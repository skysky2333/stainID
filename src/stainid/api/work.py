"""Slow work of the server's requests, on a small pool of background threads, with a queue position and an estimate.

A request whose answer takes a while (opening a field whose core was not read yet, making a session's pictures, a
Segment Anything click while the encoder is busy, the labels overview) hands the work to this pool and waits a few
seconds (`WAIT_SECONDS`). Done by then, it answers as usual. Otherwise it answers 202 with what is going on
(`pending`): the task's id, its state (waiting or running), how many tasks are ahead of it, done/total when the work
counts its steps, and the seconds it should still take, estimated from the recent durations of the same kind of work.
The page then asks again (`then: "retry"`: the same request, answered at once once the work is done) or polls
`/api/session/work/<id>` for the result (`then: "result"`, for work that is an action, e.g. making a session).

The pool has `WORKERS` threads, at a lower CPU priority than the request threads. Work made ahead (`AHEAD`: the next
fields of a session, its pictures) never takes the last free thread, so work someone is waiting for (`NOW`) always
starts at once. The same work asked twice (the same `key`) is done once.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import statistics
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response

log = logging.getLogger(__name__)

ASKER: contextvars.ContextVar[str | None] = contextvars.ContextVar("stainid_asker", default=None)  # set per request by the app
WORKERS = max(2, int(os.environ.get("STAINID_WORKERS", "3")))
WAIT_SECONDS = float(os.environ.get("STAINID_REQUEST_WAIT", "8"))  # how long a request waits for its work before answering 202
KEEP_SECONDS = 300.0  # finished work stays known this long (its result is fetched once the page polls)
NICE = 5  # pool threads run this much nicer than the request threads (Linux: per thread)
URGENT, NOW, SOON, AHEAD = -1, 0, 1, 2  # a click / someone is waiting / someone will soon / made ahead
GUESS_SECONDS = 5.0  # the estimate of a kind of work never seen yet
PLAIN_ERROR = ("Something went wrong on the server. Try again in a moment; if it keeps happening, tell the person running stainID "
               "(the details are in the server's log).")


@dataclass
class Task:
    kind: str
    key: Any
    run: Callable[[Task], Any]
    priority: int = NOW
    label: str = ""
    admin: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: str = "waiting"  # waiting, running, done, failed
    done: int = 0
    total: int = 0
    queued: float = field(default_factory=time.monotonic)
    started: float | None = None
    finished: float | None = None
    result: Any = None
    error: str = ""
    status_code: int = 500
    event: threading.Event = field(default_factory=threading.Event)
    context: contextvars.Context = field(default_factory=contextvars.copy_context)  # the request's project (`state.CURRENT`)
    viewers: set = field(default_factory=set)  # the accounts that asked for it (None: someone not signed in); see `may_see`

    def step(self, done: int, total: int | None = None) -> None:
        """Progress of the work: `done` of `total` steps."""
        self.done = int(done)
        if total is not None:
            self.total = int(total)


_COND = threading.Condition()
_QUEUE: list[Task] = []
_BY_KEY: dict[Any, Task] = {}
_BY_ID: dict[str, Task] = {}
_RUNNING: list[Task] = []
_DURATIONS: dict[str, deque] = {}
_THREADS: list[threading.Thread] = []


_SAVED: tuple[str, dict[str, list[float]]] | None = None  # durations of earlier servers (`durations_file`), read once
_SAVED_AT = [0.0]


def durations_file():
    from stainid.api import registry  # noqa: PLC0415

    return registry.SETTINGS.parent / "work_durations.json"


def _saved() -> dict[str, list[float]]:
    global _SAVED
    path = str(durations_file())
    if _SAVED is None or _SAVED[0] != path:
        import json  # noqa: PLC0415

        try:
            found = json.loads(durations_file().read_text())
            _SAVED = (path, {str(k): [float(x) for x in v][-20:] for k, v in found.items() if isinstance(v, list)})
        except (OSError, ValueError, TypeError):
            _SAVED = (path, {})
    return _SAVED[1]


def _save_durations() -> None:
    """Keep the recent durations for the next server (a cold start would otherwise guess `GUESS_SECONDS` for work that
    takes minutes), at most every 30 s."""
    import json  # noqa: PLC0415

    if time.monotonic() - _SAVED_AT[0] < 30:
        return
    _SAVED_AT[0] = time.monotonic()
    merged = {**_saved(), **{k: list(v) for k, v in _DURATIONS.items()}}
    try:
        from stainid.tables import write_text_atomic  # noqa: PLC0415

        write_text_atomic(durations_file(), json.dumps(merged))
    except OSError:
        pass


def typical(kind: str) -> float:
    """How long this kind of work took lately (median of the last 20; also from earlier servers), a guess before it
    ever ran."""
    seen = _DURATIONS.get(kind) or _saved().get(kind)
    return statistics.median(seen) if seen else GUESS_SECONDS


def overdue(task: Task, now: float) -> bool:
    """Running longer than this kind of work usually takes (and saying nothing about how far it is)."""
    return task.state == "running" and task.started is not None and not (task.total and task.done) and now - task.started > typical(task.kind)


def _left(task: Task, now: float) -> float:
    expected = typical(task.kind)
    if task.state != "running" or task.started is None:
        return expected
    spent = now - task.started
    if task.total and task.done:
        return max(0.0, spent / task.done * (task.total - task.done))
    if spent > expected:  # longer than usual: no promise of "half a second" for minutes on end; at least as long again as half of it
        return max(1.0, spent * 0.5)
    return max(0.5, expected - spent)


def _ordered() -> list[Task]:
    return sorted(_QUEUE, key=lambda t: (t.priority, t.queued))


def status(task: Task) -> dict:
    """What a page shows about a task: its state, how many are ahead, done/total and an estimate in seconds."""
    with _COND:
        now = time.monotonic()
        ahead: list[Task] = []
        if task.state == "waiting":
            queue = _ordered()
            ahead = queue[: queue.index(task)] if task in queue else []
            busy = sum(_left(t, now) for t in _RUNNING) + sum(_left(t, now) for t in ahead)
            eta = busy / WORKERS + typical(task.kind) if len(_RUNNING) + len(ahead) >= WORKERS else typical(task.kind)
        else:
            eta = _left(task, now) if task.state == "running" else 0.0
        out = {"id": task.id, "kind": task.kind, "label": task.label, "state": task.state, "ahead": len(ahead) if task.state == "waiting" else 0,
               "running": len(_RUNNING), "done": task.done, "total": task.total, "eta_s": round(eta, 1), "overdue": overdue(task, now)}
    if task.state == "failed":
        out |= {"error": task.error, "status_code": task.status_code}
    return out


def _forget_old(now: float) -> None:
    for task_id, task in list(_BY_ID.items()):
        if task.finished is not None and now - task.finished > KEEP_SECONDS:
            _BY_ID.pop(task_id, None)
            if _BY_KEY.get(task.key) is task:
                _BY_KEY.pop(task.key, None)
        elif task.state == "failed" and task.finished is not None and now - task.finished > 5 and _BY_KEY.get(task.key) is task:
            _BY_KEY.pop(task.key, None)  # asked again a little later: tried again (its status stays readable by id)


def submit(kind: str, key: Any, run: Callable[[Task], Any], priority: int = NOW, label: str = "", admin: bool = False,
           fresh: Callable[[Task], bool] | None = None, alone: bool = False) -> Task:
    """The task doing `run(task)` for `key` (the one already queued, running or just done, when there is one; its
    priority raised to this one). `fresh(task)` says whether a finished task's result still holds (default: it does
    until `KEEP_SECONDS`). `alone`: started at once on a thread of its own instead of queued behind the workers (quick
    reads a page waits for, such as where the study stands, that must not wait for field preparations)."""
    with _COND:
        now = time.monotonic()
        _forget_old(now)
        task = _BY_KEY.get(key)
        if task is not None and task.state == "done" and fresh is not None and not fresh(task):
            task = None
        if task is not None:
            if task.state == "waiting" and priority < task.priority:
                task.priority = priority
                _COND.notify_all()
            task.viewers.add(_asker())
            return task
        task = Task(kind, key, run, priority, label, admin)
        task.viewers.add(_asker())
        _BY_KEY[key] = task
        _BY_ID[task.id] = task
        if alone:
            task.state, task.started = "running", now  # not one of `_RUNNING`: it takes no worker
            threading.Thread(target=_run_one, args=(task,), name=f"stainid-{kind}", daemon=True).start()
            return task
        _QUEUE.append(task)
        _start_workers()
        _COND.notify_all()
        return task


def forget(kind: str) -> None:
    """What this kind of work made no longer holds (what it was made from changed): the next request makes it anew at
    once. Work already running still finishes for whoever follows it, but is not handed to later requests."""
    with _COND:
        for key, task in list(_BY_KEY.items()):
            if task.kind == kind:
                _BY_KEY.pop(key, None)


def _asker() -> str | None:
    """Who asks for the work (set by the app for every request, `session.asker`): the account's id, or a guest's own
    anonymous id ("guest:…"), so that guests never share work; else (no request) the label attribution."""
    found = ASKER.get()
    if found is not None:
        return found
    from stainid.labels import store  # noqa: PLC0415

    return store.ATTRIBUTION.get()


def may_see(task: Task, account_id: str | None, admin: bool) -> bool:
    """Whether this asker (`session.asker`: an account's id or a guest's anonymous id) may follow the task and read its
    result: admins every task; anyone else only work they asked for themselves. Nobody (no account, no guest id) is no
    one's: never another guest's work."""
    if admin:
        return True
    return account_id is not None and not task.admin and account_id in task.viewers


def within(seconds: float) -> Callable[[Task], bool]:
    """A `fresh` test: a finished task's result is reused for this long (the page asking again for it), then made anew."""
    return lambda task: task.finished is not None and time.monotonic() - task.finished < seconds


def find(task_id: str) -> Task | None:
    with _COND:
        return _BY_ID.get(task_id)


def wait(task: Task, seconds: float) -> bool:
    return task.event.wait(max(0.0, seconds))


def outcome(task: Task) -> Any:
    """The result of a finished task (its HTTP error raised again)."""
    if task.state == "failed":
        raise HTTPException(task.status_code, task.error)
    return task.result


def pending(task: Task, then: str = "retry") -> JSONResponse:
    """202: the work is not done yet; the page shows `pending` and asks again (`retry`) or polls the task (`result`)."""
    return JSONResponse({"pending": status(task), "then": then}, status_code=202)


def answer(kind: str, key: Any, run: Callable[[Task], Any], priority: int = NOW, label: str = "", then: str = "retry",
           seconds: float | None = None, admin: bool = False, fresh: Callable[[Task], bool] | None = None, alone: bool = False,
           saved: Path | None = None):
    """`run`'s result when it is done within `seconds` (default `WAIT_SECONDS`), else a 202 `pending` answer.
    `saved`: a file every result is also kept in (JSON); while new work is not done within `SAVED_WAIT`, the request
    gets that last result instead of waiting: a server started again, or a busy one, answers at once with what it knew,
    and the page's next poll gets the new one."""
    if saved is not None:
        run = _keeping(run, saved)
        if saved.exists():
            seconds = SAVED_WAIT if seconds is None else min(seconds, SAVED_WAIT)
    task = submit(kind, key, run, priority, label, admin, fresh, alone)
    if wait(task, WAIT_SECONDS if seconds is None else seconds):
        return outcome(task)
    if saved is not None:
        last = _last(saved)
        if last is not None:
            return last
    return pending(task, then)


SAVED_WAIT = 0.3  # how long a request with a saved answer waits for new work before taking the saved one


def _keeping(run: Callable[[Task], Any], saved: Path) -> Callable[[Task], Any]:
    def run_and_keep(task: Task) -> Any:
        result = run(task)
        if not isinstance(result, Response):
            try:
                from fastapi.encoders import jsonable_encoder  # noqa: PLC0415

                from stainid.tables import write_text_atomic  # noqa: PLC0415

                saved.parent.mkdir(parents=True, exist_ok=True)
                from stainid import disk_cache  # noqa: PLC0415

                write_text_atomic(saved, json.dumps(jsonable_encoder(result)))
                disk_cache.wrote(saved)
            except (OSError, TypeError, ValueError):
                pass
        return result

    return run_and_keep


def _last(saved: Path) -> Any:
    from stainid import disk_cache  # noqa: PLC0415

    try:
        found = json.loads(saved.read_text())
    except (OSError, ValueError):
        return None
    disk_cache.used(saved)
    return found


def _start_workers() -> None:
    _THREADS[:] = [t for t in _THREADS if t.is_alive()]
    while len(_THREADS) < WORKERS:
        thread = threading.Thread(target=_work, name=f"stainid-work-{len(_THREADS)}", daemon=True)
        thread.start()
        _THREADS.append(thread)


def _next() -> Task:
    with _COND:
        while True:
            ahead_running = sum(t.priority >= AHEAD for t in _RUNNING)
            for task in _ordered():
                if task.priority >= AHEAD and ahead_running >= WORKERS - 1:
                    continue  # the last free thread stays for work someone waits for
                _QUEUE.remove(task)
                task.state, task.started = "running", time.monotonic()
                _RUNNING.append(task)
                return task
            _COND.wait(timeout=30)


def _work() -> None:
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), min(19, os.getpriority(os.PRIO_PROCESS, 0) + NICE))
    except (AttributeError, OSError):  # not Linux, or not allowed
        pass
    while True:
        _run_one(_next())


def _run_one(task: Task) -> None:
    try:
        task.result = task.context.run(task.run, task)
        task.state = "done"
    except HTTPException as error:
        task.state, task.error, task.status_code = "failed", str(error.detail), error.status_code
    except Exception as error:  # noqa: BLE001 - reported to the page that waits for it
        log.exception("background work %s %s failed", task.kind, task.key)
        # the admin's pages show what went wrong; annotators get a plain message (the detail is in the server log)
        task.state, task.error, task.status_code = "failed", f"{type(error).__name__}: {error}" if task.admin else PLAIN_ERROR, 500
    with _COND:
        task.finished = time.monotonic()
        if task.state == "done":
            _DURATIONS.setdefault(task.kind, deque(_saved().get(task.kind, []), maxlen=20)).append(task.finished - (task.started or task.finished))
            _save_durations()
        if task in _RUNNING:
            _RUNNING.remove(task)
        _COND.notify_all()
    task.event.set()


def idle() -> bool:
    """Nothing queued or running (tests)."""
    with _COND:
        return not _QUEUE and not _RUNNING


def overview() -> dict:
    """The pool's queue as a whole: what is running and waiting (for the admin)."""
    with _COND:
        tasks = [*_RUNNING, *_ordered()]
    return {"workers": WORKERS, "tasks": [status(t) for t in tasks]}
