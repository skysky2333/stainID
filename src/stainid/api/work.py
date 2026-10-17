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
import logging
import os
import statistics
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

WORKERS = max(2, int(os.environ.get("STAINID_WORKERS", "3")))
WAIT_SECONDS = float(os.environ.get("STAINID_REQUEST_WAIT", "8"))  # how long a request waits for its work before answering 202
KEEP_SECONDS = 300.0  # finished work stays known this long (its result is fetched once the page polls)
NICE = 5  # pool threads run this much nicer than the request threads (Linux: per thread)
URGENT, NOW, SOON, AHEAD = -1, 0, 1, 2  # a click / someone is waiting / someone will soon / made ahead
GUESS_SECONDS = 5.0  # the estimate of a kind of work never seen yet


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


def typical(kind: str) -> float:
    """How long this kind of work took lately (median of the last 20), a guess before it ever ran."""
    seen = _DURATIONS.get(kind)
    return statistics.median(seen) if seen else GUESS_SECONDS


def _left(task: Task, now: float) -> float:
    expected = typical(task.kind)
    if task.state != "running" or task.started is None:
        return expected
    spent = now - task.started
    if task.total and task.done:
        return max(0.0, spent / task.done * (task.total - task.done))
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
               "running": len(_RUNNING), "done": task.done, "total": task.total, "eta_s": round(eta, 1)}
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
            return task
        task = Task(kind, key, run, priority, label, admin)
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
           seconds: float | None = None, admin: bool = False, fresh: Callable[[Task], bool] | None = None, alone: bool = False):
    """`run`'s result when it is done within `seconds` (default `WAIT_SECONDS`), else a 202 `pending` answer."""
    task = submit(kind, key, run, priority, label, admin, fresh, alone)
    if wait(task, WAIT_SECONDS if seconds is None else seconds):
        return outcome(task)
    return pending(task, then)


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
        task.state, task.error, task.status_code = "failed", f"{type(error).__name__}: {error}", 500
    with _COND:
        task.finished = time.monotonic()
        if task.state == "done":
            _DURATIONS.setdefault(task.kind, deque(maxlen=20)).append(task.finished - (task.started or task.finished))
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
