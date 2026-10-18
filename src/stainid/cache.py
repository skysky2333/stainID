"""Results computed from project files, kept until those files change.

The app polls how far every step has got every few seconds. Without a cache each poll would read the tile tables again
(several MB) and list output folders holding tens of thousands of files, which on a busy USB or network drive takes
seconds. Here a value is kept with the signature (modification time and size) of the files and folders it was made
from, and made again as soon as one of them changes:

- a file changes its signature when it is written or replaced, a folder when a file is added, removed or renamed in it;
- a signature taken within `RACY_SECONDS` of a change is not trusted (file systems store times coarsely, so a second
  change in the same instant would look the same), so such a value is made again on the next call;
- a value made from folders is kept at most `FOLDER_AGE` seconds: a folder's time only changes when a name in it
  does (not when a file in it is rewritten in place), and some network file systems do not update it at all;
- `every` lets a caller look at most that often when making the value is slow (`SLOW` seconds or more): folders
  that a running step writes to change all the time, so big ones are listed again every few seconds instead of on
  every request. Quick values are always checked.

Values are shared between requests: callers must not change what they get.
"""
from __future__ import annotations

import csv
import os
import threading
import time
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RACY_SECONDS = 2.5
FOLDER_AGE = 120.0
SLOW = 0.05


def signature(path: Path | str) -> tuple[int, int] | None:
    """(modification time in ns, size) of a file or folder; None when it does not exist."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


@dataclass
class _Entry:
    value: Any
    signatures: tuple
    made: float  # time.monotonic() when made
    checked: float  # time.monotonic() when last found up to date
    trusted: bool  # no input changed just before it was made
    slow: bool  # making it took SLOW seconds or more

    def fresh_enough(self, now: float, every: float) -> bool:
        return self.slow and now - self.checked < every


_ENTRIES: dict[Hashable, _Entry] = {}
_LOCKS: dict[Hashable, threading.Lock] = {}
_GUARD = threading.Lock()


def memo(key: Hashable, inputs: Iterable[Path | str], make: Callable[[], Any], every: float = 0.0, folders: bool = False,
         persist: Path | None = None, background: bool = False) -> Any:
    """`make()`, kept until one of `inputs` (files or folders) changes; when it is slow to make, looked at no more often
    than `every` seconds. `folders`: some inputs are folders (see `FOLDER_AGE`).
    Parallel requests for the same key wait for one `make()` instead of each doing the work.
    `persist`: a file the value is also kept in (pickled, with the key and the inputs' signatures), so that a server
    started again finds it there while nothing changed instead of working it out again (heavy cohort-wide values).
    `background`: once there is a value, a change is worked out on a thread of its own while callers keep getting the
    value they had (what pages poll, such as how far the steps are: never a request waiting seconds for a recount)."""
    inputs = tuple(inputs)
    entry = _ENTRIES.get(key)
    now = time.monotonic()
    if entry is not None and (entry.fresh_enough(now, every) or (background and key in _REFRESHING)):
        return entry.value
    with _GUARD:
        lock = _LOCKS.setdefault(key, threading.Lock())
    with lock:
        entry = _ENTRIES.get(key)
        now = time.monotonic()
        if entry is not None and entry.fresh_enough(now, every):
            return entry.value
        signatures = tuple(signature(p) for p in inputs)
        if entry is not None and entry.trusted and entry.signatures == signatures and (not folders or now - entry.made < FOLDER_AGE):
            entry.checked = now
            return entry.value
        if entry is not None and background:
            _refresh(key, lock, signatures, make, persist)
            return entry.value
        if entry is None and persist is not None:
            saved = _load(persist, key, signatures)
            if saved is not None:
                _ENTRIES[key] = _Entry(saved, signatures, now, now, True, True)
                return saved
        wall = time.time()
        value = make()
        trusted = all(s is None or s[0] / 1e9 < wall - RACY_SECONDS for s in signatures)
        _ENTRIES[key] = _Entry(value, signatures, now, now, trusted, time.monotonic() - now >= SLOW)
        if persist is not None and trusted:
            _save(persist, key, signatures, value)
        return value


def remember(key: Hashable, inputs: Iterable[Path | str], value: Any) -> None:
    """Keep a value made by other means (a fresh look) as `memo(key, inputs, ...)`'s, so the next caller agrees with it."""
    now, wall = time.monotonic(), time.time()
    signatures = tuple(signature(p) for p in inputs)
    trusted = all(s is None or s[0] / 1e9 < wall - RACY_SECONDS for s in signatures)
    _ENTRIES[key] = _Entry(value, signatures, now, now, trusted, True)


_REFRESHING: set = set()


def _refresh(key: Hashable, lock: threading.Lock, signatures: tuple, make: Callable[[], Any], persist: Path | None) -> None:
    """Work a value out again on a thread of its own (one at a time per key); callers get the old one meanwhile."""
    with _GUARD:
        if key in _REFRESHING:
            return
        _REFRESHING.add(key)

    def work() -> None:
        try:
            with lock:
                now, wall = time.monotonic(), time.time()
                value = make()
                trusted = all(s is None or s[0] / 1e9 < wall - RACY_SECONDS for s in signatures)
                _ENTRIES[key] = _Entry(value, signatures, now, time.monotonic(), trusted, time.monotonic() - now >= SLOW)
            if persist is not None and trusted:
                _save(persist, key, signatures, value)
        except Exception:  # noqa: BLE001 - the old value stays; the next caller tries again
            pass
        finally:
            with _GUARD:
                _REFRESHING.discard(key)

    threading.Thread(target=work, name="stainid-refresh", daemon=True).start()


def _load(path: Path, key: Hashable, signatures: tuple) -> Any:
    import pickle  # noqa: PLC0415

    try:
        with open(path, "rb") as handle:
            saved_key, saved_signatures, value = pickle.load(handle)  # noqa: S301 - a file this app wrote in its own cache
    except Exception:  # noqa: BLE001 - missing, unreadable or from another version: made again
        return None
    if saved_key != repr(key) or saved_signatures != signatures:
        return None
    _note("used", path)
    return value


def _save(path: Path, key: Hashable, signatures: tuple, value: Any) -> None:
    import pickle  # noqa: PLC0415

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f".{path.name}.{os.getpid()}.part")
        with open(part, "wb") as handle:
            pickle.dump((repr(key), signatures, value), handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(part, path)
        _note("wrote", path)
    except Exception:  # noqa: BLE001 - only a quicker start lost
        pass


def _note(what: str, path: Path) -> None:
    """Record a kept value's file with the project's cache space (`disk_cache.used` / `wrote`)."""
    from stainid import disk_cache  # noqa: PLC0415

    disk_cache.used(path) if what == "used" else disk_cache.wrote(path)


def forget() -> None:
    """Drop everything (tests, or after a change the signatures cannot see)."""
    with _GUARD:
        _ENTRIES.clear()


def rows(path: Path | str, columns: tuple[str, ...] | None = None) -> list[dict[str, str]]:
    """The rows of a CSV table ([] when it does not exist), only `columns` of them when given (missing ones are "")."""
    path = Path(path)

    def read() -> list[dict[str, str]]:
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if columns is None:
                    return list(reader)
                return [{c: r.get(c) or "" for c in columns} for r in reader]
        except FileNotFoundError:
            return []

    return memo(("rows", str(path), columns), [path], read)


def names(folder: Path | str, every: float = 5.0) -> frozenset[str]:
    """The names in a folder (empty when it does not exist), from one listing; a big folder is listed again at most
    every `every` seconds."""
    folder = Path(folder)

    def listing() -> frozenset[str]:
        try:
            return frozenset(entry.name for entry in os.scandir(folder))
        except (FileNotFoundError, NotADirectoryError):
            return frozenset()

    return memo(("names", str(folder)), [folder], listing, every, folders=True)
