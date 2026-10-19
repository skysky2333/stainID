"""Files the app makes from a project's data to show it quickly, kept within a size limit.

Every such file lives under `<project>/data/cache/<kind>/` (`KINDS`: core zoom pyramids, thumbnails, field pictures,
overlays, outlines, label crops …) and can be made again from the project's data at any time. When a project's caches
together grow past its limit (`limit_bytes`: the project's `cache_limit_gb` setting, else the server's `--cache-limit`,
else 100 GB; a project made from a dataset package starts with 20 GB), the files used least recently are deleted until they are under `LOW_WATER` of it; whatever asks for one
again later makes it again (and shows its usual "preparing" state meanwhile).

Only files under `data/cache/` are ever deleted. Study data and results are never caches, even when they are pictures:
slides, core images, tables, detections, masks, label logs and the label pool, labelling sessions, models, job records
and the QC pictures of the steps all live elsewhere under `data/` and are never touched here.

How code uses it (see docs/development.md, "Adding a cache"):

- `folder(root, kind)` / `path(root, kind, name)`: where a cache file goes (the only way to name one);
- `write(path, data)` writes one atomically and records it; code that writes some other way calls `wrote(path)` after;
- `used(path)` on every read, so that what is in use is kept.

Files that make up one picture together (a pyramid's `.tiles` pack and its `.json` index, a label crop and its
`.outline.json`) share the part of their name before the first dot and are deleted together. Temporary files (`*.tmp`)
are never counted; leftovers of a crash are removed after a day.

Bookkeeping is kept light: `used` and `wrote` only note the path in memory; a background thread writes the notes into
the project's index (`data/cache/.index.sqlite`) every `FLUSH_SECONDS` and cleans up when needed. File times are not
used for "last used" (modification times mark freshness elsewhere, and exFAT access times are unreliable), except for
files found by a scan of the folders (`scan`: at server start and every `RESCAN_SECONDS`, so files written before this
bookkeeping existed, or by code that forgot to record them, are counted too): their modification time is their first
"last use". Several servers and step processes may share a project: SQLite keeps the index consistent, and one cleanup
at a time runs per project (a lock file). A file used in the last `IN_USE_SECONDS` (by any process: notes reach the
index well within that time) or written in that time is never deleted, so pictures open on someone's screen stay.

The drive comes first: many projects on one drive could together be allowed more than it holds. Whenever the free space
of a drive falls below its floor (`min_free_bytes`: `stainid serve --cache-min-free`, by default 50 GB or 10 % of the
drive, whichever is smaller), the caches stop growing: the files used least recently, across every project on that
drive this process knows (`know`: the server tells it every project on its list), are deleted until the free space is
back above the floor (by `1 / LOW_WATER`), whatever each project's own limit. Writers look at the free space every few
seconds (`FREE_CHECK_SECONDS`) and wake the cleanup at once when it is low.
"""
from __future__ import annotations

import atexit
import fcntl
import heapq
import logging
import os
import re
import shutil
import sqlite3
import stat as statmod
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

import yaml

from stainid import cache

log = logging.getLogger(__name__)

GB = 10**9
DEFAULT_LIMIT = 100 * GB
LIMIT_ENV = "STAINID_CACHE_LIMIT"  # the server's default (`stainid serve --cache-limit`), inherited by its steps
SETTING = "cache_limit_gb"  # the project's own limit, in stainid.yaml (null: the server's default)
LOW_WATER = 0.9  # a cleanup goes down to this fraction of the limit
MIN_FREE_ENV = "STAINID_CACHE_MIN_FREE"  # `stainid serve --cache-min-free`: "50G", "10%" or "50G or 10%" (the smaller)
DEFAULT_MIN_FREE = "50G or 10%"
FREE_CHECK_SECONDS = 5.0  # a writer looks at the drive's free space at most this often
IN_USE_SECONDS = 600.0  # files used or written this recently are never deleted
CLEAR_IN_USE_SECONDS = 60.0  # ... by "Clear all"
FLUSH_SECONDS = 15.0
RESCAN_SECONDS = 3600.0
STALE_TEMPORARY_SECONDS = 86400.0
INDEX_NAME = ".index.sqlite"
LOCK_NAME = ".index.lock"

# Every kind of cache, with what it is called in the app. New kinds are added here (a test checks it).
KINDS: dict[str, str] = {
    "core_pyramids": "Zoomable core pictures",
    "thumbs": "Small core and tile pictures",
    "fields": "Tile pictures",
    "overlays": "Tile overlays (stain, threads, excluded areas)",
    "outlines": "Object outlines of tiles",
    "core_outlines": "Object outlines of cores",
    "core_threads": "Traced threads of cores",
    "label_crops": "Labelling pictures",
    "object_crops": "Pictures of labelled objects",
    "slide_pictures": "Scan pictures",
    "core_tiles": "Zoomable core pictures (older format)",
    "images": "Images at the analysis scale (image sets)",
    "sam_embeddings": "Prepared fields for click-to-outline",
    "candidate_index": "Candidate positions and calls of every core (for matching labels)",
    "answers": "The app's heavy answers, kept for a quick start",
    "label_overview": "The Labels page's counts per stain",
    "tissue": "Tissue of core images (the areas pages' \"not analysed\" layer)",
    "thread_ladder": "AT8 thread networks of fields at every thread threshold (the threads slider)",
}
_KIND = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_MARKER = f"{os.sep}data{os.sep}cache{os.sep}"


# ---------------------------------------------------------------- naming

def cache_root(project_root: Path | str) -> Path:
    return Path(project_root) / "data" / "cache"


_MADE: set[str] = set()  # folders known to exist


def folder(project_root: Path | str, kind: str) -> Path:
    """The folder of one kind of cache of a project (made if need be)."""
    if not _KIND.match(kind):
        raise ValueError(f"not a cache kind: {kind!r}")
    target = cache_root(project_root) / kind
    key = str(target)
    if key not in _MADE or not target.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        _MADE.add(key)
    return target


def path(project_root: Path | str, kind: str, name: str) -> Path:
    """Where the cache file `name` of this kind goes."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"not a cache file name: {name!r}")
    return folder(project_root, kind) / name


def unit(relative: str) -> str:
    """The files deleted together: same folder, same name up to the first dot."""
    head, _, name = relative.rpartition("/")
    return f"{head}/{name.split('.', 1)[0]}"


def kind_label(kind: str) -> str:
    return KINDS.get(kind, kind.replace("_", " ").capitalize())


def _split(full: str) -> tuple[str, str] | None:
    """(cache root, path inside it) of a file under some project's data/cache/<kind>/, else None."""
    at = full.rfind(_MARKER)
    if at < 0:
        return None
    relative = full[at + len(_MARKER):].replace(os.sep, "/")
    if "/" not in relative or relative.startswith(".") or relative.endswith(".tmp"):
        return None
    return full[:at + len(_MARKER) - 1], relative


# ---------------------------------------------------------------- what code calls

_USED: dict[str, float] = {}  # path -> when, not yet in an index
_WROTE: dict[str, float] = {}
_NOTES = threading.Lock()


def used(*paths: Path | str) -> None:
    """Note that these cache files were just read (cheap: written to the index in the background)."""
    now = time.time()
    for p in paths:
        _USED[os.fspath(p)] = now
    _start()


def wrote(*paths: Path | str) -> None:
    """Note that these cache files were just written (call after the file is complete)."""
    now = time.time()
    for p in paths:
        _WROTE[os.fspath(p)] = now
    _start()
    _look_at_space(paths)


def answer_file(project_root: Path | str, name: str) -> Path:
    """Where one of the app's heavy answers is kept for a quick start ("answers"); whoever writes or reads it records
    that (`cache.memo(persist=)`, `work.answer(saved=)` do)."""
    return path(project_root, "answers", name)


def write(target: Path | str, data: bytes) -> None:
    """Write a cache file so that readers see the old or the new version, never half of one (also from parallel
    writers), and record it."""
    target = Path(target)
    temporary = target.with_name(f"{target.name}.{os.getpid()}-{threading.get_ident()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, target)
    wrote(target)


_FREE_LOOKED: dict[str, float] = {}  # folder -> when a writer last looked at its drive's free space


def _look_at_space(paths) -> None:
    """Wake the cleanup at once when a drive just written to is below its floor (looked at every few seconds)."""
    now = time.monotonic()
    for p in paths:
        folder = os.path.dirname(os.fspath(p))
        if now - _FREE_LOOKED.get(folder, -FREE_CHECK_SECONDS) < FREE_CHECK_SECONDS:
            continue
        _FREE_LOOKED[folder] = now
        space = drive_space(folder)
        if space and space["free_bytes"] < space["min_free_bytes"]:
            _WAKE.set()
            return


# ---------------------------------------------------------------- the limit

def _setting(project_root: Path) -> float | None:
    config = project_root / "stainid.yaml"

    def read() -> float | None:
        try:
            value = (yaml.safe_load(config.read_text()) or {}).get(SETTING)
        except (OSError, yaml.YAMLError, AttributeError):
            return None
        return float(value) if isinstance(value, (int, float)) and value > 0 else None

    return cache.memo(("cache limit", str(config)), [config], read)


def parse_size(text: str | float | int) -> int:
    """'100G', '500 MB', '1.5T' or a plain number of GB, in bytes."""
    if isinstance(text, (int, float)):
        return int(float(text) * GB)
    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([kKmMgGtT]?)[bB]?\s*", str(text))
    if not match:
        raise ValueError(f"not a size: {text!r} (for example 100G or 500M)")
    number, unit_ = float(match.group(1)), match.group(2).upper()
    size = int(number * {"K": 10**3, "M": 10**6, "G": GB, "T": 10**12, "": GB}[unit_])
    if size <= 0:
        raise ValueError("the size must be above 0")
    return size


def server_default() -> int:
    try:
        return parse_size(os.environ[LIMIT_ENV]) if os.environ.get(LIMIT_ENV) else DEFAULT_LIMIT
    except ValueError:
        return DEFAULT_LIMIT


def limit_bytes(project_root: Path | str) -> int:
    """How much space this project's caches may take."""
    setting = _setting(Path(project_root))
    return int(setting * GB) if setting else server_default()


# ---------------------------------------------------------------- the drive's free space

def parse_min_free(text: str) -> list[tuple[str, float]]:
    """'50G', '10%' or '50G or 10%' (also '50G,10%'): the floors, as ("bytes", n) or ("share", fraction)."""
    parts = [p.strip() for p in re.split(r",|\bor\b", str(text)) if p.strip()]
    if not parts:
        raise ValueError(f"not a free-space floor: {text!r} (for example 50G, 10% or '50G or 10%')")
    out = []
    for part in parts:
        if part.endswith("%"):
            try:
                share = float(part[:-1]) / 100
            except ValueError:
                raise ValueError(f"not a percentage: {part!r}") from None
            if not 0 < share < 1:
                raise ValueError("the percentage must be above 0 and below 100")
            out.append(("share", share))
        else:
            out.append(("bytes", float(parse_size(part))))
    return out


def min_free_setting() -> str:
    text = os.environ.get(MIN_FREE_ENV) or DEFAULT_MIN_FREE
    try:
        parse_min_free(text)
    except ValueError:
        return DEFAULT_MIN_FREE
    return text


def min_free_bytes(total_bytes: int) -> int:
    """The free space a drive of this size keeps: the smallest of the floors set (`MIN_FREE_ENV`)."""
    return int(min(value if kind == "bytes" else value * total_bytes for kind, value in parse_min_free(min_free_setting())))


def _existing(path: str) -> str | None:
    while path and not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent
    return path or None


def drive_space(path: Path | str) -> dict | None:
    """Free and total space of the drive a path is on, and the floor its caches keep free (None: cannot be told)."""
    found = _existing(os.fspath(path))
    if found is None:
        return None
    try:
        space = shutil.disk_usage(found)
    except OSError:
        return None
    return {"free_bytes": int(space.free), "total_bytes": int(space.total), "min_free_bytes": min_free_bytes(space.total),
            "min_free_setting": min_free_setting()}


_KNOWN: set[str] = set()  # cache roots this process knows of (its projects): the free-space cleanup looks at all of them


def know(*project_roots: Path | str) -> None:
    """Count these projects' caches in when a drive runs low on space (nothing is scanned or read now)."""
    for root in project_roots:
        _KNOWN.add(str(cache_root(root)))


# ---------------------------------------------------------------- the index

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, kind TEXT NOT NULL, unit TEXT NOT NULL, size INTEGER NOT NULL,
                                  used REAL NOT NULL);
CREATE INDEX IF NOT EXISTS files_unit ON files (unit);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
_DB_LOCKS: dict[str, threading.Lock] = {}
_DB_GUARD = threading.Lock()


def _db_lock(root: str) -> threading.Lock:
    with _DB_GUARD:
        return _DB_LOCKS.setdefault(root, threading.Lock())


def _connect(root: str) -> sqlite3.Connection:
    """The index of a cache root, made if need be; a damaged one is thrown away (the next scan builds it again)."""
    Path(root).mkdir(parents=True, exist_ok=True)
    file = os.path.join(root, INDEX_NAME)
    db = None
    try:
        db = sqlite3.connect(file, timeout=60)
        db.executescript(_SCHEMA)
        return db
    except sqlite3.OperationalError:
        raise  # locked for too long, or the drive cannot be written: try again later
    except sqlite3.DatabaseError as error:
        log.warning("the cache index %s was damaged (%s); building it again", file, error)
        if db is not None:
            db.close()
        for name in (INDEX_NAME, INDEX_NAME + "-journal"):
            try:
                os.remove(os.path.join(root, name))
            except FileNotFoundError:
                pass
        _SCANNED.pop(root, None)
        db = sqlite3.connect(file, timeout=60)
        db.executescript(_SCHEMA)
        return db


@contextmanager
def _index(root: str) -> Iterator[sqlite3.Connection]:
    """The index of one cache root, in one transaction (one thread of this process at a time; other processes wait
    for SQLite's lock)."""
    with _db_lock(root):
        db = _connect(root)
        try:
            with db:
                yield db
        finally:
            db.close()


def _meta(db: sqlite3.Connection, key: str) -> str | None:
    row = db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _set_meta(db: sqlite3.Connection, key: str, value: str) -> None:
    db.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))


def _row(root: str, relative: str, when: float | None) -> tuple | None:
    """(path, kind, unit, size, used) of a file as it is on disk now (None: gone, or not a plain file)."""
    try:
        info = os.lstat(os.path.join(root, relative))
    except OSError:
        return None
    if not statmod.S_ISREG(info.st_mode):
        return None
    return relative, relative.split("/", 1)[0], unit(relative), info.st_size, info.st_mtime if when is None else when


_UPSERT = ("INSERT INTO files (path, kind, unit, size, used) VALUES (?, ?, ?, ?, ?) ON CONFLICT(path) DO UPDATE SET "
           "size = excluded.size, used = max(files.used, excluded.used)")


def _record(root: str, used_: dict[str, float], wrote_: dict[str, float]) -> None:
    with _index(root) as db:
        for relative, when in wrote_.items():
            row = _row(root, relative, when)
            if row is None:
                db.execute("DELETE FROM files WHERE path = ?", (relative,))
            else:
                db.execute(_UPSERT, row)
        for relative, when in used_.items():
            if relative in wrote_:
                continue
            if db.execute("UPDATE files SET used = max(used, ?) WHERE path = ?", (when, relative)).rowcount == 0:
                row = _row(root, relative, when)
                if row is not None:
                    db.execute(_UPSERT, row)


_FLUSHING = threading.Lock()


def flush() -> None:
    """Write the notes of this process into the indexes (the background thread does this every few seconds)."""
    with _FLUSHING:
        global _USED, _WROTE
        with _NOTES:  # swapped, not copied: `used` and `wrote` take no lock
            used_, _USED = _USED, {}
            wrote_, _WROTE = _WROTE, {}
        by_root: dict[str, tuple[dict, dict]] = {}
        for notes, slot in ((used_, 0), (wrote_, 1)):
            for full, when in notes.items():
                split = _split(full)
                if split is not None:
                    root, relative = split
                    group = by_root.setdefault(root, ({}, {}))[slot]
                    group[relative] = max(when, group.get(relative, 0.0))
        for root, (u, w) in by_root.items():
            _KNOWN.add(root)
            try:
                _record(root, u, w)
                _DIRTY.add(root)
            except (OSError, sqlite3.Error) as error:  # a drive gone or read-only: the notes are dropped
                log.warning("could not update the cache index in %s: %s", root, error)


# ---------------------------------------------------------------- scanning

_SCANNED: dict[str, float] = {}  # cache root -> when last scanned (by this process)
_WATCHED: set[str] = set()  # cache roots this process scans now and then (servers)


def _walk(root: str) -> Iterator[tuple[str, os.stat_result]]:
    """Every file under the kind folders of a cache root, as (path inside it, its stat)."""
    stack = [""]
    while stack:
        prefix = stack.pop()
        try:
            entries = list(os.scandir(os.path.join(root, prefix) if prefix else root))
        except OSError:
            continue
        for entry in entries:
            if not prefix and not _KIND.match(entry.name):
                continue  # the index itself, or a folder that is no kind of cache
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(relative)
                elif prefix and entry.is_file(follow_symlinks=False):
                    yield relative, entry.stat(follow_symlinks=False)
            except OSError:
                continue


def scan(project_root: Path | str) -> None:
    """Bring the index up to date with the cache folders: count files it does not know (their modification time as
    their last use), forget files that are gone, and remove temporary files a crash left behind."""
    root = str(cache_root(project_root))
    if not os.path.isdir(root):
        return
    now = time.time()
    found: dict[str, os.stat_result] = {}
    for relative, info in _walk(root):
        if relative.endswith(".tmp"):
            if now - info.st_mtime > STALE_TEMPORARY_SECONDS:
                _remove(root, relative)
            continue
        found[relative] = info
    with _index(root) as db:
        known = dict(db.execute("SELECT path, size FROM files"))
        db.executemany("DELETE FROM files WHERE path = ?", [(p,) for p in known if p not in found])
        db.executemany(_UPSERT, [(p, p.split("/", 1)[0], unit(p), i.st_size, i.st_mtime) for p, i in found.items()
                                 if known.get(p) != i.st_size])
        _set_meta(db, "scanned", str(now))
    _SCANNED[root] = now
    _DIRTY.add(root)


def watch(project_root: Path | str) -> None:
    """Keep this project's index up to date from this process (a server): scan it soon, and every hour."""
    root = str(cache_root(project_root))
    _KNOWN.add(root)
    if root not in _WATCHED:
        _WATCHED.add(root)
        _start()
        _WAKE.set()


# ---------------------------------------------------------------- cleaning up

_DIRTY: set[str] = set()  # cache roots whose size should be checked
_BUSY: dict[str, str] = {}  # cache root -> "clearing", while Clear all runs


def _safe(root: str, relative: str) -> str | None:
    """The full path of a file inside the kind folders of this cache root, or None (never anything else)."""
    parts = relative.split("/")
    if len(parts) < 2 or any(p in ("", ".", "..") for p in parts) or parts[0].startswith(".") or not _KIND.match(parts[0]):
        return None
    full = os.path.join(root, *parts)
    real_root = os.path.realpath(root)
    if not os.path.realpath(os.path.dirname(full)).startswith(real_root + os.sep):
        return None
    return full


def _remove(root: str, relative: str) -> bool:
    full = _safe(root, relative)
    if full is None:
        return False
    try:
        if statmod.S_ISDIR(os.lstat(full).st_mode):
            return False
        os.unlink(full)
    except FileNotFoundError:
        pass
    except OSError as error:
        log.warning("could not delete the cache file %s: %s", full, error)
        return False
    parent = os.path.dirname(full)
    while os.path.dirname(parent) != root and parent.startswith(root + os.sep):  # empty sub-folders of a kind go too
        try:
            os.rmdir(parent)
        except OSError:
            break
        parent = os.path.dirname(parent)
    return True


@contextmanager
def _cleaning(root: str) -> Iterator[bool]:
    """One cleanup of a cache root at a time, across processes: True when this one may clean."""
    try:
        handle = open(os.path.join(root, LOCK_NAME), "a+")
    except OSError:
        yield False
        return
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        handle.close()


def _total(root: str) -> int:
    with _index(root) as db:
        return int(db.execute("SELECT coalesce(sum(size), 0) FROM files").fetchone()[0])


def evict(project_root: Path | str, target: int, keep_seconds: float = IN_USE_SECONDS) -> dict:
    """Delete the least recently used cache files of a project until they take at most `target` bytes, keeping any used
    or written in the last `keep_seconds`. Returns what was deleted ({"files", "bytes"}), or nothing when
    another process is cleaning up this project right now ("skipped")."""
    root = str(cache_root(project_root))
    with _cleaning(root) as allowed:
        if not allowed:
            return {"files": 0, "bytes": 0, "skipped": True}
        total = _total(root)
        done = _drop(((root, *u) for u in _units(root)), lambda freed: total - freed <= target, keep_seconds, "limit")
        files, freed = done.get(root, (0, 0))
        return {"files": files, "bytes": freed, "skipped": False}


def _units(root: str) -> list[tuple[str, int, float]]:
    """(unit, size, last used) of a cache root, least recently used first."""
    with _index(root) as db:
        return db.execute("SELECT unit, sum(size), max(used) FROM files GROUP BY unit ORDER BY max(used)").fetchall()


def _drop(candidates, enough, keep_seconds: float, reason: str) -> dict[str, tuple[int, int]]:
    """Delete units (root, unit, size, last used), in the order given, until `enough(bytes freed)`; stop at the first one
    used in the last `keep_seconds`. The caller holds the cleaning lock of every root. Returns {root: (files, bytes)}."""
    now = time.time()
    freed = 0
    done: dict[str, tuple[int, int]] = {}
    pending: dict[str, list[str]] = {}
    for root, name, size, last in candidates:
        if enough(freed) or last > now - keep_seconds:
            break
        with _index(root) as db:
            members = [p for (p,) in db.execute("SELECT path FROM files WHERE unit = ?", (name,))]
        if any(_newer(root, p, now - keep_seconds) for p in members):
            continue  # made again just now (by a process whose note is not in yet)
        files = sum(_remove(root, p) for p in members)
        before = done.get(root, (0, 0))
        done[root] = (before[0] + files, before[1] + size)
        freed += size
        pending.setdefault(root, []).extend(members)
        if len(pending[root]) >= 500:
            _forget(root, pending.pop(root))
    for root, members in pending.items():
        _forget(root, members)
    for root, (files, bytes_) in done.items():
        if files:
            with _index(root) as db:
                _set_meta(db, "cleaned", f"{now} {files} {bytes_} {reason}")
            log.info("deleted %d cache files (%.1f GB) in %s (%s)", files, bytes_ / GB, root,
                     "the drive was low on space" if reason == "space" else "over its limit")
    return done


def free_space() -> list[dict]:
    """Keep every drive's floor of free space: where a drive has less, delete the least recently used cache files of all
    the projects on it this process knows, until it is back above the floor (by `1 / LOW_WATER`). Returns what was
    deleted per drive."""
    drives: dict[int, list[str]] = {}
    for root in sorted(_KNOWN | _WATCHED):
        if root in _BUSY or not os.path.isfile(os.path.join(root, INDEX_NAME)):
            continue
        try:
            drives.setdefault(os.stat(root).st_dev, []).append(root)
        except OSError:
            continue
    out = []
    for roots in drives.values():
        space = drive_space(roots[0])
        if space is None or space["free_bytes"] >= space["min_free_bytes"]:
            continue
        need = int(space["min_free_bytes"] / LOW_WATER) - space["free_bytes"]
        with ExitStack() as stack:
            allowed = [r for r in roots if stack.enter_context(_cleaning(r))]
            merged = heapq.merge(*[[(r, *u) for u in _units(r)] for r in allowed], key=lambda c: c[3])
            done = _drop(merged, lambda freed: freed >= need, IN_USE_SECONDS, "space")
        freed = sum(b for _, b in done.values())
        if freed < need:
            log.warning("the drive of %s has %.1f GB free, below its floor of %.1f GB, and its caches cannot give back more than "
                        "%.1f GB now", roots[0], space["free_bytes"] / GB, space["min_free_bytes"] / GB, freed / GB)
        out.append({"roots": roots, "needed": need, "files": sum(f for f, _ in done.values()), "bytes": freed})
    return out


def _newer(root: str, relative: str, since: float) -> bool:
    try:
        return os.lstat(os.path.join(root, relative)).st_mtime > since
    except OSError:
        return False


def _forget(root: str, paths: list[str]) -> None:
    if paths:
        with _index(root) as db:
            db.executemany("DELETE FROM files WHERE path = ?", [(p,) for p in paths])


def enforce(project_root: Path | str) -> dict | None:
    """Clean up when the project's caches take more than its limit (down to `LOW_WATER` of it)."""
    root = str(cache_root(project_root))
    if not os.path.isfile(os.path.join(root, INDEX_NAME)):
        return None
    limit = limit_bytes(Path(root).parent.parent)
    if _total(root) <= limit:
        return None
    return evict(Path(root).parent.parent, int(limit * LOW_WATER))


def check(project_root: Path | str) -> None:
    """Look at the size of this project's caches soon (after its limit changed)."""
    _DIRTY.add(str(cache_root(project_root)))
    _start()
    _WAKE.set()


def clear(project_root: Path | str) -> dict:
    """Delete every cache file of a project except those in use right now (Clear all)."""
    root = str(cache_root(project_root))
    _BUSY[root] = "clearing"
    try:
        flush()
        scan(project_root)
        return evict(project_root, 0, CLEAR_IN_USE_SECONDS)
    finally:
        _BUSY.pop(root, None)


def clear_in_background(project_root: Path | str) -> None:
    root = str(cache_root(project_root))
    if root in _BUSY:
        return
    _BUSY[root] = "clearing"  # at once, so the page that asked sees it
    threading.Thread(target=_clear_quietly, args=(project_root,), name="stainid-cache-clear", daemon=True).start()


def _clear_quietly(project_root: Path | str) -> None:
    try:
        clear(project_root)
    except (OSError, sqlite3.Error) as error:
        log.warning("could not clear the caches of %s: %s", project_root, error)


def usage(project_root: Path | str) -> dict:
    """How much space the project's caches take, per kind, and its limit."""
    root = str(cache_root(project_root))
    flush()
    kinds, scanned, cleaned = [], None, None
    if os.path.isdir(root):
        with _index(root) as db:
            kinds = db.execute("SELECT kind, sum(size), count(*) FROM files GROUP BY kind ORDER BY sum(size) DESC").fetchall()
            scanned, cleaned = _meta(db, "scanned"), _meta(db, "cleaned")
    setting = _setting(Path(project_root))
    last = None
    if cleaned:
        when, files, freed, *reason = cleaned.split()
        last = {"when": float(when), "files": int(files), "bytes": int(freed), "reason": reason[0] if reason else "limit"}
    _KNOWN.add(root)
    return {"used_bytes": sum(k[1] for k in kinds), "files": sum(k[2] for k in kinds), "limit_bytes": limit_bytes(project_root),
            "limit_gb": setting, "default_bytes": server_default(), "low_water": LOW_WATER,
            "kinds": [{"kind": k, "label": kind_label(k), "bytes": b, "files": n} for k, b, n in kinds],
            "counted": scanned is not None, "busy": _BUSY.get(root), "last_cleanup": last,
            "in_use_minutes": round(IN_USE_SECONDS / 60), "drive": drive_space(root)}


# ---------------------------------------------------------------- the background thread

_WAKE = threading.Event()
_THREAD: threading.Thread | None = None
_THREAD_GUARD = threading.Lock()


def _start() -> None:
    global _THREAD
    if _THREAD is not None and _THREAD.is_alive():
        return
    with _THREAD_GUARD:
        if _THREAD is None or not _THREAD.is_alive():
            _THREAD = threading.Thread(target=_loop, name="stainid-disk-cache", daemon=True)
            _THREAD.start()


def tick() -> None:
    """One round of the background work: write the notes, scan what is due, clean up what is over its limit."""
    flush()
    now = time.time()
    for root in list(_WATCHED):
        if now - _SCANNED.get(root, 0.0) > RESCAN_SECONDS:
            try:
                scan(Path(root).parent.parent)
            except (OSError, sqlite3.Error) as error:
                log.warning("could not count the caches in %s: %s", root, error)
                _SCANNED[root] = now
    while _DIRTY:
        root = _DIRTY.pop()
        if root in _BUSY:
            continue
        try:
            enforce(Path(root).parent.parent)
        except (OSError, sqlite3.Error) as error:
            log.warning("could not clean up the caches in %s: %s", root, error)
    try:
        free_space()
    except (OSError, sqlite3.Error) as error:
        log.warning("could not make space on the drive: %s", error)


def _loop() -> None:
    while True:
        _WAKE.wait(FLUSH_SECONDS)
        _WAKE.clear()
        try:
            tick()
        except Exception:  # noqa: BLE001 - the bookkeeping must never stop
            log.exception("cache bookkeeping failed")


@atexit.register
def _last_flush() -> None:  # a step process: its notes go in before it ends
    try:
        flush()
    except Exception:  # noqa: BLE001
        pass
