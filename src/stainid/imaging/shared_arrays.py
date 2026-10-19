"""Arrays several processes on this machine map at once (a decoded core image, core-wide masks), as files in shared
memory (/dev/shm) that each process maps with `np.memmap`: the process that makes one writes it, the tile processes map
it instead of decoding the image again, and nothing is copied between them.

Files are named `stainid-<pid>-<name>`; the maker removes its files when done, and `clean_stale()` removes those left by
a process that no longer runs (killed). Python's `multiprocessing.shared_memory` is not used: before Python 3.13 each
process attaching a block registers it with the resource tracker, which can remove it while others still use it.
"""
from __future__ import annotations

import itertools
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

PREFIX = "stainid-"
_counter = itertools.count()
_attached: OrderedDict = OrderedDict()
ATTACHED_MAX = 12  # mapped arrays a process keeps open (the newest), so a tile process reuses its maps across tiles


def folder() -> Path:
    shm = Path("/dev/shm")
    return shm if shm.is_dir() and os.access(shm, os.W_OK) else Path(tempfile.gettempdir())


def create(name: str, shape: tuple[int, ...], dtype) -> tuple[str, np.ndarray]:
    """A new zero-filled array in shared memory: its path (for `attach`) and the array (writable)."""
    path = folder() / f"{PREFIX}{os.getpid()}-{next(_counter)}-{name}"
    array = np.memmap(path, dtype=np.dtype(dtype), mode="w+", shape=tuple(int(s) for s in shape))
    return str(path), array


def attach(path: str, shape: tuple[int, ...], dtype, writable: bool = False) -> np.ndarray:
    """The array at `path` (`create`), mapped once per process and kept among the newest ATTACHED_MAX."""
    key = (path, writable)
    if key in _attached:
        _attached.move_to_end(key)
        return _attached[key]
    array = np.memmap(path, dtype=np.dtype(dtype), mode="r+" if writable else "r", shape=tuple(int(s) for s in shape))
    _attached[key] = array
    while len(_attached) > ATTACHED_MAX:
        _attached.popitem(last=False)
    return array


def forget_removed() -> list[str]:
    """Drop this process's maps of arrays whose file is gone (their maker removed them; the memory is freed once no
    process maps them); returns their paths."""
    gone = sorted({k[0] for k in _attached if not os.path.exists(k[0])})
    for key in [k for k in _attached if k[0] in gone]:
        del _attached[key]
    return gone


def remove(*paths: str) -> None:
    """Remove arrays made by `create` (processes that map them keep their map until they drop it)."""
    for path in paths:
        for key in [k for k in _attached if k[0] == path]:
            del _attached[key]
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def clean_stale() -> int:
    """Remove the shared arrays of processes that no longer run; returns how many."""
    removed = 0
    for path in folder().glob(f"{PREFIX}*"):
        try:
            pid = int(path.name[len(PREFIX):].split("-", 1)[0])
        except ValueError:
            continue
        if pid != os.getpid() and not _running(pid):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed


__all__ = ["attach", "clean_stale", "create", "folder", "forget_removed", "remove"]
