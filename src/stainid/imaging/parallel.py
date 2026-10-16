"""A few threads for the heavy whole-field filters of one process (OpenCV and scipy.ndimage release the GIL).

Detection runs one process per core image; a process is single-threaded Python, so with the processes bounded by memory
most processors sat idle. `threads(n)` (or `STAINID_THREADS`) lets the independent parts of a filter run at once: the 12
line openings of the linear-artifact mask, and each 1-D Gaussian pass of the AT8 ridge filter split into strips across
the axis it does not run along (every line is filtered on its own, so the strips give the same numbers). Results never
depend on the number of threads.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

_state = {"threads": None, "pool": None}


def threads(count: int | None = None) -> int:
    """The threads to use (set it with `count`; default `STAINID_THREADS`, else 1)."""
    if count is not None:
        _state["threads"] = max(1, int(count))
        if _state["pool"] is not None:
            _state["pool"].shutdown(wait=True)
            _state["pool"] = None
    if _state["threads"] is None:
        try:
            _state["threads"] = max(1, int(os.environ.get("STAINID_THREADS", "1")))
        except ValueError:
            _state["threads"] = 1
    return _state["threads"]


def map_all(function, items) -> list:
    """`[function(item) for item in items]`, on the threads when there are several."""
    items = list(items)
    if threads() <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    if _state["pool"] is None:
        _state["pool"] = ThreadPoolExecutor(threads(), thread_name_prefix="stainid-filter")
    return list(_state["pool"].map(function, items))


def strips(length: int, parts: int | None = None) -> list[slice]:
    """`length` cut into about `parts` (default: the threads) contiguous slices."""
    parts = max(1, min(parts or threads(), length))
    bounds = [round(i * length / parts) for i in range(parts + 1)]
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


__all__ = ["map_all", "strips", "threads"]
