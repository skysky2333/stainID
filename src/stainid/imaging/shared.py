"""Pictures derived from one field crop, computed once while the crop is worked on.

The detection steps each start from the field's RGB picture and recompute the stain separation and the tissue mask from
it (an AT8 field made them six and eight times). Inside `shared_pictures(rgb)` the pure functions of that picture
(`rgb_to_hed`, `field_tissue_mask` with the fixed cut-off) remember their result for that very array (the same object,
not an equal copy) and hand out that one array, made read-only: a step that tried to change it in place would raise
instead of silently changing it for the others. Outside the block, or for any other array, they compute as before.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

import numpy as np

_local = threading.local()


@contextmanager
def shared_pictures(rgb: np.ndarray):
    """Remember the pictures derived from `rgb` until the block ends (one block per thread at a time; blocks nest)."""
    previous = getattr(_local, "entry", None)
    _local.entry = (rgb, {})
    try:
        yield
    finally:
        _local.entry = previous


def remembered(name: str, rgb: np.ndarray, make):
    """`make()` for the current shared picture, computed once per block (read-only); else just `make()`."""
    entry = getattr(_local, "entry", None)
    if entry is None or entry[0] is not rgb:
        return make()
    values = entry[1]
    if name not in values:
        value = make()
        value.flags.writeable = False
        values[name] = value
    return values[name]


__all__ = ["remembered", "shared_pictures"]
