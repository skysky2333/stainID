"""The AT8 thread skeleton as saved segments, so the viewer draws it and the measures count it without tracing again.

A segment is a piece of skeleton between junctions ("piece"), or a cluster of junction pixels ("junction"), and never
crosses a candidate outline: a piece running into a tau candidate is cut where it enters, so the measures can leave out
the skeleton inside accepted tau+ neurons (the old tracing ran through their somata). Every skeleton pixel belongs to
exactly one segment, so the summed `n_px` / `width_sum_um` equal the skeleton's pixel count and width sum.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

EPSILON_PX = 0.75
NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # straight steps first


def walk(pixels: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Order the (y, x) pixels of a thin piece into paths, starting from its ends (straight steps before diagonal ones)."""
    left = set(pixels)
    degree = {p: sum((p[0] + dy, p[1] + dx) in pixels for dy, dx in NEIGHBOURS) for p in pixels}
    paths = []
    while left:
        start = min(left, key=lambda p: (degree[p], p))
        path, at = [start], start
        left.discard(start)
        while True:
            step = next(((at[0] + dy, at[1] + dx) for dy, dx in NEIGHBOURS if (at[0] + dy, at[1] + dx) in left), None)
            if step is None:
                break
            path.append(step)
            left.discard(step)
            at = step
        paths.append(path)
    return paths


def _polyline(path: list[tuple[int, int]], epsilon: float) -> np.ndarray:
    points = np.array([(x, y) for y, x in path], np.int32)
    if len(points) > 2:
        points = cv2.approxPolyDP(points.reshape(-1, 1, 2), epsilon, False)[:, 0, :]
    if len(points) == 1:
        points = np.repeat(points, 2, axis=0)  # a lone pixel: drawn as a dot
    return points


def segments(skeleton: np.ndarray, width_um: np.ndarray, owner: np.ndarray, pixel_size_um: float,
             epsilon: float = EPSILON_PX) -> list[dict]:
    """The skeleton (bool) as segments: `kind`, `n_px`, `length_um`, `width_um` (mean), `width_sum_um`, `owner` (the value
    of `owner` under the segment, 0 = none) and `points` ((n, 2) x, y pixel indices; pieces reach into the junction
    pixels next to their ends so the drawn network has no gaps)."""
    if not skeleton.any():
        return []
    count = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT)
    junction = skeleton & (count >= 4)  # itself and three or more neighbours
    junctions = set(zip(*map(np.ndarray.tolist, np.nonzero(junction))))
    out = []
    plain = skeleton & ~junction
    # pieces: connected runs of plain skeleton with one owner (a run crossing an outline is cut there); an owner's runs
    # are labelled in the box of its skeleton pixels (the same pieces, in the same order, as in the whole field)
    owned = np.where(plain, owner, 0)
    owner_boxes = ndimage.find_objects(owned)
    for value in np.unique(owner[plain]):
        area = (slice(0, owner.shape[0]), slice(0, owner.shape[1])) if value == 0 else owner_boxes[int(value) - 1]
        labels, _ = ndimage.label(plain[area] & (owner[area] == value), np.ones((3, 3)))
        for index, box in enumerate(ndimage.find_objects(labels), start=1):
            if box is None:
                continue
            ys, xs = np.nonzero(labels[box] == index)
            ys, xs = ys + box[0].start + area[0].start, xs + box[1].start + area[1].start
            for path in walk(set(zip(ys.tolist(), xs.tolist()))):
                own = width_um[[p[0] for p in path], [p[1] for p in path]]
                for end, at in ((0, path[0]), (len(path), path[-1])):
                    near = next(((at[0] + dy, at[1] + dx) for dy, dx in NEIGHBOURS if (at[0] + dy, at[1] + dx) in junctions), None)
                    if near is not None:
                        path.insert(end, near)
                out.append({"kind": "piece", "n_px": len(own), "length_um": len(own) * pixel_size_um, "width_um": round(float(own.mean()), 3),
                            "width_sum_um": float(own.sum()), "owner": int(value), "points": _polyline(path, epsilon)})
    labels, _ = ndimage.label(junction, np.ones((3, 3)))
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        ys, xs = np.nonzero(labels[box] == index)
        ys, xs = ys + box[0].start, xs + box[1].start
        owners = owner[ys, xs]
        out.append({"kind": "junction", "n_px": int(len(ys)), "length_um": len(ys) * pixel_size_um,
                    "width_um": round(float(width_um[ys, xs].mean()), 3), "width_sum_um": float(width_um[ys, xs].sum()),
                    "owner": int(np.bincount(owners).argmax()), "points": _polyline([(int(round(ys.mean())), int(round(xs.mean())))], epsilon)})
    return out


def rasterize(rows, shape: tuple[int, int], origin: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Draw segments' polylines (core pixels, `points` as flat lists) into a bool picture of `shape` at `origin`."""
    out = np.zeros(shape, np.uint8)
    for points in rows:
        pts = (np.asarray(points, np.int32).reshape(-1, 2) - np.asarray(origin)).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, 1, 1)
    return out.astype(bool)
