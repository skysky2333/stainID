"""Splitting and merging objects in "Fix the field" (keys S and M): the geometry, and how the edits are stored.

Both are object labels like the other edits: they train and evaluate the models and never replace detection's objects
in the analysis. An edit replaces its originals by new objects:

- each piece (split) or the merged object is an object people added: a `class` label (its type: the original's for a
  piece, the largest piece's for a merge; the keys retype it) and an `outline` label at the piece's centre, with
  `meta` {"edit": "split" | "merge", "group": <token>} (`labels.store`);
- each original gets a `replaced` label (`TASK`, id "replaced:<tile>:<x>:<y>", its `candidate` when it is one, `value`
  "split" or "merge", `meta` {"group": …, "into": [[x, y], …]}): it is no longer asked about and not confirmed when the
  field is finished, and the Cores view's Annotations layer shows it with a "split" or "merged" badge and people's
  outlines on hover.

Pieces are never matched onto candidates by position (`labels.decisions`): they are their own objects.

Geometry (field-picture pixels, polygons as [[x, y], …]):
- `split_auto`: a watershed of the object's mask into two, on its distance transform with the stain (DAB) as a tie
  break; seeds the two strongest centres of the mask, or a clicked point and the part of the mask farthest from it;
- `split_line`: the mask cut along a drawn line; only an object the line crosses right through falls in two;
- `merge`: the union of the outlines, bridged with a growing closing where they do not touch.
"""
from __future__ import annotations

import secrets

import cv2
import numpy as np

TASK = "replaced"
EDITS = ("split", "merge")
MIN_PIECE_FRACTION = 0.05  # a piece smaller than this share of the object is not a piece
BALANCED = 0.15  # an automatic split with a piece smaller than this share falls back to a straight cut between its seeds
PAD = 4


def group_token() -> str:
    return secrets.token_hex(4)


def _mask(polygons: list, pad: int = PAD) -> tuple[np.ndarray, np.ndarray]:
    """The polygons filled on a grid around them: (mask, origin [x0, y0] of the grid)."""
    allpts = np.concatenate([np.asarray(p, np.float64).reshape(-1, 2) for p in polygons])
    origin = np.floor(allpts.min(axis=0)).astype(int) - pad
    size = np.ceil(allpts.max(axis=0)).astype(int) - origin + pad + 1
    mask = np.zeros((int(size[1]), int(size[0])), np.uint8)
    for polygon in polygons:
        ring = np.round((np.asarray(polygon, np.float64).reshape(-1, 2) - origin) * 16).astype(np.int32)
        cv2.fillPoly(mask, [ring], 1, lineType=cv2.LINE_8, shift=4)
    return mask, origin


def _outline(mask: np.ndarray, origin: np.ndarray) -> list[list[float]] | None:
    """The outer outline of a mask's largest piece, in field pixels."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    simple = cv2.approxPolyDP(contour, 0.7, True)[:, 0, :]
    if len(simple) < 3:
        return None
    return (simple + origin).round(1).tolist()


def _pieces(labels: np.ndarray, origin: np.ndarray, total: int) -> list[list[list[float]]] | None:
    """The outlines of labels 1 and 2 of a two-way split, None unless both are real pieces."""
    out = []
    for value in (1, 2):
        part = labels == value
        if part.sum() < max(4, MIN_PIECE_FRACTION * total):
            return None
        outline = _outline(part, origin)
        if outline is None:
            return None
        out.append(outline)
    return out


def _dab(picture: np.ndarray | None, origin: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    """The stain's strength (DAB optical density) of the picture under the grid, 0–1; None without a picture."""
    if picture is None:
        return None
    from stainid.imaging.color import rgb_to_hed  # noqa: PLC0415

    h, w = shape
    x0, y0 = int(origin[0]), int(origin[1])
    out = np.zeros(shape, np.float32)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(picture.shape[1], x0 + w), min(picture.shape[0], y0 + h)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    dab = np.maximum(rgb_to_hed(picture[sy0:sy1, sx0:sx1])[..., 2], 0).astype(np.float32)
    out[sy0 - y0: sy1 - y0, sx0 - x0: sx1 - x0] = dab
    top = float(out.max())
    return out / top if top > 0 else out


def split_auto(polygon, picture: np.ndarray | None = None, seed: list[float] | None = None) -> list[list[list[float]]] | None:
    """An object split in two by a watershed (None when it does not fall in two real pieces). `picture`: the field's RGB
    picture (the stain breaks ties: the split runs along the paler line); `seed`: a point on the object the new split
    keeps apart from the part of it farthest away."""
    from skimage.segmentation import watershed  # noqa: PLC0415

    mask, origin = _mask([polygon])
    total = int(mask.sum())
    if total < 16:
        return None
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    smooth = cv2.GaussianBlur(distance, (0, 0), 1.0)
    from skimage.feature import peak_local_max  # noqa: PLC0415

    straight = False  # no neck to follow: cut straight between the seeds
    ys, xs = np.nonzero(mask)
    spread = max(2, int(np.sqrt(total) / 6))
    if seed is not None:
        sx, sy = int(round(seed[0] - origin[0])), int(round(seed[1] - origin[1]))
        if not (0 <= sy < mask.shape[0] and 0 <= sx < mask.shape[1]) or not mask[sy, sx]:
            nearest = int(np.argmin((xs - sx) ** 2 + (ys - sy) ** 2))
            sx, sy = int(xs[nearest]), int(ys[nearest])
        # the other seed: the centre of the mask farthest from the click (a centre, so it holds its own part)
        peaks = peak_local_max(smooth, min_distance=spread, labels=mask, num_peaks=6, exclude_border=False)
        if len(peaks):
            far = peaks[int(np.argmax((peaks[:, 1] - sx) ** 2 + (peaks[:, 0] - sy) ** 2))]
            other = (int(far[0]), int(far[1]))
        else:
            index = int(np.argmax((xs - sx) ** 2 + (ys - sy) ** 2))
            other = (int(ys[index]), int(xs[index]))
        if (other[0] - sy) ** 2 + (other[1] - sx) ** 2 < spread ** 2:  # the click is on that centre: the farthest point
            index = int(np.argmax((xs - sx) ** 2 + (ys - sy) ** 2))
            other = (int(ys[index]), int(xs[index]))
        seeds = [(sy, sx), other]
    else:
        peaks = peak_local_max(smooth, min_distance=spread, labels=mask, num_peaks=4, exclude_border=False)
        strong = [(int(q[0]), int(q[1])) for q in peaks if smooth[q[0], q[1]] >= 0.5 * float(smooth.max())]
        if len(strong) >= 2 and _neck(smooth, strong[0], strong[1]):  # two centres with a neck between (two touching objects)
            seeds = strong[:2]
        else:  # one centre (an oval, a blob): the middles of its two halves along its long axis
            seeds, straight = _halves(xs, ys), True
    markers = np.zeros(mask.shape, np.int32)
    for i, (y, x) in enumerate(seeds, start=1):
        markers[max(0, y - 1): y + 2, max(0, x - 1): x + 2] = i
    markers[mask == 0] = 0
    top = float(smooth.max()) or 1.0
    elevation = -smooth / top
    dab = _dab(picture, origin, mask.shape)
    if dab is not None:
        elevation = elevation - 0.35 * dab  # the cut prefers the paler line between two dark centres
    labels = watershed(elevation, markers, mask=mask.astype(bool))
    sizes = np.bincount(labels.ravel(), minlength=3)[1:3]
    if straight or sizes.min() < BALANCED * total:  # no neck, or one seed flooded nearly everything: a straight cut between the seeds
        (ay, ax), (by, bx) = seeds
        nearer = (xs - ax) ** 2 + (ys - ay) ** 2 <= (xs - bx) ** 2 + (ys - by) ** 2
        labels = np.zeros(mask.shape, np.int32)
        labels[ys, xs] = np.where(nearer, 1, 2)
    return _pieces(labels, origin, total)


def _neck(distance: np.ndarray, a: tuple[int, int], b: tuple[int, int], dip: float = 0.85) -> bool:
    """Whether the object narrows between two centres: the distance to the edge along the line between them falls below
    `dip` of the lower centre's (an oval's ridge does not, a dumbbell's neck does)."""
    steps = max(2, int(np.hypot(a[0] - b[0], a[1] - b[1])))
    rows = np.linspace(a[0], b[0], steps).round().astype(int)
    cols = np.linspace(a[1], b[1], steps).round().astype(int)
    return float(distance[rows, cols].min()) < dip * min(float(distance[a]), float(distance[b]))


def _halves(xs: np.ndarray, ys: np.ndarray) -> list[tuple[int, int]]:
    """The middles of a mask's two halves along its long axis, as (row, column) of pixels of the mask."""
    points = np.c_[xs, ys].astype(np.float64)
    centred = points - points.mean(axis=0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    side = centred @ axis >= 0
    out = []
    for half in (side, ~side):
        middle = points[half].mean(axis=0) if half.any() else points.mean(axis=0)
        nearest = int(np.argmin(((points - middle) ** 2).sum(axis=1)))
        out.append((int(ys[nearest]), int(xs[nearest])))
    return out


def crosses(polygon, line) -> bool:
    """Whether a drawn line crosses an object right through: it starts and ends outside the outline and passes inside."""
    ring = np.asarray(polygon, np.float32).reshape(-1, 1, 2)
    points = np.asarray(line, np.float64).reshape(-1, 2)
    if len(points) < 2:
        return False
    inside = [cv2.pointPolygonTest(ring, (float(x), float(y)), False) > 0 for x, y in points]
    if inside[0] or inside[-1]:
        return False
    if any(inside):
        return True
    dense = []  # a fast stroke: points between the recorded ones
    for (ax, ay), (bx, by) in zip(points[:-1], points[1:]):
        steps = max(1, int(np.hypot(bx - ax, by - ay)))
        dense += [(ax + (bx - ax) * t / steps, ay + (by - ay) * t / steps) for t in range(steps)]
    return any(cv2.pointPolygonTest(ring, (float(x), float(y)), False) > 0 for x, y in dense)


def split_line(polygon, line, width: float = 1.5) -> list[list[list[float]]] | None:
    """An object cut along a drawn line (None unless the line crosses it right through and it falls in two)."""
    if not crosses(polygon, line):
        return None
    mask, origin = _mask([polygon])
    total = int(mask.sum())
    cut = mask.copy()
    points = np.round((np.asarray(line, np.float64).reshape(-1, 2) - origin)).astype(np.int32)
    cv2.polylines(cut, [points.reshape(-1, 1, 2)], False, 0, thickness=max(1, int(round(2 * width))))
    count, labels = cv2.connectedComponents(cut, connectivity=4)
    if count < 3:
        return None
    sizes = np.bincount(labels.ravel())[1:]
    order = np.argsort(-sizes)[:2] + 1
    two = np.zeros_like(labels)
    two[labels == order[0]] = 1
    two[labels == order[1]] = 2
    return _pieces(two, origin, total)


MERGE_GAP_PX = 80  # objects further apart than this (field px) are not merged


def merge_gap(polygons: list) -> float:
    """How far apart the objects are, as a merge would bridge them: the largest gap (field px) on the shortest links
    that join them all (a minimum spanning tree over the nearest distances between their outlines); 0 when they touch."""
    rings = [np.asarray(p, np.float64).reshape(-1, 2) for p in polygons]
    n = len(rings)
    if n < 2:
        return 0.0
    boxes = [(r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max()) for r in rings]
    gap = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            box_gap = max(0.0, max(a[0], b[0]) - min(a[2], b[2]), max(a[1], b[1]) - min(a[3], b[3]))
            if box_gap > 4 * MERGE_GAP_PX:  # far apart: no need to measure exactly
                gap[i, j] = gap[j, i] = box_gap
                continue
            inside = any(cv2.pointPolygonTest(rings[j].astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) >= 0 for x, y in rings[i][:: max(1, len(rings[i]) // 32)])
            d = 0.0 if inside else float(np.sqrt(((rings[i][:, None, :] - rings[j][None, :, :]) ** 2).sum(-1)).min())
            gap[i, j] = gap[j, i] = d
    joined, widest = {0}, 0.0
    while len(joined) < n:  # Prim's tree: add the nearest object each time
        best = min(((gap[i, j], j) for i in joined for j in range(n) if j not in joined))
        widest = max(widest, best[0])
        joined.add(best[1])
    return widest


def merge(polygons: list, max_gap_px: int = MERGE_GAP_PX) -> list[list[float]] | None:
    """One outline for several objects: the union of their outlines, bridged by a closing just wide enough to make it
    one piece; None for fewer than two or when they lie more than `max_gap_px` apart (`merge_gap`, checked first so a
    refusal is immediate)."""
    polygons = [p for p in polygons if p is not None and len(p) >= 3]
    if len(polygons) < 2 or merge_gap(polygons) > max_gap_px:
        return None
    widest = merge_gap(polygons)
    mask, origin = _mask(polygons, pad=PAD + int(widest) + 4)
    joined = mask
    size = 1 if widest <= 0 else int(widest) + 3 | 1
    while cv2.connectedComponents(joined, connectivity=8)[0] - 1 > 1 or size == 1:
        if size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            if cv2.connectedComponents(joined, connectivity=8)[0] - 1 <= 1:
                break
        size += 4
        if size > 2 * max_gap_px + 5:
            return None
    joined = joined | mask
    return _outline(joined, origin)


def centre(polygon) -> tuple[float, float]:
    """A polygon's centroid (its vertex mean for a degenerate one), field px."""
    ring = np.asarray(polygon, np.float64).reshape(-1, 2)
    moments = cv2.moments(ring.astype(np.float32))
    if moments["m00"]:
        return round(moments["m10"] / moments["m00"], 1), round(moments["m01"] / moments["m00"], 1)
    return round(float(ring[:, 0].mean()), 1), round(float(ring[:, 1].mean()), 1)


def area(polygon) -> float:
    return float(cv2.contourArea(np.asarray(polygon, np.float32).reshape(-1, 1, 2)))


def is_piece(meta) -> bool:
    """Whether a label's `meta` says it is a piece of a split or the object of a merge."""
    return isinstance(meta, dict) and meta.get("edit") in EDITS


def pieces_mask(labels):
    """Which rows of a label frame are pieces of a split or merged objects (a boolean Series)."""
    import pandas as pd  # noqa: PLC0415

    if "meta" not in labels:
        return pd.Series(False, index=labels.index)
    return labels.meta.map(is_piece).astype(bool)
