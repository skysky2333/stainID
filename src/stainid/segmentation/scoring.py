"""Scoring one field: objects people outlined against the pipeline's candidate masks.

Inputs are plain arrays, so the rules can be tested on synthetic fields: the people's objects as polygons, the
candidates as labels in one or more label images (`layers`; a candidate is a (layer, label) key, so masks of
different layers may overlap), and which candidates the pipeline counted.

Rules (constants below):
- A candidate is *found* for an object when they match one to one at IoU >= `IOU` (greedy, best IoU first); also
  reported at the looser `IOU_LOW`.
- A candidate *overlaps* an object when it covers at least `OVERLAP_OF_OBJECT` of the object or lies at least
  `OVERLAP_OF_CANDIDATE` inside it.
- An object that is not found is, in this order: *missed* (no candidate overlaps it), *merged* (the candidate that
  overlaps it most also covers at least `MERGE_COVER` of another outlined object while covering `MERGE_COVER` of
  this one), *split* (at least two candidates each cover `SPLIT_PART` of it while lying mostly inside it) or a *poor
  outline* (overlapped, but no candidate fits it well enough).
- Objects of class "unsure" are not counted, and a counted candidate that overlaps only unsure objects is left out
  of the precision.
- Only objects (and counted candidates) whose centre lies inside the field's square count; candidates in the margin
  around it can still match the objects near its edge.
- Boundary IoU (Cheng et al. 2021) compares the bands within `BAND` of the object's diameter (at least 2 px) inside
  the two outlines' edges, so it shows how well the edges agree, which plain IoU hides for large objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

IOU = 0.5
IOU_LOW = 0.3
OVERLAP_OF_OBJECT = 0.1
OVERLAP_OF_CANDIDATE = 0.5
MERGE_COVER = 0.5
SPLIT_PART = 0.15
BAND = 0.1
UNSURE = "unsure"
OUTCOMES = ("found", "missed", "merged", "split", "poor outline")
EXTRA = ("right", "nothing there", "piece of an object", "covers several objects", "poor outline")

Box = tuple[slice, slice]


@dataclass
class Truth:
    """One outlined object: its polygon (field pixels), class, and its mask cut to its bounding box."""
    points: list[list[float]]
    value: str
    box: Box = field(init=False)
    mask: np.ndarray = field(init=False)

    def rasterize(self, shape: tuple[int, int]) -> None:
        xy = np.round(np.asarray(self.points, float)).astype(np.int32)
        x0, y0 = max(0, xy[:, 0].min()), max(0, xy[:, 1].min())
        x1, y1 = min(shape[1], xy[:, 0].max() + 1), min(shape[0], xy[:, 1].max() + 1)
        x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
        crop = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(crop, [xy - [x0, y0]], 1)
        self.box, self.mask = (slice(y0, y1), slice(x0, x1)), crop.astype(bool)

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    @property
    def centre(self) -> tuple[float, float]:
        ys, xs = np.nonzero(self.mask)
        return (float(xs.mean()) + self.box[1].start, float(ys.mean()) + self.box[0].start) if len(xs) else (np.nan, np.nan)


@dataclass
class Candidates:
    """Candidate masks: `keys[i]` = (layer, label) in `layers` (k x H x W)."""
    layers: np.ndarray
    keys: list[tuple[int, int]]
    area: np.ndarray = field(init=False)
    boxes: list[Box | None] = field(init=False)

    def __post_init__(self) -> None:
        counts = [np.bincount(layer.ravel()) for layer in self.layers]
        found = [ndimage.find_objects(layer) for layer in self.layers]
        self.area = np.array([counts[k][v] if v < len(counts[k]) else 0 for k, v in self.keys], float)
        self.boxes = [found[k][v - 1] if 0 < v <= len(found[k]) else None for k, v in self.keys]

    def centre(self, i: int) -> tuple[float, float]:
        box = self.boxes[i]
        if box is None:
            return np.nan, np.nan
        ys, xs = np.nonzero(self.layers[self.keys[i][0]][box] == self.keys[i][1])
        return float(xs.mean()) + box[1].start, float(ys.mean()) + box[0].start

    def mask(self, i: int, box: Box) -> np.ndarray:
        layer, label = self.keys[i]
        return self.layers[layer][box] == label


def intersections(truths: list[Truth], candidates: Candidates) -> np.ndarray:
    """Pixels each object shares with each candidate (objects x candidates)."""
    index = {key: i for i, key in enumerate(candidates.keys)}
    inter = np.zeros((len(truths), len(candidates.keys)))
    for t, truth in enumerate(truths):
        for layer in range(len(candidates.layers)):
            values, counts = np.unique(candidates.layers[layer][truth.box][truth.mask], return_counts=True)
            for value, count in zip(values, counts):
                i = index.get((layer, int(value))) if value else None
                if i is not None:
                    inter[t, i] = count
    return inter


def iou(inter: np.ndarray, truth_area: np.ndarray, cand_area: np.ndarray) -> np.ndarray:
    union = truth_area[:, None] + cand_area[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match(scores: np.ndarray, threshold: float) -> np.ndarray:
    """One-to-one matches at `scores` >= threshold, best first: for each row its column, or -1."""
    rows = np.full(scores.shape[0], -1)
    taken = np.zeros(scores.shape[1], bool)
    pairs = np.argwhere(scores >= threshold)
    for t, c in sorted(pairs.tolist(), key=lambda p: -scores[p[0], p[1]]):
        if rows[t] < 0 and not taken[c]:
            rows[t], taken[c] = c, True
    return rows


def overlapping(inter: np.ndarray, truth_area: np.ndarray, cand_area: np.ndarray) -> np.ndarray:
    """Which candidate overlaps which object (see the module rules)."""
    of_object = np.divide(inter, truth_area[:, None], out=np.zeros_like(inter), where=truth_area[:, None] > 0)
    of_candidate = np.divide(inter, cand_area[None, :], out=np.zeros_like(inter), where=cand_area[None, :] > 0)
    return (inter > 0) & ((of_object >= OVERLAP_OF_OBJECT) | (of_candidate >= OVERLAP_OF_CANDIDATE))


def outcomes(inter: np.ndarray, truth_area: np.ndarray, cand_area: np.ndarray, matched: np.ndarray) -> tuple[list[str], list[int]]:
    """Each object's outcome (`OUTCOMES`) and, for split ones, into how many pieces."""
    cover = np.divide(inter, truth_area[:, None], out=np.zeros_like(inter), where=truth_area[:, None] > 0)
    inside = np.divide(inter, cand_area[None, :], out=np.zeros_like(inter), where=cand_area[None, :] > 0)
    near = overlapping(inter, truth_area, cand_area)
    out, pieces = [], []
    for t in range(inter.shape[0]):
        parts = int(((cover[t] >= SPLIT_PART) & (inside[t] >= 0.5)).sum())
        if matched[t] >= 0:
            out.append("found")
        elif not near[t].any():
            out.append("missed")
        elif (c := int(inter[t].argmax())) >= 0 and cover[t, c] >= MERGE_COVER and (np.delete(cover[:, c], t) >= MERGE_COVER).any():
            out.append("merged")
        elif parts >= 2:
            out.append("split")
        else:
            out.append("poor outline")
        pieces.append(parts if out[-1] == "split" else 0)
    return out, pieces


def extra_outcomes(inter: np.ndarray, truth_area: np.ndarray, cand_area: np.ndarray, matched: np.ndarray) -> list[str]:
    """What each candidate is (`EXTRA`): right (matched), or why not."""
    cover = np.divide(inter, truth_area[:, None], out=np.zeros_like(inter), where=truth_area[:, None] > 0)
    inside = np.divide(inter, cand_area[None, :], out=np.zeros_like(inter), where=cand_area[None, :] > 0)
    near = overlapping(inter, truth_area, cand_area)
    out = []
    for c in range(inter.shape[1]):
        if matched[c] >= 0:
            out.append("right")
        elif not near[:, c].any():
            out.append("nothing there")
        elif (cover[:, c] >= MERGE_COVER).sum() >= 2:
            out.append("covers several objects")
        elif (inside[:, c] >= 0.5).any():
            out.append("piece of an object")
        else:
            out.append("poor outline")
    return out


def _band(mask: np.ndarray, width: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    inner = cv2.erode(np.pad(mask, width).astype(np.uint8), kernel)[width:-width, width:-width].astype(bool)
    return mask & ~inner


def boundary_iou(truth: Truth, candidates: Candidates, i: int) -> float:
    """Boundary IoU of an object and one candidate, in the box around both."""
    box = candidates.boxes[i]
    if box is None:
        return 0.0
    y0, y1 = min(truth.box[0].start, box[0].start), max(truth.box[0].stop, box[0].stop)
    x0, x1 = min(truth.box[1].start, box[1].start), max(truth.box[1].stop, box[1].stop)
    union = (slice(y0, y1), slice(x0, x1))
    g = np.zeros((y1 - y0, x1 - x0), bool)
    g[truth.box[0].start - y0: truth.box[0].stop - y0, truth.box[1].start - x0: truth.box[1].stop - x0] = truth.mask
    p = candidates.mask(i, union)
    width = max(2, int(round(BAND * 2 * np.sqrt(truth.area / np.pi))))
    gb, pb = _band(g, width), _band(p, width)
    total = (gb | pb).sum()
    return float((gb & pb).sum() / total) if total else 0.0


def inside_box(x: float, y: float, box: Box) -> bool:
    return box[1].start <= x < box[1].stop and box[0].start <= y < box[0].stop


@dataclass
class SetScore:
    """One candidate set (all proposals, or the counted detections) against the objects of one field."""
    best_iou: np.ndarray  # per object: its best IoU with any candidate of the set
    best: np.ndarray  # per object: that candidate (index into the set's columns), -1 when none overlaps
    outcome: list[str]
    pieces: list[int]
    found_low: np.ndarray  # per object: found at IOU_LOW
    boundary: list[float]  # per object: boundary IoU with its best candidate (nan when none)
    extra: list[str]  # per candidate of the set: `EXTRA`
    extra_low: np.ndarray  # per candidate: matched at IOU_LOW


def score_set(truths: list[Truth], candidates: Candidates, inter: np.ndarray, columns: list[int]) -> SetScore:
    sub = inter[:, columns] if columns else np.zeros((len(truths), 0))
    truth_area = np.array([t.area for t in truths], float)
    cand_area = candidates.area[columns] if columns else np.zeros(0)
    scores = iou(sub, truth_area, cand_area)
    real = np.array([t.value != UNSURE for t in truths], bool)
    matched = match(np.where(real[:, None], scores, 0), IOU)
    matched_low = match(np.where(real[:, None], scores, 0), IOU_LOW)
    by_candidate = np.full(len(columns), -1)
    by_candidate_low = np.full(len(columns), -1)
    by_candidate[matched[matched >= 0]] = np.nonzero(matched >= 0)[0]
    by_candidate_low[matched_low[matched_low >= 0]] = np.nonzero(matched_low >= 0)[0]
    outcome, pieces = outcomes(sub, truth_area, cand_area, matched)
    near = overlapping(sub, truth_area, cand_area)
    best = np.array([int(scores[t].argmax()) if len(columns) and near[t].any() else -1 for t in range(len(truths))])
    best_iou = np.array([scores[t, b] if b >= 0 else 0.0 for t, b in enumerate(best)])
    boundary = [boundary_iou(truth, candidates, columns[b]) if b >= 0 else np.nan for truth, b in zip(truths, best)]
    extra = extra_outcomes(sub[real], truth_area[real], cand_area, by_candidate)
    # a counted candidate that overlaps only objects people were unsure about is left out of the precision
    unsure_only = near[~real].any(axis=0) & ~near[real].any(axis=0) if (~real).any() else np.zeros(len(columns), bool)
    extra = ["unsure" if u and e == "nothing there" else e for e, u in zip(extra, unsure_only)]
    return SetScore(best_iou, best, outcome, pieces, matched_low >= 0, boundary, extra, by_candidate_low >= 0)


def end_to_end(proposal: str, detection: str) -> str:
    """One outcome per object over both sets: counted (found among the detections), rejected (a fitting proposal the
    model did not count), else what went wrong with the proposals."""
    if detection == "found":
        return "counted"
    return "rejected" if proposal == "found" else proposal
