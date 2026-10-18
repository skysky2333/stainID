"""Generous DAB proposals: the component tree of the smoothed DAB over several thresholds, plus a local-background pass.

At each level (a multiple of the slide threshold) the stained pixels form components; a component at a higher level lies
inside one at the lower level, which makes a tree. Walking it from the roots:
- `split` mode (cells: NeuN, AT8 bodies): a node holding two or more significant children is a merge of objects, so its
  children are walked separately; along a chain of single children the object is the node whose boundary has the
  strongest DAB edge ("best-fitting component"), among nodes of an allowed size; a chain that never gets small enough is
  cut by a marker watershed on its distance transform (merged neurons above 650 µm² are split instead of dropped),
  unless it looks like a stained rim or band rather than touching cells (`cell_pieces`);
- `roots` mode (plaques: a dense core inside a diffuse halo is one object): the lowest node of allowed size.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.segmentation import watershed

from stainid.v2.geometry import Region, components, disk


@dataclass
class Node:
    level: float
    box: tuple[slice, slice]
    mask: np.ndarray
    area: int
    children: list["Node"] = field(default_factory=list)

    def region(self, source: str) -> Region:
        return Region(self.box[0].start, self.box[1].start, self.mask, source, level=self.level)


def edge_score(field_, region: Region) -> float:
    """Mean DAB gradient on the region's boundary, in slide thresholds per pixel."""
    box, local = region.padded(1, field_.shape)
    eroded = cv2.erode(local.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    boundary = local & ~eroded
    if not boundary.any():
        return 0.0
    return float(field_.gradient[box][boundary].mean() / max(field_.threshold, 1e-6))


def level_mask(field_, level: float, allowed: np.ndarray) -> np.ndarray:
    mask = (field_.smooth >= level * field_.threshold) & allowed
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, disk(1)).astype(bool) & allowed


def build_tree(field_, levels: list[float], allowed: np.ndarray, minimum_px: int) -> list[Node]:
    """The roots of the component tree (components of at least `minimum_px` at each level, linked to their parent)."""
    levels = sorted(levels)
    previous_labels, previous_nodes, roots = None, {}, []
    for level in levels:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(level_mask(field_, level, allowed).astype(np.uint8), 8)
        nodes = {}
        for index in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[index])
            if area < minimum_px:
                continue
            box = (slice(y, y + h), slice(x, x + w))
            local = labels[box] == index
            node = Node(level, box, local, area)
            nodes[index] = node
            parent = None
            if previous_labels is not None:
                under = previous_labels[box][local]
                under = under[under > 0]
                if under.size:
                    parent = previous_nodes.get(int(np.bincount(under).argmax()))
            (parent.children if parent is not None else roots).append(node)
        previous_labels, previous_nodes = labels, nodes
    return roots


CELL_SPLIT = {"max_edge_fraction": 0.3, "max_aspect": 4.0, "min_radius_um": 3.0}


def watershed_pieces(node: Node, minimum_px: int, min_radius_px: float = 3.0) -> list[np.ndarray]:
    """A merged component cut at the necks of its distance transform (one marker per local maximum at least
    `min_radius_px` from the edge, 19 px apart as the legacy NeuN watershed): the pieces as local masks."""
    mask = np.pad(node.mask, 1)
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    peaks = (distance == cv2.dilate(distance, np.ones((19, 19), np.uint8))) & (distance >= max(3.0, min_radius_px))
    count, markers = cv2.connectedComponents(peaks.astype(np.uint8))
    if count <= 2:
        return []
    pieces = watershed(-distance, markers, mask=mask)[1:-1, 1:-1]
    return [pieces == k for k in range(1, count) if (pieces == k).sum() >= minimum_px]


def cell_pieces(field_, node: Node, minimum_px: int, maximum_px: int, rules: dict | None = None) -> list[Region]:
    """The cells of a merged component, or [] when it does not look like touching cells: a brown rim along a tissue edge
    or hole, a long band (minimum-area rectangle more than `max_aspect` long) or too thin to hold a cell body."""
    rules = {**CELL_SPLIT, **(rules or {})}
    ps = field_.pixel_size_um
    if field_.edge_band[node.box][node.mask].mean() >= rules["max_edge_fraction"]:
        return []
    contours, _ = cv2.findContours(np.pad(node.mask, 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (_, _), (w, h), _ = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    if max(w, h) > rules["max_aspect"] * max(min(w, h), 1.0):
        return []
    radius_px = rules["min_radius_um"] / ps
    out = []
    for piece in watershed_pieces(node, minimum_px, radius_px):
        ys, xs = np.nonzero(piece)
        local = piece[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
        if local.sum() <= maximum_px and cv2.distanceTransform(np.pad(local, 1).astype(np.uint8), cv2.DIST_L2, 5).max() >= radius_px:
            out.append(Region(int(node.box[0].start + ys.min()), int(node.box[1].start + xs.min()), local.copy(), f"split@{node.level:g}",
                              level=node.level))
    return out


def select(field_, roots: list[Node], minimum_px: int, maximum_px: int, significant_px: int, split_max_px: int | None,
           mode: str, split_rules: dict | None = None) -> list[Region]:
    out: list[Region] = []

    def roots_mode(node: Node) -> None:
        if minimum_px <= node.area <= maximum_px:
            out.append(node.region(f"dab@{node.level:g}"))
        elif node.area > maximum_px:
            for child in node.children:
                roots_mode(child)

    def split_mode(node: Node) -> None:
        chain = [node]
        while True:
            significant = [c for c in chain[-1].children if c.area >= significant_px]
            if len(significant) != 1:
                break
            chain.append(significant[0])
        if len(significant) >= 2:  # the chain is a merge of the objects below it: walk them, not the merge
            for child in significant:
                split_mode(child)
            return
        fitting = [n for n in chain if minimum_px <= n.area <= maximum_px]
        if fitting:
            scored = [(edge_score(field_, n.region("")), -n.level, i) for i, n in enumerate(fitting)]
            best = fitting[max(scored)[2]]
            out.append(best.region(f"dab@{best.level:g}"))
        elif split_max_px and maximum_px < chain[-1].area <= split_max_px:  # cut the tightest (darkest) node of the merge
            out.extend(cell_pieces(field_, chain[-1], minimum_px, maximum_px, split_rules))

    for root in roots:
        (roots_mode if mode == "roots" else split_mode)(root)
    return out


def tree_regions(field_, levels: list[float], allowed: np.ndarray, minimum_um2: float, maximum_um2: float, significant_um2: float,
                 split_max_um2: float | None, mode: str, split_rules: dict | None = None) -> list[Region]:
    pixel_area = field_.pixel_size_um ** 2
    minimum_px = max(1, round(minimum_um2 / pixel_area))
    significant_px = max(1, round(significant_um2 / pixel_area))
    roots = build_tree(field_, levels, allowed, min(minimum_px, significant_px))
    return select(field_, roots, minimum_px, round(maximum_um2 / pixel_area), significant_px,
                  None if not split_max_um2 else round(split_max_um2 / pixel_area), mode, split_rules)


def stained_level(field_, smooth: np.ndarray, level: float, contrast: float, valid: np.ndarray, smooth_um: float | None = None) -> float:
    """`level` × the slide threshold, raised to the median of `smooth` over `valid` tissue + `contrast` × the threshold
    where the background is darker (a pale halo stands out from its tissue, not from glass). The median is the tile's
    standard crop's (`Field.standard`); for a local window cut around an object, of the tile's own field
    (`Field.reference`: its DAB smoothed over `smooth_um`, its tissue outside the excluded areas)."""
    out = float(level) * field_.threshold
    if float(contrast) <= 0:
        return out
    own = getattr(field_, "reference", None)
    if own is not None and smooth_um is not None:
        key = ("stained_median", float(smooth_um))
        if key not in own.memo:
            mine = own.tissue & ~own.exclusion
            if (mine & own.standard).sum() >= 100:
                mine = mine & own.standard
            blurred = cv2.GaussianBlur(own.dab, (0, 0), float(smooth_um) / own.pixel_size_um)
            own.memo[key] = float(np.median(blurred[mine])) if mine.sum() >= 100 else None
        median = own.memo[key]
        return max(out, median + float(contrast) * field_.threshold) if median is not None else out
    standard = getattr(field_, "standard", None)  # the standard crop's tissue (a wider detection window moves no level)
    if standard is not None and (valid & standard).sum() >= 100:
        valid = valid & standard
    if valid.sum() >= 100:
        out = max(out, float(np.median(smooth[valid])) + float(contrast) * field_.threshold)
    return out


def coarse_regions(field_, smooth_um: float, level: float, contrast: float, allowed: np.ndarray, minimum_um2: float,
                   maximum_um2: float, source: str = "coarse") -> list[Region]:
    """Components of the DAB smoothed over `smooth_um` at `stained_level`: a pale diffuse plaque as one region, where at the
    pixel scale it is a cloud of specks below the threshold."""
    pixel_area = field_.pixel_size_um ** 2
    smooth = cv2.GaussianBlur(field_.dab, (0, 0), float(smooth_um) / field_.pixel_size_um)
    at = stained_level(field_, smooth, level, contrast, allowed, smooth_um)
    mask = cv2.morphologyEx(((smooth >= at) & allowed).astype(np.uint8), cv2.MORPH_CLOSE, disk(1)).astype(bool) & allowed
    lo, hi = max(1, round(minimum_um2 / pixel_area)), round(maximum_um2 / pixel_area)
    return [Region(box[0].start, box[1].start, local, source, level=None) for box, local, area in components(mask) if lo <= area <= hi]


def walk_roots(field_, levels: list[float], allowed: np.ndarray, minimum_um2: float, maximum_um2: float, significant_um2: float,
               keep) -> None:
    """`roots` mode with a say for the caller: the lowest node of allowed size is offered to `keep(region)`; when it
    returns False (the node merges objects already kept, or is mostly covered by them) its children are offered in its
    place, so a faint deposit inside a large faint field that joins it to other deposits is still proposed."""
    pixel_area = field_.pixel_size_um ** 2
    minimum_px = max(1, round(minimum_um2 / pixel_area))
    maximum_px = round(maximum_um2 / pixel_area)
    roots = build_tree(field_, levels, allowed, min(minimum_px, max(1, round(significant_um2 / pixel_area))))
    stack = list(reversed(roots))
    while stack:
        node = stack.pop()
        if node.area > maximum_px or (node.area >= minimum_px and not keep(node.region(f"dab@{node.level:g}"))):
            stack.extend(reversed(node.children))


def local_background_regions(field_, contrast: float, allowed: np.ndarray, minimum_um2: float, maximum_um2: float) -> list[Region]:
    """Components standing `contrast` × threshold above the 60 µm local background (and above half the threshold)."""
    pixel_area = field_.pixel_size_um ** 2
    mask = ((field_.smooth - field_.local_background) >= contrast * field_.threshold) & (field_.smooth >= 0.5 * field_.threshold) & allowed
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, disk(1)).astype(bool) & allowed
    lo, hi = max(1, round(minimum_um2 / pixel_area)), round(maximum_um2 / pixel_area)
    return [Region(box[0].start, box[1].start, local, "localbg", level=None) for box, local, area in components(mask) if lo <= area <= hi]


__all__ = ["CELL_SPLIT", "build_tree", "cell_pieces", "coarse_regions", "edge_score", "local_background_regions", "select", "stained_level",
           "tree_regions", "walk_roots", "watershed_pieces"]
