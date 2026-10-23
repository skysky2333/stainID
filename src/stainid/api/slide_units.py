"""Whole slides as the unit of the pages people work on (a whole-slide project: `stainid.imageset.slides`).

The pipeline, the label log and the tables keep the regions as their units; the pages show a slide as one picture. This
module moves what the pages need between the two:

- reading: each region's layers (detection's excluded areas, the brush's starting areas, people's areas, white matter,
  detection's objects) put at the region's place on one grid of the slide (`compose`, `moved`);
- writing: what was painted over the whole slide cut into one part per region (`split_pngs`, `split_png`), its rings in
  the region's own pixels, so every label is a region's as it is in a TMA study;
- who finished a slide: the people who finished every one of its regions (`with_slides`).

A slide's masks are coarser than a core's (`mask_scale`: a slide is ~100,000 px across, its brush masks stay ~2,000 px
across in the browser). Nothing here changes a pipeline output.
"""
from __future__ import annotations

import base64
import hashlib
from collections import OrderedDict

import cv2
import numpy as np

from stainid import exclusions
from stainid.api import slide_pyramid
from stainid.api.state import get_state
from stainid.imageset.slides import Region, Slide, clip_ring, split_mask

MASK_SIDE = 2048  # the longest side of a slide's brush mask (mask px)
EXCLUDED_SCALE = 32  # slide px per pixel of a slide's "Excluded from analysis" picture (a core's: 8)
COORDINATES = ("x_core", "y_core")  # the point keys of the annotation answers (core px)
SHAPES = ("points", "rings")  # their polygon keys: lists of [x, y]


def slide_unit(unit_id: str) -> Slide | None:
    """The slide a unit id names in a whole-slide project (None for a core, or in any other project)."""
    return slide_pyramid.slides(get_state()).get(str(unit_id))


def mask_scale(s: Slide) -> int:
    """Slide px per mask px of a slide's brush: the core's (`exclusions.SCALE`), doubled until the mask is at most
    `MASK_SIDE` across (so each is a whole number of the regions' own cells)."""
    scale = exclusions.SCALE
    while max(s.width, s.height) / scale > MASK_SIDE:
        scale *= 2
    return scale


def unit_stains(stains_of: dict[str, list[str]]) -> dict[str, list[str]]:
    """The slides of the regions with analysed tiles (`labels.areas.core_stains`): a slide's one stain each. Empty in
    other projects, whose units stay their cores."""
    slides = slide_pyramid.slides(get_state())
    if not slides:
        return {}
    have = {c for c, stains in stains_of.items() if stains}
    return {s.slide_id: [s.stain] for s in slides.values() if any(c in have for c in s.cores)}


def with_slides(found: dict[tuple[str, str], set[str]]) -> dict[tuple[str, str], set[str]]:
    """`found` ((core, stain) -> who did it) with each slide added: the people who did every one of its regions."""
    slides = slide_pyramid.slides(get_state())
    if not slides:
        return found
    out = dict(found)
    for s in slides.values():
        people: set[str] | None = None
        for core in s.cores:
            people = set(found.get((core, s.stain), set())) if people is None else people & found.get((core, s.stain), set())
            if not people:
                break
        if people:
            out[(s.slide_id, s.stain)] = people
    return out


# ---------------------------------------------------------------- reading: regions onto the slide

def compose(s: Slide, layers, scale: int, fill: int = exclusions.OUTSIDE) -> np.ndarray:
    """Each region's grid of codes (`layers`: (region, codes, its scale) or None) at the region's place on one grid of
    the slide (`scale` slide px per cell, nearest), `fill` where no region is."""
    height, width = -(-s.height // scale), -(-s.width // scale)
    out = np.full((height, width), fill, np.uint8)
    for item in layers:
        if item is None:
            continue
        region, codes, own = item
        w, h = max(1, -(-region.width // scale)), max(1, -(-region.height // scale))
        part = codes if (own == scale and codes.shape == (h, w)) else cv2.resize(codes, (w, h), interpolation=cv2.INTER_NEAREST)
        x0, y0 = region.x // scale, region.y // scale
        part = part[:max(0, height - y0), :max(0, width - x0)]
        target = out[y0:y0 + part.shape[0], x0:x0 + part.shape[1]]
        target[...] = np.where(part != fill, part, target)
    return out


_COMPOSED: OrderedDict[tuple, tuple[str, np.ndarray]] = OrderedDict()


def _kept(key: tuple, version: str, make) -> np.ndarray:
    kept = _COMPOSED.get(key)
    if kept is not None and kept[0] == version:
        _COMPOSED.move_to_end(key)
        return kept[1]
    codes = make()
    _COMPOSED[key] = (version, codes)
    while len(_COMPOSED) > 12:
        _COMPOSED.popitem(last=False)
    return codes


def excluded(project, s: Slide) -> tuple[str, np.ndarray] | None:
    """The slide's "Excluded from analysis" codes at `EXCLUDED_SCALE` (each region's layer, `exclusions.view`) and a
    version; None while no region has its layer."""
    made = [(r, exclusions.digest(project, r.core_id, s.stain)) for r in s.regions]  # cheap: kept until a mask file changes
    if all(d is None for _, d in made):
        return None
    version = hashlib.sha1("|".join(f"{r.core_id}:{d}" for r, d in made).encode()).hexdigest()[:16]

    def make() -> np.ndarray:
        views = [(r, exclusions.view(project, r.core_id, s.stain)) for r, _ in made]
        return compose(s, [(r, v.codes, v.scale) if v is not None else None for r, v in views], EXCLUDED_SCALE)

    return version, _kept(("excluded", str(project.root), s.slide_id), version, make)


def excluded_info(project, s: Slide) -> dict:
    """As `store_view.excluded_info` for a core image: the slide's layer (status, size, scale, version, the share of each
    kind); the picture is `excluded_png`."""
    from stainid.api import store_view  # noqa: PLC0415

    found = excluded(project, s)
    if found is None:
        return {"status": "not_made", "kinds": [store_view.kind_info(k, None) for k in store_view.EXCLUDED_KINDS]}
    version, codes = found
    return {"status": "done", "scale": EXCLUDED_SCALE, "width": int(codes.shape[1]), "height": int(codes.shape[0]), "version": version,
            "kinds": store_view._shares(codes)}


_PNGS: OrderedDict[tuple, bytes] = OrderedDict()


def excluded_png(project, s: Slide) -> bytes | None:
    from stainid.api import store_view  # noqa: PLC0415

    found = excluded(project, s)
    if found is None:
        return None
    key = (str(project.root), s.slide_id, found[0])
    if key not in _PNGS:
        _PNGS[key] = store_view._rgba(found[1])
        while len(_PNGS) > 6:
            _PNGS.popitem(last=False)
    return _PNGS[key]


def current_areas(project, s: Slide, session=None) -> dict:
    """As `store_view.current_areas` for a core image (what the brush starts from: each type's area as it stands, without
    the AI reviewer's edits), over the whole slide at `mask_scale`."""
    from stainid.api import store_view  # noqa: PLC0415

    scale = mask_scale(s)
    made = [(r, exclusions.digest(project, r.core_id, s.stain)) for r in s.regions]
    if all(d is None for _, d in made):
        return {"status": "not_made", "kinds": {}, "scale": scale}
    version = hashlib.sha1("|".join(f"{r.core_id}:{d}:{exclusions.edits_digest(project, r.core_id, s.stain, False, session) if d else '-'}"
                                    for r, d in made).encode()).hexdigest()[:16]

    def make() -> np.ndarray:
        layers = []
        for r, d in made:
            layer = exclusions.final_say(project, r.core_id, s.stain, ai=False, session=session) if d else None
            layers.append((r, layer.codes, layer.scale) if layer is not None else None)
        return compose(s, layers, scale)

    codes = _kept(("current", str(project.root), s.slide_id, str(session)), version, make)
    kinds = {}
    for kind in store_view.PAINTED_TYPES:
        cells = codes == int(kind)
        if cells.any():
            kinds[kind] = _png(cells)
    return {"status": "done", "scale": scale, "x": 0, "y": 0, "width": codes.shape[1] * scale, "height": codes.shape[0] * scale,
            "version": version, "kinds": kinds}


def _png(cells: np.ndarray) -> str:
    rgba = np.zeros((*cells.shape, 4), np.uint8)
    rgba[cells] = 255
    ok, encoded = cv2.imencode(".png", rgba)
    if not ok:
        raise RuntimeError("could not encode a mask")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode()


def moved(value, region: Region):
    """An answer about a region (its points `x_core`, `y_core` and polygons `points`, `rings` in core px) with every
    point and polygon moved into slide px; anything else as it is."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in COORDINATES and isinstance(item, (int, float)) and not isinstance(item, bool) and item == item:
                out[key] = item + (region.x if key == "x_core" else region.y)
            elif key in SHAPES and isinstance(item, list):
                out[key] = _points(item, region)
            else:
                out[key] = moved(item, region)
        return out
    if isinstance(value, list):
        return [moved(v, region) for v in value]
    return value


def _points(item, region: Region):
    """[[x, y], …] (or lists of those) moved; anything else as it is."""
    if not item:
        return item
    first = item[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2 and all(isinstance(v, (int, float)) for v in first[:2]):
        return [[p[0] + region.x, p[1] + region.y, *p[2:]] for p in item]
    if isinstance(first, list):
        return [_points(i, region) for i in item]
    return item


def flat_moved(flat, region: Region) -> list:
    """A flat ring or line [x0, y0, …] moved into slide px."""
    return [v + (region.x if k % 2 == 0 else region.y) for k, v in enumerate(flat)]


def agreement(parts: list[dict | None]) -> dict | None:
    """The regions' agreements (`exclusions.agreement`) as one: the cells summed, the F1 again."""
    found = [p for p in parts if p]
    if not found:
        return None
    out = {k: sum(int(p.get(k) or 0) for p in found) for k in found[0] if k != "f1"}
    people, auto, both = out.get("cells_people_excluded", 0), out.get("cells_auto_excluded", 0), out.get("cells_both", 0)
    out["f1"] = round(2 * both / (people + auto), 3) if people + auto else None
    return out


def people_areas(project, s: Slide, ai: bool = True, session=None) -> dict:
    """As `store_view.people_areas` for a core image, over the whole slide (slide px)."""
    from stainid.api import store_view  # noqa: PLC0415

    parts = [(r, store_view.people_areas(project, r.core_id, s.stain, ai=ai, session=session)) for r in s.regions]
    return {"regions": [g for r, p in parts for g in moved(p.get("regions", []), r)], "agreement": agreement([p.get("agreement") for _, p in parts]),
            "edits": sorted((e for _, p in parts for e in p.get("edits", [])), key=lambda e: e.get("time") or 0)}


# ---------------------------------------------------------------- writing: the slide cut into its regions

def _mask(data: str) -> np.ndarray:
    """A painted mask (a PNG data URL or base64; painted = alpha, else red, above 127)."""
    raw = base64.b64decode(data.split(",", 1)[1] if data.startswith("data:") else data)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("the painted area is not a PNG picture")
    if image.ndim == 3:
        image = image[..., 3] if image.shape[2] == 4 else image[..., 2]
    return image > 127


def _encoded(cells: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", cells.astype(np.uint8) * 255)
    if not ok:
        raise RuntimeError("could not encode a mask")
    return base64.b64encode(encoded.tobytes()).decode()


def _region_rings(part: np.ndarray, scale: float, dx: float, dy: float, region: Region) -> tuple[list, list[bool]]:
    from stainid.labels import exclusions as marks  # noqa: PLC0415

    rings, holes = marks.from_png(_encoded(part), scale)
    out = [clip_ring([[x + dx, y + dy] for x, y in ring], region) for ring in rings]
    return [[[round(x, 1), round(y, 1)] for x, y in ring] for ring in out], holes


def split_pngs(s: Slide, pngs: dict[str, str], scale: float) -> dict[str, tuple[list, list[bool], list[str]]]:
    """What was painted over the slide, one mask per type (`labels.exclusions.from_pngs`), as each region's rings (region
    px), hole flags and types; only the regions something was painted on."""
    from stainid.labels import exclusions as marks  # noqa: PLC0415

    out: dict[str, tuple[list, list[bool], list[str]]] = {}
    for kind, data in pngs.items():
        if kind not in marks.KINDS:
            raise ValueError(f"unknown type of excluded area {kind!r}")
        if not data:
            continue
        for core, (part, dx, dy) in split_mask(_mask(data), scale, s).items():
            rings, holes = _region_rings(part, scale, dx, dy, s.find(core))
            if rings:
                kept = out.setdefault(core, ([], [], []))
                kept[0].extend(rings)
                kept[1].extend(holes)
                kept[2].extend([kind] * len(rings))
    return out


def split_png(s: Slide, data: str, scale: float) -> dict[str, tuple[list, list[bool]]]:
    """A whole area painted over the slide (white matter) as each region's rings (region px) and hole flags: every
    region, those with nothing painted with none (the area of each region is its whole say)."""
    out: dict[str, tuple[list, list[bool]]] = {r.core_id: ([], []) for r in s.regions}
    if not data:
        return out
    for core, (part, dx, dy) in split_mask(_mask(data), scale, s).items():
        out[core] = _region_rings(part, scale, dx, dy, s.find(core))
    return out
