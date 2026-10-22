"""The "Grey / white matter" round: people paint the white matter of each core (to leave out), core by core, first on
its NeuN image (detected neurons drawn as a guide: grey matter is where they are), then on its other stain images, where
the NeuN area is carried over through the registration (`stainid.registration.cores`) for them to adjust or accept as
carried with one click.

- Its area type is white matter (`AREA`), kept in the label log beside the bad areas (`labels.exclusions`) in the same
  form: rings in core pixels with a hole flag each (`points`, `value` "kinds:white,…;holes:0,1,…"), `tile_id` the core
  id. Unlike a bad-area edit, which adds to or takes from an area, a white-matter label is the whole area of its stain
  image as the person left it (no rings: no white matter): one label per person, session and stain image (id
  `matter:<core>:<stain>`, the latest save winning, the log keeping the history), with `meta` {carried_from: the stain
  it was carried from or None, accepted: saved as carried, registration: the carry's status}.
- The one say per stain image (`final`) is the latest saved label of anyone; the Cores view shows it.
- Every stain image saved also gets a `done` label (value `DONE`, id `done:matter:<core>:<stain>`); a core is finished
  for a person when all its stain images are (`labels.areas.finished`, `next_core` and `progress` work on these sessions
  as on "Mark bad areas"). Each core is done by one person.
- Like every human annotation these areas only train and evaluate models (a grey / white matter model); the analysis
  never reads them.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stainid import exclusions

KIND = "matter"
TITLE = "Grey / white matter"
TASK = "matter"
DONE = "matter"  # the `value` of the `done` label of a stain image whose white matter was saved
AREA = "white"  # the area type: white matter
SOURCE = "matter"
STAIN_ORDER = ("NeuN", "6E10", "AT8")  # NeuN first: the neurons show where the grey matter is
HOLD_SECONDS = 120.0
PRACTICE = "practice-matter"  # the tutorial's practice session (`stainid.tutorial.matter`): nothing is kept
SCALE = exclusions.SCALE  # core px per mask px


def is_matter(meta: dict) -> bool:
    return meta.get("kind") == KIND


def label_id(core_id: str, stain: str) -> str:
    return f"{TASK}:{core_id}:{stain}"


def done_id(core_id: str, stain: str) -> str:
    return f"done:{DONE}:{core_id}:{stain}"


def ordered(stains) -> list[str]:
    order = {s: i for i, s in enumerate(STAIN_ORDER)}
    return sorted(stains, key=lambda s: (order.get(s, 9), s))


def source_of(stains) -> str:
    """The stain painted first, whose area the others start from: NeuN when the core has it."""
    found = ordered(stains)
    return found[0] if found else ""


def checked(labels: pd.DataFrame, stain: str | None = None) -> dict[tuple[str, str], set[str]]:
    """(core_id, stain) -> who saved that stain image's white matter."""
    if labels.empty or "task" not in labels:
        return {}
    done = labels[(labels.task == "done") & (labels.value.fillna("").astype(str) == DONE)]
    if stain is not None:
        done = done[done.stain == stain]
    return {(str(core), str(s)): set(group.source) for (core, s), group in done.groupby(["tile_id", "stain"])}


@dataclass(frozen=True)
class Area:
    """One saved white-matter area of a stain image (rings in core px, a hole flag each)."""
    core_id: str
    stain: str
    person: str
    time: float
    session: str
    rings: tuple
    holes: tuple
    carried_from: str | None = None
    accepted: bool = False
    registration: str | None = None

    def mask(self, shape: tuple[int, int], scale: int = SCALE) -> np.ndarray:
        """The area on a grid of `scale` core px per cell."""
        if not self.rings:
            return np.zeros(shape, bool)
        return exclusions.rasterize(list(self.rings), list(self.holes), shape, scale)

    def to_api(self) -> dict:
        return {"person": self.person, "time": self.time, "session": self.session, "carried_from": self.carried_from,
                "accepted": self.accepted, "registration": self.registration}


def _parts(points, value) -> tuple[list, list[bool]]:
    from stainid.labels import exclusions as marks  # noqa: PLC0415  (the same form as a bad-area edit)

    rings, holes, _ = marks.parts_of({"points": points, "value": value})
    return [r.tolist() for r in rings], holes


def areas_of(labels: pd.DataFrame, core_id: str, stain: str) -> list[Area]:
    """Every saved white-matter area of one stain image in `labels` (one per person and session), oldest first."""
    if labels.empty or "task" not in labels:
        return []
    rows = labels[(labels.task == TASK) & (labels.tile_id.astype(str) == str(core_id)) & (labels.stain == stain)]
    out = []
    for r in rows.itertuples():
        rings, holes = _parts(r.points, r.value)
        meta = r.meta if isinstance(getattr(r, "meta", None), dict) else {}
        out.append(Area(str(core_id), stain, str(r.source), float(r.time or 0.0), str(r.session or ""), tuple(tuple(tuple(p) for p in g) for g in rings),
                        tuple(holes), meta.get("carried_from"), bool(meta.get("accepted")), meta.get("registration")))
    return sorted(out, key=lambda a: (a.time, a.person))


def latest(labels: pd.DataFrame, core_id: str, stain: str, person: str | None = None) -> Area | None:
    """The latest saved area of a stain image (of one person, when given)."""
    found = [a for a in areas_of(labels, core_id, stain) if person is None or a.person == person]
    return found[-1] if found else None


def final(labels: pd.DataFrame, core_id: str, stain: str) -> Area | None:
    """The one say per stain image: the latest area anyone saved."""
    return latest(labels, core_id, stain)


def records(core_id: str, stain: str, rings, holes, stamp: dict, carried_from: str | None, accepted: bool, registration: str | None) -> list[dict]:
    """The labels of one save: the stain image's whole white-matter area and its `done` label."""
    from stainid.labels import exclusions as marks  # noqa: PLC0415

    meta = {"carried_from": carried_from, "accepted": accepted, "registration": registration}
    return [{**stamp, "id": label_id(core_id, stain), "stain": stain, "task": TASK, "points": [list(map(list, r)) for r in rings],
             "value": marks.value_text(list(holes), [AREA] * len(rings)), "meta": meta},
            {**stamp, "id": done_id(core_id, stain), "stain": stain, "task": "done", "value": DONE}]


def png(rings, holes, width: int, height: int, scale: int = SCALE) -> str:
    """The area as a white-on-transparent PNG data URL, one pixel per `scale` core px (what the brush starts from)."""
    import cv2  # noqa: PLC0415

    shape = (max(1, -(-int(height) // scale)), max(1, -(-int(width) // scale)))
    cells = exclusions.rasterize(list(rings), list(holes), shape, scale) if rings else np.zeros(shape, bool)
    rgba = np.zeros((*shape, 4), np.uint8)
    rgba[cells] = 255
    ok, encoded = cv2.imencode(".png", rgba)
    if not ok:
        raise RuntimeError("could not encode the white matter")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode()


def regions(area: Area | None, shape: tuple[int, int] | None, scale: int = SCALE) -> list[dict]:
    """An area as merged polygons in core px ({kind: "white", rings: [outer, hole, …]}), for the Cores view."""
    if area is None or not area.rings or shape is None:
        return []
    codes = area.mask(shape, scale).astype(np.uint8)
    return exclusions.regions_of(codes, scale, {1: AREA})
