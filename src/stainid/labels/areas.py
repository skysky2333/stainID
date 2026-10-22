"""The "Mark bad areas" round: people check the excluded areas of whole core images, a core at a time with all its stain
images, and fix them with the brush. It is apart from the object rounds because its unit is a whole core image (folds and
bubbles cross tile borders) and its labels are areas, not objects.

- A task is one core (`core_id`) with each of its stain images (6E10, AT8, NeuN, whichever exist); status, edits and
  digests stay per core image (core x stain) underneath.
- A person looks at each stain image and saves the core ("Finish core"): every stain image gets a `done` label
  (`value` `AREAS`, `tile_id` the core id, `stain` the stain, id `done:areas:<core>:<stain>`), also when nothing was
  changed (that confirms detection's areas), and each stain image whose areas were painted or cleared one edit
  (`exclude`, rings in core pixels, `tile_id` the core id; `labels.exclusions`).
- Like every human annotation, people's areas only train and evaluate models: the bad-area detector (`evaluation`:
  detection's automatic areas against people's) and the object models (labels in people's areas are left out). The
  analysis uses detection's automatic areas only (`exclusions.automatic`). A sample of cores is enough.
- Each core is checked by one person; a share (`SECOND_SHARE`, drawn with the session's seed) by a second person
  independently, to measure how well people agree (`agreement`). People's areas take both, the latest edit winning;
  cores where they disagree are listed for an admin.
- Cores come in the session's order: "problems" (the default: those detection's own signals say most likely need fixing
  first, `priority`, then the rest at random) or "random".

The status of a core image (who checked it) is `status.areas_checked`, read from the label log like every review.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from stainid import exclusions
from stainid.labels import status

KIND = "areas"
SOURCE = "areas"
TITLE = "Mark bad areas"
AREAS = status.AREAS
SECOND_SHARE = 0.10
ORDERS = ("problems", "random")
HOLD_SECONDS = 120.0
STAIN_ORDER = ("6E10", "AT8", "NeuN")


def is_areas(meta: dict) -> bool:
    return meta.get("kind") == KIND


def done_id(core_id: str, stain: str) -> str:
    return f"done:{AREAS}:{core_id}:{stain}"


def core_stains(tiles: pd.DataFrame) -> dict[str, list[str]]:
    """Each core with analysed tiles and its stain images, in `STAIN_ORDER`."""
    if tiles.empty:
        return {}
    pairs = tiles[["core_id", "stain"]].astype(str).drop_duplicates()
    order = {s: i for i, s in enumerate(STAIN_ORDER)}
    out: dict[str, list[str]] = {}
    for core, group in pairs.groupby("core_id", sort=True):
        out[core] = sorted(group.stain, key=lambda s: (order.get(s, 9), s))
    return out


def second_checks(cores: list[str], seed: int, share: float = SECOND_SHARE) -> list[str]:
    """The cores (about `share` of them, the same for the same seed) that get a second, independent check."""
    def draw(core: str) -> float:
        return int(hashlib.sha1(f"{seed}:{core}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    return [c for c in cores if draw(c) < share]


def needed(meta: dict, core_id: str) -> int:
    return 2 if core_id in set(meta.get("second") or []) else 1


# ---------------------------------------------------------------- which cores first

def signals(project, core_id: str, stain: str) -> dict | None:
    """Detection's own signs that a core image's excluded areas need a look, from the stage-2 outputs (None before
    them): the share of its analysed area detection excluded, its tissue share, unproposed stained regions per mm²
    and the share of strongly stained area."""
    from stainid import candidates  # noqa: PLC0415

    layer = exclusions.automatic(project, core_id, stain)
    if layer is None:
        return None
    auto = layer.codes
    analysed = max(1, int((auto != exclusions.OUTSIDE).sum()))
    excluded = float(np.isin(auto, (1, 2, 5)).sum()) / analysed
    tissue = 1.0 - float((auto == 3).sum()) / analysed
    mm2 = analysed * (layer.scale * project.pixel_size_um) ** 2 / 1e6
    out = {"excluded_share": excluded, "tissue_share": tissue, "unproposed_per_mm2": 0.0, "dab_share": 0.0}
    path = candidates.path(project, stain, core_id, "unproposed")
    if path.exists():
        out["unproposed_per_mm2"] = len(candidates.unproposed(project, stain, core_id)) / max(mm2, 1e-6)
    tiles = candidates.path(project, stain, core_id, "tiles")
    if tiles.exists():
        frame = candidates.read_table(tiles)
        area = pd.to_numeric(frame.get("valid_area_mm2"), errors="coerce").sum() if "valid_area_mm2" in frame else 0.0
        strong = [c for c in frame.columns if c.startswith("dab_area_mm2_at_")]
        if strong and area:
            out["dab_share"] = float(pd.to_numeric(frame[strong[-1]], errors="coerce").sum() / area)
    return out


def priority(found: dict[str, dict]) -> dict[str, float]:
    """Per core, how likely its areas need fixing: the largest robust z-score among its stain images' signals (a large
    or unusual excluded share, a low tissue share, many unproposed stained regions, a very high stained share)."""
    rows = [(core, s) for core, s in found.items() if s]
    if not rows:
        return {}
    frame = pd.DataFrame([s for _, s in rows], index=[c for c, _ in rows])
    score = pd.Series(0.0, index=frame.index)
    for column, sign in (("excluded_share", 1), ("tissue_share", -1), ("unproposed_per_mm2", 1), ("dab_share", 1)):
        values = sign * frame[column].astype(float)
        mad = float((values - values.median()).abs().median()) or float(values.std() or 1.0) or 1.0
        score = np.maximum(score, (values - values.median()) / (1.4826 * mad))
    return {str(k): float(v) for k, v in score.groupby(level=0).max().items()}


def order(cores: list[str], scores: dict[str, float], how: str, seed: int, flagged: float = 2.0) -> list[str]:
    """The cores in the order they are handed out: "problems" those scoring at least `flagged` first (highest first),
    then the rest at random; "random" all at random."""
    rng = np.random.default_rng(seed)
    shuffled = [cores[i] for i in rng.permutation(len(cores))]
    if how != "problems":
        return shuffled
    first = sorted([c for c in shuffled if scores.get(c, 0.0) >= flagged], key=lambda c: -scores[c])
    return first + [c for c in shuffled if scores.get(c, 0.0) < flagged]


# ---------------------------------------------------------------- how far a session is

def finished(meta: dict, checked: dict[tuple[str, str], set[str]]) -> dict[str, set[str]]:
    """Who finished each core of the session: checked every one of its stain images (in any session)."""
    out = {}
    for core, stains in (meta.get("stains_of") or {}).items():
        people = [checked.get((core, s), set()) for s in stains]
        out[core] = set.intersection(*people) if people else set()
    return out


def next_core(meta: dict, done: dict[str, set[str]], person: str, held: dict[str, set[str]], skip: str = "") -> tuple[str | None, int]:
    """The next core for this person: one they hold already, else the first in the session's order that still needs
    people (a second check is always someone else) and that nobody else is on; and how many cores others hold now."""
    cores = list(meta.get("cores") or [])
    todo = [c for c in cores if person not in done.get(c, set()) and len(done.get(c, set())) < needed(meta, c)]
    mine = [c for c in todo if person in held.get(c, set()) and c != skip]
    if mine:
        return mine[0], 0
    free = [c for c in todo if len(done.get(c, set())) + len(held.get(c, set()) - {person}) < needed(meta, c)]
    pick = [c for c in free if c != skip]
    return (pick[0] if pick else None), len(todo) - len(free)


def progress(meta: dict, done: dict[str, set[str]], held: dict[str, set[str]]) -> dict:
    cores = list(meta.get("cores") or [])
    complete = [c for c in cores if len(done.get(c, set())) >= needed(meta, c)]
    working = [c for c in cores if c not in set(complete) and (done.get(c) or held.get(c))]
    people: dict[str, int] = {}
    for c in cores:
        for p in done.get(c, set()):
            people[p] = people.get(p, 0) + 1
    images = sum(len((meta.get("stains_of") or {}).get(c, [])) for c in complete)
    return {"total": len(cores), "decided": len(complete), "open_now": len(working), "left": len(cores) - len(complete) - len(working),
            "images_checked": images, "second_checks": len(meta.get("second") or []),
            "people": [{"person": p, "decided": n} for p, n in sorted(people.items(), key=lambda kv: (-kv[1], kv[0]))]}


# ---------------------------------------------------------------- agreement and detection's areas

def _person_area(edits: list, person: str, shape: tuple[int, int]) -> np.ndarray:
    """What one person's edits of a core image set, as codes (0 untouched, 1–4 painted, 9 cleared)."""
    out = np.zeros(shape, np.uint8)
    for edit in edits:
        if edit.source != person:
            continue
        for kind, area in exclusions.edit_areas(edit, shape):
            out[area] = 9 if kind == exclusions.INCLUDE or exclusions.removal(kind) is not None else int(kind) if str(kind).isdigit() else 1
    return out


def agreement(project, checked: dict[tuple[str, str], set[str]], min_cells: int = 4) -> dict:
    """How well two people who checked the same core image agree: per image, the overlap (IoU) of the cells either of
    them changed (painted or cleared, the same way); both changing nothing counts as agreeing. Images where they
    disagree (IoU under 0.5) are listed for an admin to settle in the Cores view."""
    rows = []
    for (core, stain), people in sorted(checked.items()):
        if len(people) < 2:
            continue
        layer = exclusions.annotated(project, core, stain)
        if layer is None:
            continue
        first, second = sorted(people)[:2]
        a = _person_area(layer.edits, first, layer.codes.shape)
        b = _person_area(layer.edits, second, layer.codes.shape)
        changed = (a > 0) | (b > 0)
        same = (a == b) & (a > 0)
        n = int(changed.sum())
        iou = 1.0 if n < min_cells else float(same.sum()) / n
        rows.append({"core_id": core, "stain": stain, "people": [first, second], "iou": round(iou, 3), "cells_changed": n, "agree": iou >= 0.5})
    agreeing = [r for r in rows if r["agree"]]
    return {"pairs": len(rows), "agree": len(agreeing), "agreement": (len(agreeing) / len(rows)) if rows else None,
            "mean_iou": float(np.mean([r["iou"] for r in rows])) if rows else None, "disagreements": [r for r in rows if not r["agree"]]}


def evaluation(project, checked: dict[tuple[str, str], set[str]], tiles: pd.DataFrame, by: str = "tma") -> list[dict]:
    """The bad-area detector against people: detection's automatic areas (`exclusions.automatic`) against people's final
    say (`exclusions.final_say`) on the checked core images, per stain, type and `by` (a tile column: "tma" or
    "donor_id"; "all" sums every group): the area people added (mm²), the area they cleared and the pixel F1."""
    group_of = tiles.drop_duplicates(["core_id", "stain"]).set_index(["core_id", "stain"])[by].astype(str).to_dict() if not tiles.empty and by in tiles else {}
    sums: dict[tuple[str, str, int], list[float]] = {}
    for (core, stain) in sorted(checked):
        layer = exclusions.final_say(project, core, stain)
        detected = exclusions.automatic(project, core, stain)
        if layer is None or detected is None:
            continue
        auto = detected.codes
        final = layer.codes
        mm2 = (layer.scale * project.pixel_size_um) ** 2 / 1e6
        for code in (1, 2, 3, 4, 5):
            a, f = auto == code, final == code
            if not (a.any() or f.any()):
                continue
            for group in (group_of.get((core, stain), ""), "all"):
                key = (stain, group, code)
                found = sums.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                found[0] += float((f & ~a).sum()) * mm2  # added
                found[1] += float((a & ~f).sum()) * mm2  # cleared
                found[2] += float((a & f).sum())
                found[3] += float(a.sum())
                found[4] += float(f.sum())
                found[5] += 1
    out = []
    for (stain, group, code), (added, cleared, both, auto_n, final_n, images) in sorted(sums.items()):
        out.append({"stain": stain, by: group, "type": code, "type_name": exclusions.CODES[code], "images": int(images),
                    "added_mm2": round(added, 4), "cleared_mm2": round(cleared, 4),
                    "f1": round(2 * both / (auto_n + final_n), 3) if auto_n + final_n else None})
    return out
