"""How a core's stain images line up: the matrix from one stain image's pixels to another's, for carrying what people
mark on one stain onto the others (the grey / white matter of `labels.matter`, painted on NeuN and carried to 6E10 and
AT8).

The project's registration table (input `registration`, `rigid.register_triplet_manifest`'s output: per core each
moving stain against a reference stain, the native matrix from the moving image's pixels to the reference's) is the
first source. A core image pair it has no row for is registered the same way (`rigid.register_previews` on the two
whole-core pictures, `rigid.native_matrix`) once and kept in the project's cache (`registration`). Nothing here is an
analysis input.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stainid import cache

PREVIEW_PX = 1024  # the whole-core pictures registered when the table has no row


@dataclass(frozen=True)
class Carry:
    """From `source`'s pixels to `target`'s: `matrix` (3 x 3), the registration's status ("pass", "review", "fail",
    "same" for one image) and where it came from ("table", "computed", "same")."""
    matrix: np.ndarray
    status: str
    how: str
    dice: float | None = None

    def apply(self, rings) -> list[list[list[float]]]:
        """Rings (lists of [x, y] in the source's pixels) in the target's pixels."""
        out = []
        for ring in rings:
            points = np.asarray(ring, np.float64).reshape(-1, 2)
            moved = points @ self.matrix[:2, :2].T + self.matrix[:2, 2]
            out.append([[round(float(x), 1), round(float(y), 1)] for x, y in moved])
        return out

    def to_api(self) -> dict:
        return {"status": self.status, "how": self.how, "dice": None if self.dice is None else round(float(self.dice), 3),
                "matrix": [[round(float(v), 6) for v in row] for row in self.matrix[:2]]}


def _square(rows) -> np.ndarray:
    return np.vstack([np.asarray(rows, np.float64).reshape(2, 3), [0.0, 0.0, 1.0]])


def table(project) -> dict[tuple[str, str], dict]:
    """(core_id, moving stain) -> its row of the registration table ({} without one)."""
    path = Path(project.input("registration"))

    def make() -> dict:
        import csv  # noqa: PLC0415

        with path.open(newline="", encoding="utf-8") as handle:
            return {(str(r.get("core_id") or r.get("annotation_id")), str(r["moving_stain"])): r for r in csv.DictReader(handle)}

    if not path.exists():
        return {}
    return cache.memo(("registration table", str(path)), [path], make)


def _row_matrix(row: dict) -> np.ndarray:
    return _square([[row[f"native_m{i}{j}"] for j in range(3)] for i in range(2)])


def _to_reference(project, core_id: str, stain: str) -> tuple[np.ndarray, str, str, float | None] | None:
    """The matrix from a stain image's pixels to its reference stain's, that stain, the status and the dice."""
    row = table(project).get((core_id, stain))
    if row is None:
        return None
    dice = float(row["registered_tissue_dice"]) if row.get("registered_tissue_dice") not in (None, "") else None
    return _row_matrix(row), str(row["reference_stain"]), str(row.get("status") or ""), dice


def _from_table(project, core_id: str, source: str, target: str) -> Carry | None:
    up = _to_reference(project, core_id, source)
    down = _to_reference(project, core_id, target)
    worst = {"pass": 0, "review": 1, "fail": 2}
    if up and up[1] == target:  # source -> target directly
        return Carry(up[0], up[2], "table", up[3])
    if down and down[1] == source:  # target -> source: the inverse
        return Carry(np.linalg.inv(down[0]), down[2], "table", down[3])
    if up and down and up[1] == down[1]:  # both against one reference
        status = max(up[2], down[2], key=lambda s: worst.get(s, 3))
        dices = [d for d in (up[3], down[3]) if d is not None]
        return Carry(np.linalg.inv(down[0]) @ up[0], status, "table", min(dices) if dices else None)
    return None


def _picture(state, core_id: str, stain: str) -> np.ndarray:
    import cv2  # noqa: PLC0415

    from stainid.api import pyramid  # noqa: PLC0415

    path = pyramid.thumbnail(state, core_id, stain, PREVIEW_PX)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _native(state, core_id: str, stain: str) -> tuple[int, int]:
    row = state.core_image(core_id, stain)
    return int(float(row["native_height_px"])), int(float(row["native_width_px"]))


def _computed(state, core_id: str, source: str, target: str) -> Carry:
    """Registered now from the two whole-core pictures (target the reference), kept in the cache."""
    return _registered(state.root, core_id, source, target, lambda c, s: _picture(state, c, s), lambda c, s: _native(state, c, s))


def _registered(project_root, core_id: str, source: str, target: str, picture, native) -> Carry:
    """The pair registered from `picture(core, stain)` (RGB whole-core pictures, `PREVIEW_PX` across) and kept in the
    project's `registration` cache, or what is kept there already (the app and `carry_project` share it)."""
    from stainid import disk_cache  # noqa: PLC0415
    from stainid.registration import rigid  # noqa: PLC0415

    kept = disk_cache.folder(project_root, "registration") / f"{core_id}_{source}_to_{target}.json"
    if kept.exists():
        try:
            found = json.loads(kept.read_text())
            disk_cache.used(kept)
            return Carry(_square(found["matrix"]), found["status"], "computed", found.get("dice"))
        except (ValueError, KeyError):
            pass
    reference, moving = picture(core_id, target), picture(core_id, source)
    result = rigid.register_previews(reference, moving)
    matrix = rigid.native_matrix(result.matrix, reference.shape[:2], native(core_id, target), native(core_id, source))
    carry = Carry(_square(matrix), result.status, "computed", float(result.final_dice))
    from stainid.tables import write_text_atomic  # noqa: PLC0415

    write_text_atomic(kept, json.dumps({"matrix": carry.matrix[:2].tolist(), "status": carry.status, "dice": carry.dice}))
    disk_cache.wrote(kept)
    return carry


def carry(state, core_id: str, source: str, target: str) -> Carry:
    """From `source`'s pixels to `target`'s on one core: the registration table, else registered now."""
    if source == target:
        return Carry(np.eye(3), "same", "same", 1.0)
    found = _from_table(state.project, core_id, source, target)
    return found if found is not None else _computed(state, core_id, source, target)


def _project_native(project):
    from stainid.tables import read_csv  # noqa: PLC0415

    path = project.input("core_images")
    rows = {(str(r["core_id"]), str(r["stain"])): r for r in read_csv(path)} if path.exists() else {}

    def native(core_id: str, stain: str) -> tuple[int, int]:
        row = rows[(core_id, stain)]
        return int(float(row["native_height_px"])), int(float(row["native_width_px"]))

    return native, rows


def project_picture(project):
    """`picture(core, stain)` for `carry_project` without the app: the app's whole-core thumbnail base when it is kept,
    else the core image read at 1/8, shrunk to `PREVIEW_PX` across (as the app's pictures are)."""
    import cv2  # noqa: PLC0415

    from stainid import disk_cache  # noqa: PLC0415
    from stainid.imaging.core_images import read_bgr  # noqa: PLC0415
    from stainid.imaging.thumbnails import base_path, shrink  # noqa: PLC0415

    _, rows = _project_native(project)

    def picture(core_id: str, stain: str) -> np.ndarray:
        base = base_path(disk_cache.folder(project.root, "thumbs"), core_id, stain)
        bgr = cv2.imread(str(base), cv2.IMREAD_COLOR) if base.exists() else None
        if bgr is None:
            bgr = read_bgr(project.root / rows[(core_id, stain)]["image_path"], cv2.IMREAD_REDUCED_COLOR_8)
        if bgr is None:
            raise FileNotFoundError(f"no picture of {core_id} {stain}")
        return cv2.cvtColor(shrink(bgr, PREVIEW_PX), cv2.COLOR_BGR2RGB)

    return picture


def carry_project(project, core_id: str, source: str, target: str, picture=None) -> Carry:
    """`carry` without the app (a pipeline step): the registration table, else the pair registered from the whole-core
    pictures (`picture`, default `project_picture`) and kept in the same cache as the app's, so both carry with the
    same matrix."""
    if source == target:
        return Carry(np.eye(3), "same", "same", 1.0)
    found = _from_table(project, core_id, source, target)
    if found is not None:
        return found
    native, _ = _project_native(project)
    return _registered(project.root, core_id, source, target, picture or project_picture(project), native)
