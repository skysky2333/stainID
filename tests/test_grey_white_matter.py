# ruff: noqa: F811  (the rounds fixture is shared from test_annotation_rounds)
"""The "Grey / white matter" round (labels/matter.py, api/routes/matter.py): a core with its stain images, NeuN first;
the white matter painted on NeuN is carried to the other stains through the registration (registration/cores.py) and
saved per stain image, "accept as carried" in one click; one person on a core at a time; a finished core opens to look
at; the Cores view shows each stain image's one say; the analysis' layers are untouched."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi")

from test_annotation_rounds import _plain_mask, rounds_project  # noqa: E402,F401
from test_mark_bad_areas import CORE, _square  # noqa: E402

from stainid import exclusions  # noqa: E402
from stainid.api import state  # noqa: E402
from stainid.labels import matter, store  # noqa: E402
from stainid.registration import cores as registration  # noqa: E402

COLUMNS = ["core_id", "moving_stain", "reference_stain", "status", "registered_tissue_dice"] + [f"native_m{i}{j}" for i in range(2) for j in range(3)]


def _registration(project, shift_6e10: float = 80.0) -> None:
    """The registration table: 6E10 lies `shift_6e10` px left of NeuN (NeuN x = 6E10 x + shift), AT8 as NeuN."""
    import pandas as pd

    path = project.input("registration")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [[CORE, "6E10", "NeuN", "pass", 0.95, 1, 0, shift_6e10, 0, 1, 0], [CORE, "AT8", "NeuN", "review", 0.7, 1, 0, 0, 0, 1, 0]]
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)


def test_the_carry_follows_the_registration_table(rounds_project):
    project = state.get_state().project
    _registration(project)
    s = state.get_state()
    to_6e10 = registration.carry(s, CORE, "NeuN", "6E10")
    assert to_6e10.how == "table" and to_6e10.status == "pass"
    assert to_6e10.apply([[[100, 10], [200, 10], [200, 50]]]) == [[[20.0, 10.0], [120.0, 10.0], [120.0, 50.0]]]
    between = registration.carry(s, CORE, "6E10", "AT8")  # both against NeuN: through it, the weaker status
    assert between.apply([[[0, 0], [1, 1], [0, 1]]])[0][0] == [80.0, 0.0] and between.status == "review"
    assert registration.carry(s, CORE, "NeuN", "NeuN").how == "same"


def test_white_matter_is_painted_on_neun_carried_to_the_other_stains_and_saved_per_stain_image(rounds_project):
    admin, (ann, ben), _ = rounds_project
    project = state.get_state().project
    _plain_mask(project)
    _registration(project)
    before = exclusions.digest(project, CORE, "6E10")
    made = admin.post("/api/labels/sessions", json={"stain": "all", "kind": "matter", "open": True, "seed": 1})
    assert made.status_code == 200, made.text
    meta = made.json()
    name = meta["name"]
    assert meta["kind"] == "matter" and meta["stains_of"][CORE] == ["NeuN", "6E10", "AT8"]
    picker = ann.get(f"/api/annotate/{name}").json()["meta"]["core_rows"]
    assert [(r["core_id"], r["tma"], r["state"]) for r in picker] == [(CORE, "1", "not_started")]

    assert ann.get(f"/api/annotate/{name}/matter/next").json()["core_id"] == CORE
    core = ann.get(f"/api/annotate/{name}/matter/cores/{CORE}").json()
    assert core["source"] == "NeuN" and not core["read_only"] and [i["stain"] for i in core["images"]] == ["NeuN", "6E10", "AT8"]
    assert "group" not in str(core) and "disease" not in str(core)  # blinded
    taken = ben.get(f"/api/annotate/{name}/matter/cores/{CORE}")
    assert taken.status_code == 409 and "Ann is working on this core" in taken.text  # one person on a core at a time
    assert ben.get(f"/api/annotate/{name}/matter/next").json()["core_id"] is None
    assert ann.get(f"/api/annotate/{name}/cores/{CORE}/NeuN/objects").status_code == 200  # the neurons as a guide
    assert ann.get(f"/api/annotate/{name}/matter/cores/{CORE}/NeuN/area").json()["from"] == "new"
    assert ann.get(f"/api/annotate/{name}/matter/cores/{CORE}/6E10/area").json()["from"] == "waiting"
    early = ann.post(f"/api/annotate/{name}/matter/cores/{CORE}/6E10/save", json={"accepted": True})
    assert early.status_code == 400  # NeuN first: the others start from it

    # white matter on NeuN: cells rows 100-149, columns 40-79 (core px 320-640 across, 800-1200 down)
    neun = ann.post(f"/api/annotate/{name}/matter/cores/{CORE}/NeuN/save", json={"png": _square(150, (100, 150, 40, 80)), "scale": 8})
    assert neun.status_code == 200, neun.text
    assert not neun.json()["finished"]
    carried = ann.get(f"/api/annotate/{name}/matter/cores/{CORE}/6E10/area").json()
    assert carried["from"] == "carried" and carried["carried_from"] == "NeuN" and carried["registration"]["how"] == "table"
    accepted = ann.post(f"/api/annotate/{name}/matter/cores/{CORE}/6E10/save", json={"accepted": True})
    assert accepted.status_code == 200, accepted.text
    labels = store.current(project.output("reviews"))
    on_6e10 = matter.final(labels, CORE, "6E10")
    assert on_6e10.carried_from == "NeuN" and on_6e10.accepted and on_6e10.person == "Ann"
    xs = [p[0] for ring in on_6e10.rings for p in ring]
    neun_xs = [p[0] for ring in matter.final(labels, CORE, "NeuN").rings for p in ring]
    assert min(xs) == pytest.approx(min(neun_xs) - 80, abs=0.5)  # carried exactly, 80 px to the left
    shape = (150, 150)
    assert on_6e10.mask(shape)[120, 35] and not on_6e10.mask(shape)[120, 75] and matter.final(labels, CORE, "NeuN").mask(shape)[120, 75]

    # AT8 adjusted by hand: a smaller area; then the core is finished and opens to look at
    at8 = ann.post(f"/api/annotate/{name}/matter/cores/{CORE}/AT8/save", json={"png": _square(150, (120, 150, 40, 80)), "scale": 8})
    assert at8.status_code == 200 and at8.json()["finished"] and at8.json()["progress"]["decided"] == 1
    again = ann.get(f"/api/annotate/{name}/matter/cores/{CORE}").json()
    assert again["read_only"] and all(i["saved_by_me"] for i in again["images"])
    assert ann.get(f"/api/annotate/{name}/matter/cores/{CORE}?edit=true").json()["read_only"] is False  # to work on it again
    assert ann.get(f"/api/annotate/{name}/matter/cores/{CORE}/AT8/area").json()["from"] == "saved"
    picker = ann.get(f"/api/annotate/{name}").json()["meta"]["core_rows"]
    assert picker[0]["state"] == "done" and picker[0]["done_by"] == ["Ann"]

    # one say per stain image, in the Cores view; bad areas and the analysis' layer untouched
    notes = admin.get(f"/api/cores/{CORE}/6E10/annotations").json()
    assert notes["matter"]["carried_from"] == "NeuN" and notes["matter"]["regions"][0]["kind"] == matter.AREA
    assert admin.get(f"/api/cores/{CORE}/AT8/annotations").json()["matter"]["accepted"] is False
    assert exclusions.digest(project, CORE, "6E10") == before and exclusions.edits(project, CORE, "6E10") == []
    overview = admin.get("/api/labels/matter").json()
    assert overview["cores_done"] == 1 and [s["stain"] for s in overview["stains"]][0] == "NeuN"


def test_two_people_s_white_matter_is_compared(rounds_project):
    project = state.get_state().project
    a = matter.records(CORE, "NeuN", [[[0, 0], [400, 0], [400, 400], [0, 400]]], [False], {"source": "Ann", "session": "s", "tile_id": CORE}, None, False, None)
    b = matter.records(CORE, "NeuN", [[[0, 0], [400, 0], [400, 200], [0, 200]]], [False], {"source": "Ben", "session": "s", "tile_id": CORE}, None, False, None)
    store.append(project.output("reviews"), a + b)
    labels = store.current(project.output("reviews"))
    found = matter.checked(labels)
    assert found[(CORE, "NeuN")] == {"Ann", "Ben"}
    result = matter.agreement(project, labels, found)
    assert result["pairs"] == 1 and result["agree"] == 1 and result["mean_iou"] == pytest.approx(0.5, abs=0.03)
    assert np.isclose(result["agreement"], 1.0)
