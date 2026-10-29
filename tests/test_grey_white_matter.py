# ruff: noqa: F811  (the rounds fixture is shared from test_annotation_rounds)
"""The "Grey / white matter" round (labels/matter.py, api/routes/matter.py): a core with its stain images, NeuN first;
the white matter painted on NeuN is carried to the other stains through the registration (registration/cores.py) and
saved per stain image, "accept as carried" in one click; one person on a core at a time; a finished core opens to look
at; the Cores view shows each stain image's one say; the analysis' layers are untouched."""
from __future__ import annotations

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


def test_reset_this_core_replaces_every_stain_of_this_person_on_save(rounds_project):
    admin, (ann, _), _ = rounds_project
    project = state.get_state().project
    _plain_mask(project)
    _registration(project)
    name = admin.post("/api/labels/sessions", json={"stain": "all", "kind": "matter", "open": True, "seed": 1}).json()["name"]
    base = f"/api/annotate/{name}/matter/cores/{CORE}"
    ann.get(base)
    assert ann.post(f"{base}/NeuN/save", json={"png": _square(150, (100, 150, 40, 80)), "scale": 8}).status_code == 200
    assert ann.post(f"{base}/6E10/save", json={"accepted": True}).status_code == 200
    fresh = ann.get(f"{base}?fresh=true").json()
    assert not any(i["saved_by_me"] for i in fresh["images"])  # as if Ann had never worked on it
    assert ann.get(f"{base}/NeuN/area?fresh=true").json()["from"] == "new"
    assert ann.get(f"{base}/6E10/area?fresh=true").json()["from"] == "waiting"
    redone = ann.post(f"{base}/NeuN/save", json={"png": _square(150, (120, 150, 40, 80)), "scale": 8, "reset": True})
    assert redone.status_code == 200, redone.text
    labels = store.current(project.output("reviews"))
    mine = labels[(labels.source == "Ann") & (labels.tile_id == CORE)]
    assert set(mine.stain) == {"NeuN"}  # the old 6E10 acceptance is gone; the log keeps it
    assert matter.final(labels, CORE, "6E10") is None
    assert len(store.history(project.output("reviews"))) > len(labels)
    after = ann.get(base).json()
    assert [i["stain"] for i in after["images"] if i["saved_by_me"]] == ["NeuN"]


def test_one_person_per_core_no_second_checks(rounds_project):
    admin, _, _ = rounds_project
    meta = admin.post("/api/labels/sessions", json={"stain": "all", "kind": "matter", "open": True, "seed": 3}).json()
    assert meta["second"] == []
    assert "agreement" not in admin.get("/api/labels/matter").json()


def test_the_tutorial_practises_on_a_neun_core_and_keeps_nothing(rounds_project):
    from fastapi.testclient import TestClient

    from stainid import tutorial

    admin, _, _ = rounds_project
    project = state.get_state().project
    entry = tutorial.add_matter(project.root, CORE, "6E10", "Images: synthetic test project")
    assert entry["stains"] == ["NeuN", "6E10"] and tutorial.matter()["core_id"] == CORE
    guest = TestClient(admin.app)  # anyone: the tutorial needs no account
    assert guest.get("/api/tutorial").json()["matter"]["core_id"] == CORE
    log = store.history(project.output("reviews"))
    name = matter.PRACTICE
    detail = guest.get(f"/api/annotate/{name}").json()["meta"]
    assert detail["kind"] == "matter" and detail["core_rows"][0]["core_id"] == CORE
    core = guest.get(f"/api/annotate/{name}/matter/cores/{CORE}").json()
    assert core["practice"] and not core["read_only"] and core["source"] == "NeuN"
    assert guest.get(f"/api/annotate/{name}/matter/cores/{CORE}/NeuN/area").json()["from"] == "new"
    painted = _square(150, (100, 150, 40, 80))
    saved = guest.post(f"/api/annotate/{name}/matter/cores/{CORE}/NeuN/save", json={"png": painted, "scale": 8})
    assert saved.status_code == 200 and saved.json()["saved"][0]["task"] == matter.TASK  # echoed
    carried = guest.post(f"/api/annotate/{name}/matter/cores/{CORE}/6E10/carry", json={"png": painted, "scale": 8})
    assert carried.status_code == 200 and carried.json()["from"] == "carried" and carried.json()["kinds"].get(matter.AREA)
    assert guest.post(f"/api/annotate/{name}/matter/cores/{CORE}/6E10/save", json={"accepted": True}).status_code == 200
    assert guest.post(f"/api/annotate/{name}/matter/cores/{CORE}/hold").status_code == 200
    assert store.history(project.output("reviews")).equals(log)  # nothing kept


def test_all_grey_matter_saves_every_stain_with_no_white_matter_and_finishes_the_core(rounds_project):
    admin, (ann, _), _ = rounds_project
    project = state.get_state().project
    _plain_mask(project)
    _registration(project)
    name = admin.post("/api/labels/sessions", json={"stain": "all", "kind": "matter", "open": True, "seed": 1}).json()["name"]
    base = f"/api/annotate/{name}/matter/cores/{CORE}"
    ann.get(base)
    assert ann.post(f"{base}/NeuN/save", json={"png": _square(150, (100, 150, 40, 80)), "scale": 8}).status_code == 200  # replaced below
    done = ann.post(f"{base}/all-grey", json={})
    assert done.status_code == 200, done.text
    assert done.json()["finished"] and done.json()["progress"]["decided"] == 1
    labels = store.current(project.output("reviews"))
    for stain in ("NeuN", "6E10", "AT8"):
        area = matter.final(labels, CORE, stain)
        assert area is not None and area.rings == () and area.all_grey  # saved, with no white matter: not "not done yet"
    assert matter.checked(labels)[(CORE, "AT8")] == {"Ann"}
    row = ann.get(f"/api/annotate/{name}").json()["meta"]["core_rows"][0]
    assert row["state"] == "done" and row["done_by_me"]
    assert ann.get(base).json()["read_only"]  # finished: it opens to look at
    assert ann.get(f"{base}/6E10/area?look=true").json()["saved"]["all_grey"]


def test_the_picker_order_next_core_follows():
    from stainid.labels import picker

    rows = [{"core_id": c, "tma": t, "core": k} for c, t, k in
            [("x10", "10", "A-2"), ("x2b", "2", "B-1"), ("x2a10", "2", "A-10"), ("x2a2", "2", "A-2")]]
    order = picker.cores_natural(rows)
    assert order == ["x2a2", "x2a10", "x2b", "x10"]  # TMA 2 before 10, A-2 before A-10
    assert picker.rotated(order, "x2a10") == ["x2b", "x10", "x2a2", "x2a10"] and picker.rotated(order, "gone") == order
    places = [{"tile_id": "t3", "tma_name": "LIP-2", "core": "A-1", "box": [0, 512, 512, 512]},
              {"tile_id": "t2", "tma_name": "LIP-2", "core": "A-1", "box": [512, 0, 512, 512]},
              {"tile_id": "t1", "tma_name": "LIP-2", "core": "A-1", "box": [0, 0, 512, 512]},
              {"tile_id": "t0", "tma_name": "LIP-10", "core": "A-1", "box": [0, 0, 512, 512]}]
    assert picker.natural(places) == ["t1", "t2", "t3", "t0"]  # top to bottom, left to right
    scores = {"t1": {"mean": 0.2, "objects": 3}, "t2": {"mean": 0.9, "objects": 1}, "t3": {"mean": None, "objects": 0}}
    assert picker.by_confidence(["t1", "t2", "t3", "t0"], scores) == ["t2", "t1", "t3", "t0"]  # least confident first, no objects last
