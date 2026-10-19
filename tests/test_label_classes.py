"""Label classes as identity and subtype: the built-in classes, labels made before subtypes existed mapped onto them,
identity training sets unchanged by the mapping, the keys of the labelling pages, and the classes in the settings."""
from __future__ import annotations

import io
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from stainid.labels import store, vocabulary
from stainid.labels.vocabulary import NEGATIVE, POSITIVE, SKIP

# ---------- the built-in classes ----------


def test_each_stain_has_identities_with_subtypes_and_plain_descriptions():
    at8, plaques, neun = (vocabulary.defaults(s) for s in ("AT8", "6E10", "NeuN"))
    tau = at8.identity("tau+ neuron")
    assert tau.role == POSITIVE and [s.id for s in tau.subtypes] == ["pretangle", "mature tangle", "ghost tangle"]
    assert [c.id for c in at8.classes] == ["tau+ neuron", "neuritic plaque", "astroglial tau", "thread", "not tau", "unsure"]
    assert [s.name for s in at8.identity("astroglial tau").subtypes] == ["tufted astrocyte", "thorn-shaped astrocyte"]
    assert [s.id for s in plaques.identity("plaque").subtypes] == ["compact", "diffuse", "cored-neuritic", "small"]
    assert {"CAA", "intraneuronal", "not a plaque"} <= {c.id for c in plaques.classes} and plaques.identity("artifact or unclear") is None
    assert [(s.id, s.hidden) for s in neun.identity("NeuN+ neuron").subtypes] == [("strong", True), ("weak", True), ("large pyramidal", True)]
    for words in (at8, plaques, neun, vocabulary.defaults("Iba1")):  # a stain stainID has no classes for gets generic ones
        assert vocabulary.problems(words) == []
        assert [c.role for c in words.classes if c.role == POSITIVE] == [POSITIVE]  # one positive identity: what the model finds
        assert words.identity("unsure").role == SKIP and words.identity(words.reject).role == NEGATIVE
        assert words.role(words.resolve("artifact or unclear")[0]) == NEGATIVE  # old labels of the class that is gone
        assert all(c.description for c in words.classes) and all(s.description for c in words.classes for s in c.subtypes)
    assert at8.display("tau+ neuron", "pretangle") == "tau+ neuron · pretangle" and at8.display("tau+ neuron") == "tau+ neuron (type not recorded)"
    assert at8.display("thread") == "thread-like stain (not a neuron)" and at8.rejecting("thread") and at8.identity("thread").key == "e"


@pytest.mark.parametrize("stain, value, subtype, expected", [
    ("AT8", "tau+ neuron: tangle", None, ("tau+ neuron", "mature tangle")),
    ("AT8", "tau+ neuron: pretangle", None, ("tau+ neuron", "pretangle")),
    ("AT8", "tau+ neuron", None, ("tau+ neuron", "")),
    ("AT8", "not a tau+ neuron", None, ("not tau", "")),
    ("AT8", "tau+ neuron (type not recorded)", None, ("tau+ neuron", "")),
    ("6E10", "plaque (type not recorded)", None, ("plaque", "")),
    ("6E10", "plaque", None, ("plaque", "")),
    ("6E10", "compact plaque", None, ("plaque", "compact")),
    ("6E10", "diffuse plaque", None, ("plaque", "diffuse")),
    ("6E10", "small plaque", None, ("plaque", "small")),
    ("6E10", "not a plaque", None, ("not a plaque", "")),
    ("NeuN", "NeuN+ neuron", None, ("NeuN+ neuron", "")),
    ("NeuN", "artifact or unclear", None, ("not a neuron", "")),  # the class is gone: its labels are the reject
    ("6E10", "artifact or unclear", None, ("not a plaque", "")),
    ("AT8", "artifact or unclear", None, ("not tau", "")),
    ("AT8", "tau+ neuron", "ghost tangle", ("tau+ neuron", "ghost tangle")),  # a new label: identity and subtype
    ("AT8", "", None, ("", "")),
    ("AT8", float("nan"), float("nan"), ("", "")),
    # class maps of dataset packages: "identity / subtype" or "identity: subtype", by id or by name
    ("6E10", "plaque / compact", None, ("plaque", "compact")),
    ("6E10", "plaque/compact", None, ("plaque", "compact")),
    ("AT8", "tau+ neuron/pretangle", None, ("tau+ neuron", "pretangle")),
    ("6E10", "plaque/no such type", None, ("plaque", "no such type")),  # kept: the type is shown as written
    ("6E10", "plaque: compact (dense-core)", None, ("plaque", "compact")),
    ("6E10", "CAA / vascular amyloid", None, ("CAA", "")),
    ("AT8", "Tau+ Neuron / Mature Tangle", None, ("tau+ neuron", "mature tangle")),
    ("AT8", "something else", None, ("something else", "")),  # unknown: kept as it is (a negative, as before)
])
def test_labels_made_before_subtypes_map_onto_identity_and_subtype(stain, value, subtype, expected):
    assert vocabulary.defaults(stain).resolve(value, subtype) == expected


def _old_consensus(labels: pd.DataFrame, stain: str) -> dict:
    """The identity truth as the flat vocabulary worked it out before subtypes: object -> positive (None on a tie)."""
    positive = {"NeuN": {"NeuN+ neuron"}, "6E10": {"compact plaque", "diffuse plaque", "small plaque", "plaque"},
                "AT8": {"tau+ neuron: tangle", "tau+ neuron: pretangle", "tau+ neuron"}}[stain]
    rows = labels[(labels.stain == stain) & (labels.task == "class") & ~labels.value.isin({"unsure", ""})]
    out = {}
    for (tile, x, y), group in rows.groupby(["tile_id", rows.x.round().astype(int), rows.y.round().astype(int)]):
        votes = group.value.isin(positive).mean()
        out[(tile, x, y)] = None if votes == 0.5 else bool(votes > 0.5)
    return out


@pytest.mark.parametrize("stain, values", [
    ("AT8", ["tau+ neuron: tangle", "tau+ neuron: pretangle", "tau+ neuron", "not a tau+ neuron", "unsure", ""]),
    ("6E10", ["compact plaque", "diffuse plaque", "small plaque", "plaque", "not a plaque", "unsure", ""]),
    ("NeuN", ["NeuN+ neuron", "not a neuron", "artifact or unclear", "unsure", ""]),
])
def test_identity_training_sets_are_unchanged(stain, values):
    """Many objects, each labelled by one to four people with any of the values labels had before subtypes: every object's
    positive / negative (what the identity model learns) is exactly what it was."""
    rng = np.random.default_rng(0)
    rows = [{"source": f"p{p}", "stain": stain, "task": "class", "tile_id": f"T{i % 7}", "x": float(i * 10), "y": 5.0, "value": str(rng.choice(values))}
            for i in range(600) for p in range(int(rng.integers(1, 5)))]
    labels = pd.DataFrame(rows).assign(subtype=None)
    new = store.consensus(labels, stain, vocabulary.defaults(stain))
    new_truth = {(r.tile_id, round(r.x), round(r.y)): r.positive for r in new.itertuples()}
    assert new_truth == _old_consensus(labels, stain)
    assert set(new.value) - {""} <= {c.id for c in vocabulary.defaults(stain).classes}


def test_types_follow_the_people_who_gave_one():
    labels = pd.DataFrame([
        {"source": "Ann", "stain": "AT8", "task": "class", "tile_id": "T", "x": 1.0, "y": 1.0, "value": "tau+ neuron: tangle", "subtype": None},
        {"source": "Ben", "stain": "AT8", "task": "class", "tile_id": "T", "x": 1.0, "y": 1.0, "value": "tau+ neuron", "subtype": "mature tangle"},
        {"source": "Cy", "stain": "AT8", "task": "class", "tile_id": "T", "x": 1.0, "y": 1.0, "value": "tau+ neuron", "subtype": None},
        {"source": "Ann", "stain": "AT8", "task": "class", "tile_id": "T", "x": 9.0, "y": 1.0, "value": "tau+ neuron", "subtype": "pretangle"},
        {"source": "Ben", "stain": "AT8", "task": "class", "tile_id": "T", "x": 9.0, "y": 1.0, "value": "tau+ neuron", "subtype": "ghost tangle"},
    ])
    found = store.consensus(labels, "AT8").set_index("x")
    assert (found.loc[1.0, "value"], found.loc[1.0, "subtype"], found.loc[1.0, "agreement"]) == ("tau+ neuron", "mature tangle", 1.0)
    assert (found.loc[9.0, "value"], found.loc[9.0, "subtype"], found.loc[9.0, "positive"]) == ("tau+ neuron", "", True)


def test_dataset_class_maps_may_name_types(tmp_path):
    """A dataset package's class map may target a type ("plaque/compact"): its labels are stored as identity and type,
    and the scores take the AI's "compact plaque" as the same type."""
    from stainid.imageset import annotations, package, scoring
    from stainid.project import load_project

    choices = package.class_choices()
    assert {"plaque", "plaque/compact", "CAA", "intraneuronal"} <= set(choices["amyloid"]) and "tau+ neuron/pretangle" in choices["tau"]
    assert "unsure" not in choices["neun"]
    frame = pd.DataFrame([{"kind": "point", "x": 5.0, "y": 6.0, "tile_id": "T", "stain": "6E10", "class": "plaque/compact"},
                          {"kind": "point", "x": 9.0, "y": 6.0, "tile_id": "T", "stain": "AT8", "class": "tau+ neuron/pretangle"},
                          {"kind": "point", "x": 9.0, "y": 9.0, "tile_id": "T", "stain": "6E10", "class": "vascular"}])
    records = annotations._label_records(load_project(tmp_path), frame, "import: x", "import-x")
    assert [(r["value"], r["subtype"]) for r in records] == [("plaque", "compact"), ("tau+ neuron", "pretangle"), ("CAA", None)]
    assert scoring._kind("6E10", "plaque/compact") == scoring._kind("6E10", "compact plaque") == "plaque/compact"
    assert scoring._same_class("plaque/compact", "compact plaque", "6E10") and not scoring._same_class("plaque/compact", "diffuse plaque", "6E10")
    assert scoring._same_class("plaque", "diffuse plaque", "6E10")  # no type marked: any plaque


# ---------- the keys ----------


@pytest.mark.parametrize("stain, direct, summary", [
    ("6E10", {"1": ["plaque", "compact"], "2": ["plaque", "diffuse"], "3": ["plaque", "cored-neuritic"], "4": ["plaque", "small"],
              "q": ["CAA", None], "w": ["intraneuronal", None], "delete": ["not a plaque", None], "`": ["unsure", None]},
     "Plaques 1–4 · Not plaques Q–W · Del not a plaque · ` unsure · B brush"),
    ("AT8", {"1": ["tau+ neuron", "pretangle"], "2": ["tau+ neuron", "mature tangle"], "3": ["tau+ neuron", "ghost tangle"],
             "q": ["neuritic plaque", None], "w": ["astroglial tau", None], "e": ["thread", None], "delete": ["not tau", None], "`": ["unsure", None]},
     "Tau+ neurons 1–3 · Not tau+ neurons Q–E · Del not tau · ` unsure · B brush"),
    ("NeuN", {"1": ["NeuN+ neuron", None], "delete": ["not a neuron", None], "`": ["unsure", None]},
     "NeuN+ neurons 1 · Del not a neuron · ` unsure · B brush"),
])
def test_one_key_picks_a_type_per_stain(stain, direct, summary):
    """Numbers pick the counted types, Q W E R the named look-alikes, Del the single reject and ` unsure (0 too): one
    key each, no second step; the panel's groups and line come from the same assignment."""
    words = vocabulary.defaults(stain)
    keys = words.keymap(subtypes=True)
    assert keys["direct"] == direct and keys["summary"] == summary and "then" not in keys
    assert keys["aliases"] == {"0": "`", "backspace": "delete"} and keys["brush"] == "b"
    assert [g["group"] for g in keys["groups"]] == [g for g in ("counted", "lookalike", "reject", "unsure") if g != "lookalike" or stain != "NeuN"]
    clashes = {" ", "enter", "escape", "shift", "tab", "arrowleft", "arrowright", "arrowup", "arrowdown", "b", "s", "z", "v", "a", "o"}
    assert not clashes & set(keys["direct"])  # nothing the pages or the browser use (no Ctrl/Cmd keys are taken at all)
    assert vocabulary.problems(words) == []


def test_roles_of_the_reject_and_unsure_and_sessions_without_types():
    for stain in ("6E10", "AT8", "NeuN"):
        words = vocabulary.defaults(stain)
        assert words.role(words.reject) == NEGATIVE and words.role(words.unsure) == SKIP and words.skipped(words.unsure)
        assert words.reject not in words.colour_slots() and words.unsure not in words.colour_slots()  # dotted grey, dotted amber
    plain = vocabulary.defaults("AT8").keymap(subtypes=False)  # a session that does not ask for types: the identity on 1
    assert plain["direct"]["1"] == ["tau+ neuron", None] and "2" not in plain["direct"] and plain["direct"]["e"] == ["thread", None]
    slots = vocabulary.defaults("6E10").colour_slots()
    assert len(set(slots.values())) == len(slots) and slots["plaque/compact"] != slots["plaque/diffuse"]  # each type its own colour


def test_classes_people_add_take_5_to_9_and_t_y_u():
    words = vocabulary.from_config("6E10", [
        {"id": "plaque", "subtypes": [{"id": "compact"}, {"id": "diffuse"}, {"id": "cored-neuritic"}, {"id": "small"}, {"id": "burnt-out", "name": "burnt-out"}]},
        {"id": "pigment", "name": "pigment", "role": "negative", "description": "Brown pigment."},
        {"id": "artifact or unclear", "name": "artifact or unclear", "role": "negative", "key": "5"},  # a class no more
    ])
    direct = words.keymap()["direct"]
    assert direct["5"] == ["plaque", "burnt-out"] and direct["t"] == ["pigment", None] and direct["q"] == ["CAA", None]
    assert words.identity("artifact or unclear") is None
    old = vocabulary.from_config("6E10", [{"id": "plaque", "key": "1", "subtypes": [{"id": "compact", "key": "q"}, {"id": "diffuse", "key": "w"}]},
                                          {"id": "CAA", "key": "2"}, {"id": "not a plaque", "key": "4"}])
    assert old.keymap()["direct"] == vocabulary.defaults("6E10").keymap()["direct"]  # settings saved with the old keys get today's
    wrong = vocabulary.from_config("6E10", [{"id": "CAA", "key": "7"}])
    assert any("must be a letter" in p for p in vocabulary.problems(wrong))


def test_the_page_logic_of_the_keys(tmp_path):
    """The quick check's key handling (webapp/src/labels/classKeys.ts), run by Node when it is installed."""
    node = shutil.which("node") or str(Path.home() / ".local/node/bin/node")
    if not Path(node).exists():
        pytest.skip("Node.js is not installed")
    script = Path(__file__).parent / "js" / "class_keys.test.ts"
    done = subprocess.run([node, "--experimental-strip-types", "--no-warnings", str(script)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr


def test_classes_are_listed_in_the_order_of_their_keys_with_unsure_last():
    from stainid.api.routes.annotate import class_groups
    from stainid.labels.overview import class_counts

    for stain in ("NeuN", "6E10", "AT8"):
        words = vocabulary.defaults(stain)
        keys = {i: k for k, (i, sub) in words.keymap()["direct"].items() if sub is None}
        listed = [c for c in words.by_key() if not c.hidden]
        numbers = [int(keys[c.id]) for c in listed if c.role != SKIP and keys.get(c.id, "").isdigit()]
        assert numbers == sorted(numbers), (stain, numbers)  # 1, 2, 3 … never 1, 3, 2
        assert listed[-1].role == SKIP  # unsure (0 on the pages) last
        assert [g["id"] for g in class_groups(words, set())][: len(listed)] == [c.id for c in listed]
        assert [c["id"] for c in class_counts(pd.DataFrame(), words)] == [c.id for c in listed]


def test_every_page_gives_a_class_the_same_colour():
    """One palette for every page (webapp/src/classPalette.ts), in the order of `class_groups`, run by Node when installed."""
    node = shutil.which("node") or str(Path.home() / ".local/node/bin/node")
    if not Path(node).exists():
        pytest.skip("Node.js is not installed")
    script = Path(__file__).parent / "js" / "class_palette.test.ts"
    done = subprocess.run([node, "--experimental-strip-types", "--no-warnings", str(script)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    from stainid.api.routes.annotate import class_groups

    words = vocabulary.defaults("AT8")
    slots = {g["id"]: g["colour"] for g in class_groups(words, set())}
    assert all(slots[c] == i for c, i in words.colour_slots().items() if c in slots)  # the slots the page turns into colours
    assert slots[words.reject] == -2 and slots[words.unsure] == -3


# ---------- the classes in the settings, and labelling with them ----------

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from people import annotator  # noqa: E402

from stainid.api import state  # noqa: E402
from stainid.api.app import create_app  # noqa: E402

TILE = "LIP-1_B-2_6E10_S01"


@pytest.fixture()
def client(synthetic_project, monkeypatch):
    from stainid.api.routes import annotate
    from stainid.labels import pool

    monkeypatch.chdir(synthetic_project)
    monkeypatch.setattr(annotate, "prepare_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(annotate, "measure_added_soon", lambda *args, **kwargs: None)
    state.reset()
    project = state.get_state().project
    project.model("amyloid").parent.mkdir(parents=True, exist_ok=True)
    project.model("amyloid").write_bytes(b"model")
    folder = pool.folder(project, "6E10")
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"x": 100.0, "y": 200.0, "radius_px": 20.0, "ai_value": "compact plaque", "ai_probability": 0.5, "tma": "1", "complete": True,
                   "f1": 1.0}]).to_csv(folder / f"{TILE}.csv", index=False)
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    yield admin
    state.reset()


def _annotator(client) -> TestClient:
    return annotator(client, "Ann", every=True)


def test_quick_check_answers_carry_identity_and_type(client):
    made = client.post("/api/labels/sessions", json={"name": "q", "stain": "6E10", "kind": "queue", "rule": "random", "open": True, "tools": ["class"]})
    assert made.status_code == 200 and made.json()["subtypes"] is True
    ann = _annotator(client)
    meta = ann.get("/api/annotate/q").json()["meta"]
    assert meta["subtypes"] and meta["keys"]["direct"]["1"] == ["plaque", "compact"] and meta["keys"]["direct"]["2"] == ["plaque", "diffuse"]
    assert [c["id"] for c in meta["vocabulary"]["classes"]][:3] == ["plaque", "CAA", "intraneuronal"]
    assert meta["vocabulary"]["aliases"]["compact plaque"] == ["plaque", "compact"]
    item = ann.get("/api/annotate/q/next").json()["object"]
    saved = ann.post("/api/annotate/q/labels", json={"records": [{"task": "class", "tile_id": item["tile_id"], "x": item["x"], "y": item["y"],
                                                                   "value": "plaque", "subtype": "diffuse"}]}).json()
    assert (saved[0]["value"], saved[0]["subtype"]) == ("plaque", "diffuse")
    bad = ann.post("/api/annotate/q/labels", json={"records": [{"task": "class", "tile_id": TILE, "x": 1.0, "y": 1.0, "value": "plaque", "subtype": "pretangle"}]})
    assert bad.status_code == 400 and "no type" in bad.json()["detail"]
    assert ann.post("/api/annotate/q/labels", json={"records": [{"task": "class", "tile_id": TILE, "x": 1.0, "y": 1.0, "value": "a cat"}]}).status_code == 400
    overview = next(s for s in client.get("/api/labels/overview").json()["stains"] if s["stain"] == "6E10")
    plaque = next(c for c in overview["classes"] if c["id"] == "plaque")
    assert plaque["objects"] == 1 and {s["id"]: s["objects"] for s in plaque["subtypes"]}["diffuse"] == 1 and plaque["untyped"] == 0
    assert overview["values"] == {"plaque · diffuse": 1}
    table = pd.read_csv(io.StringIO(client.get("/api/labels/export.csv").text))
    row = table[table.task == "class"].iloc[0]
    assert (row.value, row.identity, row.subtype, row["class"], row.role) == ("plaque", "plaque", "diffuse", "plaque · diffuse", "positive")


def test_fields_without_types_confirm_only_the_identity(client):
    client.post("/api/labels/sessions", json={"name": "f", "stain": "6E10", "tile_ids": [TILE], "open": True, "subtypes": False})
    ann = _annotator(client)
    assert ann.get("/api/annotate/f").json()["meta"]["keys"]["direct"]["1"] == ["plaque", None]
    ann.post(f"/api/annotate/f/fields/{TILE}/finish")
    confirmed = store.current(state.get_state().project.output("reviews"))
    confirmed = confirmed[(confirmed.task == "class") & (confirmed.session == "f")].iloc[0]
    assert confirmed.value == "plaque" and confirmed.subtype is None and bool(confirmed.confirmed)


def test_classes_in_the_settings_round_trip(client):
    shown = client.get("/api/labels/classes").json()
    assert set(shown["stains"]) == {"NeuN", "6E10", "AT8"} and shown["subtype_wait_s"] == 1.0
    classes = shown["stains"]["AT8"]["classes"]
    tau = classes[0]
    tau["subtypes"][0]["name"] = "early tangle (pretangle)"  # rename (the id stays)
    tau["subtypes"][2]["hidden"] = True
    tau["subtypes"].append({"id": "globose", "name": "globose tangle", "description": "A round tangle.", "key": "4"})
    tau["role"] = "negative"  # a built-in class keeps its role
    classes.append({"id": "microglial tau", "name": "microglial tau", "role": "negative", "description": "Tau in microglia.", "key": "t", "subtypes": []})
    del classes[3]  # a built-in class left out comes back
    saved = client.put("/api/labels/classes/AT8", json={"classes": classes, "subtype_wait_s": 1.5})
    assert saved.status_code == 200, saved.text
    again = client.get("/api/labels/classes").json()
    at8 = {c["id"]: c for c in again["stains"]["AT8"]["classes"]}
    assert at8["tau+ neuron"]["role"] == "positive" and at8["tau+ neuron"]["subtypes"][0]["name"] == "early tangle (pretangle)"
    assert [s["id"] for s in at8["tau+ neuron"]["subtypes"]] == ["pretangle", "mature tangle", "ghost tangle", "globose"]
    assert at8["tau+ neuron"]["subtypes"][2]["hidden"] and at8["microglial tau"]["role"] == "negative" and "thread" in at8
    assert again["subtype_wait_s"] == 1.5 and again["stains"]["6E10"]["classes"] == shown["stains"]["6E10"]["classes"]
    written = yaml.safe_load((state.get_state().root / "stainid.yaml").read_text())
    assert set(written["labels"]["classes"]) == {"AT8"} and written["labels"]["subtype_wait_s"] == 1.5
    words = vocabulary.of(state.get_state().project, "AT8")
    direct = words.keymap()["direct"]
    assert (direct["1"], direct["2"], direct["4"], direct["t"]) == (["tau+ neuron", "pretangle"], ["tau+ neuron", "mature tangle"],
                                                                   ["tau+ neuron", "globose"], ["microglial tau", None]) and "3" not in direct
    assert words.display("tau+ neuron", "pretangle") == "tau+ neuron · early tangle (pretangle)"

    clash = [{**c, "key": "q"} if c["id"] == "thread" else c for c in again["stains"]["AT8"]["classes"]]
    refused = client.put("/api/labels/classes/AT8", json={"classes": clash})
    assert refused.status_code == 400 and "same key" in refused.json()["detail"]
    reserved = [{**c, "key": "z"} if c["id"] == "thread" else c for c in again["stains"]["AT8"]["classes"]]
    assert client.put("/api/labels/classes/AT8", json={"classes": reserved}).status_code == 400

    reset = client.put("/api/labels/classes/AT8", json={"classes": None}).json()
    assert reset["stains"]["AT8"]["classes"] == reset["builtin"]["AT8"]["classes"]
    assert "AT8" not in (yaml.safe_load((state.get_state().root / "stainid.yaml").read_text()).get("labels", {}).get("classes") or {})


def test_example_pictures_are_stored_with_the_labels_and_shown_to_labellers(client):
    from PIL import Image

    picture = io.BytesIO()
    Image.new("RGB", (1200, 900), (180, 120, 60)).save(picture, "PNG")
    made = client.post("/api/labels/classes/AT8/examples", files={"file": ("example.png", picture.getvalue(), "image/png")})
    assert made.status_code == 200, made.text
    name = made.json()["file"]
    stored = state.get_state().project.output("reviews") / "class_examples" / "AT8" / name
    with Image.open(stored) as image:
        assert max(image.size) == 480
    assert client.get(f"/api/annotate/class-examples/AT8/{name}").headers["content-type"] == "image/jpeg"
    assert client.get("/api/annotate/class-examples/AT8/stainid.yaml").status_code == 404  # only example pictures by their names
    assert client.post("/api/labels/classes/AT8/examples", files={"file": ("notes.txt", b"not a picture")}).status_code == 400

    # a labelled object's picture, from the labels view, as the example of its class and type
    spot = {"tile_id": TILE, "x": 100.0, "y": 200.0}
    store.append(state.get_state().project.output("reviews"), [{"id": f"class:{TILE}:100:200", "source": "Ann", "stain": "6E10", "task": "class",
                                                                **spot, "value": "plaque", "subtype": "compact"}])
    assert client.post("/api/labels/classes/6E10/examples/from-object", json={**spot, "identity": "plaque", "subtype": "pretangle"}).status_code == 400
    for _ in range(100):  # 409 while the core image is read in the background
        made = client.post("/api/labels/classes/6E10/examples/from-object", json={**spot, "identity": "plaque", "subtype": "compact"})
        if made.status_code != 409:
            break
        time.sleep(0.1)
    assert made.status_code == 200, made.text
    plaque = next(c for c in client.get("/api/labels/classes").json()["stains"]["6E10"]["classes"] if c["id"] == "plaque")
    assert plaque["subtypes"][0]["example"] == made.json()["file"] and plaque["example"] == ""
