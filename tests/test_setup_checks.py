"""Guided setup: cross-checks between map, donor table and slides, their fixes, the screens and the API."""
import pytest
import yaml
from setup_files import docx, xlsx
from test_setup_import import CASES, WORD_MAP

from stainid.project import load_project
from stainid.setup import checks, guide
from stainid.tables import read_csv, write_csv


def layout_row(tma, label, donor="", region="frontal", group="AD", **extra):
    return {"tma": tma, "core_label": label, "donor_id": donor, "region": region if donor else "", "disease_group": group if donor else "", **extra}


LAYOUT = [layout_row("8", "A-1", "1673", group="CT", cerad="B"), layout_row("8", "B-1", "1673", "occipital", "CT"),
          layout_row("8", "A-2", "2323", cerad="B"), layout_row("8", "B-2", "2323", "occipital"),
          layout_row("8", "C-1", "JOG_007", group=""), layout_row("8", "C-2", "")]
DONORS = [{"donor_id": "1673", "tma": "8", "disease_group": "CT", "cerad": "0"}, {"donor_id": "2323", "tma": "8", "disease_group": "AD", "cerad": "0"},
          {"donor_id": "JOG_007", "tma": "8", "disease_group": "AD"}, {"donor_id": "JOG_007", "tma": "9", "disease_group": "AD"},
          {"donor_id": "5555", "tma": "8", "disease_group": "CT"}]


def by_id(issues):
    return {i["id"]: i for i in issues}


def test_everything_that_does_not_fit_together_is_found():
    positions = {("8", p) for p in ("A-1", "B-1", "C-1", "A-2", "B-2", "C-2", "A-3")} | {("9", "A-1")}
    issues = by_id(checks.find_issues(LAYOUT, DONORS, positions, {"8", "9"}, ["CT", "AD"]))
    assert issues["tma_without_map"]["level"] == "problem" and "TMA 9" in issues["tma_without_map"]["title"]
    assert issues["positions_not_on_map"]["fixes"][0]["args"] == {"positions": [["8", "A-3"]]}
    assert issues["map_without_group"]["fixes"][0]["args"] == {"donors": ["JOG_007"]} or "JOG_007" in issues["map_without_group"]["items"][0]["text"]
    assert issues["duplicate_donors"]["fixes"][0]["args"] == {"donors": ["JOG_007"]}
    assert "5555" in issues["sheet_donor_not_on_map"]["items"][0]["text"]
    pathology = issues["pathology_disagrees"]["items"]
    assert pathology[0]["text"] == "Donor 1673: map says CERAD B, donor table says 0. Use the donor table?"
    assert [f["action"] for f in pathology[0]["fixes"]] == ["pathology_from_sheet", "pathology_from_map"]
    assert "single_core" not in issues  # every donor-region here has one core: no replicates expected


def test_disagreements_on_group_and_tma_and_missing_donors():
    donors = [{"donor_id": "1673", "tma": "9", "disease_group": "AD"}]
    issues = by_id(checks.find_issues(LAYOUT[:4], donors, set(), {"8"}, ["CT", "AD", "X"]))
    assert issues["tma_disagrees"]["items"][0]["text"] == "Donor 1673: map TMA 8, donor table TMA 9"
    assert issues["group_disagrees"]["items"][0]["text"] == "Donor 1673: map says CT, donor table says AD"
    assert "2323" in issues["map_donor_not_in_sheet"]["items"][0]["text"]


def test_lonely_cores_when_replicates_are_the_rule():
    layout = [layout_row("1", "A-1", "0001"), layout_row("1", "A-2", "0001"), layout_row("1", "B-1", "0002"), layout_row("1", "B-2", "0002"),
              layout_row("1", "C-1", "0003")]
    issues = by_id(checks.find_issues(layout, [], set(), {"1"}, ["AD"]))
    assert issues["single_core"]["level"] == "note" and "Donor 0003" in issues["single_core"]["items"][0]["text"]
    assert by_id(checks.find_issues(layout, [], set(), {"1"}, []))["unknown_groups"]["fixes"][0]["args"] == {"codes": ["AD"]}


@pytest.fixture()
def study(tmp_path):
    (tmp_path / "stainid.yaml").write_text(yaml.safe_dump({"name": "checks", "groups": [{"code": "CT", "label": "Control"}, {"code": "AD", "label": "AD"}]}))
    project = load_project(tmp_path)
    write_csv(project.input("tma_layout"), LAYOUT, list(dict.fromkeys(k for r in LAYOUT for k in r)))
    write_csv(project.input("donor_metadata"), DONORS, ["donor_id", "tma", "disease_group", "cerad"])
    return project


def test_fixes_change_the_files(study):
    assert checks.apply_fix(study, "pathology_from_sheet", {"donors": ["1673"]}).startswith("Pathology taken from the donor table")
    assert next(r for r in read_csv(study.input("tma_layout")) if r["donor_id"] == "1673")["cerad"] == "0"
    checks.apply_fix(study, "keep_map_tma_rows", {"donors": ["JOG_007"]})
    assert [d["tma"] for d in read_csv(study.input("donor_metadata")) if d["donor_id"] == "JOG_007"] == ["8"]
    checks.apply_fix(study, "groups_from_sheet", {"donors": ["JOG_007"]})
    assert next(r for r in read_csv(study.input("tma_layout")) if r["donor_id"] == "JOG_007")["disease_group"] == "AD"
    checks.apply_fix(study, "remove_donors", {"donors": ["5555"]})
    checks.apply_fix(study, "add_empty_positions", {"positions": [["8", "A-3"]]})
    assert ("8", "A-3", "") in {(r["tma"], r["core_label"], r["donor_id"]) for r in read_csv(study.input("tma_layout"))}
    checks.apply_fix(study, "add_groups", {"codes": ["ASYMP"]})
    assert [g["code"] for g in load_project(study.root).config["groups"]] == ["CT", "AD", "ASYMP"]
    remaining = by_id(checks.project_issues(load_project(study.root)))
    assert not {"duplicate_donors", "sheet_donor_not_on_map", "map_without_group"} & set(remaining)
    with pytest.raises(ValueError):
        checks.apply_fix(study, "make_coffee", {})


def test_estimate_and_durations(study):
    estimate = guide.estimate(study, ["dearray", "nuclei", "summarize"])
    assert estimate["steps"][0]["id"] == "dearray" and estimate["seconds"] > 0 and estimate["text"].startswith("about")
    assert guide.duration(30) == "a minute" and guide.duration(1800) == "30 minutes" and guide.duration(5 * 3600) == "5 hours"


# ---------------------------------------------------------------- the API

@pytest.fixture()
def client(tmp_path, monkeypatch, tmp_path_factory):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from stainid.api import state
    from stainid.api.app import create_app

    slides = tmp_path / "data" / "slides"
    slides.mkdir(parents=True)
    for name in ("TMA LIP-8 NeuN.vsi", "TMA LIP-8 AT8.vsi", "TMA LIP-9 NeuN.vsi"):
        (slides / name).write_text("")
    (tmp_path / "stainid.yaml").write_text(yaml.safe_dump({"name": "setup", "tma": {"rows": 2, "columns": 3}}))
    monkeypatch.setenv("STAINID_PROJECT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    state.reset()
    app = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert app.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    yield app
    state.get_state().jobs.close()
    state.reset()


def test_the_guided_setup_from_slides_to_checks(client, tmp_path):
    overview = client.get("/api/setup").json()
    assert not overview["set_up"] and overview["screens"][0]["summary"] == "3 scans found in the slides folder"

    slides = client.get("/api/slides").json()
    assert slides["prefix_guess"] == "LIP-" and {r["tma"] for r in slides["rows"]} == {"8", "9"} and all(r["new"] for r in slides["rows"])
    assert client.put("/api/slides", json={"rows": slides["rows"]}).status_code == 200

    cores = client.get("/api/setup/cores").json()
    assert len(cores["slides"]) == 3 and not cores["slides"][0]["found"]
    nine = next(s for s in cores["slides"] if s["tma"] == "9")
    changed = client.put("/api/setup/cores/grid", json={"slide_path": nine["slide_path"], "rows": 2, "columns": 2, "orientation": "rotate_180"}).json()
    assert changed["job"]["argv"] == ["dearray", "--slide", "TMA LIP-9 NeuN.vsi"]
    assert next(s for s in changed["slides"] if s["tma"] == "9")["grid_source"] == "manual"
    client.delete(f"/api/jobs/{changed['job']['id']}")
    assert client.put("/api/setup/cores/grid", json={"slide_path": nine["slide_path"], "rows": 2}).status_code == 400

    rule = client.put("/api/setup/sample-ids", json={"plain_prefix": "BRC_", "prefixed_prefix": "", "region": "code"}).json()["sample_ids"]
    assert rule["source"] == "setting" and rule["examples"] == ["BRC_1673_FR", "JOG_003_OCC"]
    read = client.post("/api/setup/map/read", files={"file": ("LIP8.docx", docx([("TMA LIP-8", WORD_MAP)]))}).json()
    found = read["maps"][0]
    assert found["tma"] == "8" and found["summary"]["donors"] == 4 and found["summary"]["controls"] == 1
    saved = client.put("/api/setup/map/8", json={"grid": found["grid"], "source": "LIP8.docx"}).json()
    assert {r["donor_id"]: r["sample_region_id"] for r in read_csv(tmp_path / "data" / "tma_layout.csv") if r["region"] == "frontal"} == \
        {"1673": "BRC_1673_FR", "1700": "BRC_1700_FR", "JOG_003": "JOG_003_FR"}
    assert saved["sample_ids"]["source"] == "layout" and saved["sample_ids"]["regions"]["occipital"] == "OCC"
    eight = next(t for t in saved["tmas"] if t["tma"] == "8")
    assert eight["saved"]["donors"] == 4 and eight["sources"] == ["LIP8.docx"]
    assert [g["code"] for g in client.get("/api/project").json()["config"]["groups"]] == ["AD", "ASYMP", "CT"]
    again = client.get("/api/setup/map/8").json()
    assert again["saved"] and again["grid"][0][0].startswith("1673 frontal")
    pasted = client.post("/api/setup/map/read-text", json={"text": "1141 FR AD\t1141 OCC AD\n", "name": "TMA 9"}).json()
    assert pasted["maps"][0]["tma"] == "9"
    assert pasted["maps"][0]["cells"][1]["sample_region_id"] == "BRC_1141_OCC"  # previewed with the project's naming
    assert client.post("/api/setup/map/read", files={"file": ("map.jpg", b"\xff\xd8")}).status_code == 400

    tables = client.post("/api/setup/donors/read", files={"file": ("cases.xlsx", xlsx({"Cases": CASES}))}).json()["tables"]
    preview = client.post("/api/setup/donors/preview", json={"rows": tables[0]["rows"]}).json()
    assert preview["total"] == 5 and {r["donor_id"] for r in preview["records"]} == {"1673", "1700", "2323", "JOG_003", "JOG_007"}
    assert next(c for c in preview["columns"] if c["header"] == "Age")["field"] == "age"
    assert client.put("/api/setup/donors", json={"rows": tables[0]["rows"]}).json()["count"] == 5

    issues = {i["id"]: i for i in client.get("/api/setup/checks").json()}
    assert "TMA 9" in issues["tma_without_map"]["title"] and "sheet_donor_not_on_map" not in issues
    assert issues["group_disagrees"]["items"][0]["text"] == "Donor 1673: map says AD, donor table says CT"
    assert "1935" in issues["map_donor_not_in_sheet"]["items"][0]["text"]
    fix = issues["group_disagrees"]["items"][0]["fixes"][0]
    assert client.post("/api/setup/checks/fix", json=fix).json()["message"] == "Group taken from the donor table for 1 donor"
    left = next(i for i in client.get("/api/setup/checks").json() if i["id"] == "group_disagrees")
    assert [i["text"] for i in left["items"]] == ["Donor JOG_003: map says ASYMP, donor table says AD"]
    assert client.post("/api/setup/checks/fix", json={"action": "nothing", "args": {}}).status_code == 400
    overview = client.get("/api/setup").json()
    assert [s["id"] for s in overview["screens"]] == ["slides", "cores", "map", "donors", "checks", "run"]
    assert overview["estimate"]["seconds"] > 0 and overview["remaining"][0]["id"] == "dearray"


def test_step_summaries_count_in_plain_words(client):
    screens = {s["id"]: s for s in client.get("/api/setup").json()["screens"]}
    assert screens["cores"]["summary"] == "Register slides first" and screens["map"]["summary"] == "Register slides first"
    client.put("/api/slides", json={"rows": client.get("/api/slides").json()["rows"]})
    screens = {s["id"]: s for s in client.get("/api/setup").json()["screens"]}
    assert screens["slides"]["summary"] == "3 scans registered"
    assert screens["cores"]["summary"] == "Cores found on 0 of 3 slides"
    assert screens["map"]["summary"] == "Maps for 0 of 2 TMAs"
    assert "0 to" not in screens["checks"]["summary"]
    assert checks.one("Kept the row on the map's TMA for 1 donors (1 rows removed)") == "Kept the row on the map's TMA for 1 donor (1 row removed)"


def test_the_donor_id_style_takes_0_to_8_digits(client):
    assert client.put("/api/setup/id-style", json={"digits": 9, "prefixed_digits": 3}).status_code == 400
    assert client.put("/api/setup/id-style", json={"digits": "x", "prefixed_digits": 3}).status_code == 400
    assert client.put("/api/setup/id-style", json={"digits": None, "prefixed_digits": 3}).status_code == 400
    assert client.put("/api/setup/id-style", json={"digits": 5, "prefixed_digits": 2}).json()["id_style"] == {"digits": 5, "prefixed_digits": 2}
