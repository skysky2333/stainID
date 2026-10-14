from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from stainid.slides.overview import slide_picture
from stainid.slides.table import missing_values, read_name, read_slides, save_slides, slide_rows

STAINS = ["NeuN", "6E10", "AT8"]


def test_saving_keeps_a_slide_grid_size_the_editor_does_not_send(tmp_path):
    table = tmp_path / "slides.csv"
    table.write_text("slide_path,tma,stain,rows,columns\n/s/TMA 1 NeuN.vsi,1,NeuN,,\n/s/TMA 2 NeuN.vsi,2,NeuN,4,6\n")
    edited = [{"slide_path": "/s/TMA 1 NeuN.vsi", "tma": "1", "stain": "NeuN"}, {"slide_path": "/s/TMA 2 NeuN.vsi", "tma": "2", "stain": "NeuN"}]
    save_slides(table, edited, ["NeuN"])
    rows = {r["slide_path"]: r for r in read_slides(table)}
    assert (rows["/s/TMA 2 NeuN.vsi"]["rows"], rows["/s/TMA 2 NeuN.vsi"]["columns"]) == ("4", "6")
    assert rows["/s/TMA 1 NeuN.vsi"]["rows"] == ""


def test_tables_without_a_grid_size_keep_three_columns(tmp_path):
    table = tmp_path / "slides.csv"
    save_slides(table, [{"slide_path": "/s/TMA 1 AT8.vsi", "tma": "1", "stain": "AT8"}], ["AT8"])
    assert table.read_text().splitlines()[0] == "slide_path,tma,stain"


# ---------------------------------------------------------------- what a file name says, and how sure that is

@pytest.mark.parametrize("name, prefix, tma, stain", [
    ("TMA LIP-3 AT8.vsi", "", "3", "AT8"),
    ("TMA 3mm LIP-1 6E10.vsi", "LIP-", "1", "6E10"),        # 3mm is a size, not a TMA
    ("2024-05-01 scan 7 AT8.vsi", "", "7", "AT8"),         # a date is not a TMA
    ("20240501_TMA5_AT8_40x.svs", "", "5", "AT8"),          # the number after "TMA" wins
    ("study_tma12_NeuN.ndpi", "", "12", "NeuN"),
    ("Block-04 6e10 scan.svs", "", "4", "6E10"),            # one number only, stain in any case
])
def test_a_clear_file_name_gives_the_tma_and_stain(name, prefix, tma, stain):
    reading = read_name(Path(name), STAINS, prefix)
    assert (reading.tma, reading.stain) == (tma, stain) and not reading.tma_options and not reading.stain_options


def test_an_unclear_file_name_leaves_the_choice_to_people():
    several = read_name(Path("Slide 3 block 12 NeuN.vsi"), STAINS)
    assert several.tma == "" and several.tma_options == ("3", "12") and several.stain == "NeuN"
    marked_twice = read_name(Path("TMA 3 LIP-1 AT8.vsi"), STAINS, "LIP-")
    assert marked_twice.tma == "" and set(marked_twice.tma_options) == {"1", "3"}
    two_stains = read_name(Path("LIP-2 AT8 NeuN.vsi"), STAINS, "LIP-")
    assert two_stains.tma == "2" and two_stains.stain == "" and two_stains.stain_options == ("NeuN", "AT8")
    nothing = read_name(Path("scan.tif"), STAINS)
    assert (nothing.tma, nothing.stain, nothing.tma_options, nothing.stain_options) == ("", "", (), ())


def test_editor_rows_say_what_came_from_the_file_name(tmp_path):
    for name in ("TMA LIP-1 NeuN.vsi", "TMA LIP-1 AT8.vsi", "Slide 3 block 12 NeuN.vsi", "overview.vsi"):
        (tmp_path / name).touch()
    table = tmp_path / "slides.csv"
    save_slides(table, [{"slide_path": str(tmp_path / "TMA LIP-1 AT8.vsi"), "tma": "1", "stain": "AT8"}], STAINS)
    rows = {Path(r["slide_path"]).name: r for r in slide_rows(tmp_path, table, STAINS, "LIP-")}
    assert rows["TMA LIP-1 AT8.vsi"]["new"] is False and rows["TMA LIP-1 AT8.vsi"]["from_name"] == {"tma": False, "stain": False}
    assert rows["TMA LIP-1 NeuN.vsi"]["from_name"] == {"tma": True, "stain": True} and rows["TMA LIP-1 NeuN.vsi"]["tma"] == "1"
    assert rows["Slide 3 block 12 NeuN.vsi"]["tma"] == "" and rows["Slide 3 block 12 NeuN.vsi"]["options"]["tma"] == ["3", "12"]
    assert rows["overview.vsi"]["from_name"] == {"tma": False, "stain": False}
    missing = missing_values(list(rows.values()), STAINS)
    assert len(missing["tma"]) == 2 and len(missing["stain"]) == 1


def test_saving_needs_a_tma_and_a_stain_for_every_slide_and_one_slide_per_pair(tmp_path):
    table = tmp_path / "slides.csv"
    rows = [{"slide_path": "/s/a.vsi", "tma": "", "stain": "NeuN"}, {"slide_path": "/s/b.vsi", "tma": " ", "stain": ""},
            {"slide_path": "/s/c.vsi", "tma": "2", "stain": "AT8"}]
    with pytest.raises(ValueError, match="2 slides still need a TMA") as error:
        save_slides(table, rows, STAINS)
    assert "1 slide still needs a stain (b.vsi)" in str(error.value) and not table.exists()
    with pytest.raises(ValueError, match="TMA 2 has more than one AT8 slide"):
        save_slides(table, [{"slide_path": "/s/a.vsi", "tma": "2", "stain": "AT8"}, {"slide_path": "/s/c.vsi", "tma": "2 ", "stain": "AT8"}], STAINS)
    save_slides(table, [{"slide_path": "/s/a.vsi", "tma": " 7", "stain": "NeuN"}], STAINS)
    assert read_slides(table)[0]["tma"] == "7"


# ---------------------------------------------------------------- a picture of each slide

def test_slide_picture_from_the_scan_file_or_the_find_cores_preview(tmp_path):
    scan = tmp_path / "TMA 1 NeuN.tif"
    cv2.imwrite(str(scan), np.full((300, 900, 3), 200, np.uint8))
    picture = slide_picture(scan, tmp_path / "qc", long_side=300)
    assert picture is not None and picture.shape == (100, 300, 3)
    vsi = tmp_path / "TMA 2 NeuN.vsi"
    vsi.write_bytes(b"not a tiff")
    assert slide_picture(vsi, tmp_path / "qc") is None
    previews = tmp_path / "qc" / "grids" / "previews"
    previews.mkdir(parents=True)
    cv2.imwrite(str(previews / "TMA_2_NeuN.jpg"), np.full((400, 400, 3), 120, np.uint8))
    assert slide_picture(vsi, tmp_path / "qc", long_side=200).shape == (200, 200, 3)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from stainid.api.app import create_app

    slides = tmp_path / "data" / "slides"
    slides.mkdir(parents=True)
    for name in ("TMA LIP-8 NeuN.tif", "TMA LIP-8 AT8.tif", "scan 2 of 3.tif"):
        cv2.imwrite(str(slides / name), np.full((120, 360, 3), 210, np.uint8))
    (tmp_path / "stainid.yaml").write_text(yaml.safe_dump({"name": "slides", "tma": {"prefix": "LIP-"}}))
    (tmp_path / "data" / "tma_layout.csv").write_text("tma,core_label,donor_id,region,disease_group\n3,A-1,1,frontal,AD\n")
    monkeypatch.setenv("STAINID_PROJECT", str(tmp_path))
    app = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert app.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    return app


def test_slides_api_offers_the_known_tmas_and_marks_guesses(client, tmp_path):
    info = client.get("/api/slides").json()
    rows = {Path(r["slide_path"]).name: r for r in info["rows"]}
    assert info["tmas"] == ["3"] and info["stains"] == STAINS
    assert rows["TMA LIP-8 AT8.tif"]["from_name"] == {"tma": True, "stain": True}
    unclear = rows["scan 2 of 3.tif"]
    assert unclear["tma"] == "" and unclear["stain"] == "" and unclear["options"]["tma"] == ["2", "3"]
    refused = client.put("/api/slides", json={"rows": info["rows"]})
    assert refused.status_code == 400 and "1 slide still needs a TMA" in refused.json()["detail"]
    unclear.update(tma="3", stain="NeuN")
    saved = client.put("/api/slides", json={"rows": list(rows.values())}).json()
    assert saved["tmas"] == ["3", "8"] and all(not r["new"] for r in saved["rows"])
    picture = client.get("/api/slides/picture", params={"path": unclear["slide_path"], "size": 120})
    assert picture.status_code == 200 and picture.headers["content-type"] == "image/jpeg"
    assert client.get("/api/slides/picture", params={"path": str(tmp_path / "stainid.yaml")}).status_code == 404
