"""The Results page says what it does not show and why: the rows of stage 5's table of reasons (results_missing_*.csv)
beside the main tables, never a pseudo-value, and nothing extra for a project without that table."""
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stainid.api import state
from stainid.api.app import create_app


@pytest.fixture()
def client(synthetic_project, monkeypatch):
    monkeypatch.chdir(synthetic_project)
    state.reset()
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    yield admin
    state.reset()


@pytest.fixture()
def tables(client, synthetic_project):
    folder = Path(synthetic_project) / "data" / "analysis" / "v2" / "measures"
    folder.mkdir(parents=True, exist_ok=True)
    rows = [{"sample_region_id": f"D{i}_MFG", "donor_id": f"D{i}", "tma": "1", "region": "MFG", "disease_group": "CTRL",
             "v2_plaque_density_mm2": value, "neun_profile_density_mm2": 10.0 + i}
            for i, value in enumerate([0.0, 2.5, None, None, None])]
    pd.DataFrame(rows).to_csv(folder / "results_donor_region.csv", index=False)
    return folder


def write_reasons(folder):
    pd.DataFrame([
        {"sample_region_id": "D2_MFG", "donor_id": "D2", "tma": "1", "region": "MFG", "disease_group": "CTRL", "measure": "v2_plaque_density_mm2", "reason": "too_few_objects", "n_objects": 3},
        {"sample_region_id": "D3_MFG", "donor_id": "D3", "tma": "1", "region": "MFG", "disease_group": "CTRL", "measure": "v2_plaque_density_mm2", "reason": "no_grey_matter", "n_objects": 0},
        # a row the results table does not have at all
        {"sample_region_id": "D9_MFG", "donor_id": "D9", "tma": "1", "region": "MFG", "disease_group": "AD", "measure": "v2_plaque_density_mm2", "reason": "no_image", "n_objects": None},
        # a listed value the table does have: the value wins
        {"sample_region_id": "D1_MFG", "donor_id": "D1", "tma": "1", "region": "MFG", "disease_group": "CTRL", "measure": "v2_plaque_density_mm2", "reason": "too_few_objects", "n_objects": 1},
    ]).to_csv(folder / "results_missing_donor_region.csv", index=False)


NAME = "v2/measures/results_donor_region.csv"


def test_without_the_table_of_reasons_nothing_extra_is_said(client, tables):
    feature = client.get(f"/api/tables/{NAME}/feature", params={"column": "v2_plaque_density_mm2"}).json()
    assert feature["gaps"] is None and len(feature["points"]) == 2
    page = client.get(f"/api/tables/{NAME}/rows").json()
    assert "gaps" not in page and "reasons" not in page


def test_a_plot_gets_every_row_without_a_value_and_why_and_no_pseudo_value(client, tables):
    write_reasons(tables)
    feature = client.get(f"/api/tables/{NAME}/feature", params={"column": "v2_plaque_density_mm2", "facet": "region"}).json()
    assert [p["v2_plaque_density_mm2"] for p in feature["points"]] == [0.0, 2.5]  # a true zero is a value
    gaps = {g["sample_region_id"]: g for g in feature["gaps"]}
    assert set(gaps) == {"D2_MFG", "D3_MFG", "D4_MFG", "D9_MFG"}  # D1 has its value; D4 is empty with no reason listed
    assert (gaps["D2_MFG"]["text"], gaps["D3_MFG"]["text"], gaps["D4_MFG"]["text"], gaps["D9_MFG"]["text"]) == ("too few plaques", "no grey matter", "not measured", "no image")
    assert "3 found" in gaps["D2_MFG"]["why"]
    assert gaps["D9_MFG"]["disease_group"] == "AD" and gaps["D2_MFG"]["region"] == "MFG"
    assert all("v2_plaque_density_mm2" not in g for g in feature["gaps"])
    # a measure nothing is listed for and with every value there: nothing missing
    assert client.get(f"/api/tables/{NAME}/feature", params={"column": "neun_profile_density_mm2"}).json()["gaps"] == []


def test_a_table_page_gives_the_reason_of_each_empty_cell_and_counts_them(client, tables):
    write_reasons(tables)
    page = client.get(f"/api/tables/{NAME}/rows", params={"columns": "sample_region_id,v2_plaque_density_mm2"}).json()
    assert page["columns"] == ["sample_region_id", "v2_plaque_density_mm2"]
    by_row = dict(zip((r["sample_region_id"] for r in page["rows"]), page["reasons"], strict=True))
    assert by_row["D0_MFG"] == {} and by_row["D1_MFG"] == {}
    assert by_row["D2_MFG"]["v2_plaque_density_mm2"]["text"] == "too few plaques"
    assert by_row["D4_MFG"]["v2_plaque_density_mm2"]["text"] == "not measured"
    assert page["gaps"] == {"rows": 5, "cells": {"too few plaques": 1, "no grey matter": 1, "not measured": 1}, "absent": {"no image": 1}}


def test_a_table_of_reasons_with_no_rows_means_nothing_is_missing(client, tables):
    (tables / "results_missing_donor_region.csv").write_text("sample_region_id,measure,reason,n_objects\n")
    feature = client.get(f"/api/tables/{NAME}/feature", params={"column": "neun_profile_density_mm2"}).json()
    assert feature["gaps"] == []
    empties = client.get(f"/api/tables/{NAME}/feature", params={"column": "v2_plaque_density_mm2"}).json()["gaps"]
    assert {g["text"] for g in empties} == {"not measured"}
