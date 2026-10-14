"""Guided setup: reading TMA maps and the donor case sheet from the files labs keep them in."""
import pytest
from setup_files import docx, xlsx

from stainid.setup import donors, maps
from stainid.setup.files import parse_text, read_tables
from stainid.setup.naming import Groups, IdStyle, braak, cerad, donor_id, find_pathology, region_name

GROUPS = Groups([{"code": "CT", "label": "Control"}, {"code": "ASYMP", "label": "Asymptomatic AD"}, {"code": "AD", "label": "AD"}])


# ---------------------------------------------------------------- naming

@pytest.mark.parametrize("text, expected", [("1673", "1673"), ("17 00", "1700"), ("698", "0698"), ("0698", "0698"), ("JOG3", "JOG_003"),
                                            ("JOG 3", "JOG_003"), ("jog_003", "JOG_003"), ("BRC#1673", "BRC_1673"), ("A-12-B", "A-12-B")])
def test_donor_ids_are_read_through_typing_slips(text, expected):
    assert donor_id(text) == expected


def test_id_style_and_series_prefix():
    assert donor_id("7", prefix="JOG") == "JOG_007"
    assert donor_id("698", IdStyle(digits=0)) == "698"
    assert donor_id("JOG3", IdStyle(prefixed_digits=2)) == "JOG_03"


def test_groups_map_common_words_to_the_projects_codes():
    groups = Groups([{"code": "CT", "label": "Control"}, {"code": "ASYM", "label": "ASYMAD"}, {"code": "AD", "label": "AD"}])
    assert [groups.code(t) for t in ("Control", "ctrl", "ASYMAD", "Asym AD", "asymp", "Alzheimer's", "AD", "FR")] == \
        ["CT", "CT", "ASYM", "ASYM", "ASYM", "AD", "AD", ""]
    assert groups.section("CONTROL <65 (90 cases)") == "CT" and groups.section("ASYMAD (40)") == "ASYM"
    assert Groups().code("ASYMAD") == "ASYMP"
    assert groups.find("1673 FR AD")[0] == "AD" and groups.find("AD12 FR")[0] == ""


def test_pathology_and_regions():
    assert find_pathology("C:C - B:6")[:2] == ("C", "6")
    assert find_pathology("C: 0 – B:II".replace("–", "-"))[:2] == ("0", "2")
    assert find_pathology("CERAD moderate, Braak IV")[:2] == ("B", "4")
    assert (cerad("frequent"), braak("VI"), braak("stage 3"), cerad("x")) == ("C", "6", "3", "")
    assert (region_name("FR"), region_name("OC C"), region_name("hippo"), region_name("XYZ")) == ("frontal", "occipital", "hippocampus", "xyz")


# ---------------------------------------------------------------- map cells

@pytest.mark.parametrize("text, donor, region, group, pathology", [
    ("1673 FR\nAD\nC:C - B:6", "1673", "frontal", "AD", ("C", "6")),
    ("17 00 FR", "1700", "frontal", "", ("", "")),
    ("1700 OC C\nCT", "1700", "occipital", "CT", ("", "")),
    ("JOG3 FR\nASYMAD\nC:0 – B:2", "JOG_003", "frontal", "ASYMP", ("0", "2")),
    ("2323 OCC AD C:B-B:IV", "2323", "occipital", "AD", ("B", "4")),
    ("0698 FR asym ad", "0698", "frontal", "ASYMP", ("", "")),
])
def test_map_cells(text, donor, region, group, pathology):
    cell = maps.parse_cell(text, GROUPS)
    assert (cell["donor_id"], cell["region"], cell["disease_group"], (cell["cerad"], cell["braak"]), cell["note"]) == \
        (donor, region, group, pathology, "")


def test_controls_empty_and_unreadable_cells():
    assert maps.parse_cell("CBLM", GROUPS)["tissue_control"] == "CBLM"
    assert maps.parse_cell("  STOM ", GROUPS)["donor_id"] == ""
    assert maps.parse_cell("", GROUPS)["raw"] == "" and maps.parse_cell("x", GROUPS)["note"] == ""
    assert maps.parse_cell("1 9 35", GROUPS)["note"] == "No brain region in this cell"
    assert maps.parse_cell("see notes ??", GROUPS)["note"] == maps.UNREADABLE
    assert maps.parse_cell("AD", GROUPS)["note"] == "This cell has no donor"


def test_a_saved_cell_reads_back_the_same():
    cell = maps.parse_cell("JOG3 FR\nAD\nC:C - B:6", GROUPS)
    row = maps.layout_rows("8", [{"label": "A-1", **cell}])[0]
    again = maps.parse_cell(maps.cell_text(row), GROUPS)
    assert {k: again[k] for k in ("donor_id", "region", "disease_group", "cerad", "braak")} == \
        {"donor_id": "JOG_003", "region": "frontal", "disease_group": "AD", "cerad": "C", "braak": "6"}


# ---------------------------------------------------------------- map grids from files

WORD_MAP = [["", "A", "B", "C"],
            ["1", "1673 FR", "17 00 FR", "CBLM"], ["", "AD", "CT", ""], ["", "C:C - B:6", "C:0-B:1", ""],
            ["2", "JOG3 FR", "1 9 35 OCC", ""], ["", "ASYMAD", "AD", ""], ["", "C:A - B:3", "", ""]]


def lab_word_map(rows: int) -> list[list[str]]:
    """A Word map as the lab draws it: 6 columns, three table rows per core position (id + region / group /
    pathology), the control core's text on the middle line, no header row or column (title and row numbers are
    outside the table), with the typing slips of real maps."""
    ids = ["1673 FR", "17 00  FR", "1 9 35   FR", "JOG 14   FR", "1696   OC C", "2323 OCC"]
    group = ["AD", "A D", "CT", "AD", "CT", "AD"]
    pathology = ["C:C - B:6", "C: C  - B: 6", "C:C  –  B : 6", "C:C - B6", "C:0 - B:0", "C:B - B:3"]
    table = []
    for r in range(rows):
        shift = lambda values: values[r % 6:] + values[:r % 6]  # noqa: E731
        lines = [shift(ids), shift(group), shift(pathology)]
        if r == 0:  # the control core sits at A-1: its "CBLM" is in the group line, the other two lines are empty
            lines[0][0], lines[1][0], lines[2][0] = "", "CBLM", ""
        table += lines
    return table


@pytest.mark.parametrize("rows", [5, 4])
def test_the_labs_word_maps_three_table_rows_per_position(rows):
    content = docx([("LIP- 8", lab_word_map(rows))])
    found = maps.read_maps(read_tables("LIP8.docx", content), GROUPS, ["8"])
    assert len(found) == 1 and found[0]["tma"] == "8"
    cells = {c["label"]: c for c in maps.parse_map(found[0]["grid"], GROUPS)}
    assert len(cells) == 6 * rows and len(found[0]["grid"]) == rows
    assert (cells["A-1"]["tissue_control"], cells["A-1"]["donor_id"], cells["A-2"]["tissue_control"]) == ("CBLM", "", "")
    read = [(cells[f"{c}-1"]["donor_id"], cells[f"{c}-1"]["region"], cells[f"{c}-1"]["disease_group"], cells[f"{c}-1"]["cerad"],
             cells[f"{c}-1"]["braak"]) for c in "BCDEF"]
    assert read == [("1700", "frontal", "AD", "C", "6"), ("1935", "frontal", "CT", "C", "6"), ("JOG_014", "frontal", "AD", "C", "6"),
                    ("1696", "occipital", "CT", "0", "0"), ("2323", "occipital", "AD", "B", "3")]
    assert all(not c["note"] for c in cells.values())
    assert cells[f"A-{rows}"]["donor_id"] and cells[f"F-{rows}"]["disease_group"]


def test_a_word_map_with_three_table_rows_per_map_row():
    tables = read_tables("LIP8.docx", docx([("TMA LIP-8", WORD_MAP)]))
    assert tables[0].name == "TMA LIP-8" and tables[0].source == "LIP8.docx"
    found = maps.read_maps(tables, GROUPS, ["8", "9"])
    assert len(found) == 1 and found[0]["tma"] == "8"
    cells = {c["label"]: c for c in maps.parse_map(found[0]["grid"], GROUPS)}
    assert set(cells) == {"A-1", "B-1", "C-1", "A-2", "B-2", "C-2"}
    assert (cells["A-1"]["donor_id"], cells["A-1"]["disease_group"], cells["A-1"]["cerad"]) == ("1673", "AD", "C")
    assert (cells["B-2"]["donor_id"], cells["B-2"]["region"]) == ("1935", "occipital")
    assert cells["C-1"]["tissue_control"] == "CBLM" and cells["C-2"]["raw"] == ""
    assert cells["A-2"]["donor_id"] == "JOG_003" and cells["A-2"]["disease_group"] == "ASYMP"


def test_an_excel_map_with_one_cell_per_core_and_several_maps_in_a_sheet():
    rows = [["TMA 9"], ["", "A", "B"], [1, "1141 FR\nAD", "1141 OCC\nAD"], [2, "698 FR CT", None], [],
            ["TMA 10"], ["2001 FR AD", "2002 FR CT"]]
    found = maps.read_maps(read_tables("maps.xlsx", xlsx({"Sheet1": rows})), GROUPS, ["9", "10"])
    assert [(m["name"], m["tma"]) for m in found] == [("TMA 9", "9"), ("TMA 10", "10")]
    cells = maps.parse_map(found[0]["grid"], GROUPS)
    assert [(c["label"], c["donor_id"], c["region"]) for c in cells] == \
        [("A-1", "1141", "frontal"), ("B-1", "1141", "occipital"), ("A-2", "0698", "frontal"), ("B-2", "", "")]
    assert maps.summary(cells)["donors"] == 2


def test_csv_and_pasted_cells():
    assert parse_text("a,b\n1,\"x\ny\"\n") == [["a", "b"], ["1", "x\ny"]]
    assert parse_text("1673 FR\tCBLM\n\"JOG3 FR\nAD\"\t\n") == [["1673 FR", "CBLM"], ["JOG3 FR\nAD", ""]]
    found = maps.read_maps(read_tables("LIP-12 map.csv", b"1673 FR AD,1700 FR CT\n"), GROUPS, ["12"])
    assert found[0]["tma"] == "12" and found[0]["grid"] == [["1673 FR AD", "1700 FR CT"]]


def test_pictures_and_old_formats_say_what_to_do():
    with pytest.raises(ValueError, match="type the map into the grid"):
        read_tables("map photo.jpg", b"\xff\xd8")
    with pytest.raises(ValueError, match="save it as .xlsx"):
        read_tables("cases.xls", b"")


def test_saving_one_tma_keeps_the_others(tmp_path):
    path = tmp_path / "tma_layout.csv"
    path.write_text("tma,core_label,donor_id,region,disease_group,sample_region_id\n1,A-1,0001,frontal,AD,BRC_0001_FR\n8,A-1,9999,frontal,CT,\n")
    cells = maps.parse_map([["1673 FR AD", "CBLM"]], GROUPS)
    assert maps.save_map(path, "8", maps.layout_rows("8", cells, "LIP8.docx")) == 2
    rows = maps.read_layout_rows(path)
    assert [(r["tma"], r["core_label"], r["donor_id"]) for r in rows] == [("1", "A-1", "0001"), ("8", "A-1", "1673"), ("8", "B-1", "")]
    assert rows[0]["sample_region_id"] == "BRC_0001_FR" and rows[2]["tissue_control"] == "CBLM" and rows[1]["map_source"] == "LIP8.docx"
    assert maps.grid_of(maps.map_of(path, "8"), (2, 2)) == [["1673 frontal\nAD", "CBLM"], ["", ""]]


# ---------------------------------------------------------------- donor case sheet

CASES = [
    ["Troncoso cohort 2026"],
    ["BRC#", "Source", "Clinical Dx", "Final Dx", "CERAD", "Braak", "Age", "Sex", "Race", "PMD", "APOE", "TMA", "Brain weight (g)"],
    ["CONTROL <65 (90 cases)"],
    [1673, "BLSA", "Normal", "Normal", "0", "I", "62", "F", "W", "~15", "3/3", "LIP-8", 1300],
    [1700, "BLSA", "Normal", "Normal", "A", "2", ">90", "M", "B", "NA", "E3/E4", "LIP-8", None],
    ["ASYMAD (40 cases)"],
    [2323, "BLSA", "Normal", "AD path", "B", "IV", "88", "male", "W", "8.5", "34", "LIP-9", None],
    ["JOG#", "Source", "Clinical Dx", "Final Dx", "CERAD", "Braak", "Age", "Sex", "Race", "PMD", "APOE", "TMA", ""],
    ["AD"],
    [3, "JHU", "AD", "AD", "C", "VI", "75", "f", "W", "12", "4/4", "LIP-8", None],
    [7, "JHU", "AD", "AD", "C", "V", "80", "F", "W", "6", "3/4", "LIP-9", None],
    ["notes: two rows below are pending", "", "", "", "", "", "", "", "", "", "", "", ""],
]


def test_the_case_sheet_is_read_like_a_person_would():
    sheet = donors.read_donors(read_tables("cases.xlsx", xlsx({"Cases": CASES}))[0].rows, GROUPS, map_prefixes={"JOG"})
    assert sheet.header_row == 1
    fields = {c["header"]: c["field"] for c in sheet.columns}
    assert fields["BRC#"] == "donor_id" and fields["PMD"] == "pmd_hours" and fields["Final Dx"] == "final_diagnosis" and fields["Brain weight (g)"] == ""
    by_id = {r["donor_id"]: r for r in sheet.records}
    assert list(by_id) == ["1673", "1700", "2323", "JOG_003", "JOG_007"]
    assert [by_id[d]["disease_group"] for d in by_id] == ["CT", "CT", "ASYMP", "AD", "AD"]
    assert (by_id["1673"]["pmd_hours"], by_id["1673"]["pmd_hours_raw"], by_id["1700"]["pmd_hours"]) == ("15", "~15", "")
    assert (by_id["1700"]["age"], by_id["1700"]["age_topcoded"], by_id["1700"]["age_raw"]) == ("90", "true", ">90")
    assert (by_id["1700"]["apoe"], by_id["1700"]["apoe_e4_count"], by_id["2323"]["apoe"], by_id["JOG_003"]["apoe_e4_count"]) == ("3/4", "1", "3/4", "2")
    assert (by_id["2323"]["braak"], by_id["2323"]["cerad"], by_id["2323"]["sex"], by_id["JOG_003"]["sex"]) == ("4", "B", "M", "F")
    assert (by_id["JOG_007"]["tma"], by_id["JOG_007"]["sheet_tma"], by_id["1673"]["brain_weight_g"]) == ("9", "LIP-9", "1300")
    assert by_id["1673"]["sheet_row"] == "4"
    assert any("Sections" in n and "CT" in n for n in sheet.notes) and any("approximate" in n for n in sheet.notes)


def test_without_a_map_the_first_id_series_stays_plain():
    sheet = donors.read_donors(read_tables("cases.xlsx", xlsx({"Cases": CASES}))[0].rows, GROUPS)
    assert [r["donor_id"] for r in sheet.records][-2:] == ["JOG_003", "JOG_007"]
    plain = donors.read_donors(read_tables("cases.xlsx", xlsx({"Cases": CASES}))[0].rows, GROUPS, map_prefixes=set())
    assert [r["donor_id"] for r in plain.records][-2:] == ["JOG_003", "JOG_007"]


def test_column_matches_can_be_changed():
    rows = read_tables("cases.xlsx", xlsx({"Cases": CASES}))[0].rows
    sheet = donors.read_donors(rows, GROUPS, mapping={12: "skip", 3: "skip"})
    assert "brain_weight_g" not in sheet.records[0] and "final_diagnosis" not in sheet.records[0]
    with pytest.raises(ValueError, match="column names"):
        donors.read_donors([["a", "b"], ["1", "2"]], GROUPS)


def test_choosing_and_saving_donors(tmp_path):
    sheet = donors.read_donors(read_tables("cases.xlsx", xlsx({"Cases": CASES}))[0].rows, GROUPS, map_prefixes={"JOG"})
    kept = donors.select_donors(sheet.records, {"1673", "JOG_003"}, {"8"})
    assert [r["donor_id"] for r in kept] == ["1673", "1700", "JOG_003"]
    path = tmp_path / "donor_metadata.csv"
    path.write_text("donor_id,disease_group,age\n0001,AD,70\n1673,AD,1\n")
    assert donors.save_donors(path, kept) == 4
    table = {r["donor_id"]: r for r in donors.read_donor_table(path)}
    assert table["0001"]["age"] == "70" and table["1673"]["disease_group"] == "CT" and path.read_text().startswith("donor_id,tma,disease_group,age")
    assert donors.save_donors(path, kept, "replace") == 3


# ---------------------------------------------------------------- sample-region ids

PROJECT_LAYOUT = [{"tma": "1", "core_label": "A-1", "donor_id": "1673", "region": "frontal", "sample_region_id": "BRC_1673_FR"},
                  {"tma": "1", "core_label": "B-1", "donor_id": "1673", "region": "occipital", "sample_region_id": "BRC_1673_OCC"},
                  {"tma": "1", "core_label": "C-1", "donor_id": "JOG_003", "region": "frontal", "sample_region_id": "JOG_003_FR"},
                  {"tma": "1", "core_label": "D-1", "donor_id": "", "region": "", "sample_region_id": ""}]


def test_sample_region_ids_follow_the_projects_saved_maps():
    rule = maps.infer_sample_ids(PROJECT_LAYOUT)
    assert (rule.source, rule.plain_prefix, rule.prefixed_prefix, rule.regions) == ("layout", "BRC_", "", {"frontal": "FR", "occipital": "OCC"})
    cells = {c["label"]: c for c in maps.parse_map([["2002 FR AD", "JOG7 OC C AD", "2002 frontal", "2003 TEMP", "CBLM"]], GROUPS, sample_ids=rule)}
    assert [cells[p]["sample_region_id"] for p in ("A-1", "B-1", "C-1", "D-1", "E-1")] == ["BRC_2002_FR", "JOG_007_OCC", "BRC_2002_FR", "BRC_2003_TEMP", ""]
    assert maps.layout_rows("9", [cells["A-1"]])[0]["sample_region_id"] == "BRC_2002_FR"


def test_sample_region_ids_without_saved_maps_use_the_setting():
    from stainid.project import DEFAULTS

    rule = maps.sample_ids_for(DEFAULTS, [])
    assert rule.source == "setting" and rule.make("1673", "frontal", "FR") == "1673_frontal"  # the same as an empty column gives
    coded = maps.sample_ids_for({"sample_region_ids": {"plain_prefix": "BRC_", "region": "code"}}, [])
    assert (coded.make("1673", "frontal", "FR"), coded.make("JOG_003", "frontal", "FR"), coded.make("1141", "hippocampus", "")) == \
        ("BRC_1673_FR", "JOG_003_FR", "BRC_1141_HIP")
