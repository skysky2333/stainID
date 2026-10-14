from stainid.slides.table import read_slides, save_slides


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
