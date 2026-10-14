"""Build a fully synthetic project that is not set up yet, for trying the guided setup (and taking its screenshots).

    python tools/demo/make_setup_demo.py OUT_DIR [--stage slides|cores] [--seed 7]

The study "Setup demo (synthetic)" has three TMAs (DEMO-1 to DEMO-3) stained for NeuN and AT8. DEMO-1 and DEMO-2 are
3 x 5 arrays like the project setting; DEMO-3 is a smaller 2 x 5 array, so Find cores has to find its grid size by
itself. The folder `incoming/` holds what a lab would bring, with the usual slips:

- `DEMO-1 map.docx`: a Word table with three table rows per map row (donor + region / group / pathology), stray
  spaces ("17 00 FR", "OC C"), a prefixed id ("JOG3 FR") and control cores (CBLM, STOM);
- `DEMO-2 map.xlsx`: an Excel grid with one cell per core (lines inside the cell), row and column headers;
- `DEMO-3 map photo.jpg`: a photo of a map, which has to be typed into the grid;
- `cases.xlsx`: the case sheet: section rows (CONTROL / ASYMAD / AD), a JOG# sub-table, "~15" and ">90", one id used
  twice, one pathology that disagrees with the map, one donor missing and one extra.

With --stage cores (the default) the slides are registered and Find cores has run on them (on synthetic slide
pictures, through the same code as the real step; the real step cannot read these pictures as scans). With
--stage slides only the files exist. Nothing is read from a real study. Serve it like the main demo:

    HOME=/tmp/setup-demo-home stainid --project OUT_DIR serve --no-browser --host 127.0.0.1 --port 18556
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np

from stainid.project import DEFAULTS, Project, load_project, save_project
from stainid.slides import dearray
from stainid.slides.table import save_slides, scan_slides
from stainid.slides.vsi import SceneInfo, SlidePreview
from stainid.workflows.slides import dearray_slides

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_tissue import draw_core  # noqa: E402

PREFIX, STAINS, CELL, MARGIN = "DEMO-", ("NeuN", "AT8"), 230, 150
GROUPS = [{"code": "CT", "label": "Control", "color": 3}, {"code": "ASYMP", "label": "Asymptomatic AD", "color": 1}, {"code": "AD", "label": "AD", "color": 2}]
SEVERITY = {"CT": 0.05, "ASYMP": 0.4, "AD": 0.85}
PATHOLOGY = {"CT": ("0", "1"), "ASYMP": ("B", "3"), "AD": ("C", "6")}
# TMA: (rows, columns, [(donor as written on the map, donor id, group)], controls)
TMAS = {
    "1": (3, 5, [("1673", "1673", "CT"), ("17 00", "1700", "CT"), ("2323", "2323", "ASYMP"), ("698", "0698", "AD"), ("1141", "1141", "AD"),
                 ("JOG3", "JOG_003", "AD")], ["CBLM", "STOM"]),
    "2": (3, 5, [("1935", "1935", "CT"), ("2001", "2001", "ASYMP"), ("2002", "2002", "CT"), ("2150", "2150", "AD"), ("JOG7", "JOG_007", "ASYMP"),
                 ("JOG12", "JOG_012", "AD")], ["CBLM", "STOM"]),
    "3": (2, 5, [("2410", "2410", "CT"), ("2433", "2433", "AD"), ("2502", "2502", "ASYMP"), ("JOG15", "JOG_015", "AD")], ["CBLM"]),
}


def layout(tma: str) -> dict[tuple[int, int], dict]:
    """Which donor and region sits at each (row, column): the two regions of a donor side by side, controls at the end."""
    rows, columns, donors, controls = TMAS[tma]
    places = [(r, c) for r in range(rows) for c in range(columns)]
    out = {}
    for k, (written, donor, group) in enumerate(donors):
        for j, region in enumerate(("FR", "OCC")):
            out[places[2 * k + j]] = {"written": written, "donor": donor, "group": group, "region": region}
    for k, control in enumerate(controls):
        out[places[2 * len(donors) + k]] = {"control": control}
    return out


def cell_lines(cell: dict | None, slips: bool = True) -> list[str]:
    if not cell:
        return []
    if "control" in cell:
        return [cell["control"]]
    region = "OC C" if slips and cell["donor"] == "1700" and cell["region"] == "OCC" else cell["region"]
    cerad, braak = PATHOLOGY[cell["group"]]
    if cell["donor"] == "2323":
        cerad = "B"  # the case sheet says 0
    return [f"{cell['written']} {region}", "ASYMAD" if cell["group"] == "ASYMP" else cell["group"], f"C:{cerad} – B:{braak}"]


# ---------------------------------------------------------------- the lab's files

def word_map(tma: str) -> bytes:
    rows, columns, _, _ = TMAS[tma]
    places = layout(tma)
    table = [[""] + [chr(65 + c) for c in range(columns)]]
    for r in range(rows):
        lines = [cell_lines(places.get((r, c))) for c in range(columns)]
        for line in range(3):
            table.append([str(r + 1) if line == 0 else ""] + [cell[line] if line < len(cell) else "" for cell in lines])

    def paragraph(text: str) -> str:
        return f'<w:p><w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'
    body = paragraph(f"TMA {PREFIX}{tma}") + "<w:tbl>" + "".join("<w:tr>" + "".join(f"<w:tc>{paragraph(c)}</w:tc>" for c in row) + "</w:tr>"
                                                              for row in table) + "</w:tbl>"
    document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/'
                f'2006/main"><w:body>{body}</w:body></w:document>')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                         '<Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType='
                         '"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
        archive.writestr("_rels/.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                         '<Relationship Id="r1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                         'Target="word/document.xml"/></Relationships>')
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


def excel_map(tma: str, path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment

    rows, columns, _, _ = TMAS[tma]
    places = layout(tma)
    book = Workbook()
    sheet = book.active
    sheet.title = f"{PREFIX}{tma}"
    sheet.append([f"TMA map {PREFIX}{tma}"])
    sheet.append([""] + [chr(65 + c) for c in range(columns)])
    for r in range(rows):
        sheet.append([r + 1] + ["\n".join(cell_lines(places.get((r, c)))) for c in range(columns)])
    for row in sheet.iter_rows(min_row=3):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True)
    book.save(path)


def photo_map(tma: str, path: Path) -> None:
    rows, columns, _, _ = TMAS[tma]
    places = layout(tma)
    image = np.full((rows * 120 + 160, columns * 190 + 120, 3), (226, 232, 238), np.uint8)
    cv2.putText(image, f"TMA {PREFIX}{tma}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (40, 40, 60), 2, cv2.LINE_AA)
    for r in range(rows):
        for c in range(columns):
            x, y = 60 + c * 190, 100 + r * 120
            cv2.rectangle(image, (x, y), (x + 180, y + 110), (60, 60, 80), 2)
            for i, line in enumerate(cell_lines(places.get((r, c)))):
                cv2.putText(image, line, (x + 10, y + 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 120), 2, cv2.LINE_AA)
    matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 2.5, 1.0)
    cv2.imwrite(str(path), cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderValue=(90, 80, 70)), [cv2.IMWRITE_JPEG_QUALITY, 82])


def case_sheet(path: Path, rng: np.random.Generator) -> None:
    from openpyxl import Workbook

    header = ["BRC#", "Source", "Clinical Dx", "Final Dx", "CERAD", "Braak", "Age", "Sex", "Race", "PMD", "APOE", "TMA", "Brain weight (g)"]
    people = [(cell["donor"], cell["group"], tma) for tma in TMAS for cell in layout(tma).values() if "donor" in cell and cell["region"] == "FR"]
    people = [p for p in people if p[0] != "2502"] + [("2999", "AD", "3")]  # 2502 is missing from the sheet; 2999 is not on a map
    book = Workbook()
    sheet = book.active
    sheet.title = "Cases"
    sheet.append(["Demo brain bank, cases on the DEMO arrays (synthetic)"])
    sheet.append(header)

    def row(donor: str, group: str, tma: str) -> list:
        cerad, braak = PATHOLOGY[group]
        cerad = "0" if donor == "2323" else cerad  # the map says B
        number = int(donor.split("_")[-1])
        age = ">90" if donor == "1141" else str(int(rng.integers(62, 90)))
        pmd = "~15" if donor == "1673" else "NA" if donor == "2001" else f"{rng.uniform(3, 20):.1f}"
        apoe = rng.choice(["3/3", "3/4", "4/4", "2/3"] if group != "CT" else ["3/3", "2/3", "3/4"])
        diagnosis = {"CT": "Normal", "ASYMP": "AD pathology, not demented", "AD": "Alzheimer disease"}[group]
        return [number, "Demo", "Normal" if group != "AD" else "Dementia", diagnosis, cerad, braak, age, str(rng.choice(["F", "M"])), "W", pmd,
                str(apoe), f"{PREFIX}{tma}", int(rng.integers(1000, 1450))]
    for kind in ("BRC", "JOG"):
        if kind == "JOG":
            sheet.append(["JOG#"] + header[1:])
        for title, group in (("CONTROL <65 (synthetic cases)", "CT"), ("ASYMAD", "ASYMP"), ("AD (synthetic cases)", "AD")):
            members = [p for p in people if p[1] == group and p[0].startswith("JOG") == (kind == "JOG")]
            if members:
                sheet.append([title])
            for donor, group_, tma in members:
                sheet.append(row(donor, group_, tma))
            if kind == "JOG" and group == "ASYMP":
                sheet.append(row("JOG_007", "ASYMP", "5"))  # a second "JOG 7", on a TMA of another study
    book.save(path)


# ---------------------------------------------------------------- slides

def slide_picture(tma: str, stain: str, rng: np.random.Generator) -> np.ndarray:
    rows, columns, _, _ = TMAS[tma]
    image = np.full((2 * MARGIN + 3 * CELL, 2 * MARGIN + columns * CELL + 300, 3), 238, np.uint8)  # every slide is the same size
    cv2.rectangle(image, (image.shape[1] - 260, 40), (image.shape[1] - 40, 220), (210, 200, 190), -1)  # the label
    for (r, c), cell in layout(tma).items():
        seed = abs(hash((tma, r, c))) % (1 << 31)
        group = cell.get("group", "CT")
        core = draw_core(np.random.default_rng(seed + (stain == "AT8")), stain, SEVERITY[group], seed, 600)
        small = cv2.resize(core.rgb, (CELL - 40, CELL - 40), interpolation=cv2.INTER_AREA)
        x = MARGIN + c * CELL + 20 + int(rng.integers(-6, 7))
        y = MARGIN + r * CELL + 20 + int(rng.integers(-6, 7))
        area = image[y:y + CELL - 40, x:x + CELL - 40]
        mask = small.min(axis=2) < 225
        area[mask] = small[mask]
    return image


def build(out: Path, stage: str, seed: int) -> None:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new folder")
    rng = np.random.default_rng(seed)
    root = out.resolve()
    save_project(Project(root=root, config=DEFAULTS), {"name": "Setup demo (synthetic)", "tma": {"rows": 3, "columns": 5}, "groups": GROUPS})
    project = load_project(root)
    slides = root / "data" / "slides"
    slides.mkdir(parents=True)
    pictures = {}
    for tma in TMAS:
        for stain in STAINS:
            name = f"TMA {PREFIX}{tma} {stain}.tif"
            pictures[name] = slide_picture(tma, stain, rng)
            cv2.imwrite(str(slides / name), cv2.cvtColor(cv2.resize(pictures[name], None, fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR))
    incoming = root / "incoming"
    incoming.mkdir()
    (incoming / f"{PREFIX}1 map.docx").write_bytes(word_map("1"))
    excel_map("2", incoming / f"{PREFIX}2 map.xlsx")
    photo_map("3", incoming / f"{PREFIX}3 map photo.jpg")
    case_sheet(incoming / "cases.xlsx", rng)
    print(f"slides and the lab's files in {root}", flush=True)
    if stage == "slides":
        return

    save_slides(project.input("slides_table"), scan_slides(slides, project.input("slides_table"), list(project.stains)), list(project.stains))
    save_project(project, {"tma": {"prefix": PREFIX}})

    def preview(path, target_long_side):
        rgb = pictures[Path(path).name]
        scene = SceneInfo(0, "full", rgb.shape[1] * 18, rgb.shape[0] * 18)
        return SlidePreview(Path(path), rgb, scene, SceneInfo(1, "preview", rgb.shape[1], rgb.shape[0]), 18.0, 18.0, 0.2738, 0.2738)
    dearray.read_slide_preview = preview  # the synthetic pictures stand in for the slide reader
    dearray_slides(load_project(root))
    print(f"done: {root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path, help="new (empty) folder for the demo project")
    parser.add_argument("--stage", choices=("slides", "cores"), default="cores", help="how far the setup has got (default: cores found)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    build(args.out_dir, args.stage, args.seed)


if __name__ == "__main__":
    main()
