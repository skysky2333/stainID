"""The image list of an image-set or whole-slide project (`inputs.image_list`, data/images.csv): one row per image file
or slide, with the donor it comes from and its stain.

| column | |
|---|---|
| `image_id` | a short unique name (made from the file name when empty) |
| `path` | the file (absolute, or relative to the project folder); it is read where it lies, never copied |
| `stain` | the stain as the lab calls it (6E10, 4G8, AT8, PHF-1, NeuN, ...); `imaging.stains` says how it is read |
| `donor_id` | the donor; `region` the brain region (optional) |
| `slide_id`, `site`, `batch` | optional: which slide, lab or staining run the image comes from. Stain thresholds are set per batch (else per slide, else per site, else one for the whole stain) |
| `pixel_size_um` | optional: the image's own pixel size (else the file's, else the project's imaging setting) |
| `disease_group` | optional (else the donor table's group) |
| `split` | optional: train / val / test as the dataset's paper defines it |
"""
from __future__ import annotations

import re
from pathlib import Path

from stainid.imaging import sources
from stainid.imaging.settings import family_of
from stainid.project import Project
from stainid.tables import read_csv, write_csv

COLUMNS = ["image_id", "path", "stain", "donor_id", "region", "slide_id", "site", "batch", "pixel_size_um", "disease_group", "split"]
REQUIRED = ("path", "stain", "donor_id")
SCAN_LIMIT = 20_000  # files listed from one folder at most


def read_images(project: Project) -> list[dict[str, str]]:
    path = project.input("image_list")
    return [{k: (v or "").strip() for k, v in row.items() if k} for row in read_csv(path)] if path.exists() else []


def resolve(project: Project, row: dict) -> Path:
    path = Path(row["path"]).expanduser()
    return path if path.is_absolute() else Path(project.root / path)


def image_id(row: dict, taken: set[str]) -> str:
    """The row's id, or one made from its file name (unique within `taken`)."""
    base = row.get("image_id") or Path(row["path"]).stem or "image"
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("_") or "image"
    name, n = base, 2
    while name in taken:
        name, n = f"{base}_{n}", n + 1
    taken.add(name)
    return name


def clean_rows(rows: list[dict]) -> list[dict[str, str]]:
    """Rows with every column as text, ids filled in and unique; rows without a path are dropped."""
    taken: set[str] = set()
    out = []
    for row in rows:
        row = {k: "" if v is None else str(v).strip() for k, v in row.items() if k}
        if not row.get("path"):
            continue
        row["image_id"] = image_id(row, taken)
        out.append({**{c: row.get(c, "") for c in COLUMNS}, **{k: v for k, v in row.items() if k not in COLUMNS}})
    return out


def save_images(project: Project, rows: list[dict]) -> int:
    rows = clean_rows(rows)
    if not rows:
        raise ValueError("No images to save: every row needs a file")
    columns = list(dict.fromkeys(COLUMNS + [k for r in rows for k in r]))
    write_csv(project.input("image_list"), rows, columns)
    return len(rows)


def guess_stain(name: str, known: list[str]) -> str:
    """The stain named in a file name ("case12_4G8_frontal.png" -> 4G8)."""
    words = [w for w in re.split(r"[^A-Za-z0-9-]+", name) if w]
    for word in words:
        for stain in known:
            if word.lower() == stain.lower():
                return stain
    for word in words:
        if family_of(word):
            return word
    return ""


def scan_folder(folder: Path, known_stains: list[str], slides: bool) -> list[dict[str, str]]:
    """Every image (or slide) in a folder and its subfolders, with the stain guessed from the file name and the donor
    from the first subfolder the file is in (when the files are sorted into one folder per donor)."""
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a folder")
    wanted = (sources.SLIDES | sources.TIFFS) if slides else sources.IMAGE_SUFFIXES
    out = []
    for path in sorted(_walk(folder)):
        if path.suffix.lower() not in wanted and not sources.is_zarr(path):
            continue
        parts = path.relative_to(folder).parts
        out.append({"path": str(path), "stain": guess_stain(path.stem, known_stains), "donor_id": parts[0] if len(parts) > 1 else "",
                    "image_id": "", "region": ""})
        if len(out) >= SCAN_LIMIT:
            break
    return out


def _walk(folder: Path, depth: int = 3):
    """Files up to `depth` folders down (an OME-Zarr folder counts as one file); hidden names are skipped."""
    try:
        entries = sorted(folder.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.is_dir() and not sources.is_zarr(entry):
            if depth > 0:
                yield from _walk(entry, depth - 1)
        else:
            yield entry


def donor_rows(project: Project) -> list[dict[str, str]]:
    """The donors of the image list, as the donor import compares a case sheet with them (in place of a TMA map)."""
    return [{"donor_id": d, "tma": ""} for d in dict.fromkeys(r.get("donor_id", "") for r in read_images(project)) if d]


def problems(project: Project, rows: list[dict[str, str]]) -> list[str]:
    """What is missing in the image list, in plain words (empty: nothing)."""
    out = []
    for column in REQUIRED:
        empty = sum(not r.get(column) for r in rows)
        if empty:
            out.append(f"{empty} of {len(rows)} images have no {'donor' if column == 'donor_id' else column}")
    return out
