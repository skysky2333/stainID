"""Slides table: which whole-slide scan is which TMA and stain (`slide_path,tma,stain`), and optionally the slide's
own core grid when it differs from the project's TMA layout: `rows`, `columns` (empty = found automatically by Find
cores, which then writes the size it chose with `grid_source` = auto) and `orientation` (empty = as scanned, or
rotate_180 / flip_left_right / flip_top_bottom when the slide lies turned or mirrored on the scan)."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from stainid.tables import read_csv, write_records

SLIDE_SUFFIXES = (".vsi", ".svs", ".ndpi", ".scn", ".mrxs", ".czi", ".tif", ".tiff")
COLUMNS = ["slide_path", "tma", "stain"]
GRID_COLUMNS = ["rows", "columns", "grid_source", "orientation"]  # optional per-slide grid


def find_slides(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in SLIDE_SUFFIXES and not p.name.startswith(".")) if Path(folder).is_dir() else []


def guess(path: Path, stains: list[str]) -> dict[str, str]:
    """Guess TMA number and stain from a file name such as `TMA LIP-3 AT8.vsi` -> tma 3, stain AT8."""
    stem = path.stem
    stain = next((s for s in sorted(stains, key=len, reverse=True) if re.search(rf"(?<![A-Za-z0-9]){re.escape(s)}(?![A-Za-z0-9])", stem, re.I)), "")
    rest = re.sub(re.escape(stain), " ", stem, flags=re.I) if stain else stem
    number = re.search(r"\d+", rest)
    return {"slide_path": str(path), "tma": (number.group(0).lstrip("0") or "0") if number else "", "stain": stain}


def guess_prefix(paths: list[Path], stains: list[str]) -> str:
    """The core-name prefix the file names suggest: `TMA LIP-3 AT8.vsi` -> `LIP-`; "" when they show none (or only "TMA")."""
    found = []
    for path in paths:
        stem = re.sub("|".join(re.escape(s) for s in sorted(stains, key=len, reverse=True)) or "$^", " ", Path(path).stem, flags=re.I)
        match = re.search(r"([A-Za-z]+)([-_ ]?)0*\d+", stem)
        if match and match.group(1).upper() != "TMA":
            found.append(f"{match.group(1).upper()}{match.group(2).strip() or '-'}")
    common = Counter(found).most_common(1)
    return common[0][0] if common and common[0][1] * 2 > len(paths) else ""


def read_slides(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if Path(path).exists() else []


def scan_slides(folder: Path, table: Path, stains: list[str]) -> list[dict[str, str]]:
    """Known slides keep their assignment; new files in the folder get a guess."""
    known = {row["slide_path"]: row for row in read_slides(table)}
    return [known.get(str(p)) or guess(p, stains) for p in find_slides(folder)]


def _write(table: Path, rows: list[dict[str, str]]) -> None:
    grid = [c for c in GRID_COLUMNS if any(row.get(c) for row in rows)]
    write_records(table, [{c: row.get(c, "") for c in COLUMNS + grid} for row in rows], COLUMNS + grid)


def save_slides(table: Path, rows: list[dict[str, str]], stains: list[str]) -> None:
    problems = [f"{Path(r['slide_path']).name}: choose a TMA and a stain" for r in rows if not r["tma"] or r["stain"] not in stains]
    pairs = [(r["tma"], r["stain"]) for r in rows]
    problems += [f"TMA {t} has more than one {s} slide" for t, s in sorted(set(pairs)) if pairs.count((t, s)) > 1]
    if problems:
        raise ValueError("; ".join(problems))
    known = {row["slide_path"]: row for row in read_slides(table)}
    # a slide's grid set in the table is kept when the editor does not send it back
    _write(table, [{**row, **{c: row[c] if c in row else known.get(row["slide_path"], {}).get(c, "") for c in GRID_COLUMNS}} for row in rows])


def set_slide_grid(table: Path, slide_path: str, rows: int | None, columns: int | None, orientation: str = "", source: str = "manual") -> None:
    """Give one slide its own grid (`source` manual), record the size Find cores chose (`source` auto), or clear it
    (rows / columns None: found automatically next time)."""
    slides = read_slides(table)
    if not any(s["slide_path"] == slide_path for s in slides):
        raise KeyError(slide_path)
    for slide in slides:
        if slide["slide_path"] == slide_path:
            slide.update({"rows": str(rows or ""), "columns": str(columns or ""), "orientation": orientation,
                          "grid_source": source if rows and columns else ""})
    _write(table, slides)
