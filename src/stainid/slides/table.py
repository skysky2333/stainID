"""Slides table: which whole-slide scan is which TMA and stain (`slide_path,tma,stain`), and optionally the slide's
own core grid when it differs from the project's TMA layout: `rows`, `columns` (empty = found automatically by Find
cores, which then writes the size it chose with `grid_source` = auto) and `orientation` (empty = as scanned, or
rotate_180 / flip_left_right / flip_top_bottom when the slide lies turned or mirrored on the scan)."""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from stainid.tables import read_csv, write_records

SLIDE_SUFFIXES = (".vsi", ".svs", ".ndpi", ".scn", ".mrxs", ".czi", ".tif", ".tiff")
COLUMNS = ["slide_path", "tma", "stain"]
GRID_COLUMNS = ["rows", "columns", "grid_source", "orientation"]  # optional per-slide grid


def find_slides(folder: Path) -> list[Path]:
    """The scans in a folder, as clean paths: a folder set as `../scans` gives `/.../scans/x.vsi`, the form the slides
    table keeps, so a scan already registered is not taken for a new one."""
    folder = Path(os.path.normpath(folder))
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in SLIDE_SUFFIXES and not p.name.startswith(".")) if folder.is_dir() else []


@dataclass(frozen=True)
class NameReading:
    """What a scan's file name says about its TMA and stain. A value is only filled in when the name says it clearly;
    otherwise it is "" and `*_options` lists what the name might mean (empty when it says nothing)."""
    tma: str = ""
    stain: str = ""
    tma_options: tuple[str, ...] = ()
    stain_options: tuple[str, ...] = ()


DATE = re.compile(r"(?<!\d)(\d{4}[-_.]\d{1,2}[-_.]\d{1,2}|\d{1,2}[-_.]\d{1,2}[-_.]\d{4}|\d{8})(?!\d)")


def _word(text: str) -> str:
    return rf"(?<![A-Za-z0-9]){re.escape(text)}(?![A-Za-z0-9])"


def read_name(path: Path, stains: list[str], prefix: str = "") -> NameReading:
    """Read the TMA number and the stain from a file name such as `TMA LIP-3 AT8.vsi` (TMA 3, AT8).

    The stain counts when exactly one of the project's stains is in the name as a word of its own. The TMA counts when
    one number follows the core-name prefix (`LIP-3` with prefix LIP-) or the word TMA (`TMA 3`, `tma12`), or when the
    name holds only one number. Numbers glued to a unit (`3mm`, `40x`) and dates are not TMA numbers. Two stains, or
    several numbers none of which is marked as the TMA, leave the value empty: people choose it."""
    stem = Path(path).stem
    found = [s for s in sorted(stains, key=len, reverse=True) if re.search(_word(s), stem, re.I)]
    rest = stem
    for stain in found:
        rest = re.sub(_word(stain), " ", rest, flags=re.I)
    rest = DATE.sub(" ", rest)
    marker = "|".join(re.escape(m) for m in dict.fromkeys(m for m in (prefix.rstrip("-_ #").strip(), "TMA") if m))
    numbers, marked = [], []
    for match in re.finditer(r"(?<!\d)\d{1,4}(?![\dA-Za-z])", rest):
        value = match.group(0).lstrip("0") or "0"
        numbers.append(value)
        if re.search(rf"(?<![A-Za-z])(?:{marker})[-_ #]*$", rest[:match.start()], re.I):
            marked.append(value)
    numbers, marked = list(dict.fromkeys(numbers)), list(dict.fromkeys(marked))
    clear = marked or numbers
    order = {s: i for i, s in enumerate(stains)}
    return NameReading(tma=clear[0] if len(clear) == 1 else "", stain=found[0] if len(found) == 1 else "",
                       tma_options=tuple(clear[:6]) if len(clear) > 1 else (),
                       stain_options=tuple(sorted(found, key=order.get)) if len(found) > 1 else ())


def guess(path: Path, stains: list[str], prefix: str = "") -> dict[str, str]:
    """A slides-table row for a new scan: the TMA number and stain its file name gives clearly ("" otherwise)."""
    reading = read_name(path, stains, prefix)
    return {"slide_path": str(path), "tma": reading.tma, "stain": reading.stain}


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


def scan_slides(folder: Path, table: Path, stains: list[str], prefix: str = "") -> list[dict[str, str]]:
    """Known slides keep their assignment; new files in the folder get what their file name says clearly."""
    known = {row["slide_path"]: row for row in read_slides(table)}
    return [known.get(str(p)) or guess(p, stains, prefix) for p in find_slides(folder)]


def slide_rows(folder: Path, table: Path, stains: list[str], prefix: str = "") -> list[dict]:
    """The rows of the slides editor: every scan in the folder with its TMA and stain and whether it is new; for a new
    scan also which values came from its file name (`from_name`) and, where the name is unclear, what it might mean
    (`options`)."""
    known = {row["slide_path"]: row for row in read_slides(table)}
    rows = []
    for path in find_slides(folder):
        if str(path) in known:
            rows.append({**known[str(path)], "new": False, "from_name": {"tma": False, "stain": False}, "options": {"tma": [], "stain": []}})
            continue
        reading = read_name(path, stains, prefix)
        rows.append({"slide_path": str(path), "tma": reading.tma, "stain": reading.stain, "new": True,
                     "from_name": {"tma": bool(reading.tma), "stain": bool(reading.stain)},
                     "options": {"tma": list(reading.tma_options), "stain": list(reading.stain_options)}})
    return rows


def missing_values(rows: list[dict], stains: list[str]) -> dict[str, list[str]]:
    """The scans that still need a TMA, and those that still need one of the project's stains."""
    return {"tma": [r["slide_path"] for r in rows if not str(r.get("tma") or "").strip()],
            "stain": [r["slide_path"] for r in rows if r.get("stain") not in stains]}


def _still_need(paths: list[str], what: str) -> str:
    names = ", ".join(Path(p).name for p in paths[:3]) + (" …" if len(paths) > 3 else "")
    return f"{len(paths)} slide{'s still need' if len(paths) != 1 else ' still needs'} {what} ({names})"


def _write(table: Path, rows: list[dict[str, str]]) -> None:
    grid = [c for c in GRID_COLUMNS if any(row.get(c) for row in rows)]
    write_records(table, [{c: row.get(c, "") for c in COLUMNS + grid} for row in rows], COLUMNS + grid)


def save_slides(table: Path, rows: list[dict[str, str]], stains: list[str]) -> None:
    rows = [{**r, "tma": str(r.get("tma") or "").strip()} for r in rows]
    missing = missing_values(rows, stains)
    problems = [_still_need(paths, what) for paths, what in ((missing["tma"], "a TMA"), (missing["stain"], "a stain")) if paths]
    pairs = [(r["tma"], r["stain"]) for r in rows if r["tma"] and r["stain"] in stains]
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
