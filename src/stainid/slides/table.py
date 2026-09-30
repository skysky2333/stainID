"""Slides table: which whole-slide scan is which TMA and stain (`slide_path,tma,stain`)."""
from __future__ import annotations

import re
from pathlib import Path

from stainid.tables import read_csv, write_records

SLIDE_SUFFIXES = (".vsi", ".svs", ".ndpi", ".scn", ".mrxs", ".czi", ".tif", ".tiff")
COLUMNS = ["slide_path", "tma", "stain"]


def find_slides(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in SLIDE_SUFFIXES and not p.name.startswith(".")) if Path(folder).is_dir() else []


def guess(path: Path, stains: list[str]) -> dict[str, str]:
    """Guess TMA number and stain from a file name such as `TMA LIP-3 AT8.vsi` -> tma 3, stain AT8."""
    stem = path.stem
    stain = next((s for s in sorted(stains, key=len, reverse=True) if re.search(rf"(?<![A-Za-z0-9]){re.escape(s)}(?![A-Za-z0-9])", stem, re.I)), "")
    rest = re.sub(re.escape(stain), " ", stem, flags=re.I) if stain else stem
    number = re.search(r"\d+", rest)
    return {"slide_path": str(path), "tma": (number.group(0).lstrip("0") or "0") if number else "", "stain": stain}


def read_slides(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if Path(path).exists() else []


def scan_slides(folder: Path, table: Path, stains: list[str]) -> list[dict[str, str]]:
    """Known slides keep their assignment; new files in the folder get a guess."""
    known = {row["slide_path"]: row for row in read_slides(table)}
    return [known.get(str(p)) or guess(p, stains) for p in find_slides(folder)]


def save_slides(table: Path, rows: list[dict[str, str]], stains: list[str]) -> None:
    problems = [f"{Path(r['slide_path']).name}: choose a TMA and a stain" for r in rows if not r["tma"] or r["stain"] not in stains]
    pairs = [(r["tma"], r["stain"]) for r in rows]
    problems += [f"TMA {t} has more than one {s} slide" for t, s in sorted(set(pairs)) if pairs.count((t, s)) > 1]
    if problems:
        raise ValueError("; ".join(problems))
    write_records(table, [{c: row[c] for c in COLUMNS} for row in rows], COLUMNS)
