"""TMA maps: which donor, region and group sits at each core position, read from the grids labs draw them as.

A map is drawn like the array: map rows 1, 2, … from the top and columns A, B, … from the left, so a cell's position
is "<column>-<row>" (B-2), the core label Find cores gives. A cell names the donor and region and usually the group,
sometimes the pathology: "1673 FR / AD / C:C - B:6", on one to three lines of the cell or on one to three table rows
per map row (Word maps often use three). Typing slips are read through: stray spaces ("17 00 FR", "OC C"), en
dashes, prefixed ids ("JOG3 FR" -> JOG_003, see `naming.donor_id`). Cells naming a control tissue ("CBLM", "STOM")
are control cores; empty cells are positions without a donor.

`read_maps` finds the maps in tables (`files.read_tables`), `parse_map` reads each cell, and `save_map` writes one
TMA's rows into the project's `tma_layout` (the other TMAs are kept).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from stainid.setup.files import Table
from stainid.setup.naming import (
    EMPTY_MARKS,
    REGIONS,
    Groups,
    IdStyle,
    clean,
    control_tissue,
    donor_id,
    find_pathology,
    id_prefix,
    key,
    region_name,
    tma_number,
)
from stainid.slides.layout import REQUIRED_COLUMNS
from stainid.tables import read_csv, write_csv

TITLE = re.compile(r"\b(TMA|LIP|MAP|ARRAY|SLIDE|BLOCK|LAYOUT)\b", re.I)
CELL_ID = re.compile(r"([A-Z]*?)(\d+)([A-Z]*)")
UNREADABLE = "The donor could not be read from this cell"
LAYOUT_COLUMNS = list(REQUIRED_COLUMNS) + ["sample_region_id", "tissue_control", "cerad", "braak", "map_source", "map_pathology_raw"]


def column_label(index: int) -> str:
    return chr(ord("A") + index) if index < 26 else column_label(index // 26 - 1) + chr(ord("A") + index % 26)


def position(row: int, column: int) -> str:
    """The core label of a map cell (0-based row and column): B-2 for row 1, column 1."""
    return f"{column_label(column)}-{row + 1}"


def split_position(label: str) -> tuple[int, int]:
    """0-based (row, column) of a core label such as B-2."""
    match = re.fullmatch(r"([A-Z]+)-(\d+)", label)
    if not match:
        raise ValueError(f"Not a core position: {label!r}")
    column = 0
    for letter in match.group(1):
        column = column * 26 + ord(letter) - ord("A") + 1
    return int(match.group(2)) - 1, column - 1


# ---------------------------------------------------------------- one cell

def parse_cell(text: str, groups: Groups, style: IdStyle = IdStyle()) -> dict[str, str]:
    """What one map cell says: donor_id, region (and the code as written), disease_group, cerad, braak,
    tissue_control, and a `note` in plain words when part of it could not be read."""
    cell = {"raw": clean(text), "donor_id": "", "region": "", "region_code": "", "disease_group": "", "cerad": "", "braak": "",
            "pathology_raw": "", "tissue_control": "", "note": ""}
    rest = clean(text).upper()
    if key(rest) in EMPTY_MARKS or not re.search(r"[A-Z0-9]", rest):
        return cell
    cerad, braak, span = find_pathology(rest)
    if span:
        cell.update(cerad=cerad, braak=braak, pathology_raw=f"C:{cerad} - B:{braak}")
        rest = rest[:span[0]] + " " + rest[span[1]:]
    lines = [line for line in rest.split("\n") if line.strip()]
    group_lines = [line for line in lines if groups.code(line)]
    if group_lines:
        cell["disease_group"] = groups.code(group_lines[-1])
        lines.remove(group_lines[-1])
        rest = "\n".join(lines)
    else:
        code, where = groups.find(rest)
        if where:
            cell["disease_group"] = code
            rest = rest[:where[0]] + " " + rest[where[1]:]
    control = control_tissue(rest)
    if control:
        cell["tissue_control"] = control
        return cell
    match = CELL_ID.fullmatch(key(rest))
    if not match:
        if key(rest):
            cell["note"] = UNREADABLE
        elif cell["disease_group"] or cell["cerad"]:
            cell["note"] = "This cell has no donor"
        return cell
    prefix, number, region = match.groups()
    cell.update(donor_id=donor_id(prefix + number, style), region_code=region, region=region_name(region) if region else "")
    if not region:
        cell["note"] = "No brain region in this cell"
    return cell


def cell_text(row: dict[str, str]) -> str:
    """A saved map row written back as a cell, for editing: "1673 frontal\\nAD\\nC:C - B:6"."""
    if row.get("tissue_control") and not row.get("donor_id"):
        return row["tissue_control"]
    if not row.get("donor_id"):
        return ""
    lines = [f"{row['donor_id']} {row.get('region', '')}".strip(), row.get("disease_group", "")]
    if row.get("cerad") or row.get("braak"):
        lines.append(f"C:{row.get('cerad') or '?'} - B:{row.get('braak') or '?'}")
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------- a table as a map grid

def _kind(text: str, groups: Groups) -> str:
    """identity (a donor or a control: a new map row starts here), info (only a group, region or pathology) or empty."""
    if not key(text):
        return "empty"
    cell = parse_cell(text, groups)
    if cell["donor_id"] or cell["tissue_control"] or (cell["note"] == UNREADABLE and re.search(r"\d", text)):
        return "identity"
    return "info"


def _is_title(row: list[str]) -> bool:
    filled = [c for c in row if c]
    return len(filled) == 1 and bool(TITLE.search(filled[0])) and len(row) > 1


def map_grid(rows: list[list[str]], groups: Groups) -> list[list[str]]:
    """The map cells of a table, one list per map row: header rows (A, B, C …) and columns (1, 2, 3 …) and title rows
    are dropped, and a map row drawn over several table rows (donor / group / pathology) is joined into one."""
    rows = [[clean(c) for c in row] for row in rows if any(clean(c) for c in row) and not _is_title([clean(c) for c in row])]
    if not rows:
        return []
    if all(re.fullmatch(r"[A-Z]", c.upper()) for c in rows[0] if c) and sum(bool(c) for c in rows[0]) >= 2:
        rows = rows[1:]
    first = [r[0] for r in rows if r and r[0]]
    if len(first) >= 2 and all(re.fullmatch(r"(ROW\s*)?\d{1,2}", c.upper()) for c in first) and \
            [int(re.sub(r"\D", "", c)) for c in first] == sorted(int(re.sub(r"\D", "", c)) for c in first):
        rows = [r[1:] for r in rows]
    width = max(len(r) for r in rows)
    rows = [(r + [""] * width)[:width] for r in rows]
    grid: list[list[list[str]]] = []
    for row in rows:
        if any(_kind(c, groups) == "identity" for c in row) or not grid:
            grid.append([[c] if c else [] for c in row])
        else:
            for parts, c in zip(grid[-1], row):
                if c:
                    parts.append(c)
    cells = [["\n".join(parts) for parts in row] for row in grid]
    while cells and all(not any(r[-1:]) for r in cells) and len(cells[0]) > 1:  # empty columns on the right
        cells = [r[:-1] for r in cells]
    return cells


def guess_tma(name: str, known: list[str]) -> str:
    """The TMA a map is for, from its sheet, heading or file name ("LIP8.docx" -> 8); only a TMA that has slides when
    `known` is given."""
    numbers = [str(int(n)) for n in re.findall(r"\d+", name)]
    if known:
        return next((n for n in reversed(numbers) if n in known), "")
    return numbers[-1] if numbers else ""


def read_maps(tables: list[Table], groups: Groups, known_tmas: list[str] = ()) -> list[dict]:
    """The maps in some tables: name, source file, guessed TMA and the grid of cell texts. A sheet with several maps
    one below the other (each under a title such as "TMA 8") gives one map each."""
    maps = []
    for table in tables:
        parts, title, current = [], table.name, []
        for row in table.rows:
            if not _is_title(row):
                current.append(row)
                continue
            if any(any(row) for row in current):  # a title below a map starts the next map
                parts.append((title, current))
                current = []
            title = next(c for c in row if c)
        parts.append((title, current))
        for name, rows in parts:
            grid = map_grid(rows, groups)
            if grid:
                maps.append({"name": name, "source": table.source, "tma": guess_tma(name, list(known_tmas)) or
                             guess_tma(Path(table.source).stem, list(known_tmas)), "grid": grid})
    return maps


# ---------------------------------------------------------------- sample-region ids

@dataclass(frozen=True)
class SampleIds:
    """How a core's donor-region is named (`sample_region_id`): <prefix><donor_id>_<region>. `plain_prefix` goes before
    plain numeric donor ids (BRC_1673_FR), `prefixed_prefix` before ids that already carry one (JOG_003_FR); the region
    is its name (frontal) or its code (FR), or what `regions` says for it. `source` says where the rule comes from:
    "layout" (learned from the saved map rows) or "setting" (`sample_region_ids` in stainid.yaml)."""
    plain_prefix: str = ""
    prefixed_prefix: str = ""
    region: str = "name"  # name | code
    regions: dict = field(default_factory=dict)
    source: str = "setting"

    @classmethod
    def from_config(cls, config: dict) -> SampleIds:
        rule = config.get("sample_region_ids") or {}
        return cls(str(rule.get("plain_prefix", "")), str(rule.get("prefixed_prefix", "")), str(rule.get("region", "name")))

    def make(self, donor: str, region: str, region_code: str = "") -> str:
        if not donor:
            return ""
        if region in self.regions:
            suffix = self.regions[region]
        elif self.region == "code":
            suffix = key(region_code) if region_code and key(region_code) != key(region) else REGIONS.get(region, (key(region),))[0]
        else:
            suffix = region
        prefix = self.prefixed_prefix if id_prefix(donor) else self.plain_prefix
        return f"{prefix}{donor}_{suffix}" if suffix else f"{prefix}{donor}"

    def describe(self) -> dict:
        return {"plain_prefix": self.plain_prefix, "prefixed_prefix": self.prefixed_prefix, "region": self.region, "regions": self.regions,
                "source": self.source, "examples": [self.make("1673", "frontal", "FR"), self.make("JOG_003", "occipital", "OCC")]}


def infer_sample_ids(rows: list[dict[str, str]], fallback: SampleIds = SampleIds()) -> SampleIds:
    """The naming the saved map rows use, e.g. BRC_1673_FR and JOG_003_OCC -> plain prefix "BRC_", region codes
    frontal -> FR, occipital -> OCC. `fallback` when no row shows it."""
    plain, prefixed, suffixes = Counter(), Counter(), {}
    for row in rows:
        donor, region, sample = row.get("donor_id", ""), row.get("region", ""), row.get("sample_region_id", "")
        at = sample.find(f"{donor}_") if donor and sample else -1
        if at < 0:
            continue
        (prefixed if id_prefix(donor) else plain)[sample[:at]] += 1
        suffixes.setdefault(region, Counter())[sample[at + len(donor) + 1:]] += 1
    if not suffixes:
        return fallback
    regions = {region: counts.most_common(1)[0][0] for region, counts in suffixes.items() if region}
    coded = any(suffix != region for region, suffix in regions.items())
    pick = lambda counts, default: counts.most_common(1)[0][0] if counts else default  # noqa: E731
    return SampleIds(pick(plain, fallback.plain_prefix), pick(prefixed, fallback.prefixed_prefix), "code" if coded else "name", regions, "layout")


def sample_ids_for(config: dict, layout: list[dict[str, str]]) -> SampleIds:
    """The project's rule: learned from its saved map rows when they show one, else its `sample_region_ids` setting."""
    return infer_sample_ids(layout, SampleIds.from_config(config))


# ---------------------------------------------------------------- parsed maps and the layout file

def parse_map(grid: list[list[str]], groups: Groups, style: IdStyle = IdStyle(), sample_ids: SampleIds = SampleIds()) -> list[dict]:
    """Every cell of a map grid, read, with its core label and sample-region id."""
    cells = [{"label": position(r, c), **parse_cell(text, groups, style)} for r, row in enumerate(grid) for c, text in enumerate(row)]
    for cell in cells:
        cell["sample_region_id"] = sample_ids.make(cell["donor_id"], cell["region"], cell["region_code"])
    return cells


def summary(cells: list[dict]) -> dict:
    donors = {c["donor_id"] for c in cells if c["donor_id"]}
    return {"positions": len(cells), "cores": sum(bool(c["donor_id"]) for c in cells), "donors": len(donors),
            "controls": sum(bool(c["tissue_control"]) for c in cells), "empty": sum(not c["raw"] for c in cells),
            "unreadable": sum(bool(c["note"]) for c in cells), "groups": sorted({c["disease_group"] for c in cells if c["disease_group"]}),
            "regions": sorted({c["region"] for c in cells if c["region"]})}


def layout_rows(tma: str, cells: list[dict], source: str = "") -> list[dict[str, str]]:
    """Rows of `tma_layout` for one TMA (one per map position)."""
    return [{"tma": str(tma), "core_label": c["label"], "donor_id": c["donor_id"], "region": c["region"],
             "disease_group": c["disease_group"] if c["donor_id"] else "", "sample_region_id": c.get("sample_region_id", ""),
             "tissue_control": c["tissue_control"],
             "cerad": c["cerad"], "braak": c["braak"], "map_source": source, "map_pathology_raw": c.get("pathology_raw", "")}
            for c in cells]


def _order(row: dict[str, str]) -> tuple:
    try:
        r, c = split_position(row["core_label"])
    except ValueError:
        r, c = 10**6, 0
    return (len(row["tma"]), row["tma"], r, c)


def read_layout_rows(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if Path(path).exists() else []


def save_map(path: Path, tma: str, rows: list[dict[str, str]]) -> int:
    """Put one TMA's map into the layout file, replacing that TMA's rows and keeping the others. Returns the row count."""
    tma = str(tma)
    if not tma:
        raise ValueError("Choose which TMA this map is for")
    labels = [r["core_label"] for r in rows]
    if len(set(labels)) != len(labels):
        raise ValueError("The map has the same position twice")
    kept = [r for r in read_layout_rows(path) if r["tma"] != tma]
    columns = LAYOUT_COLUMNS + [c for r in kept for c in r if c not in LAYOUT_COLUMNS]
    write_csv(Path(path), sorted(kept + [{**r, "tma": tma} for r in rows], key=_order), list(dict.fromkeys(columns)))
    return len(rows)


def map_of(path: Path, tma: str) -> list[dict[str, str]]:
    """One TMA's saved rows."""
    return [r for r in read_layout_rows(path) if r["tma"] == str(tma)]


def grid_of(rows: list[dict[str, str]], size: tuple[int, int] = (0, 0)) -> list[list[str]]:
    """Saved map rows back as a grid of cell texts, at least `size` (rows, columns) large."""
    places = {}
    for row in rows:
        try:
            places[split_position(row["core_label"])] = cell_text(row)
        except ValueError:
            continue
    height = max([size[0]] + [r + 1 for r, _ in places])
    width = max([size[1]] + [c + 1 for _, c in places])
    return [[places.get((r, c), "") for c in range(width)] for r in range(height)]


__all__ = ["map_grid", "parse_cell", "parse_map", "read_maps", "save_map", "layout_rows", "grid_of", "map_of", "summary", "tma_number"]
