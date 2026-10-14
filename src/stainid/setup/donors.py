"""Donor information from the lab's case spreadsheet (one row per donor) into the project's `donor_metadata` table.

Case sheets are made for people, not programs, so the importer reads them the way a person would:

- the header row is the row whose cells look most like column names (Age, Sex, PMI, APOE, CERAD, Braak, TMA, …);
  each column is matched to a stainID column by its name, and the match can be changed;
- a row with only its first cell filled ("CONTROL <65 (90 cases)") is a section heading that gives the group of the
  rows below it (Control, asymptomatic AD, AD, or a group code of the project);
- a row that repeats the header ("JOG#", "Age", …) starts a sub-table whose ids form their own series: plain numbers
  there are written with that prefix (JOG_007) when the TMA map uses it (or, without a map, for every series but the
  first);
- approximate values are used as numbers and kept as written beside them: "~15" -> 15, ">90" -> 90 (age_topcoded).

`read_donors` gives the column matches, the records and notes in plain words; `select_donors` keeps the donors that
matter for the study; `save_donors` writes the table.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from stainid.setup.naming import Groups, IdStyle, braak, cerad, clean, donor_id, id_prefix, key, tma_number
from stainid.tables import read_csv, write_csv

FIELDS: dict[str, tuple[str, tuple[str, ...]]] = {  # stainID column: (what to call it, header words that mean it)
    "donor_id": ("Donor ID", ("donor id", "donor", "case", "case id", "case no", "case number", "subject", "subject id", "id",
                              "brain id", "brain bank id", "sample id", "autopsy id", "autopsy number", "np number")),
    "disease_group": ("Group", ("group", "diagnostic group", "diagnosis group", "dx group", "cohort", "cohort group", "category", "cohort category")),
    "age": ("Age", ("age", "age at death", "death age", "aad", "age death", "age years", "age yrs")),
    "sex": ("Sex", ("sex", "gender", "m f", "sex m f")),
    "race": ("Race", ("race", "ethnicity", "race ethnicity")),
    "pmd_hours": ("Post-mortem delay (hours)", ("pmd", "pmi", "pmd h", "pmi h", "pmd hrs", "pmi hrs", "pmd hours", "pmi hours", "post mortem delay",
                                                 "post mortem interval", "postmortem delay", "postmortem interval")),
    "apoe": ("APOE genotype", ("apoe", "apoe genotype", "apoe status", "apoe allele", "apoe alleles")),
    "cerad": ("CERAD", ("cerad", "cerad score", "neuritic plaques", "cerad np")),
    "braak": ("Braak stage", ("braak", "braak stage", "braak nft", "braak score", "nft stage")),
    "thal": ("Thal phase", ("thal", "thal phase", "thal stage")),
    "tma": ("TMA", ("tma", "tma no", "tma number", "array", "tma block", "block")),
    "clinical_diagnosis": ("Clinical diagnosis", ("clinical diagnosis", "clinical dx", "clin dx", "clinical")),
    "final_diagnosis": ("Final diagnosis", ("final diagnosis", "final dx", "neuropathology diagnosis", "neuropath dx", "np dx", "path dx",
                                            "pathological diagnosis", "diagnosis", "dx", "primary diagnosis")),
    "source": ("Source", ("source", "brain bank", "bank", "study", "cohort source")),
    "mmse": ("MMSE", ("mmse", "last mmse", "mmse score")),
}
TOKENS = {"age": "age", "sex": "sex", "gender": "sex", "pmi": "pmd_hours", "pmd": "pmd_hours", "apoe": "apoe", "cerad": "cerad",
          "braak": "braak", "thal": "thal", "mmse": "mmse", "race": "race"}
ORDER = ["donor_id", "tma", "disease_group", "age", "age_topcoded", "sex", "race", "pmd_hours", "apoe", "apoe_e4_count", "apoe_e2_count",
         "cerad", "braak", "thal", "mmse", "clinical_diagnosis", "final_diagnosis", "source"]
SKIP = "skip"


def header_words(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9#]+", " ", clean(text).lower())).strip()


def match_column(header: str) -> str:
    """The stainID column a header means ("" when none: the column is kept under its own name)."""
    words = header_words(header)
    if not words:
        return ""
    if re.fullmatch(r"[a-z]{0,6}\s?#|no|number", words):
        return "donor_id"
    for name, (_, aliases) in FIELDS.items():
        if words in aliases:
            return name
    tokens = set(words.replace("#", " ").split())
    return next((TOKENS[t] for t in tokens if t in TOKENS), "")


def own_name(header: str) -> str:
    """A column name for a column that is kept as it is: "Brain weight (g)" -> brain_weight_g."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", clean(header).lower())).strip("_") or "column"


def _score(row: list[str]) -> int:
    return sum(bool(match_column(c)) for c in row if c and not re.fullmatch(r"[\d.,~<>+\- ]+", c))


def series(cell: str) -> str:
    """The id series a sub-table header names: "JOG#" -> JOG, "BRC #" -> BRC ("" for other headers)."""
    match = re.fullmatch(r"([A-Za-z]{2,6})\s*(#|no\.?|id)", clean(cell), re.I)
    return match.group(1).upper() if match else ""


def number(value: str) -> tuple[str, str]:
    """A number written loosely: ("15", "~") for "~15", ("90", ">") for ">90" or "90+", ("", "") for none."""
    text = clean(value).replace(",", ".")
    match = re.fullmatch(r"([~<>≈]|ca\.?|about)?\s*(\d+(?:\.\d+)?)\s*(\+|h|hrs?|hours?|y|yrs?|years?)?", text, re.I)
    if not match:
        return "", ""
    mark = ">" if match.group(1) == ">" or match.group(3) == "+" else "~" if match.group(1) else ""
    return match.group(2).rstrip("0").rstrip(".") if "." in match.group(2) else match.group(2), mark


def apoe(value: str) -> tuple[str, str, str]:
    """APOE genotype as "3/4" with the numbers of e4 and e2 alleles ("", "", "" when not a genotype)."""
    alleles = re.findall(r"[234]", re.sub(r"(?i)apoe|ε|e", "", clean(value)))
    if len(alleles) != 2 or not re.fullmatch(r"(?i)\s*(apoe)?\s*[εe]?[234]\s*[/,;\- ]?\s*[εe]?[234]\s*", clean(value)):
        return "", "", ""
    alleles.sort()
    return "/".join(alleles), str(alleles.count("4")), str(alleles.count("2"))


def sex(value: str) -> str:
    return {"M": "M", "MALE": "M", "MAN": "M", "F": "F", "FEMALE": "F", "WOMAN": "F"}.get(key(value), clean(value))


@dataclass
class DonorSheet:
    header_row: int                    # 0-based row of the column names
    columns: list[dict]                # index, header, field (stainID column, "" = kept as is, "skip"), samples
    records: list[dict[str, str]]
    notes: list[str] = field(default_factory=list)


def find_header(rows: list[list[str]]) -> int:
    scores = [_score(row) for row in rows[:60]]
    best = max(range(len(scores)), key=lambda i: (scores[i], -i)) if scores else 0
    if not scores or scores[best] < 2:
        raise ValueError("The column names (Donor, Age, Sex, …) were not found in the first 60 rows of this sheet")
    return best


def read_donors(rows: list[list[str]], groups: Groups, style: IdStyle = IdStyle(), mapping: dict[int, str] | None = None,
                map_prefixes: set[str] | None = None) -> DonorSheet:
    """Read a case sheet. `mapping` (column index -> stainID column, "" to keep the column under its own name, "skip")
    overrides the automatic column matches; `map_prefixes` are the id prefixes the TMA map uses (e.g. {"JOG"})."""
    header = find_header(rows)
    names = rows[header]
    columns = []
    for index, name in enumerate(names):
        chosen = (mapping or {}).get(index, match_column(name) if name else SKIP)
        samples = [clean(r[index]) for r in rows[header + 1:] if index < len(r) and clean(r[index]) and sum(bool(clean(c)) for c in r) > 1
                   and _score(r) < 2][:4]  # data rows only, not section headings or repeated headers
        columns.append({"index": index, "header": clean(name), "field": chosen, "samples": samples})
    taken: set[str] = set()
    for column in columns:  # a stainID column is filled from the first sheet column matched to it
        if column["field"] not in ("", SKIP) and column["field"] in taken:
            column["field"] = ""
        taken.add(column["field"])
    id_column = next((c["index"] for c in columns if c["field"] == "donor_id"), None)
    if id_column is None:
        raise ValueError("No column is matched to the donor ID: choose it in the column list")
    kinds = [series(names[id_column])] if series(names[id_column]) else []
    kind, section, sections, skipped = (kinds[0] if kinds else ""), "", {}, 0
    records = []
    for number_, row in enumerate(rows[header + 1:], start=header + 2):
        row = (row + [""] * len(names))[:len(names)]
        filled = [c for c in row if clean(c)]
        if not filled:
            continue
        if _score(row) >= 2:  # the header again: a sub-table, maybe with its own id series
            if series(row[id_column]):
                kind = series(row[id_column])
                kinds += [kind] if kind not in kinds else []
            continue
        ident = clean(row[id_column])
        is_id = bool(re.fullmatch(r"[A-Z]*\d+", key(ident)))
        if len(filled) == 1 and not is_id:  # a section heading: the group of the rows below
            section = clean(filled[0])
            sections.setdefault(section, 0)
            continue
        if not is_id:
            skipped += 1
            continue
        records.append(_record(row, columns, ident, kind, kinds, section, groups, style, map_prefixes, number_))
        if section:
            sections[section] += 1
    notes = [f"Column names found on row {header + 1}."]
    if sections:
        notes.append("Sections: " + "; ".join(f"“{s}” → {groups.section(s) or 'no group'} ({n} donors)" for s, n in sections.items()) + ".")
    if len(kinds) > 1:
        notes.append("Sub-tables with their own id series: " + ", ".join(
            f"{k} ({'ids written ' + k + '_001 …' if _prefix(k, kinds, map_prefixes) else 'ids as written'})" for k in kinds) + ".")
    loose = [(r["donor_id"], r[f"{c}_raw"], r[c]) for r in records for c in ("age", "pmd_hours") if r.get(f"{c}_raw") and r.get(c)]
    if loose:
        notes.append(f"{len(loose)} approximate or capped values were used as numbers (e.g. donor {loose[0][0]}: {loose[0][1]} → {loose[0][2]}); "
                     "the values as written are kept in age_raw and pmd_hours_raw.")
    unread = sum(bool(r.get(f"{c}_raw")) and not r.get(c) for r in records for c in ("age", "pmd_hours"))
    if unread:
        notes.append(f"{unread} ages or post-mortem delays are not numbers (such as NA) and were left empty.")
    if skipped:
        notes.append(f"{skipped} rows without a donor number were skipped.")
    return DonorSheet(header, columns, records, notes)


def _prefix(kind: str, kinds: list[str], map_prefixes: set[str] | None) -> str:
    if not kind:
        return ""
    if map_prefixes is not None and (map_prefixes or len(kinds) < 2):
        return kind if kind in map_prefixes else ""
    return "" if kind == kinds[0] else kind


def _record(row, columns, ident, kind, kinds, section, groups, style, map_prefixes, row_number) -> dict[str, str]:
    record: dict[str, str] = {"donor_id": donor_id(ident, style, _prefix(kind, kinds, map_prefixes))}
    for column in columns:
        name, value = column["field"], clean(row[column["index"]])
        if name == SKIP or name == "donor_id":
            continue
        if name == "":
            record[own_name(column["header"])] = value
        elif name in ("age", "pmd_hours"):
            amount, mark = number(value)
            record[name] = amount
            if value and (mark or not amount) and key(value) not in ("NA", "NONE", "UNKNOWN", ""):
                record[f"{name}_raw"] = value
            if name == "age":
                record["age_topcoded"] = "true" if mark == ">" else ""
        elif name == "apoe":
            genotype, e4, e2 = apoe(value)
            record.update(apoe=genotype or value, apoe_e4_count=e4, apoe_e2_count=e2)
        elif name == "cerad":
            record["cerad"] = cerad(value) or value
        elif name == "braak":
            record["braak"] = braak(value) or value
        elif name == "sex":
            record["sex"] = sex(value)
        elif name == "tma":
            record["tma"] = tma_number(value)
            if value != record["tma"]:
                record["sheet_tma"] = value
        elif name == "disease_group":
            record["disease_group"] = groups.code(value) or groups.section(value)
            record["group_raw"] = value
        else:
            record[name] = value
    if not record.get("disease_group") and section:
        record["disease_group"] = groups.section(section)
        record["group_raw"] = section
    record["sheet_row"] = str(row_number)
    return record


def columns_of(records: list[dict[str, str]]) -> list[str]:
    seen = list(dict.fromkeys(k for r in records for k in r))
    return [c for c in ORDER if c in seen] + [c for c in seen if c not in ORDER]


def select_donors(records: list[dict[str, str]], map_donors: set[str], study_tmas: set[str], keep: str = "study") -> list[dict[str, str]]:
    """keep "study": donors on the TMA maps, and donors the sheet puts on one of the study's TMAs (so a donor missing
    from a map is noticed); "all": every donor of the sheet."""
    if keep == "all" or not (map_donors or study_tmas):
        return list(records)
    return [r for r in records if r["donor_id"] in map_donors or (r.get("tma") and r["tma"] in study_tmas)]


def read_donor_table(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if Path(path).exists() else []


def save_donors(path: Path, records: list[dict[str, str]], mode: str = "update") -> int:
    """mode "update": the sheet's donors replace the same donors in the table and others are kept; "replace": the
    table becomes the sheet's donors. Returns the number of donors in the table."""
    if not records:
        raise ValueError("No donors to save")
    existing = read_donor_table(path) if mode == "update" else []
    new_ids = {r["donor_id"] for r in records}
    rows = [r for r in existing if r["donor_id"] not in new_ids] + list(records)
    write_csv(Path(path), rows, columns_of(rows))
    return len(rows)


def prefixes(donors: set[str]) -> set[str]:
    return {id_prefix(d) for d in donors} - {""}
