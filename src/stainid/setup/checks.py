"""Cross-checks between the TMA map, the donor table and the slides, in plain words, each with a fix where one is clear.

`find_issues` works on plain rows, so it is easy to test; `project_issues` reads them from a project and `apply_fix`
changes the project's files. An issue is a dict:

    {"id", "level": "problem" | "warning" | "note", "title", "detail", "items": [{"text", "fixes": [fix, ...]}],
     "fixes": [fix, ...]}     with fix = {"label", "action", "args"}

"problem" stops a later step (Attach TMA map would fail), "warning" probably needs a look, "note" is for information.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

from stainid import cache
from stainid.project import Project, save_project
from stainid.setup.naming import braak, cerad
from stainid.tables import read_csv, write_csv

SHOW = 12  # items listed per issue; the fixes apply to all


def one(text: str) -> str:
    """Counts of one read as one: "1 donors on the map are" -> "1 donor on the map is"."""
    text = re.sub(r"\b1 donors\b", "1 donor", text).replace("(1 rows ", "(1 row ")
    if not text.startswith("1 "):
        return text
    if not text.startswith("1 donor ") or text.startswith("1 donor ids"):
        text = re.sub(r"^1 ((?:\w+[ -])*?)(\w+?)s\b", r"1 \1\2", text, count=1)
    for many, single in ((" are ", " is "), (" have ", " has "), (" were ", " was ")):
        text = text.replace(many, single, 1)
    return text


def _fix(label: str, action: str, **args) -> dict:
    return {"label": one(label), "action": action, "args": args}


def _issue(id_: str, level: str, title: str, detail: str = "", items: list[dict] | None = None, fixes: list[dict] | None = None) -> dict:
    return {"id": id_, "level": level, "title": one(title), "detail": detail, "items": (items or [])[:SHOW], "more": max(0, len(items or []) - SHOW),
            "fixes": fixes or []}


def _names(values, limit: int = 8) -> str:
    values = sorted(values, key=lambda v: (len(str(v)), str(v)))
    return ", ".join(map(str, values[:limit])) + (f" and {len(values) - limit} more" if len(values) > limit else "")


def _tmas(values) -> str:
    return _names(values).replace(", ", ", TMA ") if values else ""


def find_issues(layout: list[dict], donors: list[dict], positions: set[tuple[str, str]], slide_tmas: set[str],
                group_codes: list[str]) -> list[dict]:
    """Everything that does not fit together, most important first."""
    issues = []
    on_map = [r for r in layout if r.get("donor_id")]
    map_tmas = {r["tma"] for r in layout}
    map_tma = defaultdict(set)
    map_group = defaultdict(set)
    for r in on_map:
        map_tma[r["donor_id"]].add(r["tma"])
        if r.get("disease_group"):
            map_group[r["donor_id"]].add(r["disease_group"])

    # slides and map
    no_map = sorted(slide_tmas - map_tmas, key=lambda t: (len(t), t))
    if no_map:
        issues.append(_issue("tma_without_map", "problem", f"TMA {_tmas(no_map)} {'has' if len(no_map) == 1 else 'have'} slides but no map yet",
                             "Add the map on the Map screen: which donor sits on each core.", fixes=[_fix("Go to the Map screen", "goto", screen="map")]))
    unmapped = sorted((t, p) for t, p in positions - {(r["tma"], r["core_label"]) for r in layout} if t in map_tmas)
    if unmapped:
        by_tma = Counter(t for t, _ in unmapped)
        issues.append(_issue("positions_not_on_map", "problem", f"{len(unmapped)} core positions found on the slides are not on the map",
                             "Attach TMA map needs an entry for every position. If these positions hold no donor (empty, or extra rows of the "
                             "grid), mark them as positions without a donor; if they do, add them on the Map screen.",
                             [{"text": f"TMA {t}: {_names([p for tt, p in unmapped if tt == t])}", "fixes": []} for t in sorted(by_tma, key=lambda t: (len(t), t))],
                             [_fix("Mark them as positions without a donor", "add_empty_positions", positions=[list(p) for p in unmapped]),
                              _fix("Go to the Map screen", "goto", screen="map")]))
    off_slide = sorted((r["tma"], r["core_label"]) for r in on_map if r["tma"] in {t for t, _ in positions}
                       and (r["tma"], r["core_label"]) not in positions)
    if off_slide:
        issues.append(_issue("map_positions_not_on_slides", "warning", f"{len(off_slide)} donor positions on the map were not found on the slides",
                             "The map has more rows or columns than the grid fitted on the slide. Check the grid size on the Cores screen.",
                             [{"text": f"TMA {t} {p}", "fixes": []} for t, p in off_slide], [_fix("Go to the Cores screen", "goto", screen="cores")]))
    two_tmas = {d: t for d, t in map_tma.items() if len(t) > 1}
    if two_tmas:
        issues.append(_issue("donor_on_two_tmas", "warning", f"{len(two_tmas)} donors are on more than one TMA",
                             "Each donor usually sits on one TMA (model checks leave one TMA out at a time). Check the ids on the maps.",
                             [{"text": f"Donor {d}: TMA {_tmas(t)}", "fixes": []} for d, t in sorted(two_tmas.items())]))

    # groups
    missing_group = sorted({r["donor_id"] for r in on_map if not r.get("disease_group")})
    sheet_group = {d["donor_id"]: d.get("disease_group", "") for d in donors}
    if missing_group:
        fillable = [d for d in missing_group if sheet_group.get(d)]
        issues.append(_issue("map_without_group", "warning" if fillable or not donors else "problem",
                             f"{len(missing_group)} donors on the map have no group",
                             "The group (e.g. Control or AD) comes from the map or from the donor table.",
                             [{"text": f"Donor {d}", "fixes": []} for d in missing_group],
                             [_fix(f"Take the group from the donor table ({len(fillable)} donors)", "groups_from_sheet", donors=fillable)] if fillable else []))
    used = {g for groups in map_group.values() for g in groups} | {g for g in sheet_group.values() if g}
    unknown = sorted(used - set(group_codes))
    if unknown:
        issues.append(_issue("unknown_groups", "note", f"Group {_names(unknown)} {'is' if len(unknown) == 1 else 'are'} not in Settings yet",
                             "Add them to give each group a name, a colour and a place in the plots.",
                             fixes=[_fix("Add to Settings", "add_groups", codes=unknown)]))

    # replicates
    cores = Counter((r["tma"], r["donor_id"], r.get("region", "")) for r in on_map)
    if cores and sum(n > 1 for n in cores.values()) > len(cores) / 2:
        lonely = sorted(k for k, n in cores.items() if n == 1)
        if lonely:
            issues.append(_issue("single_core", "note", f"{len(lonely)} donor-regions have only one core",
                                 "Most donor-regions of this study have two or more cores (replicates). These have one, so their numbers "
                                 "rest on one core; check the map if a replicate is expected.",
                                 [{"text": f"Donor {d}, {region or 'no region'} (TMA {t})", "fixes": []} for t, d, region in lonely]))

    if not donors:
        return issues

    # map and donor table
    sheet = defaultdict(list)
    for d in donors:
        sheet[d["donor_id"]].append(d)
    not_in_sheet = sorted(set(map_tma) - set(sheet))
    if not_in_sheet:
        issues.append(_issue("map_donor_not_in_sheet", "warning", f"{len(not_in_sheet)} donors on the map are not in the donor table",
                             "Check the ids for typing slips, or add them to the donor table with the TMA and group from the map.",
                             [{"text": f"Donor {d} (TMA {_tmas(map_tma[d])})", "fixes": []} for d in not_in_sheet],
                             [_fix("Add them to the donor table", "add_donors", donors=not_in_sheet)]))
    not_on_map = sorted({d["donor_id"] for d in donors if d.get("tma") in map_tmas and d["donor_id"] not in map_tma})
    if not_on_map:
        issues.append(_issue("sheet_donor_not_on_map", "warning", f"{len(not_on_map)} donors the donor table puts on a TMA of this study are not on its map",
                             "Maybe a typing slip in the map or in the sheet.",
                             [{"text": f"Donor {d}: TMA {_tmas({r.get('tma') for r in sheet[d]})} in the donor table", "fixes": []} for d in not_on_map],
                             [_fix("Remove them from the donor table", "remove_donors", donors=not_on_map)]))
    duplicates = {d: rows for d, rows in sheet.items() if len(rows) > 1}
    if duplicates:
        items, decidable = [], []
        for d, rows in sorted(duplicates.items()):
            match = [r for r in rows if r.get("tma") and r["tma"] in map_tma.get(d, set())]
            text = f"Donor {d}: {len(rows)} rows" + (f" (sheet rows {', '.join(r.get('sheet_row', '?') for r in rows)})" if rows[0].get("sheet_row") else "")
            if len(match) == 1:
                decidable.append(d)
                items.append({"text": text + f"; the map puts it on TMA {match[0]['tma']}", "fixes": []})
            else:
                items.append({"text": text + "; the map does not say which is right", "fixes": []})
        issues.append(_issue("duplicate_donors", "problem" if len(decidable) < len(duplicates) else "warning",
                             f"{len(duplicates)} donor ids are used more than once in the donor table",
                             "Different sub-tables can reuse a number. Keep the row on the TMA the map puts the donor on.", items,
                             [_fix(f"Keep the row on the map's TMA ({len(decidable)} donors)", "keep_map_tma_rows", donors=decidable)] if decidable else []))
    single = {d: rows[0] for d, rows in sheet.items() if len(rows) == 1}
    tma_differs = sorted(d for d, r in single.items() if d in map_tma and r.get("tma") and r["tma"] not in map_tma[d])
    if tma_differs:
        issues.append(_issue("tma_disagrees", "warning", f"{len(tma_differs)} donors are on a different TMA in the donor table than on the map",
                             "The map shows where the cores are, so it is usually right.",
                             [{"text": f"Donor {d}: map TMA {_tmas(map_tma[d])}, donor table TMA {single[d]['tma']}", "fixes": []} for d in tma_differs],
                             [_fix("Use the map's TMA in the donor table", "tma_from_map", donors=tma_differs)]))
    group_differs = sorted(d for d, r in single.items() if map_group.get(d) and r.get("disease_group") and {r["disease_group"]} != map_group[d])
    if group_differs:
        issues.append(_issue("group_disagrees", "warning", f"{len(group_differs)} donors have a different group on the map than in the donor table",
                             "One of the two has a typing slip.",
                             [{"text": f"Donor {d}: map says {_names(map_group[d])}, donor table says {single[d]['disease_group']}",
                               "fixes": [_fix("Use the donor table", "groups_from_sheet", donors=[d]), _fix("Use the map", "groups_from_map", donors=[d])]}
                              for d in group_differs],
                             [_fix("Use the donor table for all", "groups_from_sheet", donors=group_differs),
                              _fix("Use the map for all", "groups_from_map", donors=group_differs)]))
    pathology = []
    for r in on_map:
        d = single.get(r["donor_id"])
        if not d:
            continue
        for score, read, name in (("cerad", cerad, "CERAD"), ("braak", braak, "Braak")):
            mine, theirs = read(r.get(score, "")), read(d.get(score, ""))
            if mine and theirs and mine != theirs:
                pathology.append((r["donor_id"], name, r.get(score, ""), d.get(score, "")))
    pathology = sorted(set(pathology))
    if pathology:
        donors_ = sorted({p[0] for p in pathology})
        issues.append(_issue("pathology_disagrees", "warning", f"{len(donors_)} donors have different pathology on the map than in the donor table",
                             "The donor table usually holds the final neuropathology.",
                             [{"text": f"Donor {d}: map says {name} {mine}, donor table says {theirs}. Use the donor table?",
                               "fixes": [_fix("Use the donor table", "pathology_from_sheet", donors=[d]), _fix("Keep the map", "pathology_from_map", donors=[d])]}
                              for d, name, mine, theirs in pathology],
                             [_fix("Use the donor table for all", "pathology_from_sheet", donors=donors_)]))

    return issues


def project_issues(project: Project) -> list[dict]:
    """Everything that does not fit together yet (read from shared, read-only rows: the setup overview polls this)."""
    layout = cache.rows(project.input("tma_layout"))
    donors = cache.rows(project.input("donor_metadata"))
    manifest = cache.rows(project.input("core_manifest"), ("tma", "core_label"))
    positions = {(r["tma"], r["core_label"]) for r in manifest}
    slide_tmas = {s["tma"] for s in cache.rows(project.input("slides_table")) if s.get("tma")}
    return find_issues(layout, donors, positions, slide_tmas, [g["code"] for g in project.config.get("groups") or []])


def _rows(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if Path(path).exists() else []


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    columns = list(dict.fromkeys(k for r in rows for k in r))
    write_csv(Path(path), rows, columns or ["donor_id"])


def apply_fix(project: Project, action: str, args: dict) -> str:
    """Carry out one fix; returns what was done, in plain words."""
    return one(_apply(project, action, args))


def _apply(project: Project, action: str, args: dict) -> str:
    layout_path, donors_path = project.input("tma_layout"), project.input("donor_metadata")
    layout, donors = _rows(layout_path), _rows(donors_path)
    chosen = set(args.get("donors") or [])
    on_map = {}
    for r in layout:
        if r.get("donor_id"):
            on_map.setdefault(r["donor_id"], r)
    sheet = {d["donor_id"]: d for d in donors}
    if action == "add_empty_positions":
        added = [{"tma": t, "core_label": p, "donor_id": "", "region": "", "disease_group": ""} for t, p in args["positions"]]
        from stainid.setup.maps import save_map

        for tma in sorted({r["tma"] for r in added}):
            save_map(layout_path, tma, [r for r in layout if r["tma"] == tma] + [r for r in added if r["tma"] == tma])
        return f"{len(added)} positions marked as positions without a donor"
    if action == "groups_from_sheet":
        for r in layout:
            if r.get("donor_id") in chosen and sheet.get(r["donor_id"], {}).get("disease_group"):
                r["disease_group"] = sheet[r["donor_id"]]["disease_group"]
        _write(layout_path, layout)
        return f"Group taken from the donor table for {len(chosen)} donors"
    if action == "groups_from_map":
        for d in donors:
            if d["donor_id"] in chosen:
                d["disease_group"] = on_map[d["donor_id"]]["disease_group"]
        _write(donors_path, donors)
        return f"Group taken from the map for {len(chosen)} donors"
    if action in ("pathology_from_sheet", "pathology_from_map"):
        for r in layout:
            if r.get("donor_id") in chosen and r["donor_id"] in sheet:
                d = sheet[r["donor_id"]]
                for score in ("cerad", "braak"):
                    if action == "pathology_from_sheet" and d.get(score):
                        r[score] = d[score]
                    elif action == "pathology_from_map" and r.get(score):
                        d[score] = r[score]
        _write(layout_path if action == "pathology_from_sheet" else donors_path, layout if action == "pathology_from_sheet" else donors)
        return f"Pathology taken from the {'donor table' if action == 'pathology_from_sheet' else 'map'} for {len(chosen)} donors"
    if action == "add_donors":
        new = [{"donor_id": d, "tma": on_map[d]["tma"], "disease_group": on_map[d].get("disease_group", "")} for d in sorted(chosen) if d in on_map]
        _write(donors_path, donors + new)
        return f"{len(new)} donors added to the donor table"
    if action == "remove_donors":
        _write(donors_path, [d for d in donors if d["donor_id"] not in chosen])
        return f"{len(chosen)} donors removed from the donor table"
    if action == "keep_map_tma_rows":
        tmas = defaultdict(set)
        for r in layout:
            if r.get("donor_id"):
                tmas[r["donor_id"]].add(r["tma"])
        kept = [d for d in donors if d["donor_id"] not in chosen or d.get("tma") in tmas[d["donor_id"]]]
        _write(donors_path, kept)
        return f"Kept the row on the map's TMA for {len(chosen)} donors ({len(donors) - len(kept)} rows removed)"
    if action == "tma_from_map":
        for d in donors:
            if d["donor_id"] in chosen and d["donor_id"] in on_map:
                d["tma"] = on_map[d["donor_id"]]["tma"]
        _write(donors_path, donors)
        return f"TMA taken from the map for {len(chosen)} donors"
    if action == "add_groups":
        groups = list(project.config.get("groups") or [])
        codes = [c for c in args.get("codes", []) if c not in {g["code"] for g in groups}]
        save_project(project, {"groups": groups + [{"code": c, "label": c} for c in codes]})
        return f"Added group {', '.join(codes)} to Settings"
    raise ValueError(f"Unknown fix: {action}")
