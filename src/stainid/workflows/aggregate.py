"""Aggregate field/object outputs into core- and donor-region feature tables."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from stainid.analysis import field_aggregation, mask_aggregation
from stainid.project import Project
from stainid.tables import read_csv, write_records

REGION_KEYS = ("core_id", "sample_region_id", "donor_id", "tma", "region", "disease_group")


def aggregate_fields(project: Project, level: str = "sample_region_id", output: Path | None = None) -> Path:
    parts = project.output("fields") / "parts"
    tiles, objects = [], []
    for path in sorted(parts.glob("*_features.csv")):
        tiles += read_csv(path)
        objects += [r for r in read_csv(path.with_name(path.name.replace("_features", "_objects"))) if r.get("stain")]
    tile_groups, object_groups = defaultdict(list), defaultdict(list)
    for row in tiles:
        tile_groups[row[level]].append(row)
    for row in objects:
        object_groups[(row[level], row["stain"])].append(row)
    rows = []
    for key, group in sorted(tile_groups.items()):
        out = {k: group[0][k] for k in REGION_KEYS}
        for stain in ("6E10", "AT8"):
            stain_tiles = [row for row in group if row["stain"] == stain]
            if stain_tiles:
                out |= field_aggregation.summarize(stain, stain_tiles, object_groups[(key, stain)])
        rows.append(out)
    output = output or project.output("tables") / f"fields_{'core' if level == 'core_id' else 'donor_region'}.csv"
    write_records(output, rows)
    print(f"{len(rows)} {level} rows -> {output}", flush=True)
    return output


def aggregate_masks(project: Project, level: str = "sample_region_id", output: Path | None = None) -> Path:
    masks, neun_tiles = project.output("masks"), project.output("neun") / "tiles"
    tissue: dict[tuple[str, str], float] = defaultdict(float)
    for path in (project.output("fields") / "parts").glob("*_features.csv"):
        for row in read_csv(path):
            tissue[(row["tile_id"], row["stain"])] = float(row["v2_tissue_area_mm2"])
    manifest = read_csv(project.input("tile_manifest"))
    region_of: dict[str, dict] = {}
    out: dict[str, dict[str, object]] = defaultdict(dict)
    for stain in ("NeuN", "AT8", "6E10"):
        parts = sorted((masks / stain / "parts").glob("*.csv"))
        if not parts:
            continue
        objects = [r for p in parts for r in read_csv(p) if r.get("mask_status") == "ok"]
        processed = {p.name.rsplit(f"_{stain}_objects", 1)[0] for p in parts}
        tiles = {r["tile_id"]: r for r in manifest if r["stain"] == stain and r["core_id"] in processed}
        if stain == "NeuN":
            tiles = {k: v for k, v in tiles.items() if (neun_tiles / f"{k}_features.csv").exists()}
            for tile_id in tiles:
                tissue[(tile_id, stain)] = float(read_csv(neun_tiles / f"{tile_id}_features.csv")[0]["tissue_area_mm2"])
        by_key, key_tiles = defaultdict(list), defaultdict(set)
        for row in objects:
            by_key[row[level]].append(row)
        for row in tiles.values():
            key_tiles[row[level]].add(row["tile_id"])
            region_of[row[level]] = row
        for key, tile_ids in key_tiles.items():
            out[key] |= mask_aggregation.summarize(stain, by_key.get(key, []), sum(tissue.get((t, stain), 0.0) for t in tile_ids))
    rows = [{k: region_of[key][k] for k in dict.fromkeys((level, "sample_region_id", "donor_id", "tma", "region", "disease_group"))} | features
            for key, features in sorted(out.items())]
    output = output or project.output("tables") / f"masks_{'core' if level == 'core_id' else 'donor_region'}.csv"
    write_records(output, rows)
    print(f"{len(rows)} {level} rows -> {output}", flush=True)
    return output
