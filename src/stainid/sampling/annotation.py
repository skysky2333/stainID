from __future__ import annotations

import csv
import hashlib
import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STAINS = ("6E10", "AT8", "NeuN")
USABLE_STATUSES = {"partial", "substantial"}


@dataclass(frozen=True)
class CoreTriplet:
    tma: str
    core_label: str
    rows: dict[str, dict[str, str]]
    signal_ranks: dict[str, float]

    @property
    def reference(self) -> dict[str, str]:
        return self.rows[STAINS[0]]

    @property
    def usable(self) -> bool:
        return all(self.rows[stain]["tissue_status"] in USABLE_STATUSES for stain in STAINS)

    @property
    def minimum_tissue_fraction(self) -> float:
        return min(float(self.rows[stain]["tissue_fraction"]) for stain in STAINS)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "tma",
        "core_label",
        "stain",
        "core_role",
        "disease_group",
        "region",
        "tissue_status",
        "tissue_fraction",
        "dab_od_p90",
        "output_path",
        "output_width_px",
        "output_height_px",
        "source_pixel_width_um",
        "source_pixel_height_um",
    }
    missing = sorted(required - set(rows[0] if rows else ()))
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(missing)}")
    return rows


def percentile_ranks(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], float]:
    distributions: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row["core_role"] == "biological"
            and row["tissue_status"] in USABLE_STATUSES
            and row["dab_od_p90"]
        ):
            distributions[row["slide"]].append(float(row["dab_od_p90"]))
    for values in distributions.values():
        values.sort()

    ranks = {}
    for row in rows:
        if row["slide"] not in distributions or not row["dab_od_p90"]:
            continue
        values = distributions[row["slide"]]
        value = float(row["dab_od_p90"])
        midpoint = (bisect_left(values, value) + bisect_right(values, value) - 1) / 2
        ranks[(row["tma"], row["core_label"], row["stain"])] = (
            midpoint / (len(values) - 1) if len(values) > 1 else 0.5
        )
    return ranks


def build_triplets(rows: list[dict[str, str]]) -> list[CoreTriplet]:
    ranks = percentile_ranks(rows)
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["core_role"] != "biological":
            continue
        key = (row["tma"], row["core_label"])
        if row["stain"] in grouped[key]:
            raise ValueError(f"Duplicate {row['stain']} image for LIP-{key[0]} {key[1]}")
        grouped[key][row["stain"]] = row

    triplets = []
    for (tma, core_label), stain_rows in sorted(
        grouped.items(), key=lambda item: (int(item[0][0]), item[0][1])
    ):
        if set(stain_rows) != set(STAINS):
            raise ValueError(f"Incomplete stain triplet for LIP-{tma} {core_label}")
        identity_fields = ("donor_id", "region", "disease_group", "sample_region_id")
        for field in identity_fields:
            if len({stain_rows[stain][field] for stain in STAINS}) != 1:
                raise ValueError(f"Stain metadata disagree for LIP-{tma} {core_label}: {field}")
        triplets.append(
            CoreTriplet(
                tma=tma,
                core_label=core_label,
                rows=stain_rows,
                signal_ranks={
                    stain: ranks.get((tma, core_label, stain), float("nan"))
                    for stain in STAINS
                },
            )
        )
    return triplets


def target_ranks(tma: str, disease_group: str, region: str) -> dict[str, float]:
    levels = (0.15, 0.50, 0.85)
    group_index = {"AD": 0, "ASYMP": 1, "CT": 2}[disease_group]
    region_index = {"frontal": 0, "occipital": 1}[region]
    tma_index = int(tma) - 1
    return {
        "6E10": levels[(tma_index + group_index + region_index) % 3],
        "AT8": levels[(tma_index + 2 * group_index + region_index) % 3],
        "NeuN": levels[(tma_index + group_index + 2 * region_index) % 3],
    }


def selection_distance(triplet: CoreTriplet, targets: dict[str, float]) -> float:
    distances = [abs(triplet.signal_ranks[stain] - targets[stain]) for stain in STAINS]
    return float(np.mean(distances))


def select_balanced_triplets(
    triplets: list[CoreTriplet],
) -> tuple[list[tuple[CoreTriplet, dict[str, float], float]], list[CoreTriplet]]:
    strata: dict[tuple[str, str, str], list[CoreTriplet]] = defaultdict(list)
    challenges = []
    for triplet in triplets:
        if not triplet.usable:
            challenges.append(triplet)
            continue
        reference = triplet.reference
        strata[(triplet.tma, reference["disease_group"], reference["region"])].append(
            triplet
        )

    selected = []
    for stratum in sorted(strata, key=lambda value: (int(value[0]), value[1], value[2])):
        targets = target_ranks(*stratum)
        candidates = strata[stratum]
        if not candidates:
            raise ValueError(f"No usable candidates for stratum {stratum}")
        chosen = min(
            candidates,
            key=lambda triplet: (
                selection_distance(triplet, targets),
                -triplet.minimum_tissue_fraction,
                triplet.core_label,
            ),
        )
        selected.append((chosen, targets, selection_distance(chosen, targets)))
    return selected, challenges


def blinded_order(triplets: list[CoreTriplet], seed: int) -> list[CoreTriplet]:
    return sorted(
        triplets,
        key=lambda triplet: hashlib.sha256(
            f"{seed}:{triplet.tma}:{triplet.core_label}".encode()
        ).hexdigest(),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def image_rows(
    triplets: list[CoreTriplet], prefix: str, seed: int
) -> tuple[list[dict[str, object]], dict[tuple[str, str], str]]:
    identifiers = {
        (triplet.tma, triplet.core_label): f"{prefix}{index:03d}"
        for index, triplet in enumerate(blinded_order(triplets, seed), start=1)
    }
    rows = []
    for triplet in blinded_order(triplets, seed):
        identifier = identifiers[(triplet.tma, triplet.core_label)]
        for stain in STAINS:
            row = triplet.rows[stain]
            rows.append(
                {
                    "annotation_id": identifier,
                    "image_id": f"{identifier}_{stain}",
                    "stain": stain,
                    "image_path": str(Path("data") / row["output_path"]),
                    "slide_path": row.get("slide_path", ""),
                    "full_scene": row.get("full_scene", ""),
                    "center_x_px": row.get("center_x_px", ""),
                    "center_y_px": row.get("center_y_px", ""),
                    "diameter_x_px": row.get("diameter_x_px", ""),
                    "diameter_y_px": row.get("diameter_y_px", ""),
                    "native_width_px": row["output_width_px"],
                    "native_height_px": row["output_height_px"],
                    "pixel_width_um": row["source_pixel_width_um"],
                    "pixel_height_um": row["source_pixel_height_um"],
                    "tissue_status": row["tissue_status"],
                    "tissue_fraction": row["tissue_fraction"],
                }
            )
    return rows, identifiers


def selection_key_rows(
    selected: list[tuple[CoreTriplet, dict[str, float], float]],
    identifiers: dict[tuple[str, str], str],
) -> list[dict[str, object]]:
    rows = []
    for triplet, targets, distance in selected:
        reference = triplet.reference
        rows.append(
            {
                "annotation_id": identifiers[(triplet.tma, triplet.core_label)],
                "tma": triplet.tma,
                "core_label": triplet.core_label,
                "donor_id": reference["donor_id"],
                "sample_region_id": reference["sample_region_id"],
                "disease_group": reference["disease_group"],
                "region": reference["region"],
                "cerad": reference["cerad"],
                "braak": reference["braak"],
                "technical_replicate": reference["technical_replicate"],
                "minimum_tissue_fraction": f"{triplet.minimum_tissue_fraction:.6f}",
                "selection_distance": f"{distance:.6f}",
                **{
                    f"{stain}_signal_rank": f"{triplet.signal_ranks[stain]:.6f}"
                    for stain in STAINS
                },
                **{f"{stain}_target_rank": f"{targets[stain]:.2f}" for stain in STAINS},
                **{
                    f"{stain}_tissue_status": triplet.rows[stain]["tissue_status"]
                    for stain in STAINS
                },
            }
        )
    return sorted(rows, key=lambda row: str(row["annotation_id"]))


def challenge_key_rows(
    triplets: list[CoreTriplet], identifiers: dict[tuple[str, str], str]
) -> list[dict[str, object]]:
    rows = []
    for triplet in triplets:
        reference = triplet.reference
        rows.append(
            {
                "annotation_id": identifiers[(triplet.tma, triplet.core_label)],
                "tma": triplet.tma,
                "core_label": triplet.core_label,
                "donor_id": reference["donor_id"],
                "sample_region_id": reference["sample_region_id"],
                "disease_group": reference["disease_group"],
                "region": reference["region"],
                **{
                    f"{stain}_tissue_status": triplet.rows[stain]["tissue_status"]
                    for stain in STAINS
                },
                **{
                    f"{stain}_tissue_fraction": triplet.rows[stain]["tissue_fraction"]
                    for stain in STAINS
                },
            }
        )
    return sorted(rows, key=lambda row: str(row["annotation_id"]))


def create_annotation_manifests(
    manifest_path: Path, output_dir: Path, seed: int = 20260923
) -> dict[str, Path]:
    triplets = build_triplets(read_manifest(manifest_path))
    selected, challenges = select_balanced_triplets(triplets)
    selected_triplets = [triplet for triplet, _, _ in selected]
    annotation_rows, annotation_ids = image_rows(selected_triplets, "M", seed)
    challenge_rows, challenge_ids = image_rows(challenges, "Q", seed + 1)

    paths = {
        "annotation_manifest": output_dir / "annotation_manifest.csv",
        "qupath_manifest": output_dir / "qupath_manifest.json",
        "selection_key": output_dir / "selection_key.csv",
        "qc_challenge_manifest": output_dir / "qc_challenge_manifest.csv",
        "qc_challenge_key": output_dir / "qc_challenge_key.csv",
    }
    write_csv(paths["annotation_manifest"], annotation_rows)
    write_json(paths["qupath_manifest"], annotation_rows)
    write_csv(paths["selection_key"], selection_key_rows(selected, annotation_ids))
    write_csv(paths["qc_challenge_manifest"], challenge_rows)
    write_csv(paths["qc_challenge_key"], challenge_key_rows(challenges, challenge_ids))
    return paths


__all__ = [
    "CoreTriplet",
    "build_triplets",
    "create_annotation_manifests",
    "read_manifest",
    "select_balanced_triplets",
    "target_ranks",
]
