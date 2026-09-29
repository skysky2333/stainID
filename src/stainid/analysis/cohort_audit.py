from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

GROUPS = ("CT", "ASYMP", "AD")
USABLE_STATUSES = {"partial", "substantial"}
COPATHOLOGY_FIELDS = (
    "diagnosis_mentions_caa",
    "diagnosis_mentions_cerebrovascular",
    "diagnosis_mentions_lbd",
    "diagnosis_mentions_late",
    "diagnosis_mentions_infarct",
    "diagnosis_mentions_hemorrhage",
    "diagnosis_mentions_artag",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def distribution(values: list[str]) -> str:
    counts = Counter(value for value in values if value)
    return ";".join(f"{value}:{counts[value]}" for value in sorted(counts))


def median(values: list[str]) -> str:
    numeric = [float(value) for value in values if value]
    return f"{statistics.median(numeric):.1f}" if numeric else ""


def build_stage_overlap(donors: list[dict[str, str]]) -> list[dict[str, object]]:
    target = [row for row in donors if row["disease_group"] in {"ASYMP", "AD"}]
    stages = sorted(
        {(row["cerad"], row["braak"]) for row in target},
        key=lambda stage: (stage[0], int(stage[1])),
    )
    rows = []
    for cerad, braak in stages:
        counts = Counter(
            row["disease_group"]
            for row in target
            if row["cerad"] == cerad and row["braak"] == braak
        )
        rows.append(
            {
                "cerad": cerad,
                "braak": braak,
                "asymp_donors": counts["ASYMP"],
                "ad_donors": counts["AD"],
                "exact_stage_matched_pairs": min(counts["ASYMP"], counts["AD"]),
            }
        )
    return rows


def build_audit(
    donors: list[dict[str, str]],
    images: list[dict[str, str]],
    registrations: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    core_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in images:
        core_rows[row["core_id"]].append(row)
    core_reference = {core_id: rows[0] for core_id, rows in core_rows.items()}
    fully_usable = {
        core_id
        for core_id, rows in core_rows.items()
        if len(rows) == 3
        and all(row["tissue_status"] in USABLE_STATUSES for row in rows)
    }
    registration_by_core: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in registrations:
        registration_by_core[row["core_id"]].append(row)

    summary = []
    for group in GROUPS:
        group_donors = [row for row in donors if row["disease_group"] == group]
        donor_ids = {row["donor_id"] for row in group_donors}
        group_cores = [
            core_id
            for core_id, row in core_reference.items()
            if row["donor_id"] in donor_ids
        ]
        paired = {
            donor_id
            for donor_id in donor_ids
            if {
                row["region"]
                for row in core_reference.values()
                if row["donor_id"] == donor_id
            }
            == {"frontal", "occipital"}
        }
        statuses = Counter(
            row["status"]
            for core_id in group_cores
            for row in registration_by_core[core_id]
        )
        summary.append(
            {
                "disease_group": group,
                "donors": len(group_donors),
                "female": sum(row["sex"] == "F" for row in group_donors),
                "male": sum(row["sex"] == "M" for row in group_donors),
                "age_observed_n": sum(bool(row["age_lower_bound"]) for row in group_donors),
                "age_lower_bound_median": median(
                    [row["age_lower_bound"] for row in group_donors]
                ),
                "age_lower_bound_min": min(
                    float(row["age_lower_bound"])
                    for row in group_donors
                    if row["age_lower_bound"]
                ),
                "age_lower_bound_max": max(
                    float(row["age_lower_bound"])
                    for row in group_donors
                    if row["age_lower_bound"]
                ),
                "age_topcoded": sum(
                    row["age_topcoded"] == "TRUE" for row in group_donors
                ),
                "pmi_observed_n": sum(bool(row["pmd_hours"]) for row in group_donors),
                "pmi_median_hours": median([row["pmd_hours"] for row in group_donors]),
                "donors_with_any_copathology": sum(
                    any(row.get(field) == "TRUE" for field in COPATHOLOGY_FIELDS)
                    for row in group_donors
                ),
                "cerad_distribution": distribution([row["cerad"] for row in group_donors]),
                "braak_distribution": distribution([row["braak"] for row in group_donors]),
                "donors_with_both_regions": len(paired),
                "biological_core_positions": len(group_cores),
                "fully_usable_triplets": sum(
                    core_id in fully_usable for core_id in group_cores
                ),
                "registration_pass_pairs": statuses["pass"],
                "registration_review_pairs": statuses["review"],
                "registration_fail_pairs": statuses["fail"],
            }
        )

    tma_balance = []
    for tma in sorted({row["tma"] for row in donors}, key=int):
        for group in GROUPS:
            donor_ids = {
                row["donor_id"]
                for row in donors
                if row["tma"] == tma and row["disease_group"] == group
            }
            group_cores = [
                core_id
                for core_id, row in core_reference.items()
                if row["tma"] == tma and row["donor_id"] in donor_ids
            ]
            tma_balance.append(
                {
                    "tma": tma,
                    "disease_group": group,
                    "donors": len(donor_ids),
                    "biological_core_positions": len(group_cores),
                    "fully_usable_triplets": sum(
                        core_id in fully_usable for core_id in group_cores
                    ),
                }
            )
    return summary, tma_balance


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


__all__ = [
    "build_audit",
    "build_stage_overlap",
    "read_csv",
    "write_csv",
]
