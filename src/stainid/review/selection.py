from __future__ import annotations

import random
from collections import defaultdict


def evenly_spaced_candidates(
    rows: list[dict[str, object]],
    limit: int,
    value_key: str = "deposit_area_um2",
) -> list[dict[str, object]]:
    if limit <= 0:
        raise ValueError("Candidate review limit must be positive")
    ordered = sorted(rows, key=lambda row: (float(row[value_key]), str(row["candidate_id"])))
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
    return [ordered[index] for index in indices]


def select_candidate_review(
    rows: list[dict[str, object]],
    per_group: int,
    seed: int = 20260924,
    group_by_image: bool = False,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["source_set"]),
            str(row["tma"]),
            str(row["candidate_class"]),
        )
        if group_by_image:
            key += (str(row["source_image_id"]),)
        grouped[key].append(row)
    selected = [
        row
        for key in sorted(grouped)
        for row in evenly_spaced_candidates(grouped[key], per_group)
    ]
    random.Random(seed).shuffle(selected)
    return [
        {**row, "review_id": f"AR{index:03d}"}
        for index, row in enumerate(selected, start=1)
    ]


__all__ = [
    "evenly_spaced_candidates",
    "select_candidate_review",
]
