from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from stainid.tables import write_csv


@dataclass(frozen=True)
class ReviewSource:
    manifest: Path
    proposals_dir: Path
    proposal_source: str
    default_selection_role: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def union_columns(rows: list[dict[str, object]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for column in row:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return columns


def assemble_development_features(
    annotation_manifest: Path,
    field_manifest: Path,
    review_sources: list[ReviewSource],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    annotations = {row["image_id"]: row for row in read_csv(annotation_manifest)}
    fields = {row["image_id"]: row for row in read_csv(field_manifest)}
    feature_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []
    candidate_ids: set[str] = set()

    for source in review_sources:
        for review in read_csv(source.manifest):
            image_id = review["image_id"]
            if image_id not in annotations or image_id not in fields:
                raise ValueError(f"Reviewed image is absent from source manifests: {image_id}")
            annotation = annotations[image_id]
            field = fields[image_id]
            if annotation["stain"] != field["stain"]:
                raise ValueError(f"Stain mismatch for reviewed image: {image_id}")
            summary_path = source.proposals_dir / f"{image_id}_summary.csv"
            objects_path = source.proposals_dir / f"{image_id}_objects.csv"
            summaries = read_csv(summary_path)
            if len(summaries) != 1 or summaries[0]["image_id"] != image_id:
                raise ValueError(f"Expected one matching summary row for {image_id}")
            selection_role = review.get("selection_role") or source.default_selection_role
            common: dict[str, object] = {
                "annotation_id": annotation["annotation_id"],
                "image_id": image_id,
                "stain": annotation["stain"],
                "tma": review["tma"],
                "selection_role": selection_role,
                "proposal_source": source.proposal_source,
                "field_id": field["field_id"],
                "target_dab_quantile": field["target_dab_quantile"],
                "field_x_px": field["x_px"],
                "field_y_px": field["y_px"],
                "field_width_px": field["width_px"],
                "field_height_px": field["height_px"],
                "pixel_width_um": annotation["pixel_width_um"],
                "pixel_height_um": annotation["pixel_height_um"],
                "source_tissue_status": annotation["tissue_status"],
                "source_tissue_fraction": annotation["tissue_fraction"],
            }
            feature_rows.append(
                {**common, **{key: value for key, value in summaries[0].items() if key != "image_id"}}
            )
            for row in read_csv(objects_path):
                if row["image_id"] != image_id:
                    raise ValueError(f"Object row belongs to another image: {image_id}")
                candidate_id = row["candidate_id"]
                if candidate_id in candidate_ids:
                    raise ValueError(f"Duplicate development candidate ID: {candidate_id}")
                candidate_ids.add(candidate_id)
                object_rows.append(
                    {**common, **{key: value for key, value in row.items() if key != "image_id"}}
                )

    return feature_rows, object_rows


def write_development_features(
    feature_rows: list[dict[str, object]],
    object_rows: list[dict[str, object]],
    feature_output: Path,
    object_output: Path,
) -> tuple[Path, Path]:
    write_csv(feature_output, feature_rows, union_columns(feature_rows))
    write_csv(object_output, object_rows, union_columns(object_rows))
    return feature_output, object_output


__all__ = [
    "ReviewSource",
    "assemble_development_features",
    "union_columns",
    "write_development_features",
]
