from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from stainid.pipelines.threshold_baseline import summarize_field, write_overlay
from stainid.registration.aligned_fields import render_montage
from stainid.tables import write_csv


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def analyze_pilot(
    core_id: str,
    pilot_manifest: Path,
    analysis_manifest: Path,
    calibration_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    with pilot_manifest.open(encoding="utf-8") as handle:
        pilot = json.load(handle)
    if pilot["core_id"] != core_id:
        raise ValueError("Pilot manifest core_id disagrees")
    images = {
        row["stain"]: row
        for row in read_csv(analysis_manifest)
        if row["core_id"] == core_id
    }
    if set(images) != {"6E10", "AT8", "NeuN"}:
        raise ValueError(f"Core does not contain a complete stain triplet: {core_id}")
    tma = int(images["NeuN"]["tma"])
    calibrations = {
        (int(row["tma"]), row["stain"]): float(row["threshold_dab_od"])
        for row in read_csv(calibration_path)
    }
    if any((tma, stain) not in calibrations for stain in images):
        raise ValueError(f"Missing slide calibration for LIP-{tma}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    objects = []
    overlays = {}
    for stain in ("6E10", "AT8", "NeuN"):
        path = Path(pilot["images"][stain])
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read aligned pilot field: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        row = images[stain]
        threshold = calibrations[tma, stain]
        summary, candidates, tissue, positive = summarize_field(
            rgb,
            stain,
            threshold,
            float(row["pixel_width_um"]),
            float(row["pixel_height_um"]),
        )
        summaries.append(
            {
                "core_id": core_id,
                "tma": tma,
                "stain": stain,
                "threshold_dab_od": threshold,
                **summary,
            }
        )
        for index, candidate in enumerate(candidates, start=1):
            objects.append(
                {
                    "core_id": core_id,
                    "stain": stain,
                    "object_id": f"{core_id}_{stain}_O{index:05d}",
                    **candidate,
                }
            )
        overlays[stain] = output_dir / f"{core_id}_{stain}_baseline.jpg"
        write_overlay(rgb, tissue, positive, overlays[stain])

    summary_path = output_dir / f"{core_id}_baseline.csv"
    object_path = output_dir / f"{core_id}_baseline_objects.csv"
    montage_path = output_dir / f"{core_id}_baseline_montage.jpg"
    write_csv(summary_path, summaries)
    write_csv(
        object_path,
        objects,
        [
            "core_id",
            "stain",
            "object_id",
            "centroid_x_px",
            "centroid_y_px",
            "area_um2",
            "equivalent_diameter_um",
            "perimeter_um",
            "circularity",
            "solidity",
            "aspect_ratio",
            "mean_dab_od",
            "max_dab_od",
            "touches_edge",
        ],
    )
    render_montage(overlays, montage_path)
    return summary_path, object_path, montage_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("core_id")
    parser.add_argument("--pilot-dir", type=Path, default=Path("data/pilot/aligned_fields"))
    parser.add_argument("--analysis-manifest", type=Path, default=Path("data/analysis/core_images.csv"))
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("data/analysis/slide_dab_calibration.csv"),
    )
    args = parser.parse_args()
    for path in analyze_pilot(
        args.core_id,
        args.pilot_dir / f"{args.core_id}.json",
        args.analysis_manifest,
        args.calibration,
        args.pilot_dir,
    ):
        print(path)


if __name__ == "__main__":
    main()
