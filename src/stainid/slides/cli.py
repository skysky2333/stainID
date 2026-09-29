"""Slide-level commands: native-resolution core export and TMA layout attachment."""
from __future__ import annotations

import argparse
from pathlib import Path

from stainid.slides.export import export_cores
from stainid.slides.layout import attach_layout


def export_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stainid-export-cores", description="Export native-resolution core PNGs listed in the core manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("data/core_manifest.csv"))
    parser.add_argument("--slide", action="append", default=[])
    parser.add_argument("--core", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-workers", type=int, default=2)
    args = parser.parse_args(argv)
    export_cores(args.manifest, set(args.slide) or None, set(args.core) or None, args.overwrite, args.write_workers)


def layout_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="stainid-attach-layout", description="Attach TMA map metadata (donor, region, group) to the core manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("data/core_manifest.csv"))
    parser.add_argument("--layout", type=Path, default=Path("data/tma_layout.csv"))
    args = parser.parse_args(argv)
    attach_layout(args.manifest, args.layout)
    print(args.manifest)
