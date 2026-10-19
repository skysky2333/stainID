"""The one tutorial of the server: a "Fix the field" practice field and a "Mark bad areas" practice core from a public
dataset, the same for every project (the study's too, and any made later), so a tutorial never depends on a
project's outputs and never shows a study's tissue.

`stainid tutorial build --from <project> --field <tile> --core <core>` makes it once: a small self-contained
stainID project in `~/.stainid/tutorial` (`$STAINID_TUTORIAL`), with the field's and the core's pictures (as plain
files), their tile and core-image rows, detection's outputs for them (candidates with calls and both outlines, the
stage-2 masks, …), and `tutorial.json` (what to practise on, the credit line shown in the tutorial, the source's
licence and citation, and a hint for the "Change the type" step). The practice sessions (`practice-2-<stain>`,
`practice-areas`) are then answered from this project (`api.state.TUTORIAL`, the app's middleware), and nothing of
them is ever kept (`stainid.practice`).

The threads practice (AT8 only: threads mode of "Fix the field", its thread-threshold slider) is a field of another
public dataset with AT8 threads: `--threads-from <project> --threads-field <tile> --threads-credit …` (or `stainid
tutorial threads` to add it to a built tutorial) keeps its field picture and its thread network at every thread
threshold (`labels.thread_threshold.compute`) in `threads/`, served by `/api/annotate/practice-threads/…`.

Only use images whose licence allows public display and redistribution: the build records the source's licence.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

INFO = "tutorial.json"
VERSION = 1


def folder() -> Path:
    """Where the tutorial lives: `$STAINID_TUTORIAL`, else `~/.stainid/tutorial` (next to the server's settings)."""
    return Path(os.environ.get("STAINID_TUTORIAL") or Path.home() / ".stainid" / "tutorial")


_INFO: dict = {}


def info() -> dict | None:
    """`tutorial.json` of the built tutorial (None before `stainid tutorial build`), read again when it changes."""
    path = folder() / INFO
    try:
        stamp = path.stat().st_mtime_ns
    except FileNotFoundError:
        return None
    kept = _INFO.get(str(path))
    if kept and kept[0] == stamp:
        return kept[1]
    try:
        found = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    _INFO[str(path)] = (stamp, found)
    return found


def built() -> bool:
    return info() is not None


THREADS = "threads"  # the threads practice's folder in the tutorial: its field picture and `threads.json`


def threads() -> dict | None:
    """The threads practice (None when the tutorial has none): its field's tile id, picture size, credit line and the
    thread network at every thread threshold (`labels.thread_threshold.compute`)."""
    path = folder() / THREADS / "threads.json"
    try:
        stamp = path.stat().st_mtime_ns
    except FileNotFoundError:
        return None
    kept = _INFO.get(str(path))
    if kept and kept[0] == stamp:
        return kept[1]
    try:
        found = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    _INFO[str(path)] = (stamp, found)
    return found


def threads_picture() -> Path | None:
    path = folder() / THREADS / "field.jpg"
    return path if path.exists() else None


def add_threads(source_root: Path | str, field_tile: str, credit: str, out: Path | str | None = None) -> dict:
    """Add the threads practice to a tutorial folder (the built one by default): an AT8 field of `source_root` (read
    only; only images whose licence allows public display), its field picture and its thread ladder. Replaces an
    earlier one; returns what `threads()` serves."""
    import cv2  # noqa: PLC0415

    from stainid.imaging.core_images import read_bgr  # noqa: PLC0415
    from stainid.labels import (
        frames,  # noqa: PLC0415
        thread_threshold,  # noqa: PLC0415
    )
    from stainid.project import load_project  # noqa: PLC0415

    source = load_project(source_root)
    target = (Path(out) if out is not None else folder()) / THREADS
    table = pd.read_csv(source.analysis_tiles())
    row = table[table.tile_id.astype(str) == field_tile]
    if row.empty or str(row.iloc[0]["stain"]) != "AT8":
        raise KeyError(f"{field_tile} is not an AT8 tile of {source.root}")
    tile = row.iloc[0].to_dict()
    ladder = thread_threshold.compute(source, tile)
    if ladder["status"] != "done":
        raise ValueError(f"the threads of {field_tile} cannot be made (stage 2 of its core missing?)")
    bgr = read_bgr(source.root / tile["image_path"])
    x0, y0 = frames.field_origin(tile, source.context_px)
    x1 = min(bgr.shape[1], int(tile["x_px"]) + int(tile["width_px"]) + source.context_px)
    y1 = min(bgr.shape[0], int(tile["y_px"]) + int(tile["height_px"]) + source.context_px)
    work = target.with_name(THREADS + ".building")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    cv2.imwrite(str(work / "field.jpg"), bgr[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    box = {"x": int(tile["x_px"]) - x0, "y": int(tile["y_px"]) - y0, "width": int(tile["width_px"]), "height": int(tile["height_px"])}
    raw = yaml.safe_load((source.root / "stainid.yaml").read_text()) or {}
    dataset = raw.get("dataset") or {}
    found = {"tile_id": field_tile, "credit": credit, "width": x1 - x0, "height": y1 - y0, "box": box,
             "pixel_size_um": float(source.pixel_size_um), "source": {k: dataset.get(k) for k in ("id", "title", "licence", "citation", "doi") if dataset.get(k)},
             **{k: ladder[k] for k in ("status", "factors", "detection", "steps", "dab_threshold", "ridge_threshold")}}
    (work / "threads.json").write_text(json.dumps(found))
    if target.exists():
        shutil.rmtree(target)
    work.rename(target)
    _INFO.clear()
    return found


def practice() -> dict | None:
    """What the "Fix the field" tutorial practises on: {stain, tile_id, round2}, the credit line and the hint."""
    found = info()
    if not found:
        return None
    field = found["field"]
    return {"stain": field["stain"], "tile_id": field["tile_id"], "round2": f"practice-2-{field['stain']}", "credit": found.get("credit", ""),
            "hint": found.get("hint") or {}}


# ---------------------------------------------------------------- building it

def _rel(project, path: Path) -> Path:
    return Path(path).resolve().relative_to(project.root.resolve())


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame()


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _copy_picture(source, row: dict, target: Path) -> tuple[int, int]:
    """The core image as a plain picture file of the tutorial (a link file is rendered at the core image's size)."""
    from stainid.imaging.core_images import is_link, read_bgr  # noqa: PLC0415

    path = source.root / row["image_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not is_link(path) and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        shutil.copyfile(path, target.with_suffix(path.suffix.lower()))
        from PIL import Image  # noqa: PLC0415

        with Image.open(target.with_suffix(path.suffix.lower())) as image:
            return image.size
    import cv2  # noqa: PLC0415

    bgr = read_bgr(path)
    if bgr is None:
        raise FileNotFoundError(f"cannot read the picture of {row['core_id']} ({path})")
    cv2.imwrite(str(target.with_suffix(".jpg")), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return bgr.shape[1], bgr.shape[0]


def _hint(project, stain: str, tile: dict) -> dict:
    """The "Change the type" step's example: the field's clearest cored plaque (a large accepted plaque with a dense
    core, the upper left preferred) and the key that makes it cored / neuritic."""
    from stainid.labels import candidate_store  # noqa: PLC0415
    from stainid.labels import vocabulary as vocab  # noqa: PLC0415

    words = vocab.of(project, stain)
    key = next((c.key for c in vocab.assign_keys(words) if c.subtype == "cored-neuritic"), None)
    if key is None:
        return {}
    found = candidate_store.read_core(project, stain, str(tile["core_id"]))
    if found is None or found.empty:
        return {}
    mine = found[(found.tile_id == tile["tile_id"]) & found.decision.astype(str).map(candidate_store.accepted)]
    if mine.empty:
        return {}
    x0, y0, w, h = (float(tile[k]) for k in ("x_px", "y_px", "width_px", "height_px"))
    core = mine.get("m_core_fraction", pd.Series(0.0, index=mine.index)).fillna(0).astype(float)
    area = mine.area_um2.astype(float)
    upper_left = ((mine.x_core.astype(float) - x0) < w / 2) & ((mine.y_core.astype(float) - y0) < h / 2)
    score = area * (0.2 + core) * np.where(upper_left, 2.0, 1.0)
    best = mine.loc[score.idxmax()]
    fx, fy = (float(best.x_core) - x0) / w, (float(best.y_core) - y0) / h
    where = ("top" if fy < 1 / 3 else "bottom" if fy > 2 / 3 else "middle") + " " + ("left" if fx < 1 / 3 else "right" if fx > 2 / 3 else "centre")
    where = {"middle centre": "middle"}.get(where, where)
    return {"candidate_id": str(best.candidate_id), "key": key.upper() if len(key) == 1 else key,
            "text": f"For example, the big plaque near the {where} has a clear dense core: point at it and press {key.upper()} to make it cored / neuritic."}


def build(source_root: Path | str, field_tile: str, areas_core: str, credit: str, out: Path | str | None = None,
          threads_from: Path | str | None = None, threads_field: str = "", threads_credit: str = "") -> dict:
    """Make the tutorial from a project (read only): its field `field_tile` and its core `areas_core` (every stain image
    of that core), with `credit` shown in the tutorial, and with `threads_from` the threads practice (`add_threads`).
    Replaces an earlier tutorial; returns `tutorial.json`."""
    from stainid.labels import candidate_store  # noqa: PLC0415
    from stainid.project import load_project  # noqa: PLC0415
    from stainid.sampling.tiles import known_tiles  # noqa: PLC0415

    source = load_project(source_root)
    target = Path(out) if out is not None else folder()
    tiles = known_tiles(source)
    row = tiles[tiles.tile_id.astype(str) == field_tile]
    if row.empty:
        raise KeyError(f"{field_tile} is not a tile of {source.root}")
    field = row.iloc[0].to_dict()
    stain, field_core = str(field["stain"]), str(field["core_id"])
    images_path = source.input("core_images")
    images = _read(images_path)
    of_areas = images[images.core_id == areas_core]
    if of_areas.empty:
        raise KeyError(f"{areas_core} is not a core image of {source.root}")
    area_stains = sorted(set(of_areas.stain))
    cores = [field_core] if field_core == areas_core else [field_core, areas_core]
    work = target.with_name(target.name + ".building")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # the core images, as plain pictures
    kept = images[images.core_id.isin(cores)].copy()
    for i, r in kept.iterrows():
        picture = Path("data/images/tutorial") / f"{r['core_id']}_{r['stain']}"
        width, height = _copy_picture(source, r.to_dict(), work / picture)
        suffix = next(p.suffix for p in (work / picture).parent.iterdir() if p.stem == picture.name)
        kept.loc[i, "image_path"] = str(picture.with_suffix(suffix))
        for column, value in (("native_width_px", width), ("native_height_px", height), ("scale", 1), ("source_path", ""), ("source_box", ""),
                              ("source_pixel_size_um", "")):
            if column in kept:
                kept.loc[i, column] = str(value)
    pictures = dict(zip(zip(kept.core_id, kept.stain), kept.image_path))
    _write(kept, work / _rel(source, images_path))
    for key in ("image_list",):  # the image list, when the project keeps one: the same pictures
        try:
            listed_path = source.input(key)
        except (KeyError, ValueError):
            continue
        listed = _read(listed_path)
        if not listed.empty and "image_id" in listed:
            listed = listed[listed.image_id.isin(set(kept.image_id))].copy()
            if "path" in listed:
                by_image = kept.drop_duplicates("image_id").set_index("image_id")
                listed["path"] = [by_image.image_path.get(i, p) for i, p in zip(listed.image_id, listed.path)]
                for column, there in (("pixel_size_um", "pixel_width_um"), ("width_px", "native_width_px"), ("height_px", "native_height_px")):
                    if column in listed and there in by_image:
                        listed[column] = [by_image[there].get(i, v) for i, v in zip(listed.image_id, listed[column])]
            _write(listed, work / _rel(source, listed_path))

    # the tile table (the field and the core's tiles), pointing at the copied pictures
    analysis = source.analysis_tiles()
    table = _read(analysis)
    if not table.empty:
        table = table[table.core_id.isin(cores)].copy()
        if "image_path" in table:
            table["image_path"] = [pictures.get((c, s), p) for c, s, p in zip(table.core_id, table.stain, table.image_path)]
        _write(table, work / _rel(source, analysis))

    # detection's outputs of those cores: every per-core file of every v2 stage
    stages = candidate_store.folder(source, stain).parent.parent
    for stage in sorted(p for p in stages.iterdir() if p.is_dir()):
        for stain_dir in (stage / s for s in {stain, *area_stains}):
            if not stain_dir.is_dir():
                continue
            for core in cores:
                for path in stain_dir.glob(f"{core}.*"):
                    if path.stem == core:
                        copy = work / _rel(source, path)
                        copy.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, copy)

    # its settings: the source's stains, models and pixel size; its name and credit
    raw = yaml.safe_load((source.root / "stainid.yaml").read_text()) or {}
    dataset = raw.get("dataset") or {}
    config = {k: v for k, v in raw.items() if k not in ("name", "dataset")}
    config["name"] = "stainID tutorial"
    config["dataset"] = {k: dataset.get(k) for k in ("id", "title", "licence", "citation", "doi") if dataset.get(k)}
    (work / "stainid.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

    built_project = load_project(work)
    hint = _hint(built_project, stain, field)
    meta = {"version": VERSION, "built": time.time(), "credit": credit, "source": {"name": source.name, **config["dataset"]},
            "field": {"stain": stain, "tile_id": field_tile, "core_id": field_core},
            "areas": {"core_id": areas_core, "stains": area_stains}, "hint": hint}
    if threads_from is not None:
        found = add_threads(threads_from, threads_field, threads_credit, work)
        meta["threads"] = {"tile_id": found["tile_id"], "credit": found["credit"]}
    (work / INFO).write_text(json.dumps(meta, indent=1))
    if target.exists():
        shutil.rmtree(target)
    work.rename(target)
    _INFO.clear()
    return meta


def add_parser(sub) -> None:
    p = sub.add_parser("tutorial", help="the one tutorial every project serves (a public dataset's field and core)")
    what = p.add_subparsers(dest="tutorial_command", required=True)
    b = what.add_parser("build", help="make it from a project's field and core (only images whose licence allows public display)")
    b.add_argument("--from", dest="source", type=Path, required=True, help="the project the field and core come from")
    b.add_argument("--field", required=True, help="the tile id of the 'Fix the field' practice field")
    b.add_argument("--core", required=True, help="the core id of the 'Mark bad areas' practice core")
    b.add_argument("--credit", required=True, help='the credit line shown in the tutorial, e.g. "Images: … dataset (2023), MIT licence"')
    b.add_argument("--out", type=Path, default=None, help="where (default: $STAINID_TUTORIAL or ~/.stainid/tutorial)")
    for parser, required in ((b, False), (what.add_parser("threads", help="add (or replace) the AT8 threads practice of a built tutorial"), True)):
        parser.add_argument("--threads-from", type=Path, required=required, help="the project the AT8 threads practice field comes from")
        parser.add_argument("--threads-field", default="", required=required, help="its AT8 tile id")
        parser.add_argument("--threads-credit", default="", required=required, help="its credit line")
        if parser is not b:
            parser.add_argument("--out", type=Path, default=None, help="the tutorial folder (default: $STAINID_TUTORIAL or ~/.stainid/tutorial)")
    what.add_parser("show", help="what the built tutorial practises on")


def run(args) -> None:
    if args.tutorial_command == "threads":
        found = add_threads(args.threads_from, args.threads_field, args.threads_credit, args.out)
        target = (Path(args.out) if args.out is not None else folder()) / INFO
        meta = json.loads(target.read_text())
        meta["threads"] = {"tile_id": found["tile_id"], "credit": found["credit"]}
        target.write_text(json.dumps(meta, indent=1))
        _INFO.clear()
        print(f"threads practice added: {found['tile_id']}; {found['credit']}")
        return
    if args.tutorial_command == "build":
        meta = build(args.source, args.field, args.core, args.credit, args.out, args.threads_from, args.threads_field, args.threads_credit)
        print(f"tutorial built in {args.out or folder()}: field {meta['field']['tile_id']}, core {meta['areas']['core_id']} "
              f"({', '.join(meta['areas']['stains'])}); {meta['credit']}")
    else:
        found = info()
        print(json.dumps(found, indent=1) if found else f"no tutorial built yet in {folder()} (stainid tutorial build)")
