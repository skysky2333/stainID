"""Dataset packages: a public (or another lab's) dataset described in one folder, made into a stainID project.

    package/
      dataset.yaml     id, title, source/doi, licence, citation, lab or sites, stains [{stain, antibody, chromogen,
                       counterstain, also_on_slide, vectors}], pixel_size_um, image_kind (tiles | regions | wsi), class_map
      images.csv       image_id, path, width_px, height_px, pixel_size_um, stain, slide_id, donor_id, region, site, split
      annotations.csv  image_id, rater, kind (point | box | polygon | image_label), x, y, x2, y2, polygon, label,
                       stainid_class, confidence; positions in the image's own pixels
      donors.csv       optional: donor_id, diagnosis, braak, cerad, thal, adnc, apoe, age, sex, pmi, ...

`read_package` checks a package and says what is in it (`preview`); `create_project` makes the project: its kind and
imaging settings from dataset.yaml (with the stains' own measured colour vectors), the image list pointing at the
package's files (nothing is copied), and the donor table. The labels are brought in when the images are prepared
(`stainid.imageset.annotations`), because they are placed on the project's tiles.

A big public dataset is often downloaded bit by bit: images.csv lists every image, and an image whose `path` is empty,
or whose file is not there (yet), is *not downloaded*. The preview counts the images present and listed, the project is
made with the present ones, and `add_new_files` ("Look for new files") adds those downloaded since.
"""
from __future__ import annotations

import csv
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from stainid import cache
from stainid.imaging import chromogens
from stainid.imaging.settings import FAMILIES, family_of
from stainid.labels import vocabulary as vocab
from stainid.project import DEFAULTS, Project, save_project
from stainid.tables import write_csv

KIND_OF = {"tiles": "images", "regions": "images", "images": "images", "wsi": "slides", "slides": "slides"}
ANNOTATION_KINDS = ("point", "box", "polygon", "image_label")
IGNORE = "ignore"
FAMILY_STAIN = {"amyloid": "6E10", "tau": "AT8", "neun": "NeuN"}
DONOR_GROUP_COLUMNS = ("disease_group", "diagnosis", "adnc", "group", "cognition")
CACHE_LIMIT_GB = 20  # the space a dataset project's pictures for viewing may take (editable in Settings; `stainid.disk_cache`)
PRESENT, NOT_DOWNLOADED, MISSING = "present", "not downloaded", "missing"
FILES_RECHECK_SECONDS = 30.0  # the Images and Checks screens look for newly downloaded files at most this often


def _number(value) -> float | None:
    """A positive number, or None for anything else (an empty cell, or a note such as "per image")."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def class_choices() -> dict[str, list[str]]:
    """Every class a label can be mapped to, per model family: the stain's identity classes and, as "identity/subtype",
    their types (`labels.vocabulary`; "unsure" left out)."""
    out = {}
    for family, stain in FAMILY_STAIN.items():
        classes = [c for c in vocab.defaults(stain).classes if c.role != vocab.SKIP]
        out[family] = [name for c in classes for name in (c.id, *(f"{c.id}/{s.id}" for s in c.subtypes))]
    return out


@dataclass
class Package:
    folder: Path
    meta: dict
    images: list[dict[str, str]]  # shared between requests (`cache.memo`): never changed in place
    annotations_path: Path | None
    donors: list[dict[str, str]]

    @property
    def id(self) -> str:
        return str(self.meta.get("id") or self.folder.parent.name or self.folder.name)

    @property
    def title(self) -> str:
        return str(self.meta.get("title") or self.id)

    @property
    def kind(self) -> str:
        kinds = {r.get("image_kind", "") for r in self.images} - {""}
        value = str(self.meta.get("image_kind") or (kinds.pop() if len(kinds) == 1 else "tiles"))
        return KIND_OF.get(value.lower(), "images")

    def annotations(self) -> list[dict[str, str]]:
        return _rows(self.annotations_path) if self.annotations_path else []


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(handle)]


def package_folder(path: Path | str) -> Path:
    """The package folder: the one given, or its `package/` subfolder (when a dataset folder was chosen)."""
    path = Path(path).expanduser()
    if (path / "dataset.yaml").exists():
        return path
    if (path / "package" / "dataset.yaml").exists():
        return path / "package"
    raise ValueError(f"{path} is not a dataset package: it has no dataset.yaml (nor a package/ folder with one)")


def read_package(path: Path | str) -> Package:
    folder = package_folder(path)
    meta = yaml.safe_load((folder / "dataset.yaml").read_text(encoding="utf-8")) or {}
    if not isinstance(meta, dict):
        raise ValueError("dataset.yaml must be a mapping of settings")
    listed = folder / "images.csv"
    if not listed.exists():
        raise ValueError(f"{folder} has no images.csv")
    images = cache.memo(("package images", str(listed)), [listed], lambda: _rows(listed))
    if not images:
        raise ValueError("images.csv lists no images")
    missing = {"image_id", "path"} - set(images[0])
    if missing:
        raise ValueError(f"images.csv needs the columns {', '.join(sorted(missing))}")
    annotations = folder / "annotations.csv"
    donors = _rows(folder / "donors.csv") if (folder / "donors.csv").exists() else []
    return Package(folder, meta, images, annotations if annotations.exists() else None, donors)


def image_path(package: Package, row: dict) -> Path | None:
    """An image's file: absolute as written, else relative to the package folder; None when the package gives none (the
    image is not downloaded yet)."""
    written = (row.get("path") or "").strip()
    if not written:
        return None
    path = Path(written).expanduser()
    return Path(os.path.normpath(path if path.is_absolute() else package.folder / path))


def _listing(folder: Path) -> frozenset[str]:
    try:
        return frozenset(entry.name for entry in os.scandir(folder))
    except OSError:
        return frozenset()


def file_states(package: Package, fresh: bool = False) -> list[str]:
    """For every image of the package: `PRESENT`, `NOT_DOWNLOADED` (no path given) or `MISSING` (its file is not where
    the path says). Looked up in one listing per folder, not file by file (a package may list 80,000 crops); the
    listings are shared for a few seconds (`cache.names`), or made now with `fresh`."""
    names: dict[Path, frozenset[str]] = {}
    out = []
    for row in package.images:
        path = image_path(package, row)
        if path is None:
            out.append(NOT_DOWNLOADED)
            continue
        if path.parent not in names:
            names[path.parent] = _listing(path.parent) if fresh else cache.names(path.parent)
        out.append(PRESENT if path.name in names[path.parent] else MISSING)
    return out


def file_counts(states: list[str]) -> dict[str, int]:
    counts = Counter(states)
    return {"listed": len(states), "present": counts[PRESENT], "not_downloaded": counts[NOT_DOWNLOADED], "missing": counts[MISSING]}


def _missing_examples(package: Package, states: list[str], count: int = 5) -> list[str]:
    paths = (image_path(package, r) for r, state in zip(package.images, states) if state == MISSING)
    return [str(p) for p, _ in zip(paths, range(count))]


def _vectors(raw, stain: str) -> dict[str, list[float]]:
    """A stain's measured colour vectors as dataset.yaml gives them, keyed by stainID's chromogen names."""
    if not raw:
        return {}
    try:
        return chromogens.clean_vectors(raw)
    except ValueError as error:
        raise ValueError(f"dataset.yaml, stain {stain}: {error}") from None


def stain_entries(package: Package, states: list[str] | None = None) -> list[dict]:
    """The package's stains with their colour, stainID family and measured colour vectors (`vectors`: chromogen ->
    optical-density colour measured on the dataset's own slides); a slide carrying two chromogens lists the other one in
    `others` (from `also_on_slide`, or from a second entry with the same slide stain written as "AT8+pTDP-43"). With
    `states` (`file_states`) each entry also counts its images that are `present`."""
    entries = [dict(s) for s in package.meta.get("stains") or [] if isinstance(s, dict) and s.get("stain")]
    used = Counter(r.get("stain", "") for r in package.images)
    by_name = {str(s["stain"]): s for s in entries}
    for stain in used:
        if not stain or stain in by_name:
            continue
        parts = [p.strip() for p in stain.replace("/", "+").split("+") if p.strip()]
        listed = [by_name.get(p, {"stain": p}) for p in parts]
        if len(listed) > 1:  # "AT8+pTDP-43": the first stain stainID analyses is the one of interest, the others' colours are left out
            main = next((e for e in listed if e.get("family") or family_of(e.get("antibody") or e["stain"])), listed[0])
            entries.append({**main, "stain": stain, "also_on_slide": [e.get("chromogen") or "DAB" for e in listed if e is not main]})
        else:
            entries.append({"stain": stain})
    entries = [e for e in entries if used.get(str(e["stain"]))] or entries  # only the stains the images name
    present = Counter(r.get("stain", "") for r, state in zip(package.images, states or []) if state == PRESENT)
    out = []
    for entry in entries:
        stain = str(entry["stain"])
        parts = [p.strip() for p in stain.replace("/", "+").split("+") if p.strip()]
        others = [chromogens.canonical(o.get("chromogen") if isinstance(o, dict) else o) for o in entry.get("also_on_slide") or []]
        family = entry.get("family") or family_of(entry.get("antibody") or "") or next((family_of(p) for p in parts if family_of(p)), "")
        out.append({"stain": stain, "antibody": entry.get("antibody") or "", "chromogen": chromogens.canonical(entry.get("chromogen")) or "DAB",
                    "chromogen_written": entry.get("chromogen") or "", "counterstain": chromogens.canonical(str(entry.get("counterstain") or "").split("(")[0]) or "hematoxylin",
                    "others": [o for o in others if o], "family": family or "", "images": int(used.get(stain, 0)),
                    "vectors": _vectors(entry.get("vectors"), stain), **({"present": int(present.get(stain, 0))} if states is not None else {})})
    return out


def default_class_map(package: Package) -> dict[str, str]:
    """The package's own class map, completed with the `stainid_class` the annotations name (label -> class)."""
    mapping = {str(k): str(v) for k, v in (package.meta.get("class_map") or {}).items()}
    for row in package.annotations():
        label = row.get("label", "")
        if label and label not in mapping and row.get("stainid_class"):
            mapping[label] = row["stainid_class"]
    return mapping


def preview(path: Path | str) -> dict:
    """What a package holds, in the words of the import page; `files` counts the images present and listed."""
    package = read_package(path)
    annotations = package.annotations()
    raters = Counter(r.get("rater") or "rater" for r in annotations)
    kinds = Counter(r.get("kind") or "point" for r in annotations)
    labels = Counter(r.get("label") or r.get("stainid_class") or "" for r in annotations)
    states = file_states(package)
    pixel = _number(package.meta.get("pixel_size_um")) or next((_number(r.get("pixel_size_um")) for r in package.images if _number(r.get("pixel_size_um"))), None)
    return {"folder": str(package.folder), "id": package.id, "title": package.title, "kind": package.kind,
            "licence": package.meta.get("licence") or package.meta.get("license") or "", "citation": package.meta.get("citation") or "",
            "source": package.meta.get("source") or package.meta.get("source_urls") or "", "doi": package.meta.get("doi") or "",
            "sites": package.meta.get("sites") or package.meta.get("lab") or "", "notes": package.meta.get("notes") or "",
            "images": len(package.images), "donors": len({r.get("donor_id") for r in package.images if r.get("donor_id")}),
            "donor_rows": len(package.donors), "pixel_size_um": pixel, "stains": stain_entries(package, states),
            "annotations": len(annotations), "raters": dict(raters), "annotation_kinds": dict(kinds), "labels": dict(labels.most_common()),
            "class_map": default_class_map(package), "classes": class_choices(), "families": FAMILIES,
            "files": {**file_counts(states), "missing_examples": _missing_examples(package, states)}, "cache_limit_gb": CACHE_LIMIT_GB,
            "splits": dict(Counter(r.get("split", "") for r in package.images if r.get("split")))}


def imaging_settings(package: Package, chosen: dict[str, dict] | None = None) -> dict:
    """The project's imaging section: the package's pixel size, and per stain its family, colours and measured colour
    vectors (`chosen` overrides them, as edited on the import page)."""
    stains = {}
    for entry in stain_entries(package):
        own = (chosen or {}).get(entry["stain"], {})
        vectors = own.get("vectors", entry["vectors"])
        stains[entry["stain"]] = {"family": own.get("family", entry["family"]) or IGNORE, "chromogen": own.get("chromogen", entry["chromogen"]),
                                  **({"others": own.get("others", entry["others"])} if own.get("others", entry["others"]) else {}),
                                  **({"antibody": entry["antibody"]} if entry["antibody"] else {}),
                                  **({"counterstain": entry["counterstain"]} if entry["counterstain"] != "hematoxylin" else {}),
                                  **({"vectors": vectors} if vectors else {})}
    pixel = _number(package.meta.get("pixel_size_um"))  # a text such as "per image (…)" leaves it to each image
    return {"native_pixel_size_um": pixel, "rescale": "on_the_fly" if package.kind == "slides" else "on_import",
            "stains": stains}


def donor_rows(package: Package) -> list[dict[str, str]]:
    """The donor table: the package's donors.csv (the diagnosis as the group when there is no group column), plus every
    donor of the images that it does not list."""
    out = []
    for row in package.donors:
        if not row.get("donor_id"):
            continue
        group = next((row[c] for c in DONOR_GROUP_COLUMNS if row.get(c)), "")
        out.append({"donor_id": row["donor_id"], "disease_group": row.get("disease_group") or group,
                    **{k: v for k, v in row.items() if k not in ("donor_id", "disease_group")}})
    listed = {r["donor_id"] for r in out}
    out += [{"donor_id": d, "disease_group": ""} for d in dict.fromkeys(r.get("donor_id", "") for r in package.images) if d and d not in listed]
    return out


IMAGE_COLUMNS = ("image_id", "stain", "donor_id", "region", "slide_id", "site", "batch", "pixel_size_um", "disease_group", "split", "width_px",
                 "height_px")


def image_rows(package: Package, states: list[str]) -> list[dict[str, str]]:
    """The image list rows of the package's images that are present."""
    return [{**{k: r.get(k, "") for k in IMAGE_COLUMNS}, "path": str(image_path(package, r))} for r, state in zip(package.images, states)
            if state == PRESENT]


def create_project(path: Path | str, folder: Path | str, name: str = "", class_map: dict[str, str] | None = None,
                   stains: dict[str, dict] | None = None) -> Project:
    """Make a project for a package in `folder` (which must not hold a project yet), with the images present now."""
    package = read_package(path)
    root = Path(folder).expanduser().resolve()
    if (root / "stainid.yaml").exists():
        raise FileExistsError(f"{root} already holds a stainID project")
    mapping = {**default_class_map(package), **(class_map or {})}
    changes = {"name": name.strip() or package.title, "kind": package.kind, "imaging": imaging_settings(package, stains),
               "donor_ids": {"digits": 0, "prefixed_digits": 0}, "tma": {"prefix": ""}, "cache_limit_gb": CACHE_LIMIT_GB,
               "dataset": {"id": package.id, "title": package.title, "package": str(package.folder),
                           "licence": str(package.meta.get("licence") or package.meta.get("license") or ""),
                           "citation": str(package.meta.get("citation") or ""), "doi": str(package.meta.get("doi") or ""),
                           "class_map": mapping}}
    project = save_project(Project(root=root, config=DEFAULTS), changes)
    from stainid.imageset.table import save_images

    rows = image_rows(package, file_states(package))
    if rows:  # none downloaded yet: the image list stays empty until "Look for new files" finds some
        save_images(project, rows)
    donors = donor_rows(package)
    if donors:
        columns = list(dict.fromkeys(k for r in donors for k in r))
        write_csv(project.input("donor_metadata"), donors, columns)
    return project


# ---------------------------------------------------------------- a project made from a package, as its files arrive

def project_package(project: Project) -> Package | None:
    """The package a project was made from (None: none, or it cannot be read now, e.g. its drive is not connected)."""
    folder = (project.config.get("dataset") or {}).get("package")
    if not folder:
        return None
    try:
        return read_package(folder)
    except (ValueError, OSError, yaml.YAMLError):
        return None


def _in_study(project: Project, package: Package) -> list[bool]:
    """For every image of the package: is it in the study's image list (by its id, or by its file)?"""
    from stainid.imageset.table import read_images

    rows = read_images(project)
    ids, paths = {r["image_id"] for r in rows if r.get("image_id")}, {os.path.normpath(r["path"]) for r in rows if r.get("path")}
    return [bool(r.get("image_id") and r["image_id"] in ids) or str(image_path(package, r)) in paths for r in package.images]


def _project_files(project: Project, package: Package, fresh: bool = False) -> dict:
    states, listed = file_states(package, fresh), _in_study(project, package)
    return {**file_counts(states), "in_study": sum(listed), "new": sum(s == PRESENT and not inside for s, inside in zip(states, listed)),
            "kind": package.kind, "title": package.title}


def dataset_files(project: Project, fresh: bool = False) -> dict | None:
    """Of the package's images: how many are listed, present, in the study, newly downloaded (present but not in the
    study yet), not downloaded and missing; None for a project not made from a package. The folders are looked at
    again at most every `FILES_RECHECK_SECONDS` (or now, with `fresh`)."""
    package = project_package(project)
    if package is None:
        return None
    inputs = [package.folder / "images.csv", project.input("image_list")]
    if fresh:
        found = _project_files(project, package, fresh=True)
        cache.remember(("dataset files", str(project.root)), inputs, found)  # what the next look (in the background) starts from
        return found
    from stainid import disk_cache  # noqa: PLC0415

    return cache.memo(("dataset files", str(project.root)), inputs, lambda: _project_files(project, package), FILES_RECHECK_SECONDS, folders=True,
                      persist=disk_cache.answer_file(project.root, "dataset-files.pkl"), background=True)


def add_new_files(project: Project) -> int:
    """Look for new files: add every image of the package that is present now and not in the study's image list yet
    (the rows already there, with any changes made to them, are kept). Returns how many were added."""
    from stainid.imageset.table import read_images, save_images

    package = project_package(project)
    if package is None:
        raise ValueError("This study was not made from a dataset package, or the package cannot be read now (is its drive connected?)")
    states, listed = file_states(package, fresh=True), _in_study(project, package)  # the folders as they are now
    new = image_rows(package, [s if not inside else MISSING for s, inside in zip(states, listed)])
    if new:
        save_images(project, read_images(project) + new)
    return len(new)
