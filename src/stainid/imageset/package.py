"""Dataset packages: a public (or another lab's) dataset described in one folder, made into a stainID project.

    package/
      dataset.yaml     id, title, source/doi, licence, citation, lab or sites, stains [{stain, antibody, chromogen,
                       counterstain, also_on_slide}], pixel_size_um, image_kind (tiles | regions | wsi), class_map
      images.csv       image_id, path, width_px, height_px, pixel_size_um, stain, slide_id, donor_id, region, site, split
      annotations.csv  image_id, rater, kind (point | box | polygon | image_label), x, y, x2, y2, polygon, label,
                       stainid_class, confidence; positions in the image's own pixels
      donors.csv       optional: donor_id, diagnosis, braak, cerad, thal, adnc, apoe, age, sex, pmi, ...

`read_package` checks a package and says what is in it (`preview`); `create_project` makes the project: its kind and
imaging settings from dataset.yaml, the image list pointing at the package's files (nothing is copied), and the donor
table. The labels are brought in when the images are prepared (`stainid.imageset.annotations`), because they are placed
on the project's tiles.
"""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from stainid.imaging import chromogens
from stainid.imaging.settings import FAMILIES, family_of
from stainid.labels.tasks import CLASSES
from stainid.project import DEFAULTS, Project, save_project
from stainid.tables import write_csv

KIND_OF = {"tiles": "images", "regions": "images", "images": "images", "wsi": "slides", "slides": "slides"}
ANNOTATION_KINDS = ("point", "box", "polygon", "image_label")
IGNORE = "ignore"
# stainID classes a package may map its labels to, per family (besides "ignore")
EXTRA_CLASSES = {"amyloid": ["plaque", "vascular", "artifact or unclear"], "tau": ["tau+ neuron", "neuritic plaque", "thread"], "neun": []}
FAMILY_STAIN = {"amyloid": "6E10", "tau": "AT8", "neun": "NeuN"}
DONOR_GROUP_COLUMNS = ("disease_group", "diagnosis", "adnc", "group", "cognition")


def class_choices() -> dict[str, list[str]]:
    """Every class a label can be mapped to, per model family."""
    return {family: [c for c in CLASSES[FAMILY_STAIN[family]] if c != "unsure"] + EXTRA_CLASSES[family] for family in FAMILY_STAIN}


@dataclass
class Package:
    folder: Path
    meta: dict
    images: list[dict[str, str]]
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
    if not (folder / "images.csv").exists():
        raise ValueError(f"{folder} has no images.csv")
    images = _rows(folder / "images.csv")
    if not images:
        raise ValueError("images.csv lists no images")
    missing = {"image_id", "path"} - set(images[0])
    if missing:
        raise ValueError(f"images.csv needs the columns {', '.join(sorted(missing))}")
    annotations = folder / "annotations.csv"
    donors = _rows(folder / "donors.csv") if (folder / "donors.csv").exists() else []
    return Package(folder, meta, images, annotations if annotations.exists() else None, donors)


def image_path(package: Package, row: dict) -> Path:
    """An image's file: absolute as written, else relative to the package folder."""
    path = Path(row["path"]).expanduser()
    return path if path.is_absolute() else Path(package.folder / path).resolve()


def stain_entries(package: Package) -> list[dict]:
    """The package's stains with their colour and stainID family; a slide carrying two chromogens lists the other one in
    `others` (from `also_on_slide`, or from a second entry with the same slide stain written as "AT8+pTDP-43")."""
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
    out = []
    for entry in entries:
        stain = str(entry["stain"])
        parts = [p.strip() for p in stain.replace("/", "+").split("+") if p.strip()]
        others = [chromogens.canonical(o.get("chromogen") if isinstance(o, dict) else o) for o in entry.get("also_on_slide") or []]
        family = entry.get("family") or family_of(entry.get("antibody") or "") or next((family_of(p) for p in parts if family_of(p)), "")
        out.append({"stain": stain, "antibody": entry.get("antibody") or "", "chromogen": chromogens.canonical(entry.get("chromogen")) or "DAB",
                    "chromogen_written": entry.get("chromogen") or "", "counterstain": entry.get("counterstain") or "hematoxylin",
                    "others": [o for o in others if o], "family": family or "", "images": int(used.get(stain, 0))})
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
    """What a package holds, in the words of the import page."""
    package = read_package(path)
    annotations = package.annotations()
    raters = Counter(r.get("rater") or "rater" for r in annotations)
    kinds = Counter(r.get("kind") or "point" for r in annotations)
    labels = Counter(r.get("label") or r.get("stainid_class") or "" for r in annotations)
    files = [image_path(package, r) for r in package.images[:50]]
    pixel = package.meta.get("pixel_size_um") or next((r.get("pixel_size_um") for r in package.images if r.get("pixel_size_um")), None)
    return {"folder": str(package.folder), "id": package.id, "title": package.title, "kind": package.kind,
            "licence": package.meta.get("licence") or package.meta.get("license") or "", "citation": package.meta.get("citation") or "",
            "source": package.meta.get("source") or package.meta.get("source_urls") or "", "doi": package.meta.get("doi") or "",
            "sites": package.meta.get("sites") or package.meta.get("lab") or "", "notes": package.meta.get("notes") or "",
            "images": len(package.images), "donors": len({r.get("donor_id") for r in package.images if r.get("donor_id")}),
            "donor_rows": len(package.donors), "pixel_size_um": float(pixel) if pixel else None, "stains": stain_entries(package),
            "annotations": len(annotations), "raters": dict(raters), "annotation_kinds": dict(kinds), "labels": dict(labels.most_common()),
            "class_map": default_class_map(package), "classes": class_choices(), "families": FAMILIES,
            "files_found": sum(f.exists() for f in files), "files_checked": len(files),
            "splits": dict(Counter(r.get("split", "") for r in package.images if r.get("split")))}


def imaging_settings(package: Package, chosen: dict[str, dict] | None = None) -> dict:
    """The project's imaging section: the package's pixel size, and per stain its family and colours (`chosen` overrides
    them, as edited on the import page)."""
    stains = {}
    for entry in stain_entries(package):
        own = (chosen or {}).get(entry["stain"], {})
        stains[entry["stain"]] = {"family": own.get("family", entry["family"]) or IGNORE, "chromogen": own.get("chromogen", entry["chromogen"]),
                                  **({"others": own.get("others", entry["others"])} if own.get("others", entry["others"]) else {}),
                                  **({"antibody": entry["antibody"]} if entry["antibody"] else {}),
                                  **({"counterstain": entry["counterstain"]} if entry["counterstain"] != "hematoxylin" else {})}
    pixel = package.meta.get("pixel_size_um")
    return {"native_pixel_size_um": float(pixel) if pixel else None, "rescale": "on_the_fly" if package.kind == "slides" else "on_import",
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


def image_rows(package: Package) -> list[dict[str, str]]:
    keep = ("image_id", "stain", "donor_id", "region", "slide_id", "site", "batch", "pixel_size_um", "disease_group", "split", "width_px", "height_px")
    return [{**{k: r.get(k, "") for k in keep}, "path": str(image_path(package, r))} for r in package.images]


def create_project(path: Path | str, folder: Path | str, name: str = "", class_map: dict[str, str] | None = None,
                   stains: dict[str, dict] | None = None) -> Project:
    """Make a project for a package in `folder` (which must not hold a project yet)."""
    package = read_package(path)
    root = Path(folder).expanduser().resolve()
    if (root / "stainid.yaml").exists():
        raise FileExistsError(f"{root} already holds a stainID project")
    mapping = {**default_class_map(package), **(class_map or {})}
    changes = {"name": name.strip() or package.title, "kind": package.kind, "imaging": imaging_settings(package, stains),
               "donor_ids": {"digits": 0, "prefixed_digits": 0},
               "dataset": {"id": package.id, "title": package.title, "package": str(package.folder),
                           "licence": str(package.meta.get("licence") or package.meta.get("license") or ""),
                           "citation": str(package.meta.get("citation") or ""), "doi": str(package.meta.get("doi") or ""),
                           "class_map": mapping}}
    project = save_project(Project(root=root, config=DEFAULTS), changes)
    from stainid.imageset.table import save_images

    save_images(project, image_rows(package))
    donors = donor_rows(package)
    if donors:
        columns = list(dict.fromkeys(k for r in donors for k in r))
        write_csv(project.input("donor_metadata"), donors, columns)
    return project
