"""The Checks screen of an image-set or whole-slide project: the image list, the imaging settings and the donor table
compared with each other, in the same form as the TMA checks (`stainid.setup.checks`)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from stainid import cache
from stainid.imageset.table import read_images, resolve
from stainid.imaging.settings import FAMILIES, analysis_stain
from stainid.imaging.sources import available
from stainid.project import Project
from stainid.setup.checks import _fix, _issue, _names

FILE_CHECKS = 3000  # files looked for at most (the image list may be long; the rest are found by Prepare images)


def _missing_files(project: Project, rows: list[dict]) -> list[str]:
    paths = [str(resolve(project, r)) for r in rows[:FILE_CHECKS]]
    return cache.memo(("missing image files", str(project.input("image_list"))), [project.input("image_list")],
                      lambda: [p for p in paths if not Path(p).exists()])


def image_issues(project: Project) -> list[dict]:
    rows = read_images(project)
    if not rows:
        return [_issue("no_images", "problem", "No images yet", "Add the images of the study on the Images screen.",
                       fixes=[_fix("Go to the Images screen", "goto", screen="images")])]
    issues = []
    goto = [_fix("Go to the Images screen", "goto", screen="images")]
    no_donor = [r["image_id"] or r["path"] for r in rows if not r.get("donor_id")]
    if no_donor:
        issues.append(_issue("images_without_donor", "problem", f"{len(no_donor)} images have no donor",
                             "Every image needs the donor it comes from; images without one are left out.",
                             [{"text": _names(no_donor), "fixes": []}], goto))
    stains = Counter(r.get("stain", "") for r in rows)
    unknown = sorted(s for s in stains if not analysis_stain(project, s))
    if unknown:
        level = "problem" if len(unknown) == len(stains) else "warning"
        issues.append(_issue("stains_not_analysed", level, f"{len(unknown)} stains are not analysed: {_names([s or '(no stain)' for s in unknown])}",
                             "Say under Imaging what each stain is (" + ", ".join(v.lower() for k, v in FAMILIES.items() if k != "ignore")
                             + "), or that it is not analysed. Images of these stains are left out until then.",
                             [{"text": f"{s or '(no stain)'}: {stains[s]} images", "fixes": []} for s in unknown], goto))
    missing = _missing_files(project, rows)
    if missing:
        issues.append(_issue("image_files_missing", "problem", f"{len(missing)} image files are not where the image list says",
                             "The drive may not be connected, or the files were moved. Images that cannot be read are left out.",
                             [{"text": p, "fixes": []} for p in missing], goto))
    if project.kind == "slides" and not available()["openslide"]:
        issues.append(_issue("no_slide_reader", "warning", "The whole-slide reader is not installed",
                             'Most whole-slide scans need OpenSlide: install it with  pip install "stainid[slides]"  and start stainID again. '
                             "Tiled TIFF files can be read without it, more slowly."))
    donors = {d["donor_id"]: d for d in cache.rows(project.input("donor_metadata"))}
    on_images = sorted({r["donor_id"] for r in rows if r.get("donor_id")})
    not_listed = [d for d in on_images if d not in donors]
    if donors and not_listed:
        issues.append(_issue("donors_not_in_table", "warning", f"{len(not_listed)} donors of the images are not in the donor table",
                             "They are analysed, but their group and other details are unknown.", [{"text": _names(not_listed), "fixes": []}],
                             [_fix("Add them to the donor table", "add_image_donors", donors=not_listed)]))
    no_group = [d for d in on_images if d in donors and not donors[d].get("disease_group")]
    if no_group:
        issues.append(_issue("donors_without_group", "note", f"{len(no_group)} donors have no group",
                             "Results are still made; plots by group leave them out. Add the group on the Donors screen.",
                             [{"text": _names(no_group), "fixes": []}], [_fix("Go to the Donors screen", "goto", screen="donors")]))
    codes = {g["code"] for g in project.config.get("groups") or []}
    groups = sorted({d.get("disease_group") for d in donors.values() if d.get("disease_group")} - codes)
    if codes and groups:
        issues.append(_issue("unknown_groups", "warning", f"Groups {_names(groups)} are not in Settings",
                             "They are kept as written; adding them to Settings gives them a name and a colour.",
                             fixes=[_fix("Add them to Settings", "add_groups", codes=groups)]))
    return issues


def add_image_donors(project: Project, chosen: set[str]) -> list[dict[str, str]]:
    """Donor rows to add for donors of the image list that the donor table does not have."""
    images = read_images(project)
    return [{"donor_id": d, "disease_group": next((r.get("disease_group", "") for r in images if r.get("donor_id") == d), "")} for d in sorted(chosen)]
