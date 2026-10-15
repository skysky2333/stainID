"""How far the guided setup has got (one entry per screen: Slides, Cores, Map, Donors, Checks, Run) and a rough
estimate of how long running everything that is left takes."""
from __future__ import annotations

import json
from pathlib import Path

from stainid import cache
from stainid.project import Project
from stainid.setup.checks import project_issues
from stainid.slides.dearray import slide_slug
from stainid.slides.table import find_slides
from stainid.workflows.steps import best_device

# seconds per unit of work, roughly (the step cards give the same figures in words)
SECONDS = {"dearray": 60, "export": 90, "qc": 3, "tiles": 3, "images": 3, "calibrate": 60, "neun": 5, "fields": 60, "summarize": 60}
NUCLEI_SECONDS = {"cuda": 4, "mps": 90, "cpu": 400}  # Cellpose-SAM per ~560 µm tile
TILE_UM = 560


def _rows(path: Path) -> list[dict[str, str]]:
    """A table's rows, shared and read-only (see `stainid.cache`): the overview is asked for every few seconds."""
    return cache.rows(path)


def slide_reports(project: Project) -> list[dict]:
    """Every slide in the slides table with what Find cores did on it (grid size, how it was chosen, fit) and whether
    someone has said the grid looks right."""
    folder = project.output("qc") / "grids"
    in_manifest = {r["slide_path"] for r in _rows(project.input("core_manifest"))}
    out = []
    for slide in _rows(project.input("slides_table")):
        path = Path(slide["slide_path"])
        slug = slide_slug(path)
        report_path = folder / f"{slug}.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        out.append({"slide_path": slide["slide_path"], "name": path.name, "tma": slide["tma"], "stain": slide["stain"],
                    "rows": slide.get("rows", ""), "columns": slide.get("columns", ""), "grid_source": slide.get("grid_source", ""),
                    "orientation": slide.get("orientation", ""), "image": f"{slug}.jpg" if (folder / f"{slug}.jpg").exists() else "",
                    "report": report, "checked": bool(report.get("checked")),
                    "found": str(path.resolve()) in in_manifest or bool(report) and not report.get("error"), "error": report.get("error", "")})
    return out


def mark_checked(project: Project, slide_path: str, checked: bool = True) -> None:
    path = project.output("qc") / "grids" / f"{slide_slug(Path(slide_path))}.json"
    report = json.loads(path.read_text()) if path.exists() else {"slide_path": slide_path}
    report["checked"] = checked
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1))


def estimate(project: Project, remaining: list[str]) -> dict:
    """Rough run time of the steps in `remaining` on this computer, from the number of slides, cores and tiles."""
    slides = len(_rows(project.input("slides_table")))
    manifest = _rows(project.input("core_manifest"))
    if not project.is_tma:  # image sets: one core image per image; whole slides: about 30 regions of tissue per slide
        images = len(_rows(project.input("image_list")))
        cores = len(_rows(project.input("core_images"))) or images * (30 if project.kind == "slides" else 1)
        slides, manifest = len({r["tma"] for r in _rows(project.input("core_images"))}) or max(1, images // 20), []
        biological = cores
        diameter = 1700.0 if project.kind == "slides" else 400.0
    elif manifest:
        positions = len(manifest)
        biological = sum(r.get("core_role") == "biological" for r in manifest) or sum(r.get("provisional_tissue_status") != "empty" for r in manifest)
        diameters = sorted(float(r["estimated_core_diameter_um"]) for r in manifest if r.get("estimated_core_diameter_um"))
        diameter = diameters[len(diameters) // 2] if diameters else 1500.0
    else:
        positions = slides * int(project.config["tma"]["rows"]) * int(project.config["tma"]["columns"])
        biological, diameter = int(positions * 0.9), 1500.0
    tiles = len(_rows(project.analysis_tiles())) or int(biological * max(1.0, 0.85 * (diameter / TILE_UM) ** 2))
    device = best_device()
    units = {"dearray": slides, "export": slides, "qc": biological, "tiles": biological, "images": biological, "calibrate": slides,
             "neun": tiles / 3, "fields": biological * 2 / 3, "summarize": 1}
    steps = []
    for step in remaining:
        if step == "nuclei":
            seconds = tiles * NUCLEI_SECONDS[device]
        elif step in SECONDS:
            seconds = units.get(step, 0) * SECONDS[step]
        else:
            continue
        steps.append({"id": step, "seconds": int(seconds)})
    total = sum(s["seconds"] for s in steps)
    return {"seconds": total, "steps": steps, "device": device, "slides": slides, "cores": biological, "tiles": tiles,
            "text": f"about {duration(total)}" if total else "", "slowest": max(steps, key=lambda s: s["seconds"])["id"] if steps else ""}


def count(n: int, thing: str) -> str:
    """"1 slide", "3 slides"."""
    return f"{n:,} {thing}" if n == 1 else f"{n:,} {thing}s"


def duration(seconds: float) -> str:
    if seconds < 90:
        return "a minute"
    if seconds < 3600:
        return f"{round(seconds / 60)} minutes"
    if seconds < 36 * 3600:
        hours = seconds / 3600
        return f"{hours:.0f} hours" if hours >= 3 else f"{hours:.1f} hours".replace(".0 ", " ")
    return f"{seconds / 86400:.1f} days"


def screens(project: Project, steps: list[dict] | None = None) -> list[dict]:
    """The guided setup screens with their state (done / todo / attention / optional) and one line on what was found."""
    from stainid.workflows.steps import describe

    steps = steps if steps is not None else describe(project)
    if not project.is_tma:
        return image_screens(project, steps)
    by_id = {s["id"]: s for s in steps}
    in_folder = find_slides(project.input("slides_dir"))
    slides = _rows(project.input("slides_table"))
    known = {s["slide_path"] for s in slides}
    new = [p for p in in_folder if str(p) not in known]
    reports = slide_reports(project)
    found = [r for r in reports if r["found"] and not r["error"]]
    failed = [r for r in reports if r["error"]]
    layout = _rows(project.input("tma_layout"))
    slide_tmas = sorted({s["tma"] for s in slides if s.get("tma")}, key=lambda t: (len(t), t))
    mapped = sorted({r["tma"] for r in layout} & set(slide_tmas), key=lambda t: (len(t), t))
    donors = _rows(project.input("donor_metadata"))
    issues = project_issues(project) if slides else []
    problems = [i for i in issues if i["level"] == "problem"]
    warnings = [i for i in issues if i["level"] == "warning"]
    remaining = [s for s in steps if s["stage"] in ("1 · Set up", "2 · Detect", "3 · Results") and s["command"] and s["id"] != "masks"
                 and (s["progress"]["state"] != "done" or s["progress"].get("outdated"))]

    def entry(id_: str, title: str, state: str, summary: str) -> dict:
        return {"id": id_, "title": title, "state": state, "summary": summary}

    return [
        entry("slides", "Slides", "todo" if not slides or new else "done",
              f"{count(len(new), 'new scan')} in the slides folder" if slides and new else f"{count(len(slides), 'scan')} registered" if slides
              else f"{count(len(in_folder), 'scan')} found in the slides folder" if in_folder else "No slide scans found yet"),
        entry("cores", "Cores", "attention" if failed else "done" if slides and len(found) == len(slides) else "todo",
              "Register slides first" if not slides else f"No cores found on {count(len(failed), 'slide')}" if failed else
              f"Cores found on {len(found)} of {count(len(slides), 'slide')}" + (f", {sum(r['checked'] for r in found)} checked" if found else "")),
        entry("map", "Map", "done" if slide_tmas and len(mapped) == len(slide_tmas) else "todo",
              f"Maps for {len(mapped)} of {count(len(slide_tmas), 'TMA')}" if slide_tmas else "Register slides first"),
        entry("donors", "Donors", "done" if donors else "optional", count(len(donors), "donor") if donors else "Optional: age, sex and other information per donor"),
        entry("checks", "Checks", "attention" if problems else "warning" if warnings else "done" if layout else "todo",
              ", ".join(part for part in (f"{len(problems)} to fix" if problems else "", f"{len(warnings)} to look at" if warnings else "") if part)
              if problems or warnings else "Everything fits together" if layout else "After the map"),
        entry("run", "Run", "done" if not remaining else "running" if any(by_id[s["id"]].get("job") and by_id[s["id"]]["job"]["status"] in ("running", "queued")
                                                                          for s in remaining) else "todo",
              f"{count(len(remaining), 'step')} left" if remaining else "Everything has run"),
    ]


def image_screens(project: Project, steps: list[dict]) -> list[dict]:
    """The guided setup of an image set or whole slides: Images (or Slides) -> Donors -> Checks -> Run. There is no TMA
    map and no core grid: every image, or every tissue region of a slide, is analysed like a core."""
    from stainid.imageset.table import read_images
    from stainid.imaging.settings import analysis_stain

    by_id = {s["id"]: s for s in steps}
    images = read_images(project)
    thing = "slide" if project.kind == "slides" else "image"
    analysed = [r for r in images if analysis_stain(project, r.get("stain", ""))]
    donors = _rows(project.input("donor_metadata"))
    issues = project_issues(project) if images else []
    problems = [i for i in issues if i["level"] == "problem"]
    warnings = [i for i in issues if i["level"] == "warning"]
    remaining = [s for s in steps if s["stage"] in ("1 · Set up", "2 · Detect", "3 · Results") and s["command"] and s["id"] != "masks"
                 and (s["progress"]["state"] != "done" or s["progress"].get("outdated"))]
    ready = bool(images) and all(r.get("donor_id") for r in images) and bool(analysed)
    on_images = {r["donor_id"] for r in images if r.get("donor_id")}

    def entry(id_: str, title: str, state: str, summary: str) -> dict:
        return {"id": id_, "title": title, "state": state, "summary": summary}

    listed = f"{count(len(images), thing)} from {count(len(on_images), 'donor')}" + ("" if len(analysed) == len(images) else f", {len(analysed)} analysed")
    return [
        entry("images", "Slides" if project.kind == "slides" else "Images", "done" if ready else "attention" if images else "todo",
              listed if images else f"No {thing}s yet"),
        entry("donors", "Donors", "done" if donors else "optional", count(len(donors), "donor") if donors else "Optional: group, age, sex and other details per donor"),
        entry("checks", "Checks", "attention" if problems else "warning" if warnings else "done" if images else "todo",
              ", ".join(part for part in (f"{len(problems)} to fix" if problems else "", f"{len(warnings)} to look at" if warnings else "") if part)
              if problems or warnings else "Everything fits together" if images else f"After the {thing}s"),
        entry("run", "Run", "done" if not remaining else "running" if any(by_id[s["id"]].get("job") and by_id[s["id"]]["job"]["status"] in ("running", "queued")
                                                                          for s in remaining) else "todo",
              f"{count(len(remaining), 'step')} left" if remaining else "Everything has run"),
    ]
