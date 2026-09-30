"""The workflow as data: every step says what it needs, what it makes, where it puts it, and how far it has got.

The CLI, the background job runner and the web app all read this registry, so a step is described in one place.
"""
from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from stainid.project import Project
from stainid.slides.table import read_slides
from stainid.tables import read_csv

STAINS = ["NeuN", "6E10", "AT8"]
GPU = "mps" if sys.platform == "darwin" and platform.machine() == "arm64" else "cpu"


@dataclass(frozen=True)
class Option:
    key: str
    label: str
    kind: str
    default: object
    help: str = ""
    choices: tuple = ()
    advanced: bool = False


@dataclass(frozen=True)
class Resource:
    id: str
    label: str
    description: str
    path: Callable[[Project], Path]
    ready: Callable[[Project], bool] | None = None
    download: str = ""
    template: str = ""
    setting: str = ""
    format: str = ""


@dataclass(frozen=True)
class Step:
    id: str
    stage: str
    title: str
    summary: str
    details: str
    needs: tuple[str, ...]
    produces: tuple[str, ...]
    command: tuple[str, ...] | None
    options: tuple[Option, ...] = ()
    heavy: bool = False
    view: str | None = None
    duration: str = ""
    status: Callable[[Project], dict] = field(default=lambda project: {}, repr=False)
    uses: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()

    def reads(self) -> set[str]:
        return set(self.needs) | set(self.uses)


def _rows(path: Path) -> list[dict[str, str]]:
    return read_csv(path) if path.exists() else []


def _state(done: int, total: int, unit: str, note: str = "", outdated: bool = False) -> dict:
    state = "done" if total and done >= total else "partial" if done else "todo"
    return {"state": state, "done": done, "total": total, "unit": unit, "note": note, "outdated": outdated}


def _made_with_other_model(project: Project, output: str, recorded: Callable[[dict], dict], current: dict) -> bool:
    path = project.output(output) / "provenance.json"
    return path.exists() and recorded(json.loads(path.read_text())) != current


def _tiles(project: Project, stains: set[str]) -> list[dict[str, str]]:
    return [t for t in _rows(project.input("tile_manifest")) if t["stain"] in stains]


def _column_filled(column: str) -> Callable[[Project], bool]:
    def ready(project: Project) -> bool:
        rows = _rows(project.input("core_manifest"))
        return bool(rows) and any(r.get(column) for r in rows)
    return ready


def _slides_status(project: Project) -> dict:
    rows = read_slides(project.input("slides_table"))
    ready = sum(bool(r["tma"]) and r["stain"] in STAINS for r in rows)
    return _state(ready, len(rows), "slides", "" if rows else "No slides registered yet")


def _dearray_status(project: Project) -> dict:
    slides = {str(Path(s["slide_path"]).resolve()) for s in read_slides(project.input("slides_table"))}
    found = {r["slide_path"] for r in _rows(project.input("core_manifest"))}
    return _state(len(slides & found), len(slides), "slides")


def _manifest_status(column: str, unit: str = "core images", only: Callable[[dict], bool] = lambda r: True):
    def status(project: Project) -> dict:
        rows = [r for r in _rows(project.input("core_manifest")) if only(r)]
        return _state(sum(bool(r.get(column)) for r in rows), len(rows), unit)
    return status


def _select_status(project: Project) -> dict:
    tiles = _rows(project.input("tile_manifest"))
    cores = {r["core_id"] for r in tiles}
    return _state(len(tiles), len(tiles), "fields", f"{len(cores)} cores") if tiles else _state(0, 0, "fields")


def _calibration_status(project: Project) -> dict:
    wanted = {(t["tma"], t["stain"]) for t in _rows(project.input("tile_manifest"))}
    have = {(r["tma"], r["stain"]) for r in _rows(project.input("calibration"))}
    return _state(len(wanted & have), len(wanted), "slides")


def _count_files(folder: Path, names: list[str]) -> int:
    existing = {p.name for p in folder.iterdir()} if folder.exists() else set()
    return sum(n in existing for n in names)


def _nuclei_status(project: Project) -> dict:
    tiles = _tiles(project, {"AT8"})
    return _state(_count_files(project.output("nuclei"), [f"{t['tile_id']}.npz" for t in tiles]), len(tiles), "AT8 fields",
                  "AT8 needs nuclei; 6E10 uses them for plaque-niche features if present")


def _neun_status(project: Project) -> dict:
    tiles = _tiles(project, {"NeuN"})
    outdated = _made_with_other_model(project, "neun", lambda r: {"neun": str(Path(r["neun_model_bundle"]).resolve())},
                                      {"neun": str(project.model("neun").resolve())})
    return _state(_count_files(project.output("neun") / "tiles", [f"{t['tile_id']}_features.csv" for t in tiles]), len(tiles), "fields", outdated=outdated)


def _fields_status(project: Project) -> dict:
    keys = sorted({f"{t['core_id']}_{t['stain']}_features.csv" for t in _tiles(project, {"6E10", "AT8"})})
    outdated = _made_with_other_model(project, "fields", lambda r: r, {k: project.relative(project.model(k)) for k in ("amyloid", "tau")})
    return _state(_count_files(project.output("fields") / "parts", keys), len(keys), "core images", outdated=outdated)


def _masks_status(project: Project) -> dict:
    keys = {(t["core_id"], t["stain"]) for t in _tiles(project, set(STAINS))}
    done = sum(_count_files(project.output("masks") / s / "parts", [f"{c}_{s}_objects.csv" for c, st in keys if st == s]) for s in STAINS)
    return _state(done, len(keys), "core images")


def _summary_status(project: Project) -> dict:
    tables = [project.output("tables") / f"results_{level}.csv" for level in ("donor_region", "core")]
    made = [len(_rows(t)) for t in tables]
    note = " · ".join(f"{n} {unit}" for n, unit in zip(made, ("donor-regions", "cores")) if n)
    return _state(1 if any(made) else 0, 1, "table", note)


def _training_status(project: Project) -> dict:
    root = project.output("reviews") / "training"
    sets = [json.loads(m.read_text()) for m in root.glob("*/meta.json")] if root.exists() else []
    return _state(len(sets), len(sets), "training sets")


def _trained_status(project: Project) -> dict:
    from stainid.review.store import read_labels
    from stainid.training.sets import POSITIVE, SKIP

    root = project.output("trained_models")
    count = len(list(root.glob("*/bundle.joblib"))) if root.exists() else 0
    sets = project.output("reviews") / "training"
    counts = {stain: [0, 0] for stain in POSITIVE}
    for meta in sets.glob("*/meta.json") if sets.exists() else []:
        stain = json.loads(meta.read_text())["stain"]
        labels = read_labels(meta.parent)
        for label in labels.label if not labels.empty else []:
            if label not in SKIP:
                counts[stain][label not in POSITIVE[stain]] += 1
    note = " · ".join(f"{s}: {p} positive / {n} negative labels" for s, (p, n) in counts.items() if p or n)
    return _state(count, count, "trained models", note or "No labels yet. Training needs at least 10 positive and 10 negative labels per stain")


RESOURCES = {r.id: r for r in (
    Resource("slides_dir", "Slide scans folder", "The folder holding your whole-slide scans (.vsi, .svs, .ndpi, …).", lambda p: p.input("slides_dir"),
             setting="inputs.slides_dir",
             format="A folder with one whole-slide scan per TMA and stain, for example:\n  TMA 1 NeuN.vsi (+ its _TMA 1 NeuN_ folder)\n  TMA 1 AT8.vsi\n"
                    "  TMA 2 6E10.svs\nOlympus .vsi, Aperio .svs, Hamamatsu .ndpi, Leica .scn, Zeiss .czi and pyramidal .tif all work. "
                    "Putting the TMA number and the stain in each file name lets stainID fill in the slides table for you."),
    Resource("slides_table", "Slides table", "Which scan is which TMA and stain.", lambda p: p.input("slides_table")),
    Resource("core_manifest", "Core table", "One row per core position on each slide: where it is, which donor it belongs to, its QC.",
             lambda p: p.input("core_manifest")),
    Resource("core_groups", "Donor & group per core", "The TMA map attached to the core table.", lambda p: p.input("core_manifest"),
             _column_filled("core_role")),
    Resource("core_quality", "Core quality measures", "Tissue coverage and focus per core, in the core table.", lambda p: p.input("core_manifest"),
             _column_filled("tissue_status")),
    Resource("grid_images", "Grid check images", "One picture per slide with the fitted core grid drawn on it.", lambda p: p.output("qc") / "grids"),
    Resource("tma_layout", "TMA map", "Which donor, brain region and diagnostic group sits at each core position.", lambda p: p.input("tma_layout"),
             template="tma_layout", setting="inputs.tma_layout",
             format="A CSV table (save from Excel as CSV) with one row per core position:\n"
                    "  tma,core_label,donor_id,region,disease_group\n  1,A-1,,,\n  1,B-1,1141,frontal,AD\n  1,B-2,1141,occipital,AD\n  1,C-1,0698,frontal,CT\n"
                    "core_label is column letter + row number. Leave donor_id empty for orientation or control cores. "
                    "Optional columns: cerad, braak, sample_region_id. The template lists every position already."),
    Resource("core_images", "Core images", "Every core cut out of the slide at full resolution (PNG).", lambda p: p.input("core_manifest").parent / "cores"),
    Resource("core_qc_images", "Core QC images", "Tissue and focus overlays for every core.", lambda p: p.input("core_manifest").parent / "qc" / "cores"),
    Resource("tile_manifest", "Analysis fields", "The fields (about 560 µm squares) measured in every core.", lambda p: p.input("tile_manifest")),
    Resource("calibration", "Stain thresholds", "One brown-stain (DAB) threshold per slide.", lambda p: p.input("calibration")),
    Resource("nuclei", "Nuclei", "Every cell nucleus found by Cellpose, one file per field.", lambda p: p.output("nuclei")),
    Resource("neun_results", "NeuN results", "Neuron detections and per-field counts.", lambda p: p.output("neun")),
    Resource("field_results", "6E10 / AT8 results", "Plaque and tau detections and per-field measurements.", lambda p: p.output("fields")),
    Resource("mask_results", "Object outlines", "An outline and shape measurements for every detected object.", lambda p: p.output("masks")),
    Resource("results", "Results table", "One row per donor and brain region with every measurement (results_donor_region.csv).",
             lambda p: p.output("tables") / "results_donor_region.csv"),
    Resource("model_neun", "NeuN model", "Decides which candidate objects are neurons.", lambda p: p.model("neun"),
             setting="models.neun", format="A model bundle file (.joblib) made by stainID's training. Train one on the Models page from objects you label, "
                                        "or choose a bundle someone shared with you (for example the one used for a published study)."),
    Resource("model_amyloid", "6E10 model", "Decides which deposits are plaques and whether they are compact or diffuse.", lambda p: p.model("amyloid"),
             setting="models.amyloid", format="A model bundle file (.joblib) made by stainID's training. Train one on the Models page from objects you label, "
                                        "or choose a bundle someone shared with you (for example the one used for a published study)."),
    Resource("model_tau", "AT8 model", "Decides which candidates are tau+ neurons.", lambda p: p.model("tau"),
             setting="models.tau", format="A model bundle file (.joblib) made by stainID's training. Train one on the Models page from objects you label, "
                                        "or choose a bundle someone shared with you (for example the one used for a published study)."),
    Resource("model_cellpose", "Cellpose-SAM weights", "Finds cells and nuclei.", lambda p: p.model("cellpose"), download="cellpose"),
    Resource("model_sam", "Segment Anything weights", "Draws object outlines.", lambda p: p.model("sam"), download="sam"),
    Resource("model_phikon", "Phikon weights", "Image features used by the 6E10 model.", lambda p: p.model("huggingface_home"), download="huggingface_home"),
    Resource("training_sets", "Training sets", "Candidate objects you label to train a model.", lambda p: p.output("reviews") / "training"),
    Resource("trained_models", "Trained models", "Models you trained, with their accuracy reports.", lambda p: p.output("trained_models")),
)}

SHARD = (Option("shard_index", "Part", "number", 0, "Split a long run over several processes: this process runs part N …", advanced=True),
         Option("shard_count", "of parts", "number", 1, "… of this many parts (1 = everything in one go).", advanced=True))
FRESH = Option("fresh", "Start over", "bool", False, "Move existing results aside (nothing is deleted) and run everything again. "
               "Use this after switching to a different model.", advanced=True)
DEVICE = Option("device", "Processor", "select", GPU, "mps = Apple GPU (Apple-silicon Macs), cuda = NVIDIA GPU, cpu works everywhere but is slower.",
                ("cpu", "mps", "cuda"))

STEPS = (
    Step("slides", "1 · Set up", "Register slides", "Tell stainID which scan is which TMA and stain.",
         "Pick the folder with your whole-slide scans. stainID lists them and guesses the TMA number and stain from each file name; "
         "check the guesses and fix any that are wrong, then save.", ("slides_dir",), ("slides_table",), None, status=_slides_status),
    Step("dearray", "1 · Set up", "Find cores", "Locate every core on each slide by fitting the TMA grid.",
         "Reads a small preview of each slide, fits a rows × columns grid (set in Settings) and re-centres each core on its tissue. "
         "Check the grid pictures afterwards: every circle should sit on a core. The first time, stainID downloads the Bio-Formats "
         "slide reader (about 60 MB, needs internet); later runs start straight away.",
         ("slides_table",), ("core_manifest", "grid_images"), ("dearray",),
         (Option("redo", "Redo all slides", "bool", False, "Fit the grid again on slides that were already done.", advanced=True),),
         heavy=True, view="/setup/grids", duration="about 1 minute per slide", status=_dearray_status, writes=("core_manifest",)),
    Step("layout", "1 · Set up", "Attach TMA map", "Say which donor, region and group sits at each core position.",
         "Upload a spreadsheet (CSV) with one row per core position: tma, core_label (e.g. B-2), donor_id, region and disease_group. "
         "Leave donor_id empty for orientation or control cores. Download the template to start.",
         ("core_manifest", "tma_layout"), ("core_groups",), ("layout",), status=_manifest_status("core_role", "core positions"),
         writes=("core_manifest",)),
    Step("export", "1 · Set up", "Export cores", "Cut every core out of the slide at full resolution.",
         "Writes one PNG per core and stain. Finished cores are skipped, so this can be stopped and restarted.",
         ("core_manifest",), ("core_images",), ("export",),
         (Option("workers", "Parallel writers", "number", 2, "How many images are written at once.", advanced=True),
          Option("overwrite", "Export again", "bool", False, "Re-export cores that already exist.", advanced=True)),
         heavy=True, view="/cohort", duration="about 1–2 minutes per slide",
         status=_manifest_status("export_status", only=lambda r: r.get("core_role") == "biological"), writes=("core_manifest",)),
    Step("qc", "1 · Set up", "Check core quality", "Measure tissue coverage, fragments and focus of every core.",
         "Flags cores that are mostly empty, torn or out of focus so they can be reviewed. Field selection only uses tissue and focus, never the stain.",
         ("core_images",), ("core_quality", "core_qc_images"), ("qc",),
         (Option("workers", "Parallel workers", "number", 2, advanced=True), Option("overwrite", "Measure again", "bool", False, advanced=True)),
         heavy=True, view="/cohort", duration="a few seconds per core",
         status=_manifest_status("tissue_status", only=lambda r: r.get("export_status") == "complete"), writes=("core_manifest",)),
    Step("select", "1 · Set up", "Choose analysis fields", "Pick evenly spread ~560 µm fields inside every core.",
         "Scores candidate windows for tissue and focus, then spreads the chosen fields across the core. The first few fields per core are "
         "analysed; the rest are kept for stability checks.",
         ("core_groups", "core_quality"), ("tile_manifest",), ("select",),
         (Option("fields_per_core", "Fields per core", "number", 8, "How many fields to choose in each core."),
          Option("primary", "Fields analysed", "number", 4, "How many of them are measured (the first N)."),
          Option("field_size", "Field size (pixels)", "number", 2048, "Side of each square field; 2048 px is about 560 µm at 0.27 µm/px. "
                 "Use a smaller size for small cores.", advanced=True)),
         view="/cohort", duration="a few minutes", status=_select_status),
    Step("calibrate", "1 · Set up", "Calibrate stain thresholds", "Find the brown-stain (DAB) cut-off for every slide.",
         "Pools tissue pixels from all cores of a slide and sets one threshold per slide, blind to diagnosis. Everything downstream is measured "
         "relative to it, so slides stained a little darker or lighter stay comparable.",
         ("tile_manifest",), ("calibration",), ("calibrate",), view="/calibration", duration="a few minutes", status=_calibration_status),
    Step("nuclei", "2 · Detect", "Find nuclei", "Outline every cell nucleus with Cellpose-SAM.",
         "Needed before the AT8 step (tau+ neurons are found around nuclei); also used for 6E10 plaque-neighbourhood features.",
         ("tile_manifest", "model_cellpose"), ("nuclei",), ("nuclei",),
         (Option("stain", "Stains", "stains", ["AT8", "6E10"], "NeuN detection finds its own cells, so it does not need this step.", ("AT8", "6E10")),
          DEVICE, Option("batch_size", "Batch size", "number", 8, advanced=True), *SHARD),
         heavy=True, duration="about 20 s per field on a Mac GPU", status=_nuclei_status),
    Step("neun", "2 · Detect", "Detect NeuN neurons", "Find and classify NeuN-stained neuronal profiles.",
         "Candidates come from brown-stain contours and Cellpose cells; the NeuN model scores each one with its shape, stain and 55 µm "
         "neighbourhood.", ("tile_manifest", "calibration", "model_neun", "model_cellpose"), ("neun_results",), ("neun",),
         (Option("device", "Processor", "select", "cpu", "cpu reproduces published results exactly; mps / cuda (GPU) are faster but Cellpose "
                 "outlines can differ slightly.", ("cpu", "mps", "cuda")),
          Option("threads", "CPU threads", "number", 8, advanced=True), FRESH, *SHARD),
         heavy=True, view="/cohort", duration="about 1 minute per field", status=_neun_status),
    Step("fields", "2 · Detect", "Detect plaques and tau", "6E10 plaques (compact / diffuse) and AT8 tau+ neurons and threads.",
         "6E10: segments deposits, keeps real plaques with the 6E10 model and sorts them into compact and diffuse. AT8: finds tau+ neurons "
         "around nuclei and traces neuropil threads, so run Find nuclei for AT8 first (AT8 cores without nuclei are skipped).",
         ("tile_manifest", "calibration", "model_amyloid", "model_tau"), ("field_results",), ("fields",),
         (Option("stain", "Stains", "stains", ["6E10", "AT8"], choices=("6E10", "AT8")), FRESH, *SHARD),
         heavy=True, view="/cohort", duration="about 1 minute per core", status=_fields_status, uses=("nuclei", "model_phikon")),
    Step("masks", "2 · Detect", "Outline objects", "Draw an exact outline around every detected object with Segment Anything.",
         "Adds precise size and shape measurements (area, roundness, dense cores). Optional: the main counts do not need it.",
         ("model_sam",), ("mask_results",), ("masks",),
         (Option("stain", "Stains", "stains", ["NeuN", "6E10", "AT8"], "Stains without detections yet are skipped.", ("NeuN", "6E10", "AT8")),
          DEVICE, FRESH, *SHARD),
         heavy=True, view="/cohort", duration="about 1 minute per core", status=_masks_status, uses=("neun_results", "field_results")),
    Step("summarize", "3 · Results", "Make results tables", "Combine everything into one row per donor and brain region.",
         "Sums counts and areas over fields and replicate cores before dividing, so every density is area-weighted. Writes "
         "results_donor_region.csv (and per-core tables) that open in Excel, R or Python.",
         (), ("results",), ("summarize",),
         (Option("level", "One row per", "select", "sample_region_id", "donor-region (recommended) or single core.", ("sample_region_id", "core_id")),),
         view="/results", duration="about a minute", status=_summary_status, uses=("neun_results", "field_results", "mask_results")),
    Step("download_models", "Models", "Download public models", "Fetch the Cellpose-SAM, Segment Anything and Phikon weights.",
         "These models are published by their authors and are the same for every study. The NeuN, 6E10 and AT8 models are specific to "
         "your staining and are trained on the Models page instead.", (), ("model_cellpose", "model_sam", "model_phikon"), ("download-models",),
         (Option("which", "Models", "stains", ["cellpose", "sam", "huggingface_home"], choices=("cellpose", "sam", "huggingface_home")),),
         heavy=True, view="/models", duration="a few minutes (about 2 GB)", status=lambda p: _state(
             sum(p.model(k).exists() for k in ("cellpose", "sam", "huggingface_home")), 3, "models")),
    Step("training_set", "Models", "Create a training set", "Pick candidate objects to label for training a model.",
         "Chooses fields spread across TMAs and groups, finds every candidate object exactly as the detection step does, and stores their "
         "measurements. You then label them blind on the Label page.",
         ("tile_manifest", "calibration"), ("training_sets",), ("training-set",),
         (Option("stain", "Stain", "select", "AT8", choices=tuple(STAINS)), Option("name", "Name", "text", ""),
          Option("fields", "Fields", "number", 12, "How many fields to sample."),
          Option("per_field", "Objects per field", "number", 10),
          Option("strategy", "Which objects", "select", "uncertain", "uncertain: half the objects are ones the current model is unsure about.",
                 ("uncertain", "random")),
          Option("enrich", "Prefer fields with detections", "bool", True, "Plaques and tau+ neurons are rare; sample fields that have some.")),
         heavy=True, view="/models", duration="a few minutes", status=_training_status, uses=("nuclei", "model_cellpose", "model_phikon")),
    Step("train", "Models", "Train a model", "Train a new model from your labelled training sets.",
         "Fits the same kind of model the pipeline uses, checks it by leaving one TMA out at a time, and compares it with the current model on "
         "your labels. The new model is only used after you choose it on the Models page.",
         ("training_sets",), ("trained_models",), ("train",), (Option("stain", "Stain", "select", "AT8", choices=tuple(STAINS)),),
         view="/models", duration="under a minute", status=_trained_status),
)
BY_ID = {s.id: s for s in STEPS}


def to_argv(step_id: str, options: dict) -> list[str]:
    argv = list(BY_ID[step_id].command)
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            if value != _default(step_id, key):
                argv.append(flag if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, list):
            for item in value:
                argv += [flag, str(item)]
        else:
            argv += [flag, str(value)]
    return argv


def _default(step_id: str, key: str):
    return next((o.default for o in BY_ID[step_id].options if o.key == key), False)


def wait_reason(earlier: Step, later: Step) -> str:
    """Why a later step must wait for an earlier running / waiting one ("" = no need): it makes something the later step reads,
    or both write the same file (e.g. the core table)."""
    made = set(earlier.produces) & later.reads()
    if made:
        return f"Waits for “{earlier.title}” to finish (needs: {', '.join(RESOURCES[r].label for r in sorted(made))})"
    shared = set(earlier.writes + earlier.produces) & set(later.writes + later.produces)
    return f"Waits for “{earlier.title}” to finish (both write: {RESOURCES[sorted(shared)[0]].label})" if shared else ""


def depends_on(step: Step) -> list[str]:
    """Steps that make what this step needs or uses: the arrows of the pipeline map."""
    return [s.id for s in STEPS if s.id != step.id and set(s.produces) & step.reads()]


def resource_info(project: Project, resource_id: str) -> dict:
    resource = RESOURCES[resource_id]
    path = resource.path(project)
    exists = resource.ready(project) if resource.ready else path.exists() and (path.is_file() or any(path.iterdir()))
    maker = next((s for s in STEPS if resource_id in s.produces), None) if not resource.download else None
    return {"id": resource.id, "label": resource.label, "description": resource.description, "path": project.relative(path), "exists": exists,
            "made_by": {"id": maker.id, "title": maker.title} if maker else None, "download": resource.download,
            "template": resource.template, "setting": resource.setting, "format": resource.format}


def describe(project: Project) -> list[dict]:
    out = []
    for step in STEPS:
        info = {k: v for k, v in asdict(step).items() if k != "status"}
        info["options"] = [asdict(o) for o in step.options]
        info["needs"] = [resource_info(project, r) for r in step.needs]
        info["uses"] = [resource_info(project, r) for r in step.uses]
        info["produces"] = [resource_info(project, r) for r in step.produces]
        info["after"] = depends_on(step)
        info["progress"] = step.status(project)
        out.append(info)
    return out
