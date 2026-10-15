"""The workflow as data: every step says what it needs, what it makes, where it puts it, and how far it has got.

The CLI, the background job runner and the web app all read this registry, so a step is described in one place.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from stainid import cache
from stainid.project import Project

STAINS = ["NeuN", "6E10", "AT8"]


def best_device() -> str:
    """The processor for the deep models: the Apple GPU on Apple-silicon Macs, an NVIDIA GPU when one is installed, else the CPU."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mps"
    return "cuda" if shutil.which("nvidia-smi") else "cpu"


GPU = best_device()


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
    waits_for_uses: bool = True  # False: start while a step that makes a `uses` input runs (takes what is there already)

    def reads(self) -> set[str]:
        return set(self.needs) | set(self.uses)

    def waits_on(self) -> set[str]:
        """What a running step must finish making before this one starts."""
        return self.reads() if self.waits_for_uses else set(self.needs)


def _rows(path: Path, columns: tuple[str, ...] | None = None) -> list[dict[str, str]]:
    """A table's rows (shared, read-only: see `stainid.cache`); only `columns` of them for the big tile tables."""
    return cache.rows(path, columns)


TILE_COLUMNS = ("tile_id", "core_id", "tma", "stain")


def _state(done: int, total: int, unit: str, note: str = "", outdated: bool = False) -> dict:
    state = "done" if total and done >= total else "partial" if done else "todo"
    return {"state": state, "done": done, "total": total, "unit": unit, "note": note, "outdated": outdated}


def _made_with_other_model(project: Project, output: str, recorded: Callable[[dict], dict], current: dict) -> bool:
    path = project.output(output) / "provenance.json"
    return path.exists() and recorded(json.loads(path.read_text())) != current


def _tiles(project: Project, stains: set[str]) -> list[dict[str, str]]:
    return [t for t in _rows(project.analysis_tiles(), TILE_COLUMNS) if t["stain"] in stains]


def _expected(project: Project, stains: set[str], name: Callable[[dict], str]) -> frozenset[str]:
    """The output file names a step makes for the analysis tiles of these stains (`name` of each tile), kept until the
    tile table changes. `name` must be a module-level function, so the same names share one entry."""
    path = project.analysis_tiles()
    return cache.memo(("expected", str(path), tuple(sorted(stains)), name.__qualname__), [path],
                      lambda: frozenset(name(t) for t in _tiles(project, stains)))


def _whole_core_status(project: Project) -> dict:
    tiles = _rows(project.input("whole_core_manifest"), TILE_COLUMNS)
    cores = {(r["core_id"], r["stain"]) for r in tiles}
    return _state(len(tiles), len(tiles), "tiles", f"{len(cores)} core images") if tiles else _state(0, 0, "tiles")


def _column_filled(column: str) -> Callable[[Project], bool]:
    def ready(project: Project) -> bool:
        rows = _rows(project.input("core_manifest"))
        return bool(rows) and any(r.get(column) for r in rows)
    return ready


def _slides_status(project: Project) -> dict:
    rows = _rows(project.input("slides_table"))
    ready = sum(bool(r["tma"]) and r["stain"] in STAINS for r in rows)
    return _state(ready, len(rows), "slides", "" if rows else "No slides registered yet")


def _dearray_status(project: Project) -> dict:
    slides = {str(Path(s["slide_path"]).resolve()) for s in _rows(project.input("slides_table"))}
    found = {r["slide_path"] for r in _rows(project.input("core_manifest"))}
    return _state(len(slides & found), len(slides), "slides")


def _manifest_status(column: str, unit: str = "core images", only: Callable[[dict], bool] = lambda r: True):
    def status(project: Project) -> dict:
        rows = [r for r in _rows(project.input("core_manifest")) if only(r)]
        return _state(sum(bool(r.get(column)) for r in rows), len(rows), unit)
    return status


def _calibration_status(project: Project) -> dict:
    wanted = {(t["tma"], t["stain"]) for t in _rows(project.analysis_tiles(), TILE_COLUMNS)}
    have = {(r["tma"], r["stain"]) for r in _rows(project.input("calibration"))}
    return _state(len(wanted & have), len(wanted), "slides")


def _count_files(folder: Path, names: frozenset[str]) -> int:
    return len(cache.names(folder) & names)


def _nuclei_file(tile: dict) -> str:
    return f"{tile['tile_id']}.npz"


def _neun_file(tile: dict) -> str:
    return f"{tile['tile_id']}_features.csv"


def _field_file(tile: dict) -> str:
    return f"{tile['core_id']}_{tile['stain']}_features.csv"


def _mask_file(tile: dict) -> str:
    return f"{tile['core_id']}_{tile['stain']}_objects.csv"


def _nuclei_status(project: Project) -> dict:
    names = _expected(project, set(STAINS), _nuclei_file)
    return _state(_count_files(project.output("nuclei"), names), len(names), "tiles",
                  "NeuN and AT8 detection need these outlines; 6E10 uses them for plaque-niche features")


def _neun_status(project: Project) -> dict:
    names = _expected(project, {"NeuN"}, _neun_file)
    outdated = _made_with_other_model(project, "neun", lambda r: {"neun": str(Path(r["neun_model_bundle"]).resolve())},
                                      {"neun": str(project.model("neun").resolve())})
    return _state(_count_files(project.output("neun") / "tiles", names), len(names), "tiles", outdated=outdated)


def _fields_status(project: Project) -> dict:
    names = _expected(project, {"6E10", "AT8"}, _field_file)
    outdated = _made_with_other_model(project, "fields", lambda r: r, {k: project.relative(project.model(k)) for k in ("amyloid", "tau")})
    return _state(_count_files(project.output("fields") / "parts", names), len(names), "core images", outdated=outdated)


def _masks_status(project: Project) -> dict:
    per_stain = {s: _expected(project, {s}, _mask_file) for s in STAINS}
    done = sum(_count_files(project.output("masks") / s / "parts", names) for s, names in per_stain.items())
    return _state(done, sum(map(len, per_stain.values())), "core images")


def _summary_status(project: Project) -> dict:
    tables = [project.output("tables") / f"results_{level}.csv" for level in ("donor_region", "core")]
    made = [len(_rows(t)) for t in tables]
    note = " · ".join(f"{n} {unit}" for n, unit in zip(made, ("donor-regions", "cores")) if n)
    return _state(1 if any(made) else 0, 1, "table", note)


def _labels_status(project: Project) -> dict:
    from stainid.labels import pool

    prepared = {stain: len(cache.memo(("prepared", str(pool.folder(project, stain))), [pool.folder(project, stain)],
                                      lambda stain=stain: pool.prepared(project, stain), folders=True)) for stain in STAINS}
    total = sum(prepared.values())
    return _state(total, total, "fields prepared", " · ".join(f"{s}: {n}" for s, n in prepared.items() if n))


def _trained_status(project: Project) -> dict:
    from stainid.labels import store

    root = project.output("trained_models")
    count = len(list(root.glob("*/bundle.joblib"))) if root.exists() else 0
    note = cache.memo(("label counts", str(project.output("reviews"))), [store.log_path(project.output("reviews"))],
                      lambda: _label_counts(project.output("reviews")))
    return _state(count, count, "trained models", note or "No labels yet. Training needs at least 10 positive and 10 negative labels per stain")


def _label_counts(reviews: Path) -> str:
    from stainid.labels import store

    labels = store.current(reviews)
    counts = {stain: store.consensus(labels, stain).positive.value_counts() for stain in STAINS}
    return " · ".join(f"{s}: {int(c.get(True, 0))} positive / {int(c.get(False, 0))} negative labels" for s, c in counts.items() if len(c))


def _has_content(path: Path) -> bool:
    """A file, or a folder with something in it (kept until the folder changes: output folders hold thousands)."""
    def look() -> bool:
        if path.is_file():
            return True
        try:
            with os.scandir(path) as entries:
                return next(entries, None) is not None
        except OSError:
            return False

    return cache.memo(("has content", str(path)), [path], look, folders=True)


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
    Resource("tiles", "Tiles", "Every ~560 µm tile of tissue in each core: detection runs on these, and labelling fields are drawn "
             "from them.", lambda p: p.analysis_tiles()),
    Resource("calibration", "Stain thresholds", "One brown-stain (DAB) threshold per slide.", lambda p: p.input("calibration")),
    Resource("nuclei", "Nuclei", "Every cell nucleus found by Cellpose, one file per tile.", lambda p: p.output("nuclei")),
    Resource("neun_results", "NeuN results", "Neuron detections and per-tile counts.", lambda p: p.output("neun")),
    Resource("field_results", "6E10 / AT8 results", "Plaque and tau detections and per-tile measurements.", lambda p: p.output("fields")),
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
    Resource("label_store", "Labels", "Every label people made (and imported earlier label sets).", lambda p: p.output("reviews") / "labels" / "labels.jsonl"),
    Resource("candidate_pool", "Candidate objects", "The objects of each prepared field, with their measurements and the AI's call.",
             lambda p: p.output("reviews") / "pool"),
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
         "Reads a small preview of each slide, fits the grid of cores and re-centres each core on its tissue. A slide without its own "
         "rows and columns in the slides table gets the grid size that fits best (the size in Settings unless another fits clearly "
         "better), and the size it chose is written to the slides table. Check the grid pictures afterwards: every circle should sit "
         "on a core. The first time, stainID downloads the Bio-Formats slide reader (about 60 MB, needs internet); later runs start "
         "straight away.",
         ("slides_table",), ("core_manifest", "grid_images"), ("dearray",),
         (Option("redo", "Redo all slides", "bool", False, "Fit the grid again on slides that were already done.", advanced=True),
          Option("slide", "Only these slides", "text", "", "File names of slides to find again, separated by commas (the others are left as they are).",
                 advanced=True)),
         heavy=True, view="/setup/cores", duration="about 1 minute per slide", status=_dearray_status, writes=("core_manifest", "slides_table")),
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
         "Flags cores that are mostly empty, torn or out of focus so they can be reviewed. Only tissue and focus are measured, never the stain.",
         ("core_images",), ("core_quality", "core_qc_images"), ("qc",),
         (Option("workers", "Parallel workers", "number", 2, advanced=True), Option("overwrite", "Measure again", "bool", False, advanced=True)),
         heavy=True, view="/cohort", duration="a few seconds per core",
         status=_manifest_status("tissue_status", only=lambda r: r.get("export_status") == "complete"), writes=("core_manifest",)),
    Step("tiles", "1 · Set up", "Cut cores into tiles", "Cover each core's tissue with ~560 µm tiles; detection looks at every one.",
         "Lays a grid of square tiles over each core image and keeps every square that holds tissue, including the narrower squares at the "
         "image edges, so detection covers the whole core without overlap.", ("core_groups", "core_quality"), ("tiles",), ("tiles",),
         (Option("tile_size", "Tile size (pixels)", "number", 2048, "Side of each square tile; 2048 px is about 560 µm at 0.27 µm/px.",
                 advanced=True),),
         heavy=True, view="/cohort", duration="a few seconds per core", status=_whole_core_status),
    Step("calibrate", "1 · Set up", "Calibrate stain thresholds", "Find the brown-stain (DAB) cut-off for every slide.",
         "Pools tissue pixels from every tile of every core of a slide and sets one threshold per slide, blind to diagnosis. Everything "
         "downstream is measured relative to it, so slides stained a little darker or lighter stay comparable. Slides that already have a "
         "threshold keep it.", ("tiles",), ("calibration",), ("calibrate",),
         (Option("redo", "Calibrate again", "bool", False, "Work out the thresholds again for slides that already have one. The detection "
                 "steps must then run again with Start over.", advanced=True),),
         heavy=True, view="/calibration", duration="about a minute per slide", status=_calibration_status),
    Step("nuclei", "2 · Detect", "Find nuclei", "Outline every cell with Cellpose-SAM, for all stains.",
         "The slow step (about 90 s per tile on an Apple GPU, about 4 s on an NVIDIA A100-class GPU), and the only one that needs a GPU. "
         "NeuN and AT8 detection read these outlines; 6E10 uses them for plaque-neighbourhood features.",
         ("tiles", "model_cellpose"), ("nuclei",), ("nuclei",),
         (Option("stain", "Stains", "stains", ["NeuN", "AT8", "6E10"], choices=("NeuN", "AT8", "6E10")),
          DEVICE, Option("batch_size", "Batch size", "number", 8, advanced=True),
          Option("workers", "Tiles at once", "number", 2, "Two tiles in flight keep the GPU busy while the other one's CPU steps run.",
                 advanced=True), *SHARD),
         heavy=True, duration="about 90 s per tile on a Mac GPU, 4 s with CUDA", status=_nuclei_status),
    Step("neun", "2 · Detect", "Detect NeuN neurons", "Find and classify NeuN-stained neuronal profiles.",
         "Candidates come from brown-stain contours and the Cellpose cells of Find nuclei (computed here for tiles that have none yet); "
         "the NeuN model scores each one with its shape, stain and 55 µm neighbourhood.",
         ("tiles", "calibration", "model_neun", "model_cellpose"), ("neun_results",), ("neun",),
         (Option("device", "Processor", "select", "cpu", "Only matters for tiles without Find nuclei outlines. cpu reproduces published results exactly; mps / cuda (GPU) are faster but Cellpose "
                 "outlines can differ slightly.", ("cpu", "mps", "cuda")),
          Option("threads", "CPU threads", "number", 8, advanced=True), FRESH, *SHARD),
         heavy=True, view="/cohort", duration="seconds per tile once Find nuclei has run", status=_neun_status, uses=("nuclei",)),
    Step("fields", "2 · Detect", "Detect plaques and tau", "6E10 plaques (compact / diffuse) and AT8 tau+ neurons and threads.",
         "6E10: segments deposits, keeps real plaques with the 6E10 model and sorts them into compact and diffuse. AT8: finds tau+ neurons "
         "around nuclei and traces neuropil threads, so run Find nuclei for AT8 first (AT8 cores without nuclei are skipped).",
         ("tiles", "calibration", "model_amyloid", "model_tau"), ("field_results",), ("fields",),
         (Option("stain", "Stains", "stains", ["6E10", "AT8"], choices=("6E10", "AT8")), FRESH, *SHARD),
         heavy=True, view="/cohort", duration="about 1 minute per core", status=_fields_status, uses=("nuclei", "model_phikon")),
    Step("masks", "2 · Detect", "Outline objects", "Draw an exact outline around every detected object with Segment Anything.",
         "Adds precise size and shape measurements (area, roundness, dense cores). Optional: the main counts do not need it.",
         ("model_sam",), ("mask_results",), ("masks",),
         (Option("stain", "Stains", "stains", ["NeuN", "6E10", "AT8"], "Stains without detections yet are skipped.", ("NeuN", "6E10", "AT8")),
          DEVICE, FRESH, *SHARD),
         heavy=True, view="/cohort", duration="about 1 minute per core", status=_masks_status, uses=("neun_results", "field_results")),
    Step("summarize", "3 · Results", "Make results tables", "Combine everything into one row per donor and brain region.",
         "Sums counts and areas over tiles and replicate cores before dividing, so every density is area-weighted. Writes "
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
    Step("prepare_labels", "Models", "Prepare fields for labelling", "Find every candidate object in fields people will label.",
         "Finds the candidates exactly as the detection step does, stores their measurements and the AI's call, so labelling sessions open "
         "instantly and every label can train a model. Fields are full ~560 µm tiles, mostly tissue, spread over TMAs and groups. Labelling "
         "sessions run this for their own fields.",
         ("tiles", "calibration"), ("candidate_pool",), ("labels", "prepare"),
         (Option("stain", "Stain", "select", "AT8", choices=tuple(STAINS)), Option("fields", "Fields", "number", 20, "How many more fields to prepare."),
          Option("session", "Session", "text", "", "Prepare every field of this labelling session instead.", advanced=True)),
         # light: a session's few fields must not wait for a long detection run to finish
         heavy=False, view="/labels", duration="10–30 s per field", status=_labels_status, uses=("nuclei", "model_cellpose"),
         waits_for_uses=False),  # a session's fields usually have their nuclei already; missing ones are found on the spot
    Step("train", "Models", "Train a model", "Train a new model from the labels people made.",
         "Fits the same kind of model the pipeline uses, checks it by leaving one TMA out at a time, and compares it with the current model on "
         "your labels. The new model is only used after you choose it on the Models page.",
         ("label_store", "candidate_pool"), ("trained_models",), ("train",), (Option("stain", "Stain", "select", "AT8", choices=tuple(STAINS)),
          Option("session", "Sessions", "text", "", "Train only on the labels of these sessions (session IDs; reference = the imported "
                 "reference labels). Empty: every session.", advanced=True)),
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


def wait_reason(earlier: Step, later: Step, running_from: str = "") -> str:
    """Why a later step must wait for an earlier running / waiting one ("" = no need): it makes something the later step reads,
    or both write the same file (e.g. the core table). `running_from` names where the earlier one runs, if not in the app."""
    which = f"“{earlier.title}”" + (f", running from the {running_from}," if running_from else "")
    made = set(earlier.produces) & later.waits_on()
    if made:
        return f"Waits for {which} to finish (needs: {', '.join(RESOURCES[r].label for r in sorted(made))})"
    shared = set(earlier.writes + earlier.produces) & set(later.writes + later.produces)
    return f"Waits for {which} to finish (both write: {RESOURCES[sorted(shared)[0]].label})" if shared else ""


def depends_on(step: Step) -> list[str]:
    """Steps that make what this step needs or uses: the arrows of the pipeline map."""
    return [s.id for s in STEPS if s.id != step.id and set(s.produces) & step.reads()]


def resource_info(project: Project, resource_id: str) -> dict:
    resource = RESOURCES[resource_id]
    path = resource.path(project)
    exists = resource.ready(project) if resource.ready else _has_content(path)
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
