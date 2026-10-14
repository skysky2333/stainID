"""Project configuration: where inputs, models and outputs live for one TMA study."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_NAME = "stainid.yaml"
SUPPORTED_STAINS = {"NeuN": "neun", "6E10": "amyloid", "AT8": "tau"}
PRECISIONS = ("fp32", "tf32", "bf16")

DEFAULTS: dict = {
    "name": "stainID project",
    "pixel_size_um": 0.2738,
    "field_context_px": 128,
    "stains": dict(SUPPORTED_STAINS),
    "tma": {"prefix": "TMA-", "rows": 5, "columns": 6},
    "groups": [],
    # How donor ids are written when the guided setup reads TMA maps and case sheets: plain numbers with at least `digits`
    # digits (698 -> 0698; 0 keeps them as typed), prefixed ones as <PREFIX>_<number with `prefixed_digits` digits> (JOG3 -> JOG_003).
    "donor_ids": {"digits": 4, "prefixed_digits": 3},
    # How the guided setup names a core's donor-region (sample_region_id) when no saved TMA map shows the project's own
    # naming: <plain_prefix or prefixed_prefix><donor_id>_<region name, or its code with region: code>. The default gives
    # 1673_frontal, as when the column is left empty; plain_prefix BRC_ with region code gives BRC_1673_FR and JOG_003_FR.
    "sample_region_ids": {"plain_prefix": "", "prefixed_prefix": "", "region": "name"},
    "analysis": {"tiles": "whole_core"},
    # Probability a NeuN candidate or tau-neuron candidate needs to be counted. null keeps the cut-off stored in the model.
    # Random audits of accepted objects found many non-neurons at the model defaults (NeuN 0.5, AT8 0.4):
    # workspace/docs/object_acceptance_audit.md.
    "acceptance": {"neun": None, "tau": None},
    # Arithmetic of the deep models on CUDA (cpu always runs fp32): fp32, tf32 or bf16. Measured on an A100-class card:
    # Cellpose bf16 2.9 s/tile (about 1% of cells differ from fp32), tf32 3.8 s (fp32 masks cell for cell), fp32 10 s;
    # Segment Anything tf32 1.5x fp32 (median area change 0%), bf16 1.4x (0.14%), so tf32 is its better choice.
    "precision": {"cellpose": "bf16", "sam": "tf32"},
    "inputs": {
        "slides_dir": "data/slides",
        "slides_table": "data/slides.csv",
        "core_manifest": "data/core_manifest.csv",
        "core_images": "data/analysis/core_images.csv",
        "field_manifest": "data/analysis/cohort_systematic_tiles.csv",
        "tile_manifest": "data/analysis/cohort_primary_tiles.csv",
        "whole_core_manifest": "data/analysis/cohort_whole_core_tiles.csv",
        "calibration": "data/analysis/slide_dab_calibration.csv",
        "manual_exclusions": "data/annotations/cohort_manual_exclusions.json",
        "donor_metadata": "data/donor_metadata.csv",
        "tma_layout": "data/tma_layout.csv",
    },
    "models": {
        "neun": "data/validation/neun_final_predictions/model.joblib",
        "amyloid": "data/models/amyloid_frozen/bundle.joblib",
        "tau": "data/analysis/tau_neuron_benchmark/bundle.joblib",
        "cellpose": "data/models/cellpose/cpsam_v2",
        "sam": "data/models/sam/sam_vit_b_01ec64.pth",
        "plaque_cnn_dir": "data/models/plaque_cnn",
        "huggingface_home": "data/models/foundation/hf",
    },
    "outputs": {
        "qc": "data/qc",
        "neun": "data/analysis/cohort_neun",
        "fields": "data/analysis/cohort_v3",
        "nuclei": "data/analysis/tile_nuclei",
        "masks": "data/analysis/cohort_object_masks",
        "tables": "data/analysis",
        "reviews": "data/annotations",
        "trained_models": "data/models/trained",
        "jobs": "data/jobs",
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _diff(config: dict, base: dict) -> dict:
    out = {}
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            nested = _diff(value, base[key])
            if nested:
                out[key] = nested
        elif base.get(key) != value:
            out[key] = value
    return out


@dataclass(frozen=True)
class Project:
    root: Path
    config: dict

    @property
    def name(self) -> str:
        return self.config["name"]

    @property
    def pixel_size_um(self) -> float:
        return float(self.config["pixel_size_um"])

    @property
    def context_px(self) -> int:
        return int(self.config["field_context_px"])

    @property
    def stains(self) -> dict[str, str]:
        return dict(self.config["stains"])

    @property
    def tma_prefix(self) -> str:
        return str(self.config["tma"]["prefix"])

    @property
    def config_file(self) -> Path:
        return self.root / CONFIG_NAME

    @property
    def is_configured(self) -> bool:
        return self.config_file.exists()

    def precision(self, model: str) -> str:
        value = self.config["precision"][model]
        if value not in PRECISIONS:
            raise ValueError(f"precision.{model} must be one of {', '.join(PRECISIONS)}, not {value!r}")
        return value

    def analysis_tiles(self) -> Path:
        """The tiles detection runs on: every tissue tile of each core (whole_core) or only the sampled fields."""
        return self.input("whole_core_manifest" if self.config["analysis"]["tiles"] == "whole_core" else "tile_manifest")

    def acceptance(self, model: str, bundle: dict) -> float:
        """The probability cut-off for counting an object: the project setting if given, else the model's own."""
        value = self.config["acceptance"].get(model)
        if value is not None:
            return float(value)
        return float(bundle["context_threshold"] if model == "neun" else bundle["probability_threshold"])

    def resolve(self, section: str, key: str) -> Path:
        path = Path(self.config[section][key])
        return path if path.is_absolute() else self.root / path

    def input(self, key: str) -> Path:
        return self.resolve("inputs", key)

    def model(self, key: str) -> Path:
        return self.resolve("models", key)

    def output(self, key: str) -> Path:
        return self.resolve("outputs", key)

    def relative(self, path: Path) -> str:
        path = Path(path)
        return str(path.relative_to(self.root)) if path.is_absolute() and path.is_relative_to(self.root) else str(path)

    def tma_name(self, tma: str) -> str:
        return f"{self.tma_prefix}{tma}"

    def apply_environment(self) -> None:
        """Point third-party model caches at the project's model folder (offline, reproducible)."""
        os.environ.setdefault("HF_HOME", str(self.model("huggingface_home")))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH", str(self.model("cellpose").parent))
        os.environ.setdefault("STAINID_PLAQUE_CNN_DIR", str(self.model("plaque_cnn_dir")))


def load_project(path: Path | str | None = None) -> Project:
    """Load `stainid.yaml` from `path`, $STAINID_PROJECT, or the current directory; missing keys use DEFAULTS."""
    candidate = Path(path or os.environ.get("STAINID_PROJECT", "."))
    config_file = candidate / CONFIG_NAME if candidate.is_dir() else candidate
    root = config_file.parent.resolve()
    override = yaml.safe_load(config_file.read_text()) if config_file.exists() else {}
    return Project(root=root, config=_merge(DEFAULTS, override or {}))


def save_project(project: Project, changes: dict) -> Project:
    """Merge `changes` into the project's configuration and write only the values that differ from DEFAULTS."""
    config = _merge(project.config, changes)
    project.root.mkdir(parents=True, exist_ok=True)
    project.config_file.write_text(yaml.safe_dump(_diff(config, DEFAULTS) or {"name": config["name"]}, sort_keys=False))
    return Project(root=project.root, config=config)


def write_default_config(directory: Path) -> Path:
    target = Path(directory) / CONFIG_NAME
    if target.exists():
        raise FileExistsError(target)
    target.write_text(yaml.safe_dump(DEFAULTS, sort_keys=False))
    return target
