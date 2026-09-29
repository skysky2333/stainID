"""Project configuration: where inputs, models and outputs live for one TMA study."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_NAME = "stainid.yaml"

DEFAULTS: dict = {
    "name": "stainID project",
    "pixel_size_um": 0.2738,
    "field_context_px": 128,
    "stains": {"NeuN": "neun", "6E10": "amyloid", "AT8": "tau"},
    "inputs": {
        "core_manifest": "data/core_manifest.csv",
        "core_images": "data/analysis/core_images.csv",
        "field_manifest": "data/analysis/cohort_systematic_tiles.csv",
        "tile_manifest": "data/analysis/cohort_primary_tiles.csv",
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
        "neun": "data/analysis/cohort_neun",
        "fields": "data/analysis/cohort_v3",
        "nuclei": "data/analysis/tile_nuclei",
        "masks": "data/analysis/cohort_object_masks",
        "tables": "data/analysis",
        "reviews": "data/annotations",
        "jobs": "data/jobs",
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
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

    def _resolve(self, section: str, key: str) -> Path:
        path = Path(self.config[section][key])
        return path if path.is_absolute() else self.root / path

    def input(self, key: str) -> Path:
        return self._resolve("inputs", key)

    def model(self, key: str) -> Path:
        return self._resolve("models", key)

    def output(self, key: str) -> Path:
        return self._resolve("outputs", key)

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


def write_default_config(directory: Path) -> Path:
    target = Path(directory) / CONFIG_NAME
    if target.exists():
        raise FileExistsError(target)
    target.write_text(yaml.safe_dump(DEFAULTS, sort_keys=False))
    return target
