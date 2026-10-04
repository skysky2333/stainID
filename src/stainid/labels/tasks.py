"""What can be labelled, per stain: the object class (one vocabulary per stain), object outlines and field tags."""
from __future__ import annotations

STAINS = ("NeuN", "6E10", "AT8")
MODEL_KEY = {"NeuN": "neun", "6E10": "amyloid", "AT8": "tau"}
CLASSES = {
    "NeuN": ["NeuN+ neuron", "not a neuron", "artifact or unclear", "unsure"],
    "6E10": ["compact plaque", "diffuse plaque", "small plaque", "not a plaque", "unsure"],
    "AT8": ["tau+ neuron: tangle", "tau+ neuron: pretangle", "not a tau+ neuron", "unsure"],
}
POSITIVE = {
    "NeuN": {"NeuN+ neuron"},
    "6E10": {"compact plaque", "diffuse plaque", "small plaque", "plaque"},
    "AT8": {"tau+ neuron: tangle", "tau+ neuron: pretangle", "tau+ neuron"},
}
NEGATIVE = {"NeuN": "not a neuron", "6E10": "not a plaque", "AT8": "not a tau+ neuron"}
SKIP = {"unsure", ""}
FIELD_TAGS = ["white matter", "mixed grey and white matter", "poor image quality", "fold or tear", "unusable"]
INSTRUCTIONS = {
    "NeuN": "NeuN+ neuron: a brown (DAB) neuronal cell body. Not a neuron: blue-only nuclei (glia, vessels) or background. "
            "Artifact: folds, edges, stain precipitate.",
    "6E10": "Compact plaque: a dense, well-defined brown core. Diffuse plaque: loose, fluffy brown deposit without a dense core. "
            "Small plaque: a real plaque 10–15 µm across, too small to type. Not a plaque: vessel wall, cell, speck under 10 µm, edge or fold.",
    "AT8": "Tangle: a dense brown flame- or globe-shaped neuron. Pretangle: a neuron with diffuse brown cytoplasm around its nucleus. "
           "Not a tau+ neuron: threads, dots, glia, vessels or background.",
}
CROP_UM = {"NeuN": 45.0, "6E10": 85.0, "AT8": 55.0}


def ai_value(stain: str, model_class: str | None, mature: bool | None = None) -> str:
    """The pipeline's call for an object, in the stain's label vocabulary (None = not detected = negative)."""
    if stain == "NeuN":
        return "NeuN+ neuron" if model_class == "neuron" else NEGATIVE[stain]
    if stain == "6E10":
        return {"compact": "compact plaque", "diffuse": "diffuse plaque", "small_plaque": "small plaque"}.get(model_class or "", NEGATIVE[stain])
    if model_class is None:
        return NEGATIVE[stain]
    return "tau+ neuron: tangle" if mature else "tau+ neuron: pretangle"


def positive(stain: str, value: str) -> bool:
    return value in POSITIVE[stain]
