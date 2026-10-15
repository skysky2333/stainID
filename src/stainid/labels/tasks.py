"""What can be labelled, per stain: the object class (`labels.vocabulary`: identity and subtype), object outlines and field tags."""
from __future__ import annotations

from stainid.labels import vocabulary as vocab

STAINS = ("NeuN", "6E10", "AT8")
MODEL_KEY = {"NeuN": "neun", "6E10": "amyloid", "AT8": "tau"}
# The AI's calls, in the words of the labels made before subtypes existed (`vocabulary.LEGACY` maps them onto classes).
NEGATIVE = {"NeuN": "not a neuron", "6E10": "not a plaque", "AT8": "not a tau+ neuron"}
FIELD_TAGS = ["white matter", "mixed grey and white matter", "poor image quality", "fold or tear", "unusable"]
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


def positive(stain: str, value, vocabulary: vocab.Vocabulary | None = None, subtype=None) -> bool:
    """Whether a label (or an AI call) says the object is the stain's object: its identity's role is `positive`."""
    return (vocabulary or vocab.defaults(stain)).positive(value, subtype)


def skipped(stain: str, value, vocabulary: vocab.Vocabulary | None = None) -> bool:
    """Labels that say nothing about the object ("unsure", or empty)."""
    return (vocabulary or vocab.defaults(stain)).skipped(value)
