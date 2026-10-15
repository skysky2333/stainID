"""Per-project imaging settings of an image set or whole-slide project: pixel size, rescaling, and for every stain its
model family and colours.

`imaging.stains` in stainid.yaml maps each stain (antibody) as the images name it to::

    4G8:    {family: amyloid, chromogen: DAB}
    NAB228: {family: amyloid, chromogen: NovaRed}
    AT8:    {family: tau, chromogen: blue, others: [DAB]}     # AT slide: AT8 blue, pTDP-43 brown; AT8 is the one analysed
    IBA1:   {family: ignore}

The family decides which of stainID's steps and models read the stain: `neun` (NeuN), `amyloid` (6E10-type: 6E10, 4G8,
NAB228, ...), `tau` (AT8-type: AT8, PHF-1, CP13, ...), or `ignore`. In the tables every image carries the family's
pipeline stain name (NeuN, 6E10 or AT8) in `stain` and its own antibody in `antibody`, so every step, model and page
works on it unchanged.
"""
from __future__ import annotations

from stainid.imaging import chromogens
from stainid.project import SUPPORTED_STAINS, Project

ANALYSIS_STAIN = {family: stain for stain, family in SUPPORTED_STAINS.items()}  # neun -> NeuN, amyloid -> 6E10, tau -> AT8
FAMILIES = {"amyloid": "Amyloid plaques (like 6E10)", "tau": "Tau (like AT8)", "neun": "Neurons (NeuN)", "ignore": "Not analysed"}
# Antibodies stainID knows the family of; anything else is asked for.
KNOWN_FAMILY = {
    "6e10": "amyloid", "4g8": "amyloid", "nab228": "amyloid", "abeta": "amyloid", "aβ": "amyloid", "amyloid": "amyloid", "amyloid-beta": "amyloid",
    "beta-amyloid": "amyloid", "6f/3d": "amyloid", "10d5": "amyloid", "82e1": "amyloid", "at8": "tau", "phf-1": "tau",
    "phf1": "tau", "cp13": "tau", "ptau": "tau", "p-tau": "tau", "tau": "tau", "at100": "tau", "at180": "tau", "neun": "neun",
    "rbfox3": "neun", "a60": "neun",
}
TOLERANCE = 0.03  # images within 3% of the analysis scale are read as they are


PREFIXES = (("abeta", "amyloid"), ("aβ", "amyloid"), ("a-beta", "amyloid"), ("amyloid", "amyloid"), ("anti-abeta", "amyloid"),
            ("ptau", "tau"), ("p-tau", "tau"), ("phospho-tau", "tau"), ("anti-tau", "tau"))


def family_of(stain: str) -> str:
    """The model family stainID guesses for a stain name ("" when it does not know): Aβ40, Abeta42 and anti-Abeta 1-16
    are amyloid, pTau-S202 is tau."""
    key = str(stain).strip().lower()
    known = SUPPORTED_STAINS.get(stain, "") or KNOWN_FAMILY.get(key, "") or KNOWN_FAMILY.get(key.replace(" ", ""), "")
    return known or next((family for prefix, family in PREFIXES if key.startswith(prefix)), "")


def stain_settings(project: Project) -> dict[str, dict]:
    return dict(project.config.get("imaging", {}).get("stains") or {})


def setting(project: Project, stain: str) -> dict:
    """The settings of one stain, with the family guessed and DAB as the colour when they are not set."""
    own = stain_settings(project).get(stain, {})
    return {"family": own.get("family") or family_of(stain) or "", "chromogen": chromogens.canonical(own.get("chromogen")) or "DAB",
            "others": list(own.get("others") or []), "counterstain": own.get("counterstain") or chromogens.COUNTERSTAIN,
            "vectors": own.get("vectors") or None, **({"antibody": own["antibody"]} if own.get("antibody") else {})}


def analysis_stain(project: Project, stain: str) -> str:
    """The pipeline stain an image of this stain is analysed as (NeuN, 6E10, AT8), or "" when it is not analysed."""
    return ANALYSIS_STAIN.get(setting(project, stain)["family"], "")


def recipe_for(project: Project, stain: str) -> chromogens.Recipe:
    own = setting(project, stain)
    return chromogens.recipe(own["chromogen"], own["others"], own["counterstain"], own["vectors"])


def native_pixel_size(project: Project, image_value: str | float | None, file_value: float | None = None) -> float | None:
    """An image's own pixel size: the image list's, else the file's, else the project's setting."""
    for value in (image_value, file_value, project.config.get("imaging", {}).get("native_pixel_size_um")):
        try:
            if value not in (None, "") and float(value) > 0:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def scale_factor(project: Project, native_um: float | None) -> float:
    """How much an image is enlarged to reach the analysis scale (1.0: it is read as it is)."""
    if not native_um:
        return 1.0
    factor = native_um / project.pixel_size_um
    return 1.0 if abs(factor - 1.0) <= TOLERANCE else factor


def rescale_mode(project: Project) -> str:
    mode = project.config.get("imaging", {}).get("rescale") or "on_import"
    if mode not in ("on_import", "on_the_fly"):
        raise ValueError(f"imaging.rescale must be on_import or on_the_fly, not {mode!r}")
    return mode
