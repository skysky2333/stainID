"""Chromogens: the colour each stain is developed in, and how an image in any of them becomes the brown-on-blue (H-DAB)
picture every stainID step reads.

The detection steps, the stain thresholds and the models all measure DAB with the Ruifrok-Johnston H-E-DAB matrix
(`stainid.imaging.color`). An image stained in another colour (Fast Red, NovaRed, a blue chromogen), or one carrying two
chromogens (e.g. AT8 in blue and pTDP-43 in DAB on one slide), is unmixed with its own stain vectors into amounts of
each stain (colour deconvolution), and drawn again with the hematoxylin and DAB colours: the stain of interest becomes
DAB, the counterstain stays hematoxylin, and every other chromogen is left out. A plain DAB image with a hematoxylin
counterstain is read as it is.

The vectors are optical-density colours (red, green, blue). DAB, hematoxylin, eosin, Fast Red, Fast Blue and AEC are the
published Ruifrok-Johnston (2001) values; NovaRed and the generic blue are approximations. A lab can measure its own
from an area of pure stain and give them as `vectors` under the stain in the project's imaging settings (a dataset
package's measured vectors are carried there when a project is made from it); any colour without one keeps these.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stainid.imaging.color import HED_FROM_RGB

VECTORS: dict[str, tuple[float, float, float]] = {
    "hematoxylin": (0.650, 0.704, 0.286),
    "DAB": (0.268, 0.570, 0.776),
    "eosin": (0.072, 0.990, 0.105),
    "FastRed": (0.214, 0.851, 0.478),
    "NovaRed": (0.250, 0.720, 0.648),
    "AEC": (0.274, 0.680, 0.680),
    "FastBlue": (0.749, 0.606, 0.267),
    "blue": (0.834, 0.513, 0.203),  # a blue chromogen (e.g. Vector Blue / BCIP-NBT-like), approximate
}
# How each chromogen is called in the app, and other spellings found in dataset descriptions.
NAMES = {"DAB": "brown (DAB)", "FastRed": "red (Fast Red)", "NovaRed": "red-brown (NovaRed)", "AEC": "red (AEC)",
         "FastBlue": "blue (Fast Blue)", "blue": "blue", "hematoxylin": "blue counterstain (hematoxylin)", "eosin": "pink (eosin)"}
ALIASES = {"dab": "DAB", "brown": "DAB", "fastred": "FastRed", "fast red": "FastRed", "fast_red": "FastRed", "novared": "NovaRed",
           "nova red": "NovaRed", "vector novared": "NovaRed", "aec": "AEC", "fastblue": "FastBlue", "fast blue": "FastBlue", "blue": "blue",
           "vector blue": "blue", "bcip/nbt": "blue", "hematoxylin": "hematoxylin", "haematoxylin": "hematoxylin", "h": "hematoxylin",
           "eosin": "eosin"}
COUNTERSTAIN = "hematoxylin"

# The colours the converted picture is drawn with: the rows of the inverse of `HED_FROM_RGB`, so `rgb_to_hed` reads the
# amounts back exactly (hematoxylin row 0, DAB row 2).
_RGB_FROM_HED = np.linalg.inv(HED_FROM_RGB.astype(np.float64))
_LOG_FLOOR = np.log(1e-6)


def canonical(name: str | None) -> str:
    """The chromogen key for a name as datasets write it ("Fast Red", "dab", "Vector NovaRed"); "" if unknown."""
    if not name:
        return ""
    if name in VECTORS:
        return name
    return ALIASES.get(str(name).strip().lower(), "")


def clean_vectors(raw) -> dict[str, list[float]]:
    """Measured stain vectors as a lab or dataset gives them ({chromogen: [r, g, b]} in optical density), keyed by the
    chromogen names used here ("Fast Red" -> FastRed); raises ValueError, in plain words, for anything else."""
    if not isinstance(raw, dict):
        raise ValueError("the colour vectors must name each colour and give its three values, e.g. {blue: [0.80, 0.55, 0.24]}")
    out = {}
    for name, value in raw.items():
        key = canonical(str(name))
        if not key:
            raise ValueError(f"unknown colour {name!r} in the colour vectors (known: {', '.join(VECTORS)})")
        try:
            numbers = [float(v) for v in value]
        except (TypeError, ValueError):
            numbers = []
        if len(numbers) != 3 or min(numbers) < 0 or max(numbers) <= 0:
            raise ValueError(f"the colour vector of {key} must be three numbers of at least 0 (red, green, blue optical density), not {value!r}")
        out[key] = [round(v, 4) for v in numbers]
    return out


@dataclass(frozen=True)
class Recipe:
    """How one image is read: the chromogens on it (counterstain first), and which of them is the stain of interest."""
    chromogens: tuple[str, ...]
    interest: str
    vectors: tuple[tuple[float, float, float], ...] | None = None  # own vectors, one per chromogen (else `VECTORS`)

    @property
    def is_plain_dab(self) -> bool:
        return self.interest == "DAB" and set(self.chromogens) <= {"DAB", COUNTERSTAIN} and self.vectors is None

    def as_dict(self) -> dict:
        return {"chromogens": list(self.chromogens), "interest": self.interest, **({"vectors": [list(v) for v in self.vectors]} if self.vectors else {})}

    @classmethod
    def from_dict(cls, data: dict | None) -> Recipe:
        data = data or {}
        chromogens = tuple(data.get("chromogens") or (COUNTERSTAIN, "DAB"))
        vectors = tuple(tuple(map(float, v)) for v in data["vectors"]) if data.get("vectors") else None
        return cls(chromogens, data.get("interest") or "DAB", vectors)


def recipe(interest: str, others: list[str] | tuple[str, ...] = (), counterstain: str = COUNTERSTAIN,
           vectors: dict[str, list[float]] | None = None) -> Recipe:
    """The recipe for a stain developed in `interest`, with `others` also on the slide (their names as datasets write them)."""
    main = canonical(interest) or "DAB"
    rest = [c for c in (canonical(o) for o in others) if c and c != main]
    chromogens = tuple(dict.fromkeys([canonical(counterstain) or COUNTERSTAIN, main, *rest]))
    if len(chromogens) > 3:
        raise ValueError("At most three colours (the counterstain and two chromogens) can be told apart in an RGB image")
    own = None
    if vectors:
        own = tuple(tuple(map(float, vectors.get(c) or VECTORS[c])) for c in chromogens)
    return Recipe(chromogens, main, own)


def _matrix(rows: list[tuple[float, float, float]]) -> np.ndarray:
    """Stain vectors as rows (normalised); with two stains the third is their cross product (the residual)."""
    matrix = np.array(rows, dtype=np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    if len(matrix) == 2:
        third = np.cross(matrix[0], matrix[1])
        matrix = np.vstack([matrix, third / np.linalg.norm(third)])
    return matrix


def amounts(rgb: np.ndarray, recipe_: Recipe) -> np.ndarray:
    """The amount of each chromogen of the recipe at every pixel (H x W x n, optical density units)."""
    rows = list(recipe_.vectors) if recipe_.vectors else [VECTORS[c] for c in recipe_.chromogens]
    matrix = _matrix(rows)
    od = -np.log10(np.maximum(rgb.astype(np.float64), 1.0) / 255.0)
    out = od.reshape(-1, 3) @ np.linalg.inv(matrix)
    return np.maximum(out[:, :len(rows)], 0).reshape(*rgb.shape[:2], len(rows))


def to_analysis_rgb(rgb: np.ndarray, recipe_: Recipe) -> np.ndarray:
    """The image as stainID reads it: counterstain in hematoxylin colour, the stain of interest in DAB colour, any
    other chromogen left out. A plain DAB image comes back unchanged."""
    if recipe_.is_plain_dab:
        return rgb
    found = amounts(rgb[..., :3], recipe_)
    counter = found[..., recipe_.chromogens.index(COUNTERSTAIN)] if COUNTERSTAIN in recipe_.chromogens else np.zeros(rgb.shape[:2])
    stain = found[..., recipe_.chromogens.index(recipe_.interest)]
    # back from base-10 optical density to the natural-log scale of `rgb_to_hed`, then to RGB
    hed = np.stack([counter, np.zeros_like(counter), stain], axis=-1) * np.log(10) / -_LOG_FLOOR
    out = np.exp((hed.reshape(-1, 3) @ _RGB_FROM_HED) * _LOG_FLOOR).reshape(rgb.shape[:2] + (3,))
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def chromogen_choices() -> list[dict]:
    """The colours offered in the app, plain name first."""
    return [{"id": key, "label": NAMES[key]} for key in ("DAB", "FastRed", "NovaRed", "AEC", "blue", "FastBlue")]
