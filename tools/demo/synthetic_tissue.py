"""Procedural TMA core images that look roughly like DAB / hematoxylin immunohistochemistry.

Every core is drawn as optical densities of two stains (hematoxylin and DAB) and turned into RGB with the
Beer-Lambert law, using the same stain vectors stainID's colour deconvolution uses, so the DAB threshold, the DAB
overlay and tissue detection behave on these images as on real scans. Nothing here is derived from real tissue.

`draw_core` returns the RGB image together with every object it drew (position, outline, class), so the demo
project's "detections" sit exactly on the drawn objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

PIXEL_UM = 0.2738
HEMATOXYLIN = np.array([0.65, 0.70, 0.29], np.float32) / np.linalg.norm([0.65, 0.70, 0.29])
DAB = np.array([0.27, 0.57, 0.78], np.float32) / np.linalg.norm([0.27, 0.57, 0.78])
GLASS_OD = 0.045


@dataclass
class Drawn:
    """One drawn object: `kind` is what it really is, `x, y` its centre and `outline` its contour (core pixels)."""
    kind: str
    x: float
    y: float
    outline: np.ndarray
    dab: float
    extra: dict = field(default_factory=dict)

    @property
    def area_um2(self) -> float:
        return float(max(cv2.contourArea(self.outline.astype(np.float32)), 1.0) * PIXEL_UM**2)

    @property
    def diameter_um(self) -> float:
        return float(2 * np.sqrt(self.area_um2 / np.pi))


@dataclass
class Core:
    rgb: np.ndarray
    tissue: np.ndarray          # bool, usable tissue (holes removed)
    outline_mask: np.ndarray    # bool, the tissue disc including holes
    dab_positive: np.ndarray    # bool, DAB optical density above 0.45
    nuclei: np.ndarray          # (n, 5): x, y, semi-axis a, semi-axis b, angle (degrees)
    objects: list[Drawn]
    threads: list[np.ndarray]   # AT8 neuropil threads as polylines


def smooth_noise(rng: np.random.Generator, shape: tuple[int, int], scale: int) -> np.ndarray:
    """Zero-mean, unit-variance noise varying over about `scale` pixels."""
    small = rng.standard_normal((max(2, shape[0] // scale + 2), max(2, shape[1] // scale + 2))).astype(np.float32)
    big = cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return (big - big.mean()) / (big.std() + 1e-6)


def blob(rng: np.random.Generator, radius: float, roughness: float = 0.15, points: int = 28, stretch: float = 1.0,
         angle: float = 0.0) -> np.ndarray:
    """A closed irregular outline around (0, 0)."""
    theta = np.linspace(0, 2 * np.pi, points, endpoint=False)
    wobble = sum(rng.normal(0, roughness / k) * np.cos(k * theta + rng.uniform(0, 2 * np.pi)) for k in range(2, 6))
    r = radius * (1 + wobble)
    x, y = r * np.cos(theta) * stretch, r * np.sin(theta) / np.sqrt(stretch)
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([x * c - y * s, x * s + y * c], axis=1)


def pyramid(rng: np.random.Generator, radius: float, angle: float, apex: float = 1.6) -> np.ndarray:
    """A pyramidal / triangular soma pointing along `angle`."""
    theta = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    r = radius * (1 + 0.18 * np.cos(3 * theta) + rng.normal(0, 0.04, theta.size))
    r *= 1 + (apex - 1) * np.clip(np.cos(theta), 0, None) ** 6
    x, y = r * np.cos(theta), r * np.sin(theta) * 0.85
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([x * c - y * s, x * s + y * c], axis=1)


def flame(rng: np.random.Generator, length: float, width: float, angle: float) -> np.ndarray:
    """A flame-shaped tangle: a rounded body tapering into a long apical tail."""
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    along = np.cos(t)
    half = width * np.sin(t) * (0.55 + 0.45 * (1 - along) / 2) ** 1.2
    x = length * (along * np.where(along > 0, 1.0, 0.45))
    y = half * (1 + rng.normal(0, 0.05, t.size))
    bend = rng.normal(0, 0.12) * (x / length) ** 2 * length
    y = y + bend
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([x * c - y * s, x * s + y * c], axis=1)


def _fill(layer: np.ndarray, polygon: np.ndarray, value: float) -> None:
    cv2.fillPoly(layer, [np.round(polygon).astype(np.int32)], float(value), lineType=cv2.LINE_8)


def _stamp(layer: np.ndarray, patch: np.ndarray, x0: int, y0: int) -> None:
    """Max-composite a patch into the layer at (x0, y0), clipped to the layer."""
    h, w = patch.shape
    ax0, ay0, ax1, ay1 = max(0, x0), max(0, y0), min(layer.shape[1], x0 + w), min(layer.shape[0], y0 + h)
    if ax0 >= ax1 or ay0 >= ay1:
        return
    region = layer[ay0:ay1, ax0:ax1]
    np.maximum(region, patch[ay0 - y0: ay1 - y0, ax0 - x0: ax1 - x0], out=region)


def fluffy(rng: np.random.Generator, radius: float, density: float, core: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A plaque-like DAB patch: granular, fibrillar deposit fading outwards, with an optional dense core.
    Returns the patch (optical density) and its outline relative to the patch centre."""
    size = int(2 * radius * 1.35) | 1
    half = size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    shape_noise = cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32), (0, 0), radius / 4)
    shape_noise /= shape_noise.std() + 1e-6
    r = np.hypot(xx, yy) / radius * (1 + 0.18 * shape_noise)
    grain = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), 1.6)
    grain = (grain - grain.min()) / (np.ptp(grain) + 1e-6)
    falloff = np.clip(1 - r**1.6, 0, 1)
    patch = density * falloff * np.clip(grain * 2.4 - 0.7, 0, None) * (1 + 0.6 * falloff)
    mask = (falloff > 0.05).astype(np.uint8)
    if core > 0:
        core_r = radius * core
        core_noise = cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32), (0, 0), max(1.0, core_r / 3))
        core_mask = np.hypot(xx, yy) / core_r * (1 + 0.35 * core_noise / (core_noise.std() + 1e-6)) < 1
        patch = np.where(core_mask, np.maximum(patch, density * 2.3 * (0.8 + 0.4 * grain)), patch)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float32) - half
    return patch.astype(np.float32), outline


def _thread(rng: np.random.Generator, x: float, y: float, length_px: float, curl: float = 0.45) -> np.ndarray:
    steps = max(4, int(length_px / 5))
    heading = rng.uniform(0, 2 * np.pi)
    turn = rng.normal(0, curl, steps).cumsum() * 0.5
    angles = heading + turn
    pts = np.stack([np.cos(angles), np.sin(angles)], axis=1) * (length_px / steps)
    return np.vstack([[0, 0], pts.cumsum(axis=0)]) + [x, y]


def tissue_shape(rng: np.random.Generator, size: int, radius: float) -> tuple[np.ndarray, float, float]:
    """A core-shaped tissue mask: a wobbly disc, sometimes with a bite or a tear."""
    cx, cy = size / 2 + rng.normal(0, size * 0.015, 2)
    theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    r = radius * (1 + sum(rng.normal(0, 0.035 / k) * np.cos(k * theta + rng.uniform(0, 6.3)) for k in range(1, 9)))
    if rng.random() < 0.35:
        start = rng.uniform(0, 2 * np.pi)
        bite = np.exp(-((np.angle(np.exp(1j * (theta - start)))) / rng.uniform(0.15, 0.4)) ** 2)
        r *= 1 - rng.uniform(0.12, 0.35) * bite
    mask = np.zeros((size, size), np.uint8)
    cv2.fillPoly(mask, [np.stack([cx + r * np.cos(theta), cy + r * np.sin(theta)], 1).astype(np.int32)], 1)
    if rng.random() < 0.3:
        a = rng.uniform(0, np.pi)
        p0 = np.array([cx, cy]) + radius * 1.1 * np.array([np.cos(a), np.sin(a)])
        p1 = np.array([cx, cy]) + radius * rng.uniform(0.1, 0.6) * np.array([np.cos(a + 0.2), np.sin(a + 0.2)])
        cv2.line(mask, tuple(p0.astype(int)), tuple(p1.astype(int)), 0, int(rng.uniform(12, 30)))
    return mask.astype(bool), float(cx), float(cy)


def _random_points(rng: np.random.Generator, mask: np.ndarray, n: int, margin: int = 0) -> np.ndarray:
    ys, xs = np.nonzero(mask[::4, ::4])
    if not len(xs) or n <= 0:
        return np.zeros((0, 2))
    pick = rng.integers(0, len(xs), n)
    pts = np.stack([xs[pick] * 4, ys[pick] * 4], 1).astype(np.float32) + rng.uniform(0, 4, (n, 2))
    return pts


def draw_core(rng: np.random.Generator, stain: str, severity: float, shape_seed: int, size: int = 4800,
              neuron_density: float = 300.0) -> Core:
    """Draw one core of `stain` for a donor-region of the given pathology `severity` (0 none … 1 severe). The three
    stains of a core are serial sections, so they share `shape_seed` (the outline of the tissue)."""
    shape_rng = np.random.default_rng(shape_seed)
    radius = size * shape_rng.uniform(0.41, 0.445)
    tissue_mask, cx, cy = tissue_shape(shape_rng, size, radius)
    area_mm2 = tissue_mask.mean() * size**2 * PIXEL_UM**2 / 1e6
    shape = (size, size)
    per_core = lambda density: int(rng.poisson(max(density, 0) * area_mm2))  # noqa: E731
    um = 1 / PIXEL_UM

    # vacuoles and small vessels: holes in the tissue
    holes = np.zeros(shape, np.uint8)
    for x, y in _random_points(rng, tissue_mask, per_core(rng.uniform(250, 600))):
        cv2.ellipse(holes, (int(x), int(y)), (int(rng.uniform(4, 14) * um / 3.6), int(rng.uniform(3, 9) * um / 3.6)),
                    rng.uniform(0, 180), 0, 360, 1, -1)
    vessels = []
    for x, y in _random_points(rng, tissue_mask, per_core(rng.uniform(6, 12))):
        a, b = rng.uniform(8, 30) * um, rng.uniform(4, 10) * um
        angle = rng.uniform(0, 180)
        cv2.ellipse(holes, (int(x), int(y)), (int(a), int(b)), angle, 0, 360, 1, -1)
        vessels.append((x, y, a, b, angle))
    tissue = tissue_mask & ~holes.astype(bool)

    soft = cv2.GaussianBlur(tissue.astype(np.float32), (0, 0), 2.0)
    edge = cv2.GaussianBlur(tissue_mask.astype(np.float32), (0, 0), 25)
    rim = np.clip((0.5 - np.abs(edge - 0.5)) * 2, 0, 1) * tissue_mask

    # neuropil background
    hema = soft * (0.13 + 0.03 * smooth_noise(rng, shape, 400) + 0.025 * smooth_noise(rng, shape, 40)
                   + 0.03 * np.abs(smooth_noise(rng, shape, 3)))
    hema += 0.08 * rim
    dab_background = {"NeuN": 0.05, "6E10": 0.03, "AT8": 0.03 + 0.05 * severity}[stain]
    dab = soft * np.clip(dab_background * (1 + 0.35 * smooth_noise(rng, shape, 150)) + 0.025 * smooth_noise(rng, shape, 4), 0, None)
    if stain == "AT8":
        dab += 0.25 * rim * rng.uniform(0.3, 1.0)

    nuclei_layer = np.zeros(shape, np.float32)
    dab_layer = np.zeros(shape, np.float32)
    cut = np.zeros(shape, np.float32)   # where a blue nucleus shows through brown cytoplasm
    objects: list[Drawn] = []
    threads: list[np.ndarray] = []

    def nucleus(x: float, y: float, a: float, b: float, angle: float, od: float) -> None:
        outline = blob(rng, np.sqrt(a * b), 0.08, 20, a / max(b, 1), np.radians(angle)) + [x, y]
        _fill(nuclei_layer, outline, od)

    nuclei = []      # (x, y, a, b, angle, od)
    glia = _random_points(rng, tissue, per_core(rng.uniform(900, 1300)))
    for x, y in glia:
        kind = rng.choice(3, p=[0.5, 0.3, 0.2])
        if kind == 0:      # oligodendrocyte: small, round, dark
            a = rng.uniform(2.6, 3.3) * um
            nuclei.append((x, y, a, a * rng.uniform(0.85, 1.0), rng.uniform(0, 180), rng.uniform(1.9, 2.6)))
        elif kind == 1:    # astrocyte: larger, oval, pale
            a = rng.uniform(3.8, 5.0) * um
            nuclei.append((x, y, a, a * rng.uniform(0.7, 0.9), rng.uniform(0, 180), rng.uniform(0.9, 1.3)))
        else:              # microglia: elongated
            a = rng.uniform(3.5, 4.8) * um
            nuclei.append((x, y, a, a * rng.uniform(0.38, 0.55), rng.uniform(0, 180), rng.uniform(1.7, 2.3)))
    for x, y, a, b, angle in vessels:     # endothelial nuclei along vessel walls
        for t in rng.uniform(0, 2 * np.pi, int(rng.integers(1, 4))):
            c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
            px, py = a * np.cos(t) * 1.15, b * np.sin(t) * 1.15
            tangent = np.degrees(np.arctan2(b * np.cos(t), -a * np.sin(t))) + angle
            nuclei.append((x + px * c - py * s, y + px * s + py * c, 4.0 * um, 1.5 * um, tangent, rng.uniform(1.5, 2.0)))

    # neurons: pyramidal somata (all stains), brown in NeuN, faint APP signal in 6E10, tau inclusions in AT8
    neuron_count = per_core(neuron_density)
    neuron_pts = _random_points(rng, tissue, neuron_count)
    for i, (x, y) in enumerate(neuron_pts):
        r = rng.uniform(5.5, 9.5) * um
        angle = rng.uniform(-0.5, 0.5) - np.pi / 2 + (rng.normal(0, 1.2) if rng.random() < 0.3 else 0)
        outline = pyramid(rng, r, angle, apex=rng.uniform(1.15, 1.45)) + [x, y]
        nuc = (x - 0.15 * r * np.cos(angle), y - 0.15 * r * np.sin(angle), 0.45 * r, 0.40 * r, np.degrees(angle))
        direction, normal = np.array([np.cos(angle), np.sin(angle)]), np.array([-np.sin(angle), np.cos(angle)])
        apex = np.array([x, y]) + 0.95 * r * direction
        length, sway = rng.uniform(1.2, 2.6) * r, rng.normal(0, 0.12 * r)
        dendrite = np.array([apex + 0.3 * r * normal, apex + 0.5 * length * direction + (0.11 * r + sway / 2) * normal,
                             apex + length * direction + sway * normal, apex + 0.5 * length * direction + (sway / 2 - 0.11 * r) * normal,
                             apex - 0.3 * r * normal])
        basal = [np.array([[x, y], [x + 1.9 * r * np.cos(angle + t), y + 1.9 * r * np.sin(angle + t)]]) for t in rng.uniform(2.3, 2.8, 2) * [1, -1]]
        for _ in range(int(rng.integers(0, 3)) if rng.random() < 0.6 else 0):    # satellite glia
            t = rng.uniform(0, 2 * np.pi)
            a = rng.uniform(2.6, 3.2) * um
            nuclei.append((x + 1.15 * r * np.cos(t), y + 1.15 * r * np.sin(t), a, a * 0.85, rng.uniform(0, 180), rng.uniform(1.9, 2.5)))
        if stain == "NeuN":
            pale = rng.random() < 0.06
            od = rng.uniform(0.35, 0.55) if pale else rng.uniform(0.8, 1.7)
            _fill(dab_layer, outline, od)
            _fill(dab_layer, dendrite, od * 0.75)
            for line in basal:
                cv2.line(dab_layer, tuple(line[0].astype(int)), tuple(line[1].astype(int)), float(od * 0.55), 2)
            cv2.ellipse(dab_layer, (int(nuc[0]), int(nuc[1])), (int(nuc[2]), int(nuc[3])), nuc[4], 0, 360, float(od * rng.uniform(1.0, 1.3)), -1)
            nucleus(*nuc, od=0.3)
            objects.append(Drawn("missed_neuron" if pale else "neuron", x, y, outline, od))
        else:
            nucleus(*nuc, od=rng.uniform(0.45, 0.65))
            cv2.circle(nuclei_layer, (int(nuc[0]), int(nuc[1])), max(2, int(0.09 * r)), 1.8, -1)
            _fill(nuclei_layer, outline, 0.12)
            _fill(nuclei_layer, dendrite, 0.10)
            if stain == "6E10":
                _fill(dab_layer, outline, rng.uniform(0.08, 0.22))

    if stain == "NeuN":
        # candidates the detector rejects: dark glial nuclei, and stain precipitate
        for k in rng.choice(len(nuclei), min(len(nuclei), int(0.6 * neuron_count)), replace=False) if nuclei else []:
            x, y, a, b, angle, _ = nuclei[k]
            objects.append(Drawn("glia", x, y, cv2.ellipse2Poly((int(x), int(y)), (int(a), int(b)), int(angle), 0, 360, 20), 0.15))
        for x, y in _random_points(rng, tissue, per_core(12)):
            outline = blob(rng, rng.uniform(2.5, 5) * um, 0.3) + [x, y]
            _fill(dab_layer, outline, rng.uniform(1.4, 2.2))
            objects.append(Drawn("precipitate", x, y, outline, 1.8))

    if stain == "6E10":
        plaques = per_core(4 + 110 * severity**1.4)
        compact_share = 0.25 + 0.3 * severity
        for x, y in _random_points(rng, tissue, plaques):
            u = rng.random()
            if u < 0.18:
                kind, r, core = "small_plaque", rng.uniform(5.5, 7.5) * um, 0.45
            elif u < 0.18 + 0.82 * compact_share:
                kind, r, core = "compact", rng.uniform(9, 26) * um, rng.uniform(0.25, 0.4)
            else:
                kind, r, core = "diffuse", rng.uniform(14, 40) * um, 0.0
            density = rng.uniform(0.8, 1.2) if kind != "diffuse" else rng.uniform(0.6, 0.95)
            patch, outline = fluffy(rng, r, density, core)
            _stamp(dab_layer, patch, int(x) - patch.shape[1] // 2, int(y) - patch.shape[0] // 2)
            objects.append(Drawn(kind, x, y, outline + [x, y], float(patch[patch > 0.05].mean()),
                                 {"dense_core_fraction": float(core**2) if core else float(rng.uniform(0, 0.05))}))
        for x, y in _random_points(rng, tissue, per_core(25 + 30 * severity)):
            outline = blob(rng, rng.uniform(1.5, 3.5) * um, 0.25) + [x, y]
            _fill(dab_layer, outline, rng.uniform(0.9, 1.6))
            objects.append(Drawn("speck", x, y, outline, 1.2))
        for x, y, a, b, angle in vessels[: max(1, int(len(vessels) * severity * 0.6))] if severity > 0.3 else []:
            ring = cv2.ellipse2Poly((int(x), int(y)), (int(a * 1.15), int(b * 1.25)), int(angle), 0, 360, 10)
            cv2.polylines(dab_layer, [ring], True, float(rng.uniform(0.9, 1.4)), int(2.2 * um))
            objects.append(Drawn("vascular", x, y, ring, 1.1))
        for x, y in _random_points(rng, tissue, per_core(3 + 10 * severity)):
            patch, outline = fluffy(rng, rng.uniform(16, 30) * um, rng.uniform(0.2, 0.3))
            _stamp(dab_layer, patch, int(x) - patch.shape[1] // 2, int(y) - patch.shape[0] // 2)
            objects.append(Drawn("missed_plaque", x, y, outline + [x, y], 0.25))

    if stain == "AT8":
        tau_neurons = per_core(1 + 160 * severity**2)
        mature_share = 0.3 + 0.4 * severity
        for x, y in _random_points(rng, tissue, tau_neurons):
            angle = -np.pi / 2 + rng.normal(0, 0.5)
            if rng.random() < mature_share:
                length, width = rng.uniform(9, 16) * um, rng.uniform(4.5, 7) * um
                outline = flame(rng, length, width, angle) + [x, y]
                od = rng.uniform(1.6, 2.6)
                _fill(dab_layer, outline, od)
                tail = np.array([x, y]) + length * np.array([np.cos(angle), np.sin(angle)])
                cv2.line(dab_layer, tuple(tail.astype(int)), tuple((tail + 1.8 * length * np.array([np.cos(angle), np.sin(angle)])).astype(int)),
                         float(od * 0.6), max(2, int(0.5 * um)))
                objects.append(Drawn("tangle", x, y, outline, od))
            else:
                r = rng.uniform(6, 9) * um
                outline = pyramid(rng, r, angle) + [x, y]
                od = rng.uniform(0.55, 0.95)
                _fill(dab_layer, outline, od)
                nuc = (x, y, 0.42 * r, 0.38 * r, np.degrees(angle))
                cv2.ellipse(cut, (int(x), int(y)), (int(nuc[2]), int(nuc[3])), nuc[4], 0, 360, 0.75, -1)
                nucleus(*nuc, od=rng.uniform(0.7, 1.0))
                objects.append(Drawn("pretangle", x, y, outline, od))
        for x, y in _random_points(rng, tissue, per_core(4 + 12 * severity)):
            r = rng.uniform(6, 9) * um
            outline = pyramid(rng, r, -np.pi / 2) + [x, y]
            _fill(dab_layer, outline, rng.uniform(0.22, 0.32))
            objects.append(Drawn("missed_pretangle", x, y, outline, 0.27))
        for x, y in _random_points(rng, tissue, per_core(20 + 40 * severity)):   # glial tau, dots, thread knots
            outline = blob(rng, rng.uniform(2.5, 5.5) * um, 0.35) + [x, y]
            _fill(dab_layer, outline, rng.uniform(0.9, 1.6))
            objects.append(Drawn("tau_dot", x, y, outline, 1.2))
        for x, y in _random_points(rng, tissue, per_core(80 + 4200 * severity**1.6)):   # neuropil threads
            line = _thread(rng, x, y, rng.uniform(5, 28) * um)
            value = float(rng.uniform(0.6, 1.6))
            cv2.polylines(dab_layer, [line.astype(np.int32)], False, value, int(rng.choice([1, 2, 2, 2, 3])))
            for bead in line[rng.random(len(line)) < 0.25]:
                cv2.circle(dab_layer, tuple(bead.astype(int)), int(rng.choice([1, 2])), value * 1.2, -1)
            threads.append(line)
        for x, y in _random_points(rng, tissue, per_core(10 * severity**2)):   # neuritic clusters
            for _ in range(int(rng.integers(12, 30))):
                px, py = np.array([x, y]) + rng.normal(0, 9 * um, 2)
                line = _thread(rng, px, py, rng.uniform(4, 12) * um)
                cv2.polylines(dab_layer, [line.astype(np.int32)], False, float(rng.uniform(1.0, 1.8)), int(rng.choice([3, 4, 5])))

    for x, y, a, b, angle, value in nuclei:
        nucleus(x, y, a, b, angle, value)

    nuclei_layer = cv2.GaussianBlur(nuclei_layer, (0, 0), 1.3)
    nuclei_layer *= 1 + 0.2 * smooth_noise(rng, shape, 3)
    dab_layer = cv2.GaussianBlur(dab_layer, (0, 0), 1.3)
    dab_layer *= np.clip(1 + 0.22 * smooth_noise(rng, shape, 3), 0.3, None)
    dab_layer *= 1 - cv2.GaussianBlur(cut, (0, 0), 1.5)
    hema = (hema + nuclei_layer * tissue_mask) * rng.uniform(0.9, 1.1)
    dab = (dab + dab_layer * tissue_mask) * rng.uniform(0.9, 1.1)

    od = GLASS_OD + 0.002 * smooth_noise(rng, shape, 300) + 0.05 * soft + 0.16 * nuclei_layer   # neutral share: less saturated blue
    rgb = np.empty((*shape, 3), np.uint8)
    for channel in range(3):
        value = 255.0 * np.exp(-(od + hema * HEMATOXYLIN[channel] + dab * DAB[channel])) + rng.normal(0, 1.6, shape).astype(np.float32)
        rgb[..., channel] = np.clip(value, 0, 255).astype(np.uint8)
    return Core(rgb=rgb, tissue=tissue, outline_mask=tissue_mask, dab_positive=(dab > 0.45) & tissue_mask, nuclei=np.array([n[:5] for n in nuclei], np.float32).reshape(-1, 5),
                objects=objects, threads=threads)
