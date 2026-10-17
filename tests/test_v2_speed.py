"""The faster detection steps give the same numbers as the code they replace (stage 2 must not change on a re-run).

Each fast piece is compared with the straightforward computation it stands for: the run-length disk dilation with
OpenCV's, the cut Gaussian kernels with skimage's ridge filter, the 8-bit lookup tables with the formulas, the byte
opening with the float opening of the linear-artifact mask, the boxed component loops with whole-field ones, the shared
nucleus-ring loop with the old faint-ring loop, and whole tiles with and without the shared pictures and threads.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from scipy import ndimage

from stainid.imaging import parallel
from stainid.imaging.color import HED_FROM_RGB, rgb_to_hed
from stainid.imaging.morphology import dilate_disk, disk, runs
from stainid.imaging.shared import shared_pictures
from stainid.imaging.tissue import field_tissue_mask, fold_mask, linear_artifact_mask
from stainid.v2.field import Field

PS = 0.2738


@pytest.fixture
def threads():
    yield parallel.threads
    parallel.threads(1)


def _field(rgb, nuclei=None, threshold=0.05):
    h, w = rgb.shape[:2]
    tile = {"tile_id": "T_W0101", "x_px": 0, "y_px": 0, "width_px": w, "height_px": h, "pixel_width_um": PS, "pixel_height_um": PS,
            "context_px": 0}
    return Field(tile, rgb, (slice(0, h), slice(0, w)), (0, 0), threshold, PS, np.zeros((h, w), bool), nuclei)


def _tissue_picture(size=420, seed=0):
    """Pale tissue with glass at one side, brown cells and rings around blue nuclei, a dark fold band and threads."""
    rng = np.random.default_rng(seed)
    rgb = np.full((size, size, 3), (205, 185, 175), np.uint8)
    rgb[:, :40] = 240  # glass
    nuclei = np.zeros((size, size), np.int32)
    label = 0
    for _ in range(14):
        x, y = int(rng.integers(70, size - 30)), int(rng.integers(30, size - 30))
        label += 1
        cv2.circle(rgb, (x, y), int(rng.integers(14, 22)), (125, 85, 50) if label % 3 else (175, 140, 115), -1)
        cv2.circle(rgb, (x, y), 9, (95, 95, 160), -1)
        cv2.circle(nuclei, (x, y), 9, int(label), -1)
    for _ in range(6):
        x0, y0 = int(rng.integers(50, size)), int(rng.integers(0, size))
        cv2.line(rgb, (x0, y0), (x0 + int(rng.integers(-90, 90)), y0 + int(rng.integers(-90, 90))), (120, 80, 45), 2)
    cv2.line(rgb, (60, size - 40), (size - 20, size - 90), (60, 50, 110), 12)  # a dark, bluish fold-like band
    noise = rng.integers(-6, 7, rgb.shape)
    return np.clip(rgb.astype(int) + noise, 0, 255).astype(np.uint8), nuclei.astype(np.uint32)


# ---------------------------------------------------------------- primitives

def test_disk_dilation_equals_opencv():
    rng = np.random.default_rng(1)
    for _ in range(300):
        h, w = (int(v) for v in rng.integers(1, 90, 2))
        mask = rng.random((h, w)) > rng.choice([0.6, 0.97, 0.999])
        radius = int(rng.integers(0, 50))
        expected = cv2.dilate(mask.astype(np.uint8), disk(radius)).astype(bool)
        assert np.array_equal(dilate_disk(mask, radius), expected)
        assert np.array_equal(dilate_disk(mask, radius, runs(mask)), expected)


def test_lookup_tables_equal_the_formulas():
    rgb = np.random.default_rng(2).integers(0, 256, (64, 80, 3), dtype=np.uint8)
    scaled = np.maximum(rgb.astype(np.float32) / 255.0, 1e-6)
    hed = np.maximum((np.log(scaled) / np.log(1e-6)) @ HED_FROM_RGB, 0.0)
    assert np.array_equal(rgb_to_hed(rgb), hed)
    darkness = np.max(-np.log(np.maximum(rgb.astype(np.float32) / 255.0, 1.0 / 255.0)), axis=2)
    mask = cv2.morphologyEx((darkness >= 0.10).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool)
    assert np.array_equal(field_tissue_mask(rgb), mask)


def test_shared_pictures_are_made_once_and_read_only():
    rgb = np.random.default_rng(3).integers(0, 256, (40, 40, 3), dtype=np.uint8)
    with shared_pictures(rgb):
        first = rgb_to_hed(rgb)
        assert rgb_to_hed(rgb) is first and not first.flags.writeable
        assert field_tissue_mask(rgb) is field_tissue_mask(rgb)
        assert rgb_to_hed(rgb.copy()) is not first  # an equal copy is another picture
        with pytest.raises(ValueError):
            first[0, 0, 0] = 1.0
    assert rgb_to_hed(rgb) is not first and np.array_equal(rgb_to_hed(rgb), first)


def test_ridge_filter_equals_skimage_sato(threads):
    from skimage.filters import sato

    from stainid.stains.tau.threads import ridge_filter

    rgb, _ = _tissue_picture()
    dab = np.maximum(rgb_to_hed(rgb)[..., 2], 0).astype(np.float32)
    dab[:, :60] = 0.0  # zeros next to stain, where the cut taps would matter most
    expected = sato(dab, sigmas=[1, 1.5, 2, 3], black_ridges=False)
    for count in (1, 3):
        threads(count)
        assert np.array_equal(ridge_filter(dab, sigmas=[1, 1.5, 2, 3]), expected)


def _linear_reference(rgb):
    """The linear-artifact mask as it was: float openings, a component loop."""
    height, width = rgb.shape[:2]
    kernel_size = min(81, max(31, round(min(height, width) * 0.035)))
    kernel_size += 1 - kernel_size % 2
    center = radius = kernel_size // 2
    thickness = max(1, kernel_size // 25)
    darkness = 1.0 - cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    non_brown = rgb[..., 0].astype(np.int16) <= rgb[..., 2].astype(np.int16) + 15
    tissue = field_tissue_mask(rgb)
    distance = cv2.distanceTransform(tissue.astype(np.uint8), cv2.DIST_L2, 5)
    internal = tissue & (distance >= max(5, kernel_size // 10))
    darkness = cv2.GaussianBlur(darkness * non_brown * internal, (0, 0), 2.0)
    candidates = np.zeros((height, width), np.uint8)
    for angle in range(0, 180, 15):
        dx, dy = round(radius * np.cos(np.deg2rad(angle))), round(radius * np.sin(np.deg2rad(angle)))
        kernel = np.zeros((kernel_size, kernel_size), np.uint8)
        cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, thickness)
        candidates |= (cv2.morphologyEx(darkness, cv2.MORPH_OPEN, kernel) >= 0.35).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    artifact = np.zeros((height, width), np.uint8)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= 0.002 * height * width and max(stats[index, 2], stats[index, 3]) >= 0.35 * max(height, width):
            artifact[labels == index] = 1
    dilation = max(5, round(kernel_size * 0.2))
    dilation += 1 - dilation % 2
    return cv2.dilate(artifact, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))).astype(bool)


def test_linear_artifact_mask_equals_the_float_opening(threads):
    rgb = np.full((600, 600, 3), (200, 180, 175), np.uint8)
    cv2.line(rgb, (20, 40), (580, 500), (40, 40, 60), 7)  # a dark scratch across the field
    cv2.line(rgb, (300, 10), (310, 590), (70, 70, 90), 5)
    cv2.circle(rgb, (450, 150), 30, (60, 60, 80), -1)  # dark but round: no artifact
    expected = _linear_reference(rgb)
    assert expected.any()
    for count in (1, 4):
        threads(count)
        assert np.array_equal(linear_artifact_mask(rgb), expected)


def test_fold_mask_component_boxes_equal_the_whole_field_loop():
    rgb, _ = _tissue_picture(size=900, seed=4)
    found = fold_mask(rgb, PS)
    # the old loop: every component as a whole-field mask
    tissue = field_tissue_mask(rgb)
    hematoxylin = np.maximum(rgb_to_hed(rgb)[..., 0], 0.0).astype(np.float32)
    blurred = cv2.GaussianBlur(hematoxylin * tissue, (0, 0), 5.0 / PS)
    support = cv2.GaussianBlur(tissue.astype(np.float32), (0, 0), 5.0 / PS)
    blurred = np.divide(blurred, support, out=np.zeros_like(blurred), where=support > 0.5)
    values = blurred[tissue & (support > 0.5)]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    candidate = (blurred >= median + max(0.02, 5.0 * 1.4826 * mad)) & tissue
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    mask = np.zeros(tissue.shape, bool)
    for index in range(1, count):
        component = labels == index
        area = stats[index, cv2.CC_STAT_AREA] * PS ** 2
        if area < 1000.0:
            continue
        contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        (_, _), (w, h), _ = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        length = max(w, h) * PS
        if length >= 150.0 and length / max(area / max(length, 1e-6), 1e-6) >= 4.0:
            mask |= component
    expected = cv2.dilate(mask.astype(np.uint8), disk(max(1, round(5.0 / PS)))).astype(bool)
    assert expected.any()  # the band is a fold
    assert np.array_equal(found, expected)


def test_boxed_shape_and_radial_features_equal_whole_field_ones():
    from stainid.stains.amyloid.segmentation import _radial_features, _remove_small_components, _shape_features
    from stainid.stains.tau.network import _shape_features as network_shape

    rng = np.random.default_rng(5)
    labels = np.zeros((300, 300), np.int32)
    for k in range(1, 9):
        cv2.ellipse(labels, (int(rng.integers(0, 300)), int(rng.integers(0, 300))), (int(rng.integers(3, 40)), int(rng.integers(3, 25))),
                    float(rng.integers(0, 180)), 0, 360, k, -1)
    dab = rng.random((300, 300)).astype(np.float32)
    for k, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        whole, local, offset = labels == k, labels[box] == k, (box[1].start, box[0].start)
        assert _shape_features(local, 0.075, offset) == _shape_features(whole, 0.075)
        assert network_shape(local, 0.075, offset) == network_shape(whole, 0.075)
        assert _radial_features(local, dab[box], offset) == _radial_features(whole, dab)
    mask = rng.random((200, 200)) > 0.7
    count, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    expected = np.zeros(mask.shape, bool)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= 4:
            expected[cc == index] = True
    assert np.array_equal(_remove_small_components(mask, 4), expected)


def _faint_reference(field, minimum, below=0.3, ring_um=5.0):
    """The faint-ring loop as it was (its own pass over the nuclei), on all tissue as stage 2 proposes now."""
    ps = field.pixel_size_um
    tissue = field.tissue
    smooth = cv2.GaussianBlur(field.dab, (0, 0), 1.0 / ps)
    height, width = field.shape
    margin, ring_px = max(1, round(14 / ps)), max(1, round(ring_um / ps))
    near, far_in, far_out = disk(ring_px), disk(ring_px + round(2 / ps)), disk(ring_px + round(8 / ps))
    out = []
    for label, bounds in enumerate(ndimage.find_objects(field.nuclei), start=1):
        if bounds is None:
            continue
        y0, y1 = max(0, bounds[0].start - margin), min(height, bounds[0].stop + margin)
        x0, x1 = max(0, bounds[1].start - margin), min(width, bounds[1].stop + margin)
        nucleus = field.nuclei[y0:y1, x0:x1] == label
        local_tissue = tissue[y0:y1, x0:x1]
        if not 10.0 <= nucleus.sum() * ps * ps <= 200.0 or local_tissue[nucleus].mean() < 0.5:
            continue
        n8 = nucleus.astype(np.uint8)
        ring = cv2.dilate(n8, near).astype(bool) & ~nucleus & local_tissue
        far = cv2.dilate(n8, far_out).astype(bool) & ~cv2.dilate(n8, far_in).astype(bool) & local_tissue
        if ring.sum() < 20 or far.sum() < 20:
            continue
        local = smooth[y0:y1, x0:x1]
        background = float(np.median(local[far]))
        contrast = (float(local[ring].mean()) - background) / field.threshold
        if not minimum <= contrast < below:
            continue
        stained = (local >= background + 0.5 * field.threshold) & (ring | nucleus)
        soma = cv2.morphologyEx((stained | nucleus).astype(np.uint8), cv2.MORPH_CLOSE, disk(2)).astype(bool)
        ys, xs = np.nonzero(nucleus)
        out.append(("ring_faint", (x0 + xs.mean(), y0 + ys.mean()), soma, (x0, y0)))
    return out


def test_shared_nucleus_rings_equal_the_separate_loops():
    from stainid.stains.tau.neurons import tau_neuron_candidates
    from stainid.v2 import tau

    rgb, nuclei = _tissue_picture(seed=6)
    field = _field(rgb, nuclei, threshold=0.04)
    with shared_pictures(rgb):
        faint = tau.faint_rings(field, 0.05)
        expected = _faint_reference(field, 0.05)
        assert len(faint) == len(expected)
        for a, b in zip(faint, expected):
            assert a[0] == b[0] and a[1] == b[1] and a[3] == b[3] and np.array_equal(a[2], b[2])
        alone = tau_neuron_candidates(rgb, field.nuclei, field.threshold, PS, None, background_exclusion=field.exclusion)
        shared = tau_neuron_candidates(rgb, field.nuclei, field.threshold, PS, None, rings=tau.rings(field, 0.05),
                                       background_exclusion=field.exclusion)
    assert alone[0] == shared[0] and np.array_equal(alone[1], shared[1])
    assert any(row["candidate_source"] == "nucleus_ring" for row in alone[0])


@pytest.mark.parametrize("stain", ["NeuN", "6E10", "AT8"])
def test_tile_is_the_same_with_shared_pictures_and_threads(stain, threads):
    from stainid.v2 import proposals
    from stainid.v2.params import DEFAULTS

    rgb, nuclei = _tissue_picture(seed=7)
    p = DEFAULTS["proposals"]
    shared = {k: v for k, v in p.items() if k not in ("NeuN", "6E10", "AT8")}

    def tile(count, share):
        threads(count)
        field = _field(rgb, nuclei, threshold=0.04)
        if share:
            with shared_pictures(rgb):
                return proposals.process_tile(field, stain, p[stain], shared), proposals.mask_piece(field, stain)
        return proposals.process_tile(field, stain, p[stain], shared), proposals.mask_piece(field, stain)

    plain, plain_mask = tile(1, False)
    fast, fast_mask = tile(3, True)
    assert plain["rows"], "the picture should give candidates"
    assert np.array_equal(plain_mask, fast_mask)
    for key in ("rows", "summary", "unproposed", "threads"):
        a, b = plain[key], fast[key]
        assert len(a) == len(b) if isinstance(a, list) else True
        for x, y in (zip(a, b) if isinstance(a, list) else [(a, b)]):
            assert x.keys() == y.keys()
            for k in x:
                same = (x[k] == y[k]) or (isinstance(x[k], float) and np.isnan(x[k]) and np.isnan(y[k]))
                if isinstance(same, np.ndarray):
                    same = same.all()
                assert same, (key, k, x[k], y[k])
