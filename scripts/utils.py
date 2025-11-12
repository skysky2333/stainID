import os
import math
import csv
import yaml
from dataclasses import dataclass, asdict, field
from typing import Dict, Tuple, List, Optional

import numpy as np
import cv2
import pandas as pd
from skimage.color import rgb2hed
from skimage.measure import label, regionprops
from skimage.morphology import disk, opening, closing, remove_small_objects, remove_small_holes
from skimage.segmentation import clear_border, watershed
from skimage import feature
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
from scipy.spatial import cKDTree


DEFAULT_PIXEL_WIDTH_UM = 0.2738
DEFAULT_PIXEL_HEIGHT_UM = 0.2738

# ---------------- New multi-channel configuration schema ----------------

# Supported HED channels
HED_CHANNELS = ("hematoxylin", "eosin", "dab")


@dataclass
class ChannelThresholdConfig:
    # If True, threshold operates on normalized [0..1] channel; otherwise on raw OD-like channel
    normalize: bool = False
    method: str = "otsu"  # 'otsu' | 'yen' | 'li'
    scale: float = 1.0
    # Absolute thresholds (choose one depending on normalize)
    absolute_normalized: Optional[float] = None
    absolute_raw: Optional[float] = None
    # Orientation control: None => auto; True => invert; False => normal
    invert: Optional[bool] = None
    # DAB-only: stain estimation mode
    stain_estimation: str = "default"  # 'default' | 'macenko'


@dataclass
class MorphologyConfig:
    open_radius: int = 2
    close_radius: int = 7
    remove_small: int = 150
    fill_holes: bool = True
    # Max area for hole filling in pixels; 0 or None means fill all holes
    fill_holes_max_area: Optional[int] = 64
    clear_border: bool = False
    # Optionally remove very large connected components (likely spurious background floods)
    remove_large_connected: bool = True
    # Threshold is interpreted as max span fraction against the larger image dimension (0..0.5)
    large_area_frac: float = 0.08
    # If provided (>0), use this as the reference larger dimension (e.g., full image when previewing crops)
    large_span_ref_dim: Optional[int] = None


@dataclass
class WatershedConfig:
    enabled: bool = False
    min_distance: int = 5
    compactness: float = 0.0


@dataclass
class ChannelConfig:
    enabled: bool = True
    threshold: ChannelThresholdConfig = field(default_factory=ChannelThresholdConfig)
    morph: MorphologyConfig = field(default_factory=MorphologyConfig)
    watershed: WatershedConfig = field(default_factory=WatershedConfig)


@dataclass
class GatingRule:
    target: str = "dab"
    by: str = "hematoxylin"
    mode: str = "touching"  # 'intersect' | 'touching'
    dilate_px: int = 2
    min_overlap_frac: float = 0.0


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_image_type(filename: str) -> str:
    base = os.path.basename(filename)
    prefix = base.split("_")[0]
    return prefix


def discover_images(input_dir: str, exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff")) -> List[str]:
    files = []
    for root, _, fns in os.walk(input_dir):
        for fn in fns:
            if fn.lower().endswith(exts):
                files.append(os.path.join(root, fn))
    files.sort()
    return files


def read_image_rgb(path: str, downsample_factor: int = 1) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if downsample_factor and downsample_factor > 1:
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // downsample_factor, h // downsample_factor), interpolation=cv2.INTER_AREA)
    return img


def rgb_to_dab_od(rgb: np.ndarray) -> np.ndarray:
    # Convert to Optical Density space and extract DAB channel via HED deconvolution
    hed = rgb2hed(rgb)
    dab = hed[:, :, 2]
    return dab


def normalize_channel(x: np.ndarray) -> np.ndarray:
    # Normalize to [0,1] robustly using percentiles to reduce outlier effects
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        hi = x.max()
        lo = x.min()
        if hi == lo:
            return np.zeros_like(x, dtype=np.float32)
    x = (x - lo) / (hi - lo)
    x = np.clip(x, 0, 1)
    return x.astype(np.float32)


def normalize_channel_on_mask(x: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Normalize to [0,1] using percentiles computed on masked pixels when provided."""
    if mask is None or not mask.any():
        return normalize_channel(x)
    vals = x[mask]
    lo, hi = np.percentile(vals, [1, 99])
    if hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max())
        if hi == lo:
            return normalize_channel(x)
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0, 1)
    return y.astype(np.float32)


def compute_oriented_channel(image_rgb: np.ndarray,
                             cfg: "SegmentationConfig",
                             channel: str = "dab") -> Tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
    """Compute oriented channel (raw + normalized) for one of H/E/D.

    Returns (raw_oriented, norm_oriented, invert_inferred, valid_mask)
    """
    ch = str(channel).lower()
    if ch not in HED_CHANNELS:
        raise ValueError(f"Unsupported channel: {ch}")
    valid = compute_valid_mask(image_rgb, cfg)
    hed = rgb2hed(image_rgb)
    if ch == "dab":
        # Use optional Macenko for DAB
        mode = cfg.channels.get("dab", ChannelConfig()).threshold.stain_estimation or "default"
        if mode == "macenko":
            raw = dab_od_via_macenko(image_rgb)
        else:
            raw = hed[:, :, 2]
    elif ch == "hematoxylin":
        raw = hed[:, :, 0]
    else:  # eosin
        raw = hed[:, :, 1]

    norm_for_orient = normalize_channel_on_mask(raw, valid)
    invert = cfg.channels.get(ch, ChannelConfig()).threshold.invert
    if invert is None:
        norm_oriented, invert_inferred = choose_dab_orientation(raw, valid_mask=valid)
    else:
        invert_inferred = bool(invert)
        norm_oriented = 1.0 - norm_for_orient if invert_inferred else norm_for_orient
    raw_oriented = raw if not invert_inferred else -raw
    return raw_oriented.astype(np.float32), norm_oriented.astype(np.float32), invert_inferred, valid


def compute_oriented_dab_channels(image_rgb: np.ndarray,
                                  cfg: "SegmentationConfig") -> Tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
    # Backward-compatibility helper
    return compute_oriented_channel(image_rgb, cfg, channel="dab")


def choose_dab_orientation(dab: np.ndarray, expected_fg_frac: Tuple[float, float] = (0.005, 0.35),
                           valid_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
    # Heuristic: try Otsu on dab and -dab; pick orientation giving reasonable foreground fraction
    dab_norm = normalize_channel_on_mask(dab, valid_mask)
    try:
        import skimage.filters as filters
        arr1 = dab_norm[valid_mask] if (valid_mask is not None and valid_mask.any()) else dab_norm
        t1 = filters.threshold_otsu(arr1)
        mask1 = dab_norm > t1
        frac1 = mask1.mean()
        arr2 = (1.0 - dab_norm)[valid_mask] if (valid_mask is not None and valid_mask.any()) else (1.0 - dab_norm)
        t2 = filters.threshold_otsu(arr2)
        mask2 = (1.0 - dab_norm) > t2
        frac2 = mask2.mean()
    except Exception:
        # Fallback to median split
        arr = dab_norm[valid_mask] if (valid_mask is not None and valid_mask.any()) else dab_norm
        med = np.median(arr)
        mask1 = dab_norm > med
        mask2 = (1.0 - dab_norm) > (1.0 - med)
        frac1 = mask1.mean()
        frac2 = mask2.mean()

    lo, hi = expected_fg_frac
    score1 = (lo <= frac1 <= hi)
    score2 = (lo <= frac2 <= hi)
    if score1 and not score2:
        return dab_norm, False
    if score2 and not score1:
        return 1.0 - dab_norm, True
    # Otherwise pick the one with frac closer to target mid
    target = (lo + hi) / 2.0
    if abs(frac1 - target) <= abs(frac2 - target):
        return dab_norm, False
    else:
        return 1.0 - dab_norm, True


def threshold_channel(ch: np.ndarray, method: str = "otsu", scale: float = 1.0, absolute: Optional[float] = None,
                      valid_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Threshold a normalized [0,1] channel. Returns (mask, threshold).
    If absolute is provided (in [0,1]), method is ignored.
    """
    if absolute is not None:
        # Absolute threshold is used directly; scale does not apply.
        t = float(absolute)
    else:
        import skimage.filters as filters
        arr = ch[valid_mask] if (valid_mask is not None and valid_mask.any()) else ch
        if method == "otsu":
            t = float(filters.threshold_otsu(arr))
        elif method == "yen":
            t = float(filters.threshold_yen(arr))
        elif method == "li":
            t = float(filters.threshold_li(arr))
        else:
            t = float(filters.threshold_otsu(arr))
    if absolute is None:
        t *= float(scale)
    t = max(0.0, min(1.0, t))
    mask = ch > t
    return mask, t


def threshold_channel_any(ch: np.ndarray, method: str = "otsu", scale: float = 1.0,
                          absolute: Optional[float] = None,
                          valid_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Threshold without assuming [0,1] range. No clamping. Uses mask for stats if provided."""
    if absolute is not None:
        t = float(absolute)
    else:
        import skimage.filters as filters
        arr = ch[valid_mask] if (valid_mask is not None and valid_mask.any()) else ch
        if method == "otsu":
            t = float(filters.threshold_otsu(arr))
        elif method == "yen":
            t = float(filters.threshold_yen(arr))
        elif method == "li":
            t = float(filters.threshold_li(arr))
        else:
            t = float(filters.threshold_otsu(arr))
        t *= float(scale)
    mask = ch > t
    return mask, t


def morph_cleanup(mask: np.ndarray,
                  open_radius: int = 0,
                  close_radius: int = 0,
                  remove_small: int = 0,
                  fill_holes_flag: bool = True,
                  clear_border_flag: bool = False) -> np.ndarray:
    m = mask.astype(bool)
    if open_radius and open_radius > 0:
        m = opening(m, footprint=disk(open_radius))
    if close_radius and close_radius > 0:
        m = closing(m, footprint=disk(close_radius))
    if remove_small and remove_small > 0:
        m = remove_small_objects(m, remove_small)
    if fill_holes_flag:
        m = remove_small_holes(m, area_threshold=64)
    if clear_border_flag:
        m = clear_border(m)
    return m


@dataclass
class SegmentationConfig:
    type: str
    engine: str = "hed"
    # Pre-mask to ignore black borders and white background
    pre_mask_ignore_black: bool = True
    pre_mask_black_rgb_thresh: int = 5
    pre_mask_ignore_white: bool = True
    pre_mask_white_v_thresh: float = 0.90
    pre_mask_white_s_max: float = 0.25
    # HSV fallback parameters (if engine == 'hsv')
    hsv_h_low: float = 0.05   # 0..1 hue lower (brown ~ 20-40 deg)
    hsv_h_high: float = 0.15  # 0..1 hue upper
    hsv_s_min: float = 0.2    # min saturation
    hsv_v_min: float = 0.0    # min value
    hsv_v_max: float = 0.9    # max value
    # Multi-channel fields
    channels_to_output: List[str] = field(default_factory=lambda: ["dab"])  # subset of HED_CHANNELS
    channels: Dict[str, ChannelConfig] = field(default_factory=dict)
    # Gating controls: store rules always; apply only if enabled
    gating_enabled: bool = False
    gating_rules: List[GatingRule] = field(default_factory=list)


def default_config_for_type(t: str) -> SegmentationConfig:
    """Hard-coded defaults aligned with prior 212.yaml values.

    This avoids relying on any external file while providing sensible
    per-channel and pipeline defaults.
    """
    cfg = SegmentationConfig(type=t)
    cfg.engine = "hed"
    # Pre-mask defaults
    cfg.pre_mask_ignore_black = True
    cfg.pre_mask_black_rgb_thresh = 5
    cfg.pre_mask_ignore_white = True
    cfg.pre_mask_white_v_thresh = 0.90
    cfg.pre_mask_white_s_max = 0.25

    # Build per-channel configuration
    channels: Dict[str, ChannelConfig] = {}
    # Hematoxylin (H)
    hema = ChannelConfig(
        enabled=True,
        threshold=ChannelThresholdConfig(
            normalize=False,
            method="otsu",
            scale=1.0,
            absolute_normalized=None,
            absolute_raw=float(10.0 ** (-1.75)),  # ~0.01778
            invert=None,
            stain_estimation="default",
        ),
        morph=MorphologyConfig(
            open_radius=2,
            close_radius=3,
            remove_small=70,
            fill_holes=True,
            clear_border=True,
            remove_large_connected=True,
            large_area_frac=0.025,
            large_span_ref_dim=None,
        ),
    )
    channels["hematoxylin"] = hema

    # Eosin (E) — keep defaults
    eos = ChannelConfig(
        enabled=True,
        threshold=ChannelThresholdConfig(
            normalize=False,
            method="otsu",
            scale=1.0,
            absolute_normalized=None,
            absolute_raw=None,
            invert=None,
            stain_estimation="default",
        ),
        morph=MorphologyConfig(
            open_radius=2,
            close_radius=7,
            remove_small=150,
            fill_holes=True,
            clear_border=False,
            remove_large_connected=True,
            large_area_frac=0.08,
            large_span_ref_dim=None,
        ),
    )
    channels["eosin"] = eos

    # DAB (D)
    dab = ChannelConfig(
        enabled=True,
        threshold=ChannelThresholdConfig(
            normalize=False,
            method="otsu",
            scale=1.0,
            absolute_normalized=None,
            absolute_raw=float(10.0 ** (-1.5)),  # ~0.03162
            invert=None,
            stain_estimation="default",
        ),
        morph=MorphologyConfig(
            open_radius=2,
            close_radius=7,
            remove_small=150,
            fill_holes=True,
            clear_border=False,
            remove_large_connected=True,
            large_area_frac=0.08,
            large_span_ref_dim=None,
        ),
    )
    channels["dab"] = dab
    cfg.channels = channels

    # Output DAB + Hema by default
    cfg.channels_to_output = ["dab", "hematoxylin"]

    # Gating off by default; store default rule so when enabled later it uses these values
    cfg.gating_enabled = False
    cfg.gating_rules = [GatingRule(target="dab", by="hematoxylin", mode="touching", dilate_px=15, min_overlap_frac=0.0)]

    return cfg


def config_path(out_root: str, t: str) -> str:
    cfg_dir = os.path.join(out_root, "configs")
    ensure_dir(cfg_dir)
    return os.path.join(cfg_dir, f"{t}.yaml")


def save_config(cfg: SegmentationConfig, out_root: str) -> None:
    path = config_path(out_root, cfg.type)
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)


def load_or_create_config(out_root: str, t: str) -> SegmentationConfig:
    path = config_path(out_root, t)
    if not os.path.exists(path):
        cfg = default_config_for_type(t)
        save_config(cfg, out_root)
        return cfg

    # Read the config and materialize dataclasses
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    # Start from defaults and apply overridable fields
    base = default_config_for_type(t)
    # Simple scalar fields
    for key in [
        "engine",
        "pre_mask_ignore_black", "pre_mask_black_rgb_thresh",
        "pre_mask_ignore_white", "pre_mask_white_v_thresh", "pre_mask_white_s_max",
        "hsv_h_low", "hsv_h_high", "hsv_s_min", "hsv_v_min", "hsv_v_max",
    ]:
        if key in data:
            setattr(base, key, data[key])
    # Channels to output
    if isinstance(data.get("channels_to_output"), list):
        base.channels_to_output = [str(x) for x in data["channels_to_output"]]
    # Channels mapping
    ch_in = data.get("channels", {}) or {}
    if isinstance(ch_in, dict) and ch_in:
        out: Dict[str, ChannelConfig] = {}
        for name in HED_CHANNELS:
            if name in ch_in and isinstance(ch_in[name], dict):
                d = ch_in[name]
                thr = d.get("threshold", {}) if isinstance(d, dict) else {}
                m = d.get("morph", {}) if isinstance(d, dict) else {}
                ch_cfg = ChannelConfig(
                    enabled=bool(d.get("enabled", True)),
                    threshold=ChannelThresholdConfig(
                        normalize=bool(thr.get("normalize", base.channels[name].threshold.normalize)),
                        method=str(thr.get("method", base.channels[name].threshold.method)),
                        scale=float(thr.get("scale", base.channels[name].threshold.scale)),
                        absolute_normalized=thr.get("absolute_normalized", base.channels[name].threshold.absolute_normalized),
                        absolute_raw=thr.get("absolute_raw", base.channels[name].threshold.absolute_raw),
                        invert=thr.get("invert", base.channels[name].threshold.invert),
                        stain_estimation=str(thr.get("stain_estimation", base.channels[name].threshold.stain_estimation)),
                    ),
                    morph=MorphologyConfig(
                        open_radius=int(m.get("open_radius", base.channels[name].morph.open_radius)),
                        close_radius=int(m.get("close_radius", base.channels[name].morph.close_radius)),
                        remove_small=int(m.get("remove_small", base.channels[name].morph.remove_small)),
                        fill_holes=bool(m.get("fill_holes", base.channels[name].morph.fill_holes)),
                        fill_holes_max_area=m.get("fill_holes_max_area", base.channels[name].morph.fill_holes_max_area),
                        clear_border=bool(m.get("clear_border", base.channels[name].morph.clear_border)),
                        remove_large_connected=bool(m.get("remove_large_connected", base.channels[name].morph.remove_large_connected)),
                        large_area_frac=float(m.get("large_area_frac", base.channels[name].morph.large_area_frac)),
                        large_span_ref_dim=m.get("large_span_ref_dim", base.channels[name].morph.large_span_ref_dim),
                    ),
                    watershed=WatershedConfig(
                        enabled=bool((d.get("watershed", {}) or {}).get("enabled", base.channels[name].watershed.enabled)),
                        min_distance=int((d.get("watershed", {}) or {}).get("min_distance", base.channels[name].watershed.min_distance)),
                        compactness=float((d.get("watershed", {}) or {}).get("compactness", base.channels[name].watershed.compactness)),
                    ),
                )
                out[name] = ch_cfg
            else:
                out[name] = base.channels[name]
        base.channels = out
    # Gating
    base.gating_enabled = bool(data.get("gating_enabled", False))
    base.gating_rules = []
    for g in data.get("gating_rules", []) or []:
        try:
            base.gating_rules.append(GatingRule(
                target=str(g.get("target", "dab")),
                by=str(g.get("by", "hematoxylin")),
                mode=str(g.get("mode", "touching")),
                dilate_px=int(g.get("dilate_px", 2)),
                min_overlap_frac=float(g.get("min_overlap_frac", 0.0)),
            ))
        except Exception:
            continue
    # If gating_enabled not in file (older config), enable if non-empty rules
    if "gating_enabled" not in data:
        base.gating_enabled = bool(base.gating_rules)
    return base


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.4,
                 draw_edges: bool = True, edge_thickness: int = 3) -> np.ndarray:
    img = image_rgb.copy()
    overlay = image_rgb.copy()
    if draw_edges:
        m8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thickness = max(1, int(edge_thickness))
        cv2.drawContours(overlay, contours, -1, color, thickness=thickness)
    else:
        overlay[mask] = (np.array(color)[None, None, :]).astype(np.uint8)
    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return out


def overlay_multi_masks(image_rgb: np.ndarray,
                        masks: Dict[str, np.ndarray],
                        colors: Dict[str, Tuple[int, int, int]],
                        alpha: float = 0.45,
                        draw_edges: bool = True,
                        edge_thickness: int = 3) -> np.ndarray:
    """Overlay multiple boolean masks with distinct RGB colors."""
    img = image_rgb.copy()
    overlay = image_rgb.copy()
    for name, m in masks.items():
        if m is None:
            continue
        col = tuple(int(c) for c in colors.get(name, (255, 0, 0)))
        if draw_edges:
            m8 = (m.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, col, thickness=max(1, int(edge_thickness)))
        else:
            overlay[m] = col
    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return out


def compute_valid_mask(image_rgb: np.ndarray, cfg: SegmentationConfig) -> np.ndarray:
    rgb = image_rgb
    h, w = rgb.shape[:2]
    valid = np.ones((h, w), dtype=bool)
    if cfg.pre_mask_ignore_black:
        thr = int(cfg.pre_mask_black_rgb_thresh)
        black = (rgb[:, :, 0] <= thr) & (rgb[:, :, 1] <= thr) & (rgb[:, :, 2] <= thr)
        valid &= ~black
    if cfg.pre_mask_ignore_white:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2].astype(np.float32) / 255.0
        s = hsv[:, :, 1].astype(np.float32) / 255.0
        white = (v >= float(cfg.pre_mask_white_v_thresh)) & (s <= float(cfg.pre_mask_white_s_max))
        valid &= ~white
    return valid


def label_and_measure(mask: np.ndarray,
                      dab_norm: np.ndarray,
                      pixel_width_um: float = DEFAULT_PIXEL_WIDTH_UM,
                      pixel_height_um: float = DEFAULT_PIXEL_HEIGHT_UM,
                      labeled_in: Optional[np.ndarray] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    labeled = label(mask) if labeled_in is None else labeled_in
    props = regionprops(labeled)
    rows = []
    px_area_um2 = pixel_width_um * pixel_height_um

    for r in props:
        coords = r.coords
        vals = dab_norm[coords[:, 0], coords[:, 1]]
        area_px = float(r.area)
        perimeter_px = float(r.perimeter)
        area_um2 = area_px * px_area_um2
        circularity = 0.0
        if perimeter_px > 0:
            circularity = float(4.0 * math.pi * area_px / (perimeter_px ** 2))

        # Approx Feret diameter by major axis length
        feret_approx_px = float(getattr(r, 'major_axis_length', 0.0) or 0.0)

        # Intensity stats within region
        mean_int = float(vals.mean()) if vals.size else 0.0
        std_int = float(vals.std()) if vals.size else 0.0
        median_int = float(np.median(vals)) if vals.size else 0.0
        q25 = float(np.percentile(vals, 25)) if vals.size else 0.0
        q75 = float(np.percentile(vals, 75)) if vals.size else 0.0
        iqr = q75 - q25
        integrated_density = float(vals.sum())

        rows.append({
            "label": int(r.label),
            "centroid_row": float(r.centroid[0]),
            "centroid_col": float(r.centroid[1]),
            "area_px": area_px,
            "area_um2": area_um2,
            "perimeter_px": perimeter_px,
            "equivalent_diameter_px": float(getattr(r, 'equivalent_diameter', 0.0) or 0.0),
            "major_axis_length_px": float(getattr(r, 'major_axis_length', 0.0) or 0.0),
            "minor_axis_length_px": float(getattr(r, 'minor_axis_length', 0.0) or 0.0),
            "eccentricity": float(getattr(r, 'eccentricity', 0.0) or 0.0),
            "solidity": float(getattr(r, 'solidity', 0.0) or 0.0),
            "extent": float(getattr(r, 'extent', 0.0) or 0.0),
            "circularity": circularity,
            "feret_diameter_approx_px": feret_approx_px,
            "intensity_mean": mean_int,
            "intensity_std": std_int,
            "intensity_median": median_int,
            "intensity_iqr": iqr,
            "intensity_integrated": integrated_density,
        })
    df = pd.DataFrame(rows)
    return labeled, df


def label_and_measure_with_raw(mask: np.ndarray,
                               dab_norm: np.ndarray,
                               intensity_raw: np.ndarray,
                               pixel_width_um: float = DEFAULT_PIXEL_WIDTH_UM,
                               pixel_height_um: float = DEFAULT_PIXEL_HEIGHT_UM,
                               labeled_in: Optional[np.ndarray] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """Label regions and measure features using both normalized and raw oriented intensity channels.

    - dab_norm: normalized [0,1] channel oriented so higher=stronger stain (relative within image)
    - intensity_raw: raw OD-like channel oriented so higher=stronger stain (absolute across images)
    """
    labeled = label(mask) if labeled_in is None else labeled_in
    props = regionprops(labeled)
    rows: List[Dict] = []
    px_area_um2 = pixel_width_um * pixel_height_um

    for r in props:
        coords = r.coords
        vals = dab_norm[coords[:, 0], coords[:, 1]]
        vals_raw = intensity_raw[coords[:, 0], coords[:, 1]]
        area_px = float(r.area)
        perimeter_px = float(r.perimeter)
        area_um2 = area_px * px_area_um2
        circularity = 0.0
        if perimeter_px > 0:
            circularity = float(4.0 * math.pi * area_px / (perimeter_px ** 2))

        feret_approx_px = float(getattr(r, 'major_axis_length', 0.0) or 0.0)

        # Normalized intensity stats within region
        mean_int = float(vals.mean()) if vals.size else 0.0
        std_int = float(vals.std()) if vals.size else 0.0
        median_int = float(np.median(vals)) if vals.size else 0.0
        q25 = float(np.percentile(vals, 25)) if vals.size else 0.0
        q75 = float(np.percentile(vals, 75)) if vals.size else 0.0
        iqr = q75 - q25
        integrated_density = float(vals.sum())

        # Raw intensity stats
        mean_int_raw = float(vals_raw.mean()) if vals_raw.size else 0.0
        std_int_raw = float(vals_raw.std()) if vals_raw.size else 0.0
        median_int_raw = float(np.median(vals_raw)) if vals_raw.size else 0.0
        q25_raw = float(np.percentile(vals_raw, 25)) if vals_raw.size else 0.0
        q75_raw = float(np.percentile(vals_raw, 75)) if vals_raw.size else 0.0
        iqr_raw = q75_raw - q25_raw
        integrated_density_raw = float(vals_raw.sum())

        rows.append({
            "label": int(r.label),
            "centroid_row": float(r.centroid[0]),
            "centroid_col": float(r.centroid[1]),
            "area_px": area_px,
            "area_um2": area_um2,
            "perimeter_px": perimeter_px,
            "equivalent_diameter_px": float(getattr(r, 'equivalent_diameter', 0.0) or 0.0),
            "major_axis_length_px": float(getattr(r, 'major_axis_length', 0.0) or 0.0),
            "minor_axis_length_px": float(getattr(r, 'minor_axis_length', 0.0) or 0.0),
            "eccentricity": float(getattr(r, 'eccentricity', 0.0) or 0.0),
            "solidity": float(getattr(r, 'solidity', 0.0) or 0.0),
            "extent": float(getattr(r, 'extent', 0.0) or 0.0),
            "circularity": circularity,
            "feret_diameter_approx_px": feret_approx_px,
            # Normalized intensity
            "intensity_mean": mean_int,
            "intensity_std": std_int,
            "intensity_median": median_int,
            "intensity_iqr": iqr,
            "intensity_integrated": integrated_density,
            # Raw OD-oriented intensity (absolute)
            "intensity_raw_mean": mean_int_raw,
            "intensity_raw_std": std_int_raw,
            "intensity_raw_median": median_int_raw,
            "intensity_raw_iqr": iqr_raw,
            "intensity_raw_integrated": integrated_density_raw,
        })
    df = pd.DataFrame(rows)
    return labeled, df


def nearest_neighbor_stats(points: np.ndarray, img_area_px: int,
                           pixel_width_um: Optional[float] = None,
                           pixel_height_um: Optional[float] = None) -> Dict[str, float]:
    # points: Nx2 array of (row, col). Return nn mean/sd in px and um (if pixel sizes provided).
    if points.shape[0] < 2:
        out = {"nn_mean_px": float("nan"), "nn_sd_px": float("nan"), "clark_evans_R": float("nan"), "clark_evans_z": float("nan"), "density_per_px": float(points.shape[0]) / float(img_area_px)}
        if pixel_width_um and pixel_height_um:
            out.update({"nn_mean_um": float("nan"), "nn_sd_um": float("nan")})
        return out
    tree = cKDTree(points)
    dists, idxs = tree.query(points, k=2)
    nn = dists[:, 1]  # nearest neighbor excluding self
    nn_mean = float(nn.mean())
    nn_sd = float(nn.std(ddof=1)) if nn.size > 1 else 0.0

    # Clark-Evans R statistic
    N = points.shape[0]
    lam = float(N) / float(img_area_px)
    re = 1.0 / (2.0 * math.sqrt(lam)) if lam > 0 else float("nan")
    ro = nn_mean
    R = ro / re if (re and re > 0) else float("nan")
    var_ro = (4.0 - math.pi) / (4.0 * math.pi * lam * N) if (lam > 0 and N > 0) else float("nan")
    z = (ro - re) / math.sqrt(var_ro) if (not math.isnan(R) and var_ro and var_ro > 0) else float("nan")

    out = {
        "nn_mean_px": nn_mean,
        "nn_sd_px": nn_sd,
        "clark_evans_R": R,
        "clark_evans_z": z,
        "density_per_px": lam,
    }
    if pixel_width_um and pixel_height_um:
        # Compute per-point NN distances in microns using anisotropic pixel sizes
        # Build vectors to nearest neighbors
        nn_indices = idxs[:, 1]
        diffs = points - points[nn_indices]
        dy = diffs[:, 0] * float(pixel_height_um)
        dx = diffs[:, 1] * float(pixel_width_um)
        d_um = np.sqrt(dx * dx + dy * dy)
        out["nn_mean_um"] = float(np.mean(d_um))
        out["nn_sd_um"] = float(np.std(d_um, ddof=1)) if d_um.size > 1 else 0.0
    return out


def grid_dispersion(points: np.ndarray, img_shape: Tuple[int, int], K: int = 10) -> Dict[str, float]:
    h, w = img_shape[:2]
    if points.shape[0] == 0:
        return {"grid_vmr": float("nan"), "grid_cv": float("nan")}
    # Compute cell indices
    rows = np.clip((points[:, 0] * K / h).astype(int), 0, K - 1)
    cols = np.clip((points[:, 1] * K / w).astype(int), 0, K - 1)
    counts = np.zeros((K, K), dtype=int)
    for r, c in zip(rows, cols):
        counts[r, c] += 1
    vals = counts.ravel().astype(float)
    mean = vals.mean()
    var = vals.var(ddof=1) if vals.size > 1 else 0.0
    vmr = var / mean if mean > 0 else float("nan")
    cv = math.sqrt(var) / mean if mean > 0 else float("nan")
    return {"grid_vmr": float(vmr), "grid_cv": float(cv)}


def image_level_summary(df_sites: pd.DataFrame, mask: np.ndarray, img_shape: Tuple[int, int],
                        pixel_width_um: float, pixel_height_um: float) -> Dict[str, float]:
    h, w = img_shape[:2]
    num_sites = int(df_sites.shape[0])
    area_px = mask.sum()
    percent_area = float(area_px) / float(h * w)
    out = {
        "num_sites": num_sites,
        "percent_area_stained": percent_area,
    }
    for col in [
        "area_um2",
        "intensity_mean",
        "intensity_raw_mean",
        "circularity",
        "solidity",
        "eccentricity",
        "feret_diameter_approx_px",
    ]:
        if col in df_sites.columns and df_sites.shape[0] > 0:
            vals = df_sites[col].values.astype(float)
            out[f"{col}_mean"] = float(np.mean(vals))
            out[f"{col}_sd"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            out[f"{col}_iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25)) if vals.size > 0 else 0.0
        else:
            out[f"{col}_mean"] = float("nan")
            out[f"{col}_sd"] = float("nan")
            out[f"{col}_iqr"] = float("nan")
    return out


def save_csv(path: str, df: pd.DataFrame) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)


def save_mask_png(path: str, mask: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    mask_u8 = (mask.astype(np.uint8) * 255)
    cv2.imwrite(path, mask_u8)


def save_overlay_jpg(path: str, overlay_rgb: np.ndarray, quality: int = 90) -> None:
    ensure_dir(os.path.dirname(path))
    bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])


def _remove_very_large(mask_bin: np.ndarray, morph: MorphologyConfig) -> np.ndarray:
    if not morph.remove_large_connected or float(getattr(morph, 'large_area_frac', 0.0)) <= 0.0:
        return mask_bin
    h, w = mask_bin.shape
    lab = label(mask_bin)
    if lab.max() <= 0:
        return mask_bin
    ref_dim = int(getattr(morph, 'large_span_ref_dim', 0) or 0)
    big_dim = float(ref_dim if ref_dim > 0 else max(h, w))
    span_frac_thr = float(morph.large_area_frac)
    for r in regionprops(lab):
        minr, minc, maxr, maxc = r.bbox
        span_r = float(maxr - minr)
        span_c = float(maxc - minc)
        span_frac = max(span_r, span_c) / big_dim if big_dim > 0 else 0.0
        if span_frac >= span_frac_thr:
            mask_bin[lab == r.label] = False
    return mask_bin


def segment_channel_hed(image_rgb: np.ndarray,
                        cfg: SegmentationConfig,
                        channel: str) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Segment a single HED channel with its per-channel config.

    Returns (mask, norm_oriented, used_threshold, raw_oriented)
    """
    name = str(channel).lower()
    ch_cfg = cfg.channels.get(name, ChannelConfig())
    raw_oriented, norm_oriented, _inv, valid = compute_oriented_channel(image_rgb, cfg, channel=name)
    if not valid.any():
        shape = image_rgb.shape[:2]
        return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=np.float32), 0.0, raw_oriented

    if ch_cfg.threshold.normalize:
        ch_use = norm_oriented
        mask_bin, used_t = threshold_channel(
            ch_use,
            method=ch_cfg.threshold.method,
            scale=ch_cfg.threshold.scale,
            absolute=ch_cfg.threshold.absolute_normalized,
            valid_mask=valid,
        )
        out_norm = ch_use
    else:
        ch_use = raw_oriented
        mask_bin, used_t = threshold_channel_any(
            ch_use,
            method=ch_cfg.threshold.method,
            scale=ch_cfg.threshold.scale,
            absolute=ch_cfg.threshold.absolute_raw,
            valid_mask=valid,
        )
        out_norm = norm_oriented

    mask_bin &= valid
    # Phase 1 morphology (no hole-filling, no border clearing)
    mask_bin = morph_cleanup(
        mask_bin,
        open_radius=ch_cfg.morph.open_radius,
        close_radius=ch_cfg.morph.close_radius,
        remove_small=ch_cfg.morph.remove_small,
        fill_holes_flag=False,
        clear_border_flag=False,
    )
    mask_bin = _remove_very_large(mask_bin, ch_cfg.morph)
    # Phase 2
    if ch_cfg.morph.fill_holes:
        max_area = getattr(ch_cfg.morph, 'fill_holes_max_area', 64)
        try:
            A = int(max_area) if max_area is not None else 0
        except Exception:
            A = 0
        if A <= 0:
            mask_bin = ndi.binary_fill_holes(mask_bin).astype(bool)
        else:
            mask_bin = remove_small_holes(mask_bin, area_threshold=A)
    if ch_cfg.morph.clear_border:
        mask_bin = clear_border(mask_bin)
    return mask_bin, out_norm, float(used_t), raw_oriented


def segment_image_hsv(image_rgb: np.ndarray, cfg: SegmentationConfig) -> Tuple[np.ndarray, np.ndarray, float]:
    """HSV thresholding: returns (mask, density_like, used_threshold).
    Density-like channel is S * (1 - V) as a proxy for stain darkness.
    """
    img = image_rgb
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].astype(np.float32) / 179.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    h_low = float(cfg.hsv_h_low)
    h_high = float(cfg.hsv_h_high)
    if h_low <= h_high:
        h_mask = (h >= h_low) & (h <= h_high)
    else:
        # wrap-around
        h_mask = (h >= h_low) | (h <= h_high)
    s_mask = s >= float(cfg.hsv_s_min)
    v_mask = (v >= float(cfg.hsv_v_min)) & (v <= float(cfg.hsv_v_max))
    valid = compute_valid_mask(image_rgb, cfg)
    mask = (h_mask & s_mask & v_mask) & valid
    # Use DAB channel morphology parameters for HSV cleanup
    dab_morph = cfg.channels.get("dab", ChannelConfig()).morph
    # Phase 1 morphology (no hole-filling, no border clearing)
    mask = morph_cleanup(
        mask,
        open_radius=dab_morph.open_radius,
        close_radius=dab_morph.close_radius,
        remove_small=dab_morph.remove_small,
        fill_holes_flag=False,
        clear_border_flag=False,
    )
    # Remove very large connected components by span using DAB morph defaults
    mask = _remove_very_large(mask, dab_morph)
    # Phase 2: hole fill if requested
    if dab_morph.fill_holes:
        max_area = getattr(dab_morph, 'fill_holes_max_area', 64)
        try:
            A = int(max_area) if max_area is not None else 0
        except Exception:
            A = 0
        if A <= 0:
            mask = ndi.binary_fill_holes(mask).astype(bool)
        else:
            mask = remove_small_holes(mask, area_threshold=A)
    # Border clearing after filling holes, if requested
    if dab_morph.clear_border:
        mask = clear_border(mask)
    density_like = normalize_channel(s * (1.0 - v))
    return mask, density_like, float(cfg.hsv_s_min)


def segment_image(image_rgb: np.ndarray, cfg: SegmentationConfig) -> Tuple[np.ndarray, np.ndarray, float]:
    if cfg.engine == 'hsv':
        return segment_image_hsv(image_rgb, cfg)
    # HED: legacy single-channel path using DAB config
    m, norm, t, _raw = segment_channel_hed(image_rgb, cfg, channel="dab")
    return m, norm, t


def rgb_to_od(rgb: np.ndarray) -> np.ndarray:
    # Convert RGB uint8 [0,255] to OD values
    rgb = rgb.astype(np.float32)
    # Avoid log(0)
    od = -np.log((rgb + 1.0) / 256.0)
    return od


def macenko_estimate_stain_vectors(rgb: np.ndarray, alpha: float = 1.0, beta: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate two stain OD vectors using Macenko method.
    Returns (v1, v2) as 3D unit vectors in OD space.
    """
    od = rgb_to_od(rgb)
    H, W, _ = od.shape
    od_2d = od.reshape(-1, 3)
    # Remove transparent/background pixels (low optical density)
    mask = (od_2d > beta).any(axis=1)
    od_sel = od_2d[mask]
    if od_sel.shape[0] < 100:
        # Fallback: use all pixels
        od_sel = od_2d
    # SVD on covariance
    # Compute 2D projection onto first two PCs
    cov = np.cov(od_sel.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    v1 = vecs[:, 0]
    v2 = vecs[:, 1]
    # Project selected OD onto the plane spanned by v1,v2
    proj = od_sel @ np.stack([v1, v2], axis=1)  # shape (N,2)
    # Compute angles
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    # Select extremes
    lo = np.percentile(angles, alpha)
    hi = np.percentile(angles, 100 - alpha)
    # Form stain vectors at extreme angles
    stain1 = v1 * np.cos(lo) + v2 * np.sin(lo)
    stain2 = v1 * np.cos(hi) + v2 * np.sin(hi)
    # Normalize
    stain1 = stain1 / (np.linalg.norm(stain1) + 1e-8)
    stain2 = stain2 / (np.linalg.norm(stain2) + 1e-8)
    return stain1.astype(np.float32), stain2.astype(np.float32)


def pick_dab_vector(stain1: np.ndarray, stain2: np.ndarray) -> np.ndarray:
    """Pick which stain vector matches DAB best by comparing with scikit-image reference."""
    try:
        # hed_from_rgb is a 3x3 matrix; the DAB row approximates DAB OD direction
        from skimage.color.colorconv import hed_from_rgb as HED_FROM_RGB
        dab_ref = HED_FROM_RGB[2, :].astype(np.float32)
        dab_ref = dab_ref / (np.linalg.norm(dab_ref) + 1e-8)
    except Exception:
        # Fallback hard-coded (approx) if import path differs
        dab_ref = np.array([0.268, 0.570, 0.776], dtype=np.float32)
        dab_ref = dab_ref / (np.linalg.norm(dab_ref) + 1e-8)
    s1 = abs(float(np.dot(stain1, dab_ref)))
    s2 = abs(float(np.dot(stain2, dab_ref)))
    return stain1 if s1 >= s2 else stain2


def dab_od_via_macenko(rgb: np.ndarray) -> np.ndarray:
    """Estimate DAB channel OD via Macenko stain vector estimation and projection."""
    s1, s2 = macenko_estimate_stain_vectors(rgb)
    v_dab = pick_dab_vector(s1, s2)  # unit vector
    od = rgb_to_od(rgb)
    dab = od @ v_dab  # concentration along DAB vector
    dab = np.clip(dab, 0, None)
    return dab


def watershed_split(mask: np.ndarray, min_distance: int = 5, compactness: float = 0.0) -> np.ndarray:
    """Split touching objects via distance-transform watershed. Returns labeled image."""
    distance = ndi.distance_transform_edt(mask)
    # Local maxima as seeds
    try:
        # Newer versions return boolean mask when indices=False
        peaks = feature.peak_local_max(distance, labels=mask.astype(np.uint8), footprint=np.ones((3, 3), dtype=np.uint8), exclude_border=False, indices=False)
        markers = label(peaks)
    except TypeError:
        # Backward compatibility if indices arg not available
        coords = feature.peak_local_max(distance, labels=mask.astype(np.uint8), footprint=np.ones((3, 3), dtype=np.uint8), exclude_border=False)
        peaks = np.zeros_like(mask, dtype=bool)
        if coords.size:
            peaks[coords[:, 0], coords[:, 1]] = True
        markers = label(peaks)
    labeled = watershed(-distance, markers, mask=mask.astype(bool), compactness=float(compactness))
    return labeled


def apply_gating(target_mask: np.ndarray,
                 gate_mask: np.ndarray,
                 mode: str = "touching",
                 dilate_px: int = 2,
                 min_overlap_frac: float = 0.0) -> np.ndarray:
    """Restrict target components to those overlapping/touching the gate mask.

    mode:
      - 'intersect': overlap with gate directly
      - 'touching': use binary dilation on gate by 'dilate_px' before checking overlap
    min_overlap_frac: per-component fraction threshold (0..1) to keep (0 means any overlap)
    """
    from skimage.morphology import binary_dilation
    if mode not in ("intersect", "touching"):
        mode = "touching"
    gate = gate_mask.astype(bool)
    if mode == "touching" and dilate_px and dilate_px > 0:
        gate = binary_dilation(gate, footprint=disk(int(dilate_px)))
    lab = label(target_mask.astype(bool))
    if lab.max() <= 0:
        return target_mask & gate
    keep = np.zeros_like(target_mask, dtype=bool)
    for r in regionprops(lab):
        coords = r.coords
        overlap = gate[coords[:, 0], coords[:, 1]]
        if min_overlap_frac and min_overlap_frac > 0.0:
            frac = float(overlap.sum()) / float(max(1, coords.shape[0]))
            cond = frac >= float(min_overlap_frac)
        else:
            cond = bool(overlap.any())
        if cond:
            keep[lab == r.label] = True
    return keep
