#!/usr/bin/env python3
import os
from typing import List, Dict

import numpy as np

import streamlit as st
import sys
import pathlib

# Ensure repository root on sys.path so absolute import works when launching via Streamlit from other CWDs
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils import (
    discover_images,
    parse_image_type,
    ensure_dir,
    read_image_rgb,
    load_or_create_config,
    save_config,
    overlay_mask,
    overlay_multi_masks,
    label_and_measure,
    watershed_split,
    compute_valid_mask,
    normalize_channel_on_mask,
    choose_dab_orientation,
    segment_channel_hed,
    apply_gating,
)


def list_types(input_dir: str) -> List[str]:
    images = discover_images(input_dir)
    types = sorted({parse_image_type(os.path.basename(p)) for p in images})
    return types


def find_sample_images(input_dir: str, t: str) -> List[str]:
    images = discover_images(input_dir)
    return [p for p in images if parse_image_type(os.path.basename(p)) == t]


def main():
    st.set_page_config(page_title="StainID Calibration", layout="wide")
    st.title("StainID Calibration (HED multi-channel)")

    with st.sidebar:
        input_dir = st.text_input("Input directory", value="data/TMA_core_exports")
        out_root = st.text_input("Output root", value="results_stainID")
        ensure_dir(out_root)

        types = list_types(input_dir)
        if not types:
            st.error(f"No images found in {input_dir}")
            st.stop()
        t = st.selectbox("Image type (numeric prefix)", options=types)

        cfg = load_or_create_config(out_root, t)

        st.markdown("### Engine")
        engine = st.selectbox("Engine", options=["hed", "hsv"], index=0 if getattr(cfg, 'engine', 'hed') == 'hed' else 1, key=f"engine_{t}")
        cfg.engine = engine
        if cfg.engine != 'hed':
            st.markdown("HSV thresholding")
            cfg.hsv_h_low = float(st.slider("Hue low (0..1)", 0.0, 1.0, float(getattr(cfg, 'hsv_h_low', 0.05)), 0.001))
            cfg.hsv_h_high = float(st.slider("Hue high (0..1)", 0.0, 1.0, float(getattr(cfg, 'hsv_h_high', 0.15)), 0.001))
            cfg.hsv_s_min = float(st.slider("Saturation min (0..1)", 0.0, 1.0, float(getattr(cfg, 'hsv_s_min', 0.2)), 0.001))
            cfg.hsv_v_min = float(st.slider("Value min (0..1)", 0.0, 1.0, float(getattr(cfg, 'hsv_v_min', 0.0)), 0.001))
            cfg.hsv_v_max = float(st.slider("Value max (0..1)", 0.0, 1.0, float(getattr(cfg, 'hsv_v_max', 0.9)), 0.001))
        else:
            st.markdown("### Channels to output (batch)")
            all_channels = ["dab", "hematoxylin", "eosin"]
            sel = st.multiselect("Select channels", options=all_channels, default=[c for c in cfg.channels_to_output if c in all_channels] or ["dab"], key=f"channels_to_output_{t}")
            cfg.channels_to_output = sel

            st.markdown("### Per-channel settings")
            for ch in all_channels:
                with st.expander(f"Channel: {ch.upper()}"):

                    # Threshold mode and params
                    cfg.channels[ch].threshold.normalize = bool(st.checkbox("Normalize channel [0..1]", value=bool(cfg.channels[ch].threshold.normalize), key=f"ch_norm_{t}_{ch}"))
                    is_norm = bool(cfg.channels[ch].threshold.normalize)
                    if is_norm:
                        mode_label = "Threshold mode"
                        th_mode = st.selectbox(mode_label, options=["auto", "absolute (normalized)"], index=0 if cfg.channels[ch].threshold.absolute_normalized is None else 1, key=f"thr_mode_{t}_{ch}")
                        if th_mode.startswith("absolute"):
                            import math
                            default_norm = float(cfg.channels[ch].threshold.absolute_normalized or 0.5)
                            default_norm = max(default_norm, 1e-6)
                            log_min, log_max = -6.0, 0.0
                            default_log = float(np.clip(math.log10(default_norm), log_min, log_max))
                            log_t = st.slider("Absolute threshold (normalized, log10)", log_min, log_max, default_log, 0.01, key=f"abs_norm_{t}_{ch}")
                            cfg.channels[ch].threshold.absolute_normalized = float(10.0 ** log_t)
                            cfg.channels[ch].threshold.absolute_raw = None
                        else:
                            cfg.channels[ch].threshold.absolute_normalized = None
                        # Show method/scale only in auto mode
                        if th_mode == "auto":
                            cfg.channels[ch].threshold.method = st.selectbox(
                                "Threshold method",
                                options=["otsu", "yen", "li"],
                                index=["otsu", "yen", "li"].index(cfg.channels[ch].threshold.method) if cfg.channels[ch].threshold.method in ["otsu", "yen", "li"] else 0,
                                key=f"thr_method_{t}_{ch}",
                            )
                            cfg.channels[ch].threshold.scale = float(st.slider("Threshold scale", 0.2, 2.0, float(cfg.channels[ch].threshold.scale), 0.01, key=f"thr_scale_{t}_{ch}"))
                    else:
                        th_mode = st.selectbox("Threshold mode", options=["auto", "absolute (raw)"] , index=0 if cfg.channels[ch].threshold.absolute_raw is None else 1, key=f"thr_mode_{t}_{ch}")
                        if th_mode.startswith("absolute"):
                            import math
                            raw_val = cfg.channels[ch].threshold.absolute_raw
                            if raw_val is None:
                                raw_val = float(10.0 ** (-1.5)) if ch == 'dab' else float(10.0 ** (-2.0))
                            default_raw = float(raw_val)
                            log_min, log_max = -3.0, 1.0
                            default_log = float(np.clip(math.log10(max(default_raw, 1e-12)), log_min, log_max))
                            log_t = st.slider("Absolute threshold (raw OD, log10)", log_min, log_max, default_log, 0.01, key=f"abs_raw_{t}_{ch}")
                            cfg.channels[ch].threshold.absolute_raw = float(10.0 ** log_t)
                            cfg.channels[ch].threshold.absolute_normalized = None
                        else:
                            cfg.channels[ch].threshold.absolute_raw = None
                        if th_mode == "auto":
                            cfg.channels[ch].threshold.method = st.selectbox(
                                "Threshold method",
                                options=["otsu", "yen", "li"],
                                index=["otsu", "yen", "li"].index(cfg.channels[ch].threshold.method) if cfg.channels[ch].threshold.method in ["otsu", "yen", "li"] else 0,
                                key=f"thr_method_{t}_{ch}",
                            )
                            cfg.channels[ch].threshold.scale = float(st.slider("Threshold scale", 0.2, 2.0, float(cfg.channels[ch].threshold.scale), 0.01, key=f"thr_scale_{t}_{ch}"))
                    inv_choice = st.selectbox(
                        "Orientation",
                        options=["auto", "normal", "invert"],
                        index=0 if cfg.channels[ch].threshold.invert is None else (2 if cfg.channels[ch].threshold.invert else 1),
                        key=f"inv_{t}_{ch}"
                    )
                    cfg.channels[ch].threshold.invert = None if inv_choice == "auto" else (inv_choice == "invert")
                    if ch == 'dab':
                        cfg.channels[ch].threshold.stain_estimation = st.selectbox(
                            "Stain estimation (DAB)",
                            options=["default", "macenko"],
                            index=0 if cfg.channels[ch].threshold.stain_estimation == 'default' else 1,
                            key=f"dab_est_{ch}"
                        )

                    st.markdown("— Morphology")
                    cfg.channels[ch].morph.open_radius = int(st.number_input("Opening radius (px)", min_value=0, max_value=50, value=int(cfg.channels[ch].morph.open_radius), key=f"open_{t}_{ch}"))
                    cfg.channels[ch].morph.close_radius = int(st.number_input("Closing radius (px)", min_value=0, max_value=50, value=int(cfg.channels[ch].morph.close_radius), key=f"close_{t}_{ch}"))
                    cfg.channels[ch].morph.remove_small = int(st.number_input("Remove small objects (px)", min_value=0, max_value=10000, value=int(cfg.channels[ch].morph.remove_small), key=f"rm_small_{t}_{ch}"))
                    cfg.channels[ch].morph.fill_holes = bool(st.checkbox("Fill holes", value=bool(cfg.channels[ch].morph.fill_holes), key=f"fill_{t}_{ch}"))
                    cfg.channels[ch].morph.fill_holes_max_area = int(st.number_input("Fill holes up to area (px, 0 = fill all)", min_value=0, max_value=1000000, value=int(getattr(cfg.channels[ch].morph, 'fill_holes_max_area', 64) or 0), key=f"fill_area_{t}_{ch}"))
                    cfg.channels[ch].morph.clear_border = bool(st.checkbox("Clear border", value=bool(cfg.channels[ch].morph.clear_border), key=f"clear_{t}_{ch}"))
                    cfg.channels[ch].morph.remove_large_connected = bool(st.checkbox("Remove very large connected area (auto)", value=bool(cfg.channels[ch].morph.remove_large_connected), key=f"rm_large_{t}_{ch}"))
                    cfg.channels[ch].morph.large_area_frac = float(st.slider("Large span threshold (fraction of larger image dimension)", 0.0, 0.5, float(cfg.channels[ch].morph.large_area_frac), 0.001, format="%.3f", key=f"large_frac_{t}_{ch}"))

                    st.markdown("— Watershed (split touching)")
                    cfg.channels[ch].watershed.enabled = bool(st.checkbox("Enable watershed split for this channel", value=bool(cfg.channels[ch].watershed.enabled), key=f"ws_enabled_{t}_{ch}"))
                    if cfg.channels[ch].watershed.enabled:
                        cfg.channels[ch].watershed.min_distance = int(st.number_input("Min distance (px)", min_value=1, max_value=100, value=int(cfg.channels[ch].watershed.min_distance), key=f"ws_min_{t}_{ch}"))
                        cfg.channels[ch].watershed.compactness = float(st.number_input("Compactness", min_value=0.0, max_value=10.0, value=float(cfg.channels[ch].watershed.compactness), step=0.1, key=f"ws_comp_{t}_{ch}"))
        # Per-channel orientation and morphology configured above

        # Per-channel orientation and morphology configured above

        st.markdown("### Gating (off by default)")
        # Robustly detect if gating is currently configured, supporting both dicts and dataclass objects
        def _has_dab_gate(grules):
            try:
                for g in (grules or []):
                    tgt = getattr(g, 'target', None)
                    if tgt is None and isinstance(g, dict):
                        tgt = g.get('target')
                    if str(tgt) == 'dab':
                        return True
            except Exception:
                pass
            return False
        # Gate toggle uses explicit flag so rules can persist while disabled
        gate_on = bool(st.checkbox(
            "Gate DAB by Hematoxylin (touching/intersect)",
            value=bool(getattr(cfg, 'gating_enabled', False)),
            key=f"gate_dab_by_hema_{t}"
        ))
        cfg.gating_enabled = gate_on
        if gate_on:
            # Use existing DAB gating values if present; otherwise default to touching, dilation 15, min_overlap 0.0
            existing = None
            try:
                for gg in (getattr(cfg, 'gating_rules', []) or []):
                    tgt = getattr(gg, 'target', None)
                    if tgt is None and isinstance(gg, dict):
                        tgt = gg.get('target')
                    if str(tgt) == 'dab':
                        existing = gg
                        break
            except Exception:
                existing = None

            def _get(obj, key, default):
                try:
                    if hasattr(obj, key):
                        return getattr(obj, key)
                    if isinstance(obj, dict):
                        return obj.get(key, default)
                except Exception:
                    pass
                return default

            default_mode = _get(existing, 'mode', 'touching') if existing is not None else 'touching'
            default_dil = int(_get(existing, 'dilate_px', 15)) if existing is not None else 15
            default_min_frac = float(_get(existing, 'min_overlap_frac', 0.0)) if existing is not None else 0.0

            mode = st.selectbox("Mode", options=["touching", "intersect"], index=(0 if str(default_mode) == 'touching' else 1), key=f"gate_mode_{t}")
            dil = int(st.number_input("Dilation (px) for touching", min_value=0, max_value=50, value=default_dil, key=f"gate_dil_{t}"))
            min_frac = float(st.slider("Min overlap fraction per region", 0.0, 1.0, default_min_frac, 0.01, key=f"gate_minfrac_{t}"))
            cfg.gating_rules = [
                {
                    "target": "dab",
                    "by": "hematoxylin",
                    "mode": mode,
                    "dilate_px": dil,
                    "min_overlap_frac": min_frac,
                }
            ]

        st.markdown("### Background mask")
        cfg.pre_mask_ignore_black = bool(st.checkbox("Ignore pure black borders", value=bool(getattr(cfg, 'pre_mask_ignore_black', True)), key=f"mask_black_{t}"))
        cfg.pre_mask_black_rgb_thresh = int(st.number_input("Black threshold (per channel 0..255)", min_value=0, max_value=50, value=int(getattr(cfg, 'pre_mask_black_rgb_thresh', 5)), key=f"mask_black_thr_{t}"))
        cfg.pre_mask_ignore_white = bool(st.checkbox("Ignore white background", value=bool(getattr(cfg, 'pre_mask_ignore_white', True)), key=f"mask_white_{t}"))
        cfg.pre_mask_white_v_thresh = float(st.slider("White V threshold (HSV)", 0.0, 1.0, float(getattr(cfg, 'pre_mask_white_v_thresh', 0.90)), 0.01, key=f"mask_white_v_{t}"))
        cfg.pre_mask_white_s_max = float(st.slider("White S max (HSV)", 0.0, 1.0, float(getattr(cfg, 'pre_mask_white_s_max', 0.25)), 0.01, key=f"mask_white_s_{t}"))

        if st.button("Save config"):
            save_config(cfg, out_root)
            st.success(f"Saved config for type {t}")

    # Main area: select sample image
    st.subheader(f"Preview for type {t}")
    sample_paths = find_sample_images(input_dir, t)
    if not sample_paths:
        st.info("No images for this type.")
        st.stop()

    sample_names = [os.path.basename(p) for p in sample_paths]
    s_idx = st.selectbox("Sample image", options=list(range(len(sample_names))), format_func=lambda i: sample_names[i], index=0)
    sel_path = sample_paths[s_idx]

    st.caption("For very large images, consider generating previews with scripts/make_previews.py to speed up UI.")
    use_preview = st.checkbox("Use JPEG preview if available", value=True, key="use_prev_main")
    downsample_factor = int(st.number_input("Downsample factor (fallback)", min_value=1, max_value=32, value=8, key="ds_factor_main"))

    # Paths and sizes
    preview_path = os.path.join(out_root, "previews", t, "source", os.path.splitext(os.path.basename(sel_path))[0] + "_preview.jpg")
    from PIL import Image
    with Image.open(sel_path) as im_full:
        full_w, full_h = im_full.size

    # Main preview image (preview JPEG or downsampled original)
    if use_preview and os.path.exists(preview_path):
        img_main = read_image_rgb(preview_path)
        prev_h, prev_w = img_main.shape[:2]
    else:
        img_main = read_image_rgb(sel_path, downsample_factor=downsample_factor)
        prev_h, prev_w = img_main.shape[:2]
    s_h_main = full_h / float(prev_h)
    s_w_main = full_w / float(prev_w)

    # Scale per-channel morphology for main preview only (do not persist)
    from copy import deepcopy
    cfg_prev_main = deepcopy(cfg)
    radius_scale_main = float(np.sqrt(s_h_main * s_w_main))
    area_scale_main = float(s_h_main * s_w_main)
    try:
        for ch, ch_cfg in cfg_prev_main.channels.items():
            ch_cfg.morph.open_radius = int(max(0, round(ch_cfg.morph.open_radius / radius_scale_main)))
            ch_cfg.morph.close_radius = int(max(0, round(ch_cfg.morph.close_radius / radius_scale_main)))
            ch_cfg.morph.remove_small = int(max(0, round(ch_cfg.morph.remove_small / area_scale_main)))
            ch_cfg.morph.large_span_ref_dim = None
            # Scale watershed min_distance per channel if enabled
            try:
                if getattr(ch_cfg, 'watershed', None) and bool(ch_cfg.watershed.enabled):
                    ch_cfg.watershed.min_distance = int(max(1, round(ch_cfg.watershed.min_distance / radius_scale_main)))
            except Exception:
                pass
    except Exception:
        pass

    # Crop controls (place before any images)
    st.markdown("---")
    st.subheader("Original-resolution crop preview")
    crop_size_px = int(st.number_input("Crop size (px, square)", min_value=128, max_value=8192, value=1024, step=64, key="crop_size"))
    crop_row_frac = float(st.slider("Crop center row (0..1)", 0.0, 1.0, 0.5, 0.01, key="crop_row"))
    crop_col_frac = float(st.slider("Crop center col (0..1)", 0.0, 1.0, 0.5, 0.01, key="crop_col"))
    crop_size = int(min(max(1, crop_size_px), min(full_h, full_w)))
    cy = int(round(crop_row_frac * max(0, full_h - 1)))
    cx = int(round(crop_col_frac * max(0, full_w - 1)))
    y0 = int(max(0, min(full_h - crop_size, cy - crop_size // 2)))
    x0 = int(max(0, min(full_w - crop_size, cx - crop_size // 2)))
    y1 = y0 + crop_size
    x1 = x0 + crop_size
    img_full = read_image_rgb(sel_path)
    img_crop = img_full[y0:y1, x0:x1]
    st.caption(f"Original-resolution crop: y={y0}:{y1}, x={x0}:{x1}  size={crop_size}x{crop_size}")

    channel_colors = {"dab": (255, 0, 0), "hematoxylin": (0, 128, 255), "eosin": (0, 200, 0)}
    preview_channels = [c for c in getattr(cfg, 'channels_to_output', []) if c in channel_colors] or ["dab"]
    draw_edges = st.checkbox("Draw edges only", value=False, key="draw_edges")
    edge_thickness = 6 if draw_edges else 3
    if draw_edges:
        edge_thickness = int(st.number_input("Edge thickness (px)", min_value=1, max_value=50, value=edge_thickness, key="edge_thick"))

    masks_main: Dict[str, np.ndarray] = {}
    norms_main: Dict[str, np.ndarray] = {}
    raws_main: Dict[str, np.ndarray] = {}
    used_t_main: Dict[str, float] = {}
    for ch in preview_channels:
        m, norm, used_t, raw = segment_channel_hed(img_main, cfg_prev_main, channel=ch)
        masks_main[ch] = m
        norms_main[ch] = norm
        raws_main[ch] = raw
        used_t_main[ch] = used_t

    # Apply gating (main)
    if bool(getattr(cfg_prev_main, 'gating_enabled', False)) and cfg_prev_main.gating_rules:
        for gr in cfg_prev_main.gating_rules:
            tgt = getattr(gr, 'target', 'dab') if not isinstance(gr, dict) else gr.get('target', 'dab')
            src = getattr(gr, 'by', 'hematoxylin') if not isinstance(gr, dict) else gr.get('by', 'hematoxylin')
            mode = getattr(gr, 'mode', 'touching') if not isinstance(gr, dict) else gr.get('mode', 'touching')
            dil = int(getattr(gr, 'dilate_px', 2) if not isinstance(gr, dict) else gr.get('dilate_px', 2))
            min_frac = float(getattr(gr, 'min_overlap_frac', 0.0) if not isinstance(gr, dict) else gr.get('min_overlap_frac', 0.0))
            if tgt in masks_main and (src in masks_main or src in ["dab", "hematoxylin", "eosin"]):
                if src not in masks_main:
                    m_src, _, _, _ = segment_channel_hed(img_main, cfg_prev_main, channel=src)
                    masks_main[src] = m_src
                masks_main[tgt] = apply_gating(masks_main[tgt], masks_main[src], mode=mode, dilate_px=dil, min_overlap_frac=min_frac)

    overlay_main = overlay_multi_masks(img_main, {k: masks_main[k] for k in preview_channels if k in masks_main}, channel_colors, alpha=0.45, draw_edges=draw_edges, edge_thickness=edge_thickness)

    # Prepare crop config and segment DAB crop before rendering any crop images
    # Config for crop: no scaling; keep span ref to full image
    cfg_prev_crop = deepcopy(cfg)
    try:
        for ch, ch_cfg in cfg_prev_crop.channels.items():
            ch_cfg.morph.large_span_ref_dim = int(max(full_h, full_w))
    except Exception:
        pass
    # Segment selected channels for crop
    masks_crop: Dict[str, np.ndarray] = {}
    norms_crop: Dict[str, np.ndarray] = {}
    raws_crop: Dict[str, np.ndarray] = {}
    used_t_crop: Dict[str, float] = {}
    for ch in preview_channels:
        m, norm, used_t, raw = segment_channel_hed(img_crop, cfg_prev_crop, channel=ch)
        masks_crop[ch] = m
        norms_crop[ch] = norm
        raws_crop[ch] = raw
        used_t_crop[ch] = used_t
    # Apply gating to crop if any
    if bool(getattr(cfg_prev_crop, 'gating_enabled', False)) and cfg_prev_crop.gating_rules:
        for gr in cfg_prev_crop.gating_rules:
            tgt = getattr(gr, 'target', 'dab') if not isinstance(gr, dict) else gr.get('target', 'dab')
            src = getattr(gr, 'by', 'hematoxylin') if not isinstance(gr, dict) else gr.get('by', 'hematoxylin')
            mode = getattr(gr, 'mode', 'touching') if not isinstance(gr, dict) else gr.get('mode', 'touching')
            dil = int(getattr(gr, 'dilate_px', 2) if not isinstance(gr, dict) else gr.get('dilate_px', 2))
            min_frac = float(getattr(gr, 'min_overlap_frac', 0.0) if not isinstance(gr, dict) else gr.get('min_overlap_frac', 0.0))
            if tgt in masks_crop and (src in masks_crop or src in ["dab", "hematoxylin", "eosin"]):
                if src not in masks_crop:
                    m_src, _, _, _ = segment_channel_hed(img_crop, cfg_prev_crop, channel=src)
                    masks_crop[src] = m_src
                masks_crop[tgt] = apply_gating(masks_crop[tgt], masks_crop[src], mode=mode, dilate_px=dil, min_overlap_frac=min_frac)
    overlay_crop = overlay_multi_masks(img_crop, {k: masks_crop[k] for k in preview_channels if k in masks_crop}, channel_colors, alpha=0.45, draw_edges=draw_edges, edge_thickness=edge_thickness)

    # Row 1: Original (preview/downsample) | Crop (original-res)
    row1_left, row1_right = st.columns(2)
    with row1_left:
        st.image(img_main, caption="Original (preview/downsample)", width='stretch')

    # Row 2: DAB overlays (left: original overlay, right: crop overlay)
    row2_left, row2_right = st.columns(2)
    with row2_left:
        st.image(
            overlay_main,
            caption=", ".join([f"{ch}: t={used_t_main.get(ch, 0):.3f}" for ch in preview_channels]),
            width='stretch',
        )
    # Crop overlay will be computed below together with crop masks

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    # Selector for which channel to show in masks and histograms
    detail_ch = st.selectbox("Preview channel for detail", options=preview_channels, index=0, key="detail_ch")
    # Precompute masked signal (main) for selected channel
    sig_main = (np.clip(norms_main[detail_ch], 0.0, 1.0) * 255).astype(np.uint8)
    masked_sig_main = np.zeros_like(sig_main)
    masked_sig_main[masks_main[detail_ch]] = sig_main[masks_main[detail_ch]]
    # Row 3: Main mask | Crop mask
    row3_left, row3_right = st.columns(2)
    with row3_left:
        st.image((masks_main[detail_ch].astype(np.uint8) * 255), caption=f"Mask ({detail_ch})", width='stretch', clamp=True)
    with row3_right:
        st.image((masks_crop[detail_ch].astype(np.uint8) * 255), caption=f"Mask (crop, {detail_ch})", width='stretch', clamp=True)


    # Complete Row 1: show crop image on the right
    with row1_right:
        st.image(img_crop, caption="Original (crop)", width='stretch')
    # Complete Row 2: show crop overlay on the right
    with row2_right:
        st.image(
            overlay_crop,
            caption=", ".join([f"{ch}: t={used_t_crop.get(ch, 0):.3f}" for ch in preview_channels]),
            width='stretch',
        )

    # Row 4: Signal channel (masked) [dab] | Signal channel (masked) [crop, dab]
    m = masks_crop[detail_ch]
    norm = norms_crop[detail_ch]
    crop_bL, crop_bR = st.columns(2)
    with crop_bL:
        st.image(masked_sig_main, caption="Signal channel (masked) [dab]", width='stretch', clamp=True)
    with crop_bR:
        sig = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
        masked_sig = np.zeros_like(sig)
        masked_sig[m] = sig[m]
        st.image(masked_sig, caption="Signal channel (masked) [crop, dab]", width='stretch', clamp=True)

    # Row 5: Histograms (left: original, right: crop)
    valid_mask_main = compute_valid_mask(img_main, cfg_prev_main)
    valid_mask_crop = compute_valid_mask(img_crop, cfg_prev_crop)
    # Build main histogram figure
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    fig_main, axm = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    if cfg_prev_main.channels[detail_ch].threshold.normalize:
        xm = norms_main[detail_ch]
        arrm = np.clip(xm.flatten(), 1e-12, None)
        axm.hist(arrm, bins=100, color='gray')
        axm.set_xscale('log'); axm.set_yscale('log')
        pos = arrm[arrm > 0]
        xmin_data = float(np.percentile(pos, 0.1)) if pos.size > 0 else float(arrm.min())
        xmin_thr = float(max(used_t_main.get(detail_ch, 1e-12), 1e-12))
        xmin = float(max(1e-12, min(xmin_data, xmin_thr) * 0.5))
        xmax = float(max(arrm.max(), max(used_t_main.get(detail_ch, 1e-12), 1e-12)))
        axm.set_xlim(xmin, xmax)
        axm.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=12))
        axm.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=120))
        axm.xaxis.set_minor_formatter(mticker.NullFormatter())
        axm.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{np.log10(max(v, 1e-12)):.2f}"))
        axm.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=12))
        axm.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=120))
        axm.yaxis.set_minor_formatter(mticker.NullFormatter())
        axm.set_xlabel("log10(normalized intensity)"); axm.set_ylabel("count (log)")
        axm.axvline(max(used_t_main.get(detail_ch, 1e-12), 1e-6), color='red')
        axm.set_title(f"Histogram (normalized, log-log) [original] ({detail_ch})")
    else:
        xm = raws_main[detail_ch]
        arrm = np.clip(xm.flatten(), 1e-12, None)
        axm.hist(arrm, bins=100, color='gray')
        axm.set_xscale('log'); axm.set_yscale('log')
        xmax = float(max(arrm.max(), max(used_t_main.get(detail_ch, 1e-12), 1e-12)))
        pos = arrm[arrm > 0]
        xmin_data = float(np.percentile(pos, 0.1)) if pos.size > 0 else float(arrm.min())
        xmin_thr = float(max(used_t_main.get(detail_ch, 1e-12), 1e-12))
        xmin = float(max(1e-12, min(xmin_data, xmin_thr) * 0.5))
        axm.set_xlim(xmin, xmax)
        axm.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=12))
        axm.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=120))
        axm.xaxis.set_minor_formatter(mticker.NullFormatter())
        axm.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{np.log10(max(v, 1e-12)):.2f}"))
        axm.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=12))
        axm.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=120))
        axm.yaxis.set_minor_formatter(mticker.NullFormatter())
        axm.set_xlabel("log10(OD)"); axm.set_ylabel("count (log)")
        axm.axvline(max(used_t_main.get(detail_ch, 1e-12), 1e-6), color='red')
        axm.set_title(f"Histogram (raw OD, log-log) [original] ({detail_ch})")

    # Build crop histogram figure
    fig_crop, axc = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    if cfg_prev_crop.channels[detail_ch].threshold.normalize:
        xc = norms_crop[detail_ch]
        arrc = np.clip(xc.flatten(), 1e-12, None)
        axc.hist(arrc, bins=100, color='gray')
        axc.set_xscale('log'); axc.set_yscale('log')
        pos = arrc[arrc > 0]
        xmin_data = float(np.percentile(pos, 0.1)) if pos.size > 0 else float(arrc.min())
        xmin_thr = float(max(used_t_crop.get(detail_ch, 1e-12), 1e-12))
        xmin = float(max(1e-12, min(xmin_data, xmin_thr) * 0.5))
        xmax = float(max(arrc.max(), max(used_t_crop.get(detail_ch, 1e-12), 1e-12)))
        axc.set_xlim(xmin, xmax)
        axc.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=12))
        axc.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=120))
        axc.xaxis.set_minor_formatter(mticker.NullFormatter())
        axc.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{np.log10(max(v, 1e-12)):.2f}"))
        axc.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=12))
        axc.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=120))
        axc.yaxis.set_minor_formatter(mticker.NullFormatter())
        axc.set_xlabel("log10(normalized intensity)"); axc.set_ylabel("count (log)")
        axc.axvline(max(used_t_crop.get(detail_ch, 1e-12), 1e-6), color='red')
        axc.set_title(f"Histogram (normalized, log-log) [crop] ({detail_ch})")
    else:
        xc = raws_crop[detail_ch]
        arrc = np.clip(xc.flatten(), 1e-12, None)
        axc.hist(arrc, bins=100, color='gray')
        axc.set_xscale('log'); axc.set_yscale('log')
        xmax = float(max(arrc.max(), max(used_t_crop.get(detail_ch, 1e-12), 1e-12)))
        pos = arrc[arrc > 0]
        xmin_data = float(np.percentile(pos, 0.1)) if pos.size > 0 else float(arrc.min())
        xmin_thr = float(max(used_t_crop.get(detail_ch, 1e-12), 1e-12))
        xmin = float(max(1e-12, min(xmin_data, xmin_thr) * 0.5))
        axc.set_xlim(xmin, xmax)
        axc.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=12))
        axc.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=120))
        axc.xaxis.set_minor_formatter(mticker.NullFormatter())
        axc.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{np.log10(max(v, 1e-12)):.2f}"))
        axc.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=12))
        axc.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=120))
        axc.yaxis.set_minor_formatter(mticker.NullFormatter())
        axc.set_xlabel("log10(OD)"); axc.set_ylabel("count (log)")
        axc.axvline(max(used_t_crop.get(detail_ch, 1e-12), 1e-6), color='red')
        axc.set_title(f"Histogram (raw OD, log-log) [crop] ({detail_ch})")

    hist_left, hist_right = st.columns(2)
    with hist_left:
        st.pyplot(fig_main)
    with hist_right:
        st.pyplot(fig_crop)

    st.markdown("---")
    st.write("Tip: tweak parameters on the left and click 'Save config'. Then run the batch CLI: `python -m scripts.segment_stain --types", t, "` .")


if __name__ == "__main__":
    main()
