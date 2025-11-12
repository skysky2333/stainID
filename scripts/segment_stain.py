#!/usr/bin/env python3
import os
import argparse
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

from .utils import (
    discover_images,
    parse_image_type,
    load_or_create_config,
    overlay_mask,
    overlay_multi_masks,
    label_and_measure,
    label_and_measure_with_raw,
    image_level_summary,
    nearest_neighbor_stats,
    grid_dispersion,
    save_mask_png,
    save_overlay_jpg,
    ensure_dir,
    read_image_rgb,
    DEFAULT_PIXEL_WIDTH_UM,
    DEFAULT_PIXEL_HEIGHT_UM,
    watershed_split,
    compute_oriented_channel,
    segment_channel_hed,
    apply_gating,
)


def process_image(path: str, out_root: str, cfg, pixel_width_um: float, pixel_height_um: float,
                  save_preview: bool = True, downsample_preview: int = 4) -> List[Dict]:
    img_rgb = read_image_rgb(path)
    base = os.path.splitext(os.path.basename(path))[0]
    t = cfg.type

    # Define overlay colors (RGB)
    default_colors = {
        "dab": (255, 0, 0),
        "hematoxylin": (0, 128, 255),
        "eosin": (0, 200, 0),
    }

    masks_by_channel = {}
    norm_by_channel = {}
    raw_by_channel = {}
    used_thr = {}

    # First pass: segment all requested channels
    for ch in cfg.channels_to_output:
        m, norm, thr, raw = segment_channel_hed(img_rgb, cfg, channel=ch)
        masks_by_channel[ch] = m
        norm_by_channel[ch] = norm
        raw_by_channel[ch] = raw
        used_thr[ch] = thr

    # Optional gating rules — apply only if enabled
    if bool(getattr(cfg, 'gating_enabled', False)) and getattr(cfg, 'gating_rules', None):
        for rule in cfg.gating_rules:
            tgt = str(rule.target)
            src = str(rule.by)
            if tgt in masks_by_channel and (src in masks_by_channel or src in cfg.channels_to_output or True):
                # Ensure gate mask exists (compute if not part of output)
                if src not in masks_by_channel:
                    m_src, _, _, _ = segment_channel_hed(img_rgb, cfg, channel=src)
                    masks_by_channel[src] = m_src
                gated = apply_gating(
                    masks_by_channel[tgt],
                    masks_by_channel[src],
                    mode=str(getattr(rule, 'mode', 'touching')),
                    dilate_px=int(getattr(rule, 'dilate_px', 2)),
                    min_overlap_frac=float(getattr(rule, 'min_overlap_frac', 0.0)),
                )
                masks_by_channel[tgt] = gated

    # Save per-channel masks, overlays, and features
    summaries: List[Dict] = []
    for ch in cfg.channels_to_output:
        m = masks_by_channel.get(ch, None)
        if m is None:
            continue
        # Watershed per-channel if enabled
        labeled = None
        try:
            ch_cfg = cfg.channels.get(ch, None)
        except Exception:
            ch_cfg = None
        if ch_cfg and getattr(ch_cfg, 'watershed', None) and bool(ch_cfg.watershed.enabled):
            labeled = watershed_split(m, min_distance=int(ch_cfg.watershed.min_distance), compactness=float(ch_cfg.watershed.compactness))

        # Features with raw + norm of this channel
        labeled, df_sites = label_and_measure_with_raw(
            m,
            norm_by_channel[ch],
            raw_by_channel[ch],
            pixel_width_um,
            pixel_height_um,
            labeled_in=labeled,
        )

        # Save mask
        mask_path = os.path.join(out_root, "masks", t, ch, f"{base}_mask.png")
        save_mask_png(mask_path, m)

        # Save per-channel overlay
        if save_preview:
            img_ov = img_rgb
            m_ov = m
            if downsample_preview and downsample_preview > 1:
                h, w = img_rgb.shape[:2]
                img_ov = cv2.resize(img_rgb, (w // downsample_preview, h // downsample_preview), interpolation=cv2.INTER_AREA)
                m_ov = cv2.resize(m_ov.astype(np.uint8), (w // downsample_preview, h // downsample_preview), interpolation=cv2.INTER_NEAREST).astype(bool)
            overlay = overlay_mask(img_ov, m_ov, color=default_colors.get(ch, (255, 0, 0)), alpha=0.45, draw_edges=True)
            overlay_path = os.path.join(out_root, "overlays", t, ch, f"{base}_overlay.jpg")
            save_overlay_jpg(overlay_path, overlay, quality=90)

        # per-site features
        features_path = os.path.join(out_root, "features", t, ch, f"{base}_sites.csv")
        ensure_dir(os.path.dirname(features_path))
        df_sites.insert(0, "channel", ch)
        df_sites.insert(0, "image", base)
        df_sites.to_csv(features_path, index=False)

        # image-level summary & spatial metrics
        points = df_sites[["centroid_row", "centroid_col"]].values if df_sites.shape[0] > 0 else np.zeros((0, 2))
        h, w = m.shape
        nn = nearest_neighbor_stats(points, img_area_px=h * w,
                                    pixel_width_um=pixel_width_um,
                                    pixel_height_um=pixel_height_um)
        grid = grid_dispersion(points, (h, w), K=10)
        summary = image_level_summary(df_sites, m, (h, w), pixel_width_um, pixel_height_um)
        summary.update(nn)
        summary.update(grid)
        summary.update({
            "image": base,
            "type": t,
            "channel": ch,
            "threshold_used": used_thr.get(ch, 0.0),
        })
        summaries.append(summary)

    # Combined overlay of selected channels
    if save_preview and any(masks_by_channel.get(ch) is not None for ch in cfg.channels_to_output):
        # Downsample once for composite
        img_ov = img_rgb
        ds_masks = {}
        if downsample_preview and downsample_preview > 1:
            h, w = img_rgb.shape[:2]
            img_ov = cv2.resize(img_rgb, (w // downsample_preview, h // downsample_preview), interpolation=cv2.INTER_AREA)
            for ch in cfg.channels_to_output:
                mm = masks_by_channel.get(ch)
                if mm is None:
                    continue
                ds = cv2.resize(mm.astype(np.uint8), (w // downsample_preview, h // downsample_preview), interpolation=cv2.INTER_NEAREST).astype(bool)
                ds_masks[ch] = ds
        else:
            for ch in cfg.channels_to_output:
                if masks_by_channel.get(ch) is not None:
                    ds_masks[ch] = masks_by_channel[ch]
        overlay_all = overlay_multi_masks(img_ov, ds_masks, default_colors, alpha=0.45, draw_edges=True, edge_thickness=3)
        overlay_all_path = os.path.join(out_root, "overlays", t, f"{base}_overlay_all.jpg")
        save_overlay_jpg(overlay_all_path, overlay_all, quality=90)

    return summaries


def main():
    p = argparse.ArgumentParser(description="Batch segmentation of DAB/brown stain via HED deconvolution.")
    p.add_argument("--input_dir", default="origianl_data/out_seg/matched_output", help="Input image directory")
    p.add_argument("--out_root", default="results_stainID", help="Output root directory")
    p.add_argument("--types", nargs="*", help="Optional list of numeric type prefixes to process")
    p.add_argument("--pixel_width_um", type=float, default=DEFAULT_PIXEL_WIDTH_UM, help="Pixel width in micrometers")
    p.add_argument("--pixel_height_um", type=float, default=DEFAULT_PIXEL_HEIGHT_UM, help="Pixel height in micrometers")
    p.add_argument("--no_preview", action="store_true", help="Disable saving overlay previews")
    p.add_argument("--downsample_preview", type=int, default=4, help="Downsample factor for overlays")
    args = p.parse_args()

    ensure_dir(args.out_root)
    images = discover_images(args.input_dir)
    if not images:
        raise SystemExit(f"No images found in {args.input_dir}")

    # group by type prefix
    grouped: Dict[str, List[str]] = defaultdict(list)
    for pth in images:
        t = parse_image_type(os.path.basename(pth))
        grouped[t].append(pth)

    selected_types = set(args.types) if args.types else set(grouped.keys())

    all_summaries: List[Dict] = []
    for t in sorted(selected_types):
        cfg = load_or_create_config(args.out_root, t)
        print(f"Processing type {t}: {len(grouped.get(t, []))} images; engine={cfg.engine}")
        for idx, img_path in enumerate(sorted(grouped.get(t, []))):
            print(f"  [{idx+1}/{len(grouped[t])}] {os.path.basename(img_path)}")
            summaries = process_image(
                img_path,
                args.out_root,
                cfg,
                pixel_width_um=args.pixel_width_um,
                pixel_height_um=args.pixel_height_um,
                save_preview=not args.no_preview,
                downsample_preview=args.downsample_preview,
            )
            all_summaries.extend(summaries)

    # write per-type summaries
    df = pd.DataFrame(all_summaries)
    if df.shape[0] > 0:
        for t, sub in df.groupby("type"):
            outp = os.path.join(args.out_root, "summary", f"{t}_image_metrics.csv")
            ensure_dir(os.path.dirname(outp))
            sub.sort_values("image").to_csv(outp, index=False)
        # Also combined
        outp_all = os.path.join(args.out_root, "summary", "all_types_image_metrics.csv")
        ensure_dir(os.path.dirname(outp_all))
        df.sort_values(["type", "image"]).to_csv(outp_all, index=False)


if __name__ == "__main__":
    # Lazy import to avoid heavy deps on import time
    import cv2  # noqa: F401  (used in process_image overlay)
    main()
