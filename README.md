StainID: DAB/brown stain segmentation toolkit

Overview
- Per-type calibration using HED color deconvolution (DAB channel) with adjustable thresholds and morphology.
- Batch processing to produce masks, overlays, per-site features, and image-level spatial metrics.
- Streamlit calibration UI with preview/fallback for large images.

Install (example)
- Python 3.9+
- pip install: numpy, scipy, pandas, scikit-image, opencv-python-headless, pyyaml, streamlit, matplotlib, pillow

Folder conventions
- Input images default: `origianl_data/out_seg/matched_output`
- Output root (default): `results_stainID`
  - configs/<TYPE>.yaml
  - masks/<TYPE>/<channel>/<name>_mask.png (per-channel)
  - overlays/<TYPE>/<channel>/<name>_overlay.jpg (per-channel)
  - overlays/<TYPE>/<name>_overlay_all.jpg (composite of selected channels)
  - features/<TYPE>/<channel>/<name>_sites.csv (per-channel)
  - summary/<TYPE>_image_metrics.csv (with a 'channel' column), summary/all_types_image_metrics.csv
 - previews/<TYPE>/source/*_preview.jpg (optional)

Usage
1) Optional: generate previews for calibration
   - python -m scripts.stainID.make_previews --input_dir origianl_data/out_seg/matched_output --out_root results_stainID --max_per_type 6 --longest_side 2048

2) Calibrate per type (Streamlit)
   - streamlit run scripts/stainID/calibrate_stain_streamlit.py
   - Pick a type (numeric prefix), select channels to output, tweak per-channel threshold/morphology, optional gating (e.g., gate DAB by Hematoxylin), and Save config.
   - Notes:
     - Use "Absolute threshold" to directly set threshold [0..1] if autos do not react as expected.
     - When viewing downsampled previews, the UI scales morphology (radii, min-size) to preview scale; saved configs remain full-resolution values.
     - Optionally enable Macenko stain estimation and watershed splitting.
     - Default behavior: absolute threshold on raw OD (log slider). The histogram uses a log x-axis.
     - Preview options: you can either use generated JPEG previews, downsample the original, or toggle "Preview original-resolution crop" to process a small square crop at full resolution (faster than full-image, no downsampling artifacts). Crop position and size are adjustable.

3) Batch segment and extract metrics (run as module)
   - python -m scripts.stainID.segment_stain --input_dir origianl_data/out_seg/matched_output --out_root results_stainID --types 210 212 213 
   - Optional pixel size (µm): `--pixel_width_um 0.2738 --pixel_height_um 0.2738` (defaults already set)

Notes
- Types are inferred as the numeric prefix before the first underscore in the filename.
- Pixel sizes are in micrometers and used to report area in µm².
- HED orientation is auto-detected by default; override in the config if needed.
- Optional settings:
  - Per-channel settings (Hematoxylin, Eosin, DAB): thresholds (auto or absolute; raw or normalized), orientation, morphology.
  - Optional gating rules (off by default), e.g., restrict DAB to regions touching Hematoxylin within N pixels.
  - hed_stain_estimation (DAB): "default" (skimage rgb2hed) or "macenko".
  - watershed_enabled (+ min_distance, compactness)
  - Normalization per-channel: when off, thresholds operate on raw OD values (0..~2.5). When on, thresholds operate on normalized [0..1].
  - Background pre-mask: ignore pure black borders and white background (tweak HSV V/S thresholds). This prevents background from biasing normalization/thresholding.
  - morph_remove_large_connected: remove very large connected components (default true) using a span-based threshold: remove components whose max(width,height)/max(H,W) exceeds the configured fraction (range ~0..0.5; default 0.08). Runs before hole-filling.
