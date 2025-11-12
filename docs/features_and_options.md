Features and Options

This document lists all pipeline features, configuration options, and tuning guidance. Configuration files are saved under `results_stainID/configs/<TYPE>.yaml` and are created/edited via the Streamlit UI (`scripts/calibrate_stain_streamlit.py`).

Pipeline Overview
- Multi‑channel HED segmentation: Hematoxylin (H), Eosin (E), DAB (D).
- Per‑channel thresholding with raw or normalized intensity.
- Per‑channel morphology cleanup and optional watershed splitting.
- Optional gating: restrict detections in one channel by another.
- Batch outputs: binary masks, overlay images, per‑site CSV features, image‑level metrics, and spatial statistics.
- Optional differential analysis over metadata using `scripts/analyze_results.py`.

CLI Entry Points
- Previews: `python -m scripts.make_previews --input_dir <dir> --out_root <root> [--max_per_type N --longest_side PX]`
- Calibration UI: `streamlit run scripts/calibrate_stain_streamlit.py`
- Batch segmentation: `python -m scripts.segment_stain --input_dir <dir> --out_root <root> [--types ...] [--pixel_width_um V --pixel_height_um V] [--no_preview] [--downsample_preview K]`
- Analysis: `python -m scripts.analyze_results --out_root <root> --metadata <csv> [--types ...] [--alpha Q] [--min_per_group N] [--agg mean sd iqr median] [--plots_dir <dir>]`

Streamlit UI Walkthrough
- Sidebar (left)
  - `Input directory`
    - Folder containing exported TMA core images. Default: `data/TMA_core_exports`.
    - Accepts nested subfolders. All `.png/.jpg/.jpeg/.tif/.tiff` files are scanned.
  - `Output root`
    - Destination for configs and results. Default: `results_stainID`.
    - UI saves `configs/<TYPE>.yaml` here; batch writes masks/overlays/features/summary here.
  - `Image type (numeric prefix)`
    - Select the numeric type to calibrate. The tool parses the first number after an underscore in filenames (e.g., `Image_212...` → `212`).
  - `Engine`
    - `hed` (default): HED color deconvolution; supports separate channels: DAB, Hematoxylin, Eosin.
    - `hsv`: HSV thresholding; useful as a simple fallback when HED is unsuitable.
  - HSV options (shown only if Engine = hsv)
    - `Hue low` / `Hue high` (0..1): hue window for the stain color. DAB‑like browns often lie ~0.05–0.15.
    - `Saturation min` (0..1): exclude low‑saturation background; raise to remove gray/white.
    - `Value min` / `Value max` (0..1): constrain brightness; lowering `Value max` removes very bright regions.
  - Channels to output (Engine = hed)
    - Multiselect `dab`, `hematoxylin`, `eosin`. Selected channels will be exported during batch segmentation (masks, overlays, features, metrics).
  - Per‑channel settings (repeat for each channel)
    - `Normalize channel [0..1]`
      - On: thresholds operate on a normalized intensity in [0,1] computed robustly (1–99th percentile), reducing local lighting effects.
      - Off: thresholds operate on a raw OD‑oriented channel (absolute scale; comparable across images when configs are fixed).
    - `Threshold mode`
      - Auto: choose method (`otsu`, `yen`, `li`). “Threshold scale” (0.2–2.0) multiplies the auto threshold; lower = more pixels selected.
      - Absolute (normalized): fixed cutoff on normalized intensity. Slider uses log10 of the value for fine control near zero.
      - Absolute (raw): fixed cutoff on raw OD‑oriented values. Slider is log10; e.g., −1.5 ≈ 10^−1.5 ≈ 0.0316.
    - `Orientation`
      - `auto` tries both orientations and picks the one with a plausible foreground fraction (avoids inverted masks).
      - `normal`/`invert` force orientation. Foreground is always “values > threshold”.
    - `Stain estimation (DAB)`
      - `default`: standard HED using skimage’s `rgb2hed`.
      - `macenko`: Macenko stain estimation for DAB; can improve robustness if DAB hue varies.
    - Morphology (applied after thresholding)
      - `Opening radius (px)`: removes small bright specks; larger values remove more noise but can erode fine structures.
      - `Closing radius (px)`: fills small gaps/bridges nearby regions.
      - `Remove small objects (px)`: area threshold (in pixels) to drop tiny components.
      - `Fill holes`: fills interiors; use with care if hollow structures are meaningful.
      - `Fill holes up to area (px, 0 = fill all)`: restricts hole filling to small holes; 0 fills all holes.
      - `Clear border`: removes components touching the image border.
      - `Remove very large connected area (auto)`: removes components whose span exceeds a fraction of the larger image dimension.
      - `Large span threshold (fraction of larger image dimension)`: typical 0.02–0.10; increases remove bigger blobs.
    - Watershed (split touching)
      - `Enable watershed split for this channel`: separates merged objects.
      - `Min distance (px)`: required separation between seeds (larger = fewer splits).
      - `Compactness`: increases roundness of regions but can over‑split if too high.
  - Gating (restrict DAB by Hematoxylin)
    - `Gate DAB by Hematoxylin (touching/intersect)`: enables gating without discarding stored rules.
    - `Mode`
      - `touching`: keep DAB pixels near Hematoxylin; uses a dilation radius to define “near”.
      - `intersect`: keep only DAB pixels overlapping Hematoxylin.
    - `Dilation (px) for touching`: neighborhood size for proximity gating (typical 10–20).
    - `Min overlap fraction per region`: for `intersect`, minimum fraction of a DAB region that must overlap Hematoxylin to be kept (0–1).
  - Background mask (pre‑mask before thresholding)
    - `Ignore pure black borders`: drop pixels with all RGB ≤ threshold to avoid scanner borders affecting normalization.
    - `Black threshold (per channel 0..255)`: typical 3–10.
    - `Ignore white background`: drop very bright low‑saturation pixels.
    - `White V threshold (HSV)` / `White S max (HSV)`: raise V or lower S to remove more white background.
  - `Save config`
    - Writes the current settings to `configs/<TYPE>.yaml` in the Output root. Batch segmentation reads these files.

- Main Preview (right)
  - `Sample image`: choose which image to preview for the selected type.
  - `Use JPEG preview if available`
    - Uses `previews/<TYPE>/source/*_preview.jpg` created by `scripts/make_previews.py` to speed up rendering.
  - `Downsample factor (fallback)`
    - If no preview exists, downsample the original by this factor. Larger values make the UI faster at the cost of detail.
  - `Original‑resolution crop preview`
    - `Crop size (px)`, `Crop center row/col (0..1)`: extracts a square ROI at native resolution for accurate inspection.
    - For preview only, morphology radii and min‑area are scaled to the image scale; saved configs keep full‑resolution values.
  - Display options
    - `Draw edges only`: overlay mask edges instead of filled regions. `Edge thickness (px)` controls edge width.
    - `Preview channel for detail`: choose which channel’s masks/histograms to display in detail panels.
  - Visual panels
    - Overlays: combined per‑channel overlays for the main view and the crop.
    - Masks: binary masks for the selected channel (main and crop).
    - Masked signal: selected channel’s intensity image (normalized) masked by its own binary mask (main and crop).
    - Histograms: log‑log histograms for the selected channel with the applied threshold marked.
      - Normalized mode: x‑axis is log10(normalized intensity). Raw mode: x‑axis is log10(OD‑oriented intensity).

Tuning Guidelines
- Choosing normalized vs raw thresholding
  - Use `normalize: True` when backgrounds vary across the image and you want a relative [0–1] threshold per image.
  - Use `normalize: False` and set `absolute_raw` when you want comparable thresholds across images (e.g., consistent DAB OD cutoffs). The Streamlit UI shows histograms with log axes for raw OD.
- Picking an auto method
  - Start with `otsu`. If foreground is very small or bimodal, try `yen` or `li`.
  - Adjust `scale` slightly (e.g., 0.9–1.1) before switching methods.
- Morphology cleanup
  - Increase `remove_small` to prune tiny specks. Use small `open_radius` to remove isolated noise.
  - Use `close_radius` to connect small gaps within objects.
  - Enable `clear_border` if edge artifacts appear.
  - If background floods occur, enable `remove_large_connected` and tune `large_area_frac` (e.g., 0.02–0.1). Larger values remove bigger blobs.
- Watershed splitting
  - Enable when merged objects should be split. Increase `min_distance` to separate nearby peaks; add `compactness` to regularize shapes.
- Gating
  - Common pattern: restrict `dab` by `hematoxylin` using `touching` with `dilate_px` ~10–20 to keep DAB near nuclei.
  - Use `intersect` when strict pixel overlap is required.

Outputs
- Masks: `results_stainID/masks/<TYPE>/<channel>/<image>_mask.png` (binary PNG).
- Overlays: per‑channel and combined previews under `results_stainID/overlays/...`.
- Per‑site features: `results_stainID/features/<TYPE>/<channel>/<image>_sites.csv` containing object‑level measurements.
- Image‑level summary: `results_stainID/summary/<TYPE>_image_metrics.csv` and combined `summary/all_types_image_metrics.csv`.

Per‑Site Feature Columns
- Identifiers and geometry
  - `label`, `centroid_row`, `centroid_col`
  - `area_px`, `area_um2` (uses `--pixel_width_um`, `--pixel_height_um`)
  - `perimeter_px`, `equivalent_diameter_px`, `major_axis_length_px`, `minor_axis_length_px`
  - `eccentricity`, `solidity`, `extent`, `circularity`, `feret_diameter_approx_px`
- Intensity (normalized, oriented so higher = stronger stain)
  - `intensity_mean`, `intensity_std`, `intensity_median`, `intensity_iqr`, `intensity_integrated`
- Intensity (raw OD‑oriented, absolute across images)
  - `intensity_raw_mean`, `intensity_raw_std`, `intensity_raw_median`, `intensity_raw_iqr`, `intensity_raw_integrated`

Image‑Level Metrics
- Aggregates of site features (counts, areas, intensity summaries) plus spatial stats:
  - Nearest neighbor: `nn_mean_px`, `nn_sd_px`, and if pixel sizes provided, `nn_mean_um`, `nn_sd_um`.
  - Clark–Evans: `clark_evans_R`, `clark_evans_z`.
  - Grid dispersion: uniformity indices across a regular grid.
- Each row includes: `image`, `type`, `channel`, and `threshold_used` for that channel.

Analysis (`scripts/analyze_results.py`)
- Inputs
  - Image summary: `summary/all_types_image_metrics.csv` (or per‑type files).
  - Site‑level aggregates: derived from `features/<TYPE>/*_sites.csv` (supports per‑channel subfolders).
  - Metadata CSV (e.g., `data/brain_summary.csv`) with `r`, `c` columns plus phenotype columns like `Region`, `Label`, `C`, `B`.
- Behavior
  - Excludes r1_c1 (KC control) automatically.
  - Matches images to metadata by parsing `r` and `c` from image names (e.g., `..._r2_c4...`).
  - Aggregates site‑level numeric columns per image (`mean`, `sd`, `iqr`, `median`).
  - Runs per‑feature tests by phenotype; applies Benjamini–Hochberg FDR (q‑values).
  - By default, excludes normalized intensity features (`intensity_*`) from testing; raw intensity features (`intensity_raw_*`) are tested.
- Options
  - `--alpha`: FDR threshold (default 0.05).
  - `--min_per_group`: minimum samples per group to run a test (default 2).
  - `--agg`: which aggregations to compute at site level (default: mean sd iqr median).
  - `--plots_dir`: custom output for plots (defaults under `<out_root>/analysis/plots`).

Performance & UI Tips
- Generate JPEG previews to speed up the UI on very large images: `scripts/make_previews.py` writes to `<out_root>/previews/...`.
- In the UI, when using a downsampled image, morphology radii/areas are scaled for preview only; saved configs keep full‑resolution values.
- Use the “Original‑resolution crop preview” in the UI to test parameters on a small ROI without downsampling artifacts.
