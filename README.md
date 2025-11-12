StainID: stain segmentation and analysis

Quickly calibrate thresholds, batch‑segment stains (Hematoxylin, Eosin, DAB), and analyze features for TMA cores.

![StainID UI](docs/screenshot.png)

Install
- Create/activate an environment.
- Install packages: `pip install numpy scipy pandas scikit-image opencv-python-headless pyyaml streamlit matplotlib pillow`

Input Prep (QuPath)

How To Use
- 1) Prepare input
  - Install QuPath.
  - Open each TMA slide and run TMA → Dearray. Increase core diameter by 1 mm until it succeeds.
  - Go to Automate → Script editor. Drag `scripts/export_core.groovy` into the editor and run it.
  - Repeat step 2-3 for each slide.
  - Exports will default to `TMA_core_exports/<SlideName>/.../*.png` inside your QuPath project folder.
  - Ensure exported core images are available under `data/TMA_core_exports` or set `--input_dir` accordingly.
- 2) (Optional) Make previews for faster UI
  - `python -m scripts.make_previews --input_dir data/TMA_core_exports --out_root results_stainID --max_per_type 6 --longest_side 2048`
- 3) Calibrate per type (UI)
  - `streamlit run scripts/calibrate_stain_streamlit.py`
  - Pick a type (numeric prefix), select channels (DAB/Hematoxylin/Eosin), adjust thresholds and morphology, optionally enable gating. Once satisfied, click “Save config”.
  - For full tuning guidelines, reference `docs/features_and_options.md`.
  - Repeate and make sure to save config for each stain types.
- 4) Batch segment
  - `python -m scripts.segment_stain --input_dir data/TMA_core_exports --out_root results_stainID --types 212 213 214`
  - Optional pixel size (µm): `--pixel_width_um 0.2738 --pixel_height_um 0.2738` (defaults. This value can be acquired from QuPath).
- 5) Analyze results (optional)
  - `python -m scripts.analyze_results --out_root results_stainID --types 212 213 214 --metadata data/brain_summary.csv`
  - Produces per‑phenotype tables and top plots under `results_stainID/analysis/`.

Shortcuts
- You can follow the example pipeline in `scripts/driver.sh`.

Folder Layout
- Input images default: `data/TMA_core_exports`
- Output root: `results_stainID`
  - `configs/<TYPE>.yaml`
  - `masks/<TYPE>/<channel>/<image>_mask.png`
  - `overlays/<TYPE>/<channel>/<image>_overlay.jpg` and `overlays/<TYPE>/<image>_overlay_all.jpg`
  - `features/<TYPE>/<channel>/<image>_sites.csv`
  - `summary/<TYPE>_image_metrics.csv`, `summary/all_types_image_metrics.csv`
  - `previews/<TYPE>/source/*_preview.jpg` (optional)

Notes
- “Type” is parsed as the first number after an underscore in the filename. Example: `Image_212.vsi - 20x_BF_01_col03_row07.png` → type `212`.
- Multi‑channel segmentation is supported; outputs are saved per channel and combined preview overlays are also created.
- Gating lets you restrict one channel’s detections by another (e.g., DAB touching Hematoxylin).
- See docs for complete options and tuning: `docs/features_and_options.md`.
