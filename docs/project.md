# Project files

A project is a folder with a settings file, `stainid.yaml`, and everything the study reads and makes. This page
describes the [settings](#settings), the [input tables](#input-tables) and the [outputs](#outputs).

## Settings

The web app creates `stainid.yaml` and edits it on the Settings page; `stainid init` writes the defaults below. The
file only keeps the keys that differ from the defaults, and every relative path in it is resolved against the project
folder. On the command line the project is found from `--project`, then `$STAINID_PROJECT`, then the current folder;
`stainid info` prints every resolved path and whether it exists, and `stainid status` shows how far each step has got.

```yaml
name: My TMA study
kind: tma                    # tma, images (an image set) or slides (whole slides): see project_kinds.md
pixel_size_um: 0.2738        # fallback when a table has no pixel size; the analysis scale of image sets and whole slides
field_context_px: 128        # margin read around each field
stains: {NeuN: neun, 6E10: amyloid, AT8: tau}
tma:
  prefix: TMA-               # core names are <prefix><tma>_<column><row>, e.g. TMA-3_B-2
  rows: 5                    # core grid on each slide
  columns: 6
donor_ids: {digits: 4, prefixed_digits: 3}   # how the guided setup writes donor ids: 698 -> 0698, JOG3 -> JOG_003
sample_region_ids: {plain_prefix: "", prefixed_prefix: "", region: name}   # donor-region names while no saved map shows one (see tma_layout)
groups:                      # diagnostic group codes used in the TMA map, in plot order
  - {code: CT, label: Control, color: 3}     # color = palette slot 1-8 (optional)
  - {code: AD, label: Alzheimer disease}
cache_limit_gb: null         # space the pictures made for viewing (data/cache) may take; null: the server's --cache-limit, else 100
precision:                   # arithmetic on CUDA GPUs (CPU always fp32): fp32, tf32 or bf16 — keep one per cohort
  cellpose: bf16             # tf32 reproduces fp32 masks cell for cell; bf16 is faster, ~1% of cells differ
  sam: tf32

imaging: {native_pixel_size_um: null, rescale: on_import, stains: {}}   # image sets and whole slides (project_kinds.md)
dataset: {}                  # the dataset package a project was made from (project_kinds.md)

inputs:
  image_list: data/images.csv                                    # image sets and whole slides: every image, its donor and stain
  slides_dir: data/slides                                        # folder with the slide scans
  slides_table: data/slides.csv                                  # which scan is which TMA and stain
  core_manifest: data/core_manifest.csv                          # stainid dearray (+ layout, export, qc)
  core_images: data/analysis/core_images.csv                     # stainid tiles
  whole_core_manifest: data/analysis/cohort_whole_core_tiles.csv # stainid tiles (every tile of every core)
  calibration: data/analysis/slide_dab_calibration.csv           # stainid calibrate
  manual_exclusions: data/annotations/cohort_manual_exclusions.json
  donor_metadata: data/donor_metadata.csv
  tma_layout: data/tma_layout.csv

models:
  neun: data/validation/neun_final_predictions/model.joblib
  amyloid: data/models/amyloid_frozen/bundle.joblib
  tau: data/analysis/tau_neuron_benchmark/bundle.joblib
  cellpose: data/models/cellpose/cpsam_v2
  sam: data/models/sam/sam_vit_b_01ec64.pth
  plaque_cnn_dir: data/models/plaque_cnn
  huggingface_home: data/models/foundation/hf

outputs:
  qc: data/qc
  neun: data/analysis/cohort_neun
  fields: data/analysis/cohort_v3
  nuclei: data/analysis/tile_nuclei
  masks: data/analysis/cohort_object_masks
  tables: data/analysis
  reviews: data/annotations
  trained_models: data/models/trained
  jobs: data/jobs
```

`huggingface_home` and `plaque_cnn_dir` are exported as `HF_HOME` (offline) and
`STAINID_PLAQUE_CNN_DIR`, and the Cellpose folder as `CELLPOSE_LOCAL_MODELS_PATH`, so no model
is downloaded at run time.

## Input tables

You provide the TMA map, and optionally donor information and hand-drawn exclusions. The other tables are made by the
steps.

| Table | Made by |
|---|---|
| `tma_layout.csv` | the guided setup's Map screen (from your Excel, Word or CSV maps), or you (template on the *Attach TMA map* card) |
| `donor_metadata.csv` | optional: the guided setup's Donors screen (from your case sheet), or you; only used by your own downstream statistics |
| `manual_exclusions` | you, optional |
| `slides.csv` | *Register slides* |
| core table (`core_manifest`) | *Find cores*, then updated by *Attach TMA map*, *Export cores* and *Check core quality* |
| tiles (`whole_core_manifest`), `core_images` | *Cut cores into tiles* |
| `calibration` | *Calibrate stain thresholds* |

**`slides.csv`**: one row per scan, written by the *Register slides* step: `slide_path`, `tma`, `stain`. Optional:

| column | |
|---|---|
| `rows`, `columns` | the slide's own core grid (e.g. a 4 x 6 array in a 5 x 6 study). Empty: *Find cores* tries the project's size and the sizes around the number of core rows and columns it counts (2-12 each), keeps the best fit (the project's size unless another fits clearly better) and writes it here |
| `grid_source` | `auto` (found by *Find cores*) or `manual` (set by you, e.g. on the Cores screen); a size that is set is always used as it is |
| `orientation` | empty (as scanned), `rotate_180`, `flip_left_right` or `flip_top_bottom` when the slide lies turned or mirrored, so that A-1 is the map's top-left position |

Supported scan formats are those Bio-Formats reads: `.vsi`, `.svs`, `.ndpi`, `.scn`, `.mrxs`, `.czi`, `.tif`.

**`tma_layout.csv`**: one row per core position of the TMA map. The web app offers a template with every position
found by *Find cores* already listed.

| column | |
|---|---|
| `tma`, `core_label` | array and position (e.g. `B-2`) |
| `donor_id`, `region`, `disease_group` | sample identity |
| `tissue_control` | non-empty for control / orientation cores (e.g. `CBLM`) |
| `map_source`, `map_pathology_raw` | optional; the file the map was read from and the pathology as written |
| `cerad`, `braak` | optional neuropathology |
| `sample_region_id` | optional; defaults to `<donor_id>_<region>`. The guided setup writes it in the project's own naming, learned from the rows already saved (e.g. `BRC_1673_FR`, `BRC_1673_OCC`, `JOG_003_FR`: `BRC_` before plain numeric ids, nothing before prefixed ones, frontal → FR, occipital → OCC); with no saved rows it uses `sample_region_ids` (`plain_prefix`, `prefixed_prefix`, `region`: `name` or `code`), editable under *Map → Advanced* |

**`donor_metadata.csv`**: one row per donor.

- Required: `donor_id`, `disease_group`.
- Optional: covariates (age, sex, APOE, PMI, …), used by downstream analyses only.

**Tiles** (`whole_core_manifest`): one row per tile of tissue in each core image.

| column | |
|---|---|
| `tile_id`, `core_id`, `tma`, `stain` | tile identity (`…_W<row><column>`; squares that were sampled fields in older projects keep `…_S<n>`) |
| `donor_id`, `sample_region_id`, `region`, `disease_group`, `technical_replicate` | sample identity |
| `image_path` | exported core image (relative to the project) |
| `x_px`, `y_px`, `width_px`, `height_px` | tile position in the core image |
| `pixel_width_um`, `pixel_height_um` | pixel size |
| `selection_order` | the order detection works through a core (100 + 100 × row + column) |
| `preview_tissue_fraction` | share of the tile that is tissue, on the core preview |

**`manual_exclusions`**: JSON mapping each core image path to a list of polygons (`polygon_core_px`,
core-image pixel coordinates) to drop before measurement.

## Outputs

| Folder | Content |
|---|---|
| `outputs.nuclei` | `<tile_id>.npz`: Cellpose-SAM label image (field + context), every stain; NeuN, AT8 and 6E10 detection read it |
| `outputs.neun` | `tiles/<tile_id>_{features,objects}.csv`; merged `tile_features.csv`, `objects.csv` |
| `outputs.fields` | `parts/<core>_<stain>_{features,objects}.csv`: 6E10 plaques, AT8 tau neurons and field summaries |
| `outputs.masks` | per-core SAM outlines and shape features |
| `outputs.tables` | `results_<level>.csv` (every stain in one table) and `neun_`, `fields_`, `masks_<level>.csv` (level = `donor_region` or `core`) |
| `outputs.qc` | `grids/<slide>.jpg`: the fitted core grid on every slide; `grids/<slide>.json`: the grid size, how it was chosen, how well it fits (and whether someone marked it as right); `grids/previews/<slide>.jpg`: the plain slide picture behind the core thumbnails |
| `outputs.trained_models` | `<stain>_<time>/bundle.joblib` and `report.json` for every model trained in the app |
| `outputs.reviews` | `reviews/<name>/`: review sets (see [webapp.md](webapp.md)) |
| `outputs.jobs` | job records and logs from the web app |

Object coordinates (`centroid_x_px`, `centroid_y_px`) are in context-crop pixels, whose origin is
`max(0, x_px − field_context_px)`.
