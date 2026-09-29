# Projects

A project is a folder containing `stainid.yaml`. Every relative path in the file is resolved
against that folder. `stainid init` writes the defaults below, and a project file only needs the
keys it changes. `stainid info` prints every resolved path and whether it exists.

The project is found from `--project`, then `$STAINID_PROJECT`, then the current directory.
Commands run with the project folder as their working directory.

```yaml
name: My TMA study
pixel_size_um: 0.2738        # fallback when a table has no pixel size
field_context_px: 128        # margin read around each field
stains: {NeuN: neun, 6E10: amyloid, AT8: tau}

inputs:
  core_manifest: data/core_manifest.csv                          # stainid-dearray (+ attach-layout, core-qc)
  core_images: data/analysis/core_images.csv                     # stainid select
  field_manifest: data/analysis/cohort_systematic_tiles.csv      # stainid select (all fields)
  tile_manifest: data/analysis/cohort_primary_tiles.csv          # stainid select (analysis fields)
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
  neun: data/analysis/cohort_neun
  fields: data/analysis/cohort_v3
  nuclei: data/analysis/tile_nuclei
  masks: data/analysis/cohort_object_masks
  tables: data/analysis
  reviews: data/annotations
  jobs: data/jobs
```

`huggingface_home` and `plaque_cnn_dir` are exported as `HF_HOME` (offline) and
`STAINID_PLAQUE_CNN_DIR`, and the Cellpose folder as `CELLPOSE_LOCAL_MODELS_PATH`, so no model
is downloaded at run time.

## Input tables

**`tma_layout.csv`**: one row per core position of the TMA map.

| column | |
|---|---|
| `tma`, `core_label` | array and position (e.g. `B-2`) |
| `donor_id`, `region`, `disease_group` | sample identity |
| `tissue_control` | non-empty for control / orientation cores |
| `cerad`, `braak` | optional neuropathology |

**`donor_metadata.csv`**: one row per donor.

- Required: `donor_id`, `disease_group`.
- Optional: covariates (age, sex, APOE, PMI, …), used by downstream analyses only.

**Analysis fields** (`tile_manifest`): one row per field.

| column | |
|---|---|
| `tile_id`, `core_id`, `tma`, `stain` | field identity |
| `donor_id`, `sample_region_id`, `region`, `disease_group`, `technical_replicate` | sample identity |
| `image_path` | exported core image (relative to the project) |
| `x_px`, `y_px`, `width_px`, `height_px` | field position in the core image |
| `pixel_width_um`, `pixel_height_um` | pixel size |
| `selection_order`, `nested_sample` | selection rank; `primary_four` / `extended_eight` |

**`manual_exclusions`**: JSON mapping each core image path to a list of polygons (`polygon_core_px`,
core-image pixel coordinates) to drop before measurement.

## Outputs

| Folder | Content |
|---|---|
| `outputs.nuclei` | `<tile_id>.npz`: Cellpose label image (field + context) |
| `outputs.neun` | `tiles/<tile_id>_{features,objects}.csv`; merged `tile_features.csv`, `objects.csv` |
| `outputs.fields` | `parts/<core>_<stain>_{features,objects}.csv`: 6E10 plaques, AT8 tau neurons and field summaries |
| `outputs.masks` | per-core SAM outlines and shape features |
| `outputs.tables` | `fields_<level>.csv`, `masks_<level>.csv` (level = `sample_region_id` or `core_id`) |
| `outputs.reviews` | `reviews/<name>/`: review sets (see [webapp.md](webapp.md)) |
| `outputs.jobs` | job records and logs from the web app |

Object coordinates (`centroid_x_px`, `centroid_y_px`) are in context-crop pixels, whose origin is
`max(0, x_px − field_context_px)`.
