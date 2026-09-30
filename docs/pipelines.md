# Methods

What each step computes, in workflow order, with its parameters. How to run the steps is in
[webapp.md](webapp.md); the files they read and write are in [project.md](project.md).

All measurements are made on native-resolution fields (default 2048 px ≈ 560 µm at
0.274 µm/px). Each field is read with a 128 px context margin so that objects at the
edge are segmented whole; an object is counted only when its centroid falls inside the field.
Colour deconvolution uses the Ruifrok–Johnston H-E-DAB matrix (`stainid.imaging.color`).

## 1. Slides to cores

| Step | Command | Notes |
|---|---|---|
| Register slides | *web app* (writes `slides.csv`) | TMA number and stain are guessed from each file name and confirmed by the user. |
| Find cores | `stainid dearray` | Fits an affine rows × columns lattice (`tma.rows` × `tma.columns`) to tissue components on a slide preview, then re-centres each core on local tissue so torn, partial and edge-clipped cores are recovered. Writes the core table and one grid picture per slide. |
| Attach TMA map | `stainid layout` | Joins donor, region, diagnostic group (and optional neuropathology) from `tma_layout.csv`. Replicate cores of a donor-region are numbered in file order. |
| Export cores | `stainid export` | Native-resolution core PNGs read directly from the scanner files. |
| Core QC | `stainid qc` | Tissue fraction and fragments, background brightness, hematoxylin / DAB OD quantiles, low-focus scanner tiles. |

QuPath alternatives for grid placement and export are in `tools/qupath/`.

## 2. Field selection — `stainid select`

For every stain-core, candidate windows are scored on a preview for tissue and focus only
(never stain intensity). The requested number of fields (default 8) is chosen to be spatially
spread across the core. The first `--primary` fields (default 4) form the analysis manifest
(`inputs.tile_manifest`); the full set (`inputs.field_manifest`) is kept for sampling-stability
checks.

## 3. Slide calibration — `stainid calibrate`

One DAB threshold per slide (TMA × stain), computed from tissue pixels pooled across all cores
of that slide and blind to diagnosis:

```
threshold = max(Otsu(pooled DAB OD), median + 6 × MAD, 0.01 OD)
```

Every downstream rule is expressed relative to this threshold, so staining-intensity differences
between slides do not change what counts as positive.

## 4. Exclusions

Applied before any measurement (`stainid.imaging.tissue`, `stainid.qc.exclusions`):

- non-tissue (brightness / saturation);
- linear folds and cut lines;
- hematoxylin-dense fold bands (≥ 150 µm long, aspect ≥ 4);
- manually drawn artifact polygons (`inputs.manual_exclusions`);
- for AT8, chromogen rims along vacuoles and tissue edges;
- hematoxylin-negative Cellpose objects (vacuoles), which are dropped as nuclei.

## 5. Nuclei — `stainid nuclei`

Cellpose-SAM on the RGB field (diameter 30 px, flow 0.4, cell-probability 0,
minimum 50 px), saved as one label image per field. Required before AT8, optional for 6E10
(peri-plaque nuclear features).

## 6. NeuN — `stainid neun`

1. Candidates from two sources: DAB-contour profiles with watershed splitting, and Cellpose-SAM
   cells (which also recover pale neurons).
2. Each candidate gets shape, stain and 55 µm local-context features.
3. A random forest (`models.neun`) accepts NeuN-positive neuronal profiles at P ≥ 0.5.

Outputs are per-field tile features and objects; `stainid neun --merge` joins them into
`tile_features.csv` and `objects.csv`. The NeuN-positive fraction of all nuclear candidates is
reported next to density as a cellularity-independent measure.

## 7. 6E10 plaques — `stainid fields --stain 6E10`

1. Candidates are segmented from slide-calibrated DAB. Touching plaques are split by a seeded
   watershed on 3 µm-smoothed DAB, with h-maxima prominence equal to the slide threshold.
2. Identity: the mean of three probabilities decides acceptance at P ≥ 0.5:
   - a random forest on morphology and 140 µm context;
   - a logistic head on Phikon foundation-model embeddings;
   - the published Wong et al. (2022) consensus CNN.
3. Deposits whose outer boundary touches a lumen or tissue edge for ≥ 25 % of its length are
   classified `vascular_or_edge` and excluded from parenchymal measures.
4. Morphotype: accepted plaques ≥ 15 µm in diameter are **compact** when their inner DAB is at
   least `morphotype_threshold` × the slide threshold (default ≈ 1.59). Otherwise they are
   **diffuse**. Plaques smaller than 15 µm are `small_plaque`.
5. Per plaque:
   - area, dense-core area and fraction, circularity, solidity and boundary irregularity;
   - nuclei inside the plaque and in a 15 µm ring around it, against the nucleus density
     measured far from plaques.

## 8. AT8 tau — `stainid fields --stain AT8`

1. Burden (`at8_positive_area_fraction`) is the slide-calibrated DAB-positive area fraction after exclusions.
2. Tau+ neuron candidates come from two sources:
   - Cellpose nuclei with a stained perinuclear ring (`ring`);
   - dense AT8 bodies that stand above a 60 µm local background (`dense body`).

   A random forest (`models.tau`) scores each candidate on shape, stain and context.
   Detections within 12 µm are merged.
3. Neuropil threads are extracted as follows:
   - Multiscale Sato ridge filter (σ = 1–3 px).
   - Keep ridges above 0.25 × threshold with smoothed DAB above 0.6 × threshold, at least
     60 % above the 8 µm local background.
   - Remove ridges within 4 µm of tissue rims and on nuclei; minimum length 4 µm.
   - Skeletonise and report length density, width, branch points and end points.

## 9. Object outlines — `stainid masks`

Segment Anything (ViT-B) is prompted at each accepted object's centroid on a native 1024 px
window. One run covers every stain (`--stain` limits it); cores without detections yet are skipped.

- NeuN and 6E10 use a fixed output.
- AT8 picks among SAM's outputs with a stain-contrast rule.

For each outline the step records area, perimeter, axes, elongation, solidity, circularity and
stain density. For plaques it also records dense-core area and core count.

## 10. Results tables — `stainid summarize`

Counts and areas are summed over fields and technical cores before densities are formed
(numerator and denominator pooled). Per-object shape features are summarised by medians with
minimum object counts.

- `--level sample_region_id` gives donor × region tables.
- `--level core_id` gives per-core tables.

`summarize` writes `results_<level>.csv`: NeuN, 6E10 / AT8 and outline features in one table. It also writes the
per-kind tables `neun_`, `fields_` and `masks_<level>.csv` (`stainid aggregate neun|fields|masks` makes one of them).
The meaning of every column is shown on the Results page and defined in `stainid.analysis.dictionary`.

## Models

| Key | Used by | Source |
|---|---|---|
| `neun` | NeuN identity | random forest trained on the project's reference labels (`stainid.stains.neun.classifier`) |
| `amyloid` | plaque identity + morphotype | bundle from `stainid.stains.amyloid.model.train_amyloid_bundle` |
| `tau` | tau+ neuron identity | random forest trained on reference labels (`stainid.stains.tau.classifier`) |
| `cellpose` | nuclei, NeuN candidates | pretrained Cellpose-SAM weights (tested with `cpsam_v2`) |
| `sam` | outlines | Segment Anything ViT-B checkpoint `sam_vit_b_01ec64.pth` |
| `plaque_cnn_dir` | Wong 2022 consensus / Plaquebox | published weights and normalisation files |
| `huggingface_home` | Phikon (`owkin/phikon`) | Hugging Face cache; run offline once populated |

Random-forest bundles depend on the stain protocol and scanner, so no weights ship with the
repository. Train them in the web app (Models page) or with `stainid training-set` and `stainid train`. Training reuses
the detection code to find candidates and compute their features, so a trained bundle drops into the pipeline unchanged
(see [webapp.md](webapp.md#training-a-model)). `stainid download-models` fetches the Cellpose-SAM, Segment Anything
and Phikon weights.
