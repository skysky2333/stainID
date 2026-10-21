# Methods

What each step computes, in workflow order, with its parameters. How to run the steps is in
[webapp.md](webapp.md); the files they read and write are in [project.md](project.md). The staged (v2) detection,
which keeps every candidate with its call for the viewer and the annotation rounds and redoes only what is out of
date, is described in [pipeline_v2.md](pipeline_v2.md).

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

## 2. Tiles — `stainid tiles`

Writes the core image table (`inputs.core_images`, one row per core and stain) and cuts every
core image into a grid of 2048 px squares (about 560 µm), keeping every square that holds tissue
(at least 0.5 %, about 1,600 µm²; edge slivers too: only their tissue counts in any area), including the narrower squares at the right and bottom image edges, so the whole core
is covered without overlap (`inputs.whole_core_manifest`). Tiles are named `…_W<row><column>`.
Every analysis runs on all tiles of every core; `stainid summarize` pools them per core and
donor-region. The study aggregation (`stainid.analysis.aggregation`) keeps a `sampling_extent`
column: `whole_core`, and `whole_core_interior` (tiles at least half tissue; the rim sensitivity
analysis, since rim tiles carry about twice the AT8-positive fraction of interior tiles) and, when
tiles are classed as grey or white matter, `whole_core_grey`.

Projects analysed before whole-core analysis also have sampled-field tables
(`inputs.field_manifest`, `inputs.tile_manifest`). Nothing makes them any more; where they exist,
the squares that were sampled fields keep their names (`…_S01`) in the tile grid and the tables are
read only so that earlier labels, labelling sessions and training sets still find their fields
(`stainid.sampling.tiles`). An older `analysis.tiles: sampled_fields` setting is ignored.

**Labelling fields.** A field shown to people for labelling is one full-size tile, whatever its share
of tissue (tiles at the rim are labelled too); the cut-off tiles at a core's right and bottom edge stay
out, so every field shows the same area. New sessions draw fields only from cores that are ready (core picture exported, slide
calibrated, detection run on the core, the stain's model chosen; see `stainid.labels.readiness`);
the Labels page shows any other core with what is missing and the step that makes it.

## 3. Slide calibration — `stainid calibrate`

One DAB threshold per slide (TMA × stain), computed from tissue pixels pooled across every tile of
every core of that slide (every 64th tissue pixel of each tile's own area) and blind to diagnosis. The
excluded areas are left out — linear artifacts and folds (NeuN: linear artifacts only), the same masks
stage 2 finds on the same crops (tile plus its context margin), none of which depends on the threshold;
folds are dark and would raise it. The AT8 edge guard is not left out: it is decided from the pixels
above the threshold. `excluded_tissue_fraction` records the share left out (`--workers` reads several
core images at once):

```
threshold = max(Otsu(pooled DAB OD), median + 6 × MAD, 0.01 OD)
```

Every downstream rule is expressed relative to this threshold, so staining-intensity differences
between slides do not change what counts as positive. Slides that already have a threshold keep it
(`--redo` works it out again). The cohort's thresholds were calibrated again on 17 Oct 2026 on today's
tiles without the excluded areas (`data/analysis/v2_audit/recalibration.csv`, the table in
`slide_dab_calibration.excluded_out.csv`), installed for the clean re-run of 18 Oct 2026 (the 14 Oct table kept
as `slide_dab_calibration.before_v2b.csv`). Leaving the excluded areas out moves no threshold by more than 3.5 %, but
15 of the 39 earlier rows were stale (made from an earlier, smaller sample), among them LIP-6 6E10 (0.026 against
0.064 OD recalibrated) and LIP-1 AT8 (0.014 against 0.039). At the stale LIP-6 level the 6E10 split seeds were low
(h = the seed level), so single plaques were cut into 2–3 pieces and pale specks under 100 µm² were proposed by the
thousand (all rejected); at 0.064 the pale deposits keep their candidates (34 of 35 in the lower half of LIP-6 A-2, 50
of 55 in B-5, as before) and the plaques are whole, as on the other slides.

## 4. Exclusions

Applied before any measurement (`stainid.imaging.tissue`, `stainid.qc.exclusions`):

- non-tissue (brightness / saturation). For tissue **area** (the denominator of every density and
  area fraction) the bare glass is measured once per core image, in the slide margin around the
  core: some scanners render it light grey (about 230/255), darker than the fixed cut-off, so tissue
  is anything clearly darker than that glass, and enclosed holes up to ~10,000 µm² (vacuoles, small
  vessel lumens) count as tissue. Measuring the glass per field instead mistakes pale tissue (white
  matter, neuropil, vacuoles) for glass in fields that show none and undercounts their tissue;
- linear folds and cut lines;
- hematoxylin-dense fold bands (≥ 150 µm long, aspect ≥ 4);
- for AT8, chromogen rims along vacuoles and tissue edges;
- hematoxylin-negative Cellpose objects (vacuoles), which are dropped as nuclei.

## 5. Nuclei — `stainid nuclei`

Cellpose-SAM on the RGB field (diameter 30 px, flow 0.4, cell-probability 0,
minimum 50 px), saved as one label image per tile in `outputs.nuclei`, for every stain: NeuN
detection uses these cells as candidates, AT8 needs them, and 6E10 uses them for peri-plaque
nuclear features. This is the slow step (about 90 s per tile on an Apple GPU, about 4 s on an
NVIDIA A100-class GPU) and the only one that benefits from a GPU:

```bash
stainid nuclei --stain NeuN --stain AT8 --stain 6E10 --device cuda
```

With CUDA the arithmetic is the project setting `precision.cellpose` (CPU always runs fp32). Measured on
an A100-class card: `bf16` (the default, as in Cellpose-SAM itself) about 2.9 s per tile with about 1% of
cells different from fp32; `tf32` about 3.8 s with the fp32 masks cell for cell; `fp32` about 10 s. A
cohort should use one setting throughout (this one used `tf32`, matching its earlier fp32 masks). Use one
process per GPU with `--shard-index i --shard-count N`. The detection steps reuse the
saved outlines and run on the CPU. Started with `--follow`, `stainid neun` and `stainid fields` run at the
same time as this step and take each core image as soon as all of its tiles have outlines, so the CPU
works alongside the GPU instead of after it.

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
2. Identity: a random forest on morphology and 140 µm context decides acceptance at P ≥ 0.5. Adding a Phikon
   foundation-model head or the published Wong et al. (2022) consensus CNN lowered leave-one-TMA-out AUC
   (0.94 vs 0.96; `data/analysis/model_validation/components.csv`), so the deployed model uses neither. (The frozen
   v1 pipeline that defines the primary amyloid endpoints is separate and does use the plaque CNNs.)
3. Deposits whose outer boundary touches a lumen or tissue edge for ≥ 25 % of its length are
   classified `vascular_or_edge` and excluded from parenchymal measures.
4. Morphotype: accepted plaques ≥ 15 µm in diameter are **compact** when their inner DAB is at
   least `morphotype_threshold` × the slide threshold (default ≈ 1.59). Otherwise they are
   **diffuse**. Plaques of 10–15 µm are `small_plaque`. Accepted deposits of 10 µm or less (78.5 µm²; reviewers: under 80 µm²) are `speck` and are not
   counted: in a blind review of 389 disputed objects none of 8 µm or less, and 4 % of 8–10 µm, were plaques.
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

With CUDA the arithmetic is the project setting `precision.sam`. On a NeuN test core
`tf32` (the default) was 1.5 times faster than fp32 with the same objects outlined, 99.9995% of label
pixels equal and a median area change of 0% (99th percentile 0.55%); `bf16` was no faster (1.4 times)
and less close (median 0.14%, 99th percentile 1.5%). The sampled-field outlines were made in fp32
(Apple GPU / CPU); the whole-core outlines of this cohort use `tf32`.

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
