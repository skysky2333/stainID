# v2 detection: stages, the candidate store, and the measures

The v2 detection (`stainid.v2`) splits detection into stages that each read the previous stage's saved outputs, so a
change to a model, a threshold or a rule re-runs only the cheap stages after it. Everything is written under
`outputs.v2` (default `data/analysis/v2/`); the current detection outputs (`cohort_neun`, `cohort_v3`,
`cohort_object_masks`) and results tables are not touched.

| Stage | Command | Reads | Writes | Cost |
|---|---|---|---|---|
| 1 Nuclei | `stainid nuclei` (unchanged) | core images | `tile_nuclei/<tile>.npz` | GPU, run once |
| 2 Proposals | `stainid v2 proposals` | core images, nuclei, slide thresholds, exclusions | `proposals/`, `tiles/`, `threads/`, `unproposed/`, `masks/` | CPU, the expensive stage |
| 4 Calls | `stainid v2 calls` | `proposals/`, the stain models, `v2.rules` | `calls/`, `candidates/` | minutes for the cohort, no image read |
| 3 Outlines | `stainid v2 outlines` | `proposals/`, `calls/` (which to outline), core images, SAM | `outlines/`, `candidates/` | GPU, optional |
| 5 Measures | `stainid v2 measures` | `candidates/`, `tiles/`, `threads/` | `measures/` | minutes, no image read |

Stage 3 runs after stage 4 because by default it outlines the candidates stage 4 accepted or scored at least 0.2
(`v2.outlines.select`); candidates outlined before are kept when a later model selects more.

## Running

`tools/v2_run.sh all` runs every stage (stage 2 with `WORKERS` core images at once, stage 3 with one process per GPU)
and `tools/v2_run.sh decide` only stages 4 and 5 (after a new model, cut-off or rule). Both are resumable: a core image is
done once its manifest is written, and every run redoes the core images that are missing or out of date. `stainid v2
status` shows, per stage and stain, how many core images are done, out of date and in all; the Run page shows the same
for the four "Detect (v2)" steps. Memory: a stage-2 worker holds one decoded core image (about 0.8 GB) and one tile's
pictures (about 1 GB); `--workers 0` picks as many as the free memory and the processors allow.

## Manifests and out-of-date detection

Each stage writes `manifests/<stage>/<stain>/<core_id>.json` after its outputs: the stage's code version
(`stainid.v2.manifest.VERSIONS`), its parameters (`stainid.v2.params`, overridable under `v2:` in stainid.yaml) and a
fingerprint of every input — the core image file, the number of tiles with nuclei, the slide threshold, the hand-drawn
exclusions, the tiles, the model bundle (folder name and sha256) and its cut-off, and the digest of each upstream
stage's manifest. A core image is out of date when any of these differs from what it would be made from now. Changing
`v2.rules` makes only stage 4 (and the store) out of date; changing `v2.proposals` stage 2 and everything after it.

## Stage 2: proposals

Per tile (the tile with its 128 px context margin; a candidate belongs to the tile holding its centroid) and stain:

- **NeuN**: the legacy DAB profiles and the legacy Cellpose cells, with the legacy filters recorded instead of applied
  (`flag_legacy_kept`, `flag_legacy_drop` = shape / excluded / near_profile / pale_nucleus); the legacy cells come before
  the profiles the legacy shape filter dropped, so where the old pipeline counted a cell the same cell is the candidate;
  the DAB component tree at 0.5, 0.75, 1 and 1.5 × the slide threshold (`dab@<level>`, best-fitting node per object: the
  one with the strongest DAB edge; merges split at their children, so merged neurons above 650 µm² that the legacy
  filter dropped are separated by the higher levels; a merge no level separates is cut into cells by a marker watershed
  on its darkest node, `split@<level>`, unless it looks like a stained rim or band — within 5 µm of a tissue edge, a long
  band, or too thin for a cell body, `stainid.v2.tree.cell_pieces`); components standing 0.5 × threshold above the 60 µm
  local background (`localbg`). A Cellpose nucleus inside a stained profile becomes a source of that candidate, not a
  candidate of its own.
- **6E10**: every deposit of the legacy segmentation (touching plaques split at h-maxima), including those the legacy
  shape rule called artifacts and never showed the classifier (`flag_artifact_shape`); faint deposits of the tree at
  0.5 and 0.75 × the threshold (roots: a dense core inside a diffuse halo is one deposit).
- **AT8**: the legacy candidates (`ring`: stained ring around a Cellpose nucleus; `dense`: dense body above the 60 µm
  background); fainter rings down to 0.15 (`ring_faint`); tree bodies at 1, 1.5, 2 × the threshold and the local
  background pass (`dab@…`, `localbg`).

De-duplication (`stainid.v2.geometry.same_object`): a new geometry that overlaps a kept candidate with IoU ≥ 0.5, or
lies ≥ 80 % inside it, or holds it (up to 3 × its area; 8 × for NeuN, a soma around its nucleus), is the same object
and only adds its source tag (`sources`); one spanning two kept candidates is a merge and is dropped; one more than
30 % covered by kept candidates is dropped. The legacy geometries come first, so where the old proposals were fine the
same outline (and the same features) is kept.

Every candidate gets the generic measurements (plain column names: area, shape, DAB and hematoxylin statistics and
deciles, ring and edge contrast, thickness, exclusion / edge / fold fractions, Cellpose overlap) and the current
model's inputs computed exactly as the legacy pipeline computes them (`m_<feature>`; for candidates of the new sources
in the legacy style of their stain). 6E10 adds the lumen-contact fraction, the smoothed-DAB quantiles (dense core at
any threshold later) and nuclei inside / within 15 and 45 µm; AT8 the share on traced threads and whether the centre is
in valid tissue.

Per core image also: `tiles/<stain>/<core>.csv.gz` (per tile: tissue and analysis areas, exclusion fractions,
DAB-positive area at 0.5–2 × the threshold, nucleus count, AT8 burden / thread / edge-guard measures, the v1 AT8
network measures, the NeuN 100 µm density windows), `threads/<core>.csv.gz` (the AT8 skeleton as segments, each with
its pixel count, width sum and the candidate it lies in), `unproposed/<stain>/<core>.csv.gz` and a 1/8-scale mask
picture `masks/<stain>/<core>.png` (bits: 1 tissue, 2 linear artifact, 4 fold, 8 manual exclusion, 16 AT8 edge guard,
32 analysed tile).

## Stage 4: calls

The stain model scores every candidate from `m_` columns (`stainid.candidates.model_matrix`). Rules (`v2.rules`):

| Stain | Before the model | After the model |
|---|---|---|
| NeuN | `rule:excluded` (≥ 50 % excluded, or a Cellpose cell in a hand-drawn exclusion), `rule:near_profile`, `rule:pale_nucleus` (legacy Cellpose filters), `rule:shape` (legacy profile size / shape filter); for the new sources only: `rule:edge` (≥ 50 % within 5 µm of an edge or hole), `rule:fold` (≥ 50 % on a fold band with a weak edge, DAB gradient < 1 threshold per px: patches of the band; neurons on a band keep a sharp edge — the legacy NeuN detection never masked folds), `rule:irregular` (solidity < 0.6) | — |
| 6E10 | `rule:excluded`; `rule:artifact_shape` off by default (the legacy shape rule; the classifier now decides) | `rule:speck` (≤ 10 µm), `rule:vascular_or_edge` (outer boundary ≥ 25 % on a lumen or tissue edge); subtype compact / diffuse / small by the model's morphotype threshold |
| AT8 | `rule:excluded` (also a centre outside valid tissue), `rule:low_contrast` (< 0.3, the legacy bar), `rule:small` (new-source bodies < 40 µm²), `rule:edge`, `rule:irregular` (new-source bodies less compact than legacy bodies: solidity < 0.7 or circularity < 0.35) | `rule:duplicate` (12 µm suppression, now over the whole core); subtype mature tangle / pretangle by the model's p90 threshold |

`class` and `subtype` are label-vocabulary ids (`stainid.labels.vocabulary`); `threshold` is the cut-off the call used.

## Stage 3: outlines

Segment Anything ViT-B, one point at the candidate as the one-click tool, all three outputs kept small → large
(`sam_masks`, "|"-separated polygons), with predicted IoU (`sam_scores`) and stability (`sam_stability`) and the default
(`sam_default`): the output the stain's
outline step uses (NeuN large, 6E10 medium, AT8 best DAB contrast) when it survives the runaway cap (more than 5 × the
candidate's area, touching the 1024 px window edge, or outside the stain's size limits), else the surviving output with
the best predicted IoU; none surviving → `sam_status` fallback_* and the candidate outline stands.

## Stage 5: measures and the mapping to the current tables

`measures/results_core.csv` and `results_donor_region.csv` have the columns of the current stainID results tables, and
`measures/cohort_morphology_core_features.csv`, `_donor_region_features.csv` and `_donor_region_wide.csv` those of the
frozen v1 tables the R analyses read (same aggregation code: `stainid.analysis.aggregation`, `field_aggregation`,
`mask_aggregation`). They are fed with per-tile and per-object rows made from the store:

| Column(s) | v2 source | Same meaning? |
|---|---|---|
| tissue areas (`tissue_area_mm2`, `v2_tissue_area_mm2`) | `tiles.analysis_area_mm2` (tissue at the core's glass level, exclusions removed, as before) | yes |
| NeuN `neun_positive_profile_count`, `neun_profile_density_mm2`, … | accepted NeuN candidates | yes; now includes recovered neurons (split merges, component tree) |
| `candidate_profile_count`, `neun_positive_fraction_of_candidates` | candidates that reached the model (no rule) | definition of "candidate" widened |
| `neun_positive_profile_area_fraction` | summed candidate areas of accepted neurons / area | approximately (was the label-mask union) |
| `neun_local_density_cv_100um` | accepted centres in the saved 100 µm windows | yes |
| 6E10 `v2_plaque_*`, `plaque_area_fraction`, `compact_fraction_v2` | accepted plaques (compact / diffuse / small) | yes |
| `amyloid_positive_area_fraction` (v1, the burden covariate) | union of the legacy-segmentation deposits in valid tissue / area | yes (split segmentation instead of the unsplit v1 one) |
| `compact_plaque_fraction` (v1) | compact / (compact + diffuse) of accepted plaques ≥ 15 µm | definition the same; the identity model is the project's forest, not the v1 CNN ensemble |
| `accepted_plaque_*`, `plaque_density_mm2` (v1) | accepted plaques | as above |
| `plaque_dense_core_*` | the stored smoothed-DAB quantiles at the model's morphotype threshold | approximately (interpolated) |
| `periplaque_nucleus_enrichment`, `nuclei_per_plaque` | per-plaque nuclei inside / within 15 µm; background = tile nuclei outside plaques and their 45 µm rings | background approximated (rings of nearby plaques are not merged) |
| `vascular_or_edge_amyloid_density_mm2`, `amyloid_speck_density_mm2` | `rule:vascular_or_edge`, `rule:speck` | yes |
| AT8 `at8_positive_area_fraction`, `tau_positive_area_fraction` (burden) | `tiles.at8_positive_area_mm2` (the legacy AT8 network map) | yes |
| `tau_neuron_*` | accepted AT8 candidates | yes; now includes recovered bodies |
| `thread_length_density_mm_per_mm2`, `thread_mean_width_um`, `thread_to_neuron_ratio_mm` | saved skeleton segments **outside accepted tau+ neurons** | changed: the old tracing ran through tau+ somata; the old length is `thread_length_with_somata_mm` in `tiles_AT8.csv` |
| `thread_branchpoints_per_mm` | per-tile branch points (whole skeleton) / length outside neurons | approximately |
| v1 `at8_noncompact_*`, `at8_compact_profile_*`, `at8_thread_length_*`, `edge_guard_field_count` | the v1 AT8 network measures, saved per tile | yes (`at8_compact_profile_median_area_um2` is empty: per-profile areas are not saved) |
| outline (mask) features `*_mask_*` | accepted candidates with a kept Segment Anything outline | yes; `plaque_multicore_fraction` is empty (core counts are not saved) |

To move the analyses over, point the R scripts' `data/analysis/<prefix>_*.csv` reads at `data/analysis/v2/measures/`.
