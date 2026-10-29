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
| White matter | `stainid v2 white-matter` | NeuN `candidates/`, `masks/` of the three stains, the NeuN images of cores to check, the registration | `white_matter/` | seconds per core; changes no analysis output (docs/white_matter.md) |

Stage 3 runs after stage 4 because by default it outlines the candidates stage 4 accepted or scored at least 0.2
(`v2.outlines.select`); candidates outlined before are kept when a later model selects more.

## Running

`tools/v2_run.sh all` runs every stage (stage 2 with `WORKERS` core images at once, stage 3 with one process per GPU)
and `tools/v2_run.sh decide` only stages 4 and 5 (after a new model, cut-off or rule). Both are resumable: a core image is
done once its manifest is written, and every run redoes the core images that are missing or out of date. `stainid v2
status` shows, per stage and stain, how many core images are done, out of date and in all; the Run page shows the same
for the Detect steps *Find candidates*, *Decide*, *Outline objects* and the Results step *Make results tables*
(*Run what's left* runs them; the first detection's steps are under *Legacy detection (v1)*). Stage 2 decodes each core
image once into shared memory (about 0.8 GB, plus 0.5 GB of core-wide masks) and works on its tiles in a pool of
single-threaded tile processes (about 1.3 GB each, up to 2.3 GB on a 6E10 tile) shared by three core images in flight
(`--workers`); `--processes` sets the pool (default: a quarter of the processors, bounded by the job's or the free
memory), `--processes 0` gives the older layout, one process per core image. See "Speed" below.

## Manifests and out-of-date detection

Each stage writes `manifests/<stage>/<stain>/<core_id>.json` after its outputs: the stage's code version
(`stainid.v2.manifest.VERSIONS`), its parameters (`stainid.v2.params`, overridable under `v2:` in stainid.yaml) and a
fingerprint of every input — the core image file, the number of tiles with nuclei, the slide threshold, the tiles, the model bundle (folder name and sha256) and its cut-off, and the digest of each upstream
stage's manifest. A core image is out of date when any of these differs from what it would be made from now. Changing
`v2.rules` makes only stage 4 (and the store) out of date; changing `v2.proposals` stage 2 and everything after it.

## Stage 2: proposals

Excluded areas are detection's own only: linear artifacts and folds (NeuN: linear artifacts; NeuN neurons on a fold
are kept and judged by `rule:fold`), no tissue, and the AT8 edge guard. Their pixels are as if they were not there: no
proposal starts or grows into them, and they count in no measure — no numerator (DAB-positive area, amyloid extent,
AT8 positive area and threads, nuclei, candidates), no denominator (analysis and valid areas), no background or seed
statistic (the 6E10 levels, the 60 µm local background, the ring around a candidate, the context features of the NeuN
and AT8 models), and not in the slide threshold (`stainid calibrate`). The hand-drawn polygons are not an input, and
people's marked and cleared areas (`stainid.exclusions`) never change the analysis: they only leave labels out of
training and evaluation (`exclusions.drop_excluded`). Smoothing at a few pixels' scale can still carry stain across
the border of an excluded area by those few pixels.

**The linear-artifact mask is relative to the core's tissue** (`v2.proposals.linear_artifact_contrast`, 0.1). A band
(fold edge, knife line, scratch) is at least 0.35 dark (1 − grey level, smoothed) and also 0.1 darker than the core
image's own tissue (`tissue.core_tissue_darkness`: the median over the tissue outside brown stain, at 1/8 scale, once
per core image; a field's own median would be the fold's in an edge field that is mostly fold). With the fixed 0.35
alone, tissue that is itself nearly 0.35 dark became a band: on Tangle-Tracer a median 50 % of each region's tissue
(36 % of its test tangles never proposed), on Keiser 37 % of the tissue, and on the study's darkest 6E10 cores (LIP-1
D-3, E-3, E-4, D-4, F-4, E-5; LIP-6 D-5) a band of ordinary tissue tens of µm wide along the core edge, plaques in it.
Only core images whose tissue is darker than 0.25 can change (9 of 349 6E10, 3 of 364 AT8 study cores; on the 7 darkest
6E10 cores 17 plaques gained, 6 lost, nearly all of them plaques by the core edge); real folds and knife lines stay
masked (597 study tiles where the old mask took over a quarter of the tissue: 12.6 % → 12.5 % of their tissue).
`null` restores the fixed cut. Calibration and the viewer's excluded-areas overlay use the same mask.

Per tile and stain, each tile detecting in a window that reaches `proposals.<stain>.context_px` beyond it: as far as
the largest object reaches from its centre (accepted whole objects of the cohort reach at most NeuN 221 px, AT8 168 px,
6E10 926 px = 253 µm; the windows are 256, 256 and 1024 px). So the tile that owns an object's centre sees all of it,
and every object is detected whole, once: a tile keeps the candidates whose centroid lies in it or within 64 px of it,
and of the copies of one object (rows of different tiles whose outlines overlap with IoU ≥ 0.3, or one lying half
inside the other; `proposals.owned_once`) the most complete stays — not cut by its crop, then centroid in its own tile,
then the largest — and it belongs to the tile holding its centroid (`tile_id`, `x_tile`, `x_field` are of that tile).
Two tiles' windows can segment one object slightly differently, its centroid then on either side of their border:
such whole copies are found by the same overlap rule; objects that only touch are never copies. With the project's 128 px context alone, a
plaque wider than that was cut at every crop and kept in pieces (2–4 at a corner of four tiles), some with straight
crop edges. The window only lets a tile see further: every crop-wide level is taken over the tile's standard crop (the
tile and the project's 128 px context, `Field.standard`) — the 6E10 legacy segmentation's background median and spread
(so its seed and extent levels), the stained levels of the coarse and unproposed searches, and the AT8 network's
positive level and edge-guard trigger — and the fold and linear-artifact masks come from the standard crops too. Taken
over the wide crop, the background moved the 6E10 extent level by up to 8% per tile and a lower level joined
neighbouring deposits into one. The 6E10 legacy split seeds are the h-maxima of each deposit's own extent component
(`segmentation.component_h_maxima`), not of the whole crop, where a brighter deposit that the tissue background joins
a deposit to could remove or add a seed depending on how far the crop reaches (and the reconstruction ran over the
whole crop: per deposit it is about 2.5 times faster). This changed 1 of about 1,200 legacy deposits on 16 tiles.

Only the objects that need the window pay for it (`proposals.whole_objects`): a tile whose window is wider than 384 px
(6E10) is first detected over the tile and 384 px, and every candidate it keeps that comes within 176 px of that crop's
edge is detected again in a local window around it (its box and 256 px, grown by 512 px on each side the object still
comes within 176 px of, or its centroid within 260 px of, up to the tile's window; corners on the window's pixel
parity), at the tile's own crop-wide levels (`Field.reference`). The cut candidate gives way to the local regions
covering at least a quarter of it whose centroid the tile keeps. The margins hold everything a candidate's measures look
at (the 140 µm review patch reaches 256 px from the centroid, the 45 µm nuclei ring 165 px from the outline, the 4 µm
smoothing about 45 px), so this is the whole window's result: on four 6E10 cores the same 9,826 candidates and outlines,
tile measures and unproposed regions as the 1024 px windows, in 215–385 s per core instead of 875–1,300 s. The tissue areas come from the
standard crop in every case (`Field.tissue_area`: a hole is filled when the standard crop encloses it), as do the fold
and linear-artifact masks.

The window changes only what detection sees. The tiles, their ids, the tissue areas (each tile's own area), the
calibration and the nuclei stay as they are: the linear-artifact and fold masks are made per tile on its standard crop
(the tile and 128 px, as the calibration makes them) and cut from there for the wider window
(`proposals.automatic_masks`), and the window's nuclei are the tile's own Cellpose crop and, beyond it, the other
tiles' nuclei of their own areas (`proposals.window_nuclei`; a nucleus on the seam keeps the part on each side).

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
  shape rule called artifacts and never showed the classifier (`flag_artifact_shape`); pale diffuse plaques as one
  region each (`coarse`: the DAB smoothed over 4 µm at max(0.75 × threshold, tissue median + 0.25 × threshold),
  ≥ 100 µm² — at the pixel scale such a plaque is a cloud of specks below the legacy seed level); faint deposits of the
  tree at 0.5, 0.75, 1 and 1.5 × the threshold (roots: a dense core inside a diffuse halo is one deposit). A tree node
  that merges deposits already kept, or is mostly covered by them, gives way to its children (`descend`): on a
  dark-background core (LIP-1 E-3: background median 0.06 OD, legacy seed level 0.26 OD, 2.5 × the threshold) the faint
  field at 0.5 × joined unseeded plaques to seeded ones, and the plaques were never proposed. Each deposit also gets the
  slide-scaled model inputs `s_<name>`: the 14 DAB intensities of the identity model in slide thresholds (max, p90,
  inner and outer mean, centre, annulus, context, contrasts), and the seed and dense core measured at levels relative to
  the threshold (seed above max(1 × threshold, median + 6 σ), core above max(1.25 × threshold, 1.2 × seed, the 85th
  percentile)) instead of the legacy absolute 0.035 / 0.10 OD (`stainid.v2.amyloid.scaled_names`). The current model
  still reads the legacy `m_` inputs; see "Retraining on slide-scaled inputs" below.
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
picture `masks/<stain>/<core>.png` (bits: 1 tissue, 2 linear artifact, 4 fold, 8 manual exclusion — not set any more, 16 AT8
edge guard, 32 analysed tile).

### Speed

Profile of the re-run's code (py-spy, LIP-1 F-3, one process with 2 filter threads): a core image kept 1.06 (6E10),
1.21 (AT8) and 1.11 (NeuN) processors busy; the second thread worked only in the AT8 ridge filter and the
linear-artifact openings (4%, 22% and 8% of the wall time). Busy time: the linear-artifact and fold masks 25 / 21 /
34% (6E10 / AT8 / NeuN), stain separation, smoothing and levels 17 / 8 / 30%, split and watershed 18% of 6E10, the
AT8 thread network (ridge filter included) 53% of AT8, proposal sources 15 / 7 / 9%, features 7 / – / 9%, nuclei 5 /
4 / 9%, decoding 3 / 2 / 4%, writing nothing. On a dense AT8 core (LIP-5 D-5) a quarter of the time went to
rebuilding a whole-field mask for every candidate.

Fixed, each output byte-identical (7 core images of all stains checked): the per-candidate mask (dense AT8 tiles 23%
faster), the per-component peaks of the 6E10 split (scipy sorted the whole picture), the ridge filter only in the
256-px squares where a thread can be (sparse AT8 tiles 29% faster; bit-exact), saturation and skimage HED by 8-bit
lookups, and the tile processes.

Scaling (9 core images near the median cost, 3 cores x 3 stains; the machine otherwise idle): the throughput is the
same, about 80 core images an hour, from 8 to 16 tile processes, and lower beyond (24: 69, 32: 70); the processor
seconds the same work takes rise from 2250 (4 processes) to 3080 (8), 6310 (16) and 12950 (32). 16 processes on 16
physical cores are no faster than 8, and 16 on 8 physical cores (both hardware threads) are 18% slower than 8 on 8:
the ceiling is memory bandwidth, not SMT. One process streaming arrays already takes the machine's 19 GB/s (read) to
25 GB/s (copy); a 5950X with two channels of DDR4-3200 usually gives about 35-45 GB/s, so the memory configuration
(both channels populated, XMP speed) is worth a look. Two threads per process, huge pages and 2 or 5 core images in
flight change nothing beyond noise. The re-run's layout (12 processes x 2 threads) did the same work in about 420 s
of machine time; 8 tile processes take 393 s and leave three quarters of the processors free, and one core image
alone now finishes in about half the time (LIP-1 F-3 NeuN: 161 s -> 81 s with 8 processes). More throughput needs
less memory traffic per tile (float64 temporaries, whole-field pictures made for a few crops), not more processes.

Stained regions no candidate proposes (`unproposed/`, per stain `proposals.<stain>.unproposed`): NeuN and AT8 as
before (smoothed DAB ≥ the threshold, ≥ 15 / 20 µm², touching no candidate); 6E10 the DAB smoothed over 4 µm at
max(0.75 × threshold, tissue median + 0.25 × threshold), ≥ 100 µm², that no single candidate covers a quarter of — a
pale diffuse plaque that only specks of candidates touch is reported, which the old rule (DAB ≥ threshold, touching no
candidate) never did.

## The bound on a slide's threshold

`stainid calibrate` sets a slide's DAB threshold from its own tissue (the larger of Otsu's split, median + 6 MAD and
0.01 OD), so it follows the slide's DAB background: on every set it lies near the tissue's median + 6 MAD and 1.3–1.5 ×
its 90th percentile. The models read absolute OD, and the AT8 one was trained on slides at 0.017–0.046 OD. Public AT8
slides are browner throughout (median tissue DAB 2.4–6 × the study's), so their thresholds came out at 0.075 (Tangle-
Tracer), 0.080 (Emory) and 0.170 (ADNP-15) OD, and the proposals there kept only the darkest part of each tau+ neuron
and missed the pale ones. `v2.calibration.max_threshold_dab_od` bounds the threshold per stain (AT8 0.05, above every
study slide, so the study's thresholds are unchanged; NeuN and 6E10 unbounded). `threshold_dab_od` is the bounded
value every step uses and `calibrated_threshold_dab_od` the tissue's own. `stainid calibrate` without `--redo` applies
a new bound to the kept slides without sampling them again; it only ever lowers a kept threshold.

## Retraining on slide-scaled inputs

The current models take DAB intensities in absolute OD, and some measures are cut at absolute OD floors; slides differ
3–4 fold in staining strength (thresholds 6E10 0.026–0.108, AT8 0.014–0.055, NeuN 0.026–0.099 OD). Stage 2 therefore
also stores slide-scaled inputs for every stain (`s_<name>`, `stainid.v2.scaled`): 6E10 as above; NeuN mean DAB, the
80th percentile around the nucleus, the context DAB and the stained area above 0.35 × threshold without the 0.012 OD
floor; AT8 the soma's DAB 90th percentile, the background median and the context DAB (its contrast and soma DAB are
already in thresholds). A new model trains on them when `v2.training.<stain>.slide_scaled` is set (off for every
stain: on the recalibrated thresholds the October 2026 comparison found scaled inputs no better leave-one-TMA-out —
6E10 AUC 0.965 against 0.968, AT8 0.897 against 0.901, NeuN 0.934 against 0.953; LIP-6 6E10 0.917 against 0.925); an AT8 model so trained also cuts mature tangles on the scaled 90th percentile (`mature_feature` in the bundle). The
current models keep reading the legacy `m_` inputs until a new one is made active.

The paragraph below describes the models before the 18 Oct 2026 re-run. LIP-6 6E10 is not 3–4 × paler: its stale
threshold (0.026 OD) was; recalibrated it is 0.064 OD, within the cohort's 0.061–0.107, and its labelled plaques' OD is
at the low end of the others (median smoothed OD 0.075 against 0.083–0.181).

The current 6E10 model takes absolute DAB OD and dense-core / seed measures cut at absolute OD. The LIP-6 6E10 slide is
3–4 × paler than the others (threshold 0.026 OD against 0.067–0.108), so its plaques have low intensities and no dense
core: the October 2026 audit found 5,099 of the cohort's 11,060 large (≥ 300 µm²) candidates rejected with p 0.25–0.5
on LIP-6 (41 % of its large plaques; 1.5–7 % elsewhere), and LIP-6's median diffuse-plaque density 0.6 /mm² against
6–31 /mm² on the other TMAs. Shape features of the ragged outlines of diffuse plaques (low circularity and solidity)
pull them down too. The next 6E10 model is trained on the annotation rounds' labels with the slide-scaled inputs
(when `v2.training.6E10.slide_scaled` is set; `stainid.training.train.with_scaled`): every input with an `s_`
version is replaced by it, and the bundle records `slide_scaled: true` and the `s_` names, which stage 4 reads from the
store like any other input. Labels on candidates without `s_` columns (the old pool) get their intensities from the
OD over their TMA's threshold and leave the core measures to the model's imputer. Without the setting, models train
on the legacy inputs.

## Stage 4: calls

The stain model scores every candidate from `m_` columns (`stainid.candidates.model_matrix`). Rules (`v2.rules`):

| Stain | Before the model | After the model |
|---|---|---|
| NeuN | `rule:excluded` (≥ 50 % excluded, or a Cellpose cell in a hand-drawn exclusion), `rule:near_profile`, `rule:pale_nucleus` (legacy Cellpose filters), `rule:shape` (legacy profile size / shape filter); for the new sources only: `rule:edge` (≥ 50 % within 5 µm of an edge or hole), `rule:fold` (≥ 50 % on a fold band with a weak edge, DAB gradient < 1 threshold per px: patches of the band; neurons on a band keep a sharp edge — the legacy NeuN detection never masked folds), `rule:irregular` (solidity < 0.6) | — |
| 6E10 | `rule:excluded`; `rule:artifact_shape` off by default (the legacy shape rule; the classifier now decides) | `rule:speck` (≤ 10 µm), `rule:vascular_or_edge` (outer boundary ≥ 25 % on a lumen or tissue edge); subtype compact / diffuse / small by the model's morphotype threshold |
| AT8 | `rule:excluded` (also a centre outside valid tissue), `rule:low_contrast` (< 0.3, the legacy bar), `rule:small` (new-source bodies < 40 µm²), `rule:edge`, `rule:irregular` (new-source bodies less compact than legacy bodies: solidity < 0.7 or circularity < 0.35; both off with `min_body_um2: 0`, `body_min_solidity: 0`, `body_min_circularity: 0`. Flame-shaped tangles fail them, so a project of such slides whose forest is refitted on its own labels may turn them off; on the study they stop few real cells, and without them the forest accepts about 7 % more objects, mostly dense thread swellings) | `rule:duplicate` (12 µm suppression, now over the whole core); subtype mature tangle / pretangle by the model's p90 threshold |

`class` and `subtype` are label-vocabulary ids (`stainid.labels.vocabulary`); `threshold` is the cut-off the call used.

## Stage 3: outlines

Segment Anything ViT-B, one point at the candidate as the one-click tool, all three outputs kept small → large
(`sam_masks`, "|"-separated polygons), with predicted IoU (`sam_scores`) and stability (`sam_stability`) and the default
(`sam_default`): the output the stain's
outline step uses (NeuN large, 6E10 medium, AT8 best DAB contrast) when it survives the runaway cap (more than 5 × the
candidate's area, touching the 1024 px window edge, or outside the stain's size limits), else the surviving output with
the best predicted IoU; none surviving → `sam_status` fallback_* and the candidate outline stands. An output under
0.25 × the area of a candidate of 100 µm² or more has shrunk onto a part of it (`fallback_shrunk`, `v2.outlines.
shrink_factor` / `shrink_min_um2`): on pale diffuse plaques Segment Anything often outlined one dark speck (a 45 µm²
outline for a 4,278 µm² plaque), and the viewer drew that speck as the plaque.

Which outline is an object's own for display is a per-stain setting, `v2.outlines.default_outline` (advanced; default
NeuN `sam`, 6E10 `stain`, AT8 `stain`: SAM clips diffuse plaques to their dense part, is as good or better on neurons
and mixed on AT8). The assembled store records it per row (`outline_default`); one function chooses for every view,
`candidates.outline_options`: the three SAM masks small → large, then the stain outline as "mask 3", shown first where
the setting is `stain` or SAM fell back, else the stored default output. The viewer's outline layer
(`candidates.outline_rings`) and the annotation pages (`labels.candidate_store.masks`) both call it. Changing it
re-assembles the store only; the measures take each object's area from the outline this setting draws
(`candidates.shown_area_um2`), so what is drawn is what is measured.

**NeuN positive area on the SAM outline depends on each core's size mix.** On the same accepted neurons, the SAM
outline is larger than the stain outline for small profiles and smaller for large ones. On the 7 NeuN ablation cores
(17,662 neurons, ablation step 6 → 7), SAM area / stain-outline area is:

| Stain-outline area | Sum ratio |
|---|---|
| 30 µm² or less | ×1.68 |
| 30–60 µm² | ×1.30 |
| 60–120 µm² | ×1.02 |
| 120–250 µm² | ×0.79 |
| Over 250 µm² | ×0.72 |

Pooled over the 7 cores, the area rises only ×1.06. Per core, `neun_positive_profile_area_fraction` moves with the
size mix:

- Cores of small profiles go up: LIP-1_A-4 ×1.43, LIP-6_A-2 ×1.46, LIP-1_F-3 ×1.32.
- Cores of large neurons go down or stay level: LIP-1_E-3 ×0.80, LIP-10_B-1 ×1.00.

The 7-core median, +32%, comes from this per-core size mix alone. The object set, the denominator and the units are
unchanged: the same accepted ids, area within 0.04%, and density +0.02%. So positive area on SAM narrows the contrast
between cores; for example, E-3 / A-4 drops from 3.5× on the stain outline to 1.9×. Neuron counts and density are
unaffected. The outline-model panel's ×0.86 (`data/analysis/outline_model_panel/summary.md`) is consistent size for
size, but its cores are large-neuron-rich, so it does not carry over to the cohort mix. NeuN keeps SAM as its default
outline (decided for the v2b re-run). Read NeuN positive-area results with this effect in mind. The decomposition is
`data/analysis/v2_audit/ablation/neun_step7_objects.csv`.

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
| `neun_positive_profile_area_fraction` | summed areas of accepted neurons on their default outline (SAM) / area | approximately (was the label-mask union); depends on the core's size mix, see the default-outline note above |
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

### Continuous type scores: tau maturity and plaque compactness

Stage 4 types each accepted tau+ neuron as *mature tangle* or *pretangle* and each plaque of at least 15 µm as
*compact* or *diffuse* by one cut on one feature. Both are gradients, so stage 5 also gives every accepted object a
score from 0 to 1 and a bin of three (`stainid.v2.scores`). They come from the features stages 2–4 stored (no image is
read, nothing is detected again), and the two-class `subtype` stays exactly as it was: the scores only add columns.

**The formula.** `score = 1 / (1 + exp(-Σ weight × (feature - centre)))`, every weight positive, so the score rises
with every feature:

| Score | Feature | Weight (per unit) | Centre |
|---|---|---|---|
| tau maturity (AT8) | ln(soma DAB 90th percentile / the model's mature cut-off, `mature_p90_threshold`, 0.159 OD) | 3.70 | 0 |
| | soma mean DAB minus its 2–10 µm ring's median, in slide thresholds (`m_normalized_contrast`) | 0.281 | 2.83 |
| | the soma's DAB coefficient of variation: an uneven, fibrillar fill (`m_soma_dab_cv`) | 1.43 | 0.597 |
| plaque compactness (6E10) | ln(inner mean DAB in slide thresholds / the model's morphotype cut-off, 1.53), the core's density | 0.749 | 0 |
| | inner minus outer mean DAB of the radial profile, in slide thresholds (`s_radial_dab_contrast`), the core's contrast | 1.47 | 0.777 |
| | share of the deposit at or above the morphotype cut-off (stored smoothed-DAB quantiles), the dense-core fraction | 5.18 | 0.197 |
| | deposit mean DAB minus its surrounding ring's, in slide thresholds (`boundary_contrast`), the edge's sharpness | 1.39 | 0.717 |

The first term of each is the feature stage 4's type rule cuts, as the log of its ratio to the active model's
cut-off; it is 0 at the cut. The other features are centred on the study's accepted objects' medians. So an object with
median other features scores exactly 0.5 at stage 4's cut, and a retrained model's cut moves the score with it. Score
0.5 is where the score's two classes meet the existing ones: on the study 94.8 % of the 36,367 accepted tau+ neurons
and 95.2 % of the 144,023 typed plaques fall on the same side of 0.5 as their type. A missing feature counts as its
centre (31 plaques without a ring).

The weights were fitted on the study's typed labels (label set "ai": 412 tau+ neurons, 240 mature tangles; 387 plaques
of at least 15 µm, 160 compact): a logistic regression with non-negative weights on the features divided by their
interquartile range over the accepted objects. Features the fit gave no weight were dropped: AT8 the soma's stained
share (`m_positive_fraction`) and the share of it a nucleus covers (`nucleus_overlap_fraction`, the nearest stored
measure of a hidden or displaced nucleus; no displacement is measured); 6E10 solidity. No fibrillar-texture measure
other than the DAB coefficient of variation is stored. `python tools/type_scores.py --project <study>` refits and
prints the terms and bin cuts (read-only); on the study it reproduces the table.

**Bins** (`v2.scores.<stain>.bins`, two cut points): the scores at which the label set's own calibration (the fitted
intercept added back: AT8 +0.20, 6E10 −0.61) gives the two labelled classes 2:1 odds either way, rounded to 0.01.

| Score | Low bin | Intermediate | High bin |
|---|---|---|---|
| tau maturity | pretangle < 0.29 | 0.29–0.62 | mature ≥ 0.62 |
| plaque compactness | diffuse < 0.48 | 0.48–0.79 | compact ≥ 0.79 |

On the study (the October re-run's store): tau+ neurons 6,442 pretangle, 8,423 intermediate, 21,502 mature (stage 4:
10,633 pretangles, 25,734 mature tangles; the intermediate bin holds 4,092 of the pretangles and 4,331 of the mature
tangles); plaques 66,978 diffuse, 27,956 intermediate, 49,089 compact (stage 4: 67,923 diffuse, 76,100 compact; the
intermediate bin holds 24,154 compact and 3,802 diffuse). Plaques under 15 µm (`small`, 9,508) get a score and no bin.
In the intermediate bins the labelled classes are even, as the rule intends: 42 pretangles and 40 mature tangles, 40
diffuse and 40 compact plaques. Of the labelled objects in the low bins 100 of 113 are pretangles and 169 of 190
diffuse plaques; in the high bins 187 of 217 are mature tangles and 99 of 117 compact plaques.

**The simple vocabulary's three amyloid steps.** The bins do not map onto *diffuse amyloid beta* / *diffuse plaque* /
*plaque with cores* one to one, and the labels cannot test it: no 6E10 object has a simple label yet. Through the
simple classes' training mapping (*plaque with cores* → compact, *diffuse plaque* → diffuse) the labelled compact
plaques lie 99 compact, 40 intermediate, 21 diffuse and the diffuse ones 18 / 40 / 169: the high bin is *plaque with
cores* and the low bin *diffuse plaque*. *Diffuse amyloid beta* (a haze with no edge) is a split at the diffuse end,
below what the bins separate; the pale regions the coarse source proposes, the nearest thing to it, score low (median
0.11; 41 of 55 accepted ones in the diffuse bin). The intermediate bin is a step the simple vocabulary does not have.

**Readouts** (`results_core.csv`, `results_donor_region.csv`; per accepted object `type_scores_<stain>.csv` with the
score, the bin and stage 4's type): `tau_maturity_*` and `plaque_compactness_*`, each `scored_count` (binned objects),
`score_median`, `score_iqr`, `score_mean`, `<bin>_fraction` (given for at least 5 tau+ neurons / 10 plaques, as
`tau_neuron_mature_fraction` and `compact_fraction_v2`) and `<bin>_density_mm2` (per mm² of the stain's analysed
tissue, `v2_tissue_area_mm2`). Across the 365 study cores the score median follows the two-class fractions (Spearman
0.97 with `compact_fraction_v2`, 0.84 with `tau_neuron_mature_fraction`). Run on the study's store, stage 5 with the
scores left every column of every table identical (as text) and added these 20 columns and the two object tables.

**Validation.** AUC of each score for the classes, on the study's labels and on the public datasets' stored stage
outputs (the study's models, no refitting; marks matched to the accepted object inside each box, Plaquebox to the
typed plaque within 25 µm of the crop's centre). *Within image* compares only objects of the same image or slide.

| Set | Classes | n | Score AUC | Type feature alone | Stage 4 type | Within image / slide |
|---|---|---|---|---|---|---|
| Study, label set "ai" | mature tangle vs pretangle | 240 / 172 | 0.878 (leave-one-TMA-out 0.867) | 0.872 | 0.791 | – |
| Study, label set "ai" | compact vs diffuse | 160 / 227 | 0.908 (leave-one-TMA-out 0.902) | 0.894 | 0.812 | – |
| Wong 2022 (4G8, 5 experts) | consensus cored vs diffuse | 412 / 4,526 | 0.685 | 0.674 | 0.618 | 0.663 (slide) |
| Keiser 2023 (4 antibodies) | consensus cored vs other plaques | 209 / 2,372 | 0.571 (4G8 0.70, 6E10 0.81, Aβ40 0.59, Aβ42 0.50) | 0.594 | 0.574 | 0.562 (image), 0.598 (slide) |
| Plaquebox hold-out (Tang 2019) | cored vs diffuse crops | 72 / 6,457 | 0.545 | 0.562 | 0.534 | 0.622 (slide) |
| Plaquebox training set | cored vs diffuse crops | 1,603 / 3,694 | 0.636 | 0.670 | 0.594 | 0.613 (slide) |
| Emory (Vizcarra 2023), experts' majority | iNFT vs Pre-NFT | 483 / 135 | 0.397 | 0.364 | 0.494 | 0.784 (image) |
| Emory, the authors' test consensus | iNFT vs Pre-NFT | 267 / 17 | 0.501 | 0.538 | 0.457 | 0.873 (image) |

The scores carry the experts' gradient where several experts typed the same object. On Wong 2022 the score's median
rises with the number of the five experts who called a plaque cored: 0.55 (none, 4,267), 0.74 (1), 0.77 (2), 0.81 (3),
0.88 (4), 0.85 (5). On Emory it rises with the experts' share of iNFT votes: 0.29 (none of them), 0.28 (a quarter),
0.52 (half), 0.53 (three quarters), 0.63 (all, 506). Within an Emory image the score ranks iNFT above Pre-NFT (0.78 and
0.87); pooled over images it does not (0.40 and 0.50), because the soma's DAB level differs more between the Emory
images and antibodies than between their two classes (PHF-1 Pre-NFTs median 0.69, iNFTs 0.65; AT8 0.08 and 0.46);
scaling by the slide threshold does not change that (0.40), as the AT8 thresholds there are bounded. Raters box cored
plaques narrowly (see the public scores), so the other plaques of Keiser and Plaquebox include many dense ones; the
score separates cored plaques better within a slide than pooled. Montages along each score (low to high, ten bands):
`/run/media/sky/wd6t/scores_check/montages/`.
