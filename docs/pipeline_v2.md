# v2 detection: stages, the candidate store, and the measures

The v2 detection (`stainid.v2`) splits detection into stages that each read the previous stage's saved outputs, so a
change to a model, a threshold or a rule re-runs only the cheap stages after it. Everything is written under
`outputs.v2` (default `data/analysis/v2/`); the current detection outputs (`cohort_neun`, `cohort_v3`,
`cohort_object_masks`) and results tables are not touched.

| Stage | Command | Reads | Writes | Cost |
|---|---|---|---|---|
| 1 Nuclei | `stainid nuclei` (unchanged) | core images | `tile_nuclei/<tile>.npz` | GPU, run once |
| 2 Proposals | `stainid v2 proposals` | core images, nuclei, slide thresholds, exclusions | `proposals/`, `tiles/`, `threads/`, `unproposed/`, `masks/`, `cells/` | CPU, the expensive stage |
| 4 Calls | `stainid v2 calls` | `proposals/`, the stain models, `v2.rules` | `calls/`, `candidates/` | minutes for the cohort, no image read |
| 3 Outlines | `stainid v2 outlines` | `proposals/`, `calls/` (which to outline), core images, SAM | `outlines/`, `candidates/` | GPU, optional |
| 5 Measures | `stainid v2 measures` | `candidates/`, `tiles/`, `threads/` | `measures/` | minutes, no image read |
| White matter | `stainid v2 white-matter` | NeuN `candidates/`, `masks/` of the three stains, the NeuN images of cores to check, the registration | `white_matter/` | seconds per core; read by stage 5 (with `exclude`, also what it counts; docs/white_matter.md) |
| Grey-matter burdens | `stainid v2 grey-matter` | `white_matter/`, the 6E10 / AT8 images with white matter, nuclei, slide thresholds | `white_matter/grey_tiles/` | 2-3 min per core image with white matter; read by stage 5 only with `v2.white_matter.analysis: exclude` (docs/white_matter.md) |

Stage 3 runs after stage 4 because by default it outlines the candidates stage 4 accepted or scored at least 0.2
(`v2.outlines.select`); candidates outlined before are kept when a later model selects more.

## Running

`tools/v2_run.sh all` runs every stage (stage 2 with `WORKERS` core images at once, stage 3 with one process per GPU)
and `tools/v2_run.sh decide` only stages 4 and 5 (after a new model, cut-off or rule); with `v2.white_matter.analysis:
exclude` both also run the white-matter step and the grey-matter burdens before stage 5. All are resumable: a core image is
done once its manifest is written, and every run redoes the core images that are missing or out of date. `stainid v2
status` shows, per stage and stain, how many core images are done, out of date and in all; the Run page shows the same
for the Detect steps *Find candidates*, *Decide*, *Outline objects* and the Results steps *Find white matter*, *Measure
grey-matter burdens* and *Make results tables* (*Run what's left* runs them; the first detection's steps are under *Legacy
detection (v1)*; docs/white_matter.md says when the two white-matter steps are not needed). Stage 2 decodes each core
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
`null` restores the fixed cut (the study and JOG pinned it until the v2c re-run, which removes the pin). Calibration and
the viewer's excluded-areas overlay use the same mask.

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

**A core that fills its picture.** Core QC (`qc.core_qc.tissue_mask`, which the whole-core tiling also uses) takes the
glass from the picture's corners outside the core's circle, with a cut-off of the corners' median darkness + 8 MAD. A
core that fills its picture leaves tissue (or a neighbour core, or white padding at a slide edge) in the corners, and
that cut-off (up to 0.6) took the whole core for glass: LIP-3 A-5, LIP-7 A-5 and E-5 and LIP-11 A-3 NeuN and LIP-11 A-3
AT8 were called empty (no tiles, no analysis) though they hold 60-90 % tissue. When the corner cut-off is above 0.1 and
leaves the core empty, the glass is now the 99th percentile of the whole picture (white and black padding left out) with
a fixed cut-off of 0.04 (the study's median is 0.043). Of the 1,134 core pictures only these five and one orientation
control change; every other mask is identical. The v2c re-run QCs and tiles the five again and runs their stage-1 nuclei
(`fix_empty.py` in the re-run's chain), the other core images' rows and tiles checked unchanged. Some cores at the array
edge called sparse or partial have the same high corner cut-off (LIP-1 A-5 AT8, LIP-10 F-1 6E10, LIP-11 A-5 and B-5
AT8, LIP-9 C-5 AT8): their tiles may miss tissue; left for after the re-run.

**Pure black is no tissue.** Scan padding and the slide's edge are pure black (RGB 0, 0, 0); the fixed tissue cut-off took
them for tissue, and at the fixed threshold they read as stained. `field_tissue_mask` (so the tissue of detection, the
areas, the calibration and the viewer) leaves them out before its opening and closing, so a stray black pixel inside
tissue is closed over, and `stainid calibrate` also leaves every pure-black pixel out of the slide's sample
(`workflows.calibrate.sample_core`). Tested on 333 slides (`/run/media/sky/Untitled/alzh/work/calibration_guard/`): two
Plaquebox training slides move (0.43 → 0.048 and 0.35 → 0.055 OD); the study's thresholds move by at most 6e-6 OD.

**Cells: the per-nucleus table** (`cells/<stain>/<core_id>.parquet`, `stainid.v2.cells`). Every Cellpose nucleus of the
tile's own area (enough hematoxylin, 10–200 µm², centre in the counted tissue) with its position (core px), area,
eccentricity, solidity, major and minor axis, mean hematoxylin and DAB and their spread within the nucleus
(`hematoxylin_sd`, `dab_sd`: chromatin texture, so nuclei can later be told neuron-like or other on 6E10 and AT8 from
size, shape, stain and texture), and `counted`: on 6E10 and AT8 a nucleus must
also hold at least as much hematoxylin as DAB. Cellpose outlines dense DAB objects there too (plaque cores, tangles,
thread swellings, about 7 % of its "nuclei"), and they are no cells. Every nucleus count of stage 2 takes the counted
cells: the tile's `nucleus_count` (the nucleus density, the periplaque background) and the 6E10 `nuclei_inside`,
`nuclei_ring15`, `nuclei_ring45`. The AT8 rings around nuclei, the thread network's nuclei and the AT8 model's own
inputs (`nucleus_count`, `nucleus_overlap_fraction` of a candidate) are unchanged. On NeuN (whose neuronal nuclei are
brown) only the hematoxylin and size rules apply; stage 5 reads that table (below). A new per-tile measure that needs
the image belongs in this per-tile step too (`tile_summary` stores the field's table, `process_tile` returns it,
`write_core` writes it), so it costs no second read of the core image. Proposals version 6 (the tissue mask, the counted
nuclei, the table).

**Biology measurements** (`stainid.v2.biology`; proposals version 7). Additions only: no model input, rule, type or
existing column reads them, and the ablation sample (21 core images, `rerun_v2c/scripts/ablation_arm.sh bio`) gives the
same labels metrics and the same 784 per-core measures as the code before them.

- *The tangle's nucleus* (AT8 and NeuN candidates, `soma_nucleus`): of the field's cells (the per-nucleus table's
  nuclei) the one with the most area inside the candidate, if at least half of it is. `soma_nucleus_visible` 1 when it
  holds at least as much hematoxylin as DAB (a blue nucleus seen through the stain; the rule by which 6E10 and AT8 count
  a cell), 0 when there is none or the stain hides it; `soma_nucleus_offset` its centre's distance from the soma's
  centroid in soma radii (sqrt(area / π)); `soma_nucleus_area_um2`, `soma_nucleus_hematoxylin_od`, `soma_nucleus_dab_od`;
  `soma_hematoxylin_share` the share of the soma's pixels with at least as much hematoxylin as DAB (no Cellpose needed).
  NaN when the field has no nuclei. On NeuN the nucleus is the neuron's own, stained brown: its area is what stage 5
  uses.
- *The plaque's radial profile* (6E10 candidates of at least 10 µm, `radial_profile`): from the deposit's peak (its DAB
  smoothed over 1 µm), the mean DAB of the valid tissue in 2 µm rings out to 46 µm (`radial_dab_um`, OD), and in slide
  thresholds over fixed bins: `profile_core_dab` (0–4 µm), `profile_inner_halo_dab` (4–10), `profile_outer_halo_dab`
  (10–20), `profile_background_dab` (25–40); `profile_core_contrast` (core minus inner halo), `profile_core_to_halo`
  (core over the mean of both halo bins), `profile_core_radius_um` (the first ring below halfway between the core and
  one threshold), `profile_extent_radius_um` (the first ring below one threshold) and `profile_halo_width_um` (their
  difference). A cored plaque with a halo: a strong core and a halo of several µm; a bare ("burnt-out") core: a strong
  core and a width near 0; a diffuse plaque: little core contrast.
- *Vessels* (every stain, on the counterstain, `vessels`): lumen candidates are the enclosed empty spaces at the core's
  glass level (holes not filled, as the tissue mask fills them), 50–20,000 µm², not touching the crop, centre in the
  tile; per lumen its shape (area, diameter, elongation, solidity, perimeter), its lining (cells centred within 4 µm
  outside it, how many are elongated, ratio ≥ 1.8, and lie along the wall), the nuclei of tissue it surrounds (a vessel
  in its perivascular space), the rim's hematoxylin, and on 6E10 the share of its 4 µm wall band above the threshold and
  the share of 16 sectors with at least 10 % of it positive (`wall_coverage`, a CAA-like ring). A lumen is a vessel when
  it is no tear (solidity ≥ 0.55), at least 8 µm across and lined along the wall by enough elongated cells (2, or one
  per 60 µm of perimeter; 1 for a lumen elongated 2.5-fold), or round and smooth and at least 15 µm across (solidity ≥
  0.85), or around at least 30 µm² of tissue. Per core image `vessels/<stain>/<core>.parquet` (every lumen with its
  call and outline, core px) and `vessels/<stain>/<core>.png` on the masks' grid (bits 1 vessel lumen, 2 its 4 µm
  wall band, 4 any other lumen).

Seconds per tile (median of 12 tiles per stain, 2 core images each, one processor, after stage 2's own pictures are
made): see "Biology measurements: pilots" under stage 5. Pilots and scripts: `/run/media/sky/Untitled/alzh/work/bio_features/`.

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

## Training between stages 2 and 4

Training reads each labelled object's features from the candidate store (stage 2's measurements, what stage 4 scores),
and from the legacy candidate pool only for an object the store does not have. A core image's store counts only when it
was assembled from the stage-2 proposals now on disk (its `candidates` manifest names the current `proposals` digest);
otherwise, after a re-run of stage 2 and before stage 4, training reads the new proposals
(`labels.candidate_store.training_kind`). Before, training matched labels only to the store: a retrain between stages 2
and 4 found no store after the outputs were moved aside (6E10: 961 of 1,267 labels dropped in v2b) or the earlier
proposals' features in a store left in place, and it took the pool's features before the store's.

A label names its candidate (`candidate_id`: stain, core image and the rounded centre). A re-run that moves an object
(a new mask, threshold or rule) changes its id. A label naming an id the table no longer has is matched by position
instead, like the labels made before ids existed: the nearest candidate within `candidates.MATCH_RADIUS_UM` (NeuN and AT8
4 µm, 6E10 6 µm), in training (`candidate_store.known`) and in every reader of the decisions (`labels.decisions._renamed`:
the rounds, the Labels page). Only a label with no candidate there any more is left on its own (its position).

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
| outline (mask) features `*_mask_*` | accepted candidates with a kept Segment Anything outline; none in measured tissue is a count, density and area fraction of 0 | yes; `plaque_multicore_fraction` is not written (the v2 store keeps no count of dense cores) |

To move the analyses over, point the R scripts' `data/analysis/<prefix>_*.csv` reads at `data/analysis/v2/measures/`.

### Cells on the NeuN section

From the NeuN per-nucleus table (stage 2, `cells`) and the accepted neurons' own outlines (the one the views draw), over
the analysed tissue of the NeuN tiles with nuclei (with `exclude` the cells in the white matter are left out as the
neurons are):

| Column | Definition |
|---|---|
| `neun_nonneuronal_cell_density_mm2` | counterstain nuclei whose centre lies in no accepted neuron, per mm² |
| `neun_cell_density_mm2` | all cells: accepted neurons plus those other nuclei, per mm² |
| `neun_neuron_fraction_of_cells` | accepted neurons / all cells (missing under 20 cells, `too_few_objects`) |

A neuron counts once whatever the nuclei inside it, so the share does not depend on whether Cellpose found the neuron's
nucleus. The tile rows carry `cell_count` (all counted nuclei) and `nonneuronal_cell_count`; a core image whose stage 2
predates the table has them missing. Proposed in `/run/media/sky/Untitled/alzh/work/feature_explore/PROPOSAL.md` (pilot:
NeuN+ share CT 0.24, ASYMAD 0.19, AD 0.17; non-neuronal cells 550, 605, 840 per mm²; a shape-only neuron-versus-other
nucleus model from this table reached leave-one-TMA-out AUC 0.87).

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

### Biology readouts (`stainid.v2.biology_measures`)

From stage 2's biology measurements (stage 2, "Biology measurements"), per core image and summed per donor-region
like the other columns; no image is read. Each needs its minimum count, else it is missing (`too_few_objects`); every
one is `not_measured` for a core image whose stage 2 predates them (`bio_measured_<stain>`, `vessels_measured_<stain>`
in `results_counts_<level>.csv`).

| Column | Definition |
|---|---|
| `tau_neuron_nucleus_visible_fraction` (`_mature`, `_pretangle`) | accepted tau+ neurons (of that subtype) with a visible nucleus; 5 neurons |
| `tau_neuron_nucleus_offset_median` | of those with a nucleus found, its offset in soma radii (median); 5 |
| `plaque_profile_core_contrast_median`, `_core_to_halo_median`, `_halo_width_um_median` | accepted plaques of at least 15 µm with a profile; 10 plaques |
| `plaque_bare_core_fraction` | of those, a core of at least 2 slide thresholds with a halo under 2 µm (a "burnt-out" core) |
| `plaque_haloed_core_fraction` | such a core with a halo of at least 4 µm |
| `neun_` / `amyloid_` / `at8_vessel_density_mm2`, `_vessel_area_fraction` | vessels of the stain's own section over its measured tissue |
| `caa_vessel_fraction`, `caa_vessel_density_mm2` | 6E10 vessels whose wall band is positive in at least half its sectors; 5 vessels |
| `plaque_vessel_touch_fraction`, `plaque_vessel_near_fraction` | plaques whose edge lies within 2 / 10 µm of a vessel lumen (masks' grid); 10 plaques |
| `tau_neuron_vessel_near_density_mm2`, `_far_`, `tau_neuron_vessel_enrichment` | AT8 tau+ neurons within 20 µm of a vessel lumen and elsewhere, per mm², and their ratio |
| `thread_vessel_near_density_mm_per_mm2`, `_far_`, `thread_vessel_enrichment` | the same for free thread (0.5 mm of thread) |
| `neun_neuron_nucleus_area_median_um2` | the nucleus inside accepted NeuN neurons (median); 20 |
| `neun_large_nucleus_density_mm2` | neuron-sized nuclei (at least 50 µm², the per-nucleus table) per mm² of the tiles with nuclei |
| `neun_large_nucleus_in_neuron_fraction` | of those, inside an accepted neuron; 20 nuclei |
| `neun_large_nucleus_weak_fraction`, `_unstained_fraction` | of those, in none and with DAB 0.5–1 slide threshold (weak) or under 0.5 (none): antigen loss rather than neuron loss |

#### Biology measurements: pilots (2026-11-03)

Scripts and outputs: `/run/media/sky/Untitled/alzh/work/bio_features/` (`scripts/pilot_features.py` measures stored
candidates on their cores with this code; `analyse_*.py`).

- **The tangle's nucleus.** Of 1,155 accepted tau+ neurons sampled from the study, a nucleus is visible in 6.6 % of the
  mature tangles and 33.5 % of the pretangles; on the 412 labelled ones a hidden nucleus picks the mature tangles with
  AUC 0.64, the soma's hematoxylin share 0.56, the offset 0.60 (mature tangles have the smaller offset: where a nucleus
  is found in one, it is often central). Offered to the maturity score (`tools/type_scores.py --extra`, held out by
  TMA): AUC 0.867 without, 0.870 with the hidden nucleus (weight 0.63 per unit), 0.869 with the offset as well, so the
  score keeps its three terms. On the Emory iNFT / Pre-NFT crops (618) the nucleus is visible in 5.2 % of both (AUC
  0.50): no signal there. Kept as a readout (the per-core visible fraction is the biology: on the ablation cores 3–10 %
  of mature tangles against 16–54 % of pretangles).
- **The plaque's radial profile.** Compact against diffuse (387 labelled plaques of at least 15 µm): the core's density
  `profile_core_dab` AUC 0.80, core radius 0.76, halo width 0.71, core contrast 0.61; core over halo 0.42 (diffuse
  plaques have the faint halo). Wong 2022 cored (1,571 plaques): 0.70, 0.67, 0.67; Keiser 2023 consensus cored (2,581):
  0.64, 0.68, 0.71. Offered to the compactness score the profile adds nothing held out on the study (AUC 0.902 →
  0.899), but on the public sets the stored score with the core density and halo width added rises from 0.67 to 0.72
  (Wong) and from 0.57 to 0.68 (Keiser): the profile is the more portable description of a core. Kept.
- **A neuritic halo from 6E10 alone** (small dark DAB puncta, 0.8–20 µm², in the 10–30 µm ring against the 30–46 µm
  ring, as APP in dystrophic neurites): dropped. Against cored labels the puncta go the wrong way (Keiser AUC 0.29: the
  puncta are the granular texture of diffuse amyloid, not neurites), and on 40 registered cores spanning the AT8
  neuritic-crown fraction (`feature_explore`, `neuritic_fraction_2x`) no per-core puncta measure follows it (Spearman
  −0.24 to +0.22, all p > 0.13). It was the costliest part of the 6E10 pass (a whole-field top-hat, 0.4–0.8 s a tile).
- **Vessels**, graded by eye (40 called vessels, all stains): 13 clearly vessels, 11 plausible, 16 not (pericellular
  spaces with a satellite nucleus, out-of-focus NeuN fields, tears): precision 33–60 %. Called vessels 8–14 µm across are
  mostly wrong (1 of 9 clear), those of at least 14 µm 12 of 31 clear and 21 plausible. Recall is low: only open lumens
  are found, and most capillaries in the sections are collapsed; densities on the ablation cores are 0.5–42 per mm²,
  far under the hundreds per mm² of cortical capillaries. The rejected lumens are mostly right (pericellular halos). The
  vessel readouts are therefore of large open vessels only and noisy; use them as exploratory.
- **NeuN stain against loss.** The per-nucleus table of 3 tiles each of 36 of the 43 NeuN-pale cores and 36 others
  (`white_matter.neun_pale`; 40,625 nuclei, the stored accepted neurons' outlines): neuron-sized nuclei (≥ 50 µm²) are
  about as many on the pale cores (median 87 per mm² against 113, Mann-Whitney p 0.24; all nuclei 829 against 703),
  but 3 % of them lie in an accepted neuron (others 84 %, p 1e-10) and 94 % carry no NeuN (DAB under half a slide
  threshold; others 8 %). Of the 28 pale cores with at least 20 such nuclei, 19 have them at half the others' density
  or more and mostly unstained (antigen loss: the neurons are there, the stain is not), 7 under half the density
  (neuron loss, or little grey matter). The weak band (0.5–1 threshold) is rare everywhere (0.6 % and 1.7 %): the
  loss is all or nothing. The nucleus inside accepted neurons is 57 µm² (median; pale cores 66).

Seconds per tile (median of 12 tiles per stain, 2 core images, one processor, after stage 2's own pictures are made;
`scripts/time_bio.py`):

| | 6E10 | AT8 | NeuN |
|---|---|---|---|
| per-candidate columns (profile; nucleus) | 0.11 (61 candidates) | 0.003 (8) | 0.17 (426) |
| vessels | 0.19 | 0.28 | 0.30 |
| per-nucleus table (`cells._all`, shared by every count; the spread adds about 0.01) | 0.33 | 0.31 | 0.37 |

The first pilot cost 3.6 s a tile on 6E10 (a ring template per plaque, the puncta top-hat) and 0.6–0.75 s of vessels:
the ring template is made once, deposits under 10 µm get no profile, the puncta are gone, the empty-space test runs on
the 8-bit values and lining nuclei are measured only near a lumen. Together the additions cost about 0.3 s (6E10, AT8)
and 0.5 s (NeuN) a tile, about 5 processor-hours over the study's ~52,000 tiles (under an hour of the re-run's 8
processes); the vessels are two thirds of it.

### Missing values: `results_missing_core.csv` and `results_missing_donor_region.csv`

A count, a density or an area fraction is 0 when nothing was found in measured tissue. A fraction or a shape measure stays
empty under its minimum count, with no pseudo-counts; tissue not measured at all stays empty. Every empty cell of
`results_<level>.csv` (also the `_whole_core` columns and the white-matter flags) has exactly one row in
`results_missing_<level>.csv` (`stainid.v2.missing`), long format: the id columns of the results table, then `measure`
(the column), `reason`, `n_objects`.

| `reason` | Meaning |
|---|---|
| `no_image` | no image of the measure's stain for this core (donor-region: for any of its cores) |
| `no_grey_matter` | the image has too little grey matter to measure (`v2.white_matter.min_grey_mm2`; donor-region: all its cores with an image) |
| `too_few_objects` | under the measure's minimum (`missing.MEASURES`) |
| `no_white_matter_boundary` | a depth or column measure of the spatial interactions on a core image without white matter to measure depth and the radial axis from |
| `no_registration` | a spatial-interaction measure on the three sections' joint grid for a core without a stored registration of both pairs (or a failed one) |
| `not_measured` | anything else: a white-matter share of a core whose white matter is undetermined, enough plaques but no far tissue for the periplaque background |

`n_objects` is what the minimum counts: outlines for the outline measures (minimum 20 NeuN, 10 plaques, 5 tau+ neurons),
scored and binned objects for the type scores (10 plaques, 5 tau+ neurons), plaques of at least 15 µm for the compact
fraction (10), tau+ neurons for the mature fraction (5), density windows for the patchiness (4), thread pieces for the
thread width and branching (whose minimum is 0.5 mm of thread); 0 for `no_image` and `no_grey_matter`. `results_counts_<level>.csv`
holds these counts per row. Under `measures/whole_core/` the same tables describe the whole-core results.

### Tangle spacing, thread ownership, pathology in white matter

New columns of `results_<level>.csv`, from stored results with no image read (`stainid.v2.organisation`, `_attach_white`
in `measures.py`). In the grey-matter tables they are grey matter only; the spacing and ownership columns also have their
`_whole_core` version.

| Column | Definition | Minimum (else empty) |
|---|---|---|
| `tangle_nn_median_um` | median distance from a mature tangle to the nearest other mature tangle of the same core image; a donor-region pools its cores' distances | 10 tangles with a neighbour |
| `tangle_l200_excess_um` | Ripley's L at 200 µm less its mean over 39 random placements of as many points in the same valid tissue (counted tissue of the analysed tiles, without white matter in grey-matter tables), so the core's shape and holes cancel; above 0 the tangles gather, below 0 they keep apart. Seeded by the core id, so a re-run gives the same numbers. A donor-region: the count-weighted mean of its cores that reach the minimum | 10 tangles in a core |
| `tau_processes_per_neuron` (`_mature`, `_pretangle`) | connected stretches of free thread, within 50 µm of the outline, that come within 2 µm of it, per accepted tau+ neuron | 5 neurons (of the subtype) |
| `tau_neurons_with_process_fraction` | share of tau+ neurons with at least one | 5 neurons |
| `tau_attached_thread_um_per_neuron` (`_mature`, `_pretangle`) | length of those stretches per neuron; a thread piece reached by k neurons gives each 1/k; a piece counts whole also when it runs past the 50 µm | 5 neurons (of the subtype) |
| `thread_attached_density_mm_per_mm2`, `thread_unattached_density_mm_per_mm2` | attached and other free thread over the AT8 area: they add up to `thread_length_density_mm_per_mm2` | none: 0 when nothing was found |
| `thread_attached_share` | attached thread / all free thread | 0.5 mm of thread |
| `plaque_density_white_mm2`, `plaque_area_fraction_white`, `amyloid_positive_area_fraction_white` | 6E10 plaques, their area and the amyloid-stained area inside the core image's white-matter mask, over its area | 1 mm² of white matter (`v2.white_matter.min_grey_mm2`) |
| `tau_neuron_density_white_mm2`, `thread_density_white_mm_per_mm2`, `at8_positive_area_fraction_white` | the same on AT8 | 1 mm² of white matter |
| `plaque_white_area_mm2`, `tau_white_area_mm2` | the white matter measured on that stain's image; 0 without white matter | none |
| `<neun,plaque,tau_neuron>_edge_touch_fraction`, `_edge_mostly_fraction` | QC: the share of accepted objects whose outline touches the 5 µm band along a tissue edge or hole or an excluded area, and of those mostly (≥ 0.5) in the band | 20 neurons, 10 plaques, 5 tau+ neurons |
| `tau_neuron_boundary_fraction` | QC: the share of accepted tau+ neurons flagged at a tissue boundary (below) | 5 tau+ neurons |
| `tau_neuron_density_no_boundary_mm2` | tau+ neurons without the flagged ones, over the same area as `tau_neuron_density_mm2` | none: 0 when nothing was found |

The white-matter columns are the whole-core numbers less the grey-matter ones; their grey-matter counterparts are the main
columns of the same row. An empty one is `not_measured` in `results_missing_<level>.csv`. `tau_neuron_threads.csv` keeps
ownership per neuron (`processes`, `attached_thread_um`). Neither these columns nor their inputs read the edge band or
`rule:edge`, except through which candidates the calls accepted: see the check below.

### Check of the AT8 edge band (2026-11-02)

Why do so many accepted tau+ neurons touch the 5 µm band along a tissue edge or hole? Measured on the whole store and on
287 crops graded by eye (`/run/media/sky/Untitled/alzh/work/feature_explore/edge/`: `grades.csv`, the pages, the scripts in
`../scripts/edge_*.py`). No detection was changed.

- Cohort: 28.0 % of accepted tau+ neurons touch the band (NeuN 15.1 %, plaques 10.9 %), 2.7 % lie mostly in it (NeuN
  4.4 %, plaques 0.05 %). It depends on the slide (LIP-6 2.6 % to LIP-11 49 %), and falls as the glass level rises
  (per core Spearman -0.47): `field.tissue` uses the fixed `TISSUE_DARKNESS` 0.10 while the slides' glass lies at 0.084–0.116,
  so pale holes are nearly invisible on some slides. The band is not comparable between slides.
- Cause (91 graded touching neurons): a separate vacuole or hole nearby 46 %, a tissue boundary or tear 18 %, only tiny
  pale specks 16 %, a halo around the soma 15 %, a vessel or large lumen 4 %. So the halo is read as a hole, but only
  in about one in seven; the usual cause is the spongy tissue around the neuron.
- Are they real? Touching: 66 % real, 25 % doubtful, 9 % not a cell body; not touching (39): 56 %, 41 %, 3 %. Every halo
  case is real. Of the 18 not-a-body among the 129 graded accepted, 17 are at a tissue boundary (a stained rim or a
  detached fragment). Touching the band is no sign of a false tangle; lying at a tissue boundary is.
- `rule:edge` drops 35,098 AT8 candidates (new sources only: the legacy `ring` and `dense` are exempt), against 36,367
  accepted. The model would have rejected 99.3 % of them anyway (1 of 65 graded is real). Of the 253 it would have accepted
  (0.7 % of accepted), 43 % of 54 graded are real, 37 % not a cell body: at most about 110 real neurons lost, 0.3 %.
- The model's own `edge_fraction` input has importance 0.008 (26th of 28).

What a fix would cost, none of it needed for the study's current numbers: dropping `rule:edge` (stage 4 and 5, minutes) adds
about as many false objects as real ones. A band at each core's glass level makes the QC comparable between slides, but
re-measures AT8 and NeuN candidates from the images (stage 2 on: the full re-run). A tissue-boundary rule from the stored
mask (no image read, stage 4 and 5: largest non-tissue area of at least 10,000 µm² within 15 µm) flags 6.3 % of accepted
tau+ neurons (LIP-1 2.8 %, LIP-3 12 %, LIP-12 12 %), catches 15 of the 18 not-a-body and costs 5 of the 76 real. With it
the tangle spacing moves little (176 cores: median change L excess 3 µm, nearest neighbour 1 µm; Spearman 0.985, 0.975).

### Tissue-boundary flag on AT8 tau+ neurons

The rule of the check above, as a flag: stage 5 flags an accepted tau+ neuron when a connected non-tissue area of at least
`v2.boundary_flag.min_nontissue_um2` (10,000 µm²) lies within `v2.boundary_flag.radius_um` (15 µm) of its centre, on the
stage-2 mask, whose cells are 2.2 µm (`organisation.mark_boundary`; no image read). **It only flags.** Flagged objects stay in
every count and every other column; detection and stages 2–4 are not touched; a changed setting changes only the flag's
columns. `objects_AT8.csv` gains `candidate_id`, `tissue_boundary` (1 / 0) and `nearby_nontissue_um2` (the area that decides
it, 0 with none near), and the results tables the two columns above (with `exclude` also as `_whole_core`). The flag does
not depend on the white matter: the grey-matter tables simply hold fewer objects. On whole slides the rule sees one region's
mask at a time, so glass cut by a region's edge counts only its part inside that region.

On the study (stage 5 from the live outputs into `/run/media/sky/Untitled/alzh/work/boundary_flag/`, 2026-11-03; every old
cell of all 42 tables identical, as text; the missing-value tables only gain rows for the new columns): 2,285 of 36,367
accepted tau+ neurons flagged whole-core (6.3 %), 2,234 of 35,407 in grey matter. By TMA (whole core) LIP-1 2.8 %, LIP-5
3.2 %, LIP-10 3.3 %, LIP-7 4.1 %, LIP-2 5.0 %, LIP-9 5.1 %, LIP-6 5.1 %, LIP-13 5.2 %, LIP-11 5.6 %, LIP-4 7.1 %, LIP-8
7.5 %, LIP-12 11.8 %, LIP-3 12.1 %. By group: AD 6.2 %, ASYMP 4.0 % (11 of 273), CT 30 % (45 of 148: few tau+ neurons, so
the edge artefacts weigh more). On the graded sample it catches 15 of 18 not-a-body and flags 5 of 76 real (and 5 of 35
doubtful). AD vs ASYMAD (LIP-1–7, `workspace/scripts/table_contrasts.R`: log density, burden, age, APOE ε4, region, TMA
and donor), grey matter, flagged in → out: tau+ neuron density standardised effect 1.837 (t 13.4) → 1.833 (t 13.3),
mature tangles 1.825 → 1.820, pretangles 1.624 → 1.598, mature share −0.31 → −0.34, maturity median −0.45 → −0.60 (p 0.24
→ 0.11); whole core alike. The flagged share itself does not differ (0.50, p 0.26). So the main tau results do not rest on
the boundary objects.

The viewer does not show the flag: the outline layer reads the candidate store, and stage 5's flag is not in it. Showing
it means carrying the flag into the store's view and a new legend entry in the web app, so `objects_AT8.csv` is its only
place for now.

### Spatial interactions

New columns of `results_<level>.csv` from stored results, no image read (`stainid.v2.interactions`, after the other
columns; the topic "Spatial interactions" of the Results page). They come from the literature review
`/run/media/sky/Untitled/alzh/reports/Spatial interaction features for AD TMAs.md` (its ranked shortlist; items 6, the
6E10 neuritic halo, and 10, vessels, are stage-2 features and not here) and the pilot
`/run/media/sky/Untitled/alzh/work/feature_explore/PROPOSAL.md`. With `v2.white_matter.analysis: exclude` they are grey
matter only (no `_whole_core` version); a core image left with too little grey matter is not measured. Every random
placement or shift is seeded by the core id, so a re-run gives the same numbers. `interactions_<level>.csv` holds the
columns with the counts behind them (`ix_<count>`, also in `results_counts_<level>.csv`); a donor-region sums its cores'
counts (a correlation: the mean over its cores weighted by their bins or plaques).

**Inputs.** The accepted calls and their shown outlines; stage 2's masks; the threads; the white matter (also on cores
with `analysis: off`, for depth); the 6E10 compactness score stage 5 has just written; the counterstain nuclei: stage 2's
per-nucleus table (`cells/`, its counted nuclei) where it exists, else stage 1's Cellpose nuclei (a nucleus of 10-200
µm² centred in its tile's own area; shape from its pixels' second moments, as `regionprops`). Until the stage-2 re-run
writes `cells/` for every core image, the 6E10 and AT8 nuclei therefore include the dense DAB objects Cellpose outlines
(about 7 %, sitting exactly at plaques and tangles); after it they do not, and the same code reads the new table. The
registered grid needs the registration table or the pair kept in the project's `registration` cache (read only; stage 5
never registers).

**Neuron-like nuclei** on the 6E10 and AT8 sections: a logistic regression on nuclear shape only (log area, eccentricity,
major and minor axis, their ratio), trained at every run on the NeuN section's nuclei (3,000 per core image, seeded),
where a nucleus whose centre lies in an accepted NeuN+ neuron's shown outline is a neuron's. Counts are soft (the sum of
the probabilities). `interactions_nucleus_model.csv` has its leave-one-TMA-out AUC. No hematoxylin feature: on the NeuN
section the brown neuronal nuclei hold less hematoxylin, which would not carry to the other stains.

| Columns | Definition | Minimum (else empty) |
|---|---|---|
| `plaque_<ring>_neuron_share_ratio` (`contact` 0-10 µm, `ring10_30`, `ring30_60`) | 6E10: the neuron-like share of the nuclei in the ring around the nearest plaque of at least 20 µm (each nucleus and tissue cell goes to its nearest plaque edge, on the 2.2 µm mask grid), over the share in tissue at least 100 µm from any plaque; divided by the same ratio averaged over 99 placements of every plaque outline at random cells of the same tissue. Below 1: neurons missing next to plaques. The contact ring is descriptive (nuclei under dense DAB are missed) | 10 plaques ≥ 20 µm, 20 nuclei in the ring, 50 far |
| `plaque_ring_neuron_share_slope` | 30-60 µm ratio minus 10-30 µm ratio | as above |
| `plaque_ring10_30_neuronlike_density_ratio`, `_other_density_ratio` | the neuron-like and the other nuclei's densities (per mm² of ring tissue) over far tissue, against the random placements | as above |
| `plaque_compact_` / `plaque_diffuse_ring10_30_neuron_share_ratio` | the 10-30 µm ratio for the nuclei whose nearest plaque is compact / diffuse | 10 plaques of the type |
| `tangle_ring15_60_neuron_share_ratio`, `tau_pretangle_ring15_60_neuron_share_ratio`, `tangle_ring15_60_other_density_ratio` | AT8: the same 15-60 µm from mature tangles / pretangles, far = 100 µm from every accepted tau+ neuron | 10 bodies of the type, 20 ring nuclei, 50 far |
| `tau_mature_no_nucleus_fraction`, `tau_pretangle_no_nucleus_fraction` | share of outlined mature tangles / pretangles with no nucleus centre inside (a "nucleus" covering over half the body is the body itself) | 10 |
| `neun_no_nucleus_fraction` | share of NeuN+ neurons with no nucleus centre inside: how often the section misses an intact neuron's nucleus | 20 |
| `tangle_ghost_excess` | mature tangles without a nucleus minus NeuN+ neurons without one | both |
| `tau_satellites_per_neuron`, `tau_satellite_enrichment` | nuclei just outside a tau+ neuron (centre within its radius + 2 µm of the outline), per neuron and over what the nucleus density puts in that band | 10 outlined |
| `neun_microcolumn_strength`, `amyloid_section_`, `tau_section_microcolumn_strength` | Buldyrev 2000's band: neuron pairs within 10 µm of the radial axis and 30-150 µm along it over the same turned by 90°, divided by that ratio for 19 random placements of as many points in the same tissue (with its axes). Radial axis: the gradient of the depth from the white matter (smoothed over 50 µm), within 2 mm of it. NeuN: accepted neurons; 6E10 / AT8: neuron-like nuclei (probability ≥ 0.5), two replicates within the core | white matter; 200 points, 100 pairs |
| `amyloid_section_neuronlike_fraction`, `tau_section_neuronlike_fraction` | mean neuron-like probability of the section's nuclei | 100 nuclei |
| `tangle_pretangle_mature_mix_50um`, `_100um` | pretangle-mature pairs within the radius over their exact expectation under random labelling of maturity among the same tau+ neurons | 10 of each, 20 pairs |
| `tau_neuron_column_alignment` | the share of tau+ neuron pairs within 300 µm lying within 30° of the radial axis, over that share for 19 random placements | white matter; 10 neurons, 20 pairs |
| `<object>_depth_excess_um` (`tau_neuron`, `tangle`, `tau_pretangle`, `thread`, `plaque`, `plaque_compact`, `plaque_diffuse`, `neun`, `neun_large_neuron`) | mean distance from the white matter of the objects (thread: length-weighted) minus that of the section's tissue, within 2 mm of the white matter | white matter; 10 objects (0.5 mm of thread) |
| `tau_neuron_deep_upper_ratio`, `neun_deep_upper_ratio` | density within 1 mm of the white matter over density 1-2 mm from it | white matter; 10 / 20 objects, 0.25 mm² in each band |
| `neun_large_neuron_peak_depth_um` | centre of the 250 µm depth band densest in NeuN+ neurons of at least 150 µm² (bands of 0.05 mm² or more) | white matter; 20 large neurons |
| `tangle_plaque_depth_difference_um` | mature tangles' depth excess minus compact plaques' (each on its own section) | both |
| `plaque_thread_coupling_rho` | Spearman rho of plaque area and thread length over the 100 µm squares of the NeuN frame that are at least 80 % grey matter on all three sections | registration; 30 squares |
| `plaque_thread_coupling_excess`, `plaque_tangle_coupling_excess` | rho (thread; tau+ neuron density) minus its mean over 99 shifts of the AT8 map by 3-10 squares (overlapping squares only) | as above |
| `plaque_thread_coupling_depth_adjusted` | the rank correlation within 250 µm depth bands (NeuN white matter) | white matter; 30 squares with a depth |
| `plaque_thread_correlation_length_um` | lag (100 µm steps) at which rho of each square's plaque area with the mean thread of the squares k steps away falls to half rho; only where rho ≥ 0.1 and it falls within 800 µm | as above |
| `plaque_compact_thread_coupling`, `plaque_diffuse_thread_coupling` | rho with compact / diffuse plaque area | as above |
| `niche<100,200>_<niche>_fraction` (`amyloid_tau`, `amyloid_only`, `tau_only`, `quiet`, `amyloid_tau_neuron_poor`) | the same joint squares (100 and 200 µm), each typed by its percentile ranks within its TMA: plaque area, tau (mean rank of thread and tau+ neuron density) and NeuN+ neuron density; high = top third, neuron-poor = bottom third | registration; 20 squares |
| `niche<size>_entropy` | evenness over the eight types (0-1) | as above |
| `niche<size>_damage_quiet_contact_ratio`, `_amyloid_tau_cohesion` | side-sharing pairs of amyloid + tau with quiet squares, and of amyloid + tau squares with each other, over their expectation under random arrangement of the core's squares | expectation ≥ 2 |
| `neun_other_to_neuron_clq`, `neun_neuron_to_other_clq` | co-location quotient on the NeuN section's nuclei (neuron: centre in an accepted neuron's outline): the share of one kind whose nearest nucleus is of the other, over the other's share | 20 of each |
| `neun_satellite_fraction`, `neun_large_satellite_fraction`, `neun_satellite_enrichment` | NeuN+ neurons (all; ≥ 150 µm²) with an other nucleus just outside, and those nuclei over their expectation | 20 / 10 neurons |
| `neun_gap_area_fraction` | share of the tissue whose empty circle among neurons lies inside the tissue and is wider than 95 % of the control cores' (computed at every run) | 20 neurons, 0.5 mm² |
| `neun_neuron_share_cv_100um` | coefficient of variation of the neurons' share of nuclei over 100 µm windows (80 % tissue, 10 nuclei) | 10 windows |
| `plaque_size_mark_correlation_50um`, `_100um` | Stoyan's mark correlation of plaque diameter: the mean product of the diameters of pairs within the radius over the squared mean | 20 plaques, 20 pairs |
| `plaque_compact_in_diffuse_fraction`, `_enrichment` | compact plaques centred inside or within 20 µm of a diffuse plaque's outline; over the share of tissue there | 10 compact, 5 diffuse |
| `plaque_compactness_diffuse_neighbourhood_rho` | Spearman rho of each plaque's compactness score (≥ 15 µm) with the diffuse plaque area within 100 µm of it | 20 plaques |

Not built: the lateral periodicity and width of the microcolumns (noisy at these counts), a k-means niche model (the
niches are typed by rule so their names stay the same from run to run), and per-plaque neuron shares by the compactness
score (too few nuclei per plaque; the compact / diffuse split carries it).

**On the study's current outputs** (stage 5 re-run in scratch, 2026-11-03, with the LIP-8-13 registrations filled in
there; grey matter only; 13 TMAs, 365 core images). The neuron-like model's leave-one-TMA-out AUC is 0.88 (0.81-0.91).
26 of the 73 columns reach a replicate-core ICC of 0.5; the full list is in `measure_reliability.csv`.
- Reliable and distinct from the main densities: `tangle_ghost_excess` (ICC 0.70), `tau_mature_no_nucleus_fraction`
  (0.67), `neun_no_nucleus_fraction` (0.63), `neun_satellite_enrichment` (0.70), `neun_gap_area_fraction` (0.70; AD
  above ASYMAD by 1.0 SD, p 0.001), `neun_neuron_share_cv_100um` (0.69; AD above ASYMAD by 1.1 SD, p 4e-4),
  `plaque_size_mark_correlation_50um` (0.60; AD below ASYMAD by 0.8 SD, p 0.003; higher in occipital),
  `plaque_compact_in_diffuse_fraction` (0.56) and the 6E10 / AT8 neuron-like fractions (0.70, 0.60).
- The niche fractions are the most reliable (ICC 0.61-0.86) and separate the groups most strongly (amyloid + tau
  squares: AD above ASYMAD by 1.2-1.3 SD, p < 1e-9; amyloid-only squares lower in AD, -1.7 SD), but they are largely
  the densities seen together (unique variance 0.09-0.41).
- Below the gate on these outputs: the ring neuron shares around plaques and tangles (ICC ≤ 0.4; the median 10-30 µm
  share is 0.96-0.97 of the random placements), the microcolumn strengths (replicates within a core agree poorly:
  Spearman 0.17-0.30), most depth profiles (few cores with enough white matter: 7-36 pairs) and the plaque-thread
  coupling (ICC 0.40, though it is clearly higher in AD than ASYMAD, 1.2 SD, p 4e-5). They stay in the tables.

### Reliability of every measure: `measure_reliability.csv`

Stage 5 also writes, beside the results tables, one row per numeric column of `results_core.csv`
(`stainid.v2.reliability`): `icc`, the replicate-core intraclass correlation (one-way random effects ICC(1) over the
donor-regions with two or more cores, all TMAs; densities and counts as log10(x + 1); at least 10 donor-regions),
`n_pairs`, `unique_variance` (the share of the column's rank variance across donor-regions not explained by the ranks of
plaque density, plaque area fraction, AT8 area fraction, tau+ neuron density, thread density and NeuN+ neuron density),
`regions` and `reliable` (ICC ≥ 0.5, the study's gate for confirmatory use). Every measure keeps its column whatever its
ICC: the Results page can filter on the table.

