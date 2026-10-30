# Grey and white matter: the algorithm, its checks, and what the analysis could do with it

The cohort's cores are 4.5 mm across and the cortex is 2-3 mm thick, so many cores reach into the white matter. The
algorithm (`stainid.v2.white_matter`, `stainid v2 white-matter`) finds it on every core from detection's stored
outputs; people correct it in the *Grey / white matter* round, and their corrections only train and score it. The
analysis counts white matter as before until the project's setting says otherwise (below).

**Cells and neurons.** Cells are the hematoxylin nuclei: every section has them, every stain. Neurons are only the
NeuN+ cells on the NeuN section (stage 4's accepted NeuN calls). The algorithm is carried by the neurons and checked by
the cells.

## The algorithm

Per NeuN core image, on a grid of 64 core px (17.5 µm), one rule set for every core (version 2, 2026-10-29):

1. NeuN+ neurons per mm² of analysed tissue (stage 2's mask: tissue in a tile), smoothed over 100 µm and divided by the
   smoothed tissue, so the core's edge and its holes do not lower it.
2. The core's own grey-matter density, the *reference*: the 90th percentile over its tissue (typically 250-900 /mm²).
3. Candidates, two ways that are merged:
   - *far below the reference*: seeds under 10 % of it, opened with a disc of 150 µm (a thin sparse band is no seed),
     grown through tissue under 25 %;
   - *the sharp drop*: tissue under 20 % of the densest tissue within 400 µm (a drop to a fifth across 400 µm is a step,
     a fade is not), where that grey side has at least 25 % of the reference and 80 /mm²; each such seed's region
     reaches the steepest part of the drop (a watershed of the density's gradient between these seeds and the grey
     matter, tissue at ≥ 40 % of the reference). The drop is measured against the grey matter beside it, so very dense
     grey matter elsewhere in the core no longer hides white matter (LIP-1_E-4).
   A core whose densest tissue is under 80 /mm² (its NeuN stain pale or failed, or a core of white matter only) has no
   grey matter to drop from and is a candidate as a whole.
4. Each part (≥ 0.25 mm²) is kept when
   - it is at least 180 µm thick and its density is at most 0.7 x the density just outside it (a real step);
   - it reaches 450 µm from the tissue's outer edge (holes and tears inside the core are not edges any more), or its
     step is clear (≤ 0.35) and its cells are white matter's: white matter at a core's edge is kept, layer I and the
     pia mostly not;
   - its cells are white matter's, against the core's grey matter (tissue at ≥ 50 % of the reference): few
     neuron-sized nuclei (stage 1's Cellpose nuclei of the NeuN section ≥ 45 µm², at most 0.3 x the grey matter's
     share), or at least as many cells per mm² (hematoxylin nuclei of the NeuN section and of the 6E10 and AT8 sections
     carried onto it through the registration, averaged), or at most 0.65 x the NeuN stain (mean DAB). When the drop in
     neurons comes with a drop in the background NeuN stain (the part's median DAB under 0.65 x the ring around it: a
     staining gradient across the core, LIP-5_E-1) only the cells count: more cells, or few neuron-sized nuclei;
   - a core with no grey matter of its own (point 3) is compared with the cohort's grey matter instead: at most 0.3 x
     its share of neuron-sized nuclei (0.28; white matter has 0.03-0.07, cores whose neurons are only unstained
     0.10-0.32), and no grey-matter pathology on its other sections (≤ 8 AT8+ neurons and ≤ 50 plaques per mm²: an AD
     core whose NeuN failed is grey matter). A pale core that is not white stays **undetermined** (flagged, see below).
5. Holes under 0.1 mm² filled; the area on the masks' grid (1/8 of the core image), its edge smoothed, inside the
   analysed tissue. It is carried to the core's 6E10 and AT8 images through the registration (`registration.cores`:
   the registration table, else the pair registered from the whole-core pictures and kept in the same cache the app
   carries people's areas with), inside each image's own analysed tissue. A failed registration makes that stain image
   undetermined.

All settings are under `v2: white_matter:` in stainid.yaml (`stainid.v2.white_matter.DEFAULTS`); a change redoes every
core at the next run.

**Which cue carries it.** The neuron density and the sharp drop in it; cells and their sizes confirm. Cells alone separate a core's white from its grey matter with a median
within-core AUC of 0.91 on the NeuN section, 0.95 on 6E10 and 0.90 on AT8 (the white matter carried by the registration),
but below 0.65 on a quarter of the cores; the NeuN stain (DAB) alone 0.96. So cells confirm the parts the neurons propose
and veto grey matter that is merely sparse in NeuN+ neurons.

## Outputs

Under `<outputs.v2>/white_matter/` (default `data/analysis/v2/white_matter/`), never in detection's own files:

| File | What |
|---|---|
| `<stain>/<core_id>.png` | the area on the masks' grid (1/8 of the core image): 1 white matter, 0 not |
| `manifests/<core_id>.json` | what the core's areas were made from (the NeuN candidate store's digest, the three stage-2 masks, the registration rows, the settings, the code version) and its rows; a core is redone only when one of them changes |
| `white_matter.csv` | per core image: `status` white / grey / undetermined, `reason`, `white_share` of the analysed tissue, `white_mm2`, `reference_per_mm2`, `neurons`, `registration` (how and how well it was carried) |

`stainid v2 white-matter` runs on detection's existing outputs (stage 4 for NeuN, stage 2 for all three stains) and
re-runs nothing of detection; `--core` limits it, `--redo` redoes every core. About 5-10 s per core with white matter
to check (one reduced read of its NeuN image), a fraction of a second for the others. Once its folder exists the app
also makes a missing or outdated core's areas in idle time, when a *Grey / white matter* core is prepared
(`api.ahead._core_pictures`).

## Where it shows

- **Cores view and every annotation page** (Mark bad areas, Fix the field, the object rounds): the "Excluded from
  analysis" layer draws it over the codes as a violet cross-hatch with an edge, without a wash, with its share in the
  legend: *white matter (found by the algorithm; still counted in the analysis)*. It is not one of the layer's codes:
  what the views draw as counted is still exactly what the measures divide by (`tests/test_excluded_view_matches_measures.py`).
- **Grey / white matter round**: on NeuN the brush starts from the algorithm's area (`from` "proposed", with
  `proposal`: status, share, version); people paint what it missed and erase what is grey matter, or *Accept as found*.
  A core it found none on, or could not read, starts empty and says so. Saving stores people's area as before (one
  store: the label log, `labels.matter`), with `meta.proposal` naming the algorithm's version they started from;
  *All grey matter* stays the person's own say. The other stains are carried from people's NeuN area as before.
- **Scores**: `GET /api/labels/matter` (admin) adds `algorithm`: per stain the images both people and the algorithm
  have, the mean IoU and how many each calls white, and the rows (IoU, Dice, each share; people's area is the truth).

## Validation (2026-10-29)

People's labels so far: one session, `grey_white_matter_oct_29` (troncoso), 4 NeuN cores, probably a trial: LIP-1_A-5
and LIP-9_F-3 none (the algorithm: none), LIP-1_B-2 none (the algorithm: 24 % — sparse deep cortex with many NeuN+
neurons; a false positive), LIP-1_E-4 a 0.13 mm² patch (the algorithm: 2.1 mm², IoU 0.03; its upper part is white
matter on a visual check, the lower part a torn edge). The earlier 329-field review (`white_matter_field_review`,
AI-assisted, sampled from low-density fields) mostly labels NeuN-pale grey matter as white matter and is not used.

A visual check (of the prototype, which flagged 93 of the same cores; the step adds LIP-11_F-3 and LIP-3_B-1, 4-10 %) of every core the first version flagged and of 36 random grey cores (whole-core picture plus a 550 µm
crop deep in the flagged area, montages under `/run/media/sky/wd6t/wm_detect/`):

| | cores |
|---|---|
| NeuN cores | 361 |
| white matter found | 95 (median 28 % of the tissue) |
| grey | 223 |
| undetermined (NeuN too pale to read) | 43 |
| checked white matter cores found / missed | 59 / 5 (sensitivity 92 %) |
| flagged cores checked as not white matter / white matter (precision among the checked) | 11 / 59 (84 %); 13 more were unclear |
| random grey cores with white matter seen | 0 of 36 |

White matter is on every TMA (most on LIP-5: 15 of 29 cores, LIP-9: 11), more often in occipital (57) than frontal (36)
cores. The false positives are: sparse zones of deep cortex in very dense cores (LIP-13_F-3, LIP-13_A-4, LIP-1_B-2,
LIP-5_F-2), edge bands deeper than 450 µm (LIP-7_E-4, LIP-8_E-1, LIP-4_A-4, LIP-7_C-5), a sulcus (LIP-6_A-4), the pia
(LIP-11_E-3) and pale-NeuN grey matter (LIP-5_E-5). Missed: white matter in pale-NeuN cores with as few cells as the
grey matter (LIP-6_F-5, LIP-10_F-4, LIP-6_B-4) and narrow white matter at a frayed edge (LIP-3_C-1, LIP-5_B-5). The
boundary itself follows the drop in neuron density within about one smoothing width (100 µm); it has not been scored
against people's outlines (none yet beyond the trial).

### Version 2 (2026-10-29): area-level recall

People found misses quickly (LIP-1_E-4's lower right, LIP-4_B-2, LIP-3_E-2) and one over-claim (LIP-5_E-1), so the
first version's core-level sensitivity (92 %) hid a recall problem in area. Version 2 keeps version 1 as its base and
adds to the same rules: the sharp drop as a second way to a candidate, depth from the outer edge only, the cells of the
other sections and the neuron-sized nuclei, the background-stain check and the cohort comparison for cores with no grey
matter of their own. Scored against white matter outlined by eye (by the assistant, on whole-core NeuN pictures with a
grid; 21 cores: 14 with white matter, 2 whole-core white, 5 grey, LIP-5_E-1 none; `labels_by_eye.json`):

| | pixel recall | pixel precision |
|---|---|---|
| version 1 | 0.43 | 0.81 |
| version 2 | 0.83 | 0.89 |

Over all 361 NeuN cores: the 95 cores version 1 called white keep at least 80 % of their area except LIP-5_E-1 (pale
stain, removed on purpose) and LIP-3_B-1 (4 %, a thin part); 61 grey cores and 7 pale cores become white (161 white,
164 grey, 36 undetermined). A visual pass over the new ones finds most plausible (neuron-poor zones with glial nuclei:
LIP-7_C-2, LIP-6_E-2, LIP-6_C-4, LIP-8_C-2, ...) and some false positives of three kinds: leptomeninges and frayed
tissue (LIP-2_C-1, LIP-10_A-5, LIP-1_A-3), bands along the pia or a sulcus (LIP-5_F-1, LIP-5_F-4, LIP-12_A-4,
LIP-6_F-4) and patchy neuron loss in AD cores with a low reference (LIP-12_D-4, LIP-10_F-3). LIP-3_E-2 stays grey: its
upper right looks like white matter but its centre has dark pyramidal perikarya and the candidate parts keep 0.7 x the
grey matter's neuron-sized nuclei. Per-core tables and montages: `/run/media/sky/Untitled/alzh/work/wm_detect/`.

### NeuN-pale cores

43 NeuN cores (12 %) have almost no NeuN+ neurons anywhere (the densest part under 80 /mm², typically 0-40) while their
6E10 and AT8 sections show grey-matter pathology at the usual densities (AD cores: 20-90 plaques and 10-40 tau+ neurons
per mm²) and their hematoxylin shows unstained neuronal cell bodies. Their NeuN stain failed or is very pale, often in
neighbouring positions at a slide's edge (LIP-7 D-1, D-2, E-1, E-2; LIP-9 D-5, E-5; LIP-11 E-5, F-5; LIP-3 A-2, A-3,
F-2, F-4). The analysis counts them today with NeuN densities of 0-30 /mm²: they are a question for the NeuN analysis
whatever is decided about white matter (`white_matter.csv`, status undetermined, lists them).

The 12 control (CT) pale cores have no pathology to settle it, so their cells were compared with known grey and white
matter (the GM and WM parts of 16 CT white-matter cores, 16 grey CT cores): stage-1 Cellpose nuclei on all three
sections, per mm², their median area and the share of large, neuron-sized nuclei (≥ 45 µm²; neurons keep their large
round nucleus when the NeuN stain fails). The large-nucleus share separates the reference GM and WM parts best (AUC
0.93 on the NeuN section, 0.72 AT8, 0.65 6E10). On the pale cores the NeuN section reads low everywhere, also on AD pale
cores that are grey matter by their plaques and tangles, so the 6E10 and AT8 sections decide. Eight CT pale cores look
like grey matter (LIP-2_A-4, LIP-2_A-5, LIP-7_D-1, D-2, E-1, E-2, LIP-11_D-2, LIP-11_B-2); LIP-8_D-4 and LIP-8_E-4 (both
cores of donor 2152, occipital) look like pure white matter (small dark round nuclei in rows along parallel fibres, no
neuron-sized nuclei on any section); LIP-11_C-2 (vacuolated, small nuclei only) probably white matter; LIP-6_E-5
unclear. `scripts`: `/run/media/sky/wd6t/wm_detect/scripts/pale_check.py`, montages `pale_ct_*.jpg`.

Version 2 decides these cores with the same rules (point 4, the cohort comparison): 7 of the 43 become white matter as a
whole (LIP-8_D-4, LIP-8_E-4, LIP-11_C-2 (its upper fragment), LIP-6_E-5, LIP-6_D-4, LIP-8_E-5, LIP-4_F-3), the other 36
stay undetermined and flagged.

## Leaving it out of the analysis: `v2.white_matter.analysis: exclude`

The setting is `off` by default (white matter counted as before); `exclude` makes stage 5 grey matter only. The mask
needs NeuN's detection first (it is made from NeuN's calls), so it is applied at the analysis stage, never in detection:

1. `stainid v2 white-matter`: the areas (above).
2. `stainid v2 grey-matter` (`v2.grey_matter`): the 6E10 and AT8 stained areas without the white matter. Stage 2 keeps
   each tile's stained areas as totals, so this pass reads each 6E10 / AT8 core image with white matter once and works
   out what stage 2 works out for every tile the white matter reaches (`proposals.tile_summary` with those cells
   counted out: the same crops, slide thresholds, nuclei, linear-artifact and fold masks and levels; no proposals, no
   scores, no outlines). Without leaving anything out it reproduces stage 2's tile values exactly (tested on the
   study's LIP-7_B-5 and in `tests/test_white_matter_analysis.py`). Outputs `white_matter/grey_tiles/<stain>/<core>.csv`
   and manifests; about 2-3 min per core image, `--workers` core images at once (about 2.5 GB each); CPU only.
3. `stainid v2 measures` with `exclude`: objects whose centre lies in the white matter are dropped (every stain; AT8
   thread pieces by their middle), its counted cells leave each tile's area (so every density and burden divides by grey
   matter only; the NeuN density windows lose them too), and the 6E10 / AT8 burdens and nuclei are the pass's. Stage 5
   stops when a core image's pass is missing or older than its white matter or its stage-2 outputs. The main columns
   are grey matter only; every whole-core number is kept next to it as `<column>_whole_core` in `results_core`,
   `results_donor_region` and the three `cohort_morphology_*` R tables, and the whole-core tables as before are written
   under `measures/whole_core/`.
4. The views shade the white matter as excluded (code 7, *white matter (found by the algorithm; left out of the
   analysis)*), so what they draw as counted is still what the measures divide by.

Flags, whenever the white-matter step has run and whatever the setting: `white_matter_status`, `white_matter_share`
(per core: of its NeuN image's analysed tissue; per donor-region: of its NeuN cores' tissue together) and `neun_pale`
(the core's NeuN stain too pale to read; per donor-region: any of its cores, with `neun_pale_cores`). Pale cores are
flagged, never left out: they are grey matter whose NeuN failed.

### Effect on the study (2026-10-29, scratch run of stage 5 and the primary R model)

Stage 5 with `exclude` from the live outputs into `/run/media/sky/wd6t/wm_detect/analysis/` (`measures/` grey matter,
`measures/whole_core/`, `group_means.csv`, `group_ratios.csv`, `r_*.csv`): 284 core images lose white matter. Mean
donor-region NeuN density rises 4-12 % in AD, 5-12 % in ASYMP and 7-22 % in CT (most in CT occipital, the region with
the most white matter), so the AD/CT neuron ratio falls from 0.84 to 0.82 (frontal) and 0.71 to 0.65 (occipital).
Plaque and tau densities and burdens rise 2-6 % in AD. The ASYMP-vs-AD contrasts keep their size and significance
(frontal log NeuN density, burden-adjusted: -0.73, p 0.022 whole core; -0.75, p 0.019 grey matter). Replicate-core ICC
of log NeuN density 0.81 to 0.82, of the NeuN+ share of candidates 0.73 to 0.81. The positive association of NeuN
density with local amyloid in the two-region model shrinks (0.29, t 2.2, to 0.21, t 1.7): part of it was white matter
(fewer neurons and fewer plaques in the same cores).

### Expected effect (estimated before the pass, counts and densities only)

From the stored outputs (stage-4 accepted objects, stage-2 counted area, the step's areas; objects by their centre;
`/run/media/sky/wd6t/wm_detect/scripts/effects.py`, table `step_full/effects.csv`). Cores without white matter do not
change. In the 95 cores with white matter (median share 28 %, quartiles 22-44 %):

| per core with white matter, median | NeuN neurons /mm² | 6E10 plaques /mm² | AT8 objects /mm² |
|---|---|---|---|
| whole core (today) | 218 | 15.7 | 0.34 |
| grey matter only | 290 | 20.1 | 0.41 |
| inside the white matter | 35 | 3.4 | 0 |
| change of the core's density | +34 % (quartiles +23 to +64 %) | +24 % (+8 to +43 %) | +27 % (+10 to +46 %) |

Per donor-region (cores pooled), 81 of 182 donor-regions have a core with white matter; their densities rise by a median
12 % (AD), 31 % (ASYMP), 28 % (CT) for NeuN, 10-17 % for 6E10, 11-37 % for AT8. The white-matter share is not even
across groups: 6 % (frontal) and 12 % (occipital) of AD tissue, 9 % and 16 % of CT tissue. So the CT cores' neurons are
diluted more today: the AD / CT ratio of mean NeuN density is 0.78 whole-core and 0.74 in grey matter only. Between
replicate cores of one donor-region the NeuN density's coefficient of variation falls from 0.21 to 0.16 (6E10 0.47 to
0.40, AT8 unchanged 0.49): leaving the white matter out removes a source of variance between replicates and slightly
widens the group difference in neurons.
