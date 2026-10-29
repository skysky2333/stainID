# Grey and white matter: the algorithm, its checks, and what the analysis could do with it

The cohort's cores are 4.5 mm across and the cortex is 2-3 mm thick, so many cores reach into the white matter. The
algorithm (`stainid.v2.white_matter`, `stainid v2 white-matter`) finds it on every core from detection's stored
outputs; people correct it in the *Grey / white matter* round, and their corrections only train and score it. The
analysis counts white matter as before until the project's setting says otherwise (below).

**Cells and neurons.** Cells are the hematoxylin nuclei: every section has them, every stain. Neurons are only the
NeuN+ cells on the NeuN section (stage 4's accepted NeuN calls). The algorithm is carried by the neurons and checked by
the cells.

## The algorithm

Per NeuN core image, on a grid of 64 core px (17.5 µm):

1. NeuN+ neurons per mm² of analysed tissue (stage 2's mask: tissue in a tile), smoothed over 100 µm and divided by the
   smoothed tissue, so the core's edge and its holes do not lower it.
2. The core's own grey-matter density, the *reference*: the 90th percentile over its tissue (typically 250-900 /mm²).
   Below 80 /mm² the core's NeuN cannot be read and the core is **undetermined**: its NeuN stain is too pale or failed
   (43 of 361 NeuN cores; see "NeuN-pale cores" below).
3. Seeds: tissue below 10 % of the reference, opened with a disc of 150 µm (a thin sparse band is no seed); grown
   through tissue below 25 % of the reference; parts under 0.25 mm² dropped; a part must reach 450 µm from the
   tissue's edge (glass, tears, holes), so a band along an edge — layer I under the pia, leptomeninges, a sulcus, a
   frayed core edge — is no white matter.
4. Each part is checked on the NeuN image itself (read at 1/4, 1.1 µm/px): kept when it has at least as many cells per
   mm² as the core's grey matter (tissue at ≥ 50 % of the reference), or at most 0.65 x its NeuN stain (mean DAB).
   White matter has more cells (oligodendrocytes in rows: 1.4 x the grey matter's, median of the confirmed cores) and
   no NeuN-stained neuropil; grey matter that is only poor in NeuN+ neurons (a pale or uneven NeuN stain, deep layer VI)
   has neither.
5. Holes under 0.1 mm² filled; the area on the masks' grid (1/8 of the core image), its edge smoothed, inside the
   analysed tissue. It is carried to the core's 6E10 and AT8 images through the registration (`registration.cores`:
   the registration table, else the pair registered from the whole-core pictures and kept in the same cache the app
   carries people's areas with), inside each image's own analysed tissue. A failed registration makes that stain image
   undetermined.

All settings are under `v2: white_matter:` in stainid.yaml (`stainid.v2.white_matter.DEFAULTS`); a change redoes every
core at the next run.

**Which cue carries it.** The neuron density. Cells alone separate a core's white from its grey matter with a median
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

### NeuN-pale cores

43 NeuN cores (12 %) have almost no NeuN+ neurons anywhere (the densest part under 80 /mm², typically 0-40) while their
6E10 and AT8 sections show grey-matter pathology at the usual densities (AD cores: 20-90 plaques and 10-40 tau+ neurons
per mm²) and their hematoxylin shows unstained neuronal cell bodies. Their NeuN stain failed or is very pale, often in
neighbouring positions at a slide's edge (LIP-7 D-1, D-2, E-1, E-2; LIP-9 D-5, E-5; LIP-11 E-5, F-5; LIP-3 A-2, A-3,
F-2, F-4). The analysis counts them today with NeuN densities of 0-30 /mm²: they are a question for the NeuN analysis
whatever is decided about white matter (`white_matter.csv`, status undetermined, lists them).

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

### Expected effect (2026-10-29, counts and densities only)

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
