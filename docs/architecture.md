# One store per object

Every object people or detection make has one owner module and one store; every view reads it through that owner,
so an edit shows everywhere at once. Human annotations — object labels and people's areas alike — only train and
evaluate models: the analysis reads detection's outputs only.

| Object | Owner module | Store file | Who reads it | How edits invalidate | Used in analysis? |
| --- | --- | --- | --- | --- | --- |
| Candidates (detection's objects: position, outline, call, decision) | `stainid.candidates`, the one reader and writer (`read_core`, `read_bbox`, `stored`, `cores`, `write_table`); `labels.candidate_store` only parses its rows for the label pages | `<outputs.v2>/candidates/<stain>/<core>.parquet` | stages 4–5, Cores view, both rounds, scoring, training | a stage's manifest digest of its inputs: a changed input remakes that core | yes |
| Outlines and the default mask | `stainid.candidates.outline_options`, the one choice (`outline_rings` draws it for the Cores view, `candidate_store.masks` lists it for the label pages; `outline_default` setting) | the candidate store (`outline`, `masks`) | stage 5 measures, Cores view, rounds | as candidates | yes |
| Calls and classes (detection's) | `stainid.v2` stage 4 (`classify`) | the candidate store (`class`, `subtype`, `decision`) | stage 5, Cores view, rounds, `labels.compare` | stage 4 manifest | yes |
| Excluded areas, automatic | `stainid.exclusions.automatic` | `<outputs.v2>/masks/<stain>/<core>.png` (stage 2; bits `v2.proposals.BITS`) | stages 2–5 (valid area, objects' excluded share), Cores view and rounds as "Excluded from analysis" (`exclusions.view`), `final_say` | stage 2 manifest; its digest (`exclusions.digest`) is in the later stages' manifests | yes |
| People's areas (added to and taken out of each type) | `stainid.exclusions.annotated` (format: `labels.exclusions`) | `exclude` labels in `<outputs.reviews>/labels.jsonl`, an ordered edit log (who, when, per type added "1"–"4" or taken out "-1"–"-4" — "marked fine" is a part taken out of a type; older edits may clear every type, "include"; latest wins), resolved per type into one merged region set; "AI reviewer" for the earlier hand-drawn polygons | Annotations layers (Cores view, rounds, Mark bad areas), `final_say` → `drop_excluded` (training, scoring, model comparison, object rounds' fading), Models → Bad-area detector | none: no stage reads them | no — train and evaluate only |
| Object labels (class, type, outline, added objects) | `stainid.labels.store` (classes and keys: `labels.vocabulary`) | `<outputs.reviews>/labels.jsonl` | `labels.decisions` (scoring, rounds), `labels.overview`, `labels.compare`, training | caches keyed on the log's signature | no — train and evaluate only |
| What a label is about, and the count | `stainid.labels.store`: `object_keys` (the candidate id; the field and position only where no candidate is named or matched), `latest_say` (each person's latest), `tally` (majority, "" when split) | — (worked out from the labels; old labels matched onto candidates by `candidate_store.match`) | `decisions.state` (rounds' calls, scoring), `decisions.consensus` (training's truth, Labels page), `store.consensus` | as object labels and the candidate store | no |
| A field's acceptance threshold (step 1) | `stainid.labels.starting` (the page: `webapp/src/labels/fieldStart.ts`) | a `threshold` label per person and field in `labels.jsonl` (the locked value; where untouched objects started on *Save and continue*) | the field tool (opens locked at it), training and threshold tuning | as object labels | no — train and evaluate only |
| Splits and merges | `stainid.labels.objectedit` | `replaced` labels (the originals) and class + outline labels with `meta.edit` (the pieces) in `labels.jsonl` | `labels.decisions` / training (pieces are their own objects, never matched onto a candidate), Cores view Annotations layer (`compare.object_edits`) | as object labels | no — train and evaluate only |
| The bad-area brush | `webapp/src/labels/brush.ts` (`BrushMasks`; the field tool's `useBrush` wraps it) | the strokes since Save, sent as one `exclude` edit per picture | the field tool (key B) and "Mark bad areas": the same areas (`store_view.current_areas`), the same keys | — | no |
| Detection's call on a labelled object | `stainid.labels.candidate_store.calls_of` (the v2 store) | the v2 candidate store | the Labels page (`labels.objects`, `labels.overview`: agreement, precision, recall), `labels.compare` (the Cores view's Annotations layer), the rounds — the one source wherever the store has the core | — | — |
| The older candidate pool (`labels.pool`) | `stainid.labels.pool` | `<outputs.reviews>/pool/<stain>/<tile>.csv` | only for cores the v2 store does not have (a project from before the v2 detection: its older sessions' pictures and features, `labels.added`, and as detection's call there); never read for a core the store has | none | no |
| Outline fixes (how detection's outlines match people's) | `stainid.labels.outline_fixes` (from `labels.compare.layer`, the Annotations layer's reader) | worked out from the label log and the candidate store (cached), nothing stored of its own | the Models page (*Outline fixes*, `/api/models/outline-fixes`, CSV for training) | — | — |
| Colours | `src/stainid/palette.json` (`stainid.palette`; the web app's `palette.ts` imports the same file; served at `/api/palette`) | the file itself | every page (classes by colour slot, states, area types, layers, markers) and the server's pictures (the excluded-areas layer, the raster layers) | — | — |
| Claims and session state | `stainid.labels.sessions`, `api.work` (holds) | `<outputs.reviews>/sessions/<name>.json`; holds in memory | the round pages, Labels page | — | no |
| Review status (done per field, areas checked per core × stain) | `stainid.labels.status` | `done` labels in `labels.jsonl` | Labels page, Cores view minimap, rounds' field draws | as object labels | no |
| People against detection | `stainid.labels.compare` | — (worked out from the labels and the candidate store) | Cores view Annotations layer, Labels overview, scoring | as both | no |

`exclusions.final_say` (people's areas over detection's) exists only as the label set for training and evaluating:
labels on objects there are left out (`drop_excluded`), and it is the target the bad-area detector is scored against.
It is never an analysis input, so painting or clearing an area changes no stage input and no digest.

Labels are counted one way everywhere: by candidate id, a label without one matched onto the candidate at its position
(within `candidates.MATCH_RADIUS_UM`), and only labels no candidate is found for (objects people added) by field and
position. A candidate seen from two fields, or clicked a few pixels apart, is one object; each person's latest label
about it counts once. The rounds' "checked by" call is the same majority as training's truth (the latest said when
people are split).

## What was removed

- **"Check detection's calls" (round 1)**, one candidate at a time. No new sessions can be made (the API answers 400),
  and an old one answers 410 to its annotators. Old sessions stay listed with their progress as history
  (`labels.rounds.items` reads their `items.json`), and their labels stay valid object labels for training and evaluation.
  Removed with it:
  - the candidate ranking (`labels.rounds.decision_threshold`, `rank`, `choose`, `save_items`, `with_shapes`, `open_items`);
  - the round-1 routes (`next_tile_objects`, `my_answers`, `_item`, `_unanswered`, the round-1 branches of `save_tile` and `warm`);
  - its practice session (`practice-1-*`) and its picture queue (`queue_pictures.round_wanted`);
  - the page (`webapp/src/labels/checkcalls.tsx`), its tutorial steps, its screenshot and the demo's round-1 session.

  The two round types left are **Fix the field** (objects) and **Mark bad areas** (brush only).
- **The second store reader and outline choice**: `labels.candidate_store` read the files itself (`_read_file`, its own
  parquet filter and calls reader) and chose outlines itself (`masks`); both now go through `stainid.candidates`.
  `candidates.match` / `match_many` (one-to-one, unused) are gone: `candidate_store.match` is the one position matcher.
- **The second count**: `store.consensus` grouped by field and position while `labels.decisions` grouped by candidate
  with its own tie rule; both now use `store.object_keys` and `store.tally`.
- **"Apply area edits"** (`v2/area_edits.py`, `api/area_batch.py`, the `v2_area_edits` step, `stainid v2 area-edits`):
  people's areas no longer reach the analysis.
