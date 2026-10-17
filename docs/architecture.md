# One store per object

Every object people or detection make has one owner module and one store; every view reads it through that owner,
so an edit shows everywhere at once. Human annotations — object labels and people's areas alike — only train and
evaluate models: the analysis reads detection's outputs only.

| Object | Owner module | Store file | Who reads it | How edits invalidate | Used in analysis? |
| --- | --- | --- | --- | --- | --- |
| Candidates (detection's objects: position, outline, call, decision) | `stainid.candidates` (`labels.candidate_store` reads it for the label pages) | `<outputs.v2>/candidates/<stain>/<core>.parquet` | stages 4–5, Cores view, both rounds, scoring, training | a stage's manifest digest of its inputs: a changed input remakes that core | yes |
| Outlines and the default mask | `stainid.candidates` (`outline_rings`; `outline_default` setting) | the candidate store (`outline`, `masks`) | stage 5 measures, Cores view, rounds | as candidates | yes |
| Calls and classes (detection's) | `stainid.v2` stage 4 (`classify`) | the candidate store (`class`, `subtype`, `decision`) | stage 5, Cores view, rounds, `labels.compare` | stage 4 manifest | yes |
| Excluded areas, automatic | `stainid.exclusions.automatic` | `<outputs.v2>/masks/<stain>/<core>.png` (stage 2; bits `v2.proposals.BITS`) | stages 2–5 (valid area, objects' excluded share), Cores view and rounds as "Excluded from analysis" (`exclusions.view`), `final_say` | stage 2 manifest; its digest (`exclusions.digest`) is in the later stages' manifests | yes |
| People's areas (painted, marked fine) | `stainid.exclusions.annotated` (format: `labels.exclusions`) | `exclude` labels in `<outputs.reviews>/labels.jsonl` (latest edit wins; "AI reviewer" for the earlier hand-drawn polygons) | Annotations layers (Cores view, rounds, Mark bad areas), `final_say` → `drop_excluded` (training, scoring, model comparison, object rounds' fading), Models → Bad-area detector | none: no stage reads them | no — train and evaluate only |
| Object labels (class, type, outline, added objects) | `stainid.labels.store` (classes and keys: `labels.vocabulary`) | `<outputs.reviews>/labels.jsonl` | `labels.decisions` (scoring), `labels.overview`, `labels.compare`, training | caches keyed on the log's signature | no — train and evaluate only |
| Claims and session state | `stainid.labels.sessions`, `api.work` (holds) | `<outputs.reviews>/sessions/<name>.json`; holds in memory | the round pages, Labels page | — | no |
| Review status (done per field, areas checked per core × stain) | `stainid.labels.status` | `done` labels in `labels.jsonl` | Labels page, Cores view minimap, rounds' field draws | as object labels | no |
| People against detection | `stainid.labels.compare` | — (worked out from the labels and the candidate store) | Cores view Annotations layer, Labels overview, scoring | as both | no |

`exclusions.final_say` (people's areas over detection's) exists only as the label set for training and evaluating:
labels on objects there are left out (`drop_excluded`), and it is the target the bad-area detector is scored against.
It is never an analysis input, so painting or clearing an area changes no stage input and no digest.

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
- **"Apply area edits"** (`v2/area_edits.py`, `api/area_batch.py`, the `v2_area_edits` step, `stainid v2 area-edits`):
  people's areas no longer reach the analysis.
