# Using the app

## Starting

Double-click `Start stainID.command` (macOS) or `Start stainID.bat` (Windows), or run:

```bash
stainid serve                              # opens http://127.0.0.1:8765 in the browser
stainid --project path/to/study serve      # open a specific project
stainid serve --no-browser --port 9000
```

The app opens the project used last, or the current folder. If that folder has no `stainid.yaml`, a welcome screen
offers to create a new project or open an existing one. If stainID is already running, starting it again just opens the
browser. The app runs on your computer only and has no login, so do not expose it on a network.

## A study, from start to finish

| | Where | What you do |
|---|---|---|
| 1 | Welcome screen | Create a project: a study name and an empty folder. |
| 2 | **Settings** | Rows and columns of cores per slide, the core-name prefix (`LIP-` gives cores like `LIP-3_B-2`) and the diagnostic groups (code, label, colour, plot order). |
| 3 | **Models** | Download the published models. Add the NeuN, 6E10 and AT8 models, or train them (see [Training a model](#training-a-model)). |
| 4 | **Workflow** | Register the slides, upload the TMA map, then run the steps from top to bottom (or press *Run all remaining steps*). |
| 5 | **Cohort & cores** | Look at the detections on the tissue, core by core. |
| 6 | **Label & check** | Label a check set to measure how accurate the detections are on your slides. |
| 7 | **Results** | Download the results table (one row per donor and region) and read what every column means. |

## The Workflow page

- **Pipeline map.** Every step is a box, arranged in the order steps can run. An arrow means "needs the result of".
  Colours show done, partly done, running, waiting, failed and "made with an older model". Hover over a step to
  highlight what it needs and what needs it; click it to open its card.
- **Now running.** Running and waiting steps, with progress, elapsed time, time left and why a step is waiting.
  *Run all remaining steps* queues every unfinished step (except the optional outlines).
- **Step cards**, one per step, in order. Finished steps fold away. Each card shows:
  - what the step does and how long it takes;
  - **Needs** and **Makes**: every file with its location and whether it exists. A missing file has a button: go to
    the step that makes it, download it, or show the expected format and a template;
  - options, with advanced ones hidden;
  - Run / Continue / Run again, a progress bar, *Show log*, and a plain-language reason when a step fails.

The *Register slides* card holds the slides-table editor, and the *Attach TMA map* card holds the template download and
the CSV upload.

## How steps run

Steps run in the background, started by a queue with two rules:

- **Order.** A step waits while a step that makes something it needs (an arrow on the map) is still running or
  waiting. The same step with the same options never runs twice at once.
- **Load.** At most two heavy steps (Cellpose, detection, outlines) run at the same time.

So you can press Run on several steps in a row and they start in the right order; each waiting step says what it is
waiting for. Just before a step starts, the queue checks that everything it needs exists. If not, the step is
**skipped** and its card says what is missing and which step makes it. When a step fails or is stopped, the steps
waiting for it are skipped too, instead of failing one after another.

Steps keep running when the browser is closed, but stop when the stainID terminal window is closed; running a step
again continues where it stopped. When you switch to a different stain model, the detection steps made with the old one
are marked **Made with an older model** until you run them again with *Start over* (under Advanced options).

## Other pages

| Page | What it is for |
|---|---|
| **Home** | Your next step (with a button to it), a checklist of every step, cohort numbers and recent jobs. |
| **Cohort & cores** | Every core of every TMA; click one to open the viewer. Layers: detections by class, outlines, DAB above the slide threshold, excluded regions, AT8 threads and field boundaries. Scroll to zoom. |
| **Stain thresholds** | The DAB threshold of every slide and why it exists. |
| **Label & check** | Training sets, check sets, the form to create a check set, and older study sets (read-only). |
| **Models** | Every model with its status and settings, download buttons, the three-step training flow, and trained models compared with the current one. |
| **Results** | The main results table (download), every other table, plots by group and region, and the meaning of every column. |
| **Settings** | Study name, TMA layout, diagnostic groups and every file location, with a folder picker. Also opens or creates another project. |
| **Help** | Getting started, common questions and a glossary. Words with a dotted underline show their definition on hover. |

## Labelling

While labelling, only the image crop is shown: the object sits under a blue cross, and the diagnosis, donor and model
answer are hidden. Keys `1`–`n` assign a label and `←`/`→` move between items.

- **Check sets** sample detected objects, at random or within a probability band, balanced across groups. Once every
  item is labelled, two tables appear: your labels against the model's answers (how accurate the results are) and
  your labels per diagnostic group.
- **Training sets** are made by the *Create a training set* step (Models page):
  - fields are spread across TMAs and groups, optionally only fields where the current model found objects;
  - candidates are found exactly as the detection step finds them;
  - half of each field's candidates are ones the current model is unsure about.

Every set is a folder under `outputs.reviews` (`reviews/<name>` or `training/<name>`):

| file | |
|---|---|
| `key.csv` | one row per item: `review_id`, `tile_id`, `x`, `y` (field-crop pixels) and the hidden columns (group, donor, region, model probability) |
| `labels.csv` | labels (`review_id`, `label`, `confidence`, `notes`, `reviewer`, `timestamp`), appended; the last label per item counts |
| `meta.json` | stain, label options, crop size, instructions, purpose |
| `features.csv`, `embeddings.npy` | training sets only: the candidates' features (and Phikon embeddings for 6E10) |

## Training a model

*Train a model* (`stainid train --stain X`) uses every labelled training set of that stain:

- **Labels.** "Unsure" is skipped. It needs at least 10 positive and 10 negative labels; 100–200 labels is a good start.
- **Model.** The same random forest the pipeline uses. For 6E10 it also fits the Phikon linear probe and keeps the
  Wong 2022 CNN as a fixed ensemble member.
- **Thresholds.** The compact/diffuse threshold (6E10) and the tangle/pretangle threshold (AT8) come from your labels
  when there are enough, otherwise from the current model.
- **Validation.** Leave-one-TMA-out (5-fold if there are too few TMAs). The current model is scored on the same labels,
  so the Models page can show both side by side.

The model is saved to `outputs.trained_models/<stain>_<time>/bundle.joblib` with `report.json`. It is used only after
you press *Use this model*, which sets `models.<neun|amyloid|tau>` in `stainid.yaml`. Detection steps never mix
results from two models: run them again with *Start over*, which moves the old results aside, then *Make results
tables* again.
