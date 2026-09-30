# Web app

Start it by double-clicking `Start stainID.command` / `Start stainID.bat`, or run:

```bash
stainid serve                              # opens http://127.0.0.1:8765 in the browser
stainid --project path/to/study serve      # open a specific project
stainid serve --no-browser --port 9000
```

Without `--project`, the app opens the project used last (remembered in `~/.stainid/app.json`), or the current folder.
If that folder has no `stainid.yaml`, a welcome screen offers to create a new project or open an existing one.

The server is a FastAPI app (`stainid.api`) serving the built React front end from `stainid/api/static` and a JSON API
under `/api`. It binds to localhost only and has no authentication, so do not expose it on a network.

## Pages

| Page | What it is for |
|---|---|
| **Home** | Your next step (with a button to it), a checklist of every step, cohort numbers, recent jobs. |
| **Workflow** | A **pipeline map** at the top shows every step as a box in the order they can run. Arrows mean "needs the result of", colours show done / running / waiting / failed / older model; hover to highlight what a step needs and what needs it, click to open it. **Now running** lists running and waiting steps with progress, elapsed time, time left and why a step is waiting; **Run all remaining steps** queues every unfinished step (except the optional outlines) in one click. Below, every step in order. Finished steps fold away; each open card shows:<ul><li>what the step does and how long it takes;</li><li>**Needs** and **Makes**, each file with its location, whether it exists, and "Show in Finder" / "Download";</li><li>options, with advanced ones hidden;</li><li>Run / Continue / Run again, a live progress bar, "Show log", and a plain-language reason when a step fails.</li></ul>The *Register slides* card holds the slides-table editor. The *Attach TMA map* card holds the template download and CSV upload. |
| **Cohort & cores** | Every core of every TMA; click one for the field viewer. The viewer's layers are detections by class, SAM outlines, DAB above the slide threshold, excluded regions, AT8 threads and field boundaries. |
| **Stain thresholds** | The per-slide DAB threshold and why it exists. |
| **Label & check** | Training sets and check sets, the form to create a check set, and older study sets (read-only). |
| **Models** | Every model with its status and settings; download buttons for published models; the three-step training flow; trained models compared with the current one, and "Use this model". |
| **Results** | The main results table (download), every table in the project, plots by diagnostic group and region, and the meaning of every column. |
| **Settings** | Study name, TMA layout (core-name prefix, rows, columns), diagnostic groups (code, label, colour, order), and every file location with a folder picker. Also open or create another project. |
| **Help** | Getting started, common questions, glossary. Dotted words anywhere in the app show their definition on hover. |

Steps run as background `stainid` processes, started by a queue with two rules:

- **Order:** a step waits while an earlier step it depends on (its `after` list in `workflows/steps.py`, the arrows on the
  map) is still running or waiting. The same step with the same options never runs twice at once.
- **Load:** at most two heavy steps run at the same time.

Each waiting step shows the reason. You can press Run on several steps in a row and they start in the right order.
Just before a step starts, the runner checks that everything it needs exists; if not, the step is **skipped** and the
card says what is missing and which step makes it. When a step fails or is stopped, the queued steps that were waiting
for it are skipped too, instead of failing one after another. The
sidebar shows how many steps are running.

Steps keep running when the browser is closed. If the server stops and starts again while a step is running, the step is
picked up again and marked finished or failed when it ends. When a stain model is switched, the detection steps made
with the old one are marked **Made with an older model** until they are run again with *Start over*.

## Labelling

Every labelling set is a folder under `outputs.reviews` with:

| file | |
|---|---|
| `key.csv` | one row per item: `review_id`, `tile_id`, `x`, `y` (field-crop pixels) and hidden columns (group, donor, region, model probability) |
| `labels.csv` | append-only labels (`review_id`, `label`, `confidence`, `notes`, `reviewer`, `timestamp`); the last label per item wins |
| `meta.json` | stain, label options, crop size, instructions, purpose |

While labelling, only the image crop is shown. The object sits under a blue cross, and diagnosis, donor and model answer
are hidden. Keys `1`–`n` assign a label and `←`/`→` move between items.

- **Check sets** (`reviews/<name>`) sample detected objects: random, or those with probability in a band. They
  are balanced across groups. After all items are labelled, two tables appear: your labels against what the model
  decided (how accurate the results are) and your labels per diagnostic group.
- **Training sets** (`training/<name>`) are made by the *Create a training set* step:
  - Fields are spread across TMAs and groups; optionally only fields where the current model found objects.
  - Every candidate object is found exactly as the detection step does.
  - Each field contributes some candidates, half of them ones the current model is unsure about.
  - Their features are saved in `features.csv`, plus `embeddings.npy` of Phikon features for 6E10.

## Training a model

*Train a model* (`stainid train --stain X`) uses every labelled training set of that stain:

- Labels: "unsure" is skipped. It needs at least 10 positive and 10 negative labels.
- Model: the same random forest the pipeline uses. For 6E10 it also fits the Phikon linear probe and keeps the Wong
  2022 CNN as a fixed ensemble member.
- Thresholds: the compact/diffuse threshold (6E10) and the tangle/pretangle threshold (AT8) come from your compact /
  diffuse and tangle / pretangle labels when there are enough, otherwise from the current model.
- Validation: leave-one-TMA-out (5-fold if there are too few TMAs). The current model is scored on the same labels.

The model is saved to `outputs.trained_models/<stain>_<time>/bundle.joblib` with `report.json`. It is used only after
"Use this model", which sets `models.<neun|amyloid|tau>` in `stainid.yaml`. Detection steps refuse to mix results from
different models; run them again with *Start over*, which moves the old results aside.

## API

Interactive docs are at `/docs` while the server runs. Main routes:

| Route | |
|---|---|
| `GET /api/project`; `PUT /api/project/config`; `POST /api/project/open`, `/api/project/create` | project settings and switching |
| `GET /api/steps` | every step: needs / makes (with paths and existence), options, progress, latest job |
| `GET/POST /api/jobs`; `DELETE /api/jobs/{id}`; `GET /api/jobs/{id}/log` | run, stop and follow steps |
| `GET/PUT /api/slides`; `GET /api/grids`, `/api/grids/{name}` | slides table; core-grid check images |
| `GET /api/templates/{tma_layout,donor_metadata}.csv`; `POST /api/upload/{kind}` | CSV templates and validated uploads |
| `GET /api/fs`; `POST /api/reveal`; `GET /api/download` | folder picker, Show in Finder, file download (project folder only) |
| `GET /api/summary`, `/api/calibration`, `/api/cores`, `/api/cores/{id}`, `/api/tmas/{tma}/layout` | cohort browsing |
| `GET /api/fields/{tile}/objects`, `/api/fields/{tile}/outlines`; `GET /api/images/...` | detections, outlines, images and overlays |
| `GET/POST /api/reviews`; `/api/reviews/{name}/items`, `/labels`, `/summary`, `/image/{id}.jpg` | labelling sets |
| `GET /api/models`, `/api/models/trained`; `POST /api/models/activate` | models |
| `GET /api/tables`, `/api/tables/{name}/columns`, `/rows`, `/feature` | output tables and column meanings |
