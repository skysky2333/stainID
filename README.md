# stainID

Stain-level morphology for brain tissue microarrays (TMAs). stainID takes scanned
immunohistochemistry slides, cuts out every core, and measures object-level pathology on
calibrated, artifact-masked native-resolution fields:

| Stain | What is measured |
|---|---|
| **NeuN** | NeuN-positive neurons: density, positive fraction, soma size and shape |
| **6E10** | Amyloid plaques: burden, density, compact / diffuse morphotype, dense cores, vascular/edge amyloid, peri-plaque nuclei |
| **AT8** | Tau pathology: positive area, tau+ neurons (tangles and pretangles), neuropil thread network |

Everything can be done from a local web app, without writing code. The app walks you through each step, from slide
scans to a results table with one row per donor and brain region. Along the way you can check detections on the tissue,
label objects blind, and train the stain models on your own slides. Every step is also a `stainid` command
for scripted use.

![Workflow](docs/images/workflow.png)

## Getting started (no coding)

1. **Install once.** You need [Python 3.10 or newer](https://www.python.org/downloads/). To read Olympus `.vsi` scans you
   also need [Java](https://adoptium.net). Download this repository (green *Code* button → *Download ZIP*), unzip it, and
   double-click **`Install stainID.command`** (macOS) or **`Install stainID.bat`** (Windows).
   On macOS the first time, right-click the file → *Open* to allow it.
2. **Start.** Double-click **`Start stainID.command`** (or `.bat`). stainID opens in your web browser. Keep the small
   terminal window open while you work, because closing it stops running steps.
3. **Create a project** for your study on the welcome screen: choose a study name and an empty folder. All settings,
   core images and results for the study are kept in that folder.
4. **Follow Home → Workflow.** Each step says what it needs, what it makes and where the files go; press *Run*.
   The Help page has a getting-started guide, a glossary and answers to common questions.

## What the app looks like

| | |
|---|---|
| ![Home](docs/images/home.png) | ![Models](docs/images/models.png) |
| **Home**: your next step and overall progress | **Models**: download published models; create a training set, label it, train, compare, switch |
| ![Core viewer](docs/images/viewer.jpg) | ![Labelling](docs/images/labelling.jpg) |
| **Cohort & cores**: every core and field with detections, outlines and masks drawn on the tissue | **Label & check**: blinded, keyboard-driven labelling |
| ![Results](docs/images/results.png) | ![Settings](docs/images/settings.png) |
| **Results**: download the main table, what every column means, plots by group | **Settings**: study layout, diagnostic groups and every file location |

## The workflow

| Step | What it does | Command |
|---|---|---|
| Register slides | say which scan is which TMA and stain (guessed from file names) | *web app* |
| Find cores | fit the TMA grid on each slide | `stainid dearray` |
| Attach TMA map | donor, brain region and diagnostic group per core position (CSV upload, template provided) | `stainid layout` |
| Export cores | native-resolution core images | `stainid export` |
| Check core quality | tissue coverage, fragments, focus | `stainid qc` |
| Choose analysis fields | evenly spread ~560 µm fields per core | `stainid select` |
| Calibrate | one DAB threshold per slide, blind to diagnosis | `stainid calibrate` |
| Find nuclei | Cellpose-SAM nuclei (needed for AT8) | `stainid nuclei --stain AT8` |
| Detect NeuN neurons | candidates + NeuN model | `stainid neun` |
| Detect plaques and tau | 6E10 plaques and morphotypes; AT8 tau+ neurons and threads | `stainid fields` |
| Outline objects (optional) | Segment Anything outlines and shape features | `stainid masks --stain NeuN` |
| Make results tables | one row per donor-region, every stain | `stainid summarize` |

Model steps: `stainid download-models`, `stainid training-set`, `stainid train`. `stainid status` prints the progress of
every step. All steps are resumable and the heavy ones can be split with `--shard-index/--shard-count`.
Method details and parameters are in [docs/pipelines.md](docs/pipelines.md).

## Install from the command line

```bash
git clone https://github.com/skysky2333/stainID.git && cd stainID
python -m venv .venv && source .venv/bin/activate
pip install -e ".[app,deep,slides]"
stainid serve                       # opens the web app; or: stainid --project path/to/study status
```

The extras are:

- `slides`: reads `.vsi` scans (aicsimageio / Bio-Formats, needs Java).
- `deep`: PyTorch, Cellpose-SAM, Segment Anything and transformers.
- `app`: the web server.
- `dev`: pytest and ruff.

The web app is committed prebuilt, so Node is only needed to change the front end (see
[docs/development.md](docs/development.md)).

## Documentation

- [docs/webapp.md](docs/webapp.md): the web app page by page, the review and training workflow, and the API
- [docs/project.md](docs/project.md): project settings (`stainid.yaml`), input tables and output files
- [docs/pipelines.md](docs/pipelines.md): what each step computes, with parameters
- [docs/development.md](docs/development.md): tests, front-end development, adding a step

## Repository layout

```
src/stainid/
  project.py        stainid.yaml loading, saving and path resolution
  cli.py            the `stainid` command
  workflows/        every step (steps.py is the registry the CLI, job runner and web app share)
  training/         training sets and model training from blinded labels
  slides/           slides table, scan reading, dearraying, core export, TMA map
  qc/               core QC, focus, artifact exclusions
  sampling/         field selection
  imaging/          colour deconvolution, tissue / fold / artifact masks, calibration
  nuclei.py         Cellpose-SAM nuclei
  stains/           neun/, amyloid/, tau/ stain-specific detection and classifiers
  masks/            Segment Anything outlines and shape features
  analysis/         core / donor-region aggregation and the column dictionary
  registration/     cross-stain core registration
  review/           blinded review sets
  api/              FastAPI backend (and the built web app in api/static)
webapp/             React + TypeScript front end
tools/qupath/       QuPath scripts for TMA grids and core export
tests/              pytest suite (synthetic data only)
```

## Acknowledgements

`stainid/stains/amyloid/wong_consensus.py` is copied from the consensus-learning code of
Wong et al. (Keiser lab, [keiserlab/consensus-learning-paper](https://github.com/keiserlab/consensus-learning-paper)).
stainID uses the following published models; their weights are not redistributed here:

- Wong DR, Tang Z, Mew NC, et al. Deep learning from multiple experts improves identification of amyloid neuropathologies. *Acta Neuropathol Commun* 10, 66 (2022).
- Tang Z, Chuang KV, DeCarli C, et al. Interpretable classification of Alzheimer's disease pathologies with a convolutional neural network pipeline. *Nat Commun* 10, 2173 (2019).
- Pachitariu M, Rariden M, Stringer C. Cellpose-SAM: superhuman generalization for cellular segmentation. *bioRxiv* (2025).
- Kirillov A, Mintun E, Ravi N, et al. Segment Anything. *ICCV* (2023).
- Filiot A, Ghermi R, Olivier A, et al. Scaling self-supervised learning for histopathology with masked image modeling (Phikon). *medRxiv* (2023).

## License

BSD 2-Clause, see [LICENSE](LICENSE). Vendored third-party code keeps its original attribution.
