# stainID

Stain-level morphology for brain tissue microarrays (TMAs). stainID takes scanned
immunohistochemistry slides, cuts out every core, and measures object-level pathology on
calibrated, artifact-masked native-resolution fields:

| Stain | What is measured |
|---|---|
| **NeuN** | NeuN-positive neurons: density, positive fraction, soma size and shape |
| **6E10** | Amyloid plaques: burden, density, compact / diffuse morphotype, dense cores, vascular/edge amyloid, peri-plaque nuclei |
| **AT8** | Tau pathology: positive area, tau+ neurons (tangles and pretangles), neuropil thread network |

The whole study runs from a local web app: it goes from slide scans to one results row per donor and brain region.
Along the way you can check detections on the tissue, label objects blind and train the stain models on your own
slides. Every step is also a `stainid` command.

![Workflow](docs/images/workflow.png)

*Screenshots show a synthetic demo project, not study data (see [The app](#the-app)).*

## What you need

| | Where it comes from |
|---|---|
| **Slide scans**, one per TMA and stain (`.vsi`, `.svs`, `.ndpi`, `.czi`, `.tif`, …) | your scanner |
| **TMA map**: which donor, brain region and diagnostic group sits at each core position | the maps your lab already has (Excel, Word or CSV), read by the guided setup; a map that is only a photo is typed into a grid |
| Donor information, optional (age, sex, APOE, PMI, …) | your case spreadsheet, read by the guided setup |
| **Stain models** (NeuN, 6E10, AT8) | trained in the app on objects you label, or copied from an earlier study with the same staining |
| **Published models** (Cellpose-SAM, Segment Anything, Phikon) | downloaded by the app (Models page) |
| Published plaque CNN weights (Wong et al. 2022; Tang et al. 2019), optional, used by the 6E10 model | copied by hand into `data/models/plaque_cnn` (they have no download link) |
| Python 3.10+, and Java for `.vsi` scans | [python.org](https://www.python.org/downloads/), [adoptium.net](https://adoptium.net) |

## Getting started

1. **Install once.** Download this repository (green *Code* button → *Download ZIP*), unzip it, and double-click
   **`Install stainID.command`** (macOS) or **`Install stainID.bat`** (Windows).
   On macOS the first time, right-click the file → *Open* to allow it.
2. **Start.** Double-click **`Start stainID.command`** (or `.bat`). stainID opens in your web browser; the admin
   password is `admin` until you change it in Settings. Keep the small terminal window open while you work, because
   closing it stops running steps.
3. **Create a project** on the Projects page (*New study*): a study name and where to keep it. Settings, core images
   and results for the study are all kept in its folder. The Projects page lists every study on this computer; several
   people can work at the same time, each in their own study.
4. **Follow the guided setup** (Home → *Start the guided setup*). Six short screens: each slide's TMA and stain
   are chosen from lists, beside a picture of the slide's label (filled in where the file name says them clearly), the cores are found on each slide (the grid size too), the TMA maps and
   the donor sheet are read from the files your lab keeps them in, and a check says in plain words what does not fit
   together. You confirm what was found; every technical detail is under *Advanced*. *Run everything* then starts
   all the analysis steps. Adding slides later works the same way (Home → *Add slides*); see
   [Adding new slides to a study](docs/guide_adding_slides.md).
5. **Follow the run on the Workflow page.** Each step shows its progress; *Details* shows what it needs, what it
   makes, where the files go, its options and its log, and any step can be run on its own.
6. **Check and use the results.** Look at detections on the tissue (Cohort & cores), review fields alone or with
   others through a link to measure accuracy and collect labels (Labels), train better models from those labels
   (Models), and download the results table (Results).

The Help page in the app has the same guide, answers to common questions and a glossary.

## The workflow

| | Step | What it does | Command |
|---|---|---|---|
| **Slides** | Register slides | say which scan is which TMA and stain (chosen from lists; filled in where the file name says it clearly) | *web app* |
| | Find cores | fit the TMA grid on each slide (the grid size is found when a slide has none set) | `stainid dearray` |
| | Attach TMA map | donor, brain region and diagnostic group for each core | `stainid layout` |
| | Export cores | native-resolution core images | `stainid export` |
| | Check core quality | tissue coverage, fragments, focus | `stainid qc` |
| **Tiles** | Cut cores into tiles | every ~560 µm tile of tissue in each core; detection looks at every one | `stainid tiles` |
| | Calibrate stain thresholds | one DAB threshold per slide from every tile of its cores, blind to diagnosis | `stainid calibrate` |
| **Detection** | Find nuclei | Cellpose-SAM cells for every stain; the slow step, the one to run on a GPU | `stainid nuclei --stain NeuN --stain AT8 --stain 6E10` |
| | Detect NeuN neurons | candidates + NeuN model | `stainid neun` |
| | Detect plaques and tau | 6E10 plaques and morphotypes; AT8 tau+ neurons and threads | `stainid fields` |
| | Outline objects | Segment Anything outlines and shape features (optional) | `stainid masks` |
| **Results** | Make results tables | one row per donor and region, every stain | `stainid summarize` |
| **Models** | Download public models | Cellpose-SAM, Segment Anything, Phikon | `stainid download-models` |
| | Prepare fields for labelling, Train a model | candidates for labelling (a field is one tile); train and compare a stain model from the labelling sessions you choose | `stainid labels prepare`, `stainid train` |

Steps run in the background, in dependency order, at most two heavy ones at a time; all of them can be stopped and
resumed. [docs/webapp.md](docs/webapp.md) explains how the queue works, and [docs/pipelines.md](docs/pipelines.md)
says what each step computes. Image sets and whole slides replace the six steps from Register slides to Cut cores into tiles with *Prepare images*
(`stainid images`, below).

## Image sets, whole slides and other labs' datasets

A study does not have to be a TMA. *New study* asks what you will analyse:

| Kind | What it is | Set up |
|---|---|---|
| **TMA study** | slide scans of tissue microarrays and their TMA maps | Slides → Cores → Map → Donors → Checks → Run |
| **Image set** | a folder of image tiles or regions, each from a known donor | Images → Donors → Checks → Run |
| **Whole slides** | whole-slide scans (.svs, .ndpi, .tif, .czi, OME-Zarr); the tissue of each slide is analysed region by region | Slides → Donors → Checks → Run |
| **From a dataset package** | a public or shared dataset described by `dataset.yaml`, `images.csv`, `annotations.csv` and `donors.csv` | made for you; then as an image set or whole slides |

There is no map and no core grid for image sets and whole slides: each image, or each region of tissue, takes the place
of a core, so detection, results, labels, models and the viewer work the same. Per study you say what each stain is
(6E10-type amyloid such as 4G8 or NAB228, AT8-type tau such as PHF-1, NeuN), the colour it was developed in (brown DAB,
Fast Red, NovaRed, a blue chromogen; also two colours on one slide, of which you analyse one) and the pixel size;
stainID brings every image to the models' scale (0.274 µm/px) and colours without copying the files. A dataset
package's annotations become one read-only labelling session per rater plus their consensus, and *Models → Score the
models on these labels* gives precision, recall, F1 and AUC against each rater and the consensus, beside how well the
raters agree with each other: a test of the models on another lab's data. Whole slides need
`pip install "stainid[slides]"` (OpenSlide, Zarr, Bio-Formats). See [docs/project_kinds.md](docs/project_kinds.md).

## The app

All screenshots show a synthetic demo project: the tissue images, donors, labels and results are generated by
[tools/demo/make_demo_project.py](tools/demo/make_demo_project.py) and contain no study data.

| | |
|---|---|
| ![Home](docs/images/home.png) | ![Models](docs/images/models.png) |
| **Home**: your next step and overall progress | **Models**: download published models; train from all labels, compare, switch |
| ![Core viewer](docs/images/viewer.jpg) | ![Labels](docs/images/labels.png) |
| **Cohort & cores**: the whole core; click any spot for a full-resolution preview with the detections drawn on it | **Labels**: progress per stain, agreement with the AI, labelling sessions and the link to share |
| ![Reviewing a field](docs/images/annotate.jpg) | ![Projects](docs/images/projects.png) |
| **Reviewing a field** (at `/p/<project>/annotate`): the AI's calls and outlines pre-filled; people correct, add and confirm | **Projects**: every study with its status; new study, open a folder, switch |
| ![Results](docs/images/results.png) | ![Settings](docs/images/settings.png) |
| **Results**: download the main table, what every column means, plots by group | **Settings**: study layout, diagnostic groups and every file location |

## Command line

```bash
git clone https://github.com/skysky2333/stainID.git && cd stainID
python -m venv .venv && source .venv/bin/activate
pip install -c constraints.txt -e ".[app,deep,slides]"   # constraints.txt: the versions the trained models need
stainid serve                              # the web app
stainid --project path/to/study status     # progress of every step
stainid --project path/to/study neun       # run one step
```

Install extras: `slides` reads scanner files (Bio-Formats, needs Java; OpenSlide and Zarr for whole-slide projects); `deep` adds PyTorch, Cellpose-SAM, Segment
Anything and transformers; `app` is the web server; `dev` adds pytest and ruff. Heavy steps can be split across
machines with `--shard-index/--shard-count`. The web app is committed prebuilt, so Node is only needed to change the
front end.

## Documentation

Read in this order:

1. [Adding new slides to a study](docs/guide_adding_slides.md): the guided setup, step by step, for people new to stainID.
2. [Using the app](docs/webapp.md): every page, how steps are queued, labelling and training models.
3. [Project files](docs/project.md): the project settings (`stainid.yaml`), the input tables and every output file.
   [Project kinds](docs/project_kinds.md): image sets, whole slides, imaging settings, dataset packages and scoring on
   imported labels.
4. [Methods](docs/pipelines.md): what each step computes, with its parameters, and the models.
5. [Development](docs/development.md): tests, front end, the API and adding a step.

## Repository layout

```
src/stainid/
  project.py        stainid.yaml loading, saving and path resolution
  cli.py            the `stainid` command
  workflows/        every step (steps.py is the registry the CLI, job runner and web app share)
  labels/           label store, candidate pool, labelling sessions, import of earlier label sets
  training/         candidate features and model training from the label store
  slides/           slides table, scan reading, dearraying, core export, TMA map
  qc/               core QC, focus, artifact exclusions
  sampling/         field selection
  imaging/          colour deconvolution, tissue / fold / artifact masks, calibration
  nuclei.py         Cellpose-SAM cells (shared by every stain)
  stains/           neun/, amyloid/, tau/ stain-specific detection and classifiers
  masks/            Segment Anything outlines and shape features
  analysis/         core / donor-region aggregation and the column dictionary
  registration/     cross-stain core registration
  review/           blind review sets
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
