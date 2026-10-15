# Project kinds: TMA studies, image sets and whole slides

stainID was built for tissue microarrays, but a study can also be a folder of image tiles from another lab or a set of
whole-slide scans. Every kind ends up with the same thing, a table of *core images* cut into ~560 µm tiles, so every
step from the stain thresholds on (nuclei, NeuN, plaques and tau, outlines, results, labels, models, the viewer) runs
unchanged. This page says how each kind gets there, how images from other labs are brought to the models' scale and
colours, how a dataset package becomes a project, and how the models are scored on another lab's labels.

| Kind (`kind:` in stainid.yaml) | What it holds | Guided setup | Its "core images" |
|---|---|---|---|
| **TMA study** (`tma`, the default) | slide scans of TMAs and their TMA maps | Slides → Cores → Map → Donors → Checks → Run | the cores found on each slide |
| **Image set** (`images`) | image tiles or regions (PNG, JPEG, TIFF), each from a known donor | Images → Donors → Checks → Run | each image |
| **Whole slides** (`slides`) | whole-slide scans (.svs, .ndpi, .tif, .mrxs, .scn, .czi, OME-Zarr) | Slides → Donors → Checks → Run | each region of tissue on a slide |

A new study's kind is chosen on the Projects page (*New study → What will you analyse?*), or comes from the dataset
package it is made from. Image sets and whole slides have no TMA map and no core grid; the setup says so and skips
those screens.

## The image list

An image set or whole-slide project lists its images in `inputs.image_list` (`data/images.csv`); the Images screen
makes it from a folder (subfolders are searched; a folder per donor gives each image its donor; the stain is read from
the file name where it says it) or from an uploaded spreadsheet.

| column | |
|---|---|
| `image_id` | a unique name, kept exactly as written (a dataset's annotations name images by it); made from the file name when empty |
| `path` | the file, absolute or relative to the project; it is read where it lies and never copied or changed |
| `stain` | the stain as the lab calls it (6E10, 4G8, NAB228, AT8, PHF-1, NeuN, …); a slide with two stains is written `AT8+pTDP-43` |
| `donor_id` | the donor (required) |
| `region` | brain region (optional); results are per donor and region (`sample_region_id = <donor>_<region>`) |
| `slide_id`, `site`, `batch` | optional; the stain threshold is set per batch, else per slide, else per site, else once for the stain |
| `pixel_size_um` | optional; else the file's own pixel size, else `imaging.native_pixel_size_um` |
| `disease_group` | optional; else the donor table's group |
| `split` | optional: the dataset paper's train / val / test |

The *Prepare images* step (`stainid images`) replaces Find cores, Export cores and Cut cores into tiles:

- **Image set**: every image is one core image. It is cut into equal tiles of at most 2048 px (an image smaller than a
  tile is one tile); every tile with any tissue is kept, as the lab chose the images.
- **Whole slides**: each slide is covered by a grid of square regions of 6144 analysis pixels (about 1.7 mm, a large
  TMA core); every region holding at least 5 % tissue on an 8 µm/px picture, cut down to its tissue plus 60 µm, is one
  core image. Regions never overlap. They are tiled like whole cores (tiles with at least 10 % tissue).

The core image table (`inputs.core_images`) records for every core image its source (`source_path`, `source_box` in the
source's level-0 pixels, `scale`), its own antibody (`antibody`) and colour (`chromogen`); `stain` is the pipeline stain
(below). Run *Prepare images* again after changing the image list or the imaging settings.

## Imaging settings

```yaml
kind: images
pixel_size_um: 0.2738            # the analysis scale: the models' scale; images are brought to it
imaging:
  native_pixel_size_um: 0.5      # the images' own pixel size when neither the list nor the files say it
  rescale: on_import             # or on_the_fly
  stains:
    4G8:    {family: amyloid, chromogen: DAB}
    NAB228: {family: amyloid, chromogen: NovaRed}
    6E10:   {family: amyloid, chromogen: FastRed}
    AT8:    {family: tau, chromogen: blue, others: [DAB]}   # SEA-AD "AT" slide: AT8 in blue, pTDP-43 in DAB
    IBA1:   {family: ignore}
```

**Model families.** `family` says which of stainID's models reads a stain: `amyloid` (6E10-type: 6E10, 4G8, NAB228,
Aβ40, Aβ42, …), `tau` (AT8-type: AT8, PHF-1, CP13, …), `neun`, or `ignore`. stainID guesses it from the name and asks
on the Images screen when it cannot. In the tables an image carries its family's pipeline stain in `stain` (6E10, AT8
or NeuN), so every step, model, label and page treats it as that stain, and its own antibody in `antibody`.

**Colours.** `chromogen` is the colour the stain was developed in: `DAB` (brown, the default), `FastRed`, `NovaRed`,
`AEC`, `blue` (a blue chromogen) or `FastBlue`; `others` lists another chromogen on the same slide, and `counterstain` is
hematoxylin unless set. An image that is not plain DAB on hematoxylin is unmixed with these stain vectors (colour
deconvolution, `stainid.imaging.chromogens`) and drawn again in hematoxylin and DAB colours: the stain of interest
becomes DAB, the counterstain stays hematoxylin and every other chromogen is left out. The stain thresholds, the DAB
layers and the detection steps then measure the stain of interest exactly as they measure DAB, so the colour
deconvolution and the thresholds follow the per-stain setting without any change to them. With two chromogens on a
slide, the stain listed (`AT8` in `AT8+pTDP-43`) is the one analysed; the Images screen shows each stain's picture as
scanned and as stainID reads it. The DAB, hematoxylin, eosin, Fast Red, Fast Blue and AEC vectors are the published
Ruifrok-Johnston values; NovaRed and the generic blue are approximations, so a lab can give its own, measured from an
area of pure stain, as `vectors: {NovaRed: [r, g, b], hematoxylin: [r, g, b]}` under the stain.

**Scale.** Images within 3 % of `pixel_size_um` are read as they are; others are enlarged or reduced to it (whole
slides from the pyramid level closest to it). The models, Cellpose's cell size and every µm figure assume this scale.

**Converting.** An image already at the analysis scale and in DAB is used where it lies: its own path goes into the
core image table. Any other becomes a small link file, `data/images/links/<core>_<stain>.json` (source, box, scale,
colour recipe), which every step opens through `stainid.imaging.core_images` (`open_image`, `read_bgr`).
`rescale: on_import` keeps the converted picture among the app's cached pictures (kind *Images at the analysis scale*,
through `stainid.disk_cache`): it is made when the images are prepared and made again whenever it is missing or older
than its link, so it counts towards the space limit and may be cleared at any time. `on_the_fly` makes it each time it
is read and stores nothing: no extra disk space, slower, and the default for whole slides (a slide's regions would
take tens of gigabytes).

## Dataset packages

A dataset package is one folder describing a public (or another lab's) dataset; the files stay where the package says
(for the public datasets, `public_datasets/<id>/raw/`).

```
package/
  dataset.yaml     id, title, source_urls, doi, licence, citation, sites, stains [{stain, antibody, chromogen,
                   counterstain, also_on_slide, family}], pixel_size_um, image_kind (tiles | regions | wsi), class_map
  images.csv       image_id, path (relative to package/ or absolute), width_px, height_px, pixel_size_um, stain,
                   slide_id, donor_id, region, site, split
  annotations.csv  image_id, rater, kind (point | box | polygon | image_label), x, y, x2, y2, polygon (JSON), label,
                   stainid_class, confidence; positions in the image's own pixels
  donors.csv       optional: donor_id, diagnosis (or disease_group), braak, cerad, thal, adnc, apoe, age, sex, pmi, …
```

*Projects → New study → From a dataset package* (or `stainid package <package> <new project folder>`) shows what the
package holds (images, donors, raters, labels, licence and citation, whether the image files are found), lets you check
which model family each stain is and which stainID class each label means, and makes the project:

- its kind (`tiles` and `regions` → image set, `wsi` → whole slides) and imaging settings from dataset.yaml (a pixel
  size written as text, e.g. "per image", is left to images.csv);
- the image list, pointing at the package's files (absolute paths; nothing is copied);
- the donor table from donors.csv (the diagnosis is the group when there is no group column), donor ids kept as
  written (`donor_ids: {digits: 0}`);
- `dataset:` in stainid.yaml: id, title, licence, citation, DOI, the package folder and the class map.

The annotations are placed when the images are prepared, because they go on the project's tiles
(`stainid.imageset.annotations`): every point, box, polygon (by its centre) and image label is moved to the core image
it lies on (for a whole slide, the region containing it), enlarged to the analysis scale, and to the tile it falls in.
Then

- `data/annotations/imported/<dataset>/annotations.csv` keeps every annotation in the project's frame (core-image and
  field-picture pixels, boxes as corners) with its rater and class; `import.json` says what was left out and why;
- every rater becomes one **read-only labelling session** (`import-<dataset>-<rater>`, a session per stain when a rater
  labelled several), and the consensus one more: the package's own `consensus` rater when it has one, else the
  objects more than half of the raters of that image marked within an object's size of each other (12 µm for plaques,
  8 µm for tau, 6 µm for neurons, or the box's own size), with the class most of them gave; image labels the same way;
- the labels go into the label store like any other (source `import: <dataset> <rater>`, boxes and polygons also as
  outlines), so they show on the Labels page and can train models; nobody can add to or change them.

The class map (`dataset.class_map`, label → stainID class or `ignore`) comes from the package and can be changed on the
import page or with `PUT /api/labels/imported/class-map`, which places the labels again. stainID's classes:
NeuN `NeuN+ neuron`, `not a neuron`; amyloid `compact plaque`, `diffuse plaque`, `small plaque`, `plaque` (type not
recorded), `not a plaque`, `vascular` (CAA), `artifact or unclear`; tau `tau+ neuron: tangle`, `tau+ neuron: pretangle`,
`tau+ neuron`, `not a tau+ neuron`, `neuritic plaque`, `thread`.

## Scoring the models on imported labels

Raters who drew **polygons** are also scored by the segmentation check (*Models → How well does detection outline
objects?*, `stainid score-segmentation`; see docs/webapp.md): their outlines are matched with every candidate the
detection step proposes and with the objects it counts, by IoU, on the fields where all of a rater's marks are
polygons. That shows objects never proposed, cut into pieces or merged, which the matching below cannot.

On the Models page, *Score the models on these labels* (or `stainid score`) compares the AI's published detections with
every rater and with the consensus, per stain, and saves `scores.json` and `scores.csv` next to the imported labels:

- **points**: an object the AI accepted finds a mark when it lies within 12 µm (plaques), 8 µm (tau) or 6 µm (neurons)
  of it, or within the object's own radius if larger; one object per mark, closest pairs first;
- **boxes** (and polygons): an accepted object whose centre is in the box finds it; `iou_recall` gives the share of
  boxes whose best object overlaps them by IoU ≥ 0.5 (the object taken as a square of its diameter);
- **image labels**: an image said to contain a class is found when the AI accepted any object of that class in it;
- **vessels**: `vascular` (CAA) marks are scored against the deposits the 6E10 step set aside as on a vessel or the
  tissue edge.

Recall ("Found") is the share of marks the AI found; precision ("Of the AI's, marked") the share of the AI's accepted
objects in the annotated images that someone marked; F1 combines them. When a rater marked only some types (e.g. only
cored plaques, as in the amyloid-YOLO set), precision counts only the AI's objects of those types, and `same_type` gives
how often the AI's type agreed with the mark's. The AUC (where the AI gives probabilities) says how well its
probability ranks marked objects above negative marks, or positive images above negative ones. *How much the raters
agree* gives every pair of raters' F1 (objects, matched the same way) and Cohen's kappa (image labels), so the AI's
figures can be read against human agreement. Precision assumes the raters marked every object of their types on the
images they annotated; only images some rater annotated count.

## Dependencies

Whole-slide reading needs the `slides` extra: `pip install "stainid[slides]"` (Bio-Formats via aicsimageio, OpenSlide
via `openslide-python` and `openslide-bin`, Zarr). aicsimageio pins `zarr < 2.16`, which needs `numcodecs < 0.16`
(later numcodecs releases removed `cbuffer_sizes`, which breaks `import zarr`); the extra pins both. Without it
stainID still reads plain pictures and tiled TIFFs (one pyramid level at a time with tifffile), says which reader is
missing on the Images and Checks screens, and leaves out images it cannot read with the reason in the step's log.

## For developers

| Module | |
|---|---|
| `stainid.project` | `KINDS`, `Project.kind`, the `imaging` and `dataset` sections |
| `stainid.imaging.sources` | one reader for pictures, OpenSlide, tifffile, OME-Zarr and Bio-Formats: size, levels, pixel size, regions |
| `stainid.imaging.chromogens` | stain vectors, colour recipes, conversion to H-DAB |
| `stainid.imaging.settings` | per-stain family and colour, native pixel size, scale factor |
| `stainid.imaging.core_images` | link files, `open_image` / `read_bgr` (every step opens core images through them), the image cache |
| `stainid.imageset.table` | the image list, folder scan |
| `stainid.imageset.prepare` | the Prepare images step: core image table, whole-slide regions, tiles |
| `stainid.imageset.package` | reading a dataset package and making a project from it |
| `stainid.imageset.annotations` | placing annotations, consensus, read-only sessions |
| `stainid.imageset.scoring` | matching, precision / recall / F1 / AUC, rater agreement |
| `stainid.imageset.checks` | the Checks screen of image sets and whole slides |
| `stainid.api.routes.imageset` | `/api/setup/images…`, `/api/labels/imported…`, `/api/models/score…` |

Steps carry the kinds they belong to (`Step.kinds`; `steps_for(project)`): a TMA study never shows Prepare images, an
image set never shows Find cores. Synthetic packages for every kind are in `tests/package_files.py`
(`tests/test_project_kinds.py`).
