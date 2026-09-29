# Web app

```bash
stainid --project path/to/study serve            # http://127.0.0.1:8765
```

The server is a FastAPI app (`stainid.api`) serving the built React front end from
`stainid/api/static` and a JSON API under `/api`. It binds to localhost by default. It has no
authentication, so do not expose it on a shared network.

## Pages

| Page | |
|---|---|
| **Overview** | Donors, donor-regions, cores and fields; donors by group; per-stain progress of detection, nuclei and outlines; recent jobs. |
| **Cohort & cores** | TMA grid (or list) per array and stain; border colour is the diagnostic group. |
| **Core viewer** | Core thumbnail with the analysed fields, plus a zoomable field viewer. Layers: detections by class, SAM outlines, DAB above slide threshold, exclusions, AT8 thread network, field boundary. The object table lists class, probability and area. |
| **Calibration** | Per-slide DAB thresholds and tissue DAB quantiles; recompute into a new file. |
| **Pipelines & jobs** | One card per workflow step with its options; jobs run as `stainid` subprocesses. At most two heavy (GPU/CPU-bound) jobs run at once, and the rest queue. Live progress and logs; cancel running jobs. |
| **Review & annotate** | Create and label blinded review sets. |
| **Models** | Configured models, whether they exist, and bundle metadata (features, thresholds, training labels). |
| **Analysis** | Feature explorer for any output table, by group and split by region or TMA, plus raw rows. |
| **Project** | Resolved configuration and paths. |

Field images and overlays are rendered on demand and cached in `data/cache/fields`.

## Review workflow

1. **Create a set.** Pick a stain, optional model class, sampling strategy (`random`, or
   `uncertain` within a probability band), size, and whether to balance
   diagnostic groups. Objects are sampled from pipeline outputs.
2. **Label.** Each item is shown as a raw crop (NeuN 45 µm, 6E10 85 µm, AT8 55 µm) with a centre
   cross and nothing else.
   - The group, donor, region, model class and probability stay hidden.
   - Keys `1`–`n` assign a label; `←`/`→` navigate.
   - Confidence and notes are optional.
3. **Unblind.** Once every item is labelled, the summary shows label counts by diagnostic group;
   `key.csv` holds the model class and probability for agreement analyses.

A set is a folder under `outputs.reviews/reviews/<name>/`:

| file | |
|---|---|
| `key.csv` | items with all hidden columns |
| `labels.csv` | append-only labels (`review_id`, `label`, `confidence`, `notes`, `reviewer`, `timestamp`); the last label per item wins |
| `meta.json` | stain, labels, crop size, instructions |

Labels feed the stain classifiers (`stainid.stains.*.classifier`).

## API

Interactive docs are at `/docs` while the server runs. Main routes:

| Route | |
|---|---|
| `GET /api/project`, `/api/summary`, `/api/calibration` | configuration, cohort summary, thresholds |
| `GET /api/cores`, `/api/cores/{id}`, `/api/tmas/{tma}/layout` | cohort browsing |
| `GET /api/fields/{tile}/objects`, `/api/fields/{tile}/outlines` | detections and SAM polygons |
| `GET /api/images/cores/{core}/{stain}.jpg`, `/api/images/fields/{tile}.jpg`, `/api/images/fields/{tile}/{layer}.png` | images and overlays |
| `GET /api/workflows`; `GET/POST /api/jobs`; `DELETE /api/jobs/{id}`; `GET /api/jobs/{id}/log` | jobs |
| `GET/POST /api/reviews`; `/api/reviews/{name}/items`, `/labels`, `/summary` | review sets |
| `GET /api/tables`, `/api/tables/{name}/columns`, `/rows`, `/feature` | output tables |
| `GET /api/models` | models |
