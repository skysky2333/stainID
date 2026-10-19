# Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -c constraints.txt -e ".[dev,app,slides]"      # add ",deep" to run Cellpose / SAM / Phikon code paths
pytest
ruff check src tests
```

Tests use synthetic images only (`tests/conftest.py` builds a one-core, three-stain project),
so they need no study data or model weights. Tests that read `.vsi` slides need the `slides`
extra.

## Front end

```bash
cd webapp
npm ci
stainid --project path/to/study serve &   # API on :8765
npm run dev                                # Vite on :5173, proxies /api
npm run build                              # production build into src/stainid/api/static/
```

The production build in `src/stainid/api/static/` is committed so that installing stainID does not need Node:
run `tools/build_webapp.sh` and commit the result whenever the front end changes. The script runs `npm ci`, `tsc` and
`vite build`, in a temporary copy of `webapp/` when the checkout is on a drive without symlinks (exFAT), and uses
`~/.local/node/bin` when Node is not on the PATH.

React 19 + TypeScript + Vite, with no UI framework:

| path | |
|---|---|
| `src/index.css` | design tokens (light / dark) |
| `src/api.ts` | typed API client; `apiBase` / `apiUrl` decide which server it talks to; `projectId` / `scoped` add the page's project to every call |
| `public/config.js` | optional `window.STAINID_SERVER` for a copy of the app served from somewhere else |
| `src/hooks.ts` | `useFetch`, `useStored` |
| `src/project.tsx` | project context: group labels / colours, TMA names; the project switcher |
| `src/slides.tsx` | the slides-table editor shared by the guided setup and the Register slides step |
| `src/glossary.ts` | glossary behind hover definitions and the Help page |
| `src/components.tsx`, `src/charts.tsx` | shared pieces (help boxes, folder picker, path lines, option fields, charts) |
| `src/pages/` | one file per page |

## The guided setup

The guided setup (web app `/setup`, API `/api/setup/...` in `stainid/api/routes/setup.py`) only calls library
functions in `stainid/setup`, each tested on its own:

| module | |
|---|---|
| `files.py` | Excel, Word (.docx tables, read from the zip's XML), CSV and pasted text into rows of cell texts |
| `naming.py` | donor ids, brain regions, diagnostic groups and CERAD / Braak as people type them |
| `maps.py` | TMA map grids (cells over one to three lines or table rows) into `tma_layout` rows, one TMA at a time |
| `donors.py` | the case sheet: header found by its column names, section rows, id series, approximate values |
| `checks.py` | map, donor table and slides compared, as issues with fixes (`find_issues`, `apply_fix`) |
| `guide.py` | the state of each setup screen, the slide reports of *Find cores*, the run-time estimate |

Grid-size detection is in `stainid/slides/dearray.py` (`choose_grid`, `grid_score`). To try the setup by hand or
take its screenshots, `tools/demo/make_setup_demo.py OUT_DIR` makes a synthetic study that is not set up yet.

## Adding a workflow step

1. Implement it in `stainid/workflows/<step>.py` as a function of a `Project`. Print one `[i/n] …` line per unit of
   work so the job runner can show progress, skip finished outputs so the step is resumable, and raise `ValueError`
   with a plain-language message for user errors; the web app shows that message on the step card.
2. Add a subcommand in `stainid/cli.py`.
3. Add a `Step` to `STEPS` in `stainid/workflows/steps.py`: title, a one-line summary and details in plain language,
   options (with help text), `heavy`, a status function, and the files it touches, as `Resource` ids:
   - `needs`: must exist before it starts (otherwise the queue skips it and says which step makes the file);
   - `uses`: read when present;
   - `produces`: what it makes for later steps;
   - `writes`: shared files it updates (such as the core table), so two steps never write one at the same time.

   Dependencies (the pipeline-map arrows and the queue order) are derived from these lists, so the job runner, the
   pipeline map and the Run page pick the step up without further changes. A new input file gets a `Resource`
   with its `download`, `template`, `setting` or `format`, which gives the missing-file buttons on the step card.

## Adding a cache

Pictures and other files the app makes from a project's data only to show it quickly are caches: they live in
`data/cache/<kind>/`, take at most the project's space limit together, and may be deleted at any time (least recently
used first) to stay under it. Everything that reads one must therefore make it again when it is missing, and show its
usual "preparing" state meanwhile, never an error. All of it goes through `stainid/disk_cache.py`:

1. Add the kind to `disk_cache.KINDS` with what it is called in Settings ("Labelling pictures").
2. Name files with `state.cache_dir(kind)` (in the API) or `disk_cache.folder(root, kind)` / `disk_cache.path(root,
   kind, name)`; never build `data/cache` paths by hand. Files that belong together (a pack and its index) share the
   part of their name before the first dot: they are deleted together.
3. Write with `disk_cache.write(path, data)` (atomic, recorded); code that must write some other way calls
   `disk_cache.wrote(path)` once the file is complete.
4. Call `disk_cache.used(path)` whenever a cached file is read or served, so what is in use is kept. Both calls only
   note the path in memory; a background thread writes the notes into the index every few seconds.
5. Test that the reader makes the file again after `disk_cache.evict(root, 0, keep_seconds=0)` (see
   `tests/test_disk_cache.py`, whose `test_every_cache_goes_through_disk_cache` fails on code that names cache folders
   itself or uses a kind not in `KINDS`).

What is **not** a cache, and is never deleted: anything outside `data/cache/`, i.e. the slides and core images, the
project tables, detections, masks and results (`data/analysis`), labels, label logs, the label pool and labelling
sessions (`data/annotations`), models, job records and the QC pictures of the steps (`data/qc`). A file that cannot be
made again from those does not belong in `data/cache/`.

The limit is per project (`cache_limit_gb` in `stainid.yaml`, set in Settings → Advanced → *Space for pictures the app
makes*), else the server's `stainid serve --cache-limit SIZE` (which steps started by that server inherit through
`$STAINID_CACHE_LIMIT`), else 100 GB; a project made from a dataset package starts with 20 GB. It is per project
rather than per server because each project's caches sit next to its data, often on different drives, and every
process that writes them (servers and steps) can enforce it from the project alone. Files used or written in the last
10 minutes are never deleted.

The drive comes before the limits: several projects on one drive may together be allowed more than it holds. Every
drive keeps a floor of free space (`stainid serve --cache-min-free`, inherited as `$STAINID_CACHE_MIN_FREE`: `50G`,
`10%`, or by default `50G or 10%`, the smaller of the two). When a drive has less, `disk_cache.free_space` (run by the
background thread every 15 s, and at once when a writer sees the drive low) deletes the least recently used cache
files of every project on that drive the process knows of (the server tells it every project on its list,
`disk_cache.know`), across projects, until the drive is back above the floor. Settings shows the drive's free space.

## Architecture

```
 browser: web app (webapp/, built into stainid/api/static), a project's pages at /p/<project>/...
    │  JSON over HTTP, /api/projects/<project>/...  (cookies; or a token when the app is served from another address)
    ▼
 server: FastAPI app (stainid/api) ── one ProjectState per open project (state.py): reads its files for the pages,
    │                                  queues and starts its steps (jobs.py); heavy steps share one limit (SLOTS)
    │  writes / reads
    ▼
 job store, one per project: outputs.jobs/<id>.json, <id>.log, <id>.exit  (stainid/jobs)
    ▲                                   ▲
    │ python -m stainid.jobs.run        │ registers itself
 app jobs: `stainid <step>` subprocesses      command-line runs: `stainid <step>` typed in a terminal
```

- **The web app only talks to the server.** Every request goes through `src/api.ts`: `api.get/post/...` for JSON and
  `apiUrl(path)` for what the browser loads by itself (images, downloads, links). `apiBase` there decides which server:
  the one that served the page (normal), one chosen on the sign-in screen, or `window.STAINID_SERVER` from
  `config.js`. Never write `/api/...` into an `<img src>` or `href` directly.
- **The server runs nothing itself.** Each step runs as `python -m stainid.cli --project ROOT <step> ...`, started
  through `python -m stainid.jobs.run EXIT_FILE -- ...`: the runner puts the command in its own process group, passes
  SIGTERM / SIGINT / SIGHUP on to the whole group and writes the exit code atomically when it ends. The server sets
  `STAINID_JOB_ID` for these commands. Steps survive a server restart: the restarted server adopts the running ones by
  process number and finishes them with the exit code from `<id>.exit`.
- **One job store.** `stainid.jobs` holds the `Job` record (`source`: `app` or `command line`; `log_file`) and its
  atomic JSON files. A workflow command typed in a terminal (one whose words match a step's `command`, without
  `STAINID_JOB_ID`) registers itself with its process number, arguments, options and, when its output goes to a file,
  that file as its log; it records how it ended (finished, failed with the error, or stopped by Ctrl+C / SIGTERM). The
  server rescans the store every 3 s, so these runs show in the app, count toward the heavy-step limit and make
  conflicting queued jobs wait (`wait_reason`). A run whose process is gone without a final record is marked
  interrupted; Stop in the app sends SIGTERM to its process group (or to the process alone when it does not lead
  one, e.g. one of several commands started by a script).
- **Several projects at once.** The project is part of every address: pages live under `/p/<project>/...` (the
  router's base, set in `main.tsx`, so links inside the app need no project) and `api.ts` sends every call of such a
  page to `/api/projects/<project>/...`, except the server-wide `/api/session` and `/api/projects`. That is the only
  place the client adds the project. On the server, `ProjectScope` (`stainid/api/app.py`, the outermost middleware)
  takes `/projects/<id>` out of the path and sets `state.CURRENT` for that request, so the routes are written once and
  call `state.get_state()`, which returns that project's `ProjectState` (one per open project, kept by id, each with
  its own `JobManager` over its own job store). There is no current project of the server: an address without a
  project (older web apps, old links) uses the project opened last (`state.open_project`); `state.reload_project`
  re-reads an open project after a change without making it the last. The server never changes its working folder or
  environment for a project; each job gets its project's model folders (`Project.environment`).
- **Project ids and the projects list** (`stainid/api/registry.py`). A project's id is its folder name made safe for
  addresses plus a hash of its path, so it is stable and needs no table. The list is the recent projects of
  `~/.stainid/app.json` plus the project folders in the projects folder (`--projects-dir` / the saved setting), minus
  those removed from the list. Its status is computed from a few file checks and the newest job records
  (`state.project_status`).
- **One heavy-step limit for the server.** `stainid/api/jobs.py: SLOTS` (a `HeavySlots`) is shared by every
  `JobManager`: a queue decides what to start while holding its lock, counting the heavy steps running in every open
  project, and (through `state.running_elsewhere`, rescanned every 10 s) heavy command-line runs in projects of the
  list that are not open. Order rules (`wait_reason`) stay per project.
- **Several people at once.** The server keeps no per-browser state besides the signed sign-in; every browser in a
  project sees the same job list and queue, and the project a browser shows is only in its address.

## API

The server is a FastAPI app (`stainid.api`). It serves the built front end from `stainid/api/static` and a JSON API
under `/api`. Everything except `/api/session` and `/api/annotate` needs the admin sign-in (`stainid/api/session.py`):
a signed cookie, or for a web app served from another address the token that sign-in returns, sent as
`Authorization: Bearer <token>` (or `?access_token=` on plain GETs). CORS is off unless `stainid serve --allow-origin
URL` (or `STAINID_ALLOW_ORIGINS`) names the addresses allowed. Job records and logs are in each project's
`outputs.jobs`. The projects list is kept in `~/.stainid/app.json`.

Every project route below is answered for one project at `/api/projects/<id>/...` (e.g.
`/api/projects/demo-project-024fe3/summary`), and for the project opened last at the plain `/api/...` address. Main
routes:

| Route | |
|---|---|
| `GET /api/projects`; `POST /api/projects/open`, `/api/projects/create` (`copy_from`, `settings`); `DELETE /api/projects/{id}`; `PUT /api/projects/settings` | the projects list (server-wide) |
| `GET /api/project`; `PUT /api/project/config` | project settings (`POST /api/project/open`, `/create`: the previous release's switch, kept) |
| `GET /api/steps` | every step: needs / makes (with paths and existence), options, progress, latest job |
| `GET/POST /api/jobs`; `DELETE /api/jobs/{id}`; `GET /api/jobs/{id}/log` | run, stop and follow steps |
| `GET/PUT /api/slides`; `GET /api/slides/picture?path=`; `GET /api/grids`, `/api/grids/{name}` | slides table (with what came from file names, `stainid/slides/table.py: read_name`); a picture of each slide (`stainid/slides/overview.py`); core-grid check images |
| `GET /api/templates/{tma_layout,donor_metadata}.csv`; `POST /api/upload/{kind}` | CSV templates and validated uploads |
| `GET /api/fs`; `POST /api/reveal`; `GET /api/download` | folder picker, Show in Finder, file download (project folder only) |
| `GET /api/summary`, `/api/calibration`, `/api/cores`, `/api/cores/{id}`, `/api/tmas/{tma}/layout` | cohort browsing |
| `GET /api/fields/{tile}/objects`, `/api/fields/{tile}/outlines`; `GET /api/images/...` | detections, outlines, images and overlays |
| `GET/POST /api/reviews`; `/api/reviews/{name}/items`, `/labels`, `/summary`, `/image/{id}.jpg` | labelling sets |
| `POST /api/labels/sessions` (with `round` 1 or 2 for an annotation round), `/open`, `/fields`; `DELETE /api/labels/sessions/{name}`; `GET /api/labels/rounds/scope`, `/api/labels/share`, `/api/labels/export.csv` | labelling sessions and annotation rounds (`routes/annotate.py`, `routes/rounds.py`) |
| `GET /api/annotate`, `/api/annotate/{name}`, `/next`; `POST /api/annotate/{name}/live`, `/tiles/{tile}/save`, `/fields/{tile}/segment`; `GET/POST /api/tutorial` (the server's one tutorial, `stainid.tutorial`, whatever the project: its field, credit line and hint, and whether the person did it), `GET/POST /api/session/tutorials` (both tutorials, kept with the account) | the annotators' pages: the next field, who holds which field, saving a field at once, one-click outlines, the tutorial |
| `GET /api/labels/classes`, `PUT /api/labels/classes/{stain}`; `GET/PUT /api/project/cache`, `POST /api/project/cache/clear` | label classes; the space for pictures made for viewing |
| `GET /api/models`, `/api/models/trained`; `POST /api/models/activate` | models |
| `GET /api/tables`, `/api/tables/{name}/columns`, `/rows`, `/feature` | output tables and column meanings |

## Conventions

- Errors are not caught to be hidden; a failing step fails loudly and the job shows its log.
- Tables are CSV, written atomically (`stainid.tables.write_csv` / `write_records`).
- Paths in tables are relative to the project root.
