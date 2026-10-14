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
| `src/api.ts` | typed API client; `apiBase` / `apiUrl` decide which server it talks to |
| `public/config.js` | optional `window.STAINID_SERVER` for a copy of the app served from somewhere else |
| `src/hooks.ts` | `useFetch`, `useStored` |
| `src/project.tsx` | project context: group labels / colours, TMA names |
| `src/glossary.ts` | glossary behind hover definitions and the Help page |
| `src/components.tsx`, `src/charts.tsx` | shared pieces (help boxes, folder picker, path lines, option fields, charts) |
| `src/pages/` | one file per page |

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
   pipeline map and the Workflow page pick the step up without further changes. A new input file gets a `Resource`
   with its `download`, `template`, `setting` or `format`, which gives the missing-file buttons on the step card.

## Architecture

```
 browser: web app (webapp/, built into stainid/api/static)
    │  JSON over HTTP, /api/...  (cookies; or a token when the app is served from another address)
    ▼
 server: FastAPI app (stainid/api) ── reads project files for the pages; queues and starts steps (stainid/api/jobs.py)
    │  writes / reads
    ▼
 job store: outputs.jobs/<id>.json, <id>.log, <id>.exit  (stainid/jobs)
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
- **Several people at once.** The server keeps no per-browser state besides the signed sign-in; every browser sees the
  same project, job list and queue.

## API

The server is a FastAPI app (`stainid.api`). It serves the built front end from `stainid/api/static` and a JSON API
under `/api`. Everything except `/api/session` and `/api/annotate` needs the admin sign-in (`stainid/api/session.py`):
a signed cookie, or for a web app served from another address the token that sign-in returns, sent as
`Authorization: Bearer <token>` (or `?access_token=` on plain GETs). CORS is off unless `stainid serve --allow-origin
URL` (or `STAINID_ALLOW_ORIGINS`) names the addresses allowed. Job records and logs are in `outputs.jobs`. The
last-used project is remembered in `~/.stainid/app.json`.

Main routes:

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

## Conventions

- Errors are not caught to be hidden; a failing step fails loudly and the job shows its log.
- Tables are CSV, written atomically (`stainid.tables.write_csv` / `write_records`).
- Paths in tables are relative to the project root.
