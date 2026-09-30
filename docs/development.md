# Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,app,slides]"      # add ",deep" to run Cellpose / SAM / Phikon code paths
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
run `npm run build` and commit the result whenever the front end changes.

React 19 + TypeScript + Vite, with no UI framework:

| path | |
|---|---|
| `src/index.css` | design tokens (light / dark) |
| `src/api.ts` | typed API client |
| `src/hooks.ts` | `useFetch`, `useStored` |
| `src/project.tsx` | project context: group labels / colours, TMA names |
| `src/glossary.ts` | glossary behind hover definitions and the Help page |
| `src/components.tsx`, `src/charts.tsx` | shared pieces (help boxes, folder picker, path lines, option fields, charts) |
| `src/pages/` | one file per page |

## Adding a workflow step

1. Implement it in `stainid/workflows/<step>.py` as a function of a `Project`. Print one
   `[i/n] …` line per unit of work so the job runner can show progress, skip finished outputs
   so the step is resumable, and raise `ValueError` with a plain-language message for user errors.
   The web app shows that message on the step card.
2. Add a subcommand in `stainid/cli.py`.
3. Add a `Step` to `STEPS` in `stainid/workflows/steps.py`. It covers the title, a one-line summary and details for
   non-coders, `needs` / `produces` resources, options (with help text), `heavy`, a status function, and `after`: the
   steps it must wait for. The job runner, the pipeline map and the Workflow page pick it up automatically.

## Conventions

- Errors are not caught to be hidden; a failing step fails loudly and the job shows its log.
- Tables are CSV, written atomically (`stainid.tables.write_csv` / `write_records`).
- Paths in tables are relative to the project root.
