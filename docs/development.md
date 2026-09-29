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

React 19 + TypeScript + Vite, with no UI framework:

| path | |
|---|---|
| `src/index.css` | design tokens (light / dark) |
| `src/api.ts` | typed API client |
| `src/hooks.ts` | `useFetch`, `useStored` |
| `src/components.tsx`, `src/charts.tsx` | shared pieces |
| `src/pages/` | one file per page |

## Adding a workflow step

1. Implement it in `stainid/workflows/<step>.py` as a function of a `Project`. Print one
   `[i/n] …` line per unit of work so the job runner can show progress, and skip finished
   outputs so the step is resumable.
2. Add a subcommand in `stainid/cli.py`.
3. Register it in `WORKFLOWS` in `stainid/api/jobs.py` (`heavy: True` if it is GPU/CPU-bound).
4. Add a card to `STEPS` in `webapp/src/pages/Pipelines.tsx`.

## Conventions

- Errors are not caught to be hidden; a failing step fails loudly and the job shows its log.
- Tables are CSV, written atomically (`stainid.tables.write_csv` / `write_records`).
- Paths in tables are relative to the project root.
