"""The guided setup: everything needed to go from slide scans and the lab's own spreadsheets to a project that runs.

- `files`: read Excel workbooks, Word tables, CSV and pasted text into plain tables of cell texts;
- `naming`: shared rules for donor ids, regions, diagnostic groups and neuropathology scores;
- `maps`: TMA maps (who sits on each core) into the project's `tma_layout`;
- `donors`: the case spreadsheet (one row per donor) into `donor_metadata`;
- `checks`: cross-checks between the map, the donor table and the slides, with fixes;
- `guide`: how far each screen of the guided setup has got, and a rough run-time estimate.

The web app's guided setup (`stainid.api.routes.setup`) only calls these functions.
"""
