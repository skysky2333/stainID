"""Rewrite the v2 store's csv.gz tables as parquet (needs pyarrow), in parallel over core images.

    python tools/v2_to_parquet.py [--project .] [--workers 12] [--remove-csv]

Without --remove-csv the csv.gz files are kept next to their parquet twins (readers already prefer the parquet);
run again with --remove-csv to delete them (a csv.gz is removed only when its parquet twin reads back the same rows).
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

KINDS = ("candidates", "proposals", "outlines", "calls", "tiles", "unproposed")


def convert(job: tuple[str, bool]) -> tuple[str, str]:
    from stainid import candidates as store

    name, remove = job
    file = Path(name)
    target = store.twin(file)
    try:
        if not target.exists():
            frame = store.read_table(file)
            store.write_table(file, frame, keep_csv=True)
        if remove:
            import pandas as pd

            if len(pd.read_parquet(target, columns=[])) != len(pd.read_csv(file, usecols=[0])):
                return name, "row counts differ: kept the csv.gz"
            file.unlink()
        return name, "ok"
    except Exception as error:  # noqa: BLE001 - reported, the next run retries
        return name, f"failed: {error}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.environ.get("STAINID_PROJECT", "."))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--remove-csv", action="store_true")
    args = parser.parse_args()
    from stainid import candidates as store
    from stainid.project import load_project

    if not store.PARQUET:
        raise SystemExit("pyarrow is not installed")
    root = store.root(load_project(args.project))
    folders = [root / k / s for k in KINDS for s in ("NeuN", "6E10", "AT8")] + [root / "threads"]
    files = [str(f / n) for f in folders if f.is_dir() for n in sorted(os.listdir(f)) if n.endswith(".csv.gz") and not n.startswith(".")]
    print(f"{len(files)} csv.gz tables under {root}", flush=True)
    done = failed = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for name, status in pool.map(convert, [(f, args.remove_csv) for f in files], chunksize=8):
            done += 1
            if status != "ok":
                failed += 1
                print(name, status, flush=True)
            if done % 500 == 0:
                print(f"{done}/{len(files)}", flush=True)
    print(f"done: {done - failed} ok, {failed} not", flush=True)


if __name__ == "__main__":
    main()
