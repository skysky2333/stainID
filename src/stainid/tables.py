from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path


def claim_field(claim: Path) -> bool:
    """Take a unit of work (a field, a core image) for this process, so any number of processes can share one queue; a
    claim left by a process that no longer runs (stopped or crashed) is taken over."""
    try:
        descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        owner = claim.read_text().strip()
        try:
            os.kill(int(owner), 0)
            return False
        except (ValueError, ProcessLookupError):
            claim.unlink(missing_ok=True)
            return claim_field(claim)
        except PermissionError:
            return False
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    return True


def file_names(folder: Path) -> set[str]:
    """Names in `folder` from one directory listing (empty if it does not exist). Use it instead of one `exists()` per
    file when checking many outputs: on exFAT and network drives each lookup of a missing name rereads the folder."""
    try:
        return {entry.name for entry in os.scandir(folder)}
    except FileNotFoundError:
        return set()


def _sync(handle) -> None:
    """Force the file's data to disk before it is renamed into place: without this a power loss or freeze can leave the
    rename on disk but the data not, i.e. an empty file under the final name that resumed runs take as finished."""
    handle.flush()
    os.fsync(handle.fileno())


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str] | None = None,
) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer columns for empty CSV: {path}")
        fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        _sync(handle)
    temporary.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    """Parallel shards write the same small files; readers must never see a half-written one."""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        handle.write(text)
        _sync(handle)
    Path(handle.name).replace(path)


def save_npz_atomic(path: Path, **arrays) -> None:
    """Compressed .npz written under a temporary name, synced and renamed into place, so a reader that sees the file
    (another step following this one, or a resumed run after a crash) never reads a half-written one."""
    import numpy as np

    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        _sync(handle)
    Path(handle.name).replace(path)


def format_csv_value(value: object) -> object:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.8f}"
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_records(path: Path, rows: list[dict[str, object]], empty_columns: list[str] | None = None) -> None:
    """Write rows with formatted values and the union of their columns (in first-seen order)."""
    columns = list(dict.fromkeys(k for row in rows for k in row)) if rows else (empty_columns or ["id"])
    write_csv(Path(path), [{k: format_csv_value(v) for k, v in row.items()} for row in rows], columns)
