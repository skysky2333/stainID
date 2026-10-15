from __future__ import annotations

import os
import time

from stainid import cache


def _age(path, seconds: float = 100.0) -> None:
    """Date a file back, as if written long ago (so its signature is trusted)."""
    when = time.time() - seconds
    os.utime(path, (when, when))


def test_rows_are_read_again_when_the_file_changes(tmp_path):
    table = tmp_path / "t.csv"
    table.write_text("a,b\n1,2\n")
    _age(table)
    assert cache.rows(table) == [{"a": "1", "b": "2"}]
    table.write_text("a,b\n1,2\n3,4\n")
    assert len(cache.rows(table)) == 2
    assert cache.rows(table, ("b", "missing")) == [{"b": "2", "missing": ""}, {"b": "4", "missing": ""}]
    table.unlink()
    assert cache.rows(table) == []


def test_kept_while_nothing_changes(tmp_path):
    table = tmp_path / "t.csv"
    table.write_text("a\n1\n")
    _age(table)
    calls = []
    make = lambda: calls.append(1) or len(calls)  # noqa: E731
    assert cache.memo(("test", str(table)), [table], make) == 1
    assert cache.memo(("test", str(table)), [table], make) == 1
    assert len(calls) == 1


def test_a_change_in_the_same_instant_is_not_missed(tmp_path):
    """File systems keep coarse times: a value made right after a change is made again next time, even when the file
    has been rewritten with the same size and the same time."""
    table = tmp_path / "t.csv"
    table.write_text("a\n1\n")
    stamp = os.stat(table).st_mtime_ns
    assert cache.rows(table) == [{"a": "1"}]
    table.write_text("a\n2\n")
    os.utime(table, ns=(stamp, stamp))
    assert cache.rows(table) == [{"a": "2"}]


def test_folder_names_follow_new_and_removed_files(tmp_path):
    folder = tmp_path / "out"
    assert cache.names(folder) == frozenset()
    folder.mkdir()
    (folder / "a.npz").write_text("")
    assert cache.names(folder) == {"a.npz"}
    (folder / "b.npz").write_text("")
    (folder / "a.npz").unlink()
    assert cache.names(folder) == {"b.npz"}


def test_step_progress_follows_new_outputs(synthetic_project):
    from stainid.project import load_project
    from stainid.workflows.steps import BY_ID

    project = load_project(synthetic_project)
    before = BY_ID["nuclei"].status(project)
    tiles = cache.rows(project.analysis_tiles())
    assert before["total"] == len(tiles) and before["done"] == 0
    folder = project.output("nuclei")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{tiles[0]['tile_id']}.npz").write_bytes(b"")
    assert BY_ID["nuclei"].status(project)["done"] == 1
