"""The pictures the app makes for viewing (data/cache/) stay within a size limit: least recently used first, never
anything in use or outside the cache folders, and whatever is cleaned up is made again when asked for."""
from __future__ import annotations

import ast
import fcntl
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from stainid import disk_cache

SRC = Path(__file__).resolve().parents[1] / "src" / "stainid"
HOUR = 3600


def _file(root: Path, relative: str, size: int = 1000, age: float = 0) -> Path:
    path = root / "data" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    if age:
        when = time.time() - age
        os.utime(path, (when, when))
    return path


def _limit(root: Path, gb: float | None) -> None:
    (root / "stainid.yaml").write_text("name: test\n" + (f"cache_limit_gb: {gb:.12f}\n" if gb is not None else ""))


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    _limit(tmp_path, None)
    return tmp_path


def _kinds(root: Path) -> dict[str, tuple[int, int]]:
    return {k["kind"]: (k["bytes"], k["files"]) for k in disk_cache.usage(root)["kinds"]}


def test_a_scan_counts_files_made_before_and_cleans_up_least_recently_used_whole_units_first(project):
    pack = _file(project, "cache/core_pyramids/C1_6E10.tiles", 5000, age=3 * HOUR)
    index = _file(project, "cache/core_pyramids/C1_6E10.json", 100, age=3 * HOUR)
    thumb = _file(project, "cache/thumbs/C1_6E10_base.jpg", 1000, age=2 * HOUR)
    crop = _file(project, "cache/label_crops/T_10_20_96.jpg", 1000, age=HOUR)
    crop_outline = _file(project, "cache/label_crops/T_10_20_96.outline.json", 50, age=HOUR)
    writing = _file(project, "cache/thumbs/C2_6E10_base.jpg.12-34.tmp", 7000)  # being written right now
    crashed = _file(project, "cache/thumbs/C3_6E10_base.jpg.12-34.tmp", 7000, age=2 * 86400)
    labels = _file(project, "annotations/labels.jsonl", 9000, age=5 * HOUR)  # study data: never a cache
    stray = _file(project, "cache/notes.txt", 9000, age=5 * HOUR)  # not in a kind folder

    disk_cache.scan(project)
    assert not crashed.exists() and writing.exists()
    assert _kinds(project) == {"core_pyramids": (5100, 2), "thumbs": (1000, 1), "label_crops": (1050, 2)}
    usage = disk_cache.usage(project)
    assert usage["used_bytes"] == 7150 and usage["limit_bytes"] == 100 * disk_cache.GB and usage["limit_gb"] is None and usage["counted"]

    disk_cache.used(thumb)  # someone looks at the thumbnail: it is the most recently used now
    disk_cache.flush()
    assert disk_cache.enforce(project) is None  # under the limit: nothing to do
    _limit(project, 1500 / disk_cache.GB)  # 1500 bytes: down to 1350
    result = disk_cache.enforce(project)
    assert result["files"] == 4 and result["bytes"] == 6150
    assert not pack.exists() and not index.exists()  # the pyramid went first, pack and index together
    assert not crop.exists() and not crop_outline.exists()  # then the crop with its outline
    assert thumb.exists() and writing.exists() and labels.exists() and stray.exists()
    assert _kinds(project) == {"thumbs": (1000, 1)}
    assert disk_cache.usage(project)["last_cleanup"]["files"] == 4


def test_what_was_used_or_made_in_the_last_minutes_is_kept_even_over_the_limit(project):
    recent = _file(project, "cache/fields/T1.jpg", 5000)
    rewritten = _file(project, "cache/fields/T2.jpg", 5000, age=5 * HOUR)
    disk_cache.scan(project)
    rewritten.write_bytes(b"y" * 5000)  # made again by a process whose note is not in yet
    _limit(project, 1e-9)
    assert disk_cache.enforce(project)["files"] == 0
    assert recent.exists() and rewritten.exists()


def test_writes_and_reads_are_noted_cheaply_and_reach_the_index_in_one_go(project):
    target = disk_cache.path(project, "overlays", "T1_dab.png")
    disk_cache.write(target, b"png" * 100)
    assert target.read_bytes() == b"png" * 100 and not list(target.parent.glob("*.tmp"))
    disk_cache.used(target, project / "data" / "tables" / "results.csv")  # not a cache: ignored
    disk_cache.flush()
    assert _kinds(project) == {"overlays": (300, 1)}
    with pytest.raises(ValueError):
        disk_cache.path(project, "../annotations", "x.jpg")
    with pytest.raises(ValueError):
        disk_cache.path(project, "thumbs", "../../x.jpg")


def test_nothing_outside_the_cache_folders_is_ever_deleted(project):
    outside = _file(project, "annotations/pool.csv", 100, age=9 * HOUR)
    real = _file(project, "analysis/results.csv", 100, age=9 * HOUR)
    link = project / "data" / "cache" / "thumbs" / "link.jpg"
    link.parent.mkdir(parents=True)
    link.symlink_to(real)
    disk_cache.scan(project)
    root = str(disk_cache.cache_root(project))
    old = time.time() - 9 * HOUR
    with disk_cache._index(root) as db:  # rows no correct code writes: a damaged or tampered index
        db.executemany("INSERT INTO files VALUES (?, ?, ?, ?, ?)", [
            ("../annotations/pool.csv", "..", "../annotations/pool", 100, old),
            ("thumbs/../../analysis/results.csv", "thumbs", "thumbs/../../analysis/results", 100, old),
            ("thumbs/link.jpg", "thumbs", "thumbs/link", 100, old),
        ])
    disk_cache.evict(project, 0, keep_seconds=0)
    assert outside.exists() and real.exists()
    assert not link.exists()  # the link itself was the cache file; what it pointed to stays


def test_one_cleanup_at_a_time_per_project(project):
    _file(project, "cache/thumbs/A_base.jpg", 1000, age=HOUR)
    disk_cache.scan(project)
    with open(disk_cache.cache_root(project) / disk_cache.LOCK_NAME, "a+") as other:  # another process cleaning
        fcntl.flock(other, fcntl.LOCK_EX)
        assert disk_cache.evict(project, 0, keep_seconds=0)["skipped"]
        fcntl.flock(other, fcntl.LOCK_UN)
    assert disk_cache.evict(project, 0, keep_seconds=0)["files"] == 1


def test_a_step_process_records_what_it_writes_before_it_ends(project):
    code = ("import sys; from stainid import disk_cache; "
            "disk_cache.write(disk_cache.path(sys.argv[1], 'thumbs', 'C9_NeuN_base.jpg'), b'j' * 1234)")
    env = {**os.environ, "PYTHONPATH": str(SRC.parent)}
    subprocess.run([sys.executable, "-c", code, str(project)], check=True, env=env, timeout=60)
    assert _kinds(project) == {"thumbs": (1234, 1)}


def test_clear_all_keeps_only_what_is_in_use_right_now(project):
    old = _file(project, "cache/core_pyramids/C1_AT8.tiles", 1000, age=HOUR)
    shown = _file(project, "cache/thumbs/C1_AT8_512.jpg", 1000, age=HOUR)
    disk_cache.scan(project)
    disk_cache.used(shown)
    result = disk_cache.clear(project)
    assert result["files"] == 1 and not old.exists() and shown.exists()


def test_the_limit_comes_from_the_project_then_the_server_then_100_gb(project, monkeypatch):
    assert disk_cache.limit_bytes(project) == 100 * disk_cache.GB
    monkeypatch.setenv(disk_cache.LIMIT_ENV, "250G")
    assert disk_cache.limit_bytes(project) == 250 * disk_cache.GB
    _limit(project, 20)
    assert disk_cache.limit_bytes(project) == 20 * disk_cache.GB
    assert disk_cache.parse_size("500M") == 500 * 10**6 and disk_cache.parse_size("1.5TB") == 1.5e12 and disk_cache.parse_size("7") == 7e9
    with pytest.raises(ValueError):
        disk_cache.parse_size("lots")


def test_a_damaged_index_is_built_again(project):
    _file(project, "cache/thumbs/A_base.jpg", 1000)
    (disk_cache.cache_root(project) / disk_cache.INDEX_NAME).write_bytes(b"not a database at all" * 100)
    disk_cache.scan(project)
    assert _kinds(project) == {"thumbs": (1000, 1)}


# ---------------------------------------------------------------- the only way to make a cache

def _code_strings(tree: ast.AST) -> list[ast.Constant]:
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body
                  and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)}
    return [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]


ONLY_LISTS = {"api/queue_pictures.py": "lists label_crops/ to see which crops are made; label_crops writes and reads them"}


def test_every_cache_goes_through_disk_cache():
    """Code names cache folders only through `disk_cache.folder` / `path` (or `ProjectState.cache_dir`), with a kind
    listed in `disk_cache.KINDS`, and records what it writes and reads there."""
    problems = []
    for file in sorted(SRC.rglob("*.py")):
        if file.name == "disk_cache.py" or "static" in file.parts:
            continue
        text = file.read_text()
        tree = ast.parse(text)
        for node in _code_strings(tree):
            if node.value == "cache" or ("data/cache" in node.value and " " not in node.value):
                problems.append(f"{file.relative_to(SRC)}:{node.lineno} names the cache folder itself")
        kinds = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("cache_dir", "folder", "path") and (
                    node.func.attr == "cache_dir" or (isinstance(node.func.value, ast.Name) and node.func.value.id == "disk_cache")):
                position = 0 if node.func.attr == "cache_dir" else 1
                if len(node.args) > position and isinstance(node.args[position], ast.Constant):
                    kinds.append((node.lineno, node.args[position].value))
        for line, kind in kinds:
            if kind not in disk_cache.KINDS:
                problems.append(f"{file.relative_to(SRC)}:{line} uses cache kind {kind!r}, which is not in disk_cache.KINDS")
        name = file.relative_to(SRC).as_posix()
        if kinds and name not in ONLY_LISTS and "def cache_dir" not in text and not re.search(r"disk_cache\.(write|wrote|used)\(", text):
            problems.append(f"{name} makes cache paths but never records a write or read (disk_cache.write / wrote / used)")
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------- cleaned up while in use: made again

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from stainid.api import state  # noqa: E402
from stainid.api.app import create_app  # noqa: E402

TILE = "LIP-1_B-2_6E10_S01"


@pytest.fixture()
def client(synthetic_project, monkeypatch):
    monkeypatch.chdir(synthetic_project)
    state.reset()
    admin = TestClient(create_app(), client=("127.0.0.1", 50000))
    assert admin.post("/api/session/admin", json={"password": "admin"}).status_code == 200
    yield admin
    state.reset()


def _everything_cleaned_up(root: Path) -> None:
    """As a cleanup over the limit would, with nothing counted as in use."""
    disk_cache.flush()
    disk_cache.scan(root)
    disk_cache.evict(root, 0, keep_seconds=0)
    assert not [p for p in (root / "data" / "cache").rglob("*") if p.is_file() and not p.name.startswith(".")]


def _until(ask, ready, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        answer = ask()
        if ready(answer):
            return answer
        time.sleep(0.1)
    raise AssertionError(f"never ready: {answer}")


def test_core_viewer_pictures_are_made_again_after_a_cleanup_even_with_the_viewer_open(client, synthetic_project):
    url = "/api/cores/LIP-1_B-2/6E10/picture"
    picture = _until(lambda: client.get(url).json(), lambda p: p["ready"])
    tile = f"/api/images/cores/LIP-1_B-2/6E10/tiles/0/1_1.jpg?v={picture['version']}"
    assert client.get(tile).status_code == 200
    assert client.get("/api/images/cores/LIP-1_B-2/6E10.jpg?size=256").status_code == 200
    _everything_cleaned_up(synthetic_project)

    assert client.get(tile).status_code == 404  # the open viewer asks for the picture again on a missing tile
    again = client.get(url).json()
    assert not again["ready"] and again["state"] in ("waiting", "preparing")  # its usual "preparing", not an error
    again = _until(lambda: client.get(url).json(), lambda p: p["ready"])
    assert again["version"] == picture["version"] and client.get(tile).status_code == 200
    assert client.get("/api/images/cores/LIP-1_B-2/6E10.jpg?size=256").status_code == 200
    assert (synthetic_project / "data/cache/core_pyramids/LIP-1_B-2_6E10.tiles").exists()


def test_tile_pictures_overlays_and_outlines_are_made_again(client, synthetic_project):
    urls = [f"/api/images/fields/{TILE}.jpg", f"/api/images/fields/{TILE}/dab.png", f"/api/images/fields/{TILE}/exclusions.png",
            f"/api/labels/fields/{TILE}/thumb.jpg"]
    first = [client.get(u) for u in urls]
    assert [r.status_code for r in first] == [200] * len(urls)
    _everything_cleaned_up(synthetic_project)
    again = [client.get(u) for u in urls]
    assert [r.status_code for r in again] == [200] * len(urls)
    assert [r.content for r in again][:3] == [r.content for r in first][:3]
    assert (synthetic_project / f"data/cache/fields/{TILE}.jpg").exists()


def test_label_crops_of_a_queue_are_made_again(client, synthetic_project):
    from stainid.labels import store

    store.append(state.get_state().project.output("reviews"), [
        {"id": f"class:{TILE}:300:320", "source": "import: test", "stain": "6E10", "task": "class", "tile_id": TILE, "x": 300.0, "y": 320.0,
         "value": "compact plaque"}])

    def listing():
        return client.get("/api/labels/sessions/reference/objects", params={"stain": "6E10"}).json()["objects"]

    def crop():
        o = listing()[0]
        return client.get(f"/api/labels/crops/6E10/{o['tile_id']}.jpg", params={"x": o["x"], "y": o["y"]})

    _until(listing, lambda objects: objects and all(o["crop"]["state"] == "ready" for o in objects))
    assert crop().status_code == 200
    _everything_cleaned_up(synthetic_project)
    assert crop().status_code in (200, 409)  # 409: being cut again from its core image, the page shows "preparing"
    _until(listing, lambda objects: all(o["crop"]["state"] == "ready" for o in objects))
    assert crop().status_code == 200


def test_settings_show_the_space_taken_and_change_the_limit(client, synthetic_project):
    assert client.get(f"/api/images/fields/{TILE}.jpg").status_code == 200
    usage = client.get("/api/project/cache").json()
    assert usage["limit_bytes"] == 100 * disk_cache.GB and usage["limit_gb"] is None
    assert {k["kind"]: k["label"] for k in usage["kinds"]}["fields"] == "Tile pictures" and usage["used_bytes"] > 0
    assert client.put("/api/project/cache", json={"limit_gb": 0.1}).status_code == 400
    usage = client.put("/api/project/cache", json={"limit_gb": 20}).json()
    assert usage["limit_gb"] == 20 and usage["limit_bytes"] == 20 * disk_cache.GB
    assert "cache_limit_gb: 20" in (synthetic_project / "stainid.yaml").read_text()
    assert client.get("/api/project").json()["config"]["cache_limit_gb"] == 20
    assert client.put("/api/project/cache", json={"limit_gb": None}).json()["limit_gb"] is None
    assert "cache_limit_gb" not in (synthetic_project / "stainid.yaml").read_text()
    assert client.post("/api/project/cache/clear").json()["busy"] == "clearing"
    _until(lambda: client.get("/api/project/cache").json(), lambda u: u["busy"] is None)


