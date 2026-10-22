from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

_OUTSIDE: list[str] = []  # projects something opened outside the temporary folders (a test, or work it left running)
_SAFE_ENV: dict[str, str] = {}


def _allowed(path: Path) -> bool:
    import tempfile

    roots = [Path(tempfile.gettempdir()).resolve()]
    factory = getattr(_CONFIG.get("config"), "_tmp_path_factory", None)
    if factory is not None:
        roots.append(factory.getbasetemp().resolve())
    return any(path == r or r in path.parents for r in roots)


_CONFIG: dict = {}


def pytest_configure(config):
    """For the whole run, before any test and for the work tests leave running in the background: no test may touch a
    real project. The run starts in an empty temporary folder (not the folder pytest was started from, which may be a
    study), without $STAINID_PROJECT and with a temporary home; a project opened anywhere but in a temporary folder
    raises at once and fails the run."""
    import os
    import shutil
    import tempfile

    from stainid import project as project_module

    _CONFIG["config"] = config
    base = Path(tempfile.mkdtemp(prefix="stainid-tests-"))
    node = Path.home() / ".local/node/bin"  # the tests that run TypeScript find Node.js here too
    if not shutil.which("node") and (node / "node").exists():
        os.environ["PATH"] = f"{node}{os.pathsep}{os.environ.get('PATH', '')}"
    for name in ("STAINID_PROJECT", "STAINID_PROJECTS_DIR", "STAINID_TUTORIAL", "STAINID_QUEUE_DIR", "STAINID_PLAQUE_CNN_DIR",
                 "STAINID_PROGRESS_FILE", "STAINID_JOB_ID"):
        os.environ.pop(name, None)
    (base / "home").mkdir()
    (base / "cwd").mkdir()
    os.environ["HOME"] = str(base / "home")
    os.chdir(base / "cwd")
    _SAFE_ENV.update(os.environ)
    from stainid.api import registry  # noqa: PLC0415 - the projects list too: never the real ~/.stainid/app.json, also between tests

    registry.SETTINGS = base / "home" / ".stainid" / "app.json"
    original = project_module.Project.__init__

    def checked_init(self, root, config):
        resolved = Path(root).expanduser().resolve()
        if not _allowed(resolved):
            _OUTSIDE.append(str(resolved))
            raise RuntimeError(f"a test opened a project outside the temporary folders: {resolved}")
        original(self, root, config)

    project_module.Project.__init__ = checked_init


def pytest_sessionfinish(session, exitstatus):
    if _OUTSIDE:
        print(f"\nPROJECTS OPENED OUTSIDE THE TEMPORARY FOLDERS: {sorted(set(_OUTSIDE))}")
        session.exitstatus = 1


@pytest.fixture(autouse=True)
def no_real_project(tmp_path_factory, monkeypatch):
    """Each test starts in an empty folder of its own with its own home, and the environment (as `pytest_configure`
    made it) is put back afterwards; a project opened outside the temporary folders during the test fails it."""
    import os

    seen = len(_OUTSIDE)
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home-dir")))
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
    yield
    for name in [n for n in os.environ if n not in _SAFE_ENV and not n.startswith("PYTEST_")]:
        del os.environ[name]  # what the code under test set (a project's model caches, `stainid serve`'s project)
    os.environ.update({n: v for n, v in _SAFE_ENV.items() if not n.startswith("PYTEST_")})
    from stainid import disk_cache

    disk_cache._KNOWN.clear()  # the projects this process knew of: the free-space cleanup must never reach another test's one
    disk_cache._WATCHED.clear()
    if len(_OUTSIDE) > seen:
        pytest.fail(f"a project was opened outside the temporary folders: {sorted(set(_OUTSIDE[seen:]))}")


@pytest.fixture(autouse=True)
def own_app_settings(no_real_project, tmp_path_factory, monkeypatch):
    """Every test gets its own app settings (projects list, password, secret) and its own job queue,
    so no test reads this computer's ~/.stainid or sees another test's jobs; open projects are closed afterwards."""
    from stainid.api import registry
    from stainid.jobs import queue

    monkeypatch.setattr(registry, "SETTINGS", tmp_path_factory.mktemp("home") / "app.json")
    monkeypatch.setenv(queue.QUEUE_ENV, str(tmp_path_factory.mktemp("queue")))  # its own job queue, run as plain processes
    monkeypatch.setenv(queue.LAUNCHER_ENV, "process")
    monkeypatch.delenv(registry.PROJECTS_DIR_ENV, raising=False)
    monkeypatch.setenv("STAINID_TUTORIAL", str(tmp_path_factory.mktemp("tutorial") / "tutorial"))  # not built: a test builds its own
    yield
    queue.reset()
    from stainid.labels import sessions

    sessions.ON_CREATE[:] = [hook for hook in sessions.ON_CREATE if not getattr(hook, "test_only", False)]
    try:
        from stainid.api import accounts, state
    except ImportError:  # pragma: no cover - without the app extra
        return
    accounts.forget()
    state.reset()


@pytest.fixture()
def synthetic_project(tmp_path: Path, monkeypatch) -> Path:
    """A one-core, three-stain project with tiny images and pre-made pipeline outputs."""
    rng = np.random.default_rng(0)
    core_dir = tmp_path / "data" / "cores" / "LIP-1" / "B-2"
    core_dir.mkdir(parents=True)
    tiles, images = [], []
    for stain in ("NeuN", "6E10", "AT8"):
        rgb = np.full((1200, 1200, 3), 225, np.uint8)
        for x, y in rng.integers(200, 1000, size=(25, 2)):
            cv2.circle(rgb, (int(x), int(y)), 12, (120, 80, 40), -1)
        cv2.imwrite(str(core_dir / f"{stain}.png"), rgb)
        images.append({"core_id": "LIP-1_B-2", "tma": 1, "stain": stain, "image_path": f"data/cores/LIP-1/B-2/{stain}.png",
                       "native_width_px": 1200, "native_height_px": 1200, "tissue_fraction": 0.9, "tissue_status": "complete"})
        tiles.append({"tile_id": f"LIP-1_B-2_{stain}_S01", "core_id": "LIP-1_B-2", "tma": 1, "core_label": "B-2", "donor_id": "0001",
                      "sample_region_id": "BRC_0001_FR", "region": "frontal", "disease_group": "AD", "technical_replicate": 1, "stain": stain,
                      "image_path": f"data/cores/LIP-1/B-2/{stain}.png", "pixel_width_um": 0.2738, "pixel_height_um": 0.2738,
                      "selection_role": "whole_core_grid", "selection_order": 1, "nested_sample": "whole_core", "x_px": 300, "y_px": 300,
                      "width_px": 512, "height_px": 512, "preview_tissue_fraction": 0.9})
    analysis = tmp_path / "data" / "analysis"
    analysis.mkdir(parents=True)
    pd.DataFrame(tiles).to_csv(analysis / "cohort_whole_core_tiles.csv", index=False)
    pd.DataFrame(images).to_csv(analysis / "core_images.csv", index=False)
    pd.DataFrame([{"tma": 1, "stain": s, "threshold_dab_od": 0.05} for s in ("NeuN", "6E10", "AT8")]).to_csv(analysis / "slide_dab_calibration.csv", index=False)
    pd.DataFrame([{"tma": 1, "core_label": "B-2", "donor_id": "0001", "region": "frontal", "disease_group": "AD"}]).to_csv(tmp_path / "data" / "tma_layout.csv", index=False)
    pd.DataFrame([{"donor_id": "0001", "tma": 1, "disease_group": "AD"}]).to_csv(tmp_path / "data" / "donor_metadata.csv", index=False)
    parts = analysis / "cohort_v3" / "parts"
    parts.mkdir(parents=True)
    common = {"tile_id": "LIP-1_B-2_6E10_S01", "core_id": "LIP-1_B-2", "tma": 1, "donor_id": "0001", "sample_region_id": "BRC_0001_FR",
              "region": "frontal", "disease_group": "AD", "technical_replicate": 1, "stain": "6E10", "selection_order": 1}
    plaques = [{**common, "centroid_x_px": 100 + 30 * i, "centroid_y_px": 200, "plaque_class": ["compact", "diffuse"][i % 2], "plaque_probability": 0.9,
                "plaque_area_um2": 300.0, "dense_core_fraction": 0.2, "equivalent_diameter_um": 20.0} for i in range(8)]
    pd.DataFrame(plaques).to_csv(parts / "LIP-1_B-2_6E10_objects.csv", index=False)
    (tmp_path / "data" / "annotations").mkdir()
    (tmp_path / "stainid.yaml").write_text(yaml.safe_dump({"name": "synthetic"}))
    monkeypatch.setenv("STAINID_PROJECT", str(tmp_path))
    return tmp_path
