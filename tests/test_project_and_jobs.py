from __future__ import annotations

from stainid.project import DEFAULTS, load_project, save_project, write_default_config
from stainid.workflows.steps import to_argv


def test_defaults_and_overrides(tmp_path):
    (tmp_path / "stainid.yaml").write_text("pixel_size_um: 0.5\noutputs:\n  neun: out/neun\n")
    project = load_project(tmp_path)
    assert project.pixel_size_um == 0.5
    assert project.output("neun") == tmp_path.resolve() / "out/neun"
    assert project.output("fields") == tmp_path.resolve() / DEFAULTS["outputs"]["fields"]


def test_write_default_config_roundtrip(tmp_path):
    write_default_config(tmp_path)
    assert load_project(tmp_path).config == DEFAULTS


def test_job_argv():
    assert to_argv("nuclei", {"stain": ["AT8", "6E10"], "device": "mps", "shard_index": 0, "reverse": False}) == [
        "nuclei", "--stain", "AT8", "--stain", "6E10", "--device", "mps", "--shard-index", "0"]
    assert to_argv("summarize", {"level": "core_id"}) == ["summarize", "--level", "core_id"]
    assert to_argv("training_set", {"stain": "AT8", "name": "r1", "enrich": False}) == ["training-set", "--stain", "AT8", "--name", "r1", "--no-enrich"]


def test_save_project_writes_only_changes(tmp_path):
    project = save_project(load_project(tmp_path), {"name": "Study", "tma": {"rows": 8}})
    assert load_project(tmp_path).config["tma"] == {"prefix": "TMA-", "rows": 8, "columns": 6}
    assert (tmp_path / "stainid.yaml").read_text() == "name: Study\ntma:\n  rows: 8\n"
    assert project.tma_name("3") == "TMA-3"
