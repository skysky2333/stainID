from __future__ import annotations

from stainid.api.jobs import to_argv
from stainid.project import DEFAULTS, load_project, write_default_config


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
    assert to_argv("aggregate_fields", {"level": "core_id"}) == ["aggregate", "fields", "--level", "core_id"]
