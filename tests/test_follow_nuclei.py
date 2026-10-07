import pytest

from stainid import workflows
from stainid.tables import file_names, write_csv


class FakeProject:
    def __init__(self, tmp_path):
        self.manifest, self.nuclei = tmp_path / "tiles.csv", tmp_path / "nuclei"
        self.nuclei.mkdir()
        write_csv(self.manifest, [{"tile_id": f"{core}_{stain}_{i}", "core_id": core, "stain": stain, "selection_order": i}
                                  for core in ("A", "B") for stain in ("NeuN", "AT8") for i in (1, 2)])

    def analysis_tiles(self):
        return self.manifest

    def output(self, key):
        return self.nuclei

    def finish(self, core, stain):
        for i in (1, 2):
            (self.nuclei / f"{core}_{stain}_{i}.npz").touch()


def test_file_names_lists_a_folder_once_and_tolerates_a_missing_one(tmp_path):
    (tmp_path / "a.npz").touch()
    assert file_names(tmp_path) == {"a.npz"} and file_names(tmp_path / "missing") == set()


def test_follow_runs_each_core_image_once_as_its_nuclei_appear(tmp_path, monkeypatch):
    project, calls = FakeProject(tmp_path), []
    project.finish("A", "NeuN")
    (project.nuclei / "B_NeuN_1.npz").touch()  # half a core image is not ready
    later = iter([("B", "NeuN"), ("A", "AT8"), ("B", "AT8")])
    monkeypatch.setattr(workflows.time, "sleep", lambda _: project.finish(*next(later)))
    workflows.follow_nuclei(project, {"NeuN", "AT8"}, lambda stain, cores: calls.append((stain, cores)))
    assert calls == [("NeuN", {"A"}), ("NeuN", {"B"}), ("AT8", {"A"}), ("AT8", {"B"})]


def test_follow_stops_when_find_nuclei_stalls(tmp_path, monkeypatch):
    project, clock = FakeProject(tmp_path), iter(range(0, 10**6, 600))
    monkeypatch.setattr(workflows.time, "time", lambda: next(clock))
    monkeypatch.setattr(workflows.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="lack nuclei"):
        workflows.follow_nuclei(project, {"NeuN"}, lambda *_: None, patience_s=3600)
