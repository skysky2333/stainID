"""A cold server on a large public project: label matching, the excluded-area pass and the Labels overview do not read
every core's files again and again (D3), and the pending-work estimate is honest when work takes longer than usual."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from stainid import candidates, exclusions
from stainid.api import state, work
from stainid.labels import candidate_store, overview


def _cores(project, n: int) -> list[str]:
    names = []
    for i in range(n):
        core = f"LIP-9_Z-{i}"
        frame = pd.DataFrame({"candidate_id": [candidate_store.candidate_id("6E10", core, 100.0 + 10 * j, 200.0) for j in range(3)],
                              "x_core": [100.0 + 10 * j for j in range(3)], "y_core": [200.0] * 3, "tile_id": [""] * 3,
                              "decision": ["accepted", "rejected", "accepted"], "class": ["plaque", "", "plaque"], "subtype": ["", "", "compact"]})
        candidates.write_table(candidate_store.folder(project, "6E10") / f"{core}.csv.gz", frame)
        names.append(core)
    return names


@pytest.fixture()
def project(synthetic_project, monkeypatch):
    monkeypatch.chdir(synthetic_project)
    state.reset()
    candidate_store._LIGHT.clear()
    candidate_store._LIGHT_SIZE[0] = 0
    candidate_store._INDEXED.clear()
    yield state.get_state().project
    state.reset()


def _forget(project):
    candidate_store._LIGHT.clear()
    candidate_store._LIGHT_SIZE[0] = 0
    candidate_store._INDEXED.clear()


def test_every_core_of_a_cohort_stays_known_and_a_cold_server_reads_them_from_the_index(project, monkeypatch):
    cores = _cores(project, 30)
    monkeypatch.setattr(candidate_store, "SAVE_AFTER", 5)
    found = candidate_store._lights(project, "6E10", cores)
    assert all(found[c] is not None and len(found[c]["ids"]) == 3 for c in cores)
    assert len(candidate_store._LIGHT) == 30  # bounded by memory, not by a count smaller than a cohort
    assert candidate_store._index_file(project, "6E10").exists()

    _forget(project)  # a new server: nothing in memory
    reads = []
    real = candidates.read_core
    monkeypatch.setattr(candidates, "read_core", lambda *a, **k: reads.append(a[2]) or real(*a, **k))
    again = candidate_store._lights(project, "6E10", cores)
    assert reads == []  # every core from the index on disk
    assert [again[c]["ids"].tolist() for c in cores] == [found[c]["ids"].tolist() for c in cores]
    frame = candidate_store.core_calls(project, "6E10", cores[:2])
    assert len(frame) == 6 and set(frame.decision) == {"accepted", "rejected"} and "compact" in set(frame.subtype)

    time.sleep(0.01)  # one core's store changes: only it is read again
    changed = pd.DataFrame({"candidate_id": [candidate_store.candidate_id("6E10", cores[3], 5.0, 5.0)], "x_core": [5.0], "y_core": [5.0],
                            "tile_id": [""], "decision": ["accepted"], "class": ["plaque"], "subtype": [""]})
    candidates.write_table(candidate_store.folder(project, "6E10") / f"{cores[3]}.csv.gz", changed)
    again = candidate_store._lights(project, "6E10", cores)
    assert reads == [cores[3]] and len(again[cores[3]]["ids"]) == 1


def test_excluded_labels_are_worked_out_once_for_the_same_labels_and_inputs(project, monkeypatch):
    labels = pd.DataFrame({"id": ["a", "b"], "source": ["Ann", "Ann"], "stain": ["6E10", "6E10"], "task": ["class", "class"],
                           "tile_id": ["LIP-1_B-2_6E10_S01"] * 2, "x": [10.0, 20.0], "y": [10.0, 20.0], "candidate": ["", ""], "time": [1.0, 2.0],
                           "value": ["plaque", "plaque"]})
    calls = []
    real = exclusions._to_drop
    monkeypatch.setattr(exclusions, "_to_drop", lambda *a: calls.append(1) or real(*a))
    first = exclusions.drop_excluded(project, labels)
    second = exclusions.drop_excluded(project, labels)
    assert len(calls) == 1 and first.equals(second)
    exclusions.drop_excluded(project, labels.assign(x=[11.0, 20.0]))  # other labels: worked out again
    assert len(calls) == 2


def test_the_labels_overview_outlives_a_restart(project, monkeypatch):
    root = project.output("reviews")
    tiles = state.get_state().tiles
    made = overview.cached_stain_overview(project, "6E10", root, tiles)
    overview._CACHE.clear()  # a new server
    monkeypatch.setattr(overview, "stain_overview", lambda *a, **k: pytest.fail("worked out again although nothing changed"))
    assert overview.cached_stain_overview(project, "6E10", root, tiles) == made


def test_work_taking_longer_than_usual_says_so_instead_of_half_a_second(monkeypatch):
    monkeypatch.setitem(work._DURATIONS, "test overdue", work.deque([2.0], maxlen=20))
    task = work.Task("test overdue", ("test overdue", 1), lambda t: None)
    task.state, task.started = "running", time.monotonic() - 60  # a minute in, for work that usually takes 2 s
    status = work.status(task)
    assert status["overdue"] is True and status["eta_s"] >= 30
    task.started = time.monotonic() - 1
    assert work.status(task)["overdue"] is False and 0.5 <= work.status(task)["eta_s"] <= 1.5


def test_durations_of_earlier_servers_give_the_first_estimate(tmp_path, monkeypatch):
    monkeypatch.setattr(work, "durations_file", lambda: tmp_path / "work_durations.json")
    (tmp_path / "work_durations.json").write_text('{"labels overview": [40.0, 50.0, 60.0]}')
    monkeypatch.delitem(work._DURATIONS, "labels overview", raising=False)
    assert work.typical("labels overview") == 50.0
    assert np.isclose(work.typical("never seen"), work.GUESS_SECONDS)
