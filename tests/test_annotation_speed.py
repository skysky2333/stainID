"""The fast paths of the annotation pages give what the slow ones gave: a field's candidates read for that field only
(filtered while the parquet store is read, outlines parsed with numpy) equal the whole core read and filtered, and the
decisions and the consensus counted for every object at once equal the object-by-object loops they replace. No timing
is checked: only that the answers are the same, on a synthetic store with every kind of cell (several rings, missing
masks, ties between people)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from stainid import candidates
from stainid.api import state, work
from stainid.labels import candidate_store, decisions, store
from stainid.labels import vocabulary as vocab

TILE = "LIP-1_B-2_6E10_S01"  # x_px = y_px = 300 in a 1200 px core image, field margin 128 px
CORE = "LIP-1_B-2"


# ---------- the code these replace (as it was) ----------

def _old_ring(value) -> np.ndarray:
    points = np.asarray(value, dtype=float) if len(value) else np.zeros((0, 2))
    if points.ndim == 1:
        points = points[: len(points) // 2 * 2].reshape(-1, 2)
    return points.reshape(-1, 2)


def _old_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return abs(float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum())) / 2


def _old_polygon(value):
    if candidate_store._missing(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text[0] in "[{":
            try:
                value = json.loads(text)
            except ValueError:
                return None
        else:
            rings = [_old_ring([float(v) for v in part.replace(",", " ").split()]) for part in text.split(";") if part.strip()]
            value = max(rings, key=_old_area).tolist() if rings else []
    ring = _old_ring(value)
    return ring.round(1).tolist() if len(ring) >= 3 else None


def _old_polygons(value):
    if candidate_store._missing(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text[0] == "[":
            try:
                value = json.loads(text)
            except ValueError:
                return None
        else:
            value = text.split("|")
    found = [_old_polygon(v) for v in value]
    return found if any(found) else None


def _old_in_field(project, stain, tile, context_px, level):
    found = candidate_store.read_core(project, stain, str(tile["core_id"]))
    x0, y0 = max(0, int(tile["x_px"]) - context_px), max(0, int(tile["y_px"]) - context_px)
    x1 = int(tile["x_px"]) + int(tile["width_px"]) + context_px
    y1 = int(tile["y_px"]) + int(tile["height_px"]) + context_px
    near = candidate_store.at_level(found[(found.x_core >= x0) & (found.x_core < x1) & (found.y_core >= y0) & (found.y_core < y1)], level)
    shift = lambda poly: None if not poly else [[px - x0, py - y0] for px, py in poly]  # noqa: E731
    return near.assign(x=near.x_core - x0, y=near.y_core - y0, own=near.tile_id == str(tile["tile_id"]),
                       outline=[shift(p) for p in near.outline], sam_outline=[shift(p) for p in near.sam_outline],
                       sam_masks=[None if not m else [shift(p) for p in m] for m in near.sam_masks]).reset_index(drop=True)


def _latest(group: pd.DataFrame) -> pd.Series:
    """The latest row of a group: by time, then the order of the rows."""
    time = pd.to_numeric(group.time, errors="coerce").fillna(0.0) if "time" in group else pd.Series(0.0, index=group.index)
    return group.assign(_t=time.to_numpy(), _q=np.arange(len(group))).sort_values(["_t", "_q"], kind="stable").iloc[-1].drop(["_t", "_q"])


def _call(group: pd.DataFrame) -> tuple[str, str, str, str]:
    """(majority identity, its majority subtype, latest identity, latest subtype) of one object's rows that say something."""
    top, _ = store.majority(group.identity.astype(str))
    sub = group.subtype.fillna("").astype(str)
    typed = sub[(group.identity.astype(str) == top) & (sub != "")] if top else sub.iloc[:0]
    last = _latest(group)
    return top, store.majority(typed)[0], str(last.identity), "" if pd.isna(last.subtype) else str(last.subtype)


def _old_state_of(rows: pd.DataFrame) -> decisions.State:
    """`decisions._state_of` candidate by candidate: the majority call (`store.tally`), the latest said when split."""
    out = decisions.State(rows=rows)
    if rows.empty:
        return out
    out.answered = {c: set(g) for c, g in rows.groupby("candidate").source}
    said = rows[rows.role != vocab.SKIP].sort_values("time", kind="stable")
    for candidate, group in said.groupby("candidate", sort=False):
        out.decided[candidate] = list(dict.fromkeys(group.source))
        top, subtype, latest, latest_subtype = _call(group)
        out.calls[candidate] = (top, subtype) if top else (latest, latest_subtype)
        out.times[candidate] = float(group.time.max())
    return out


def _old_consensus(labels: pd.DataFrame, stain: str, vocabulary=None) -> pd.DataFrame:
    """`store.consensus` object by object: keyed by candidate, else field and rounded position; each person's latest label."""
    rows = store.classes(labels, stain, vocabulary)
    if rows.empty:
        return pd.DataFrame(columns=store.CONSENSUS)
    named = rows.candidate.fillna("").astype(str) if "candidate" in rows else pd.Series("", index=rows.index)
    keys = [n or f"at:{t}:{round(float(x))}:{round(float(y))}" for n, t, x, y in zip(named, rows.tile_id, rows.x, rows.y)]
    rows = rows.assign(object=keys, _p=np.arange(len(rows)))
    rows = rows.reset_index(drop=True)
    rows = rows.loc[[_latest(g).name for _, g in rows.groupby(["source", "object"], sort=False)]].sort_values("_p")
    rows = rows[rows.role != vocab.SKIP]
    out = []
    for key, group in rows.groupby("object", sort=False):
        top, agreement = store.majority(group.identity.astype(str))
        _, subtype, latest, latest_subtype = _call(group)
        votes = (group.role == vocab.POSITIVE).to_numpy()
        time = pd.to_numeric(group.time, errors="coerce").fillna(0.0) if "time" in group else pd.Series([0.0])
        out.append({"object": key, "candidate": "" if key.startswith("at:") else key, "tile_id": group.tile_id.iloc[0],
                    "x": float(group.x.iloc[0]), "y": float(group.y.iloc[0]), "value": top, "subtype": subtype, "n": len(group),
                    "agreement": agreement, "positive": bool(votes.mean() > 0.5) if votes.mean() != 0.5 else None,
                    "sources": ", ".join(sorted(group.source)), "latest": latest, "latest_subtype": latest_subtype, "time": float(time.max())})
    return pd.DataFrame(out, columns=store.CONSENSUS)


# ---------- a synthetic store ----------

def _poly(rng, x, y, r) -> str:
    n = int(rng.integers(3, 30))
    t = np.sort(rng.uniform(0, 2 * np.pi, n))
    return " ".join(f"{v:.0f}" for p in zip(x + r * np.cos(t), y + r * np.sin(t)) for v in p)


def _store(rng, n: int = 400) -> pd.DataFrame:
    rows = []
    for i in range(n):
        x, y = float(rng.uniform(0, 1200)), float(rng.uniform(0, 1200))
        masks = "|".join("" if rng.random() < 0.15 else _poly(rng, x, y, r) for r in (4, 9, 16)) if rng.random() < 0.85 else ""
        rows.append({"candidate_id": candidate_store.candidate_id("6E10", CORE, x, y) + (f"~{i}" if i % 50 == 7 else ""), "x_core": x, "y_core": y,
                     "tile_id": TILE if rng.random() < 0.7 else "LIP-1_B-2_6E10_S02", "area_um2": float(rng.uniform(50, 500)),
                     "radius_px": float(rng.uniform(3, 20)) if rng.random() < 0.8 else np.nan, "sources": "dab@1",
                     "proposal_threshold": [0.5, 1.0, 1.5, np.nan][int(rng.integers(4))], "outline": _poly(rng, x, y, 8),
                     "sam_outline": ";".join(_poly(rng, x, y, r) for r in (2, 7)) if rng.random() < 0.5 else "", "sam_status": "ok",
                     "sam_masks": masks, "sam_scores": "|".join(f"{v:.2f}" for v in rng.uniform(0.5, 1, 3)) if masks else "",
                     "sam_default": float(rng.integers(0, 3)) if rng.random() < 0.7 else np.nan, "probability": float(rng.uniform()),
                     "model_id": "m@1", "decision": "accepted" if rng.random() < 0.5 else "rejected", "reason": "accepted",
                     "class": "plaque", "subtype": "compact" if rng.random() < 0.5 else "", "mean_dab_od": float(rng.uniform())})
    return pd.DataFrame(rows)


@pytest.fixture()
def project(synthetic_project, monkeypatch):
    monkeypatch.chdir(synthetic_project)
    state.reset()
    yield state.get_state()
    state.reset()


@pytest.mark.parametrize("parquet", [True, False])
def test_a_field_read_alone_equals_the_whole_core_read_and_filtered(project, parquet):
    rng = np.random.default_rng(3)
    frame = _store(rng)
    target = candidate_store.folder(project.project, "6E10") / f"{CORE}.csv.gz"
    if parquet:
        candidates.write_table(target, frame)
        assert candidate_store.path(project.project, "6E10", CORE).suffix == ".parquet"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False, compression="gzip")  # the store as text, without pyarrow
    tile = project.tile(TILE)
    for level in (None, 1.0):
        new = candidate_store.in_field(project.project, "6E10", tile, project.project.context_px, level)
        old = _old_in_field(project.project, "6E10", tile, project.project.context_px, level)
        assert len(new) == len(old) > 20
        for column in ("candidate_id", "x", "y", "own", "decision", "class", "subtype", "probability", "radius_px", "area_um2", "outline",
                       "sam_outline", "sam_masks", "sam_scores", "sam_default", "proposal_threshold", "tile_id"):
            assert list(map(repr, new[column])) == list(map(repr, old[column])), column
        assert candidate_store.field_cached(project.project, "6E10", tile, project.project.context_px, level)
    for text in [*frame.outline, *frame.sam_outline, *frame.sam_masks.str.split("|").explode().fillna(""), "[[1, 2], [3, 4], [5, 7]]", "", "1 2 3"]:
        assert candidate_store._polygon(text) == _old_polygon(text)
    for text in [*frame.sam_masks, "[]", "||", "1 2 3 4 5 6||"]:
        assert candidate_store._polygons(text) == _old_polygons(text)


def _decision_rows(rng, n: int) -> pd.DataFrame:
    people = [f"p{i}" for i in range(6)]
    candidates_ = [f"6E10:{CORE}:{i}:{i}" for i in range(n // 3)]
    return pd.DataFrame({"candidate": rng.choice(candidates_, n), "source": rng.choice(people, n),
                         "identity": rng.choice(["plaque", "not a plaque", "CAA", "unsure"], n, p=[0.45, 0.3, 0.15, 0.1]),
                         "subtype": rng.choice(["", "compact", "diffuse", "small"], n),
                         "role": "", "time": rng.integers(0, 50, n).astype(float),  # many ties in time: the order of the rows decides
                         "session": "s", "confirmed": False})


def test_decisions_counted_at_once_equal_the_candidate_by_candidate_loop():
    rng = np.random.default_rng(5)
    words = vocab.defaults("6E10")
    for n in (0, 7, 3000, 12000):  # a submitted tile records every candidate on it: tens of thousands of rows
        rows = _decision_rows(rng, n) if n else pd.DataFrame(columns=decisions.COLUMNS)
        rows["role"] = [words.role(i) for i in rows.identity]
        new, old = decisions._state_of(rows), _old_state_of(rows)
        assert new.decided == old.decided and new.answered == old.answered and new.calls == old.calls and new.times == old.times


def test_the_consensus_counted_at_once_equals_the_object_by_object_loop():
    rng = np.random.default_rng(8)
    n = 5000
    labels = pd.DataFrame({"id": [f"class:t:{i}" for i in range(n)], "source": rng.choice([f"p{i}" for i in range(5)], n),
                           "stain": rng.choice(["6E10", "AT8"], n, p=[0.9, 0.1]), "task": rng.choice(["class", "outline"], n, p=[0.9, 0.1]),
                           "tile_id": rng.choice(["t1", "t2", "t3"], n), "x": rng.integers(0, 40, n) + rng.choice([0.0, 0.2, 0.7], n),
                           "y": rng.integers(0, 40, n).astype(float),
                           "value": rng.choice(["plaque", "not a plaque", "CAA", "unsure", "compact"], n),
                           "subtype": rng.choice([None, "", "diffuse", "compact"], n)})
    for stain in ("6E10", "AT8", "NeuN"):
        new, old = store.consensus(labels, stain), _old_consensus(labels, stain)
        assert list(new.columns) == list(old.columns) and len(new) == len(old)
        assert new.to_dict("list") == old.to_dict("list"), stain


def test_slow_work_answers_pending_with_its_place_and_then_its_result():
    import threading

    gate = threading.Event()
    task = work.submit("test slow", ("test slow", 1), lambda t: (t.step(1, 4), gate.wait(10), "made")[-1])
    answer = work.answer("test slow", ("test slow", 1), task.run, seconds=0.05)
    assert answer.status_code == 202
    body = json.loads(answer.body)
    assert body["then"] == "retry" and body["pending"]["kind"] == "test slow" and body["pending"]["state"] in ("waiting", "running")
    gate.set()
    assert work.wait(task, 10) and work.outcome(task) == "made"
    assert work.answer("test slow", ("test slow", 1), task.run) == "made"  # asked again: the work is done once
