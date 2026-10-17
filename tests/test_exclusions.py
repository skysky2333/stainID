"""The areas annotators mark as bad (labels/exclusions.py): which labels they keep out of scoring and training."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from stainid.labels import exclusions


def _reference(labels: pd.DataFrame) -> pd.DataFrame:
    """The first version, one label at a time with cv2.pointPolygonTest: what the vectorised one must agree with."""
    marks = labels[labels.task == exclusions.TASK]
    areas = {(r.source, r.tile_id): exclusions.rings_of(r) for r in marks.itertuples()}
    objects = labels[labels.task.isin(["class", "outline"]) & labels.x.notna()]
    drop = [i for i, r in zip(objects.index, objects.itertuples())
            if (r.source, r.tile_id) in areas and exclusions.inside(*areas[(r.source, r.tile_id)], float(r.x), float(r.y))]
    return labels.drop(index=drop)


def _labels(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    square = [[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]]
    hole = [[180.0, 180.0], [220.0, 180.0], [220.0, 220.0], [180.0, 220.0]]
    marks = [{"source": "Ann", "tile_id": "T1", "task": "exclude", "x": None, "y": None, "points": [square, hole], "value": "holes:0,1"},
             {"source": "Ben", "tile_id": "T2", "task": "exclude", "x": None, "y": None, "points": [[[0, 0], [50, 0], [25, 60]]], "value": "holes:0"}]
    rows = [{"source": str(rng.choice(["Ann", "Ben", "Cy"])), "tile_id": str(rng.choice(["T1", "T2", "T3"])), "task": str(rng.choice(["class", "outline", "tag"])),
             "x": float(rng.uniform(0, 400)), "y": float(rng.uniform(0, 400)), "points": None, "value": "plaque"} for _ in range(n)]
    rows += [{"source": "Ann", "tile_id": "T1", "task": "class", "x": x, "y": y, "points": None, "value": "plaque"}
             for x, y in [(100.0, 150.0), (300.0, 300.0), (200.0, 200.0), (180.0, 200.0), (99.9, 150.0)]]  # on edges, in the hole
    return pd.DataFrame(marks + rows)


def test_the_vectorised_exclusions_agree_with_the_first_version():
    labels = _labels(3000)
    fast, slow = exclusions.counted(labels), _reference(labels)
    assert list(fast.index) == list(slow.index)
    assert len(labels) - len(fast) > 30  # it does leave labels out: Ann's in her square (not the hole), Ben's in his triangle


def test_labels_without_marked_areas_are_passed_through_quickly():
    labels = _labels(200_000).query("task != 'exclude'")
    started = time.time()
    assert exclusions.counted(labels) is labels
    marked = _labels(200_000)
    started = time.time()
    exclusions.counted(marked)
    assert time.time() - started < 5.0  # one test per marked (person, field), not one per label
