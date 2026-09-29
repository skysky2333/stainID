from __future__ import annotations

import math

import numpy as np
import pandas as pd


def records(frame: pd.DataFrame) -> list[dict]:
    """DataFrame rows as JSON-safe dicts (NaN/inf -> None, numpy scalars -> Python)."""
    if frame.empty:
        return []
    clean = frame.replace([np.inf, -np.inf], np.nan).astype(object).where(frame.notna(), None)
    return [{k: _py(v) for k, v in row.items()} for row in clean.to_dict(orient="records")]


def _py(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
