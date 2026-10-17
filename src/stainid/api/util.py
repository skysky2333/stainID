from __future__ import annotations

import gzip
import json
import math

import numpy as np
import pandas as pd


def records(frame: pd.DataFrame) -> list[dict]:
    """DataFrame rows as JSON-safe dicts (NaN/inf -> None, numpy scalars -> Python)."""
    if frame.empty:
        return []
    clean = frame.replace([np.inf, -np.inf], np.nan).astype(object).where(frame.notna(), None)
    return [{k: _py(v) for k, v in row.items()} for row in clean.to_dict(orient="records")]


GZIP_FROM_BYTES = 16 * 1024


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    raise TypeError(f"{type(value).__name__} is not JSON")


def fast_json(request, payload, status_code: int = 200):
    """A large JSON answer made quickly: written in one pass (not walked value by value first, as FastAPI does) and
    gzipped when the browser accepts it (a field's outlines shrink about five times)."""
    from fastapi.responses import Response  # noqa: PLC0415

    body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"), default=_plain).encode("utf-8")
    headers = {"Vary": "Accept-Encoding"}
    if len(body) >= GZIP_FROM_BYTES and "gzip" in (request.headers.get("accept-encoding") or "").lower():
        body = gzip.compress(body, compresslevel=5)
        headers["Content-Encoding"] = "gzip"
    return Response(body, status_code=status_code, media_type="application/json", headers=headers)


def _py(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
