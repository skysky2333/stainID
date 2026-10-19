"""Bring an AI review batch's labels (`review.ai_batch`) into the label log: one read-only session per stain and round,
"AI review round 4 · 6E10" (`session_name`, kind `KIND`), each reviewer its own account `ai:<name>`.

What each reviewer row becomes, as "Fix the field" records the same answers (`api.routes.rounds`):

- a class label about its candidate (`candidate`, the field and position of the candidate in the field picture), its
  value and subtype the vocabulary's ids; `confirmed` when it is detection's own call; `meta` keeps the review id,
  confidence, rationale, the outline judgement and the round;
- `outline_ok` "ok" (on an object that is not rejected or unsure): the outline the reviewer saw, confirmed, as the
  app records a person leaving detection's outline as it is (task `outline`, `value` the choice, e.g. "stain" or
  "sam_medium", `confirmed`). "too big", "too small" and "wrong object" record no outline (a reviewer draws none):
  the judgement stays in the class label's `meta` and in `outline_judgements.csv`;
- a missed object (`<stain>-F03-M1`, `field_x`, `field_y`): a class label without a candidate at that position, an
  object people added (`labels.added` measures it), unless a candidate of the store lies there (then it goes to
  `missed_objects.csv` only, as does one outside the part of the field asked);
- objects of both halves: where the reviewers agree on the identity each keeps their label (a type they disagree on
  is dropped: "type not recorded"); where they disagree, every reviewer's label of it is "unsure" (the label each gave
  stays in `meta.said`).

Labels on objects in excluded areas stay in the log and are left out by every reader (`exclusions.drop_excluded`), as
for any label; the report counts them. Ingesting again replaces what the session had (changed labels are written
again, labels no longer given are marked deleted). The session is listed on the Labels page and can be chosen for
training on the Models page like any other; the annotation pages hide `ai:` accounts' labels.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stainid import exclusions as excluded_areas
from stainid.labels import candidate_store, sessions, store
from stainid.labels import vocabulary as vocab
from stainid.project import Project
from stainid.review import ai_batch

KIND = "ai_review"
PREFIX = "ai:"
MISSED = re.compile(r"^(?P<stain>[^-]+)-(?P<field>F\d+)-M(?P<n>\d+)$", re.IGNORECASE)
MISSED_TOLERANCE_PX = 40.0  # a missed object may sit this far outside the box asked


def session_name(stain: str, round_: int) -> str:
    return f"ai_review_round{round_}_{re.sub(r'[^A-Za-z0-9]', '', stain).lower()}"


def session_title(stain: str, round_: int) -> str:
    return f"AI review round {round_} · {stain}"


def account(name) -> str:
    """The reviewer's account: `ai:<name>` (lower case, letters, digits, - and _)."""
    text = vocab.text(name)
    text = text[len(PREFIX):] if text.lower().startswith(PREFIX) else text
    slug = re.sub(r"[^a-z0-9_-]+", "-", text.lower()).strip("-")
    return PREFIX + (slug or "reviewer")


def _norm(value, allowed: tuple[str, ...]) -> str:
    text = vocab.text(value).lower().replace("_", " ")
    text = {"moderate": "moderate", "medium": "moderate", "mod": "moderate", "yes": "ok", "fine": "ok", "good": "ok", "wrong": "wrong object",
            "too large": "too big", "n/a": "", "na": "", "-": ""}.get(text, text)
    return text if text in allowed else ""


def read_reviews(paths: list[Path]) -> pd.DataFrame:
    """The reviewers' rows (every file; the last row of a reviewer about an id wins), columns as `ai_batch.CSV_COLUMNS`
    and `file`."""
    parts = []
    for path in paths:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        frame.columns = [c.strip().lower() for c in frame.columns]
        if "review_id" not in frame or "label" not in frame:
            raise ValueError(f"{path} lacks the review_id or label column")
        for column in ai_batch.CSV_COLUMNS:
            if column not in frame:
                frame[column] = ""
        parts.append(frame[ai_batch.CSV_COLUMNS].assign(file=path.name))
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*ai_batch.CSV_COLUMNS, "file"])
    rows = rows.assign(review_id=rows.review_id.str.strip(), label=rows.label.str.strip())
    rows = rows[(rows.review_id != "") & (rows.label != "")]
    fallback = rows.file.str.extract(r"review_labels_([A-Za-z0-9]+)")[0].fillna("reviewer")
    rows = rows.assign(source=[account(r if vocab.text(r) else f"reviewer-{f}") for r, f in zip(rows.reviewer, fallback)])
    return rows.drop_duplicates(["source", "review_id"], keep="last").reset_index(drop=True)


def resolve_overlap(said: pd.DataFrame, vocabulary: vocab.Vocabulary) -> pd.DataFrame:
    """Objects labelled by more than one reviewer: agreeing on the identity, each keeps their label (a disputed type is
    dropped); disagreeing, each label becomes "unsure" (`said` keeps what they gave) and no outline is confirmed. Columns
    used: review_id, source, identity, subtype, outline_ok; adds `agreed` (None for objects one reviewer labelled)."""
    out = said.assign(agreed=None, said="")
    unsure = vocabulary.unsure
    for _, group in out.groupby("review_id"):
        if group.source.nunique() < 2:
            continue
        identities = set(group.identity)
        if len(identities) > 1:
            out.loc[group.index, "said"] = [f"{i} / {s}" if s else i for i, s in zip(group.identity, group.subtype)]
            out.loc[group.index, "identity"] = unsure
            out.loc[group.index, "subtype"] = ""
            out.loc[group.index, "outline_ok"] = ""
            out.loc[group.index, "agreed"] = False
        else:
            if len(set(group.subtype)) > 1:
                out.loc[group.index, "subtype"] = ""
            if len(set(group.outline_ok)) > 1:
                out.loc[group.index, "outline_ok"] = ""
            out.loc[group.index, "agreed"] = True
    return out


def _records(project: Project, stain: str, manifest: pd.DataFrame, fields: pd.DataFrame, rows: pd.DataFrame, name: str, round_: int,
             tiles: pd.DataFrame) -> tuple[list[dict], dict]:
    """The label records of the reviewers' rows, and what became of every row (for the report)."""
    vocabulary = vocab.of(project, stain)
    reject, unsure = vocabulary.reject, vocabulary.unsure
    by_id = manifest.set_index("review_id")
    resolved = [ai_batch.resolve_label(vocabulary, v) for v in rows.label]
    rows = rows.assign(identity=[r[0] if r else "" for r in resolved], subtype=[r[1] if r else "" for r in resolved],
                       confidence=[_norm(c, ai_batch.CONFIDENCE) for c in rows.confidence],
                       outline_ok=[_norm(o, ai_batch.OUTLINE) for o in rows.outline_ok])
    invalid = rows[[r is None for r in resolved]]
    rows = rows[[r is not None for r in resolved]]
    missed = rows[rows.review_id.str.match(MISSED.pattern, case=False)]
    objects = rows[rows.review_id.isin(set(by_id.index))]
    unknown = rows[~rows.index.isin(missed.index) & ~rows.index.isin(objects.index)]
    objects = resolve_overlap(objects, vocabulary)
    now = time.time()
    records, judgements = [], []
    for r in objects.itertuples():
        m = by_id.loc[r.review_id]
        x, y = round(float(m.x), 1), round(float(m.y), 1)
        detected = (str(m.ai_identity), vocab.text(m.ai_subtype))
        meta = {"ai_review": round_, "review_id": r.review_id, "confidence": r.confidence, "rationale": vocab.text(r.rationale)[:200],
                "outline": r.outline_ok}
        if r.said:
            meta["said"] = r.said
        stamp = {"source": r.source, "account": r.source, "stain": stain, "tile_id": str(m.tile_id), "x": x, "y": y, "candidate": str(m.candidate_id),
                 "session": name, "time": now}
        records.append({**stamp, "id": store.object_id("class", m.tile_id, x, y), "task": "class", "value": r.identity, "subtype": r.subtype or None,
                        "confirmed": (r.identity, r.subtype) == detected, "meta": meta})
        if r.identity in (reject, unsure):
            continue
        if r.outline_ok == "ok" and vocab.text(m.outline_points):
            records.append({**stamp, "id": store.object_id("outline", m.tile_id, x, y), "task": "outline", "points": json.loads(m.outline_points),
                            "value": str(m.outline_choice), "confirmed": True, "meta": {"ai_review": round_, "review_id": r.review_id}})
        elif r.outline_ok and r.outline_ok != "ok":
            judgements.append({"review_id": r.review_id, "source": r.source, "candidate_id": m.candidate_id, "tile_id": m.tile_id, "x": x, "y": y,
                               "label": f"{r.identity} / {r.subtype}" if r.subtype else r.identity, "outline_choice": m.outline_choice,
                               "outline_ok": r.outline_ok, "rationale": vocab.text(r.rationale)})
    placed = []
    boxes = fields.set_index("field")
    for r in missed.itertuples():
        found = MISSED.match(r.review_id)
        code = found["field"].upper()
        entry = {"review_id": r.review_id, "source": r.source, "label": r.label, "field": code, "field_x": r.field_x, "field_y": r.field_y,
                 "confidence": r.confidence, "rationale": vocab.text(r.rationale)}
        try:
            fx, fy = float(r.field_x), float(r.field_y)
        except (TypeError, ValueError):
            placed.append({**entry, "outcome": "no position"})
            continue
        if code not in boxes.index:
            placed.append({**entry, "outcome": "unknown field"})
            continue
        f = boxes.loc[code]
        t = MISSED_TOLERANCE_PX
        if not (f.box_x0 - t <= fx < f.box_x1 + t and f.box_y0 - t <= fy < f.box_y1 + t):
            placed.append({**entry, "outcome": "outside the part asked"})
            continue
        placed.append({**entry, "tile_id": f.tile_id, "x": round(fx, 1), "y": round(fy, 1), "identity": r.identity, "subtype": r.subtype})
    candidates = pd.DataFrame([p for p in placed if "identity" in p])
    if len(candidates):
        near = candidate_store.match(project, stain, candidates[["tile_id", "x", "y"]], tiles)
        for p, hit in zip([p for p in placed if "identity" in p], near):
            if isinstance(hit, str) and hit:
                p["outcome"] = f"a candidate lies there ({hit})"
                p["candidate_id"] = hit
                continue
            p["outcome"] = "added"
            records.append({"id": store.object_id("class", p["tile_id"], p["x"], p["y"]), "source": p["source"], "account": p["source"], "stain": stain,
                            "task": "class", "tile_id": p["tile_id"], "x": p["x"], "y": p["y"], "value": p["identity"], "subtype": p["subtype"] or None,
                            "candidate": None, "session": name, "confirmed": False, "time": now,
                            "meta": {"ai_review": round_, "review_id": p["review_id"], "missed": True, "confidence": p["confidence"],
                                     "rationale": p["rationale"][:200]}})
    overlap = objects[objects.agreed.notna()].drop_duplicates("review_id")
    report = {"rows": int(len(rows) + len(invalid)), "invalid": invalid[["review_id", "label", "source", "file"]].to_dict("records"),
              "unknown_ids": unknown[["review_id", "source", "file"]].to_dict("records"),
              "objects": int(objects.review_id.nunique()), "of_manifest": int(len(manifest)),
              "by_reviewer": {s: int(n) for s, n in objects.source.value_counts().items()},
              "classes": {vocabulary.display(i, s): int(n) for (i, s), n in objects.groupby(["identity", "subtype"]).size().items()},
              "overlap_objects": int(len(overlap)), "overlap_agreed": int(overlap.agreed.astype(bool).sum()) if len(overlap) else 0,
              "outline": {k: int(v) for k, v in objects.outline_ok.replace("", "(none)").value_counts().items()},
              "missed": {k: int(v) for k, v in pd.Series([p.get("outcome", "") for p in placed]).value_counts().items()} if placed else {},
              "unlabelled": sorted(set(manifest.review_id) - set(objects.review_id))}
    return records, {"report": report, "judgements": judgements, "missed": placed}


def ingest(project: Project, stain: str, round_: int = ai_batch.ROUND, paths: list[Path] | None = None, dry_run: bool = False, log=print) -> dict:
    """Import the reviewers' CSVs of one stain (`review_labels_*.csv` of the batch, or `paths`) as the session
    `session_name`; returns the report (also `ingest_report.json` next to the batch)."""
    from stainid.sampling.tiles import known_tiles  # noqa: PLC0415

    folder = ai_batch.stain_folder(project, stain, round_)
    text = ("review_id", "field", "tile_id", "candidate_id", "ai_identity", "ai_subtype", "outline_choice", "outline_points")
    manifest = pd.read_csv(folder / "manifest.csv", dtype={c: str for c in text})
    manifest[list(text)] = manifest[list(text)].fillna("")
    fields = pd.read_csv(folder / "fields.csv")
    paths = paths or sorted(folder.glob("review_labels_*.csv"))
    rows = read_reviews(paths)
    name = session_name(stain, round_)
    tiles = known_tiles(project)
    records, extra = _records(project, stain, manifest, fields, rows, name, round_, tiles)
    report = extra["report"]
    frame = pd.DataFrame(records, columns=[*store.COLUMNS, "account"])
    kept = excluded_areas.drop_excluded(project, frame.reset_index(drop=True)) if len(frame) else frame
    report["in_excluded_areas"] = int(len(frame) - len(kept))
    root = project.output("reviews")
    current = store.current(root)
    old = current[current.session == name] if len(current) else current
    new_keys = {(r["source"], r["id"]) for r in records}
    gone = [{**r, "deleted": True, "time": None} for r in old.astype(object).where(old.notna(), None).to_dict("records")
            if (r["source"], r["id"]) not in new_keys]
    before = {(r.source, r.id): r for r in old.itertuples()} if len(old) else {}
    changed = [r for r in records if not _same(before.get((r["source"], r["id"])), r)]
    report |= {"session": name, "title": session_title(stain, round_), "written": len(changed), "deleted": len(gone), "unchanged": len(records) - len(changed),
               "reviewers": sorted({r["source"] for r in records}), "files": [str(p) for p in paths], "dry_run": dry_run}
    if not dry_run:
        if gone or changed:
            store.append(root, gone + changed)
        _session(root, name, stain, round_, fields, report)
        pd.DataFrame(extra["judgements"]).to_csv(folder / "outline_judgements.csv", index=False)
        pd.DataFrame(extra["missed"]).to_csv(folder / "missed_objects.csv", index=False)
        (folder / "ingest_report.json").write_text(json.dumps(report, indent=1, default=str))
    log(f"{name}: {report['objects']} of {report['of_manifest']} objects from {', '.join(report['reviewers']) or 'nobody'}; "
        f"{len(changed)} labels written, {len(gone)} deleted; overlap {report['overlap_agreed']}/{report['overlap_objects']} agreed; "
        f"{len(report['invalid'])} rows with an unknown label; {report['in_excluded_areas']} labels in excluded areas" + (" (dry run)" if dry_run else ""),
        flush=True)
    return report


def _value(value):
    """A stored value as compared: None for missing ones (None, NaN, "")."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or (isinstance(value, str) and not value):
        return None
    return value


def _same(old, new: dict) -> bool:
    """Whether a label in the log says what a new record says (so it is not written again)."""
    if old is None:
        return False
    for key in ("value", "subtype", "candidate", "tile_id", "points", "meta"):
        if _value(getattr(old, key, None)) != _value(new.get(key)):
            return False
    if bool(getattr(old, "confirmed", False)) != bool(new.get("confirmed")):
        return False
    return all(abs(float(getattr(old, k)) - float(new[k])) < 1e-6 for k in ("x", "y"))


def _session(root: Path, name: str, stain: str, round_: int, fields: pd.DataFrame, report: dict) -> dict:
    """The read-only session the labels belong to (made once, its record updated on every ingest)."""
    about = {"title": session_title(stain, round_), "stain": stain, "fields": list(fields.tile_id), "read_only": True, "ai_review": {
        "round": round_, "reviewers": report["reviewers"], "objects": report["objects"], "ingested": time.time(),
        "subfields": {f.tile_id: [f.box_x0, f.box_y0, f.box_x1, f.box_y1] for f in fields.itertuples() if bool(f.subfield)}},
        "labels": report["written"] + report["unchanged"]}
    if name in sessions.names(root):
        return sessions.update(root, name, **about)
    return sessions.create(root, name, kind=KIND, tools=["class", "outline"], ai_view="full", per_field=1, rule="random",
                           instructions=f"Labels by AI reviewer agents (accounts {PREFIX}<name>) from the picture sheets of AI review round {round_}; "
                                        "imported read-only.", **about)
