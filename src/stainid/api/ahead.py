"""Idle-time preparation ("Preparing fields ahead"): while the computer has nothing else to do, the fields of the open
"Fix the field" sessions are made ready, so that opening one is instant.

One background thread, the lowest priority of everything the app and the job queue do. It works in small units (one
core image's field pictures from one decode, one field's Segment Anything features, one field's thread ladder, one
core's thumbnails) and before each one looks whether anything else wants the computer:

- someone waits: a picture being made for a person (`render.people_waiting`), work in the app's pool (`work`), the
  Segment Anything thread busy, or a request of any page taking longer than `SLOW_REQUEST` seconds;
- the job queue: a job running, or one queued that could start (any class), or "Pause all background work";
- other programs: over a short quiet look (`LOOK_SECONDS`, every `LOOK_EVERY` seconds), the processors more than
  `CPU_BUSY` busy, the project's disk more than `DISK_BUSY` busy, or a GPU more than `GPU_BUSY` busy.

Then it waits and looks again. Its thread runs at the lowest CPU priority (nice 19) and the idle disk class, so even a
unit under way gives way.

What is made, most likely to be opened first:

1. the cores people have open or opened lately (`RECENT_SECONDS`), most recent first: every session field's picture on
   them, the cores' thumbnails, and the features and thread ladders of the fields people are on and the next ones;
2. per open session (the most recently worked on first), the next `NEAR` unfinished fields after the one last opened:
   pictures, thumbnails, Segment Anything features (on a GPU only) and, for AT8, the thread ladder (the wide one of a
   simple session);
3. the caches the admin pages read (the step overview with the v2 counts, the labelling tiles' tissue shares, the labels
   overview);
4. every other unfinished field of the open sessions, in session order: pictures, core thumbnails and AT8 thread
   ladders (Segment Anything features only near: ~18 MB a field); and every other Grey / white matter core still to do.

The open "Grey / white matter" sessions are prepared the same way, core by core (`_matter`): tier 2 the `MATTER_NEAR`
cores after the one each person opened last, in the picker's order (and the picker's first ones), tier 4 the rest; a
core is its stain images' deep-zoom pyramids (one decode each, the slow part of opening a core: 10-13 s an image), the
carry from NeuN to the other stains and detection's neurons for the guide.

Space: tiers 1-3 are what people open next and are made whatever the cache holds (the cache's own cleanup keeps it under
its limit, least recently used first). Tier 4 stops while the project's caches take more than `SHARE` of their limit
(Settings, "Cache"), after deleting what nobody used for `STALE_DAYS` days; the status says how much all of it takes
(`estimate`) and whether it fits.

Its pause switch (the Run page) is kept across restarts (`.ahead.json` in the job queue's folder).
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SLOW_REQUEST = 0.25  # a page request running this long: someone waits
LOOK_SECONDS = 1.5  # the quiet look at the processors, the disk and the GPUs
LOOK_EVERY = 30.0  # seconds of work between two quiet looks
WAIT_SECONDS = 5.0  # between two looks while something else uses the computer
CPU_BUSY = 0.25  # share of all processors busy
DISK_BUSY = 0.25  # share of the time the project's disk is busy
GPU_BUSY = 20  # % utilisation of any GPU
RECENT_SECONDS = 6 * 3600.0  # a core opened this recently is "open"
NEAR = 8  # fields per session made fully ready after the one last opened
PLAN_SECONDS = 300.0  # the plan is made again at least this often (and whenever people open another field)
SHARE = 0.85  # tier 4 stops above this share of the cache limit
STALE_DAYS = 3.0  # tier 4 first deletes cache files nobody used for this long
MB = 10**6
PICTURE_BYTES = int(0.9 * MB)  # a field picture (2,304 px with its margin, JPEG: 0.87 MB on average in the study)
SAM_BYTES = 18 * MB  # a field's Segment Anything features (nine windows of ~2 MB)
LADDER_BYTES = 1 * MB  # a field's thread ladder (the wide one: 0.3-1.4 MB)
THUMB_BYTES = int(1.5 * MB)  # a core's thumbnails (the base, 512 px, 2,048 px)
MATTER_NEAR = 3  # Grey / white matter cores made ready after the one each person opened last (and from the picker's start)
CORE_PICTURE_BYTES = 40 * MB  # one core image's deep-zoom pyramid (a ~16,000 px core: 30-45 MB)

_LOCK = threading.Lock()
_THREAD: threading.Thread | None = None
_WAKE = threading.Event()
_REPLAN = threading.Event()  # sessions changed: the current pass stops and plans again
_STATUS: dict = {"state": "starting", "why": "", "current": "", "made": 0}
_REQUESTS: dict[int, float] = {}  # page requests in flight: id -> when they started
_REQUESTS_LOCK = threading.Lock()


# ---------------------------------------------------------------- the pause switch

def _switch_file() -> Path:
    from stainid.jobs.queue import queue_dir  # noqa: PLC0415

    return queue_dir() / ".ahead.json"


_PAUSED = [-1e9, False]  # when looked at, the switch


def paused() -> bool:
    if time.monotonic() - _PAUSED[0] > 2:
        try:
            value = bool(json.loads(_switch_file().read_text()).get("paused"))
        except (OSError, ValueError):
            value = False
        _PAUSED[:] = [time.monotonic(), value]
    return bool(_PAUSED[1])


def set_paused(value: bool) -> dict:
    path = _switch_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"paused": bool(value), "time": time.time()}))
    os.replace(tmp, path)
    _PAUSED[:] = [time.monotonic(), bool(value)]
    _WAKE.set()
    return status()


# ---------------------------------------------------------------- page requests (the app's middleware)

def request_started() -> int:
    token = id(object()) ^ time.monotonic_ns()
    with _REQUESTS_LOCK:
        _REQUESTS[token] = time.monotonic()
    return token


def request_ended(token: int) -> None:
    with _REQUESTS_LOCK:
        _REQUESTS.pop(token, None)


def someone_waits_for_a_page() -> bool:
    now = time.monotonic()
    with _REQUESTS_LOCK:
        return any(now - started > SLOW_REQUEST for started in _REQUESTS.values())


# ---------------------------------------------------------------- is the computer free?

def _app_busy() -> str:
    from stainid.api import pyramid, render, sam_embeddings, work  # noqa: PLC0415

    if render.people_waiting() or someone_waits_for_a_page() or pyramid.people_waiting():
        return "someone is waiting for a page"
    if not work.idle():
        return "the app is preparing something for someone"
    if sam_embeddings.busy():
        return "Segment Anything is preparing a field"
    return ""


def _jobs_busy() -> str:
    from stainid.jobs import queue  # noqa: PLC0415

    scheduler = queue._SHARED
    if scheduler is None:
        return ""
    try:
        if scheduler.paused_all():
            return "all background work is paused (Run page)"
        jobs = list(scheduler.jobs.values())  # kept fresh by the queue's own thread (`Scheduler.tick`)
    except Exception:  # noqa: BLE001 - a queue that cannot be read: nothing known to wait for
        return ""
    if any(j.status == "running" for j in jobs):
        return "jobs are running"
    blocked = ("Waits for “", "Not started")  # waits for another job or for its inputs: it would not start anyway
    if any(j.status == "queued" and not str(j.waiting or "").startswith(blocked) for j in jobs):
        return "jobs are waiting to run"
    return ""


def _cpu_times() -> tuple[int, int] | None:
    try:
        with open("/proc/stat") as f:
            values = [int(v) for v in f.readline().split()[1:]]
    except (OSError, ValueError):
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def _disk_stat(root: Path) -> Path | None:
    try:
        device = os.stat(root).st_dev
    except OSError:
        return None
    path = Path(f"/sys/dev/block/{os.major(device)}:{os.minor(device)}/stat")
    return path if path.exists() else None


def _io_ticks(stat: Path | None) -> int | None:
    if stat is None:
        return None
    try:
        return int(stat.read_text().split()[9])  # milliseconds the device was busy
    except (OSError, ValueError, IndexError):
        return None


_GPU = [0.0, 0]


def gpu_busy() -> int:
    """The highest utilisation (%) of this computer's GPUs (0: none), looked at every 10 s."""
    if time.monotonic() - _GPU[0] < 10:
        return _GPU[1]
    found = 0
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                 timeout=10).stdout
            found = max([int(float(v)) for v in out.split() if v.strip()] or [0])
        except (OSError, ValueError, subprocess.SubprocessError):
            found = 0
    _GPU[:] = [time.monotonic(), found]
    return found


def look(root: Path, seconds: float = LOOK_SECONDS) -> str:
    """A quiet look (nothing of ours runs meanwhile): what other programs use. "" when the computer is free."""
    cpu0, disk_file = _cpu_times(), _disk_stat(root)
    disk0, start = _io_ticks(disk_file), time.monotonic()
    _WAKE.wait(seconds)
    cpu1, disk1, took = _cpu_times(), _io_ticks(disk_file), max(1e-3, time.monotonic() - start)
    if cpu0 and cpu1 and cpu1[0] > cpu0[0]:
        total, idle = cpu1[0] - cpu0[0], cpu1[1] - cpu0[1]
        if (total - idle) / total > CPU_BUSY:
            return "the processors are busy"
    if disk0 is not None and disk1 is not None and (disk1 - disk0) / 1000.0 / took > DISK_BUSY:
        return "the disk is busy"
    if gpu_busy() > GPU_BUSY:
        return "a GPU is busy"
    return ""


def others_busy() -> str:
    """For work made ahead elsewhere (Segment Anything features of every session field): the pause switch, someone
    waiting, the job queue."""
    from stainid.api import render  # noqa: PLC0415

    if paused():
        return "paused"
    if render.people_waiting() or someone_waits_for_a_page():
        return "someone is waiting for a page"
    return _jobs_busy()


def busy_now() -> str:
    """The quick look before every unit: someone waiting, the job queue, the pause switch."""
    if paused():
        return "paused"
    return _app_busy() or _jobs_busy()


# ---------------------------------------------------------------- what to make

@dataclass
class Unit:
    tier: int  # 1 cores open, 2 next fields, 3 admin caches, 4 everything else
    kind: str  # "pictures", "thumbs", "sam", "ladder", "caches", "slide" (a whole slide's picture: `image_path` is its id),
    # "core_pictures" (a Grey / white matter core: `image_path` its id, `tiles` its stains)
    state: object
    label: str
    tiles: tuple[str, ...] = ()
    image_path: str = ""
    wide: bool = False


@dataclass
class Session:
    meta: dict
    fields: list[str]  # unfinished, in session order
    near: list[str]
    last: float  # when someone last opened one of its fields
    at8: bool
    simple: bool


def _opened(state) -> dict[tuple[str, str], float]:
    """(session, tile) -> when someone last had it open."""
    from stainid.api.routes import annotate  # noqa: PLC0415

    with annotate._BUSY_LOCK:
        found = {(name, tile): max(people.values()) for ((sid, name), tile), people in annotate._BUSY.items() if sid == state.id and people}
    return found


def _sessions(state, opened: dict) -> list[Session]:
    from stainid.api.routes import rounds as round_routes  # noqa: PLC0415
    from stainid.labels import rounds, sessions  # noqa: PLC0415

    out = []
    reviews = state.project.output("reviews")
    for name in sessions.names(reviews):
        try:
            meta = sessions.meta(reviews, name)
        except Exception:  # noqa: BLE001 - a half-written session
            continue
        if meta.get("practice") or not (rounds.is_round(meta) and meta.get("open")) or rounds.round_of(meta) != 2 or not meta.get("fields"):
            continue
        finished = round_routes.finished_tiles(meta)
        needed = rounds.needed(meta)
        fields = [t for t in meta["fields"] if len(finished.get(t, set())) < needed]
        times = {t: when for (n, t), when in opened.items() if n == name}
        last_tile = max(times, key=times.get) if times else None
        order = list(meta["fields"])
        at = order.index(last_tile) if last_tile in order else -1
        left = set(fields)
        after = [t for t in order[at + 1:] + order[:max(0, at)] if t in left]
        near = ([last_tile] if last_tile in left else []) + after[:NEAR]
        out.append(Session(meta, fields, near, max(times.values()) if times else 0.0, meta.get("stain") == "AT8", rounds.is_simple(meta)))
    out.sort(key=lambda s: -s.last)  # the most recently worked on first, then in the order of the sessions folder
    return out


def _known(state, tile_ids) -> dict[str, dict]:
    tiles = state.tiles
    if tiles.empty:
        return {}
    rows = tiles[tiles.tile_id.isin(set(tile_ids))].drop_duplicates("tile_id")
    return {str(r["tile_id"]): r for r in rows.to_dict(orient="records")}


def plan(state, now: float | None = None) -> tuple[list[Unit], dict]:
    """Every unit there is to make for one project, most urgent first (made or not: each looks before it makes), and the
    counts for the status."""
    now = time.time() if now is None else now
    opened = _opened(state)
    sessions = _sessions(state, opened)
    every = list(dict.fromkeys(t for s in sessions for t in s.fields))
    info = _known(state, every)
    by_image: dict[str, list[str]] = {}
    for tile_id in every:
        if tile_id in info:
            by_image.setdefault(str(info[tile_id]["image_path"]), []).append(tile_id)
    ladders = {t: s.simple for s in sessions if s.at8 for t in s.fields}  # tile -> wide
    units: list[Unit] = []
    seen_images: set[str] = set()

    def core(tile_id: str, tier: int) -> None:
        row = info.get(tile_id)
        if row is None or str(row["image_path"]) in seen_images:
            return
        image = str(row["image_path"])
        seen_images.add(image)
        units.append(Unit(tier, "pictures", state, f"{row.get('core_id', '')} {row.get('stain', '')}: field pictures", tuple(by_image[image]), image))
        units.append(Unit(tier, "thumbs", state, f"{row.get('core_id', '')} {row.get('stain', '')}: core thumbnails", (tile_id,), image))

    def field(tile_id: str, tier: int, sam: bool = True) -> None:
        if tile_id not in info:
            return
        if sam:
            units.append(Unit(tier, "sam", state, f"{tile_id}: Segment Anything features", (tile_id,)))
        if tile_id in ladders:
            units.append(Unit(tier, "ladder", state, f"{tile_id}: thread ladder", (tile_id,), wide=ladders[tile_id]))

    recent = sorted(((when, tile) for (name, tile), when in opened.items() if now - when < RECENT_SECONDS and tile in info), reverse=True)
    for _, tile_id in recent:  # 1: the cores people have open
        core(tile_id, 1)
    for _, tile_id in recent:
        field(tile_id, 1)
    for session in sessions:  # 2: the next fields of every open session
        for tile_id in session.near:
            core(tile_id, 2)
            field(tile_id, 2)
    matter_sessions = _matter(state, now)
    for _, _, near in matter_sessions:  # 2: the Grey / white matter cores people open next
        for core_id, stains in near:
            units.append(Unit(2, "core_pictures", state, f"{core_id}: Grey / white matter pictures", tuple(stains), core_id))
    units.append(Unit(3, "caches", state, "the step overview, the labelling tiles and the labels overview"))
    for slide_id in _slides(state):  # a whole-slide project: each slide's picture, put together from its regions
        units.append(Unit(3, "slide", state, f"{slide_id}: whole-slide picture", image_path=slide_id))
    for session in sessions:  # 4: every other field
        for tile_id in session.fields:
            core(tile_id, 4)
        for tile_id in session.fields:
            if tile_id in ladders:
                field(tile_id, 4, sam=False)
    for _, todo, near in matter_sessions:  # 4: every other core still to do
        first = {c for c, _ in near}
        for core_id, stains in todo:
            if core_id not in first:
                units.append(Unit(4, "core_pictures", state, f"{core_id}: Grey / white matter pictures", tuple(stains), core_id))
    near = {t for s in sessions for t in s.near}
    from stainid.api import render  # noqa: PLC0415

    pictures = render.fields_made(state)
    matter_images = len({(c, s) for _, todo, _ in matter_sessions for c, stains in todo for s in stains})
    estimate = (len(info) * PICTURE_BYTES + len(ladders) * LADDER_BYTES + len(by_image) * THUMB_BYTES + len(near) * SAM_BYTES
                + matter_images * CORE_PICTURE_BYTES)
    counts = {"sessions": len(sessions) + len(matter_sessions), "fields": len(info), "fields_ready": sum(t in pictures for t in info), "cores": len(by_image),
              "ladder_fields": len(ladders), "near_fields": len(near), "matter_images": matter_images, "estimate_bytes": int(estimate)}
    return units, counts


def _matter(state, now: float) -> list[tuple[dict, list[tuple[str, list[str]]], list[tuple[str, list[str]]]]]:
    """Per open Grey / white matter session: its cores still to do in the picker's order (each with its stains), and
    the ones to make ready first: the `MATTER_NEAR` after the core each person opened last (`RECENT_SECONDS`), and the
    first ones of the picker (where "Start" begins). A whole slide's picture is made as a slide (tier 3)."""
    from stainid.api import slide_units  # noqa: PLC0415
    from stainid.api.routes import matter as routes  # noqa: PLC0415
    from stainid.labels import areas, matter, picker, sessions  # noqa: PLC0415

    out = []
    reviews = state.project.output("reviews")
    opened = routes.opened(state.id)
    for name in sessions.names(reviews):
        try:
            meta = sessions.meta(reviews, name)
        except Exception:  # noqa: BLE001 - a half-written session
            continue
        if meta.get("practice") or not matter.is_matter(meta) or not meta.get("open"):
            continue
        done = areas.finished(meta, routes.session_checked(meta))
        order = [c for c in routes.picker_order(meta) if slide_units.slide_unit(c) is None]
        stains = meta.get("stains_of") or {}
        left = {c for c in order if len(done.get(c, set())) < areas.needed(meta, c)}
        starts = [core for core, when in sorted(opened.get(name, {}).values(), key=lambda v: -v[1]) if now - when < RECENT_SECONDS]
        near: list[str] = []
        for start in [*starts, None]:
            for core_id in [c for c in picker.rotated(order, start) if c in left][:MATTER_NEAR]:
                if core_id not in near:
                    near.append(core_id)
        out.append((meta, [(c, list(stains.get(c, []))) for c in order if c in left], [(c, list(stains.get(c, []))) for c in near]))
    return out


def _slides(state) -> list[str]:
    from stainid.api import slide_pyramid  # noqa: PLC0415

    try:
        return list(slide_pyramid.slides(state))
    except Exception:  # noqa: BLE001  (a project without a core image table yet)
        return []


# ---------------------------------------------------------------- making one unit

def _sam_ready(state) -> bool:
    from stainid.api import sam_embeddings  # noqa: PLC0415
    from stainid.masks import clicks  # noqa: PLC0415

    weights = sam_embeddings.checkpoint(state)
    return weights.exists() and clicks.encodes_on_gpu(weights)


def done(unit: Unit) -> bool:
    """Whether a unit is made already (looked at without making anything)."""
    from stainid.api import pyramid, render, sam_embeddings  # noqa: PLC0415
    from stainid.labels import thread_threshold  # noqa: PLC0415

    state = unit.state
    if unit.kind == "pictures":
        made = render.fields_made(state)
        return all(t in made for t in unit.tiles)
    if unit.kind == "thumbs":
        tile = state.tile(unit.tiles[0])
        return pyramid.thumbnail_state(state, str(tile["core_id"]), str(tile["stain"]), 512) == "ready"
    if unit.kind == "sam":
        return not _sam_ready(state) or sam_embeddings.marker_path(state, sam_embeddings.field_stem(state, unit.tiles[0])).exists()
    if unit.kind == "ladder":
        return thread_threshold.ready(state.project, state.tile(unit.tiles[0]), unit.wide)
    if unit.kind == "caches":
        return time.monotonic() - _CACHES_WARMED.get(state.id, -1e9) < PLAN_SECONDS
    if unit.kind == "slide":
        from stainid.api import slide_pyramid  # noqa: PLC0415

        return slide_pyramid.made(state, unit.image_path)
    if unit.kind == "core_pictures":
        return all(pyramid.ready(state, unit.image_path, stain) for stain in unit.tiles)
    return True


_CACHES_WARMED: dict[str, float] = {}


def make(unit: Unit, stop) -> None:
    """Make one unit (`stop()`: something else wants the computer; looked at between the unit's own steps)."""
    from stainid.api import pyramid, render, sam_embeddings  # noqa: PLC0415
    from stainid.api.routes.annotate import MID_PX  # noqa: PLC0415
    from stainid.labels import thread_threshold  # noqa: PLC0415

    state = unit.state
    if unit.kind == "pictures":
        render.field_jpegs(state, list(unit.tiles))
    elif unit.kind == "thumbs":
        tile = state.tile(unit.tiles[0])
        for size in (512, MID_PX):
            if stop():
                return
            pyramid.thumbnail(state, str(tile["core_id"]), str(tile["stain"]), size)
    elif unit.kind == "sam":
        sam_embeddings._make_field(state, unit.tiles[0], False, stop)
    elif unit.kind == "ladder":
        thread_threshold.field_ladder(state.project, state.tile(unit.tiles[0]), unit.wide)
    elif unit.kind == "caches":
        _warm_caches(state, stop)
        _CACHES_WARMED[state.id] = time.monotonic()
    elif unit.kind == "slide":
        from stainid.api import slide_pyramid  # noqa: PLC0415

        if slide_pyramid.build(state, unit.image_path, stop) is None:
            return
        for size in slide_pyramid.THUMBS:
            if stop():
                return
            slide_pyramid.thumbnail(state, unit.image_path, size)
    elif unit.kind == "core_pictures":
        _core_pictures(state, unit.image_path, list(unit.tiles), stop)


def _core_pictures(state, core_id: str, stains: list[str], stop) -> None:
    """A Grey / white matter core ready to open: each stain image's pyramid (one decode each, looked at between them),
    the carry of the white matter from the first stain to the others (`registration.cores`, kept in the cache when
    computed) and detection's neurons for the guide (kept in memory, `routes.areas.object_rows`)."""
    from stainid.api import pyramid  # noqa: PLC0415
    from stainid.api.routes import areas as area_routes  # noqa: PLC0415
    from stainid.labels import matter  # noqa: PLC0415
    from stainid.registration import cores as registration  # noqa: PLC0415

    for stain in stains:
        if stop():
            return
        pyramid.build_ahead(state, core_id, stain)
    source = matter.source_of(stains)
    for stain in stains:
        if stain != source and not stop():
            try:
                registration.carry(state, core_id, source, stain)
            except (KeyError, FileNotFoundError, ValueError):
                pass
    if source and not stop():
        area_routes.object_rows(state.project, core_id, source)


def _warm_caches(state, stop) -> None:
    from stainid.labels import overview  # noqa: PLC0415
    from stainid.workflows.steps import describe  # noqa: PLC0415

    for warm in (lambda: describe(state.project), lambda: state.labelling_tiles):
        if stop():
            return
        try:
            warm()
        except Exception:  # noqa: BLE001 - a half-set-up project: its page says what is wrong
            pass
    tiles = state.tiles
    for stain in sorted(set(tiles.stain)) if "stain" in tiles else []:
        if stop():
            return
        try:
            overview.cached_stain_overview(state.project, stain, state.project.output("reviews"), tiles)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- the space it may take

_COUNTS: dict[str, dict] = {}  # project id -> the counts of its last plan
_SAM_LOOK: dict[str, tuple[float, bool]] = {}
SAM_KIND = "sam_embeddings"


def sam_cap(state) -> int:
    """The space Segment Anything features of fields far off may take: what is left of `SHARE` of the cache limit once
    every open field's picture, thread ladder and core thumbnails are counted (each field's picture first: every open
    needs it, the features only the outline tool)."""
    from stainid import disk_cache  # noqa: PLC0415

    counts = _COUNTS.get(state.id) or {}
    others = counts.get("fields", 0) * PICTURE_BYTES + counts.get("ladder_fields", 0) * LADDER_BYTES + counts.get("cores", 0) * THUMB_BYTES
    return max(0, int(SHARE * disk_cache.limit_bytes(state.root) - others))


def sam_over_cap(state) -> bool:
    """Whether the features made ahead for every field of a session (`sam_embeddings.prepare_session`) are to stop: they
    take their share already (looked at every 30 s)."""
    from stainid import disk_cache  # noqa: PLC0415

    when, over = _SAM_LOOK.get(state.id, (-1e9, False))
    if time.monotonic() - when > 30:
        try:
            over = disk_cache.kind_bytes(state.root, SAM_KIND) > sam_cap(state)
        except Exception:  # noqa: BLE001 - no index yet
            over = False
        _SAM_LOOK[state.id] = (time.monotonic(), over)
    return over


def room(state) -> tuple[bool, dict]:
    """Whether tier 4 may make more (the project's caches under `SHARE` of their limit, the drive above its floor).
    When over: first the least recently used Segment Anything features beyond their share (`sam_cap`) are deleted, then
    whatever nobody used for `STALE_DAYS`."""
    from stainid import disk_cache  # noqa: PLC0415

    limit = disk_cache.limit_bytes(state.root)
    used = disk_cache.used_bytes(state.root)
    if used > SHARE * limit:
        disk_cache.evict_kind(state.root, SAM_KIND, sam_cap(state))
        used = disk_cache.used_bytes(state.root)
    if used > SHARE * limit:
        disk_cache.evict(state.root, int(SHARE * limit * disk_cache.LOW_WATER), keep_seconds=STALE_DAYS * 86400)
        used = disk_cache.used_bytes(state.root)
    drive = disk_cache.drive_space(disk_cache.cache_root(state.root)) or {}
    low = bool(drive) and drive["free_bytes"] < 1.5 * drive["min_free_bytes"]
    return used <= SHARE * limit and not low, {"used_bytes": int(used), "limit_bytes": int(limit), "drive_low": low}


# ---------------------------------------------------------------- the thread

def _lowest_priority() -> None:
    tid = threading.get_native_id()
    try:
        os.setpriority(os.PRIO_PROCESS, tid, 19)
    except (AttributeError, OSError):
        pass
    try:  # the idle disk class (Linux ioprio_set: IOPRIO_WHO_PROCESS, class 3 << 13)
        libc = ctypes.CDLL(None, use_errno=True)
        number = {"x86_64": 251, "aarch64": 30}.get(os.uname().machine)
        if number:
            libc.syscall(number, 1, tid, 3 << 13)
    except (OSError, AttributeError):
        pass


def _set(**changes) -> None:
    with _LOCK:
        _STATUS.update(changes)


def status() -> dict:
    with _LOCK:
        out = dict(_STATUS)
    out["paused"] = paused()
    if out["paused"]:
        out["state"], out["why"] = "paused", "paused on the Run page"
    out["running"] = _THREAD is not None and _THREAD.is_alive()
    return out


def _projects() -> list:
    from stainid.api import state as state_module  # noqa: PLC0415

    return [s for s in state_module.open_states() if s.id != state_module.TUTORIAL]


def _plans() -> list[tuple[object, list[Unit], dict]]:
    out = []
    for state in _projects():
        try:
            from stainid.api.state import use_project  # noqa: PLC0415

            with use_project(state.id):
                units, counts = plan(state)
            _COUNTS[state.id] = counts
        except Exception as error:  # noqa: BLE001 - a project that cannot be read now: tried again at the next plan
            log.warning("preparing ahead: no plan for %s: %s", getattr(state, "root", "?"), error)
            continue
        out.append((state, units, counts))
    return out


def _signature() -> tuple:
    from stainid.api.routes import annotate, matter  # noqa: PLC0415

    cores = tuple(sorted((sid, name, person, core) for (sid, name), people in list(matter._OPENED.items()) for person, (core, _) in list(people.items())))
    with annotate._BUSY_LOCK:
        return tuple(sorted((sid, name, tile) for ((sid, name), tile) in annotate._BUSY)) + cores


def _units(plans) -> Iterator[Unit]:
    """The units of every project, tier by tier (a project's tier 1 before another's tier 2)."""
    for tier in (1, 2, 3, 4):
        for _, units, _ in plans:
            yield from (u for u in units if u.tier == tier)


def run_once(stop_after: float | None = None, quiet_look: bool = True) -> int:
    """One pass over everything there is to make (tests, and the thread). Returns how many units it made."""
    made = 0
    _REPLAN.clear()
    plans = _plans()
    total = {"sessions": sum(c["sessions"] for *_, c in plans), "fields": sum(c["fields"] for *_, c in plans),
             "fields_ready": sum(c["fields_ready"] for *_, c in plans), "estimate_bytes": sum(c["estimate_bytes"] for *_, c in plans), "near_fields": sum(c["near_fields"] for *_, c in plans)}
    _set(**total)
    signature, planned, looked = _signature(), time.monotonic(), -1e9  # a quiet look before the first unit
    full: dict[str, bool] = {}
    for unit in _units(plans):
        if signature != _signature() or _REPLAN.is_set() or time.monotonic() - planned > PLAN_SECONDS or (stop_after and made >= stop_after):
            break  # people opened another field: plan again
        try:
            if done(unit):
                continue
        except Exception:  # noqa: BLE001 - a tile that is gone
            continue
        if unit.tier == 4:
            if unit.state.id not in full:
                fits, space = room(unit.state)
                full[unit.state.id] = not fits
                _set(**space)
            if full[unit.state.id]:
                _set(state="full", why="the cache is nearly full (Settings, Cache): only the fields people open next are prepared")
                continue
        while True:  # wait for the computer to be free
            why = busy_now()
            if not why and quiet_look and time.monotonic() - looked > LOOK_EVERY:
                why = look(unit.state.root)
                looked = time.monotonic()
            if not why:
                break
            _set(state="paused" if why == "paused" else "waiting", why=why)
            _WAKE.wait(WAIT_SECONDS)
            _WAKE.clear()
            if signature != _signature() or _REPLAN.is_set():
                return made
        _set(state="preparing", why="", current=unit.label, tier=unit.tier)
        try:
            from stainid.api import render  # noqa: PLC0415
            from stainid.api.state import use_project  # noqa: PLC0415

            missing = [t for t in unit.tiles if not render.field_path(unit.state, t).exists()] if unit.kind == "pictures" else []
            with use_project(unit.state.id):
                make(unit, lambda: bool(busy_now()))
            made += 1
            with _LOCK:
                _STATUS["made"] = _STATUS.get("made", 0) + 1
                _STATUS["fields_ready"] = _STATUS.get("fields_ready", 0) + sum(render.field_path(unit.state, t).exists() for t in missing)
        except Exception as error:  # noqa: BLE001 - one unit that fails (a missing image): the others go on
            log.warning("preparing ahead: %s not made: %s", unit.label, error)
    return made


def _loop() -> None:
    _lowest_priority()
    time.sleep(30)  # the server's start (its warm-up, the first pages) first
    while True:
        try:
            if paused():
                _set(state="paused", why="paused on the Run page")
                _WAKE.wait(30)
                _WAKE.clear()
                continue
            made = run_once()
            if not made:
                with _LOCK:
                    if _STATUS.get("state") != "full":
                        _STATUS.update(state="done", why="", current="")
                _WAKE.wait(60)
                _WAKE.clear()
        except Exception:  # noqa: BLE001 - never ends: the next pass tries again
            log.exception("preparing ahead failed")
            time.sleep(60)


def wake() -> None:
    """Something new to prepare (a session made, fields added): plan again now rather than at the next pass."""
    _REPLAN.set()
    _WAKE.set()


def start() -> None:
    """Start the thread (the app's start; not with STAINID_AHEAD=0)."""
    global _THREAD
    if os.environ.get("STAINID_AHEAD", "1") == "0":
        _set(state="off", why="turned off for this server (STAINID_AHEAD=0)")
        return
    with _LOCK:
        if _THREAD is None or not _THREAD.is_alive():
            _THREAD = threading.Thread(target=_loop, name="stainid-ahead", daemon=True)
            _THREAD.start()
