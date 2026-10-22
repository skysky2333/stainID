"""The simple classes of "Fix the field (simple)": a few plain classes per stain, worked in passes, one class at a time.

A simple session (a "Fix the field" session with `mode: "simple"`, `labels.rounds`) asks fewer, plainer questions than
the detailed one: 6E10 "diffuse amyloid beta", "diffuse plaque", "plaque with cores"; AT8 "tangles", "neuritic plaque",
"neurite". People pick a class (a pass), click the candidates that belong to it (the ones detection put there are lit
already), click empty tissue to add a missed one, then go on to the next class. On Finish field every candidate of
the field nobody put in a class is saved as `NONE` ("not an object"), as the detailed mode confirms what was left.

The simple labels are their own vocabulary, kept as they are: task `TASK` ("simple"), `value` the simple class's id.
They are never rewritten as detailed classes in the label log. A class of kind `WAND` (AT8 "neurite") is not a
candidate class: its pass selects thread segments with a brush (one label per field, task `AREA_TASK`, the chosen
segments' lines in `points`): a stroke takes every segment under it of the thread network at the slider's thread level
(the thread ladder of threads mode, `labels.thread_threshold`: higher on the slider is a lower factor, so more and
fainter lines count), Shift-painting takes them out; candidates left in no class that lie on those segments are saved
as that class.

Training (`training.train.labelled`) and the Models page read them through `training_rows`: each simple class maps onto
a detailed class (identity, optional subtype) or "unused" (`mapping`, stainid.yaml `labels.simple.<stain>.training`,
`DEFAULT_TRAINING` until someone decides), and every model records the mapping it used. Detection's calls map onto the
simple classes (`SimpleClass.starts_from`) only for where a field starts.

Each stain's simple classes can be changed in stainid.yaml, `labels.simple.<stain>.classes`: a list of {id, name,
description, key, colour, kind, starts_from, hidden}. The built-in ones (`DEFAULTS`) come back when left out.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from stainid.labels import vocabulary as vocab

TASK = "simple"
AREA_TASK = "simple_area"
NONE = "none"  # "not an object": what a candidate left in no class is saved as
NONE_NAME = "not an object"
OBJECTS, WAND = "objects", "wand"  # a class of candidates; a class of thread segments (the neurite brush)
MODE = "simple"  # `mode` of a simple session's meta
UNUSED = ""  # a simple class that trains nothing
NEAR_SEGMENT_UM = 3.0  # a candidate within this of a chosen thread segment (its centre) is in the wand's class
WAND_REACH_UM = (0.0, 40.0, 8.0)  # the magic wand of earlier pages (µm between segments taken together); the brush has none


@dataclass(frozen=True)
class SimpleClass:
    id: str
    name: str
    description: str = ""
    key: str = ""
    colour: str = "#ffe600"
    kind: str = OBJECTS
    starts_from: tuple[str, ...] = ()  # detailed calls ("identity" or "identity/subtype") a candidate starts in this class from
    hidden: bool = False
    builtin: bool = False

    def to_config(self) -> dict:
        return {"id": self.id, "name": self.name, "description": self.description, "key": self.key, "colour": self.colour, "kind": self.kind,
                "starts_from": list(self.starts_from), "hidden": self.hidden}

    def to_api(self) -> dict:
        return self.to_config() | {"builtin": self.builtin}


def _c(id: str, name: str, description: str, key: str, colour: str, starts_from: tuple[str, ...] = (), kind: str = OBJECTS) -> SimpleClass:
    return SimpleClass(id, name, description, key, colour, kind, starts_from, builtin=True)


DEFAULTS: dict[str, tuple[SimpleClass, ...]] = {
    "6E10": (
        _c("diffuse_abeta", "diffuse amyloid beta", "Faint, loose brown amyloid with no clear edge: a haze rather than a plaque.", "1", "#b98cff"),
        _c("diffuse_plaque", "diffuse plaque", "A plaque of loose, granular or fleecy brown with an edge you can see, but no dense core.", "2", "#00e5ff",
           ("plaque/diffuse", "plaque/small", "plaque")),
        _c("cored_plaque", "plaque with cores", "A plaque with a solid, sharply bounded dark brown core (with or without a halo).", "3", "#ffe600",
           ("plaque/compact", "plaque/cored-neuritic")),
    ),
    "AT8": (
        _c("tangle", "tangles", "A neuron whose cell body holds brown tau: grainy, flame-shaped or globe-shaped.", "1", "#ffe600", ("tau+ neuron",)),
        _c("neuritic_plaque", "neuritic plaque", "A ring or cluster of swollen brown nerve endings, with no cell body in the middle.", "2", "#b98cff",
           ("neuritic plaque",)),
        _c("neurite", "neurite", "Brown threads and fibres (neuropil threads): painted with the brush over the threads detection found.", "3",
           "#7cff3a", kind=WAND),
    ),
    "NeuN": (
        _c("neuron", "neuron", "A brown (DAB) neuron: its nucleus and often its cell body are stained.", "1", "#ffe600", ("NeuN+ neuron",)),
    ),
}
DEFAULT_TRAINING: dict[str, dict[str, str]] = {  # simple class -> "identity" or "identity/subtype" of the detailed classes, "" unused
    "6E10": {"diffuse_abeta": UNUSED, "diffuse_plaque": "plaque/diffuse", "cored_plaque": "plaque/compact"},
    "AT8": {"tangle": "tau+ neuron", "neuritic_plaque": "neuritic plaque", "neurite": "thread"},
    "NeuN": {"neuron": "NeuN+ neuron"},
}


@dataclass(frozen=True)
class SimpleVocabulary:
    stain: str
    classes: tuple[SimpleClass, ...]
    training: dict = field(default_factory=dict, compare=False, hash=False)  # simple id -> detailed "identity[/subtype]" or "" (unused)

    @property
    def visible(self) -> list[SimpleClass]:
        return [c for c in self.classes if not c.hidden]

    def get(self, ref: str) -> SimpleClass | None:
        return next((c for c in self.classes if c.id == ref), None) or next((c for c in self.classes if c.name.lower() == str(ref).lower()), None)

    def name(self, ref: str) -> str:
        if ref == NONE:
            return NONE_NAME
        found = self.get(ref)
        return found.name if found else str(ref)

    def start(self, identity: str, subtype: str = "") -> str:
        """The simple class a candidate starts in from detection's (detailed) call: the first class naming the call with
        its subtype, else the identity alone; `NONE` when no class does."""
        full, alone = f"{identity}/{subtype}" if subtype else identity, identity
        for wanted in (full, alone):
            for c in self.visible:
                if c.kind == OBJECTS and wanted in c.starts_from:
                    return c.id
        return NONE

    def mapped(self, ref: str) -> tuple[str, str] | None:
        """The detailed (identity, subtype) a simple class trains as, None when unused."""
        target = self.training.get(ref, UNUSED)
        if not target:
            return None
        identity, _, subtype = target.partition("/")
        return identity, subtype

    def to_api(self) -> dict:
        return {"stain": self.stain, "classes": [c.to_api() for c in self.visible], "none": {"id": NONE, "name": NONE_NAME},
                "training": dict(self.training), "wand": {"from": WAND_REACH_UM[0], "to": WAND_REACH_UM[1], "start": WAND_REACH_UM[2]}}


def _key(value) -> str:
    return vocab.text(value).lower()[:1]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "class"


def default_training(stain: str, words: vocab.Vocabulary | None = None) -> dict[str, str]:
    """The proposed mapping onto the detailed classes, with "not an object" as the stain's reject."""
    words = words or vocab.defaults(stain)
    return {**DEFAULT_TRAINING.get(stain, {}), NONE: words.reject}


def from_config(stain: str, entry: dict | None, words: vocab.Vocabulary | None = None) -> SimpleVocabulary:
    """A stain's simple classes from its `labels.simple.<stain>` entry ({classes, training}); the built-in classes keep
    their id and kind and come back when left out."""
    entry = entry or {}
    builtin = {c.id: c for c in DEFAULTS.get(stain, ())}
    out: list[SimpleClass] = []
    for e in entry.get("classes") or []:
        ident = vocab.text(e.get("id")) or _slug(vocab.text(e.get("name")))
        if not ident or ident == NONE or any(c.id == ident for c in out):
            continue
        base = builtin.get(ident)
        starts = tuple(vocab.text(s) for s in (e.get("starts_from") if e.get("starts_from") is not None else (base.starts_from if base else ())) if vocab.text(s))
        kind = base.kind if base else (vocab.text(e.get("kind")) if vocab.text(e.get("kind")) in (OBJECTS, WAND) else OBJECTS)
        out.append(SimpleClass(ident, vocab.text(e.get("name")) or (base.name if base else ident), vocab.text(e.get("description")) or (base.description if base else ""),
                               _key(e.get("key")) or (base.key if base else ""), vocab.text(e.get("colour")) or (base.colour if base else "#ffe600"),
                               kind, starts, bool(e.get("hidden")), builtin=base is not None))
    out += [c for c in builtin.values() if not any(o.id == c.id for o in out)]
    training = default_training(stain, words) | {str(k): vocab.text(v) for k, v in (entry.get("training") or {}).items()}
    known = {c.id for c in out} | {NONE}
    return SimpleVocabulary(stain, tuple(out), {k: v for k, v in training.items() if k in known})


_OF: dict[tuple, SimpleVocabulary] = {}


def of(project, stain: str) -> SimpleVocabulary:
    """The project's simple classes of this stain and their training mapping (stainid.yaml `labels.simple`)."""
    entry = ((project.config.get("labels") or {}).get("simple") or {}).get(stain)
    words = vocab.of(project, stain)
    key = (stain, repr(entry), repr(words.reject))
    if key not in _OF:
        _OF[key] = from_config(stain, entry, words)
    return _OF[key]


def problems(simple: SimpleVocabulary, words: vocab.Vocabulary) -> list[str]:
    """What is wrong with a stain's simple classes or their mapping, in plain words (empty when they can be saved)."""
    out = []
    names = [c.name.lower() for c in simple.classes]
    if len(set(names)) != len(names):
        out.append("Two simple classes have the same name.")
    keys = [c.key for c in simple.visible if c.key]
    if len(set(keys)) != len(keys):
        out.append("Two simple classes have the same key.")
    if any(k not in "123456789" for k in keys):
        out.append("The keys of simple classes are numbers (1–9).")
    for ref, target in simple.training.items():
        if not target:
            continue
        identity, _, subtype = target.partition("/")
        found = words.identity(identity)
        if found is None or (subtype and found.subtype(subtype) is None):
            out.append(f"{simple.name(ref)} trains as {target}, which is not a {simple.stain} class.")
    for c in simple.classes:
        for start in c.starts_from:
            identity, _, subtype = start.partition("/")
            found = words.identity(identity)
            if found is None or (subtype and found.subtype(subtype) is None):
                out.append(f"{c.name} starts from {start}, which is not a {simple.stain} class.")
    return out


def to_settings(simple: SimpleVocabulary) -> dict:
    return {"classes": [c.to_config() for c in simple.classes], "training": dict(simple.training)}


# ---------- simple labels for training and the Labels page ----------

def labels_of(labels: pd.DataFrame, stain: str) -> pd.DataFrame:
    """The simple class labels of this stain (not the wand's areas)."""
    if labels.empty or "task" not in labels:
        return labels.iloc[0:0]
    return labels[(labels.task == TASK) & (labels.stain == stain)]


def training_rows(labels: pd.DataFrame, simple: SimpleVocabulary) -> pd.DataFrame:
    """The simple labels of the stain as class labels of the detailed classes, by the mapping (`SimpleVocabulary.mapped`;
    classes mapped to nothing are left out): what training and evaluation read. The label log keeps the simple ones;
    each row's `meta.simple` names the simple class it came from."""
    rows = labels_of(labels, simple.stain)
    if rows.empty:
        return rows
    targets = [simple.mapped(vocab.text(v)) for v in rows.value]
    keep = [t is not None for t in targets]
    rows, targets = rows[keep], [t for t in targets if t is not None]
    if rows.empty:
        return rows
    ids = [str(i).replace(f"{TASK}:", "class:", 1) for i in rows.id]
    return rows.assign(task="class", id=ids, value=[t[0] for t in targets], subtype=[t[1] or None for t in targets],
                       meta=[{"simple": vocab.text(v)} for v in rows.value])


def with_training_rows(labels: pd.DataFrame, project, stain: str) -> pd.DataFrame:
    """The labels with the simple ones of this stain added as detailed class labels (`training_rows`)."""
    extra = training_rows(labels, of(project, stain))
    return pd.concat([labels, extra], ignore_index=True) if len(extra) else labels


def used_mapping(labels: pd.DataFrame, project, stain: str) -> dict | None:
    """The mapping a training on these labels used (None when they hold no simple labels of the stain): for its report."""
    rows = labels_of(labels, stain)
    if rows.empty:
        return None
    simple = of(project, stain)
    return {"classes": {c.id: c.name for c in simple.classes} | {NONE: NONE_NAME}, "training": dict(simple.training),
            "labels": {str(k): int(v) for k, v in rows.value.value_counts().items()}}


def counts(labels: pd.DataFrame, stain: str, simple: SimpleVocabulary) -> dict[str, int]:
    """How many simple labels of each class (by name), as the Labels page shows them."""
    rows = labels_of(labels, stain)
    return {simple.name(str(k)): int(v) for k, v in rows.value.value_counts().items()}
