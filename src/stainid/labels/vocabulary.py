"""The label classes of each stain: identity classes, each with optional subtypes.

The identity says what an object is (tau+ neuron, plaque, NeuN+ neuron, or one of the things that are not: a thread, a
vessel, an artifact). Its `role` is what the stain's identity model learns from it: `positive`, `negative` or `skip`
(not used: "unsure"). The subtype says which kind of positive (or other) object it is (pretangle, mature tangle; compact,
diffuse plaque); a later model can learn it from the objects people typed, while the identity models keep learning
exactly what they learnt before.

A label stores the identity's `id` as its value and the subtype's `id` in `subtype` (both stable: renaming a class in the
settings changes its `name` only). Labels made before subtypes existed stored one string ("tau+ neuron: tangle",
"compact plaque", "plaque"); `LEGACY` maps each onto an identity and subtype, so the label log is never rewritten.
Class maps of dataset packages may name a subtype as "identity / subtype" (or "identity: subtype"), by id or name.

Each stain's classes are built in (`DEFAULTS`) and can be changed per project in stainid.yaml, `labels.classes.<stain>`:
a list of identities as `Identity.to_config` writes them (names, descriptions, keys, an example picture, hidden or not,
subtypes). Built-in classes cannot be removed (only hidden) and keep their role, so a project's settings never change
what an identity model learns from the labels people made; classes people add choose their role.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

POSITIVE, NEGATIVE, SKIP = "positive", "negative", "skip"
ROLES = (POSITIVE, NEGATIVE, SKIP)
UNTYPED = "type not recorded"  # shown for a positive object whose type nobody gave: "plaque (type not recorded)"
TYPE_SPLIT = "people disagree on the type"  # "plaque (people disagree on the type)"
RESERVED_KEYS = {"v", "a", "o", "z", "0"}  # field-page tools (v, a, o), undo in the quick check (z), "no type" (0)
DEFAULT_SUBTYPE_WAIT_S = 1.0  # how long after an identity key a number key picks its subtype


@dataclass(frozen=True)
class Subtype:
    id: str
    name: str
    description: str = ""
    key: str = ""  # a key that answers identity and subtype at once (the subtype's place, 1–9, works after the identity key)
    example: str = ""
    hidden: bool = False
    builtin: bool = False

    def to_config(self) -> dict:
        return {"id": self.id, "name": self.name, "description": self.description, "key": self.key, "example": self.example, "hidden": self.hidden}


@dataclass(frozen=True)
class Identity:
    id: str
    name: str
    role: str
    description: str = ""
    key: str = ""
    example: str = ""
    hidden: bool = False
    builtin: bool = False
    subtypes: tuple[Subtype, ...] = field(default_factory=tuple)

    def subtype(self, ref: str) -> Subtype | None:
        """A subtype by id, else by name (case does not matter)."""
        return next((s for s in self.subtypes if s.id == ref), None) or next((s for s in self.subtypes if s.name.lower() == ref.lower()), None)

    def to_config(self) -> dict:
        return {"id": self.id, "name": self.name, "role": self.role, "description": self.description, "key": self.key, "example": self.example,
                "hidden": self.hidden, "subtypes": [s.to_config() for s in self.subtypes]}


def _s(id: str, name: str, description: str, key: str = "", hidden: bool = False) -> Subtype:
    return Subtype(id, name, description, key, hidden=hidden, builtin=True)


def _i(id: str, name: str, role: str, description: str, key: str, subtypes: tuple[Subtype, ...] = ()) -> Identity:
    return Identity(id, name, role, description, key, builtin=True, subtypes=subtypes)


ARTIFACT = ("artifact or unclear", "artifact or unclear", NEGATIVE, "A fold, tear, edge, bubble, stain precipitate or blurred spot, or too unclear to say.")
UNSURE = ("unsure", "unsure", SKIP, "You cannot decide. The object is left out of training.")

DEFAULTS: dict[str, tuple[Identity, ...]] = {
    "AT8": (
        _i("tau+ neuron", "tau+ neuron", POSITIVE, "A neuron whose cell body holds brown (AT8) tau.", "1", (
            _s("pretangle", "pretangle", "Diffuse or grainy brown in the cell body around a visible nucleus; the cell keeps its shape.", "q"),
            _s("mature tangle", "mature tangle", "A dense brown flame- or globe-shaped mass filling the cell body, often hiding the nucleus.", "w"),
            _s("ghost tangle", "ghost (extracellular) tangle", "A loose, paler brown tangle with no cell or nucleus around it: the neuron has died.", "e"),
        )),
        _i("neuritic plaque", "neuritic plaque (tau crown)", NEGATIVE,
           "A ring or cluster of swollen brown nerve endings around a plaque, with no cell body in the middle.", "2"),
        _i("astroglial tau", "astroglial tau", NEGATIVE, "Brown tau in an astrocyte rather than a neuron: many fine processes around a small nucleus.", "3", (
            _s("tufted astrocyte", "tufted astrocyte", "Tufts of fine brown fibres radiating from the cell."),
            _s("thorn-shaped astrocyte", "thorn-shaped astrocyte", "Short thorn-like brown processes, often near vessels, under the brain surface or in white matter."),
        )),
        _i("thread", "thread or dystrophic neurite", NEGATIVE, "Short brown threads, dots or swollen fibres in the tissue, without a cell body.", "4"),
        _i("not tau", "not tau", NEGATIVE, "Nothing stained for tau here: a blue nucleus, a vessel, background.", "5"),
        _i(*ARTIFACT, "6"),
        _i(*UNSURE, "7"),
    ),
    "6E10": (
        _i("plaque", "plaque", POSITIVE, "A brown amyloid deposit between the cells, at least about 10 µm across.", "1", (
            _s("compact", "compact (dense-core)", "A dense, well-defined brown core.", "q"),
            _s("diffuse", "diffuse", "A loose, fluffy brown deposit without a dense core.", "w"),
            _s("cored-neuritic", "cored-neuritic", "A dense core with a halo of swollen nerve endings around it, if you can see them.", "e"),
            _s("small", "small (too small to type)", "A real plaque only 10–15 µm across: too small to tell its type.", "r"),
        )),
        _i("CAA", "CAA / vascular amyloid", NEGATIVE, "Brown amyloid in the wall of a blood vessel (cerebral amyloid angiopathy).", "2"),
        _i("intraneuronal", "intraneuronal staining", NEGATIVE,
           "Brown inside a neuron’s cell body: APP or Aβ in the cell, which 6E10 also stains. Not a plaque.", "3"),
        _i("not a plaque", "not a plaque", NEGATIVE, "A cell, a speck under 10 µm, a fibre or background.", "4"),
        _i(*ARTIFACT, "5"),
        _i(*UNSURE, "6"),
    ),
    "NeuN": (
        _i("NeuN+ neuron", "NeuN+ neuron", POSITIVE, "A brown (DAB) neuron: its nucleus and often its cell body are stained.", "1", (
            _s("strong", "strongly stained", "Dark brown nucleus and cell body.", "q"),
            _s("weak", "weakly stained", "Pale brown, but clearly a neuron.", "w"),
            _s("large pyramidal", "large pyramidal neuron", "A large triangular neuron with a thick dendrite towards the brain surface.", "e", hidden=True),
        )),
        _i("not a neuron", "not a neuron", NEGATIVE, "A blue-only nucleus (glia, the lining of a vessel) or background.", "2"),
        _i(*ARTIFACT, "3"),
        _i(*UNSURE, "4"),
    ),
}
# Any other stain (e.g. of a dataset from another lab) starts from these.
GENERIC: tuple[Identity, ...] = (
    _i("object", "object", POSITIVE, "One of the objects this stain is analysed for.", "1"),
    _i("not an object", "not an object", NEGATIVE, "Something else: another cell, a vessel, background.", "2"),
    _i(*ARTIFACT, "3"),
    _i(*UNSURE, "4"),
)

# Values of labels made before subtypes existed (and the AI's calls in the candidate pools): identity and subtype.
LEGACY: dict[str, dict[str, tuple[str, str]]] = {
    "AT8": {"tau+ neuron: tangle": ("tau+ neuron", "mature tangle"), "tau+ neuron: pretangle": ("tau+ neuron", "pretangle"),
            "not a tau+ neuron": ("not tau", ""), "tau+ neuron (type not recorded)": ("tau+ neuron", "")},
    "6E10": {"compact plaque": ("plaque", "compact"), "diffuse plaque": ("plaque", "diffuse"), "small plaque": ("plaque", "small"),
             "plaque (type not recorded)": ("plaque", ""), "vascular": ("CAA", "")},
    "NeuN": {},
}
_SPLIT = re.compile(r"\s*[/:]\s*")  # "identity / subtype", "identity/subtype", "identity: subtype"


def text(value) -> str:
    """A stored value as text: "" for missing ones (None, NaN)."""
    return "" if value is None or value != value else str(value).strip()


@dataclass(frozen=True)
class Vocabulary:
    stain: str
    classes: tuple[Identity, ...]
    subtype_wait_s: float = DEFAULT_SUBTYPE_WAIT_S
    _resolved: dict = field(default_factory=dict, init=False, compare=False, hash=False, repr=False)  # (value, subtype) -> resolve()

    def identity(self, ref: str) -> Identity | None:
        """An identity by id, else by name (case does not matter)."""
        return next((c for c in self.classes if c.id == ref), None) or next((c for c in self.classes if c.name.lower() == ref.lower()), None)

    @property
    def visible(self) -> list[Identity]:
        return [c for c in self.classes if not c.hidden]

    def resolve(self, value, subtype=None) -> tuple[str, str]:
        """(identity id, subtype id) of a stored label: "" for no identity (an empty value) or no subtype. A value no class
        knows stays as it is (and counts as negative, as it always did)."""
        key = (text(value), text(subtype))
        if key not in self._resolved:
            self._resolved[key] = _resolve(self, *key)
        return self._resolved[key]

    def role(self, identity: str) -> str:
        if not identity:
            return SKIP
        found = self.identity(identity)
        return found.role if found else NEGATIVE

    def positive(self, value, subtype=None) -> bool:
        return self.role(self.resolve(value, subtype)[0]) == POSITIVE

    def skipped(self, value, subtype=None) -> bool:
        """Labels that say nothing about the object: empty, or a class with the role `skip` ("unsure")."""
        return self.role(self.resolve(value, subtype)[0]) == SKIP

    def display(self, identity: str, subtype: str = "") -> str:
        """How a class reads on the page: "tau+ neuron · pretangle", "plaque (type not recorded)", "not tau"."""
        found = self.identity(identity) if identity else None
        if found is None:
            return identity
        if subtype:
            sub = found.subtype(subtype)
            return f"{found.name} · {sub.name if sub else subtype}"
        return f"{found.name} ({UNTYPED})" if found.subtypes and found.role == POSITIVE else found.name

    def disputed(self, identity: str) -> str:
        """An object people agree is of this identity but give different types, as the page shows it."""
        found = self.identity(identity)
        return f"{found.name if found else identity} ({TYPE_SPLIT})"

    def describe(self, value, subtype=None) -> str:
        return self.display(*self.resolve(value, subtype))

    def record(self, value: str, subtype: str = "") -> dict:
        """The label fields for a class given as an id, a name, an old value or "identity / subtype": canonical ids."""
        identity, sub = self.resolve(value, subtype)
        return {"value": identity, "subtype": sub or None}

    def keymap(self, subtypes: bool = True) -> dict:
        """The keys of the labelling pages. `direct`: key -> [identity, subtype or None], each shown class's own key
        (free number keys for classes without one) and each shown subtype's key; `then`: identity -> {"1": subtype, …},
        the shown subtypes by their place, picked by a number key right after the identity's key. Without `subtypes`
        (the session does not ask for types) only the identities have keys."""
        direct: dict[str, list] = {}
        then: dict[str, dict[str, str]] = {}
        free = iter(k for k in "123456789" if k not in {c.key for c in self.classes} | {s.key for c in self.classes for s in c.subtypes})
        for c in self.visible:
            key = c.key or next(free, "")
            if key:
                direct.setdefault(key, [c.id, None])
            shown = [s for s in c.subtypes if not s.hidden] if subtypes else []
            if shown:
                then[c.id] = {str(i + 1): s.id for i, s in enumerate(shown[:9])}
            for s in shown:
                if s.key:
                    direct.setdefault(s.key, [c.id, s.id])
        return {"direct": direct, "then": then}

    def by_key(self) -> list[Identity]:
        """The classes in the order every page lists them: by their key (1, 2, … 9, then letters), classes without a key
        after them, "unsure" (skip; key 0 on the pages) last; hidden ones too, in the same rule (webapp classKeys.byKey)."""
        keys = {ident: key for key, (ident, sub) in self.keymap()["direct"].items() if sub is None}
        place = {c.id: i for i, c in enumerate(self.classes)}

        def order(c: Identity) -> tuple:
            key = keys.get(c.id, c.key or "")
            return (c.role == SKIP, not key, not key.isdigit(), int(key) if key.isdigit() else 0, key, place[c.id])

        return sorted(self.classes, key=order)

    def to_api(self) -> dict:
        """The classes as the web app uses them, with how old values map onto them."""
        return {"stain": self.stain, "subtype_wait_s": self.subtype_wait_s,
                "classes": [{**c.to_config(), "builtin": c.builtin, "subtypes": [{**s.to_config(), "builtin": s.builtin} for s in c.subtypes]}
                            for c in self.classes],
                "aliases": {k: list(v) for k, v in LEGACY.get(self.stain, {}).items()}, "untyped": UNTYPED}

    def to_config(self) -> list[dict]:
        return [c.to_config() for c in self.classes]


def _resolve(vocabulary: Vocabulary, value: str, subtype: str) -> tuple[str, str]:
    if not value:
        return "", ""
    found = vocabulary.identity(value)
    if found is None and value in LEGACY.get(vocabulary.stain, {}):
        identity, legacy_sub = LEGACY[vocabulary.stain][value]
        return identity, subtype or legacy_sub
    if found is None:
        parts = _SPLIT.split(value, maxsplit=1)
        head = vocabulary.identity(parts[0]) if len(parts) == 2 else None
        if head is None:
            return value, subtype
        found, subtype = head, subtype or parts[1]
    sub = found.subtype(subtype) if subtype else None
    return found.id, sub.id if sub else subtype


_BUILT_IN: dict[str, Vocabulary] = {}


def defaults(stain: str) -> Vocabulary:
    """The built-in classes of a stain (those of `GENERIC` for a stain stainID has none for)."""
    if stain not in _BUILT_IN:
        _BUILT_IN[stain] = Vocabulary(stain, DEFAULTS.get(stain, GENERIC))
    return _BUILT_IN[stain]


def _clean_key(value) -> str:
    return text(value)[:1].lower()


def from_config(stain: str, entries: list[dict] | None, subtype_wait_s: float = DEFAULT_SUBTYPE_WAIT_S) -> Vocabulary:
    """A stain's classes from its `labels.classes` entry: built-in classes keep their id and role (and come back, as they
    are by default, when the entry lacks them), others are taken as written."""
    builtin = {c.id: c for c in DEFAULTS.get(stain, GENERIC)}
    if not entries:
        return replace(defaults(stain), subtype_wait_s=subtype_wait_s)
    out: list[Identity] = []
    for entry in entries:
        ident = text(entry.get("id")) or text(entry.get("name"))
        if not ident or any(c.id == ident for c in out):
            continue
        base = builtin.get(ident)
        known = {s.id: s for s in base.subtypes} if base else {}
        subtypes: list[Subtype] = []
        for sub in entry.get("subtypes") or []:
            sid = text(sub.get("id")) or text(sub.get("name"))
            if sid and not any(s.id == sid for s in subtypes):
                subtypes.append(Subtype(sid, text(sub.get("name")) or sid, text(sub.get("description")), _clean_key(sub.get("key")),
                                        text(sub.get("example")), bool(sub.get("hidden")), builtin=sid in known))
        subtypes += [s for s in known.values() if not any(t.id == s.id for t in subtypes)]
        role = base.role if base else (text(entry.get("role")) if text(entry.get("role")) in ROLES else NEGATIVE)
        out.append(Identity(ident, text(entry.get("name")) or ident, role, text(entry.get("description")), _clean_key(entry.get("key")),
                            text(entry.get("example")), bool(entry.get("hidden")), builtin=base is not None, subtypes=tuple(subtypes)))
    out += [c for c in builtin.values() if not any(o.id == c.id for o in out)]
    return Vocabulary(stain, tuple(out), subtype_wait_s)


def of(project, stain: str) -> Vocabulary:
    """The project's classes of this stain (stainid.yaml `labels`), else the built-in ones."""
    settings = project.config.get("labels") or {}
    entries = (settings.get("classes") or {}).get(stain)
    wait = float(settings.get("subtype_wait_s", DEFAULT_SUBTYPE_WAIT_S))
    key = (stain, repr(entries), wait)
    if key not in _OF_CONFIG:
        _OF_CONFIG[key] = from_config(stain, entries, wait)
    return _OF_CONFIG[key]


_OF_CONFIG: dict[tuple, Vocabulary] = {}  # vocabularies by the settings they were made from (kept small: settings rarely change)


def problems(vocabulary: Vocabulary) -> list[str]:
    """What is wrong with a stain's classes, in plain words (empty when they can be saved)."""
    out = []
    names = [c.name.lower() for c in vocabulary.classes]
    if len(set(names)) != len(names):
        out.append("Two classes have the same name.")
    if not any(c.role == POSITIVE and not c.hidden for c in vocabulary.classes):
        out.append(f"At least one shown class must be the {vocabulary.stain} object itself (a positive class).")
    keys: dict[str, str] = {}
    for c in vocabulary.classes:
        sub_names = [s.name.lower() for s in c.subtypes]
        if len(set(sub_names)) != len(sub_names):
            out.append(f"Two types of {c.name} have the same name.")
        for owner, key in [(c.name, c.key), *((f"{c.name} · {s.name}", s.key) for s in c.subtypes)]:
            if not key:
                continue
            if key in RESERVED_KEYS:
                out.append(f"The key {key.upper()} of {owner} is used for something else (tools, undo or “no type”).")
            elif key in keys:
                out.append(f"{owner} and {keys[key]} have the same key, {key.upper()}.")
            keys.setdefault(key, owner)
        if any(s.key.isdigit() for s in c.subtypes):
            out.append(f"The types of {c.name} cannot have number keys: numbers pick a type by its place after the class key.")
    return list(dict.fromkeys(out))


def entry_of(classes: list[dict]) -> list[dict]:
    """A stain's classes as stainid.yaml keeps them (only what the settings may change)."""
    keep = ("id", "name", "role", "description", "key", "example", "hidden")
    sub_keep = ("id", "name", "description", "key", "example", "hidden")
    return [{**{k: c.get(k) for k in keep if k in c}, "subtypes": [{k: s.get(k) for k in sub_keep if k in s} for s in c.get("subtypes") or []]}
            for c in classes]
