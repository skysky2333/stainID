"""The label classes of each stain: identity classes, each with optional subtypes.

The identity says what an object is (tau+ neuron, plaque, NeuN+ neuron, or one of the things that are not: a thread, a
vessel, an artifact). Its `role` is what the stain's identity model learns from it: `positive`, `negative` or `skip`
(not used: "unsure"). The subtype says which kind of positive (or other) object it is (pretangle, mature tangle; compact,
diffuse plaque); a later model can learn it from the objects people typed, while the identity models keep learning
exactly what they learnt before.

The keys (`assign_keys`, the one key assignment every page, the key lists and the tutorial read): each counted type
(a subtype of a positive identity, or the identity itself when it has no types shown) has a number, 1–4 for the
built-in ones and 5–9 for types people add; each named look-alike (a negative identity: CAA, a thread) a letter, Q W E R
built in and T Y U … for those people add; Del the stain's single reject ("not a plaque", drawn dotted grey like
detection's own rejects) and ` (backtick; 0 also works) "unsure", left out of training and evaluation. B is the brush
on the field pages. There is no two-step choice: a key picks the type at once.

A label stores the identity's `id` as its value and the subtype's `id` in `subtype` (both stable: renaming a class in the
settings changes its `name` only), so a type picked with one key is still "plaque" / "compact" for training, scoring
and the per-type measures; a label with an identity and no type stays valid ("plaque (type not recorded)"). Labels made before subtypes existed stored one string ("tau+ neuron: tangle",
"compact plaque", "plaque"), and labels of the class "artifact or unclear" (gone: debris is the reject, unclear is
"unsure", bad areas are painted with the brush); `LEGACY` maps each onto an identity and subtype, so the label log is
never rewritten.
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
REJECT_KEY, UNSURE_KEY, BRUSH_KEY = "delete", "`", "b"
KEY_ALIASES = {"0": UNSURE_KEY, "backspace": REJECT_KEY}  # quiet aliases: 0 where ` is awkward, Backspace (a Mac's delete key)
# keys the pages use for something else: field-page tools (v, a, o), save (s), undo (z), the brush (b), unsure (` and 0)
RESERVED_KEYS = {"v", "a", "o", "s", "m", "z", BRUSH_KEY, "0", UNSURE_KEY, REJECT_KEY}  # s split, m merge, z undo, b the brush
COUNTED_KEYS = "123456789"
EXTRA_COUNTED_FROM = "5"  # types people add take 5–9 (then any free number)
LOOKALIKE_KEYS = "qwer"  # the built-in look-alikes
EXTRA_LOOKALIKE_KEYS = "tyuiphjklcndfgx"  # look-alikes people add (then Q W E R when free)
DEFAULT_SUBTYPE_WAIT_S = 1.0  # no longer used (types are picked with one key); kept so older settings still load


@dataclass(frozen=True)
class Subtype:
    id: str
    name: str
    description: str = ""
    key: str = ""  # the key that picks this type (`assign_keys`; empty: one is given)
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


ARTIFACT = "artifact or unclear"  # a class no more: its labels read as the stain's reject (`LEGACY`)
UNSURE = ("unsure", "unsure", SKIP, "You cannot decide. The object is left out of training and evaluation.", UNSURE_KEY)

DEFAULTS: dict[str, tuple[Identity, ...]] = {
    "AT8": (
        _i("tau+ neuron", "tau+ neuron", POSITIVE, "A neuron whose cell body holds brown (AT8) tau.", "1", (
            _s("pretangle", "pretangle", "Diffuse or grainy brown in the cell body around a visible nucleus; the cell keeps its shape.", "1"),
            _s("mature tangle", "mature tangle", "A dense brown flame- or globe-shaped mass filling the cell body, often hiding the nucleus.", "2"),
            _s("ghost tangle", "ghost (extracellular) tangle", "A loose, paler brown tangle with no cell or nucleus around it: the neuron has died.", "3"),
        )),
        _i("neuritic plaque", "neuritic crown (tau around a plaque)", NEGATIVE,
           "A ring or cluster of swollen brown nerve endings around a plaque, with no cell body in the middle.", "q"),
        _i("astroglial tau", "tau in an astrocyte", NEGATIVE, "Brown tau in an astrocyte rather than a neuron: many fine processes around a small nucleus.", "w", (
            _s("tufted astrocyte", "tufted astrocyte", "Tufts of fine brown fibres radiating from the cell.", hidden=True),
            _s("thorn-shaped astrocyte", "thorn-shaped astrocyte", "Short thorn-like brown processes, often near vessels, under the brain surface or in white matter.",
               hidden=True),
        )),
        _i("thread", "thread or dystrophic neurite", NEGATIVE, "Short brown threads, dots or swollen fibres in the tissue, without a cell body.", "e"),
        _i("not tau", "not tau", NEGATIVE, "Not a tau+ neuron: a blue nucleus, a vessel, background, debris or an artifact.", REJECT_KEY),
        _i(*UNSURE),
    ),
    "6E10": (
        _i("plaque", "plaque", POSITIVE, "A brown amyloid deposit between the cells, at least about 10 µm across.", "1", (
            _s("compact", "compact (dense-core)", "A dense, well-defined brown core.", "1"),
            _s("diffuse", "diffuse", "A loose, fluffy brown deposit without a dense core.", "2"),
            _s("cored-neuritic", "cored-neuritic", "A dense core with a halo of swollen nerve endings around it, if you can see them.", "3"),
            _s("small", "small (too small to type)", "A real plaque only 10–15 µm across: too small to tell its type.", "4"),
        )),
        _i("CAA", "CAA / vascular amyloid", NEGATIVE, "Brown amyloid in the wall of a blood vessel (cerebral amyloid angiopathy).", "q"),
        _i("intraneuronal", "staining inside a neuron", NEGATIVE,
           "Brown inside a neuron’s cell body: APP or Aβ in the cell, which 6E10 also stains. Not a plaque.", "w"),
        _i("not a plaque", "not a plaque", NEGATIVE, "A cell, a speck under 10 µm, a fibre, background, debris or an artifact.", REJECT_KEY),
        _i(*UNSURE),
    ),
    "NeuN": (
        _i("NeuN+ neuron", "NeuN+ neuron", POSITIVE, "A brown (DAB) neuron: its nucleus and often its cell body are stained.", "1", (
            # how strongly a neuron is stained is measured, not judged: these types are kept for older labels only
            _s("strong", "strongly stained", "Dark brown nucleus and cell body.", hidden=True),
            _s("weak", "weakly stained", "Pale brown, but clearly a neuron.", hidden=True),
            _s("large pyramidal", "large pyramidal neuron", "A large triangular neuron with a thick dendrite towards the brain surface.", hidden=True),
        )),
        _i("not a neuron", "not a neuron", NEGATIVE, "A blue-only nucleus (glia, the lining of a vessel), background, debris or an artifact.", REJECT_KEY),
        _i(*UNSURE),
    ),
}
# Any other stain (e.g. of a dataset from another lab) starts from these.
GENERIC: tuple[Identity, ...] = (
    _i("object", "object", POSITIVE, "One of the objects this stain is analysed for.", "1"),
    _i("not an object", "not an object", NEGATIVE, "Something else: another cell, a vessel, background, debris.", REJECT_KEY),
    _i(*UNSURE),
)
REJECTS = {"AT8": "not tau", "6E10": "not a plaque", "NeuN": "not a neuron"}  # each stain's single reject (Del)
GENERIC_REJECT = "not an object"

# The keys of the built-in classes before one key picked a type (identity 1–7, then a type by its place or Q W T R):
# a project whose settings still carry one of these gets the built-in key of today (`from_config`).
OLD_KEYS: dict[str, dict[str, str]] = {
    "AT8": {"tau+ neuron": "1", "tau+ neuron/pretangle": "q", "tau+ neuron/mature tangle": "w", "tau+ neuron/ghost tangle": "t",
            "neuritic plaque": "2", "astroglial tau": "3", "thread": "4", "not tau": "5", "unsure": "7"},
    "6E10": {"plaque": "1", "plaque/compact": "q", "plaque/diffuse": "w", "plaque/cored-neuritic": "t", "plaque/small": "r", "CAA": "2",
             "intraneuronal": "3", "not a plaque": "4", "unsure": "6"},
    "NeuN": {"NeuN+ neuron": "1", "NeuN+ neuron/strong": "q", "NeuN+ neuron/weak": "w", "NeuN+ neuron/large pyramidal": "t", "not a neuron": "2",
             "unsure": "4"},
}

# Values of labels made before subtypes existed (and the AI's calls in the candidate pools): identity and subtype.
LEGACY: dict[str, dict[str, tuple[str, str]]] = {
    "AT8": {"tau+ neuron: tangle": ("tau+ neuron", "mature tangle"), "tau+ neuron: pretangle": ("tau+ neuron", "pretangle"),
            "not a tau+ neuron": ("not tau", ""), "tau+ neuron (type not recorded)": ("tau+ neuron", ""), ARTIFACT: ("not tau", "")},
    "6E10": {"compact plaque": ("plaque", "compact"), "diffuse plaque": ("plaque", "diffuse"), "small plaque": ("plaque", "small"),
             "plaque (type not recorded)": ("plaque", ""), "vascular": ("CAA", ""), ARTIFACT: ("not a plaque", "")},
    "NeuN": {ARTIFACT: ("not a neuron", "")},
}
GENERIC_LEGACY = {ARTIFACT: (GENERIC_REJECT, "")}
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

    @property
    def reject(self) -> str:
        """The stain's single reject ("not a plaque": Del, drawn dotted grey like detection's own rejects): the shown class
        with the Del key, else the built-in reject, else the first shown negative class."""
        found = next((c for c in self.visible if c.key == REJECT_KEY), None) or self.identity(REJECTS.get(self.stain, GENERIC_REJECT))
        found = found or next((c for c in self.visible if c.role == NEGATIVE), None)
        return found.id if found else ""

    @property
    def unsure(self) -> str:
        """The class meaning "unsure" (` or 0): the first shown class with the role skip."""
        return next((c.id for c in self.visible if c.role == SKIP), "")

    def keymap(self, subtypes: bool = True) -> dict:
        """The keys of the labelling pages (`assign_keys`). `direct`: key -> [identity, type or None]; `groups`: the
        panel's groups in order (counted types, named look-alikes, the reject, unsure), each with its title, keys text
        ("1–4") and items; `summary`: the panel's one line ("Plaques 1–4 · Not plaques Q–W · Del not a plaque · ` unsure
        · B brush"); `aliases`: quiet second keys (0 for `, Backspace for Del); `brush`: the brush key. Without
        `subtypes` (the session does not ask for types) a positive identity has one key and no types."""
        choices = assign_keys(self, subtypes)
        direct = {c.key: [c.identity, c.subtype] for c in choices}
        groups = []
        positives = [c for c in self.visible if c.role == POSITIVE]
        plural = " / ".join(_plural(c.name) for c in positives) or "objects"
        titles = {"counted": plural[:1].upper() + plural[1:], "lookalike": f"Not {plural}"}
        for group in GROUPS:
            items = [c for c in choices if c.group == group]
            if not items:
                continue
            title = titles.get(group) or (self.identity(items[0].identity).name if self.identity(items[0].identity) else items[0].identity)
            groups.append({"group": group, "title": title, "keys": _keys_text([c.key for c in items]),
                           "items": [{"key": c.key, "label": key_label(c.key), "identity": c.identity, "subtype": c.subtype, "name": c.name,
                                      "description": c.description} for c in items]})
        summary = " · ".join(f"{g['title']} {g['keys']}" if g["group"] in ("counted", "lookalike") else f"{g['keys']} {g['title']}" for g in groups)
        return {"direct": direct, "groups": groups, "summary": f"{summary} · B brush", "aliases": dict(KEY_ALIASES), "brush": BRUSH_KEY}

    def by_key(self) -> list[Identity]:
        """The classes in the order every page lists them: the counted ones, the named look-alikes (by their keys), the
        reject, "unsure" last; classes without a key (hidden ones) before "unsure" (webapp classKeys.byKey)."""
        first: dict[str, int] = {}
        for n, c in enumerate(assign_keys(self, True)):
            first.setdefault(c.identity, (GROUPS.index(c.group), n))
        place = {c.id: i for i, c in enumerate(self.classes)}

        def order(c: Identity) -> tuple:
            if c.id in first:
                return (c.role == SKIP, 0, first[c.id], place[c.id])
            return (c.role == SKIP, 1, (0, 0), place[c.id])

        return sorted(self.classes, key=order)

    def colour_slots(self) -> dict[str, int]:
        """Each class's colour slot (webapp classPalette.ts), by "identity" and "identity/type": positive identities and
        each of their types first, then the other classes (their types share their colour), hidden ones too so a colour
        never moves; the reject and "unsure" have none (dotted grey, dotted amber)."""
        out: dict[str, int] = {}
        special = {self.reject, self.unsure}
        for c in sorted(self.classes, key=lambda c: c.role != POSITIVE):
            if c.id in special:
                continue
            out[c.id] = len(out)
            if c.role == POSITIVE:
                for s in c.subtypes:
                    out[f"{c.id}/{s.id}"] = len(out)
        return out

    def to_api(self) -> dict:
        """The classes as the web app uses them, with how old values map onto them."""
        return {"stain": self.stain, "subtype_wait_s": self.subtype_wait_s,
                "classes": [{**c.to_config(), "builtin": c.builtin, "subtypes": [{**s.to_config(), "builtin": s.builtin} for s in c.subtypes]}
                            for c in self.classes],
                "aliases": {k: list(v) for k, v in _legacy(self.stain).items()}, "untyped": UNTYPED, "reject": self.reject,
                "unsure": self.unsure, "slots": self.colour_slots()}

    def to_config(self) -> list[dict]:
        return [c.to_config() for c in self.classes]


def _resolve(vocabulary: Vocabulary, value: str, subtype: str) -> tuple[str, str]:
    if not value:
        return "", ""
    found = vocabulary.identity(value)
    if found is None and value in _legacy(vocabulary.stain):
        identity, legacy_sub = _legacy(vocabulary.stain)[value]
        if identity == REJECTS.get(vocabulary.stain, GENERIC_REJECT) and vocabulary.identity(identity) is None:
            identity = vocabulary.reject or identity
        return identity, subtype or legacy_sub
    if found is None:
        parts = _SPLIT.split(value, maxsplit=1)
        head = vocabulary.identity(parts[0]) if len(parts) == 2 else None
        if head is None:
            return value, subtype
        found, subtype = head, subtype or parts[1]
    sub = found.subtype(subtype) if subtype else None
    return found.id, sub.id if sub else subtype


def _legacy(stain: str) -> dict[str, tuple[str, str]]:
    return LEGACY.get(stain, GENERIC_LEGACY)


GROUPS = ("counted", "lookalike", "reject", "unsure")


@dataclass(frozen=True)
class KeyChoice:
    """One key of the labelling pages: what it answers (identity and type) and its group in the panel."""
    key: str
    identity: str
    subtype: str | None
    group: str
    name: str
    description: str = ""


def key_label(key: str) -> str:
    """A key as the pages print it: "1", "Q", "Del", "`"."""
    return "Del" if key == REJECT_KEY else key.upper()


def _keys_text(keys: list[str]) -> str:
    """"1–4", "Q–W", "Q", "Del"."""
    labels = [key_label(k) for k in keys]
    return labels[0] if len(labels) == 1 else f"{labels[0]}–{labels[-1]}"


def _plural(name: str) -> str:
    return name if name.endswith("s") else name + "s"


def assign_keys(vocabulary: Vocabulary, subtypes: bool = True) -> list[KeyChoice]:
    """The one key assignment of a stain's classes (the panel, the key lists and the tutorial all read it, through
    `Vocabulary.keymap`), in the panel's order:
    - counted types: each shown type of a shown positive identity (with `subtypes`; the identity itself when it has no
      shown types or types are not asked), by its own key when that is a free number, else the next free from 5;
    - named look-alikes: each other shown class (a shown type of it, when types are asked), by its own key when that is
      a free letter, else the next free of T Y U …;
    - the reject: Del; "unsure": ` (and 0, `KEY_ALIASES`).
    Hidden classes and types have no key."""
    reject, unsure = vocabulary.reject, vocabulary.unsure
    counted: list[tuple[Identity, Subtype | None]] = []
    lookalike: list[tuple[Identity, Subtype | None]] = []
    for c in vocabulary.visible:
        if c.id in (reject, unsure):
            continue
        shown = [s for s in c.subtypes if not s.hidden] if subtypes else []
        items = [(c, s) for s in shown] or [(c, None)]
        (counted if c.role == POSITIVE else lookalike).extend(items)
    used: set[str] = set()

    def place(items, allowed: str, extra: str) -> list[tuple[str, Identity, Subtype | None]]:
        out: list = []
        wanted = [(c, s, (s.key if s is not None else c.key)) for c, s in items]
        for c, s, key in wanted:  # their own keys first, so a class people add never takes a built-in one's
            if key and key in allowed and key not in used and key not in RESERVED_KEYS:
                used.add(key)
                out.append((key, c, s))
            else:
                out.append((None, c, s))
        free = [k for k in extra + allowed if k not in used and k not in RESERVED_KEYS]
        for n, (key, c, s) in enumerate(out):
            if key is None and free:
                key = free.pop(0)
                used.add(key)
                out[n] = (key, c, s)
        return [o for o in out if o[0]]

    counted_free = COUNTED_KEYS[COUNTED_KEYS.index(EXTRA_COUNTED_FROM):]
    letters = LOOKALIKE_KEYS + EXTRA_LOOKALIKE_KEYS
    choices = [KeyChoice(k, c.id, s.id if s else None, "counted", s.name if s else c.name, (s or c).description)
               for k, c, s in place(counted, COUNTED_KEYS, counted_free)]
    choices += [KeyChoice(k, c.id, s.id if s else None, "lookalike", s.name if s else c.name, (s or c).description)
                for k, c, s in place(lookalike, letters, EXTRA_LOOKALIKE_KEYS)]
    if reject:
        choices.append(KeyChoice(REJECT_KEY, reject, None, "reject", vocabulary.identity(reject).name, vocabulary.identity(reject).description))
    if unsure:
        choices.append(KeyChoice(UNSURE_KEY, unsure, None, "unsure", vocabulary.identity(unsure).name, vocabulary.identity(unsure).description))
    choices.sort(key=lambda c: (GROUPS.index(c.group), COUNTED_KEYS.find(c.key) if c.group == "counted" else letters.find(c.key) if c.group == "lookalike" else 0))
    return choices


_BUILT_IN: dict[str, Vocabulary] = {}


def defaults(stain: str) -> Vocabulary:
    """The built-in classes of a stain (those of `GENERIC` for a stain stainID has none for)."""
    if stain not in _BUILT_IN:
        _BUILT_IN[stain] = Vocabulary(stain, DEFAULTS.get(stain, GENERIC))
    return _BUILT_IN[stain]


def _clean_key(value) -> str:
    found = text(value).lower()
    if found in ("del", "delete", "backspace"):
        return REJECT_KEY
    return found[:1]


def _current_key(stain: str, ref: str, key: str, base_key: str | None) -> str:
    """A built-in class's key from the settings: today's built-in key when the settings carry none or its old one."""
    if base_key is not None and (not key or OLD_KEYS.get(stain, {}).get(ref) == key):
        return base_key
    return key


def from_config(stain: str, entries: list[dict] | None, subtype_wait_s: float = DEFAULT_SUBTYPE_WAIT_S) -> Vocabulary:
    """A stain's classes from its `labels.classes` entry: built-in classes keep their id and role (and come back, as they
    are by default, when the entry lacks them), others are taken as written."""
    builtin = {c.id: c for c in DEFAULTS.get(stain, GENERIC)}
    entries = [e for e in entries or [] if (text(e.get("id")) or text(e.get("name"))) != ARTIFACT]  # a class no more (`LEGACY`)
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
                key = _current_key(stain, f"{ident}/{sid}", _clean_key(sub.get("key")), known[sid].key if sid in known else None)
                hidden = bool(sub.get("hidden")) or (ident == "NeuN+ neuron" and sid in ("strong", "weak"))  # measured, not judged
                subtypes.append(Subtype(sid, text(sub.get("name")) or sid, text(sub.get("description")), key,
                                        text(sub.get("example")), hidden, builtin=sid in known))
        subtypes += [s for s in known.values() if not any(t.id == s.id for t in subtypes)]
        role = base.role if base else (text(entry.get("role")) if text(entry.get("role")) in ROLES else NEGATIVE)
        key = _current_key(stain, ident, _clean_key(entry.get("key")), base.key if base else None)
        out.append(Identity(ident, text(entry.get("name")) or ident, role, text(entry.get("description")), key,
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
    reject, unsure = vocabulary.reject, vocabulary.unsure
    for c in vocabulary.classes:
        sub_names = [s.name.lower() for s in c.subtypes]
        if len(set(sub_names)) != len(sub_names):
            out.append(f"Two types of {c.name} have the same name.")
        counted = c.role == POSITIVE
        typed = any(not s.hidden for s in c.subtypes)
        owners = [] if counted and typed else [(c.name, c.key)]  # a typed counted class's own key is only used when types are not asked
        owners += [(f"{c.name} · {s.name}", s.key) for s in c.subtypes]
        for owner, key in owners:
            if not key or c.id in (reject, unsure):
                continue
            if key in RESERVED_KEYS:
                out.append(f"The key {key_label(key)} of {owner} is used for something else (Del, `, B brush, S save, Z undo or the field tools).")
            elif counted and key not in COUNTED_KEYS:
                out.append(f"{owner} is counted, so its key must be a number (1–9), not {key_label(key)}.")
            elif not counted and not key.isalpha():
                out.append(f"{owner} is not counted, so its key must be a letter (Q W E R T …), not {key_label(key)}.")
            elif key in keys:
                out.append(f"{owner} and {keys[key]} have the same key, {key_label(key)}.")
            keys.setdefault(key, owner)
    return list(dict.fromkeys(out))


def entry_of(classes: list[dict]) -> list[dict]:
    """A stain's classes as stainid.yaml keeps them (only what the settings may change)."""
    keep = ("id", "name", "role", "description", "key", "example", "hidden")
    sub_keep = ("id", "name", "description", "key", "example", "hidden")
    return [{**{k: c.get(k) for k in keep if k in c}, "subtypes": [{k: s.get(k) for k in sub_keep if k in s} for s in c.get("subtypes") or []]}
            for c in classes]
