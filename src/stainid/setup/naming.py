"""Shared rules for reading what people type: donor ids, brain regions, diagnostic groups and neuropathology scores.

The TMA map and the donor spreadsheet are read with the same rules, so "JOG3" on a map and "3" in the JOG part of a
case sheet both become JOG_003, and "698" and "0698" are the same donor.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−﹘﹣－"), "-")
INVISIBLE = dict.fromkeys(map(ord, "​‌‍﻿"), None)


def clean(text: str) -> str:
    """Dashes as "-", no-break and repeated spaces as one space, each line trimmed, empty lines dropped."""
    text = str(text or "").translate(DASHES).translate(INVISIBLE).replace(" ", " ").replace("\r", "\n")
    return "\n".join(line for line in (re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")) if line)


def key(text: str) -> str:
    """Letters and digits only, upper case: how codes are compared ("Asym AD" == "ASYMAD", "OC C" == "OCC")."""
    return re.sub(r"[^A-Z0-9]", "", clean(text).upper())


# ---------------------------------------------------------------- donor ids

@dataclass(frozen=True)
class IdStyle:
    """How donor ids are written. Plain numbers get at least `digits` digits (0 keeps them as typed); numbers after a
    letter prefix are written <PREFIX>_<number with `prefixed_digits` digits>: "JOG3" -> JOG_003."""
    digits: int = 4
    prefixed_digits: int = 3

    @classmethod
    def from_config(cls, config: dict) -> IdStyle:
        ids = config.get("donor_ids") or {}
        return cls(int(ids.get("digits", 4)), int(ids.get("prefixed_digits", 3)))


ID_PATTERN = re.compile(r"([A-Z]*)(\d+)")


def donor_id(text: str, style: IdStyle = IdStyle(), prefix: str = "") -> str:
    """The donor id in the project's style: stray spaces, "#", "_" and "-" are ignored ("17 00" -> 1700, "JOG 3" ->
    JOG_003). `prefix` applies to plain numbers (the id series of a sub-table such as "JOG#"). Other ids are kept
    as typed."""
    compact = key(text)
    match = ID_PATTERN.fullmatch(compact)
    if not match:
        return clean(text)
    letters, number = match.group(1) or prefix.upper(), match.group(2)
    if letters:
        return f"{letters}_{int(number):0{style.prefixed_digits}d}"
    return f"{int(number):0{style.digits}d}" if style.digits else number


def id_prefix(donor: str) -> str:
    match = re.fullmatch(r"([A-Z]+)_\d+", donor)
    return match.group(1) if match else ""


# ---------------------------------------------------------------- brain regions and control tissue

REGIONS = {
    "frontal": ("FR", "FRO", "FRONT", "FRONTAL", "F", "FCX", "MFG", "SFG", "BA9", "BA46"),
    "occipital": ("OCC", "OC", "OCCIP", "OCCIPITAL", "OCX", "V1", "BA17"),
    "temporal": ("TEMP", "TEM", "TE", "TEMPORAL", "TCX", "MTG", "STG", "ITG"),
    "parietal": ("PAR", "PARIET", "PARIETAL", "PCX", "IPL", "ANG"),
    "hippocampus": ("HIP", "HIPP", "HIPPO", "HIPPOCAMPUS", "HPC"),
    "entorhinal": ("ENT", "EC", "ERC", "ENTORHINAL"),
    "cingulate": ("CING", "CG", "ACC", "PCC", "CINGULATE"),
    "insula": ("INS", "INSULA"),
    "amygdala": ("AMY", "AMYG", "AMYGDALA"),
    "striatum": ("STR", "STRIATUM"),
    "putamen": ("PUT", "PUTAMEN"),
    "caudate": ("CAUD", "CAUDATE"),
    "thalamus": ("THAL", "THALAMUS"),
    "midbrain": ("MB", "MIDBRAIN"),
}
REGION_OF = {code: name for name, codes in REGIONS.items() for code in codes}
CONTROL_TISSUES = ("CBLM", "CEREBELLUM", "CEREB", "STOM", "STOMACH", "LIVER", "SPLEEN", "KIDNEY", "TONSIL", "PLACENTA", "LUNG",
                   "HEART", "SKIN", "COLON", "MUSCLE", "ORIENT", "ORIENTATION", "MARKER")
EMPTY_MARKS = {"", "X", "EMPTY", "BLANK", "NONE", "NA", "NOCORE", "MISSING"}


def region_name(code: str) -> str:
    """frontal for FR, occipital for OCC, …; an unknown code is kept (in lower case)."""
    return REGION_OF.get(key(code), clean(code).lower())


def control_tissue(text: str) -> str:
    compact = key(text)
    return next((word for word in CONTROL_TISSUES if word in compact), "")


# ---------------------------------------------------------------- diagnostic groups

FAMILIES = {  # code used when the project has none of the family: the words that mean it
    "CT": ("CT", "CTRL", "CON", "CONT", "CONTROL", "CONTROLS", "NC", "NORMAL", "NORMALCONTROL", "HEALTHY", "NONDEMENTED"),
    "ASYMP": ("ASYMP", "ASYM", "ASYMAD", "ASYMPAD", "ASYMPTOMATIC", "ASYMPTOMATICAD", "PRECLINICAL", "PRECLINICALAD"),
    "AD": ("AD", "ALZ", "ALZHEIMER", "ALZHEIMERS", "ALZHEIMERDISEASE", "ALZHEIMERSDISEASE", "ADD", "ADRD"),
}


WORD = r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z']*(?![A-Za-z0-9])"  # a word not glued to digits ("AD", not the AD of "AD12")


class Groups:
    """Turns what a map cell or a spreadsheet says about a donor's group into one of the project's group codes.

    The project's own codes and labels always match. Common words for control, asymptomatic AD and AD match too
    ("Control", "CTRL", "ASYM AD", "Alzheimer"), mapped to the project's code of that kind when it has one (a
    project with code ASYM gets ASYM for "ASYMAD"), otherwise to CT / ASYMP / AD."""

    def __init__(self, configured: list[dict] | None = None):
        configured = list(configured or [])
        self.codes = [str(g["code"]) for g in configured]
        self.aliases: dict[str, str] = {}
        for family, words in FAMILIES.items():
            mine = next((code for code in self.codes if key(code) in words or key(code) == family), family)
            self.aliases.update(dict.fromkeys(words, mine))
        for group in configured:
            self.aliases[key(group["code"])] = str(group["code"])
            if group.get("label"):
                self.aliases.setdefault(key(group["label"]), str(group["code"]))

    def code(self, text: str) -> str:
        """The group code for a cell that says only the group ("" when it is not one)."""
        return self.aliases.get(key(text), "")

    def find(self, text: str) -> tuple[str, tuple[int, int] | None]:
        """The group a longer text names, and where (map cells such as "1673 FR AD"): whole words only, the most specific
        match first (the longest: "ASYMAD" beats "AD"), and of equally long ones the last."""
        found: tuple[str, tuple[int, int] | None] = ("", None)
        words = list(re.finditer(WORD, text))
        i = 0
        while i < len(words):
            for width in (2, 1):  # two words first ("ASYM AD"), then one
                if i + width > len(words):
                    continue
                span = (words[i].start(), words[i + width - 1].end())
                code = self.aliases.get(key(text[span[0]:span[1]]))
                if code:
                    if not found[1] or len(key(text[span[0]:span[1]])) >= len(key(text[found[1][0]:found[1][1]])):
                        found = (code, span)
                    i += width - 1
                    break
            i += 1
        return found

    def section(self, text: str) -> str:
        """The group a section heading stands for: "CONTROL <65 (90 cases)" -> control, "High AD (218 cases) - Braak
        5/6" -> AD, "ASYMAD A+T (15 cases) - asymptomatic" -> asymptomatic AD. Only the heading's name counts (the
        part before "(", " - " or ":"), so a note such as "JOG 74 removed (…) - ASYMAD A only" is no heading ("")."""
        name = re.split(r"\(|\s[-\u2013]\s|:", clean(text), maxsplit=1)[0]
        return self.code(name) or self.find(name)[0]


# ---------------------------------------------------------------- neuropathology scores

CERAD = {"0": "0", "NONE": "0", "NO": "0", "A": "A", "SPARSE": "A", "B": "B", "MODERATE": "B", "MOD": "B", "C": "C",
         "FREQUENT": "C", "FREQ": "C"}
BRAAK = {"0": "0", "I": "1", "II": "2", "III": "3", "IV": "4", "V": "5", "VI": "6", **{str(n): str(n) for n in range(7)}}
PATHOLOGY = re.compile(r"(?:CERAD|(?<![A-Z])C)\s*[:=-]?\s*(0|A|B|C|NONE|SPARSE|MODERATE|FREQUENT)(?![A-Z])\s*[-,;/]?\s*"
                       r"(?:BRAAK|(?<![A-Z])B)\s*[:=-]?\s*(VI|V|IV|III|II|I|[0-6])(?![A-Z0-9])", re.I)


def cerad(value: str) -> str:
    """CERAD neuritic plaque score as 0 / A / B / C ("" when not a score)."""
    return CERAD.get(key(value), "")


def braak(value: str) -> str:
    """Braak stage as 0-6 (roman numerals read too; "" when not a stage)."""
    compact = key(value).removeprefix("BRAAK").removeprefix("STAGE")
    return BRAAK.get(compact, "")


def find_pathology(text: str) -> tuple[str, str, tuple[int, int] | None]:
    """CERAD and Braak written as "C:C - B:6" (or "CERAD B, Braak IV") in a text, and where."""
    match = PATHOLOGY.search(text)
    if not match:
        return "", "", None
    return cerad(match.group(1)), braak(match.group(2)), match.span()


def tma_number(text: str) -> str:
    """The TMA number in "LIP-8", "TMA 8" or "8" ("" when there is none)."""
    numbers = re.findall(r"\d+", str(text))
    return str(int(numbers[-1])) if numbers else ""
