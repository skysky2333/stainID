"""The tutorial's practice changes nothing. A request on a practice session (`/api/annotate/practice-…`) runs with
`PRACTISING` set (the app's middleware), and every store's write calls `guard` first: the label log (labels, brush
areas, bad areas), the candidate and added-object stores, the sessions, and the holds and "who is on it" claims. The
practice routes answer as if saved without writing; this is the safety net under them, so a route that forgets cannot
keep anything (it answers 403 instead)."""
from __future__ import annotations

import contextlib
import re
from contextvars import ContextVar

PRACTISING: ContextVar[bool] = ContextVar("stainid_practising", default=False)
PATH = re.compile(r"^/api/annotate/practice-[A-Za-z0-9_\-]+(/|$)")
NOT_KEPT = "This is the tutorial's practice: nothing here is kept."


class NotKept(PermissionError):
    """A store write asked for while practising."""


def guard(what: str) -> None:
    """Refuse a write while practising (`what`: the store, for the log)."""
    if PRACTISING.get():
        raise NotKept(f"{NOT_KEPT} ({what})")


def practising() -> bool:
    return PRACTISING.get()


@contextlib.contextmanager
def practice():
    """Run the block as a practice request (the middleware's doing; also for tests)."""
    token = PRACTISING.set(True)
    try:
        yield
    finally:
        PRACTISING.reset(token)
