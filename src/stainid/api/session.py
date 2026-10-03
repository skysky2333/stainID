"""Who is annotating, and the shared server mode.

Annotators sign in with a name (and the access code, when the server was started with one); the name travels in a signed
cookie, so every annotation is attributed by the server. `stainid serve --annotate` runs the shared mode, in which only
the annotation pages and their images are reachable.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from urllib.parse import quote, unquote

from fastapi import HTTPException, Request

from stainid.api import state
from stainid.api.state import shared


def cookie() -> str:
    """Separate cookies, so the owner can use the full app and the shared server side by side in one browser."""
    return "stainid_annotator_shared" if shared() else "stainid_annotator"


def access_code() -> str:
    return os.environ.get("STAINID_ACCESS_CODE", "")


def _secret() -> bytes:
    path = state.SETTINGS.parent / "secret"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32))
        path.chmod(0o600)
    return path.read_bytes() + access_code().encode()


def _signature(name: str) -> str:
    return hmac.new(_secret(), name.encode(), hashlib.sha256).hexdigest()


def sign(name: str) -> str:
    return f"{quote(name)}.{_signature(name)}"


def verify(token: str | None) -> str | None:
    if not token or "." not in token:
        return None
    encoded, signature = token.rsplit(".", 1)
    name = unquote(encoded)
    return name if hmac.compare_digest(signature, _signature(name)) else None


def annotator(request: Request) -> str:
    name = verify(request.cookies.get(cookie()))
    if name is None:
        raise HTTPException(401, "sign in with your name first")
    return name


def viewer(request: Request) -> str | None:
    """Anyone may look in the full app; in shared mode only signed-in annotators may."""
    return annotator(request) if shared() else verify(request.cookies.get(cookie()))


def admin_only() -> None:
    if shared():
        raise HTTPException(403, "not available on the shared annotation server")
