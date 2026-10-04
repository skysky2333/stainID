"""Who is using the app: the admin (password) and annotators (a name).

The admin password guards the whole app; it is `admin` until changed and is stored hashed in ~/.stainid/admin.json.
While it is still the default, the admin may only sign in from this computer. Annotators register with a name on the
public /annotate pages; both identities travel in signed cookies, so the server attributes every annotation.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from urllib.parse import quote, unquote

from fastapi import HTTPException, Request

from stainid.api import state
from stainid.tables import write_text_atomic

ADMIN_COOKIE = "stainid_admin"
ANNOTATOR_COOKIE = "stainid_annotator"
DEFAULT_PASSWORD = "admin"
LOCAL_HOSTS = ("127.0.0.1", "::1", "localhost")


def _secret() -> bytes:
    path = state.SETTINGS.parent / "secret"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32))
        path.chmod(0o600)
    return path.read_bytes()


def _signature(message: str) -> str:
    return hmac.new(_secret(), message.encode(), hashlib.sha256).hexdigest()


def _password_file():
    return state.SETTINGS.parent / "admin.json"


def default_password() -> bool:
    return not _password_file().exists()


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()


def check_password(password: str) -> bool:
    if default_password():
        return hmac.compare_digest(password, DEFAULT_PASSWORD)
    stored = json.loads(_password_file().read_text())
    return hmac.compare_digest(_hash(password, stored["salt"]), stored["hash"])


def set_password(password: str) -> None:
    salt = secrets.token_hex(16)
    _password_file().parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(_password_file(), json.dumps({"salt": salt, "hash": _hash(password, salt)}))
    _password_file().chmod(0o600)


def admin_token() -> str:
    """Changes with the password, so changing it signs every admin out."""
    stored = "default" if default_password() else json.loads(_password_file().read_text())["hash"]
    return _signature(f"admin:{stored}")


def is_local(request: Request) -> bool:
    return request.client is not None and request.client.host in LOCAL_HOSTS


def is_admin(request: Request) -> bool:
    return hmac.compare_digest(request.cookies.get(ADMIN_COOKIE, ""), admin_token())


def require_admin(request: Request) -> None:
    if not is_admin(request):
        raise HTTPException(401, "admin sign-in required")


def sign_annotator(name: str) -> str:
    return f"{quote(name)}.{_signature(f'annotator:{name}')}"


def annotator_name(request: Request) -> str | None:
    token = request.cookies.get(ANNOTATOR_COOKIE, "")
    if "." not in token:
        return None
    encoded, signature = token.rsplit(".", 1)
    name = unquote(encoded)
    return name if hmac.compare_digest(signature, _signature(f"annotator:{name}")) else None


def annotator(request: Request) -> str:
    name = annotator_name(request)
    if name is None:
        raise HTTPException(401, "register with your name first")
    return name
