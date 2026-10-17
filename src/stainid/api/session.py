"""Who is using the app: the admin (password) and annotators (a name).

The admin password guards the whole app; it is `admin` until changed and is stored hashed in ~/.stainid/admin.json.
Annotators register with a name on the
public /annotate pages; both identities travel in signed cookies, so the server attributes every annotation.

A web app served from another address (see `stainid serve --allow-origin`) cannot use this server's cookies, so signing
in also returns the same identities as a token, which it sends as `Authorization: Bearer <token>`, or as
`?access_token=<token>` on the images and downloads a browser fetches by itself.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from urllib.parse import quote, unquote

from fastapi import HTTPException, Request

from stainid.api import registry
from stainid.tables import write_text_atomic

ADMIN_COOKIE = "stainid_admin"
ANNOTATOR_COOKIE = "stainid_annotator"
DEFAULT_PASSWORD = "admin"
LOCAL_HOSTS = ("127.0.0.1", "::1", "localhost")


def allowed_origins(extra: list[str] = ()) -> list[str]:
    """Addresses of web apps served elsewhere that may use this server: `--allow-origin` and $STAINID_ALLOW_ORIGINS."""
    origins = [o.strip().rstrip("/") for o in [*os.environ.get("STAINID_ALLOW_ORIGINS", "").split(","), *extra]]
    return list(dict.fromkeys(o for o in origins if o))


def _secret() -> bytes:
    path = registry.SETTINGS.parent / "secret"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32))
        path.chmod(0o600)
    return path.read_bytes()


def _signature(message: str) -> str:
    return hmac.new(_secret(), message.encode(), hashlib.sha256).hexdigest()


def _password_file():
    return registry.SETTINGS.parent / "admin.json"


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


def _token_parts(request: Request) -> dict[str, str]:
    """The identities in the request's token: {"admin": ..., "annotator": ...} (see `session_token`)."""
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header[:7].lower() == "bearer " else ""
    if not token and request.method in ("GET", "HEAD"):
        token = request.query_params.get("access_token", "")
    return dict(part.split(".", 1) for part in token.split(",") if "." in part)


def session_token(admin: bool, annotator: str | None) -> str:
    """Who the caller is, for a web app served elsewhere: the admin and annotator credentials joined by a comma."""
    parts = [f"admin.{admin_token()}"] if admin else []
    return ",".join(parts + ([f"annotator.{sign_annotator(annotator)}"] if annotator else []))


def is_admin(request: Request) -> bool:
    token = admin_token()
    return hmac.compare_digest(request.cookies.get(ADMIN_COOKIE, ""), token) or hmac.compare_digest(_token_parts(request).get("admin", ""), token)


def require_admin(request: Request) -> None:
    if not is_admin(request):
        raise HTTPException(401, "admin sign-in required")


def sign_annotator(name: str) -> str:
    return f"{quote(name)}.{_signature(f'annotator:{name}')}"


def annotator_name(request: Request) -> str | None:
    return _annotator(request.cookies.get(ANNOTATOR_COOKIE, "")) or _annotator(_token_parts(request).get("annotator", ""))


def _annotator(token: str) -> str | None:
    if "." not in token:
        return None
    encoded, signature = token.rsplit(".", 1)
    name = unquote(encoded)
    return name if hmac.compare_digest(signature, _signature(f"annotator:{name}")) else None


PREVIEW = "admin (preview)"  # the admin trying the tutorial without a name: only on the practice sessions, where nothing is kept


def annotator(request: Request) -> str:
    name = annotator_name(request)
    if name is None and is_admin(request) and (str(request.path_params.get("name", "")).startswith("practice-")
                                               or request.url.path.endswith("/session/tutorial") and request.method == "GET"):
        return PREVIEW
    if name is None:
        raise HTTPException(401, "register with your name first")
    return name
