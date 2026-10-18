"""Who is using the app: one sign-in for everyone, by account (`stainid.api.accounts`).

Every person signs in on the same page with their name and password; their role decides what they see: an admin the
whole control panel (and every labelling set), an annotator only the annotation pages and the sets they joined with a
key. The tutorial needs no account. The account travels in a signed, HttpOnly cookie (`ACCOUNT_COOKIE`).

A web app served from another address (see `stainid serve --allow-origin`) cannot use this server's cookies, so signing
in also returns the same identity as a token, which it sends as `Authorization: Bearer <token>`, or as
`?access_token=<token>` on the images and downloads a browser fetches by itself. Scripts and agents use an account's API
token (`Bearer sid_…`, `stainid accounts token NAME`).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets

from fastapi import HTTPException, Request

from stainid.api import accounts, registry
from stainid.tables import write_text_atomic

ACCOUNT_COOKIE = "stainid_account"
DEFAULT_PASSWORD = "admin"  # the admin password until changed (earlier versions; the account `admin` keeps it)
LOCAL_HOSTS = ("127.0.0.1", "::1", "localhost")
GUEST = "guest"  # someone trying the tutorial without an account: only the practice sessions, where nothing is kept
PREVIEW = GUEST  # (the name earlier code used for it)


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
    """The admin password of earlier versions (read once, into the account `admin`)."""
    return registry.SETTINGS.parent / "admin.json"


def default_password() -> bool:
    return not _password_file().exists()


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()


def set_legacy_password(password: str) -> None:
    """Tests of the migration: the admin password as earlier versions kept it."""
    salt = secrets.token_hex(16)
    _password_file().parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(_password_file(), json.dumps({"salt": salt, "hash": _hash(password, salt)}))


def is_local(request: Request) -> bool:
    return request.client is not None and request.client.host in LOCAL_HOSTS


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header[:7].lower() == "bearer " else ""
    if not token and request.method in ("GET", "HEAD"):
        token = request.query_params.get("access_token", "")
    return token


def session_token(account: accounts.Account | None) -> str:
    """The signed-in account as a token, for a web app served elsewhere ("" when signed out)."""
    return f"account.{accounts.session_value(account)}" if account is not None else ""


def current_account(request: Request) -> accounts.Account | None:
    """The account of the request: its session cookie, else its bearer token (a sign-in token or an API token)."""
    state = request.scope.setdefault("state", {})
    if "stainid_account" in state:
        return state["stainid_account"]
    found = accounts.from_session(request.cookies.get(ACCOUNT_COOKIE, ""))
    if found is None:
        token = _bearer(request)
        if token.startswith(accounts.TOKEN_PREFIX):
            found = accounts.from_token(token)
        elif token.startswith("account."):
            found = accounts.from_session(token[len("account."):])
    state["stainid_account"] = found
    return found


def is_admin(request: Request) -> bool:
    found = current_account(request)
    return found is not None and found.admin and not found.must_change


def require_admin(request: Request) -> None:
    found = current_account(request)
    if found is None:
        raise HTTPException(401, "sign in first")
    if found.must_change:
        raise HTTPException(403, "choose a new password first")
    if not found.admin:
        raise HTTPException(403, "only admins can do this")


def require_account(request: Request) -> accounts.Account:
    found = current_account(request)
    if found is None:
        raise HTTPException(401, "sign in first")
    if found.must_change:
        raise HTTPException(403, "choose a new password first")
    return found


def _practice(request: Request) -> bool:
    return str(request.path_params.get("name", "")).startswith("practice-") or request.url.path.endswith("/session/tutorial")


def annotator_name(request: Request) -> str | None:
    """The name the signed-in person's labels carry (`Account.handle`), None when nobody is signed in."""
    found = current_account(request)
    return found.handle if found is not None and not found.must_change else None


def annotator(request: Request) -> str:
    """The person labelling: the signed-in account's name; on the tutorial's practice sessions anyone (`GUEST`)."""
    name = annotator_name(request)
    if name is None and _practice(request):
        return GUEST
    if name is None:
        raise HTTPException(401, "sign in first")
    return name
