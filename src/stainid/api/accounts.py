"""The people who use this server: one account system for the control panel and the annotation pages.

Every account has a name (unique on the server, case does not matter), a password (scrypt with its own salt; never kept
as text) and a role: `admin` (the control panel, every session) or `annotator` (only the annotation pages, and only the
labelling sets they joined). Everyone signs in on the same page; what they see follows from their role. Signing up makes
an annotator; an admin promotes, demotes, resets passwords and removes people (Settings → People), and there is always
at least one admin. The admin password of earlier versions becomes the account `admin` (same password).

Labelling sets are shared with a **join key** per session (`stainid-ab3k-9xqe` style: 10 characters of an alphabet
without look-alikes, about 50 bits): whoever enters it adds that session (of that project) to their sets. A new key
stops the old one from working; people who joined keep their access. Membership is checked on every annotation request
(`require_member`), not only in the page.

Names already in a project's label log are reserved (`reserved_names`): nobody can sign up as "Ann" and inherit Ann's
labels. An admin links an account to such a past name; the account then labels under that name and sees its history.
Labels carry the account id besides the name (`store.append` adds `account` from `ATTRIBUTION`).

Scripts and agents use API tokens of an account (`stainid accounts token NAME`, or Settings): `Authorization: Bearer
sid_…`. Tokens are kept hashed, and removing an account ends its tokens.

Everything lives in `~/.stainid/accounts.sqlite` (next to the server's settings), shared by every process of the server.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

ROLES = ("admin", "annotator")
ADMIN, ANNOTATOR = ROLES
KEY_ALPHABET = "23456789abcdefghjkmnpqrstuvwxyz"  # no 0/o, 1/l/i: read aloud or typed from paper without mistakes
KEY_LENGTH = 10  # 31^10: about 50 bits; with `JOIN_TRIES` per hour nobody guesses one
TOKEN_PREFIX = "sid_"
MIN_PASSWORD = 8
SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}
LOGIN_TRIES = 8  # failed sign-ins per name (and 3x that per address) within `TRY_WINDOW`
JOIN_TRIES = 10  # wrong keys per account (and per address) within `TRY_WINDOW`
TRY_WINDOW = 15 * 60
NAME_MAX = 40
_LOCK = threading.RLock()
_TRIES: dict[tuple, deque] = defaultdict(deque)


class AccountError(ValueError):
    """Something a person asked for that cannot be done; `status` is the HTTP status the server answers with."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class Account:
    id: str
    name: str
    role: str
    must_change: bool
    past_name: str
    generation: int
    created: float

    @property
    def admin(self) -> bool:
        return self.role == ADMIN

    @property
    def handle(self) -> str:
        """The name the account's labels carry: the past name it was linked to, else its own."""
        return self.past_name or self.name

    def to_api(self) -> dict:
        return {"id": self.id, "name": self.name, "role": self.role, "must_change": self.must_change, "past_name": self.past_name,
                "handle": self.handle, "created": self.created}


# ---------------------------------------------------------------- the store

def db_path() -> Path:
    from stainid.api import registry  # noqa: PLC0415

    return registry.SETTINGS.parent / "accounts.sqlite"


_SCHEMA = """
create table if not exists accounts (id text primary key, name text not null, name_key text not null unique, role text not null,
    kdf text not null, salt text not null, hash text not null, must_change integer not null default 0, past_name text not null default '',
    generation integer not null default 1, created real not null);
create table if not exists tokens (id text primary key, account text not null, hash text not null unique, label text not null,
    created real not null, last_used real);
create table if not exists members (account text not null, project text not null, session text not null, joined real not null,
    primary key (account, project, session));
create table if not exists join_keys (project text not null, session text not null, key text not null unique, enabled integer not null,
    created real not null, primary key (project, session));
create table if not exists tutorials (account text not null, kind text not null, done real not null, primary key (account, kind));
"""
_READY: set[str] = set()


def _db() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=15, isolation_level=None)
    connection.row_factory = sqlite3.Row
    if str(path) not in _READY:
        with _LOCK:
            connection.executescript(_SCHEMA)
            path.chmod(0o600)
            _migrate_admin(connection)
            _READY.add(str(path))
    return connection


def _migrate_admin(db: sqlite3.Connection) -> None:
    """The first admin: the admin password of earlier versions (`~/.stainid/admin.json`, else `admin`) as the account
    `admin`, so nobody is locked out."""
    from stainid.api import session  # noqa: PLC0415

    if db.execute("select 1 from accounts where role = ?", (ADMIN,)).fetchone():
        return
    if session.default_password():
        kdf, salt, digest = _hash_password(session.DEFAULT_PASSWORD)
    else:
        stored = json.loads(session._password_file().read_text())
        kdf, salt, digest = "pbkdf2", stored["salt"], stored["hash"]  # checked as before, made scrypt at the next sign-in
    name = "admin" if not db.execute("select 1 from accounts where name_key = 'admin'").fetchone() else f"admin-{secrets.token_hex(2)}"
    db.execute("insert into accounts (id, name, name_key, role, kdf, salt, hash, created) values (?, ?, ?, ?, ?, ?, ?, ?)",
               (_new_id(), name, _key_of(name), ADMIN, kdf, salt, digest, time.time()))


def _new_id() -> str:
    return "u" + secrets.token_hex(6)


def _key_of(name: str) -> str:
    return " ".join(name.split()).casefold()


def _account(row) -> Account | None:
    if row is None:
        return None
    return Account(row["id"], row["name"], row["role"], bool(row["must_change"]), row["past_name"], row["generation"], row["created"])


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), **SCRYPT).hex()
    return "scrypt", salt, digest


def _password_ok(row, password: str) -> bool:
    if row["kdf"] == "scrypt":
        return hmac.compare_digest(_hash_password(password, row["salt"])[2], row["hash"])
    if row["kdf"] == "pbkdf2":  # the admin password of earlier versions
        return hmac.compare_digest(hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(row["salt"]), 200_000).hex(), row["hash"])
    return False


def _check_password(password: str) -> None:
    if len(password) < MIN_PASSWORD:
        raise AccountError(f"choose a password of at least {MIN_PASSWORD} characters")


def _clean_name(name: str) -> str:
    name = " ".join(str(name).split())
    if not 1 <= len(name) <= NAME_MAX:
        raise AccountError(f"enter a name of 1–{NAME_MAX} characters")
    if name.casefold() in ("admin (preview)", "guest"):
        raise AccountError("choose another name")
    return name


# ---------------------------------------------------------------- reserved past names

_RESERVED: dict[str, tuple[tuple, set[str]]] = {}


def reserved_names() -> set[str]:
    """Names (case folded) that labels of the server's projects already carry: nobody can sign up as one of them."""
    from stainid.api import registry  # noqa: PLC0415
    from stainid.labels import store  # noqa: PLC0415
    from stainid.project import load_project  # noqa: PLC0415

    found: set[str] = set()
    roots = [e.path for e in registry.entries() if e.exists]
    try:
        from stainid.api import state  # noqa: PLC0415

        roots += [str(s.root) for s in state.open_states()]
    except Exception:  # noqa: BLE001 - without the app's state
        pass
    for root in dict.fromkeys(roots):
        try:
            reviews = load_project(root).output("reviews")
        except Exception:  # noqa: BLE001 - a project that cannot be read has no names to keep
            continue
        stamp = (store.signature(reviews),)
        key = str(reviews)
        if key not in _RESERVED or _RESERVED[key][0] != stamp:
            try:
                sources = set(store.history(reviews).source.dropna().astype(str))
            except Exception:  # noqa: BLE001
                sources = set()
            _RESERVED[key] = (stamp, {_key_of(s) for s in sources if s})
        found |= _RESERVED[key][1]
    return found


# ---------------------------------------------------------------- rate limits

def _limited(kind: str, keys: list[str], limit: int) -> None:
    now = time.monotonic()
    with _LOCK:
        for key in keys:
            tries = _TRIES[(kind, key)]
            while tries and now - tries[0] > TRY_WINDOW:
                tries.popleft()
            if len(tries) >= limit * (3 if key.startswith("ip:") else 1):
                raise AccountError("too many tries: wait a quarter of an hour and try again", 429)


def _failed(kind: str, keys: list[str]) -> None:
    now = time.monotonic()
    with _LOCK:
        for key in keys:
            _TRIES[(kind, key)].append(now)


def reset_limits() -> None:
    with _LOCK:
        _TRIES.clear()


# ---------------------------------------------------------------- accounts

_GOT: dict[str, tuple[float, Account | None]] = {}
GOT_FOR = 5.0  # seconds an account read is reused (every request asks; changes made here clear it at once, others within this)


def _changed() -> None:
    _GOT.clear()
    _DEFAULT.clear()


def get(account_id: str) -> Account | None:
    seen = _GOT.get(account_id)
    if seen is not None and time.monotonic() - seen[0] < GOT_FOR:
        return seen[1]
    with _db() as db:
        found = _account(db.execute("select * from accounts where id = ?", (account_id,)).fetchone())
    if len(_GOT) > 10_000:
        _GOT.clear()
    _GOT[account_id] = (time.monotonic(), found)
    return found


def by_name(name: str) -> Account | None:
    with _db() as db:
        return _account(db.execute("select * from accounts where name_key = ?", (_key_of(name),)).fetchone())


def everyone() -> list[Account]:
    with _db() as db:
        return [_account(r) for r in db.execute("select * from accounts order by name_key")]


def register(name: str, password: str, role: str = ANNOTATOR, by_admin: bool = False) -> Account:
    """A new account (signing up: an annotator). A name another account carries is refused, and so is one past labels
    carry, unless an admin makes the account (it then sees those labels as its own)."""
    name = _clean_name(name)
    _check_password(password)
    if role not in ROLES:
        raise AccountError(f"unknown role {role}")
    if not by_admin and _key_of(name) in reserved_names():  # an admin may: they make the account of the person who labelled
        raise AccountError(f"labels by “{name}” exist already: choose another name, or ask the admin to link your account to that name", 409)
    kdf, salt, digest = _hash_password(password)
    with _LOCK, _db() as db:
        if db.execute("select 1 from accounts where name_key = ? or past_name != '' and lower(past_name) = ?", (_key_of(name), _key_of(name))).fetchone():
            raise AccountError(f"the name “{name}” is taken: choose another", 409)
        account_id = _new_id()
        db.execute("insert into accounts (id, name, name_key, role, kdf, salt, hash, created) values (?, ?, ?, ?, ?, ?, ?, ?)",
                   (account_id, name, _key_of(name), role, kdf, salt, digest, time.time()))
    _changed()
    return get(account_id)


def sign_in(name: str, password: str, address: str = "") -> Account:
    """The account with this name and password (rate-limited per name and per address)."""
    keys = [f"name:{_key_of(name)}", f"ip:{address}"]
    _limited("login", keys, LOGIN_TRIES)
    with _db() as db:
        row = db.execute("select * from accounts where name_key = ?", (_key_of(name),)).fetchone()
        if row is None or not _password_ok(row, password):
            _failed("login", keys)
            raise AccountError("wrong name or password", 403)
        if row["kdf"] != "scrypt":  # an old hash: kept as scrypt from now on
            kdf, salt, digest = _hash_password(password)
            db.execute("update accounts set kdf = ?, salt = ?, hash = ? where id = ?", (kdf, salt, digest, row["id"]))
    return _account(row)


def set_password(account_id: str, new: str, current: str | None = None) -> Account:
    """A new password (the person's own: `current` must be right; after an admin's reset any new one). Signs the
    account out everywhere else."""
    _check_password(new)
    with _LOCK, _db() as db:
        row = db.execute("select * from accounts where id = ?", (account_id,)).fetchone()
        if row is None:
            raise AccountError("no such account", 404)
        if current is not None and not row["must_change"] and not _password_ok(row, current):
            raise AccountError("the current password is wrong", 403)
        kdf, salt, digest = _hash_password(new)
        db.execute("update accounts set kdf = ?, salt = ?, hash = ?, must_change = 0, generation = generation + 1 where id = ?",
                   (kdf, salt, digest, account_id))
    _changed()
    return get(account_id)


def reset_password(account_id: str) -> str:
    """An admin's reset: a one-time password to give the person, who chooses a new one when signing in with it."""
    temporary = "-".join("".join(secrets.choice(KEY_ALPHABET) for _ in range(4)) for _ in range(3))
    kdf, salt, digest = _hash_password(temporary)
    with _LOCK, _db() as db:
        if not db.execute("update accounts set kdf = ?, salt = ?, hash = ?, must_change = 1, generation = generation + 1 where id = ?",
                          (kdf, salt, digest, account_id)).rowcount:
            raise AccountError("no such account", 404)
    _changed()
    return temporary


_DEFAULT: dict[tuple, bool] = {}


def uses_default_password(account: Account | None) -> bool:
    """Whether this admin still signs in with `admin` (the first admin, until someone changes it)."""
    if account is None:
        return False
    key = (account.id, account.generation)
    if key not in _DEFAULT:
        from stainid.api import session  # noqa: PLC0415

        with _db() as db:
            row = db.execute("select * from accounts where id = ?", (account.id,)).fetchone()
        _DEFAULT[key] = row is not None and _password_ok(row, session.DEFAULT_PASSWORD)
    return _DEFAULT[key]


def _admins(db) -> int:
    return db.execute("select count(*) from accounts where role = ?", (ADMIN,)).fetchone()[0]


def set_role(account_id: str, role: str) -> Account:
    if role not in ROLES:
        raise AccountError(f"unknown role {role}")
    with _LOCK, _db() as db:
        row = db.execute("select * from accounts where id = ?", (account_id,)).fetchone()
        if row is None:
            raise AccountError("no such account", 404)
        if row["role"] == ADMIN and role != ADMIN and _admins(db) <= 1:
            raise AccountError("this is the only admin: make someone else an admin first", 409)
        db.execute("update accounts set role = ?, generation = generation + 1 where id = ?", (role, account_id))
    _changed()
    return get(account_id)


def remove(account_id: str) -> None:
    with _LOCK, _db() as db:
        row = db.execute("select * from accounts where id = ?", (account_id,)).fetchone()
        if row is None:
            raise AccountError("no such account", 404)
        if row["role"] == ADMIN and _admins(db) <= 1:
            raise AccountError("this is the only admin: make someone else an admin first", 409)
        for table in ("tokens", "members", "tutorials"):
            db.execute(f"delete from {table} where account = ?", (account_id,))  # noqa: S608 - fixed table names
        db.execute("delete from accounts where id = ?", (account_id,))
    _changed()


def link_past_name(account_id: str, past_name: str) -> Account:
    """"This account is Ann": the account labels under the past name from now on and sees its history ("" unlinks)."""
    past_name = " ".join(str(past_name).split())
    with _LOCK, _db() as db:
        if past_name:
            if _key_of(past_name) not in reserved_names():
                raise AccountError(f"no labels carry the name “{past_name}”", 404)
            other = db.execute("select name from accounts where id != ? and (lower(past_name) = ? or name_key = ?)",
                               (account_id, _key_of(past_name), _key_of(past_name))).fetchone()
            if other is not None:
                raise AccountError(f"the account “{other['name']}” is “{past_name}” already", 409)
        if not db.execute("update accounts set past_name = ? where id = ?", (past_name, account_id)).rowcount:
            raise AccountError("no such account", 404)
    _changed()
    return get(account_id)


def past_names() -> list[dict]:
    """The names past labels carry, and which account (if any) is each."""
    linked = {_key_of(a.past_name): a for a in everyone() if a.past_name}
    named = {_key_of(a.name): a for a in everyone()}
    out = []
    for key in sorted(reserved_names()):
        account = linked.get(key) or named.get(key)
        out.append({"name": key, "account": account.id if account else None, "account_name": account.name if account else None})
    return out


# ---------------------------------------------------------------- sign-in sessions and API tokens

def session_value(account: Account) -> str:
    """What the session cookie holds: the account and its generation (a new password or role signs it out), signed."""
    from stainid.api import session  # noqa: PLC0415

    message = f"{account.id}.{account.generation}"
    return f"{message}.{session._signature('account:' + message)}"


def from_session(value: str) -> Account | None:
    from stainid.api import session  # noqa: PLC0415

    parts = (value or "").split(".")
    if len(parts) != 3:
        return None
    message = f"{parts[0]}.{parts[1]}"
    if not hmac.compare_digest(parts[2], session._signature("account:" + message)):
        return None
    account = get(parts[0])
    return account if account is not None and str(account.generation) == parts[1] else None


def new_token(account_id: str, label: str = "") -> str:
    """An API token of the account (shown once; kept hashed)."""
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    with _db() as db:
        if not db.execute("select 1 from accounts where id = ?", (account_id,)).fetchone():
            raise AccountError("no such account", 404)
        db.execute("insert into tokens (id, account, hash, label, created) values (?, ?, ?, ?, ?)",
                   (secrets.token_hex(4), account_id, hashlib.sha256(token.encode()).hexdigest(), label[:80], time.time()))
    return token


def from_token(token: str) -> Account | None:
    if not token.startswith(TOKEN_PREFIX):
        return None
    with _db() as db:
        row = db.execute("select account from tokens where hash = ?", (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
        if row is None:
            return None
        db.execute("update tokens set last_used = ? where hash = ?", (time.time(), hashlib.sha256(token.encode()).hexdigest()))
    return get(row["account"])


def tokens(account_id: str) -> list[dict]:
    with _db() as db:
        return [dict(r) for r in db.execute("select id, label, created, last_used from tokens where account = ? order by created", (account_id,))]


def revoke_token(account_id: str, token_id: str) -> None:
    with _db() as db:
        db.execute("delete from tokens where account = ? and id = ?", (account_id, token_id))


# ---------------------------------------------------------------- join keys and members

def _new_key() -> str:
    raw = "".join(secrets.choice(KEY_ALPHABET) for _ in range(KEY_LENGTH))
    return f"{raw[:5]}-{raw[5:]}"


def _normal_key(key: str) -> str:
    return "".join(c for c in str(key).lower() if c in KEY_ALPHABET)


def access(project: str, session: str, make: bool = True) -> dict:
    """A session's join key (made when it has none and `make`), whether joining is on, and its members."""
    with _LOCK, _db() as db:
        row = db.execute("select * from join_keys where project = ? and session = ?", (project, session)).fetchone()
        if row is None and make:
            db.execute("insert into join_keys (project, session, key, enabled, created) values (?, ?, ?, 1, ?)", (project, session, _new_key(), time.time()))
            row = db.execute("select * from join_keys where project = ? and session = ?", (project, session)).fetchone()
        members = [{"id": r["id"], "name": r["name"], "handle": r["past_name"] or r["name"], "role": r["role"], "joined": r["joined"]}
                   for r in db.execute("select a.*, m.joined from members m join accounts a on a.id = m.account where m.project = ? and m.session = ? "
                                       "order by m.joined", (project, session))]
    return {"key": row["key"] if row else None, "enabled": bool(row["enabled"]) if row else False, "members": members}


def regenerate_key(project: str, session: str) -> dict:
    """A new key: the old one stops working; the members keep their access."""
    with _LOCK, _db() as db:
        db.execute("insert into join_keys (project, session, key, enabled, created) values (?, ?, ?, 1, ?) "
                   "on conflict (project, session) do update set key = excluded.key, created = excluded.created", (project, session, _new_key(), time.time()))
    return access(project, session)


def set_joining(project: str, session: str, enabled: bool) -> dict:
    access(project, session)
    with _db() as db:
        db.execute("update join_keys set enabled = ? where project = ? and session = ?", (int(bool(enabled)), project, session))
    return access(project, session)


def join(account_id: str, key: str, address: str = "") -> tuple[str, str]:
    """Add the session the key opens to the account's sets: (project id, session name). Wrong keys are rate-limited."""
    keys = [f"account:{account_id}", f"ip:{address}"]
    _limited("join", keys, JOIN_TRIES)
    wanted = _normal_key(key)
    with _LOCK, _db() as db:
        row = next((r for r in db.execute("select * from join_keys") if hmac.compare_digest(_normal_key(r["key"]), wanted)), None)
        if row is None or not wanted:
            _failed("join", keys)
            raise AccountError("no labelling set has this key (it may have been replaced): check it with the person who sent it", 404)
        if not row["enabled"]:  # the right key, but the person running the study closed joining this set
            raise AccountError("joining this labelling set is switched off at the moment: ask the person running the study to switch it on", 403)
        db.execute("insert or ignore into members (account, project, session, joined) values (?, ?, ?, ?)", (account_id, row["project"], row["session"], time.time()))
    return row["project"], row["session"]


def add_member(account_id: str, project: str, session: str) -> None:
    with _db() as db:
        db.execute("insert or ignore into members (account, project, session, joined) values (?, ?, ?, ?)", (account_id, project, session, time.time()))


def remove_member(account_id: str, project: str, session: str) -> None:
    with _db() as db:
        db.execute("delete from members where account = ? and project = ? and session = ?", (account_id, project, session))


def is_member(account_id: str, project: str, session: str) -> bool:
    with _db() as db:
        return db.execute("select 1 from members where account = ? and project = ? and session = ?", (account_id, project, session)).fetchone() is not None


def sets_of(account_id: str) -> list[tuple[str, str, float]]:
    """(project id, session, when joined) of every set in the account's collection, latest first."""
    with _db() as db:
        return [(r["project"], r["session"], r["joined"]) for r in db.execute("select * from members where account = ? order by joined desc", (account_id,))]


TUTORIALS = ("objects", "areas")  # "Fix the field" and "Mark bad areas"


def tutorials_done(account_id: str) -> dict[str, bool]:
    """Which tutorials the account went through (or skipped), on any project of the server."""
    with _db() as db:
        done = {r["kind"] for r in db.execute("select kind from tutorials where account = ?", (account_id,))}
    return {kind: kind in done for kind in TUTORIALS}


def set_tutorial(account_id: str, kind: str, done: bool) -> dict[str, bool]:
    """Remember that the account finished (or skipped) a tutorial, or forget it (to be offered it again)."""
    if kind not in TUTORIALS:
        raise AccountError(f"no tutorial {kind}", 400)
    with _db() as db:
        if done:
            db.execute("insert or replace into tutorials (account, kind, done) values (?, ?, ?)", (account_id, kind, time.time()))
        else:
            db.execute("delete from tutorials where account = ? and kind = ?", (account_id, kind))
    return tutorials_done(account_id)


def forget() -> None:
    """Tests: read the store afresh (another home folder)."""
    _READY.clear()
    _RESERVED.clear()
    _changed()
    reset_limits()
