"""`stainid accounts`: the server's accounts from the command line (the same store the app uses, `api.accounts`).

    stainid accounts list
    stainid accounts add NAME [--admin]           asks for the password
    stainid accounts token NAME [--label TEXT]    a new API token (printed once) for scripts: Authorization: Bearer <token>
    stainid accounts reset NAME                   a one-time password; the person chooses a new one when signing in
    stainid accounts role NAME admin|annotator
"""
from __future__ import annotations

import getpass
import sys


def add_parser(sub) -> None:
    p = sub.add_parser("accounts", help="the server's accounts: list, add, API tokens for scripts, password resets, roles")
    actions = p.add_subparsers(dest="action", required=True)
    actions.add_parser("list", help="every account and its role")
    a = actions.add_parser("add", help="a new account (asks for its password)")
    a.add_argument("name")
    a.add_argument("--admin", action="store_true", help="an admin (default: an annotator)")
    t = actions.add_parser("token", help="a new API token of an account, for scripts and agents (printed once)")
    t.add_argument("name")
    t.add_argument("--label", default="command line", help="what the token is for (shown in Settings)")
    r = actions.add_parser("reset", help="a one-time password for an account (its owner then chooses a new one)")
    r.add_argument("name")
    o = actions.add_parser("role", help="make an account an admin or an annotator (one admin always stays)")
    o.add_argument("name")
    o.add_argument("role", choices=("admin", "annotator"))


def run(args) -> None:
    from stainid.api import accounts  # noqa: PLC0415

    def account(name: str) -> accounts.Account:
        found = accounts.by_name(name)
        if found is None:
            sys.exit(f"no account named {name}")
        return found

    try:
        if args.action == "list":
            for a in accounts.everyone():
                print(f"{a.name:<30} {a.role:<10} {('labels as ' + a.past_name) if a.past_name else ''}")
        elif args.action == "add":
            password = getpass.getpass(f"Password for {args.name}: ")
            made = accounts.register(args.name, password, accounts.ADMIN if args.admin else accounts.ANNOTATOR, by_admin=True)
            print(f"{made.name}: {made.role}")
        elif args.action == "token":
            print(accounts.new_token(account(args.name).id, args.label))
        elif args.action == "reset":
            print(accounts.reset_password(account(args.name).id))
        elif args.action == "role":
            print(accounts.set_role(account(args.name).id, args.role).role)
    except accounts.AccountError as error:
        sys.exit(str(error))
