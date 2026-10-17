"""Run one command for the queue and leave its exit code in a file:
`python -m stainid.jobs.run EXIT_FILE [--pid-file FILE] [--env-file FILE] -- COMMAND...`.

The command runs in its own process group. Stopping this runner (SIGTERM, SIGINT, SIGHUP) stops that whole group, and
the exit code is written when the command ends, so a server restarted in the meantime still learns how the job ended
(a negative code means the command was ended by that signal, as in `subprocess`). `--pid-file` gets this runner's and
the command's process numbers once it starts; `--env-file` (JSON) is the command's whole environment (a job started
by systemd gets the queue's environment this way, without it showing in the unit's settings).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from stainid.tables import write_text_atomic


def main(argv: list[str]) -> int:
    options = {}
    rest = argv[1:]
    while len(rest) >= 2 and rest[0] in ("--pid-file", "--env-file"):
        options[rest[0]], rest = rest[1], rest[2:]
    if not argv or len(rest) < 2 or rest[0] != "--":
        print(__doc__.splitlines()[0], file=sys.stderr)
        return 2
    exit_file, command = Path(argv[0]), rest[1:]
    env = None
    if "--env-file" in options:
        try:
            env = json.loads(Path(options["--env-file"]).read_text())
        except (OSError, ValueError) as error:
            print(f"Could not read the environment file: {error}", flush=True)
            write_text_atomic(exit_file, "126")
            return 126
    child: subprocess.Popen | None = None
    pending: list[int] = []

    def forward(signum, frame):
        if child is None:
            pending.append(signum)
            return
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, forward)
    try:
        child = subprocess.Popen(command, start_new_session=True, env=env)
    except OSError as error:
        print(f"Could not start {command[0]}: {error}", flush=True)
        write_text_atomic(exit_file, "127")
        return 127
    if "--pid-file" in options:
        write_text_atomic(Path(options["--pid-file"]), f"{os.getpid()} {child.pid}")
    for signum in pending:
        forward(signum, None)
    code = child.wait()
    write_text_atomic(exit_file, str(code))
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
