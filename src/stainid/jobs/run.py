"""Run one command for the server and leave its exit code in a file: `python -m stainid.jobs.run EXIT_FILE -- COMMAND...`.

The command runs in its own process group. Stopping this runner (SIGTERM, SIGINT, SIGHUP) stops that whole group, and
the exit code is written when the command ends, so a server restarted in the meantime still learns how the job ended
(a negative code means the command was ended by that signal, as in `subprocess`).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from stainid.tables import write_text_atomic


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] != "--":
        print(__doc__.splitlines()[0], file=sys.stderr)
        return 2
    exit_file, command = Path(argv[0]), argv[2:]
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
        child = subprocess.Popen(command, start_new_session=True)
    except OSError as error:
        print(f"Could not start {command[0]}: {error}", flush=True)
        write_text_atomic(exit_file, "127")
        return 127
    for signum in pending:
        forward(signum, None)
    code = child.wait()
    write_text_atomic(exit_file, str(code))
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
