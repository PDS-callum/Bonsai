#!/usr/bin/env python3
"""Run a command and record its peak resident memory.

The command's output passes straight through. When it exits, its peak
resident set size in bytes is written to <out-file>, and its exit code is
passed on. (GNU time would do this, but it is not installed by default.)

Usage: peak_rss.py <out-file> -- <command> [args...]
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        raise SystemExit(__doc__)
    out_file = Path(sys.argv[1])
    process = subprocess.Popen(sys.argv[3:])
    _, status, usage = os.wait4(process.pid, 0)
    # On Linux, ru_maxrss is in kibibytes.
    out_file.write_text(f"{usage.ru_maxrss * 1024}\n", encoding="utf-8")
    sys.exit(os.waitstatus_to_exitcode(status))


if __name__ == "__main__":
    main()
