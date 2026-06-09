#!/usr/bin/env python3
"""Start a long-running local service outside the parent shell session."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def parse_env(items: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --env value {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError("invalid --env value with empty key")
        values[key] = value
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--pidfile", required=True)
    parser.add_argument("--logfile", required=True)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")

    env = os.environ.copy()
    env.update(parse_env(args.env))

    os.makedirs(os.path.dirname(os.path.abspath(args.pidfile)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.logfile)), exist_ok=True)

    with open(args.logfile, "ab", buffering=0) as log_file:
        process = subprocess.Popen(
            command,
            cwd=args.cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    with open(args.pidfile, "w", encoding="utf-8") as pid_file:
        pid_file.write(f"{process.pid}\n")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"detached_process failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
