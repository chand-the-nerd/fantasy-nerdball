#!/usr/bin/env python3
"""Run every suite and say which ones passed.

Each file under tests/ is a standalone script that prints PASS or FAIL
per check and exits non-zero on failure. No pytest, no fixtures, no
conftest — they run against a real temporary database and drive the real
app through a test client, because the bugs worth catching here have all
been integration bugs: a race between worker processes, a NaN Postgres
won't take, an orphaned row SQLite allowed and Postgres won't.

    python web/backend/tests/run_all.py           # everything it can
    python web/backend/tests/run_all.py --quick   # skip Postgres suites

The Postgres suites need a server on hand. Set NERDBALL_TEST_PG to a
connection string, or leave it and they'll be skipped with a note rather
than failing. Everything else runs against throwaway SQLite files and
needs nothing at all.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Suites that need a real Postgres, and what they cover without one.
NEEDS_POSTGRES = {
    "test_migrate.py": "SQLite to Postgres migration",
    "test_race.py": "concurrent schema creation",
    "test_backup.py": "backup and restore round trip",
}


def main() -> int:
    quick = "--quick" in sys.argv
    have_pg = bool(os.getenv("NERDBALL_TEST_PG"))

    suites = sorted(HERE.glob("test_*.py"))
    if not suites:
        print("No suites found.")
        return 1

    failed = []
    skipped = []

    for suite in suites:
        if suite.name in NEEDS_POSTGRES and (quick or not have_pg):
            skipped.append(suite.name)
            continue

        print(f"\n─── {suite.name} " + "─" * (52 - len(suite.name)))
        result = subprocess.run(
            [sys.executable, str(suite)],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if line.startswith(("PASS", "FAIL", "ALL PASS", "FAILURES")):
                print(line)

        if result.returncode != 0:
            failed.append(suite.name)
            tail = (result.stderr or result.stdout).strip().splitlines()
            for line in tail[-6:]:
                print(f"    {line}")

    print("\n" + "═" * 60)
    if skipped:
        print(f"Skipped (no Postgres): {', '.join(skipped)}")
        print("Set NERDBALL_TEST_PG to run them.")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"All {len(suites) - len(skipped)} suite(s) passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
