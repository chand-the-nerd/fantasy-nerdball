"""Copy everything out of the old SQLite file and into Postgres.

The app falls back to SQLite when DATABASE_URL isn't set, so a
deployment that ran that way for a while has real managers, squads and
runs sitting in a file on the volume. Pointing the app at Postgres does
not bring them along — Postgres starts empty, and it looks exactly like
every account was deleted.

This copies them across, table by table, in dependency order, and then
fixes the id sequences so the next insert doesn't collide with a row it
just copied. It refuses to run against a target that already has
managers in it unless you insist, because running it twice by accident
is a far more likely mistake than needing to.

Usage, from the repo root with the Railway CLI installed:

    railway run --service <app> python web/backend/migrate_to_postgres.py \\
        --sqlite /data/nerdball.db

Or locally against a tunnelled database:

    DATABASE_URL=postgresql://... python web/backend/migrate_to_postgres.py \\
        --sqlite ./data/nerdball.db

Add --force to write into a target that already has data, and --dry-run
to see what it would copy without touching anything.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqlalchemy import create_engine, func, select, text  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.db import _dumps  # noqa: E402
from app.models import Base, User  # noqa: E402

# Order matters: parents before children, which is exactly the order
# SQLAlchemy sorts them into for creation.
TABLES = Base.metadata.sorted_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the SQLite database into Postgres."
    )
    parser.add_argument(
        "--sqlite",
        required=True,
        help="Path to the old SQLite file, e.g. /data/nerdball.db",
    )
    parser.add_argument(
        "--target",
        default=os.getenv("DATABASE_URL", ""),
        help="Postgres URL. Defaults to DATABASE_URL.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Write even if the target already has managers in it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Say what would be copied, change nothing.",
    )
    return parser.parse_args()


def normalise(url: str) -> str:
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def main() -> int:
    args = parse_args()

    source_path = Path(args.sqlite)
    if not source_path.exists():
        print(f"No SQLite file at {source_path}")
        return 1

    if not args.target:
        print("No target. Pass --target or set DATABASE_URL.")
        return 1

    target_url = normalise(args.target)
    if target_url.startswith("sqlite"):
        print("The target is SQLite. That's the thing we're leaving.")
        return 1

    source = create_engine(f"sqlite:///{source_path}", future=True)
    target = create_engine(
        target_url, future=True, json_serializer=_dumps
    )

    # The target may be brand new.
    Base.metadata.create_all(target)

    with Session(target) as check:
        existing = check.scalar(
            select(func.count()).select_from(User)
        )
    if existing and not args.force:
        print(
            f"The target already has {existing} user(s). Refusing to "
            "write on top of them — pass --force if that's what you "
            "want."
        )
        return 1

    total = 0
    with Session(source) as reader, Session(target) as writer:
        for table in TABLES:
            # A table the old database never had is not an error: it was
            # added after that deployment stopped writing.
            try:
                rows = [
                    dict(row._mapping)
                    for row in reader.execute(select(table))
                ]
            except Exception as error:
                print(f"  {table.name}: skipped ({type(error).__name__})")
                continue

            if not rows:
                print(f"  {table.name}: empty")
                continue

            # Columns the old file doesn't have (anything added since)
            # simply aren't in the dicts, so they take their defaults.
            print(f"  {table.name}: {len(rows)} row(s)")
            total += len(rows)

            if args.dry_run:
                continue

            writer.execute(table.insert(), rows)

        if not args.dry_run:
            writer.commit()

    if args.dry_run:
        print(f"\nWould copy {total} row(s). Nothing was written.")
        return 0

    # Every id was copied verbatim, so the sequences are still at zero
    # and the next insert would collide with row 1.
    with Session(target) as fixer:
        for table in TABLES:
            if "id" not in table.columns:
                continue
            fixer.execute(
                text(
                    "SELECT setval("
                    "  pg_get_serial_sequence(:table, 'id'),"
                    "  COALESCE((SELECT MAX(id) FROM " + table.name + "), 1)"
                    ")"
                ),
                {"table": table.name},
            )
        fixer.commit()

    print(f"\nCopied {total} row(s) and reset the id sequences.")
    print("Check the admin pane before deleting the SQLite file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
