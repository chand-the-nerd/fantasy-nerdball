"""Daily backups, without a paid plan or a second service.

Railway's volume backups and point-in-time recovery are Pro features.
This is the cheap version: once a day the whole database is written to a
compressed JSON file on the volume the app already has, old ones are
deleted, and the admin page lists them for download.

Be clear about what this does and doesn't protect against. It covers the
thing most likely to bite — a bug, or a misconfigured INACTIVE_DAYS,
deleting rows that mattered — because yesterday's file still has them.
It does not cover losing the volume itself, since the backups are on it.
Downloading one occasionally is what closes that gap, which is why the
admin page offers them as a download rather than only listing them.

JSON rather than pg_dump because the container has no Postgres client
binaries, and a dump written by the wrong client version refuses to
restore anyway. The format is the same table-by-table copy the SQLite
migration uses, so restoring is the same code path in reverse.
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from . import events
from .config import settings
from .db import _dumps, engine
from .models import Base

log = logging.getLogger("nerdball.backup")

FORMAT_VERSION = 1


def backup_dir() -> Path:
    path = settings.data_dir / "backups"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _encode(value: Any) -> Any:
    """Make one column value safe to write and unambiguous to read back."""
    if isinstance(value, dt.datetime):
        return {"__dt__": value.isoformat()}
    if isinstance(value, dt.date):
        return {"__date__": value.isoformat()}
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if "__dt__" in value:
            return dt.datetime.fromisoformat(value["__dt__"])
        if "__date__" in value:
            return dt.date.fromisoformat(value["__date__"])
    return value


def create(label: str = "") -> Path:
    """Write one backup. Returns the file it wrote."""
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = f"-{label}" if label else ""
    target = backup_dir() / f"nerdball-{stamp}{suffix}.json.gz"

    payload: dict[str, Any] = {
        "version": FORMAT_VERSION,
        "taken_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "season": settings.current_season,
        "tables": {},
    }

    rows_total = 0
    with Session(engine) as session:
        for table in Base.metadata.sorted_tables:
            rows = []
            for row in session.execute(select(table)):
                rows.append(
                    {
                        key: _encode(value)
                        for key, value in dict(row._mapping).items()
                    }
                )
            payload["tables"][table.name] = rows
            rows_total += len(rows)

    # _dumps rather than json.dumps: the same NaN handling the database
    # itself uses, so a backup can always be read back in.
    text = _dumps(payload)
    with gzip.open(target, "wt", encoding="utf-8") as handle:
        handle.write(text)

    size = target.stat().st_size
    events.emit(
        "backup_created",
        file=target.name,
        rows=rows_total,
        bytes=size,
    )
    log.info("Wrote %s (%s rows, %s bytes)", target.name, rows_total, size)
    return target


def prune(keep: int) -> int:
    """Delete all but the newest `keep` backups."""
    if keep <= 0:
        return 0
    files = sorted(
        backup_dir().glob("nerdball-*.json.gz"),
        key=lambda path: path.name,
        reverse=True,
    )
    removed = 0
    for path in files[keep:]:
        try:
            path.unlink()
            removed += 1
        except OSError:
            log.warning("Could not delete %s", path.name)
    return removed


def listing() -> list[dict[str, Any]]:
    """Every backup on disk, newest first."""
    out = []
    for path in sorted(
        backup_dir().glob("nerdball-*.json.gz"), reverse=True
    ):
        stat = path.stat()
        out.append(
            {
                "name": path.name,
                "bytes": stat.st_size,
                "taken_at": dt.datetime.fromtimestamp(
                    stat.st_mtime, tz=dt.timezone.utc
                ).isoformat(),
            }
        )
    return out


def restore(path: Path, force: bool = False) -> dict[str, int]:
    """Load a backup back in, replacing everything currently there.

    Destructive by definition, so it refuses to run against a database
    that has data unless you say force. Tables are emptied in reverse
    dependency order and refilled in forward order, which is what keeps
    the foreign keys happy in both directions.
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("version") != FORMAT_VERSION:
        raise ValueError(
            f"Backup format {payload.get('version')} isn't one this "
            f"version understands (expected {FORMAT_VERSION})."
        )

    Base.metadata.create_all(engine)

    with Session(engine) as session:
        from .models import User

        existing = session.scalar(select(User))
        if existing is not None and not force:
            raise RuntimeError(
                "The database already has users in it. Pass force to "
                "replace them with the backup."
            )

        for table in reversed(Base.metadata.sorted_tables):
            session.execute(delete(table))

        counts: dict[str, int] = {}
        for table in Base.metadata.sorted_tables:
            rows = payload["tables"].get(table.name) or []
            if not rows:
                continue
            decoded = [
                {key: _decode(value) for key, value in row.items()}
                for row in rows
            ]
            session.execute(table.insert(), decoded)
            counts[table.name] = len(decoded)

        session.commit()

    _reset_sequences()
    events.emit("backup_restored", file=path.name, tables=len(counts))
    return counts


def _reset_sequences() -> None:
    """Put the id sequences past the rows just written."""
    if settings.database_backend != "postgres":
        return
    from sqlalchemy import text

    with engine.begin() as connection:
        for table in Base.metadata.sorted_tables:
            if "id" not in table.columns:
                continue
            connection.execute(
                text(
                    "SELECT setval("
                    "  pg_get_serial_sequence(:t, 'id'),"
                    "  COALESCE((SELECT MAX(id) FROM " + table.name + "), 1)"
                    ")"
                ),
                {"t": table.name},
            )


def start_scheduler() -> None:
    """Take one backup a day.

    Runs under the same single-process lease as the other shared jobs,
    so several workers don't all write their own copy. Takes one shortly
    after boot too: a deploy is exactly when something might go wrong,
    and the backup you wish you had is the one from just before.
    """
    if settings.backup_every_hours <= 0:
        return

    def loop() -> None:
        time.sleep(120)
        while True:
            try:
                create()
                removed = prune(settings.backup_keep)
                if removed:
                    log.info("Pruned %s old backup(s)", removed)
            except Exception:
                log.warning("Backup failed", exc_info=True)
                events.emit("backup_failed")
            time.sleep(settings.backup_every_hours * 3600)

    threading.Thread(target=loop, name="backup", daemon=True).start()


def _main() -> int:
    """Command line, for taking or restoring one by hand."""
    import argparse

    parser = argparse.ArgumentParser(description="Database backups.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("create", help="Write a backup now")
    sub.add_parser("list", help="List the backups on disk")
    restore_cmd = sub.add_parser("restore", help="Load one back in")
    restore_cmd.add_argument("file", help="Filename or full path")
    restore_cmd.add_argument(
        "--force",
        action="store_true",
        help="Replace the data that's there now",
    )

    args = parser.parse_args()

    if args.command == "create":
        print(create())
        return 0

    if args.command == "list":
        for item in listing():
            size = item["bytes"] / 1024
            print(f"{item['name']}  {size:.0f} KB  {item['taken_at']}")
        return 0

    path = Path(args.file)
    if not path.is_absolute():
        path = backup_dir() / args.file
    if not path.exists():
        print(f"No backup at {path}")
        return 1

    try:
        counts = restore(path, force=args.force)
    except RuntimeError as error:
        print(error)
        return 1

    total = sum(counts.values())
    print(f"Restored {total} row(s) across {len(counts)} table(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
