"""Database engine and session handling."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.schema import CreateColumn

from .config import settings
from .models import Base

connect_args: dict = {}
if settings.database_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}


def _finite(value: Any) -> Any:
    """Replace anything JSON can't hold with null, recursively.

    pandas and numpy hand back NaN for a missing number — an injured
    player's chance of playing, most often. Python's json module writes
    that as the bare token NaN, which is not valid JSON. SQLite accepted
    it because it stores JSON as text and never looked; Postgres parses
    it and refuses the whole row, failing a run after the optimiser had
    already done all the work.

    Doing it in the serialiser rather than at one call site means every
    JSON column in the app is covered, including the ones added later by
    somebody who never heard of this problem.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _fallback(obj: Any) -> Any:
    """Numpy scalars, which json doesn't recognise but pandas returns."""
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _finite(item())
        except Exception:
            pass
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def _dumps(value: Any) -> str:
    return json.dumps(_finite(value), default=_fallback)


engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    future=True,
    connect_args=connect_args,
    json_serializer=_dumps,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


log = logging.getLogger(__name__)


def _add_missing_columns() -> None:
    """Add columns the models have gained since the tables were created.

    create_all only ever creates whole tables, so a new column on an existing
    one is silently absent until something tries to read it. This adds those,
    and only those: nothing is ever altered, renamed or dropped, so the worst
    case is a column that goes unused.
    """
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    for table in Base.metadata.sorted_tables:
        if table.name not in tables:
            continue
        present = {column["name"] for column in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in present:
                continue
            if not column.nullable and column.server_default is None:
                # No safe value for the existing rows; leave it to a real
                # migration rather than guessing one.
                log.warning(
                    "Column %s.%s is missing and has no default; skipping",
                    table.name,
                    column.name,
                )
                continue
            spec = CreateColumn(column).compile(dialect=engine.dialect)
            try:
                with engine.begin() as connection:
                    connection.exec_driver_sql(
                        f"ALTER TABLE {table.name} ADD COLUMN {spec}"
                    )
                log.info("Added column %s.%s", table.name, column.name)
            except Exception:
                log.exception("Could not add column %s.%s", table.name, column.name)


# One arbitrary but fixed number, so every process asks for the same
# lock. Postgres advisory locks are just an integer namespace.
SCHEMA_LOCK_KEY = 8_244_101


@contextmanager
def _schema_lock() -> Iterator[None]:
    """Hold the schema open for one process at a time.

    With more than one uvicorn worker, every process runs init_db at
    once. create_all checks whether a table exists and then creates it,
    and two processes can both pass the check before either finishes —
    at which point the loser dies with a duplicate key on
    pg_type_typname_nsp_index and takes the deploy with it.

    An advisory lock serialises them: the second process waits, then
    finds the tables already there and does nothing. SQLite has no such
    thing, but it also can't run several worker processes usefully, so
    there is nothing to serialise.
    """
    if settings.database_backend != "postgres":
        yield
        return

    connection = engine.connect()
    try:
        connection.execute(
            text("SELECT pg_advisory_lock(:key)"), {"key": SCHEMA_LOCK_KEY}
        )
        connection.commit()
        yield
    finally:
        try:
            connection.execute(
                text("SELECT pg_advisory_unlock(:key)"),
                {"key": SCHEMA_LOCK_KEY},
            )
            connection.commit()
        except Exception:
            log.warning("Could not release the schema lock", exc_info=True)
        connection.close()


def init_db() -> None:
    with _schema_lock():
        Base.metadata.create_all(engine)
        _add_missing_columns()


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Standalone session for the background worker."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
