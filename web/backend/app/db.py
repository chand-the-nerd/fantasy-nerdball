"""Database engine and session handling."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import logging

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.schema import CreateColumn

from .config import settings
from .models import Base

connect_args: dict = {}
if settings.database_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    future=True,
    connect_args=connect_args,
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


def init_db() -> None:
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
