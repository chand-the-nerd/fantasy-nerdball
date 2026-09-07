"""Database models."""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    google_sub: Mapped[str | None] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(120), default="")
    avatar_url: Mapped[str] = mapped_column(String(512), default="")
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)

    # Optional link to the manager's real FPL side, so actual points can be
    # pulled and charted next to the model's projection.
    fpl_entry_id: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    settings: Mapped["UserSettings"] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    squads: Mapped[list["Squad"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Invite(Base):
    """An email the owner has allowed in, on top of ALLOWED_EMAILS."""

    __tablename__ = "invites"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    invited_by: Mapped[str] = mapped_column(String(320), default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class UserSettings(Base):
    """Per-manager optimiser settings. Maps onto the engine's Config class."""

    __tablename__ = "user_settings"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True)

    budget: Mapped[float] = mapped_column(Float, default=100.0)
    free_transfers: Mapped[int] = mapped_column(Integer, default=1)
    accept_transfer_penalty: Mapped[bool] = mapped_column(Boolean, default=True)
    exclude_unavailable: Mapped[bool] = mapped_column(Boolean, default=True)

    wildcard: Mapped[bool] = mapped_column(Boolean, default=False)
    free_hit_prev_gw: Mapped[bool] = mapped_column(Boolean, default=False)
    bench_boost: Mapped[bool] = mapped_column(Boolean, default=False)
    triple_captain: Mapped[bool] = mapped_column(Boolean, default=False)

    use_ml_weights: Mapped[bool] = mapped_column(Boolean, default=False)
    first_n_gameweeks: Mapped[int] = mapped_column(Integer, default=1)
    min_transfer_value: Mapped[float] = mapped_column(Float, default=2.0)
    transfer_horizon_gws: Mapped[int] = mapped_column(Integer, default=4)

    # Free-form overrides, keyed by the engine's Config attribute name.
    # Anything here wins over the columns above.
    overrides: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    team_modifiers: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    forced_selections: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    blacklist_players: Mapped[list[Any]] = mapped_column(JSON, default=list)

    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    user: Mapped[User] = relationship(back_populates="settings")


class Squad(Base):
    """A saved squad for one manager, one gameweek."""

    __tablename__ = "squads"
    __table_args__ = (UniqueConstraint("user_id", "season", "gameweek", name="uq_squad_gw"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    season: Mapped[str] = mapped_column(String(16), index=True)
    gameweek: Mapped[int] = mapped_column(Integer, index=True)

    formation: Mapped[str] = mapped_column(String(16), default="")
    projected_points: Mapped[float] = mapped_column(Float, default=0.0)
    squad_value: Mapped[float] = mapped_column(Float, default=0.0)
    bank: Mapped[float] = mapped_column(Float, default=0.0)
    transfers_made: Mapped[int] = mapped_column(Integer, default=0)
    penalty_points: Mapped[int] = mapped_column(Integer, default=0)
    chip: Mapped[str] = mapped_column(String(24), default="")

    # Structured squad used by the UI: starting XI, bench, captain flags.
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # Verbatim rows the engine wrote to squads/gw{n}/full_squad.csv, so a
    # later run can be handed exactly the file it expects.
    engine_rows: Mapped[list[Any]] = mapped_column(JSON, default=list)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped[User] = relationship(back_populates="squads")


class Run(Base):
    """One optimisation job."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    season: Mapped[str] = mapped_column(String(16))
    gameweek: Mapped[int] = mapped_column(Integer)

    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    log: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    squad_id: Mapped[int | None] = mapped_column(ForeignKey("squads.id", ondelete="SET NULL"))
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))


class PlayerScores(Base):
    """The scored player pool from a manager's most recent run.

    Scoring every player is the expensive half of an optimisation, so the
    Players tab reads what the run already produced rather than recomputing
    it. That also means the rankings shown are exactly the ones the model
    used, not a second opinion.
    """

    __tablename__ = "player_scores"
    __table_args__ = (
        UniqueConstraint("user_id", "season", name="uq_player_scores"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    season: Mapped[str] = mapped_column(String(16), index=True)
    gameweek: Mapped[int] = mapped_column(Integer)
    look_ahead: Mapped[int] = mapped_column(Integer, default=1)
    players: Mapped[list[Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class GameweekStat(Base):
    """Global FPL benchmarks for a gameweek, cached from bootstrap-static."""

    __tablename__ = "gameweek_stats"
    __table_args__ = (UniqueConstraint("season", "gameweek", name="uq_gw_stat"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    season: Mapped[str] = mapped_column(String(16), index=True)
    gameweek: Mapped[int] = mapped_column(Integer, index=True)
    average_score: Mapped[float] = mapped_column(Float, default=0.0)
    highest_score: Mapped[float] = mapped_column(Float, default=0.0)
    finished: Mapped[bool] = mapped_column(Boolean, default=False)
    deadline: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class GameweekResult(Base):
    """What a manager actually scored, next to what the model projected."""

    __tablename__ = "gameweek_results"
    __table_args__ = (UniqueConstraint("user_id", "season", "gameweek", name="uq_gw_result"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    season: Mapped[str] = mapped_column(String(16), index=True)
    gameweek: Mapped[int] = mapped_column(Integer, index=True)

    projected_points: Mapped[float | None] = mapped_column(Float)
    actual_points: Mapped[float | None] = mapped_column(Float)
    overall_rank: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(16), default="manual")
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
