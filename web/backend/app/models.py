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
    false,
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

    # A throwaway account behind "Continue without signing in". It never
    # counts towards the seat cap, and everything it owns is deleted when
    # the session ends or the row goes stale.
    is_guest: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=false()
    )

    # Optional link to the manager's real FPL side, so actual points can be
    # pulled and charted next to the model's projection.
    fpl_entry_id: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_seen_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # When their place was given up for inactivity. A dormant manager
    # keeps everything they had — squads, runs, settings — and simply
    # stops occupying a seat. Signing back in clears this and hands it
    # all back. Deleting an account outright happens only after the far
    # longer window in PURGE_AFTER_MONTHS.
    dormant_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # When a dormant manager's season data was cleared out. Their FPL id
    # survives this, which is the one thing worth carrying between
    # seasons — everything else describes a season that has ended.
    data_purged_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

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

    # When the invitation stops working, so a held place doesn't sit
    # unused while somebody waits for it. Null means it never expires,
    # which is what invites added by hand before this existed are — and
    # what anything you add directly should stay.
    expires_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


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
    # Playing a Free Hit this week. Unlimited transfers like a Wildcard, but
    # the side reverts afterwards, so only this gameweek is worth planning for.
    free_hit: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=false()
    )
    # Having played one last week, which is a different thing: the squad the
    # optimiser should transfer from is the one from two gameweeks ago.
    free_hit_prev_gw: Mapped[bool] = mapped_column(Boolean, default=False)
    bench_boost: Mapped[bool] = mapped_column(Boolean, default=False)
    triple_captain: Mapped[bool] = mapped_column(Boolean, default=False)

    # Set once the guided tour has been finished or skipped, so it opens by
    # itself exactly once and never again unless asked for.
    tutorial_seen: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=false()
    )

    # Which palette the app renders in. A display preference rather than an
    # engine one, but it belongs to the manager, so it lives with the rest.
    theme: Mapped[str] = mapped_column(
        String(16), default="legacy", server_default="legacy"
    )

    # How much a bench player's score counts when picking the squad, as a
    # proportion of a starter's. Zero optimises the eleven alone and lets
    # the bench be whatever the budget leaves; one makes all fifteen count
    # equally, which is what a Bench Boost actually does.
    bench_weight: Mapped[float] = mapped_column(
        Float, default=0.2, server_default="0.2"
    )

    use_ml_weights: Mapped[bool] = mapped_column(Boolean, default=False)
    first_n_gameweeks: Mapped[int] = mapped_column(Integer, default=1)
    min_transfer_value: Mapped[float] = mapped_column(Float, default=2.0)
    # Retained but no longer read: a transfer's benefit is now counted over
    # first_n_gameweeks, the same window the scores are built over. Left in
    # place because nothing here drops columns, and an unused one is
    # harmless where a lost one is not.
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
    # Also carries every option the run offered, under "options".
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # Verbatim rows the engine wrote to squads/gw{n}/full_squad.csv, so a
    # later run can be handed exactly the file it expects.
    engine_rows: Mapped[list[Any]] = mapped_column(JSON, default=list)

    # Which of the run's options is in force. The columns above and the
    # payload's own starting XI always describe this one.
    active_option: Mapped[str] = mapped_column(
        String(24), default="option-1", server_default="option-1"
    )
    # Engine rows for every option, keyed the same way, so activating one
    # can hand next week's run the fifteen that were actually kept. Kept out
    # of the payload deliberately: it is several times the size of everything
    # the browser needs, and the browser never reads it.
    option_rows: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, default=dict, nullable=True
    )

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


class Plan(Base):
    """A multi-gameweek plan: the optimiser run forward, week after week.

    Kept apart from Run and Squad on purpose. A plan is speculative — it
    assumes today's prices and today's form hold for two months — so it must
    never be mistaken for the squad you actually have.
    """

    __tablename__ = "plans"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    season: Mapped[str] = mapped_column(String(9))
    start_gameweek: Mapped[int] = mapped_column(Integer)
    weeks: Mapped[int] = mapped_column(Integer)
    # {"7": "wildcard"} — which chip is meant to be played in which gameweek.
    chips: Mapped[dict] = mapped_column(JSON, default=dict)

    status: Mapped[str] = mapped_column(String(16), default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    log: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    # One entry per planned gameweek.
    payload: Mapped[list] = mapped_column(JSON, default=list)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    user: Mapped["User"] = relationship()


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


class MetricEvent(Base):
    """An append-only record of things people did, for the admin dashboard.

    Deliberately not joined to users by a foreign key. Guest accounts are
    deleted the moment their session ends, and a cascade would take the
    history with them — leaving a dashboard that can only ever describe
    the people still signed in. The user id is kept as a plain integer so
    a member's activity can still be resolved by name, and anything left
    dangling reads as a visitor who has since gone.

    Written from a background thread and pruned on a timer: see
    metrics.py, and web/OBSERVABILITY.md for what each kind means.
    """

    __tablename__ = "metric_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    kind: Mapped[str] = mapped_column(String(48), index=True)

    # Who, at the coarsest resolution that still answers the question.
    # `visitor` is the stable identity used for unique counts: a user id
    # for a signed-in manager, a salted hash of the address for a guest.
    visitor: Mapped[str] = mapped_column(String(64), index=True, default="")
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_guest: Mapped[bool] = mapped_column(Boolean, default=False)
    # Only populated for guests, and only as far as VISITOR_IP_MODE
    # allows: there is no name to fall back on when they leave.
    label: Mapped[str] = mapped_column(String(64), default="")

    # A duration in seconds for runs, or whatever a kind wants to chart.
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class InboxItem(Base):
    """Something a person sent the admin: a request to join, or feedback.

    One table for both, because they want the same handling — arrive,
    get read, get dealt with — and a single inbox is easier to keep on
    top of than two. `kind` separates them; `email` is whoever sent it,
    which for an access request is the Google address they want let in.

    Not joined to users by a foreign key: an access request comes from
    somebody who by definition has no account yet, and feedback from a
    guest should outlive the guest.
    """

    __tablename__ = "inbox_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(24), index=True)
    email: Mapped[str] = mapped_column(String(320), default="")
    name: Mapped[str] = mapped_column(String(120), default="")
    body: Mapped[str] = mapped_column(Text, default="")

    # Who sent it, when they were signed in. Kept as a plain integer so
    # a departed account doesn't take its feedback with it.
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    from_guest: Mapped[bool] = mapped_column(Boolean, default=False)

    status: Mapped[str] = mapped_column(
        String(16), default="new", index=True, server_default="new"
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    handled_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    handled_by: Mapped[str] = mapped_column(String(320), default="")
