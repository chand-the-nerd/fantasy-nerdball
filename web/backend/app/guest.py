"""Guest access: what an unsigned-in visitor gets, and what it costs.

A guest is a real row in the users table, because everything below this
layer expects a user id, but it is marked as one, capped by the constants
here, and deleted when the session ends. Keeping the policy in a single
module means the table on the sign-in page, the limits the server
enforces and the controls the browser greys out all describe the same
product rather than three drifting approximations of it.

The frontend mirror of these numbers lives in web/frontend/src/lib/
guest.ts. Change one, change the other.
"""

from __future__ import annotations

import datetime as dt
import secrets

from fastapi import HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .config import settings
from .models import (
    GameweekResult,
    Plan,
    PlayerScores,
    Run,
    Squad,
    User,
    UserSettings,
    utcnow,
)

# ── The locked model ─────────────────────────────────────────────────────
#
# A guest gets one sensible configuration rather than a tuning panel. The
# sliders that would move these are greyed out in the browser; enforce()
# below puts them back regardless of what arrives.

GAMEWEEKS = 5
TRANSFER_STRATEGY = 1.2
BENCH_WEIGHT = 0.2
WEIGHTS = {"form": 0.4, "historic": 0.3, "difficulty": 0.3}
POSITIONS = ("GK", "DEF", "MID", "FWD")

MAX_FORCED = 2
MAX_BLACKLIST = 2

LOCKED: dict[str, object] = {
    "first_n_gameweeks": GAMEWEEKS,
    "min_transfer_value": TRANSFER_STRATEGY,
    "bench_weight": BENCH_WEIGHT,
    "accept_transfer_penalty": True,
    "exclude_unavailable": True,
    # Trained weights would override the fixed 40/30/30 below, which is
    # the one thing a guest is told the model is doing.
    "use_ml_weights": False,
    # Nothing carries over between guest sessions, so there is no squad
    # from two gameweeks ago for this to reach back to.
    "free_hit_prev_gw": False,
}


def default_weights() -> dict[str, dict[str, float]]:
    return {position: dict(WEIGHTS) for position in POSITIONS}


def new_settings() -> UserSettings:
    """The settings row a guest starts with."""
    row = UserSettings(**LOCKED)
    row.overrides = {"POSITION_SCORING_WEIGHTS": default_weights()}
    row.team_modifiers = {}
    row.forced_selections = {}
    row.blacklist_players = []
    return row


def enforce(row: UserSettings) -> None:
    """Put a guest's settings back inside the guest limits.

    Called after a save rather than instead of one: budget, free
    transfers, chips and the theme are all still theirs to set, so the
    save goes through and only the locked fields are put back.
    """
    for field, value in LOCKED.items():
        setattr(row, field, value)

    overrides = dict(row.overrides or {})
    overrides["POSITION_SCORING_WEIGHTS"] = default_weights()
    row.overrides = overrides

    # Every club neutral. Stored as an empty dict, which is what the
    # engine reads as 1.00 across the board.
    row.team_modifiers = {}


def limit(what: str, allowed: int) -> HTTPException:
    return HTTPException(
        status.HTTP_403_FORBIDDEN,
        f"Without signing in you can set {allowed} {what}. Sign in for "
        "the rest.",
    )


def members_only(feature: str) -> HTTPException:
    return HTTPException(
        status.HTTP_403_FORBIDDEN,
        f"{feature} is for signed-in users. Sign in to use it.",
    )


def check_lists(
    forced: dict | None = None, blacklist: list | None = None
) -> None:
    """Reject a save that would put a guest over either list cap."""
    if forced is not None:
        total = sum(len(names or []) for names in forced.values())
        if total > MAX_FORCED:
            raise limit("forced picks", MAX_FORCED)
    if blacklist is not None and len(blacklist) > MAX_BLACKLIST:
        raise limit("players to avoid", MAX_BLACKLIST)


# ── Lifecycle ────────────────────────────────────────────────────────────


def create(session: Session) -> User:
    """A fresh throwaway account, with the guest settings already on it."""
    purge_expired(session)

    token = secrets.token_hex(8)
    user = User(
        # .invalid is reserved by RFC 2606, so this can never collide
        # with a real address somebody later gets invited under.
        email=f"guest-{token}@guests.invalid",
        name="Guest",
        is_guest=True,
    )
    user.settings = new_settings()
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def discard(session: Session, user: User) -> None:
    """Delete a guest and everything it owns. Signing out is the end.

    Rows are removed explicitly rather than left to the foreign keys:
    ON DELETE CASCADE does nothing under SQLite without the pragma, and
    the Plan relationship has no cascade of its own.
    """
    if not user.is_guest:
        return

    for model in (Run, Plan, PlayerScores, GameweekResult, Squad):
        session.execute(delete(model).where(model.user_id == user.id))
    session.delete(user)
    session.commit()


def purge_expired(session: Session) -> int:
    """Clear out guest sessions nobody has come back to."""
    cutoff = utcnow() - dt.timedelta(hours=settings.guest_ttl_hours)
    stale = session.scalars(
        select(User).where(
            User.is_guest.is_(True), User.last_seen_at < cutoff
        )
    ).all()
    for user in stale:
        discard(session, user)
    return len(stale)


def touch(session: Session, user: User) -> None:
    """Keep an active guest from being purged mid-session.

    Written at most every ten minutes, so a busy page doesn't turn every
    request into a database write.
    """
    if not user.is_guest:
        return
    last = user.last_seen_at
    if last is not None and last.tzinfo is None:
        last = last.replace(tzinfo=dt.timezone.utc)
    if last is None or utcnow() - last > dt.timedelta(minutes=10):
        user.last_seen_at = utcnow()
        session.commit()
