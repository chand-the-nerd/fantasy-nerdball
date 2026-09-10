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
import shutil
import threading
import time

from fastapi import HTTPException, status
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from . import events
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
    events.emit("guest_blocked", feature=what, kind="limit")
    return HTTPException(
        status.HTTP_403_FORBIDDEN,
        f"Without signing in you can set {allowed} {what}. Sign in for "
        "the rest.",
    )


def members_only(feature: str) -> HTTPException:
    events.emit("guest_blocked", feature=feature, kind="locked")
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


# Guest sessions are unauthenticated, so the endpoint that makes them is
# the cheapest thing on the site to abuse: each one can queue minutes of
# CPU on a worker there is only one of. These two limits are what stand
# between a bored stranger and the bill.
_starts: dict[str, list[float]] = {}
_starts_lock = threading.Lock()


def too_many_from(key: str) -> bool:
    """Whether this caller has started more guest sessions than allowed."""
    if settings.guest_starts_per_hour <= 0:
        return False

    now = time.monotonic()
    with _starts_lock:
        seen = [t for t in _starts.get(key, []) if now - t < 3600]
        if len(seen) >= settings.guest_starts_per_hour:
            _starts[key] = seen
            return True
        seen.append(now)
        _starts[key] = seen

        if len(_starts) > 5000:
            for stale in [
                k for k, times in _starts.items() if not times
                or now - times[-1] > 3600
            ]:
                _starts.pop(stale, None)
    return False


def at_capacity(session: Session) -> bool:
    """Whether there are already as many live guests as allowed.

    A blunt instrument on purpose. Turning people away for a few minutes
    is a far better failure than an optimiser queue nobody real can get
    into, or a disk full of abandoned workspaces.
    """
    if settings.max_live_guests <= 0:
        return False
    live = session.scalar(
        select(func.count())
        .select_from(User)
        .where(User.is_guest.is_(True))
    )
    return int(live or 0) >= settings.max_live_guests


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

    user_id = user.id
    for model in (Run, Plan, PlayerScores, GameweekResult, Squad):
        session.execute(delete(model).where(model.user_id == user.id))
    session.delete(user)
    session.commit()

    # And the working directory on the volume. Without this the rows go
    # and the folders stay, which is invisible until the disk is full.
    _remove_workspace(user_id)


def _remove_workspace(user_id: int) -> None:
    try:
        from .engine.workspace import user_root

        path = user_root(user_id)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        # A folder left behind is a slow problem; an exception here would
        # be an immediate one, in the middle of somebody signing out.
        pass


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
    if stale:
        events.emit("guests_purged", count=len(stale))
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
