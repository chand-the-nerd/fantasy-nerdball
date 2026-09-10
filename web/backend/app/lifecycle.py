"""Places don't sit unused.

Two rules, both stated in the email that welcomes somebody in, and both
enforced here rather than left to the admin to remember:

An invitation lasts INVITE_TTL_HOURS. If it isn't used, it lapses and
the place goes back to whoever is waiting.

A manager who hasn't signed in for INACTIVE_DAYS gives their place up.
Both they and the admin are told, and they can ask for it back.

Removal is destructive, so the guards matter more than the sweep: admins
are never removed, guests are handled by their own shorter clock, and
nothing happens at all if the deployment hasn't set an inactivity
window. It runs on a timer under the same single-process lease as the
history scheduler, so with several workers it happens once.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from . import emails, events, mailer
from .config import settings
from .db import session_scope
from .models import (
    GameweekResult,
    InboxItem,
    Invite,
    Plan,
    PlayerScores,
    Run,
    Squad,
    User,
    utcnow,
)

log = logging.getLogger("nerdball.lifecycle")


def _aware(moment: dt.datetime | None) -> dt.datetime | None:
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=dt.timezone.utc)
    return moment


def expire_invites(session: Session) -> int:
    """Drop invitations nobody used in time, and say so."""
    if settings.invite_ttl_hours <= 0:
        return 0

    now = utcnow()
    expired = [
        invite
        for invite in session.scalars(
            select(Invite).where(Invite.expires_at.is_not(None))
        )
        if (_aware(invite.expires_at) or now) < now
    ]

    for invite in expired:
        # Only tell them if they never got in. Somebody who signed in and
        # then had their invite tidied up doesn't need an email saying
        # their invitation expired.
        signed_in = session.scalar(
            select(User).where(User.email == invite.email)
        )
        if signed_in is None:
            mailer.send_template(
                emails.invite_expired(
                    invite.email, settings.invite_ttl_hours
                ),
                to=invite.email,
            )
            events.emit("invite_expired", email=invite.email)

        session.delete(invite)

    if expired:
        session.commit()
    return len(expired)


def _protected(user: User) -> bool:
    """Accounts the sweeps never touch.

    ALLOWED_EMAILS is the deployment's own list: somebody on it is there
    because you put them there, not because a seat was going spare.
    """
    return user.is_admin or user.email in settings.allowed_emails


def make_dormant(session: Session) -> int:
    """Free the places of managers who stopped turning up.

    Their place goes back to the queue; their data does not go anywhere.
    A dormant account keeps its squads, runs and settings, stops counting
    against the seat cap, and comes back intact the moment they sign in
    again. Nothing is deleted here at all — that happens far later, in
    purge_season_data and delete_long_gone.
    """
    days = settings.inactive_days
    if days <= 0:
        return 0

    cutoff = utcnow() - dt.timedelta(days=days)
    candidates = session.scalars(
        select(User).where(
            User.is_guest.is_(False),
            User.is_admin.is_(False),
            User.dormant_at.is_(None),
            User.last_seen_at < cutoff,
        )
    ).all()

    count = 0
    for user in candidates:
        if _protected(user):
            continue

        user.dormant_at = utcnow()
        count += 1

        mailer.send_template(
            emails.gone_dormant(user.email, days, settings.current_season),
            to=user.email,
        )
        mailer.send_template(emails.admin_dormant(user.email, days))
        events.emit("member_dormant", email=user.email, days=days)

    if count:
        session.commit()
    return count


def purge_season_data(session: Session) -> int:
    """Clear a dormant manager's data once its season is over.

    A squad from a finished season can't be picked up where it was left,
    so keeping it serves nobody. The FPL team id survives, because that
    is the one thing still true next August.
    """
    dormant = session.scalars(
        select(User).where(
            User.is_guest.is_(False),
            User.dormant_at.is_not(None),
            User.data_purged_at.is_(None),
        )
    ).all()

    season = settings.current_season
    cleared = 0
    for user in dormant:
        # Anything from the season now being played is still live data:
        # they could come back tomorrow and carry on.
        leftover = session.scalar(
            select(func.count())
            .select_from(Squad)
            .where(Squad.user_id == user.id, Squad.season == season)
        )
        if leftover:
            continue

        for model in (Squad, Run, Plan, PlayerScores, GameweekResult):
            session.execute(delete(model).where(model.user_id == user.id))
        _remove_workspace(user.id)

        user.data_purged_at = utcnow()
        cleared += 1
        events.emit("season_data_purged", user=user.id)

    if cleared:
        session.commit()
    return cleared


def delete_long_gone(session: Session) -> int:
    """Finally remove accounts nobody has touched in years."""
    months = settings.purge_after_months
    if months <= 0:
        return 0

    cutoff = utcnow() - dt.timedelta(days=months * 30)
    candidates = session.scalars(
        select(User).where(
            User.is_guest.is_(False),
            User.is_admin.is_(False),
            User.last_seen_at < cutoff,
        )
    ).all()

    removed = 0
    for user in candidates:
        if _protected(user):
            continue
        email = user.email
        _delete_user(session, user)
        removed += 1
        events.emit("member_deleted", email=email, months=months)

    if removed:
        session.commit()
    return removed


def reactivate(session: Session, user: User) -> None:
    """Hand a dormant manager their place and their data back."""
    user.dormant_at = None
    user.last_seen_at = utcnow()
    session.commit()
    events.emit("member_reactivated", user=user.id)


def _delete_user(session: Session, user: User) -> None:
    """Remove an account and everything hanging off it.

    Explicit rather than relying on the foreign keys: ON DELETE CASCADE
    does nothing under SQLite without the pragma, and Plan's relationship
    has no cascade of its own.
    """
    for model in (Run, Plan, PlayerScores, GameweekResult, Squad):
        session.execute(delete(model).where(model.user_id == user.id))
    session.execute(delete(Invite).where(Invite.email == user.email))
    session.delete(user)

    _remove_workspace(user.id)


def _remove_workspace(user_id: int) -> None:
    try:
        import shutil

        from .engine.workspace import user_root

        path = user_root(user_id)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        log.warning("Left %s's workspace behind", user_id, exc_info=True)


def queue_position(session: Session, email: str) -> int | None:
    """Where an address stands in the waiting list, oldest first.

    None when there is no queue to be in, which is the answer whenever a
    place is free.
    """
    from .auth import seat_count

    if seat_count(session) < settings.max_users:
        return None

    waiting = session.scalars(
        select(InboxItem)
        .where(
            InboxItem.kind == "access_request",
            InboxItem.status == "new",
        )
        .order_by(InboxItem.created_at)
    ).all()

    for index, item in enumerate(waiting, start=1):
        if item.email == email:
            return index
    # Not in the list yet: they are about to join the end of it.
    return len(waiting) + 1


def season_looks_stale() -> bool:
    """Whether CURRENT_SEASON appears to be last season's.

    The dormant-data purge and every squad key off this string, and
    nothing goes wrong visibly when it's out of date — the app just
    quietly keeps filing this year's work under last year. A new season
    starts in August, so from September a season whose first year is
    behind us is almost certainly forgotten.
    """
    season = settings.current_season
    try:
        start_year = int(season.split("-")[0])
    except (ValueError, IndexError):
        return False

    now = utcnow()
    expected = now.year if now.month >= 8 else now.year - 1
    return start_year < expected


def sweep() -> dict[str, int]:
    """One pass of every rule, in the order they escalate."""
    with session_scope() as session:
        expired = expire_invites(session)
        dormant = make_dormant(session)
        purged = purge_season_data(session)
        deleted = delete_long_gone(session)

    result = {
        "expired": expired,
        "dormant": dormant,
        "purged": purged,
        "deleted": deleted,
    }
    if any(result.values()):
        events.emit("lifecycle_sweep", **result)

    if season_looks_stale():
        log.warning(
            "CURRENT_SEASON is %s, which looks like last season. Squads "
            "are being filed under it and dormant data won't be cleared "
            "until it's rolled over.",
            settings.current_season,
        )
        events.emit("season_stale", season=settings.current_season)

    return result


def start_scheduler() -> None:
    """Run the sweep hourly.

    Hourly rather than continuously because both windows are measured in
    days and hours: being an hour late to free a place has never mattered
    to anybody, and an idle timer costs nothing.
    """
    if (
        settings.invite_ttl_hours <= 0
        and settings.inactive_days <= 0
        and settings.purge_after_months <= 0
    ):
        return

    def loop() -> None:
        # A moment's grace at boot so a deploy isn't competing with the
        # engine warming up.
        time.sleep(90)
        while True:
            try:
                sweep()
            except Exception:
                log.warning("Lifecycle sweep failed", exc_info=True)
            time.sleep(3600)

    threading.Thread(target=loop, name="lifecycle", daemon=True).start()
