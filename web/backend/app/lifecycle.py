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

from sqlalchemy import select
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


def remove_inactive(session: Session) -> int:
    """Free the places of managers who stopped turning up."""
    days = settings.inactive_days
    if days <= 0:
        return 0

    cutoff = utcnow() - dt.timedelta(days=days)
    candidates = session.scalars(
        select(User).where(
            User.is_guest.is_(False),
            User.is_admin.is_(False),
            User.last_seen_at < cutoff,
        )
    ).all()

    removed = 0
    for user in candidates:
        # ALLOWED_EMAILS is the deployment's own list. Someone on it is
        # there because you put them there, not because a seat was going
        # spare, so the sweep leaves them alone.
        if user.email in settings.allowed_emails:
            continue

        email = user.email
        _delete_user(session, user)
        removed += 1

        mailer.send_template(
            emails.removed_for_inactivity(email, days), to=email
        )
        mailer.send_template(emails.admin_removed(email, days))
        events.emit("member_removed_inactive", email=email, days=days)

    if removed:
        session.commit()
    return removed


def _delete_user(session: Session, user: User) -> None:
    """Remove an account and everything hanging off it.

    Explicit rather than relying on the foreign keys: ON DELETE CASCADE
    does nothing under SQLite without the pragma, and Plan's relationship
    has no cascade of its own.
    """
    from sqlalchemy import delete

    for model in (Run, Plan, PlayerScores, GameweekResult, Squad):
        session.execute(delete(model).where(model.user_id == user.id))
    session.execute(delete(Invite).where(Invite.email == user.email))
    session.delete(user)

    try:
        import shutil

        from .engine.workspace import user_root

        path = user_root(user.id)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        log.warning("Left %s's workspace behind", user.id, exc_info=True)


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


def sweep() -> dict[str, int]:
    """One pass of both rules."""
    with session_scope() as session:
        expired = expire_invites(session)
        removed = remove_inactive(session)
    if expired or removed:
        events.emit("lifecycle_sweep", expired=expired, removed=removed)
    return {"expired": expired, "removed": removed}


def start_scheduler() -> None:
    """Run the sweep hourly.

    Hourly rather than continuously because both windows are measured in
    days and hours: being an hour late to free a place has never mattered
    to anybody, and an idle timer costs nothing.
    """
    if settings.invite_ttl_hours <= 0 and settings.inactive_days <= 0:
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
