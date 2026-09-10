"""Google sign-in, the access allowlist, and the current-user dependency."""

from __future__ import annotations

import datetime as dt

from authlib.integrations.starlette_client import OAuth
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import events, guest, metrics
from .config import settings
from .db import get_session
from .models import Invite, User, UserSettings, utcnow

GOOGLE_METADATA = "https://accounts.google.com/.well-known/openid-configuration"

oauth = OAuth()
if settings.google_configured:
    oauth.register(
        name="google",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        server_metadata_url=GOOGLE_METADATA,
        client_kwargs={"scope": "openid email profile"},
    )


def is_allowed(email: str, session: Session) -> bool:
    """An address gets in if it's in ALLOWED_EMAILS or holds a live invite.

    Live matters: an invitation that has run out of time is no more use
    than none at all, and the sweep that deletes expired ones runs on a
    timer rather than at the moment they lapse.
    """
    email = email.lower().strip()
    if email in settings.allowed_emails:
        return True

    invited = session.scalar(select(Invite).where(Invite.email == email))
    if invited is None:
        return False
    return not _expired(invited)


def _expired(invite: Invite) -> bool:
    if invite.expires_at is None:
        return False
    deadline = invite.expires_at
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=dt.timezone.utc)
    return utcnow() > deadline


def seat_count(session: Session) -> int:
    """How many of the league's seats are taken.

    Guests are deliberately not counted. They are throwaway accounts on
    the same table, and a busy afternoon of them would otherwise lock
    real managers out of a deployment that has plenty of room.
    """
    return int(
        session.scalar(
            select(func.count())
            .select_from(User)
            .where(User.is_guest.is_(False))
        )
        or 0
    )


def upsert_user(session: Session, *, email: str, sub: str | None, name: str, picture: str) -> User:
    """Find or create the signed-in manager, enforcing the seat cap."""
    email = email.lower().strip()
    user = session.scalar(select(User).where(User.email == email))

    if user is None:
        if seat_count(session) >= settings.max_users:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"This league is full ({settings.max_users} managers). "
                "Ask the owner to remove someone first.",
            )
        user = User(
            email=email,
            google_sub=sub,
            name=name or email.split("@")[0],
            avatar_url=picture or "",
            is_admin=settings.is_admin_email(email) or seat_count(session) == 0,
        )
        user.settings = UserSettings()
        session.add(user)
    else:
        user.google_sub = sub or user.google_sub
        user.name = name or user.name
        user.avatar_url = picture or user.avatar_url
        # Re-applied every sign-in: an address added to ALLOWED_EMAILS or
        # ADMIN_EMAILS after the account already existed still gets the page.
        if settings.is_admin_email(email):
            user.is_admin = True
        if user.settings is None:
            user.settings = UserSettings()

    user.last_seen_at = utcnow()
    session.commit()
    session.refresh(user)
    return user


def current_user(
    request: Request, session: Session = Depends(get_session)
) -> User:
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sign in to continue")
    user = session.get(User, user_id)
    if user is None:
        # A swept-up guest lands here: the cookie outlives the row.
        request.session.clear()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sign in to continue")
    guest.touch(session, user)
    metrics.set_actor(user.id, user.is_guest)
    events.note_session(user)
    return user


# ── Admin ────────────────────────────────────────────────────────────────


def current_admin(user: User = Depends(current_user)) -> User:
    """Admin routes belong to admin accounts.

    There is no separate password: you are an admin if you signed in with the
    owner address (the first entry in ALLOWED_EMAILS) or one listed in
    ADMIN_EMAILS. The flag is re-applied on every sign-in by upsert_user, so
    changing the variables and signing in again is enough.
    """
    if not user.is_admin:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"The admin page belongs to the owner's account. You're signed in "
            f"as {user.email}.",
        )
    return user
