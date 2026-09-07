"""Google sign-in, the access allowlist, and the current-user dependency."""

from __future__ import annotations

from authlib.integrations.starlette_client import OAuth
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

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
    """An address gets in if it is in ALLOWED_EMAILS or has been invited."""
    email = email.lower().strip()
    if email in settings.allowed_emails:
        return True
    invited = session.scalar(select(Invite).where(Invite.email == email))
    return invited is not None


def seat_count(session: Session) -> int:
    return int(session.scalar(select(func.count()).select_from(User)) or 0)


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
        owner_email = settings.allowed_emails[0] if settings.allowed_emails else ""
        user = User(
            email=email,
            google_sub=sub,
            name=name or email.split("@")[0],
            avatar_url=picture or "",
            is_admin=(email == owner_email) or seat_count(session) == 0,
        )
        user.settings = UserSettings()
        session.add(user)
    else:
        user.google_sub = sub or user.google_sub
        user.name = name or user.name
        user.avatar_url = picture or user.avatar_url
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
        request.session.clear()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Sign in to continue")
    return user


def current_admin(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Owner access only")
    return user
