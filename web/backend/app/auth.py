"""Google sign-in, the access allowlist, and the current-user dependency."""

from __future__ import annotations

import datetime as dt
import secrets
import threading

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


# ── Admin unlock ─────────────────────────────────────────────────────────

UNLOCK_KEY = "admin_unlocked_at"

# Failed attempts, per signed-in user. Six people share this deployment, so an
# in-memory counter is enough; there's no cluster to share state across.
_attempts: dict[int, list] = {}
_attempts_lock = threading.Lock()

MAX_ATTEMPTS = 5
LOCKOUT = dt.timedelta(minutes=15)


def _lockout_remaining(user_id: int) -> dt.timedelta | None:
    with _attempts_lock:
        record = _attempts.get(user_id)
        if not record:
            return None
        failures, last = record
        if failures < MAX_ATTEMPTS:
            return None
        remaining = (last + LOCKOUT) - utcnow()
        if remaining.total_seconds() <= 0:
            _attempts.pop(user_id, None)
            return None
        return remaining


def register_failure(user_id: int) -> None:
    with _attempts_lock:
        failures, _ = _attempts.get(user_id, (0, utcnow()))
        _attempts[user_id] = (failures + 1, utcnow())


def clear_failures(user_id: int) -> None:
    with _attempts_lock:
        _attempts.pop(user_id, None)


def check_admin_password(user_id: int, candidate: str) -> None:
    """Raises with a useful message unless the password is right."""
    if not settings.admin_configured:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The admin area is switched off. Set ADMIN_PASSWORD on the "
            "service and redeploy to enable it.",
        )

    remaining = _lockout_remaining(user_id)
    if remaining is not None:
        minutes = max(1, int(remaining.total_seconds() // 60))
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Too many failed attempts. Try again in {minutes} minute"
            f"{'s' if minutes != 1 else ''}.",
        )

    # Constant-time, so a wrong password can't be narrowed down by timing.
    if not secrets.compare_digest(candidate, settings.admin_password):
        register_failure(user_id)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong password.")

    clear_failures(user_id)


def admin_unlocked(request: Request) -> bool:
    """Whether this session has entered the password recently enough."""
    raw = request.session.get(UNLOCK_KEY)
    if not raw:
        return False
    try:
        unlocked_at = dt.datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return False
    age = utcnow() - unlocked_at
    return age < dt.timedelta(minutes=settings.admin_session_minutes)


def current_admin(
    request: Request, user: User = Depends(current_user)
) -> User:
    """Admin routes need a signed-in user who has entered the password."""
    if not settings.admin_configured:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The admin area is switched off. Set ADMIN_PASSWORD to enable it.",
        )
    if not admin_unlocked(request):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "Enter the admin password to continue."
        )
    return user
