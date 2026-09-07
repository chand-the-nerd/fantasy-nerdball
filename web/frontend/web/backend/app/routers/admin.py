"""The admin area: who can sign in, and who currently has a seat.

Gated by ADMIN_PASSWORD rather than by which account you signed in with, so
it can be handed to whoever is looking after the deployment without changing
anyone's account.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import (
    UNLOCK_KEY,
    admin_unlocked,
    check_admin_password,
    current_admin,
    current_user,
    seat_count,
)
from ..config import settings
from ..db import get_session
from ..models import Invite, User, utcnow
from ..schemas import InviteIn

router = APIRouter(prefix="/api/admin", tags=["admin"])

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class UnlockIn(BaseModel):
    password: str


@router.get("/status")
def admin_status(request: Request, user: User = Depends(current_user)) -> dict:
    """Lets the footer button know whether to show a prompt or the page."""
    return {
        "configured": settings.admin_configured,
        "unlocked": admin_unlocked(request),
        "session_minutes": settings.admin_session_minutes,
    }


@router.post("/unlock")
def unlock(
    payload: UnlockIn, request: Request, user: User = Depends(current_user)
) -> dict:
    check_admin_password(user.id, payload.password)
    request.session[UNLOCK_KEY] = utcnow().isoformat()
    return {"unlocked": True, "session_minutes": settings.admin_session_minutes}


@router.post("/lock")
def lock(request: Request, user: User = Depends(current_user)) -> dict:
    request.session.pop(UNLOCK_KEY, None)
    return {"unlocked": False}


@router.get("/members")
def members(
    admin: User = Depends(current_admin), session: Session = Depends(get_session)
) -> dict:
    users = session.scalars(select(User).order_by(User.created_at)).all()
    invites = session.scalars(select(Invite).order_by(Invite.created_at)).all()

    registered = {u.email for u in users}

    return {
        "seats_used": len(users),
        "seats_total": settings.max_users,
        "members": [
            {
                "id": u.id,
                "email": u.email,
                "name": u.name,
                "avatar_url": u.avatar_url,
                "is_admin": u.is_admin,
                "is_you": u.id == admin.id,
                "last_seen_at": u.last_seen_at,
            }
            for u in users
        ],
        # From ALLOWED_EMAILS. Set on the service, so it can't be edited here —
        # the UI shows these as fixed rather than pretending they're removable.
        "env_allowlist": [
            {"email": email, "registered": email in registered}
            for email in settings.allowed_emails
        ],
        "invites": [
            {
                "id": i.id,
                "email": i.email,
                "invited_by": i.invited_by,
                "registered": i.email in registered,
            }
            for i in invites
        ],
    }


@router.post("/invites", status_code=status.HTTP_201_CREATED)
def add_invite(
    payload: InviteIn,
    admin: User = Depends(current_admin),
    session: Session = Depends(get_session),
) -> dict:
    email = payload.email.strip().lower()
    if not EMAIL_PATTERN.match(email):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "That isn't an email address.")

    if email in settings.allowed_emails:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{email} is already allowed in via ALLOWED_EMAILS.",
        )

    if seat_count(session) >= settings.max_users:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"All {settings.max_users} seats are taken. Remove a manager first.",
        )

    existing = session.scalar(select(Invite).where(Invite.email == email))
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"{email} is already invited.")

    session.add(Invite(email=email, invited_by=admin.email))
    session.commit()
    return {"email": email}


@router.delete(
    "/invites/{invite_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
)
def remove_invite(
    invite_id: int,
    admin: User = Depends(current_admin),
    session: Session = Depends(get_session),
) -> None:
    """Stops a new account being made from this address.

    An account already created from it keeps working — remove the manager as
    well if the intention is to revoke access.
    """
    invite = session.get(Invite, invite_id)
    if invite is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such invite")
    session.delete(invite)
    session.commit()


@router.delete(
    "/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
)
def remove_member(
    user_id: int,
    admin: User = Depends(current_admin),
    session: Session = Depends(get_session),
) -> None:
    """Frees a seat. Their squads, settings and results go with them."""
    if user_id == admin.id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "You can't remove your own account."
        )
    member = session.get(User, user_id)
    if member is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such manager")

    # Deleting the account does not revoke access on its own. If the address is
    # still on the allowlist, the next sign-in simply creates it again, so the
    # invite has to go too. The members list flags this either way.
    session.delete(member)
    session.commit()
