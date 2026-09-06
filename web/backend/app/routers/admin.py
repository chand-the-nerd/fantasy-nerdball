"""Owner-only routes for managing who can sign in."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import current_admin, seat_count
from ..config import settings
from ..db import get_session
from ..models import Invite, User
from ..schemas import InviteIn

router = APIRouter(prefix="/api/admin", tags=["admin"])

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@router.get("/members")
def members(
    admin: User = Depends(current_admin), session: Session = Depends(get_session)
) -> dict:
    users = session.scalars(select(User).order_by(User.created_at)).all()
    invites = session.scalars(select(Invite).order_by(Invite.created_at)).all()
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
                "last_seen_at": u.last_seen_at,
            }
            for u in users
        ],
        "env_allowlist": settings.allowed_emails,
        "invites": [{"id": i.email and i.id, "email": i.email} for i in invites],
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

    if seat_count(session) >= settings.max_users:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"All {settings.max_users} seats are taken. Remove a manager first.",
        )

    existing = session.scalar(select(Invite).where(Invite.email == email))
    if existing is None:
        session.add(Invite(email=email, invited_by=admin.email))
        session.commit()
    return {"email": email}


@router.delete("/invites/{invite_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response, response_model=None)
def remove_invite(
    invite_id: int,
    admin: User = Depends(current_admin),
    session: Session = Depends(get_session),
) -> None:
    invite = session.get(Invite, invite_id)
    if invite is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such invite")
    session.delete(invite)
    session.commit()


@router.delete("/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response, response_model=None)
def remove_member(
    user_id: int,
    admin: User = Depends(current_admin),
    session: Session = Depends(get_session),
) -> None:
    """Frees a seat. Their squads and history go with them."""
    if user_id == admin.id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "You can't remove your own account."
        )
    member = session.get(User, user_id)
    if member is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such manager")
    session.delete(member)
    session.commit()
