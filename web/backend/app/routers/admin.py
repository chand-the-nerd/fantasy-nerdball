"""The admin area: who can sign in, and who currently has a seat.

Gated by which Google account you signed in with. The owner address (the
first entry in ALLOWED_EMAILS) and anything in ADMIN_EMAILS get in; everyone
else gets a 403. There is no separate password.
"""

from __future__ import annotations

import datetime as dt
import re

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import events, metrics
from ..auth import current_admin, current_user, seat_count
from ..config import settings
from ..db import get_session
from ..models import Invite, Run, User, utcnow
from ..schemas import InviteIn

router = APIRouter(prefix="/api/admin", tags=["admin"])

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@router.get("/status")
def admin_status(user: User = Depends(current_user)) -> dict:
    """Lets the footer know whether to offer the admin page at all."""
    return {
        "admin": user.is_admin,
        "email": user.email,
        "owner_email": settings.owner_email,
    }


@router.get("/members")
def members(
    admin: User = Depends(current_admin), session: Session = Depends(get_session)
) -> dict:
    users = session.scalars(
        select(User)
        .where(User.is_guest.is_(False))
        .order_by(User.created_at)
    ).all()
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


@router.get("/users")
def admin_users(
    admin: User = Depends(current_admin),
    session: Session = Depends(get_session),
) -> dict:
    """Every manager, their linked FPL side, and when they last turned up.

    Ordered by last activity so the people about to lose their place for
    inactivity are at the bottom, where you can see them coming.
    """
    users = session.scalars(
        select(User)
        .where(User.is_guest.is_(False))
        .order_by(User.last_seen_at.desc())
    ).all()

    cutoff_days = settings.inactive_days
    now = utcnow()

    rows = []
    for user in users:
        seen = user.last_seen_at
        if seen is not None and seen.tzinfo is None:
            seen = seen.replace(tzinfo=dt.timezone.utc)
        idle_days = (now - seen).days if seen else None

        protected = user.is_admin or user.email in settings.allowed_emails
        removal_in = None
        if cutoff_days > 0 and idle_days is not None and not protected:
            removal_in = max(0, cutoff_days - idle_days)

        runs = session.scalar(
            select(func.count())
            .select_from(Run)
            .where(Run.user_id == user.id)
        )

        rows.append(
            {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "is_admin": user.is_admin,
                "fpl_entry_id": user.fpl_entry_id,
                "created_at": user.created_at.isoformat()
                if user.created_at
                else "",
                "last_seen_at": seen.isoformat() if seen else "",
                "idle_days": idle_days,
                # None means they are never removed for inactivity.
                "removal_in_days": removal_in,
                "runs": int(runs or 0),
            }
        )

    pending = session.scalars(
        select(Invite).order_by(Invite.created_at.desc())
    ).all()
    invites = []
    for invite in pending:
        expires = invite.expires_at
        if expires is not None and expires.tzinfo is None:
            expires = expires.replace(tzinfo=dt.timezone.utc)
        hours_left = None
        if expires is not None:
            hours_left = round((expires - now).total_seconds() / 3600, 1)
        invites.append(
            {
                "email": invite.email,
                "invited_by": invite.invited_by,
                "expires_at": expires.isoformat() if expires else None,
                "hours_left": hours_left,
                "signed_in": session.scalar(
                    select(User).where(User.email == invite.email)
                )
                is not None,
            }
        )

    return {
        "users": rows,
        "invites": invites,
        "seats_used": len(rows),
        "seats_total": settings.max_users,
        "inactive_days": cutoff_days,
        "invite_ttl_hours": settings.invite_ttl_hours,
    }


@router.get("/metrics")
def admin_metrics(
    window: str = "24h",
    admin: User = Depends(current_admin),
    session: Session = Depends(get_session),
) -> dict:
    """Usage over one time window, for the dashboard in the admin pane.

    Admin-only, and the only place stored addresses are ever shown.
    """
    if window not in metrics.WINDOWS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Pick one of: {', '.join(metrics.WINDOWS)}.",
        )
    return metrics.summary(session, window)


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
    events.emit("invite_added", by=admin.id, email=email)
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
    events.emit("member_removed", by=admin.id, removed=user_id)
