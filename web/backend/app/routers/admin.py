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

from .. import emails, events, lifecycle, mailer, metrics
from ..auth import current_admin, current_user, seat_count
from ..config import settings
from ..db import get_session
from ..models import Invite, Run, Squad, User, utcnow
from ..schemas import InviteIn, RestoreSquadIn

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
    # Dormant managers are listed too. The whole point of not deleting
    # them is being able to see them and give them their place back.

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
        if (
            cutoff_days > 0
            and idle_days is not None
            and not protected
            and user.dormant_at is None
        ):
            removal_in = max(0, cutoff_days - idle_days)

        runs = session.scalar(
            select(func.count())
            .select_from(Run)
            .where(Run.user_id == user.id)
        )

        squads = session.scalar(
            select(func.count())
            .select_from(Squad)
            .where(Squad.user_id == user.id)
        )
        latest = session.scalar(
            select(Squad)
            .where(Squad.user_id == user.id)
            .order_by(Squad.season.desc(), Squad.gameweek.desc())
            .limit(1)
        )

        if user.dormant_at is not None:
            status_label = "purged" if user.data_purged_at else "dormant"
        else:
            status_label = "active"

        rows.append(
            {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "is_admin": user.is_admin,
                "status": status_label,
                "dormant_at": user.dormant_at.isoformat()
                if user.dormant_at
                else None,
                "squads_kept": int(squads or 0),
                "latest_gameweek": latest.gameweek if latest else None,
                "latest_season": latest.season if latest else None,
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
        "seats_used": len(
            [r for r in rows if r["status"] == "active"]
        ),
        "seats_total": settings.max_users,
        "inactive_days": cutoff_days,
        "invite_ttl_hours": settings.invite_ttl_hours,
        "purge_after_months": settings.purge_after_months,
        "season": settings.current_season,
    }


@router.get("/users/{user_id}/squads")
def admin_user_squads(
    user_id: int,
    admin: User = Depends(current_admin),
    session: Session = Depends(get_session),
) -> dict:
    """Every gameweek this manager has saved, newest first."""
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such manager")

    squads = session.scalars(
        select(Squad)
        .where(Squad.user_id == user_id)
        .order_by(Squad.season.desc(), Squad.gameweek.desc())
    ).all()

    return {
        "user": {"id": user.id, "email": user.email, "name": user.name},
        "squads": [
            {
                "season": squad.season,
                "gameweek": squad.gameweek,
                "formation": squad.formation,
                "projected_points": squad.projected_points,
                "squad_value": squad.squad_value,
                "transfers_made": squad.transfers_made,
                "chip": squad.chip,
                "players": len(squad.engine_rows or []),
            }
            for squad in squads
        ],
    }


@router.post("/users/{user_id}/reactivate")
def admin_reactivate(
    user_id: int,
    admin: User = Depends(current_admin),
    session: Session = Depends(get_session),
) -> dict:
    """Give a dormant manager their place, and their data, back."""
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such manager")
    if user.dormant_at is None:
        return {"ok": True, "already_active": True}

    if seat_count(session) >= settings.max_users:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"All {settings.max_users} places are taken. Free one first.",
        )

    lifecycle.reactivate(session, user)
    mailer.send_template(
        emails.access_approved(
            user.email, settings.invite_ttl_hours, settings.inactive_days
        ),
        to=user.email,
        reply_to=settings.mail_to,
    )
    return {"ok": True, "email": user.email}


@router.post("/users/{user_id}/restore-squad")
def admin_restore_squad(
    user_id: int,
    payload: RestoreSquadIn,
    admin: User = Depends(current_admin),
    session: Session = Depends(get_session),
) -> dict:
    """Put an old squad back as the manager's current one.

    Copies forward rather than rewinding: the chosen gameweek's squad is
    written into the target gameweek, and every row in between is left
    exactly as it was. An admin fixing one week's mistake should not be
    able to erase a season by accident.
    """
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such manager")

    source = session.scalar(
        select(Squad).where(
            Squad.user_id == user_id,
            Squad.season == payload.season,
            Squad.gameweek == payload.gameweek,
        )
    )
    if source is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Nothing saved for gameweek {payload.gameweek}.",
        )

    target_gw = payload.into_gameweek or payload.gameweek
    target = session.scalar(
        select(Squad).where(
            Squad.user_id == user_id,
            Squad.season == payload.season,
            Squad.gameweek == target_gw,
        )
    )
    if target is None:
        target = Squad(
            user_id=user_id, season=payload.season, gameweek=target_gw
        )
        session.add(target)

    for field in (
        "formation",
        "projected_points",
        "squad_value",
        "bank",
        "transfers_made",
        "penalty_points",
        "chip",
        "payload",
        "engine_rows",
        "active_option",
    ):
        setattr(target, field, getattr(source, field))

    session.commit()
    events.emit(
        "squad_restored",
        by=admin.id,
        user=user_id,
        source_gameweek=payload.gameweek,
        into_gameweek=target_gw,
    )
    return {
        "ok": True,
        "restored_from": payload.gameweek,
        "into": target_gw,
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
