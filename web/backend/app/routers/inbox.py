"""Letting people ask to be let in, and tell you when something's wrong.

Both end up in the same table and the same inbox in the admin pane, and
both send you an email, because a message you have to remember to go and
look for is one you find out about a fortnight late.

The access-request endpoint is the only unauthenticated write in the app,
so it is deliberately dull: an address, an optional sentence, a cap on
how often one caller can use it, and no way to tell from the response
whether an address is already a member.
"""

from __future__ import annotations

import re
import threading
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import events, mailer, metrics
from ..auth import current_admin, current_user, is_allowed, seat_count
from ..config import settings
from ..db import get_session
from ..models import Invite, InboxItem, User, utcnow
from ..schemas import AccessRequestIn, FeedbackIn

router = APIRouter(tags=["inbox"])

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

FEEDBACK_KINDS = {
    "broken": "Something is broken",
    "feature": "Feature request",
    "general": "General feedback",
}

# One caller, three requests an hour. Enough for somebody who mistypes
# their address twice, not enough to be worth automating.
RATE_LIMIT = 3
RATE_WINDOW = 3600
_recent: dict[str, list[float]] = {}
_recent_lock = threading.Lock()


def _rate_limited(key: str) -> bool:
    now = time.monotonic()
    with _recent_lock:
        seen = [t for t in _recent.get(key, []) if now - t < RATE_WINDOW]
        if len(seen) >= RATE_LIMIT:
            _recent[key] = seen
            return True
        seen.append(now)
        _recent[key] = seen

        if len(_recent) > 2000:
            for old in [
                k
                for k, times in _recent.items()
                if not times or now - times[-1] > RATE_WINDOW
            ]:
                _recent.pop(old, None)
    return False


@router.post("/api/access-request", status_code=status.HTTP_201_CREATED)
def request_access(
    payload: AccessRequestIn,
    request: Request,
    session: Session = Depends(get_session),
) -> dict:
    """Ask the admin for a seat.

    Three outcomes, and the caller is told which: they are already
    approved and should just sign in, they have already asked, or the
    request has gone through.
    """
    email = payload.email.strip().lower()
    if not EMAIL_PATTERN.match(email):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "That doesn't look like an email address.",
        )

    if _rate_limited(metrics.client_ip_from(request) or email):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "That's a few requests in a short time. Try again later.",
        )

    note = (payload.note or "").strip()[:1000]

    # Already welcome: on the allowlist, holding an invite, or signed in
    # before now. Telling them so is the useful answer — asking the admin
    # for something you already have, and hearing nothing back, is worse
    # than the small amount this discloses about who is a member.
    if is_allowed(email, session):
        return {
            "ok": True,
            "status": "already_approved",
            "message": (
                "That address is already approved. Sign in with Google "
                "above — no need to ask."
            ),
        }

    pending = session.scalar(
        select(InboxItem).where(
            InboxItem.kind == "access_request",
            InboxItem.email == email,
            InboxItem.status == "new",
        )
    )
    if pending is not None:
        return {
            "ok": True,
            "status": "pending",
            "message": (
                "You've already asked and the admin has it. You'll be "
                "able to sign in once they add your address."
            ),
        }

    session.add(
        InboxItem(
            kind="access_request",
            email=email,
            name=(payload.name or "").strip()[:120],
            body=note,
        )
    )
    session.commit()

    events.emit("access_requested", email=email)
    seats = seat_count(session)
    mailer.send(
        subject=f"Access request — {email}",
        body=(
            f"{email} has asked for access to Fantasy Nerdball.\n\n"
            f"{note or 'No message.'}\n\n"
            f"Seats: {seats} of {settings.max_users} taken.\n"
            "Approve or dismiss it in the admin pane."
        ),
        reply_to=email,
    )

    return {
        "ok": True,
        "status": "sent",
        "message": (
            "Request sent. You'll be able to sign in with Google once "
            "the admin adds your address."
        ),
    }


@router.post("/api/feedback", status_code=status.HTTP_201_CREATED)
def send_feedback(
    payload: FeedbackIn,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> dict:
    """Tell the admin something. Open to guests as well as managers."""
    if payload.kind not in FEEDBACK_KINDS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Pick one of: {', '.join(FEEDBACK_KINDS)}.",
        )

    body = payload.body.strip()[:4000]
    if not body:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Say a little about it first."
        )

    if _rate_limited(f"user:{user.id}"):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "That's a few messages in a short time. Try again later.",
        )

    session.add(
        InboxItem(
            kind=payload.kind,
            # A guest has a throwaway address, which is worse than none:
            # replying to it reaches nobody.
            email="" if user.is_guest else user.email,
            name="Guest" if user.is_guest else user.name,
            body=body,
            user_id=user.id,
            from_guest=user.is_guest,
        )
    )
    session.commit()

    events.emit("feedback_sent", kind=payload.kind, **events.actor(user))
    who = "A guest" if user.is_guest else f"{user.name} ({user.email})"
    mailer.send(
        subject=f"{FEEDBACK_KINDS[payload.kind]} — Fantasy Nerdball",
        body=f"From: {who}\n\n{body}",
        reply_to="" if user.is_guest else user.email,
    )

    return {"ok": True, "message": "Thanks — that's gone through."}


# ── The admin side ───────────────────────────────────────────────────────


def _serialise(item: InboxItem) -> dict:
    return {
        "id": item.id,
        "kind": item.kind,
        "title": (
            f"Access request - {item.email}"
            if item.kind == "access_request"
            else FEEDBACK_KINDS.get(item.kind, item.kind)
        ),
        "email": item.email,
        "name": item.name,
        "body": item.body,
        "from_guest": item.from_guest,
        "status": item.status,
        "created_at": item.created_at.isoformat() if item.created_at else "",
        "handled_by": item.handled_by,
    }


@router.get("/api/admin/inbox")
def read_inbox(
    include_done: bool = False,
    admin: User = Depends(current_admin),
    session: Session = Depends(get_session),
) -> dict:
    query = select(InboxItem).order_by(InboxItem.created_at.desc())
    if not include_done:
        query = query.where(InboxItem.status == "new")

    items = session.scalars(query.limit(200)).all()
    unread = session.scalar(
        select(InboxItem).where(InboxItem.status == "new").limit(1)
    )

    return {
        "items": [_serialise(item) for item in items],
        "unread": len([i for i in items if i.status == "new"]),
        "has_unread": unread is not None,
        "seats_used": seat_count(session),
        "seats_total": settings.max_users,
        "email_configured": mailer.configured(),
        "email": mailer.status(),
    }


@router.post("/api/admin/test-email")
def test_email(admin: User = Depends(current_admin)) -> dict:
    """Send yourself one email now, and say exactly what happened.

    Sent on this thread rather than in the background, because the point
    is the answer, not the message.
    """
    error = mailer.send_now(
        subject="Fantasy Nerdball — test",
        body=(
            "If you're reading this, access requests and feedback will "
            "reach you too."
        ),
    )
    if error:
        return {"ok": False, "detail": error}
    return {
        "ok": True,
        "detail": f"Sent to {settings.mail_to}. Give it a minute.",
    }


@router.post("/api/admin/inbox/{item_id}/approve")
def approve_request(
    item_id: int,
    admin: User = Depends(current_admin),
    session: Session = Depends(get_session),
) -> dict:
    """Let them in: adds the address to the allowlist and closes the item."""
    item = session.get(InboxItem, item_id)
    if item is None or item.kind != "access_request":
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "No such access request"
        )

    if seat_count(session) >= settings.max_users:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"All {settings.max_users} seats are taken. Remove a manager "
            "first, or raise MAX_USERS.",
        )

    existing = session.scalar(
        select(Invite).where(Invite.email == item.email)
    )
    if existing is None:
        session.add(Invite(email=item.email, invited_by=admin.email))

    item.status = "done"
    item.handled_at = utcnow()
    item.handled_by = admin.email
    session.commit()

    events.emit("access_approved", by=admin.id, email=item.email)
    mailer.send(
        subject="Fantasy Nerdball — you're in",
        body=(
            f"{item.email} can now sign in at {settings.public_base_url}\n\n"
            "Sign in with the Google account for that address."
        ),
    )
    return {"ok": True, "email": item.email}


@router.post("/api/admin/inbox/{item_id}/done")
def mark_done(
    item_id: int,
    admin: User = Depends(current_admin),
    session: Session = Depends(get_session),
) -> dict:
    """Deal with an item without letting anyone in."""
    item = session.get(InboxItem, item_id)
    if item is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such item")

    item.status = "done"
    item.handled_at = utcnow()
    item.handled_by = admin.email
    session.commit()
    return {"ok": True}
