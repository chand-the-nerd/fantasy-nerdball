"""Sign-in routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from .. import events, guest, metrics
from ..auth import is_allowed, oauth, upsert_user
from ..config import settings
from ..db import get_session
from ..models import User

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/config")
def auth_config() -> dict:
    """Tells the sign-in screen which methods are available."""
    return {
        "google": settings.google_configured,
        "dev_login": settings.dev_mode and bool(settings.dev_login_email),
        "guest": settings.guest_mode,
    }


@router.get("/login")
async def login(request: Request):
    if not settings.google_configured:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Google sign-in isn't configured. Set GOOGLE_CLIENT_ID and "
            "GOOGLE_CLIENT_SECRET, then redeploy.",
        )
    return await oauth.google.authorize_redirect(request, settings.oauth_redirect_uri)


@router.get("/callback")
async def callback(request: Request, session: Session = Depends(get_session)):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception:
        return RedirectResponse("/?error=signin_failed")

    profile = token.get("userinfo") or {}
    email = (profile.get("email") or "").lower()

    if not email or not profile.get("email_verified", True):
        events.emit("sign_in_denied", reason="no_verified_email")
        return RedirectResponse("/?error=no_verified_email")

    if not is_allowed(email, session):
        # The address is recorded here and nowhere else in the log: a
        # refused sign-in is only actionable if you know who to invite.
        events.emit("sign_in_denied", reason="not_invited", email=email)
        return RedirectResponse("/?error=not_invited")

    try:
        user = upsert_user(
            session,
            email=email,
            sub=profile.get("sub"),
            name=profile.get("name", ""),
            picture=profile.get("picture", ""),
        )
    except HTTPException:
        events.emit("sign_in_denied", reason="league_full", email=email)
        return RedirectResponse("/?error=league_full")

    request.session["user_id"] = user.id
    events.emit(
        "sign_in",
        user=user.id,
        method="google",
        # Distinguishes a first sign-in from a returning one, which is
        # how you count growth rather than activity.
        new_user=user.created_at == user.last_seen_at,
    )
    return RedirectResponse("/")


@router.post("/guest")
def guest_login(request: Request, session: Session = Depends(get_session)):
    """Start a throwaway session for someone who hasn't been invited.

    A guest account is created rather than the request being let through
    unauthenticated: every route below this expects a user id, and a
    guest still needs somewhere to put a squad while it looks at one.
    What it doesn't get is a seat, or anything that outlives the session.
    """
    if not settings.guest_mode:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Guest access is switched off on this deployment.",
        )
    user = guest.create(session)
    request.session["user_id"] = user.id
    metrics.set_actor(user.id, True)
    events.emit("guest_start", user=user.id, guest=True)
    return {"ok": True}


@router.post("/dev-login")
def dev_login(request: Request, session: Session = Depends(get_session)):
    """Local development only. Never enabled when DEV_MODE is off."""
    if not (settings.dev_mode and settings.dev_login_email):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not available")
    user = upsert_user(
        session,
        email=settings.dev_login_email,
        sub=None,
        name="Local developer",
        picture="",
    )
    request.session["user_id"] = user.id
    return {"ok": True}


@router.post("/logout")
def logout(request: Request, session: Session = Depends(get_session)) -> dict:
    """Ends the session, and for a guest ends the account with it.

    This is the promise on the sign-in page kept literally: a guest's
    squads, runs and settings are gone the moment they leave.
    """
    user_id = request.session.get("user_id")
    request.session.clear()

    if user_id:
        user = session.get(User, user_id)
        if user is not None:
            events.emit("sign_out", user=user.id, guest=bool(user.is_guest))
            if user.is_guest:
                guest.discard(session, user)
        events.forget_session(user_id)

    return {"ok": True}
