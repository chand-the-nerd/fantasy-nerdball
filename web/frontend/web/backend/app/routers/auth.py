"""Sign-in routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from ..auth import is_allowed, oauth, upsert_user
from ..config import settings
from ..db import get_session

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/config")
def auth_config() -> dict:
    """Tells the sign-in screen which methods are available."""
    return {
        "google": settings.google_configured,
        "dev_login": settings.dev_mode and bool(settings.dev_login_email),
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
        return RedirectResponse("/?error=no_verified_email")

    if not is_allowed(email, session):
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
        return RedirectResponse("/?error=league_full")

    request.session["user_id"] = user.id
    return RedirectResponse("/")


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
def logout(request: Request) -> dict:
    request.session.clear()
    return {"ok": True}
