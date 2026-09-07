"""Scheduled maintenance, triggered by whatever you like.

Deliberately idempotent and cheap when there's nothing to do: it checks which
gameweeks have finished against what's already stored, so polling it hourly
costs one bootstrap fetch and returns "up-to-date" all week. That means any
trigger works — an external scheduler, the built-in timer, or a person.
"""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status

from ..auth import current_admin
from ..config import settings
from ..models import User
from ..services import history_updater

router = APIRouter(prefix="/api/cron", tags=["cron"])


def _authorise(token: str | None, authorization: str | None) -> None:
    """Accepts the secret as a query parameter or a bearer header.

    Two ways in because schedulers differ: some send only a URL, others let you
    set headers. A header is preferable — URLs turn up in logs.
    """
    if not settings.cron_secret:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Scheduled updates are switched off. Set CRON_SECRET on the "
            "service and redeploy to enable them.",
        )

    supplied = token
    if not supplied and authorization:
        parts = authorization.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            supplied = parts[1]

    if not supplied or not secrets.compare_digest(supplied, settings.cron_secret):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Bad or missing token.")


@router.post("/update-history")
def update_history(
    token: str | None = Query(default=None),
    authorization: str | None = Header(default=None),
    gameweek: int | None = Query(default=None, ge=1, le=38),
) -> dict:
    _authorise(token, authorization)
    return history_updater.update(force_gameweek=gameweek)


# Some schedulers only send GET. Same work, same guard.
@router.get("/update-history")
def update_history_get(
    token: str | None = Query(default=None),
    authorization: str | None = Header(default=None),
    gameweek: int | None = Query(default=None, ge=1, le=38),
) -> dict:
    _authorise(token, authorization)
    return history_updater.update(force_gameweek=gameweek)


@router.get("/status")
def cron_status(admin: User = Depends(current_admin)) -> dict:
    """For the admin page: what's stored, and what's outstanding."""
    state = history_updater.read_state()
    try:
        pending = history_updater.pending_gameweek()
    except Exception:
        pending = None
    return {
        "configured": bool(settings.cron_secret),
        "internal_scheduler": settings.history_auto_update,
        "last_gameweek": state.get("last_gameweek"),
        "updated_at": state.get("updated_at"),
        "pending_gameweek": pending,
    }


@router.post("/run-now")
def run_now(admin: User = Depends(current_admin)) -> dict:
    """Manual trigger from the admin page, for when you don't want to wait."""
    return history_updater.update()
