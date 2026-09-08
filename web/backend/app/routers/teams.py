"""Team strength overview."""

from __future__ import annotations

import datetime as dt
import threading

from fastapi import APIRouter, Depends, HTTPException

from ..auth import current_user
from ..models import User, utcnow
from ..services import team_ratings

router = APIRouter(prefix="/api/teams", tags=["teams"])

_cache: dict = {}
_lock = threading.Lock()
CACHE_TTL = dt.timedelta(minutes=30)


@router.get("")
def teams(
    look_ahead: int | None = None,
    refresh: bool = False,
    user: User = Depends(current_user),
) -> dict:
    row = user.settings
    window = look_ahead or (int(getattr(row, "first_n_gameweeks", 5) or 5) if row else 5)
    key = f"la-{window}"

    with _lock:
        cached = _cache.get(key)
        if cached and not refresh and (utcnow() - cached["at"]) < CACHE_TTL:
            return cached["data"]

    try:
        data = team_ratings.build_ratings(window)
    except Exception:
        with _lock:
            if _cache.get(key):
                return _cache[key]["data"]
        raise HTTPException(503, "The FPL API isn't responding. Try again shortly.")

    with _lock:
        _cache[key] = {"data": data, "at": utcnow()}
    return data
