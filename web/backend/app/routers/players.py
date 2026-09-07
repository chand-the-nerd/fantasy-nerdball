"""The current season's player pool, for the Setup page's pickers."""

from __future__ import annotations

import datetime as dt
import threading

from fastapi import APIRouter, Depends

from ..auth import current_user
from ..models import User, utcnow
from ..services import fpl

router = APIRouter(prefix="/api/players", tags=["players"])

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

_cache: dict = {}
_lock = threading.Lock()
CACHE_TTL = dt.timedelta(minutes=30)


def _build() -> dict:
    data = fpl.bootstrap()
    teams = {team["id"]: team["name"] for team in data.get("teams", [])}

    # The optimiser matches forced and blacklisted names against web_name, so
    # that is the string the picker has to hand back — anything else silently
    # fails to match and the constraint is quietly dropped.
    counts: dict[str, int] = {}
    for element in data.get("elements", []):
        counts[element["web_name"]] = counts.get(element["web_name"], 0) + 1

    players = []
    for element in data.get("elements", []):
        name = element["web_name"]
        full = " ".join(
            part for part in (element.get("first_name"), element.get("second_name")) if part
        )
        players.append(
            {
                "id": element["id"],
                "name": name,
                "full_name": full,
                "position": POSITIONS.get(element.get("element_type"), "MID"),
                "team": teams.get(element.get("team"), ""),
                "price": round(element.get("now_cost", 0) / 10, 1),
                "status": element.get("status", "a"),
                "news": element.get("news", "") or "",
                "chance_of_playing": element.get("chance_of_playing_next_round"),
                "total_points": element.get("total_points", 0),
                "selected_by": float(element.get("selected_by_percent") or 0),
                # Two players sharing a web_name can't be told apart by the
                # optimiser; it warns and takes the first. Flag it here so the
                # picker can warn before the run instead of after.
                "ambiguous": counts.get(name, 0) > 1,
            }
        )

    players.sort(key=lambda p: (-p["total_points"], p["name"]))
    return {"season": fpl.settings.current_season, "players": players}


@router.get("")
def list_players(refresh: bool = False, user: User = Depends(current_user)) -> dict:
    with _lock:
        cached = _cache.get("data")
        fetched = _cache.get("at")
        if cached and fetched and (utcnow() - fetched) < CACHE_TTL and not refresh:
            return cached

    try:
        built = _build()
    except Exception:
        # A stale list beats an empty picker.
        with _lock:
            if _cache.get("data"):
                return _cache["data"]
        return {"season": "", "players": [], "error": "The FPL API isn't responding."}

    with _lock:
        _cache["data"] = built
        _cache["at"] = utcnow()
    return built
