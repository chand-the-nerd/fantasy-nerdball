"""Reads the public FPL API for benchmarks and a manager's real results."""

from __future__ import annotations

import datetime as dt
import threading
from typing import Any

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import GameweekResult, GameweekStat, User, utcnow

BASE_URL = "https://fantasy.premierleague.com/api"
TIMEOUT = 30

_bootstrap_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()
CACHE_TTL = dt.timedelta(minutes=15)


def _get(path: str) -> dict:
    response = requests.get(f"{BASE_URL}{path}", timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def bootstrap(force: bool = False) -> dict:
    """Cached bootstrap-static. Every page load would otherwise hit the API."""
    with _cache_lock:
        cached = _bootstrap_cache.get("data")
        fetched_at = _bootstrap_cache.get("at")
        fresh = fetched_at and (utcnow() - fetched_at) < CACHE_TTL
        if cached and fresh and not force:
            return cached

    data = _get("/bootstrap-static/")
    with _cache_lock:
        _bootstrap_cache["data"] = data
        _bootstrap_cache["at"] = utcnow()
    return data


def current_gameweek() -> int:
    """The gameweek to plan for: the next one that has not kicked off."""
    events = bootstrap().get("events", [])
    for event in events:
        if event.get("is_next"):
            return int(event["id"])
    for event in events:
        if not event.get("finished"):
            return int(event["id"])
    return int(events[-1]["id"]) if events else 1


def _parse_deadline(raw: str | None) -> dt.datetime | None:
    if not raw:
        return None
    try:
        return dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def sync_global_stats(session: Session, season: str | None = None) -> int:
    """Store each finished gameweek's global average and highest score."""
    season = season or settings.current_season
    events = bootstrap(force=True).get("events", [])
    existing = {
        row.gameweek: row
        for row in session.scalars(
            select(GameweekStat).where(GameweekStat.season == season)
        ).all()
    }

    updated = 0
    for event in events:
        gw = int(event["id"])
        row = existing.get(gw)
        if row is None:
            row = GameweekStat(season=season, gameweek=gw)
            session.add(row)
        row.average_score = float(event.get("average_entry_score") or 0)
        row.highest_score = float(event.get("highest_score") or 0)
        row.finished = bool(event.get("finished"))
        row.deadline = _parse_deadline(event.get("deadline_time"))
        row.fetched_at = utcnow()
        updated += 1

    session.commit()
    return updated


def sync_entry_history(session: Session, user: User, season: str | None = None) -> int:
    """Pull a manager's real per-gameweek points from their linked FPL side."""
    if not user.fpl_entry_id:
        return 0
    season = season or settings.current_season

    data = _get(f"/entry/{user.fpl_entry_id}/history/")
    existing = {
        row.gameweek: row
        for row in session.scalars(
            select(GameweekResult).where(
                GameweekResult.user_id == user.id, GameweekResult.season == season
            )
        ).all()
    }

    count = 0
    for entry in data.get("current", []):
        gw = int(entry["event"])
        row = existing.get(gw)
        if row is None:
            row = GameweekResult(user_id=user.id, season=season, gameweek=gw)
            session.add(row)
        row.actual_points = float(entry.get("points") or 0)
        row.overall_rank = entry.get("overall_rank")
        row.source = "fpl"
        count += 1

    session.commit()
    return count


def verify_entry(entry_id: int) -> dict:
    """Check an FPL team id exists and return its name, for the settings page."""
    data = _get(f"/entry/{entry_id}/")
    manager = " ".join(
        part for part in (data.get("player_first_name"), data.get("player_last_name")) if part
    )
    return {
        "entry_id": entry_id,
        "team_name": data.get("name", ""),
        "manager_name": manager,
        "overall_rank": data.get("summary_overall_rank"),
        "total_points": data.get("summary_overall_points"),
    }
