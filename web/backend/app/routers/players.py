"""The current season's player pool, for the Setup page's pickers."""

from __future__ import annotations

import datetime as dt
import threading

import requests
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import current_user
from ..config import settings
from ..db import get_session
from ..models import PlayerScores, User, utcnow
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


def _ranked(rows: list[dict], limit: int, max_ownership: float | None) -> dict:
    """Top N per position, optionally filtered to low-owned players."""
    out: dict[str, list[dict]] = {}
    for position in ("GK", "DEF", "MID", "FWD"):
        pool = [r for r in rows if r.get("position") == position]
        if max_ownership is not None:
            pool = [
                r
                for r in pool
                if r.get("ownership") is not None
                and float(r["ownership"]) < max_ownership
            ]
        pool.sort(key=lambda r: (r.get("score") or 0), reverse=True)
        out[position] = pool[:limit]
    return out


def _load_scores(session: Session, user: User) -> PlayerScores | None:
    return session.scalar(
        select(PlayerScores).where(
            PlayerScores.user_id == user.id,
            PlayerScores.season == settings.current_season,
        )
    )


@router.get("/best")
def best_players(
    limit: int = 5,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> dict:
    """The model's top picks per position, using this manager's own settings."""
    cache = _load_scores(session, user)
    if cache is None:
        return {
            "available": False,
            "reason": "Run the optimiser once and these fill in. Scoring every "
                      "player is the slow part of a run, so this reuses what "
                      "the run already worked out rather than doing it twice.",
        }
    return {
        "available": True,
        "gameweek": cache.gameweek,
        "look_ahead": cache.look_ahead,
        "computed_at": cache.created_at,
        "positions": _ranked(cache.players or [], limit, None),
    }


@router.get("/differentials")
def differential_players(
    limit: int = 5,
    max_ownership: float = 5.0,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> dict:
    """The same ranking, restricted to players most managers don't own."""
    cache = _load_scores(session, user)
    if cache is None:
        return {"available": False, "reason": "Run the optimiser once to fill this in."}

    ranked = _ranked(cache.players or [], limit, max_ownership)
    empty = [pos for pos, rows in ranked.items() if not rows]
    return {
        "available": True,
        "gameweek": cache.gameweek,
        "max_ownership": max_ownership,
        "positions": ranked,
        "thin_positions": empty,
    }


@router.get("/{player_id}")
def player_detail(
    player_id: int,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> dict:
    """Everything known about one player, live from FPL plus the model's score.

    Deliberately not dependent on a run having happened: the underlying stats
    and fixtures come straight from the API, so lookup works from a cold start.
    """
    try:
        data = fpl.bootstrap()
    except Exception:
        raise HTTPException(503, "The FPL API isn't responding.")

    element = next(
        (e for e in data.get("elements", []) if int(e["id"]) == player_id), None
    )
    if element is None:
        raise HTTPException(404, "No such player.")

    teams = {int(t["id"]): t for t in data.get("teams", [])}
    team = teams.get(element.get("team"), {})

    def number(key: str) -> float | None:
        raw = element.get(key)
        if raw in (None, ""):
            return None
        try:
            return round(float(raw), 2)
        except (TypeError, ValueError):
            return None

    # Upcoming fixtures over this manager's look-ahead window.
    row = user.settings
    look_ahead = int(getattr(row, "first_n_gameweeks", 1) or 1) if row else 3
    fixtures = []
    try:
        upcoming = requests.get(
            f"{fpl.BASE_URL}/fixtures/?future=1", timeout=fpl.TIMEOUT
        ).json()
        for fixture in upcoming:
            home, away = fixture.get("team_h"), fixture.get("team_a")
            if element.get("team") not in (home, away):
                continue
            at_home = element.get("team") == home
            opponent = teams.get(away if at_home else home, {})
            fixtures.append(
                {
                    "gameweek": fixture.get("event"),
                    "opponent": opponent.get("short_name") or opponent.get("name", ""),
                    "venue": "Home" if at_home else "Away",
                    "difficulty": fixture.get("team_h_difficulty" if at_home else "team_a_difficulty"),
                }
            )
            if len(fixtures) >= max(look_ahead, 5):
                break
    except Exception:
        fixtures = []

    cache = _load_scores(session, user)
    model = None
    if cache:
        model = next(
            (r for r in (cache.players or []) if r.get("id") == player_id), None
        )

    return {
        "id": element["id"],
        "name": element["web_name"],
        "full_name": " ".join(
            p for p in (element.get("first_name"), element.get("second_name")) if p
        ),
        "position": POSITIONS.get(element.get("element_type"), "MID"),
        "team": team.get("name", ""),
        "price": round(element.get("now_cost", 0) / 10, 1),
        "status": element.get("status", "a"),
        "news": element.get("news", "") or "",
        "chance_of_playing": element.get("chance_of_playing_next_round"),
        "fpl": {
            "total_points": element.get("total_points"),
            "points_per_game": number("points_per_game"),
            "form": number("form"),
            "minutes": element.get("minutes"),
            "starts": element.get("starts"),
            "goals": element.get("goals_scored"),
            "assists": element.get("assists"),
            "clean_sheets": element.get("clean_sheets"),
            "goals_conceded": element.get("goals_conceded"),
            "bonus": element.get("bonus"),
            "bps": element.get("bps"),
            "ownership": number("selected_by_percent"),
            "transfers_in_event": element.get("transfers_in_event"),
            "transfers_out_event": element.get("transfers_out_event"),
        },
        "underlying": {
            "xg": number("expected_goals"),
            "xa": number("expected_assists"),
            "xgi": number("expected_goal_involvements"),
            "xgc": number("expected_goals_conceded"),
            "xg_per_90": number("expected_goals_per_90"),
            "xa_per_90": number("expected_assists_per_90"),
            "xgi_per_90": number("expected_goal_involvements_per_90"),
            "xgc_per_90": number("expected_goals_conceded_per_90"),
        },
        "fixtures": fixtures,
        "look_ahead": look_ahead,
        "model": model,
    }
