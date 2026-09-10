"""Performance against the rest of the game."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import current_user
from ..config import settings
from ..db import get_session
from ..models import GameweekResult, GameweekStat, Squad, User
from ..schemas import ResultIn
from ..services import fpl

router = APIRouter(prefix="/api/performance", tags=["performance"])


def _refresh_globals(session: Session, season: str) -> None:
    """Top up the benchmark table, but never fail a page load over it."""
    try:
        fpl.sync_global_stats(session, season)
    except Exception:
        pass


@router.get("")
def performance(
    season: str | None = None,
    refresh: bool = False,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> dict:
    season = season or settings.current_season

    stats = session.scalars(
        select(GameweekStat).where(GameweekStat.season == season)
    ).all()
    if not stats or refresh:
        _refresh_globals(session, season)
        stats = session.scalars(
            select(GameweekStat).where(GameweekStat.season == season)
        ).all()

    if refresh and user.fpl_entry_id:
        try:
            fpl.sync_entry_history(session, user)
        except Exception:
            pass

    global_by_gw = {row.gameweek: row for row in stats}
    squads = {
        row.gameweek: row
        for row in session.scalars(
            select(Squad).where(Squad.user_id == user.id, Squad.season == season)
        ).all()
    }
    results = {
        row.gameweek: row
        for row in session.scalars(
            select(GameweekResult).where(
                GameweekResult.user_id == user.id, GameweekResult.season == season
            )
        ).all()
    }

    gameweeks = sorted(set(global_by_gw) | set(squads) | set(results))
    series = []
    for gw in gameweeks:
        stat = global_by_gw.get(gw)
        squad = squads.get(gw)
        result = results.get(gw)

        # Only chart gameweeks that have something real in them.
        if stat is not None and not stat.finished and squad is None and result is None:
            continue

        series.append(
            {
                "gameweek": gw,
                "projected": round(squad.projected_points, 1) if squad else None,
                "actual": result.actual_points if result else None,
                "global_average": stat.average_score if stat and stat.finished else None,
                "global_highest": stat.highest_score if stat and stat.finished else None,
                "overall_rank": result.overall_rank if result else None,
                "finished": bool(stat.finished) if stat else False,
                "chip": squad.chip if squad else "",
            }
        )

    scored = [
        row for row in series
        if row["actual"] is not None and row["global_average"] is not None
    ]
    total_actual = sum(row["actual"] for row in scored)
    total_average = sum(row["global_average"] for row in scored)
    beaten = sum(1 for row in scored if row["actual"] > row["global_average"])

    projected_pairs = [
        row for row in series if row["actual"] is not None and row["projected"] is not None
    ]
    model_error = (
        sum(abs(row["actual"] - row["projected"]) for row in projected_pairs)
        / len(projected_pairs)
        if projected_pairs
        else None
    )

    return {
        "season": season,
        "series": series,
        "summary": {
            "gameweeks_scored": len(scored),
            "total_points": round(total_actual, 1),
            "total_global_average": round(total_average, 1),
            "points_above_average": round(total_actual - total_average, 1),
            "gameweeks_beating_average": beaten,
            "latest_overall_rank": next(
                (row["overall_rank"] for row in reversed(series) if row["overall_rank"]),
                None,
            ),
            "model_mean_error": round(model_error, 1) if model_error is not None else None,
            "fpl_entry_linked": bool(user.fpl_entry_id),
        },
    }


@router.post("/results")
def record_result(
    payload: ResultIn,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> dict:
    """Enter a gameweek score by hand, for anyone not linking an FPL team."""
    season = payload.season or settings.current_season

    if user.fpl_entry_id:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Your points come from your linked FPL team. Unlink it first to "
            "enter scores by hand.",
        )

    row = session.scalar(
        select(GameweekResult).where(
            GameweekResult.user_id == user.id,
            GameweekResult.season == season,
            GameweekResult.gameweek == payload.gameweek,
        )
    )
    if row is None:
        row = GameweekResult(user_id=user.id, season=season, gameweek=payload.gameweek)
        session.add(row)

    row.actual_points = payload.actual_points
    row.source = "manual"

    squad = session.scalar(
        select(Squad).where(
            Squad.user_id == user.id,
            Squad.season == season,
            Squad.gameweek == payload.gameweek,
        )
    )
    if squad is not None:
        row.projected_points = squad.projected_points

    session.commit()
    return {"ok": True, "gameweek": payload.gameweek}


@router.get("/league")
def league_table(
    season: str | None = None,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> dict:
    """How everyone on this deployment is doing, side by side."""
    season = season or settings.current_season

    users = session.scalars(
        select(User).where(User.is_guest.is_(False))
    ).all()
    stats = {
        row.gameweek: row
        for row in session.scalars(
            select(GameweekStat).where(GameweekStat.season == season)
        ).all()
    }
    finished_average = sum(
        row.average_score for row in stats.values() if row.finished
    )

    standings = []
    for member in users:
        results = session.scalars(
            select(GameweekResult).where(
                GameweekResult.user_id == member.id, GameweekResult.season == season
            )
        ).all()
        total = sum(row.actual_points or 0 for row in results)
        latest_rank = next(
            (row.overall_rank for row in sorted(results, key=lambda r: -r.gameweek)
             if row.overall_rank),
            None,
        )
        standings.append(
            {
                "user_id": member.id,
                "name": member.name or member.email.split("@")[0],
                "avatar_url": member.avatar_url,
                "total_points": round(total, 1),
                "gameweeks": len(results),
                "overall_rank": latest_rank,
                "is_you": member.id == user.id,
            }
        )

    standings.sort(key=lambda row: -row["total_points"])
    for position, row in enumerate(standings, start=1):
        row["position"] = position

    return {
        "season": season,
        "global_average_total": round(finished_average, 1),
        "standings": standings,
    }
