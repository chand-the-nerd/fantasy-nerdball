"""Saved squads."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import current_user
from ..config import settings
from ..db import get_session
from ..models import Squad, User
from ..schemas import SquadOut

router = APIRouter(prefix="/api/squads", tags=["squads"])


@router.get("", response_model=list[SquadOut])
def list_squads(
    season: str | None = None,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> list[Squad]:
    season = season or settings.current_season
    return list(
        session.scalars(
            select(Squad)
            .where(Squad.user_id == user.id, Squad.season == season)
            .order_by(Squad.gameweek.desc())
        ).all()
    )


@router.get("/latest", response_model=SquadOut | None)
def latest_squad(
    season: str | None = None,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> Squad | None:
    season = season or settings.current_season
    return session.scalar(
        select(Squad)
        .where(Squad.user_id == user.id, Squad.season == season)
        .order_by(Squad.gameweek.desc())
        .limit(1)
    )


@router.get("/{gameweek}", response_model=SquadOut)
def read_squad(
    gameweek: int,
    season: str | None = None,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> Squad:
    season = season or settings.current_season
    squad = session.scalar(
        select(Squad).where(
            Squad.user_id == user.id,
            Squad.season == season,
            Squad.gameweek == gameweek,
        )
    )
    if squad is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No squad saved for gameweek {gameweek}. Run the optimiser to build one.",
        )
    return squad


@router.delete("/{gameweek}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response, response_model=None)
def delete_squad(
    gameweek: int,
    season: str | None = None,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> None:
    season = season or settings.current_season
    squad = session.scalar(
        select(Squad).where(
            Squad.user_id == user.id,
            Squad.season == season,
            Squad.gameweek == gameweek,
        )
    )
    if squad is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No squad for that gameweek")
    session.delete(squad)
    session.commit()
