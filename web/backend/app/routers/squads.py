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
from ..schemas import ActivateOptionIn, SquadOut

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


@router.post("/{gameweek}/activate", response_model=SquadOut)
def activate_option(
    gameweek: int,
    body: ActivateOptionIn,
    season: str | None = None,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> Squad:
    """Put one of the run's alternative squads in force.

    Everything the option needs was worked out during the run, so this is a
    swap rather than a re-optimisation: the summary columns, the payload the
    pitch draws from, and the engine rows next week transfers from all move
    together. They have to — leaving the engine rows behind would show one
    squad and then transfer from another.
    """
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

    payload = dict(squad.payload or {})
    options = payload.get("options") or []
    chosen = next((o for o in options if o.get("key") == body.option), None)

    if chosen is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "That option isn't available for this gameweek. Run the "
            "optimiser again to rebuild the list.",
        )

    transfers = chosen.get("transfers") or {"in": [], "out": []}
    made = int(chosen.get("transfers_made") or 0)

    payload.update(
        {
            "starting": chosen.get("starting", []),
            "bench": chosen.get("bench", []),
            "formation": chosen.get("formation", ""),
            "projected_points": chosen.get("projected_points", 0.0),
            "squad_value": chosen.get("squad_value", 0.0),
            "bank": chosen.get("bank", 0.0),
            "transfers_made": made,
            "penalty_points": int(chosen.get("penalty_points") or 0),
            "made_transfers": made > 0,
            "transfers": transfers,
            "active_option": chosen["key"],
        }
    )

    # A hold has no reasoning to show, and the recommendation's reasoning
    # would be actively misleading against a squad it didn't recommend.
    if chosen.get("kind") == "previous":
        payload["transfer_reason"] = "Holding last week's squad, by your choice."
        payload["points_gain_per_gw"] = None
    elif not chosen.get("recommended"):
        payload["transfer_reason"] = (
            f"{chosen.get('label', 'This option')} is your pick, not the "
            "optimiser's."
        )
        payload["points_gain_per_gw"] = None

    squad.payload = payload
    squad.formation = payload["formation"]
    squad.projected_points = float(payload["projected_points"] or 0.0)
    squad.squad_value = float(payload["squad_value"] or 0.0)
    squad.bank = float(payload["bank"] or 0.0)
    squad.transfers_made = made
    squad.penalty_points = int(payload["penalty_points"] or 0)
    squad.active_option = chosen["key"]

    rows = (squad.option_rows or {}).get(chosen["key"])
    if rows:
        squad.engine_rows = rows

    session.commit()
    session.refresh(squad)
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
