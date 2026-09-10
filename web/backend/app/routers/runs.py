"""Starting an optimisation and watching it run."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import events
from ..auth import current_user
from ..config import settings
from ..db import get_session
from ..engine import jobs
from ..models import Run, User
from ..schemas import RunIn, RunOut
from ..services import fpl

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.post("", response_model=RunOut, status_code=status.HTTP_202_ACCEPTED)
def start_run(
    payload: RunIn,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> Run:
    season = payload.season or settings.current_season

    active = session.scalar(
        select(Run).where(
            Run.user_id == user.id, Run.status.in_(("queued", "running"))
        )
    )
    if active is not None:
        events.emit(
            "run_rejected",
            reason="already_running",
            **events.actor(user),
        )
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "You already have an optimisation in progress.",
        )

    gameweek = payload.gameweek
    if gameweek is None:
        try:
            gameweek = fpl.current_gameweek()
        except Exception:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "Couldn't reach the FPL API to work out the gameweek. "
                "Pick one manually and try again.",
            )

    run = Run(user_id=user.id, season=season, gameweek=gameweek, status="queued")
    session.add(run)
    session.commit()
    session.refresh(run)

    try:
        jobs.enqueue(run.id)
    except RuntimeError as error:
        run.status = "failed"
        run.error = str(error)
        session.commit()
        # The queue filling up is the single worker being overrun, which
        # is the first thing that will break as usage grows. Worth an
        # event of its own rather than being one 503 among many.
        events.emit(
            "run_rejected",
            reason="queue_full",
            queue_depth=jobs.queue_depth(session),
            **events.actor(user),
        )
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error))

    events.emit(
        "run_queued",
        run=run.id,
        gameweek=gameweek,
        queue_depth=jobs.queue_depth(session),
        **events.actor(user),
    )
    return run


@router.get("/latest", response_model=RunOut | None)
def latest_run(
    user: User = Depends(current_user), session: Session = Depends(get_session)
) -> Run | None:
    return session.scalar(
        select(Run).where(Run.user_id == user.id).order_by(Run.id.desc()).limit(1)
    )


@router.get("/{run_id}", response_model=RunOut)
def read_run(
    run_id: int,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> Run:
    run = session.get(Run, run_id)
    if run is None or run.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such run")
    return run


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response, response_model=None)
def cancel_run(
    run_id: int,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> None:
    """Drops a queued run. A run already in the optimiser plays out."""
    run = session.get(Run, run_id)
    if run is None or run.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such run")
    if run.status != "queued":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "This run has already started and can't be cancelled.",
        )
    run.status = "cancelled"
    session.commit()
    events.emit("run_cancelled", run=run.id, **events.actor(user))
