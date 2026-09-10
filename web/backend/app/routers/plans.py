"""Planning several gameweeks ahead."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import events, guest
from ..auth import current_user
from ..config import settings
from ..db import get_session
from ..engine import jobs
from ..engine.planner import CHIP_FIELDS
from ..models import Plan, Run, User
from ..schemas import PlanIn, PlanOut
from ..services import fpl

router = APIRouter(prefix="/api/plans", tags=["plans"])

MIN_WEEKS = 3
MAX_WEEKS = 8


@router.post("", response_model=PlanOut, status_code=status.HTTP_202_ACCEPTED)
def start_plan(
    payload: PlanIn,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> Plan:
    # A plan is a long job whose whole point is the weeks it saves for
    # later, so it belongs to an account that will still be there.
    if user.is_guest:
        raise guest.members_only("The planner")

    season = payload.season or settings.current_season

    if not MIN_WEEKS <= payload.weeks <= MAX_WEEKS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"A plan covers between {MIN_WEEKS} and {MAX_WEEKS} gameweeks.",
        )

    # One job per manager at a time: a plan holds the worker for several
    # minutes and a second one would just queue behind it.
    busy_run = session.scalar(
        select(Run).where(Run.user_id == user.id, Run.status.in_(("queued", "running")))
    )
    busy_plan = session.scalar(
        select(Plan).where(
            Plan.user_id == user.id, Plan.status.in_(("queued", "running"))
        )
    )
    if busy_run is not None or busy_plan is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "You already have something running."
        )

    start = payload.start_gameweek
    if start is None:
        try:
            start = fpl.current_gameweek()
        except Exception:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "Couldn't reach the FPL API to work out the gameweek.",
            )

    covered = range(start, start + payload.weeks)
    chips: dict[str, str] = {}
    for raw_gw, chip in (payload.chips or {}).items():
        gameweek = int(raw_gw)
        if not chip:
            continue
        if chip not in CHIP_FIELDS:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"{chip} isn't a chip this plans for."
            )
        if gameweek not in covered:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Gameweek {gameweek} isn't in this plan, which runs "
                f"{start} to {start + payload.weeks - 1}.",
            )
        if chip in chips.values():
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"The {chip.replace('_', ' ')} is set for two gameweeks. "
                "You only get one.",
            )
        chips[str(gameweek)] = chip

    plan = Plan(
        user_id=user.id,
        season=season,
        start_gameweek=start,
        weeks=payload.weeks,
        chips=chips,
        status="queued",
    )
    session.add(plan)
    session.commit()
    session.refresh(plan)

    try:
        jobs.enqueue_plan(plan.id)
    except RuntimeError as error:
        plan.status = "failed"
        plan.error = str(error)
        session.commit()
        events.emit(
            "plan_rejected",
            reason="queue_full",
            queue_depth=jobs.queue_depth(session),
            **events.actor(user),
        )
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error))

    events.emit(
        "plan_queued",
        plan=plan.id,
        weeks=plan.weeks,
        chips=len(plan.chips or {}) or None,
        **events.actor(user),
    )
    return plan


@router.get("/latest", response_model=PlanOut | None)
def latest_plan(
    user: User = Depends(current_user), session: Session = Depends(get_session)
) -> Plan | None:
    return session.scalar(
        select(Plan)
        .where(Plan.user_id == user.id, Plan.season == settings.current_season)
        .order_by(Plan.created_at.desc())
        .limit(1)
    )


@router.get("/{plan_id}", response_model=PlanOut)
def get_plan(
    plan_id: int,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> Plan:
    plan = session.get(Plan, plan_id)
    if plan is None or plan.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such plan.")
    return plan
