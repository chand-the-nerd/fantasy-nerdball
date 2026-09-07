"""The signed-in manager's own profile and optimiser settings."""

from __future__ import annotations

import requests
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..auth import current_user
from ..db import get_session
from ..models import User, UserSettings
from ..schemas import EntryLinkIn, SettingsIn, SettingsOut, UserOut
from ..services import fpl

router = APIRouter(prefix="/api/me", tags=["me"])


def _settings_row(session: Session, user: User) -> UserSettings:
    if user.settings is None:
        user.settings = UserSettings()
        session.commit()
        session.refresh(user)
    return user.settings


@router.get("", response_model=UserOut)
def read_me(user: User = Depends(current_user)) -> User:
    return user


@router.get("/settings", response_model=SettingsOut)
def read_settings(
    user: User = Depends(current_user), session: Session = Depends(get_session)
) -> UserSettings:
    return _settings_row(session, user)


POSITIONS = ("GK", "DEF", "MID", "FWD")
WEIGHT_KEYS = ("form", "historic", "difficulty")


def _validate_position_weights(overrides: dict) -> None:
    """Each position's three weights must be a sane distribution.

    The engine treats them as proportions of a player's core score, so a set
    that doesn't sum to 1.0 silently rescales everyone and makes the output
    hard to reason about.
    """
    weights = overrides.get("POSITION_SCORING_WEIGHTS")
    if weights is None:
        return

    if not isinstance(weights, dict):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Scoring weights are malformed.")

    for position in POSITIONS:
        row = weights.get(position)
        if row is None:
            continue
        if not isinstance(row, dict) or any(key not in row for key in WEIGHT_KEYS):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"{position} weights need a form, historic and difficulty value.",
            )
        try:
            values = [float(row[key]) for key in WEIGHT_KEYS]
        except (TypeError, ValueError):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"{position} weights must be numbers."
            )
        if any(value < 0 for value in values):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"{position} weights can't be negative."
            )
        total = sum(values)
        if abs(total - 1.0) > 0.01:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"{position} weights add up to {total:.2f}. They need to total 1.00.",
            )


def _validate_forced(forced: dict) -> None:
    """A forced squad still has to fit inside the FPL position limits."""
    limits = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    for position, names in (forced or {}).items():
        if position not in limits:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"{position} isn't a position."
            )
        if len(names) > limits[position]:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"You can force at most {limits[position]} {position} players; "
                f"you've listed {len(names)}.",
            )


@router.put("/settings", response_model=SettingsOut)
def update_settings(
    payload: SettingsIn,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> UserSettings:
    row = _settings_row(session, user)

    changes = payload.model_dump(exclude_unset=True)

    if changes.get("overrides") is not None:
        _validate_position_weights(changes["overrides"])
    if changes.get("forced_selections") is not None:
        _validate_forced(changes["forced_selections"])

    for field, value in changes.items():
        if value is not None:
            setattr(row, field, value)

    chips = [row.wildcard, row.bench_boost, row.triple_captain]
    if sum(1 for chip in chips if chip) > 1:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Only one chip can be active in a gameweek.",
        )

    session.commit()
    session.refresh(row)
    return row


@router.post("/fpl-entry", response_model=UserOut)
def link_entry(
    payload: EntryLinkIn,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> User:
    """Connect a real FPL team so actual points can be charted."""
    if payload.fpl_entry_id is None:
        user.fpl_entry_id = None
        session.commit()
        session.refresh(user)
        return user

    try:
        fpl.verify_entry(payload.fpl_entry_id)
    except requests.HTTPError:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"FPL couldn't find team {payload.fpl_entry_id}. The id is the "
            "number in your team's URL on the FPL site.",
        )
    except requests.RequestException:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The FPL API isn't responding. Try again shortly.",
        )

    user.fpl_entry_id = payload.fpl_entry_id
    session.commit()
    fpl.sync_entry_history(session, user)
    session.refresh(user)
    return user


@router.get("/reference")
def reference(user: User = Depends(current_user)) -> dict:
    """Defaults and club names the Setup page needs to render its editors."""
    from ..engine.runtime_config import _base_config_class, ensure_engine_on_path
    from ..config import settings as app_settings

    default_weights: dict = {}
    try:
        ensure_engine_on_path(app_settings.engine_dir)
        base = _base_config_class()
        default_weights = base.POSITION_SCORING_WEIGHTS
    except Exception:
        default_weights = {
            pos: {"form": 0.5, "historic": 0.25, "difficulty": 0.25}
            for pos in POSITIONS
        }

    teams: list[str] = []
    try:
        teams = sorted(team["name"] for team in fpl.bootstrap().get("teams", []))
    except Exception:
        pass

    return {
        "positions": list(POSITIONS),
        "weight_keys": list(WEIGHT_KEYS),
        "default_weights": default_weights,
        "teams": teams,
        "squad_limits": {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
    }
