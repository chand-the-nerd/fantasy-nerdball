"""The signed-in manager's own profile and optimiser settings."""

from __future__ import annotations

import requests
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth import current_user
from ..config import settings
from ..db import get_session
from ..models import Squad, User, UserSettings
from ..schemas import EntryLinkIn, ImportSquadIn, SettingsIn, SettingsOut, UserOut
from ..services import fpl, fpl_import

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

    # Pulling their history is a nicety. If it fails the link itself is still
    # good, so don't fail the request over it.
    try:
        fpl.sync_entry_history(session, user)
    except Exception:
        pass

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


@router.get("/fpl-entry")
def entry_details(
    user: User = Depends(current_user), session: Session = Depends(get_session)
) -> dict:
    """What is currently linked, so the page can say so plainly."""
    if not user.fpl_entry_id:
        return {"linked": False}

    try:
        details = fpl.verify_entry(user.fpl_entry_id)
    except requests.RequestException:
        # Still linked; we just can't decorate it with the team name.
        return {"linked": True, "entry_id": user.fpl_entry_id, "unreachable": True}

    try:
        upcoming = fpl.current_gameweek()
    except Exception:
        upcoming = None

    imported = None
    if upcoming and upcoming > 1:
        squad = session.scalar(
            select(Squad).where(
                Squad.user_id == user.id,
                Squad.season == settings.current_season,
                Squad.gameweek == upcoming - 1,
            )
        )
        if squad is not None:
            imported = {
                "gameweek": squad.gameweek,
                "from_fpl": bool((squad.payload or {}).get("imported")),
            }

    return {
        "linked": True,
        "unreachable": False,
        "upcoming_gameweek": upcoming,
        "importable_gameweek": (upcoming - 1) if upcoming and upcoming > 1 else None,
        "existing_squad": imported,
        **details,
    }


@router.post("/import-squad")
def import_squad(
    payload: ImportSquadIn,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> dict:
    """Save the manager's real FPL side as a previous gameweek.

    Without this the optimiser has nothing to compare against mid-season and
    treats you as a new team, so its first set of transfers is meaningless.
    """
    if not user.fpl_entry_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Link your FPL team first, then import.",
        )

    try:
        upcoming = fpl.current_gameweek()
    except Exception:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The FPL API isn't responding. Try again shortly.",
        )

    gameweek = payload.gameweek or (upcoming - 1)
    if gameweek < 1:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "There's no gameweek before the first one to import.",
        )
    if gameweek >= upcoming:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Gameweek {gameweek} hasn't been played yet. The latest you can "
            f"import is gameweek {upcoming - 1}.",
        )

    try:
        result = fpl_import.import_summary(user.fpl_entry_id, gameweek, upcoming)
    except fpl_import.SquadImportError as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error))
    except requests.RequestException:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The FPL API isn't responding. Try again shortly.",
        )

    squad = session.scalar(
        select(Squad).where(
            Squad.user_id == user.id,
            Squad.season == settings.current_season,
            Squad.gameweek == gameweek,
        )
    )
    if squad is None:
        squad = Squad(
            user_id=user.id, season=settings.current_season, gameweek=gameweek
        )
        session.add(squad)

    data = result["payload"]
    squad.formation = data["formation"]
    squad.projected_points = 0.0
    squad.squad_value = result["squad_value"]
    squad.bank = result["bank"]
    squad.transfers_made = data["transfers_made"]
    squad.penalty_points = data["penalty_points"]
    squad.chip = data["chip"]
    squad.payload = data
    squad.engine_rows = result["engine_rows"]

    row = _settings_row(session, user)
    applied = []
    if payload.apply_budget:
        row.budget = result["budget"]
        applied.append(f"budget set to £{result['budget']}m")
    if payload.apply_free_transfers:
        row.free_transfers = result["free_transfers"]
        applied.append(f"{result['free_transfers']} free transfer"
                       f"{'s' if result['free_transfers'] != 1 else ''}")
    if payload.apply_free_transfers and result["free_hit_used"]:
        row.free_hit_prev_gw = True
        applied.append("Free Hit flagged")

    session.commit()

    return {
        "gameweek": gameweek,
        "formation": data["formation"],
        "players": len(result["engine_rows"]),
        "squad_value": result["squad_value"],
        "bank": result["bank"],
        "budget": result["budget"],
        "free_transfers": result["free_transfers"],
        "free_transfers_note": result["free_transfers_note"],
        "chip": data["chip"],
        "free_hit_used": result["free_hit_used"],
        "applied": applied,
    }
