"""The signed-in manager's own profile and optimiser settings."""

from __future__ import annotations

import unicodedata

import requests
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import guest
from ..auth import current_user
from ..config import settings
from ..db import get_session
from ..models import Squad, User, UserSettings
from ..schemas import (
    EntryLinkIn,
    ImportSquadIn,
    ManualSquadIn,
    PlayerRefIn,
    SettingsIn,
    SettingsOut,
    UserOut,
)
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
THEMES = ("legacy", "dark", "light")
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
    if user.is_guest:
        guest.check_lists(
            forced=changes.get("forced_selections"),
            blacklist=changes.get("blacklist_players"),
        )
    if changes.get("theme") is not None and changes["theme"] not in THEMES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"{changes['theme']} isn't one of the styles ({', '.join(THEMES)}).",
        )

    for field, value in changes.items():
        if value is not None:
            setattr(row, field, value)

    # Applied after the assignment rather than by rejecting the save: the
    # Setup page sends the whole row back, so a guest saving a budget
    # would otherwise be refused over fields it never let them touch.
    if user.is_guest:
        guest.enforce(row)

    chips = [row.wildcard, row.free_hit, row.bench_boost, row.triple_captain]
    if sum(1 for chip in chips if chip) > 1:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Only one chip can be active in a gameweek.",
        )

    session.commit()
    session.refresh(row)
    return row


# ── Forced picks and the avoid list ──────────────────────────────────────
#
# Both lists are editable in bulk from the Setup page. These endpoints exist
# so a single player can be added from wherever you happen to be looking at
# them — the pitch, the rankings, the lookup page — and so the reason an add
# is refused comes back as a sentence rather than as a run that fails an hour
# later with "No valid solution found".

SQUAD_LIMITS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
POSITION_WORDS = {
    "GK": "goalkeepers",
    "DEF": "defenders",
    "MID": "midfielders",
    "FWD": "forwards",
}
MAX_PER_CLUB = 3

_ACCENT_MAP = str.maketrans(
    {"ø": "o", "æ": "ae", "å": "a", "œ": "oe", "ß": "ss", "đ": "d", "ð": "d",
     "þ": "th", "ł": "l", "ı": "i"}
)


def _key(name: str) -> str:
    """Match the frontend's comparison: case- and accent-insensitive."""
    lowered = (name or "").strip().lower().translate(_ACCENT_MAP)
    decomposed = unicodedata.normalize("NFD", lowered)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _pool_by_name() -> dict[str, dict]:
    from .players import pool

    return {_key(p["name"]): p for p in pool().get("players", [])}


def _forced_pairs(row: UserSettings) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for position, names in (row.forced_selections or {}).items():
        for name in names or []:
            out.append((position, name))
    return out


def _conflict(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_409_CONFLICT, detail)


@router.post("/lists/force", response_model=SettingsOut)
def force_player(
    payload: PlayerRefIn,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> UserSettings:
    """Add one player to the forced picks, or explain why they can't go in."""
    row = _settings_row(session, user)
    name = payload.name.strip()
    key = _key(name)

    if user.is_guest and len(_forced_pairs(row)) >= guest.MAX_FORCED:
        raise guest.limit("forced picks", guest.MAX_FORCED)

    for position, existing in _forced_pairs(row):
        if _key(existing) == key:
            raise _conflict(
                f"{existing} is already a forced pick"
                + (f" ({POSITION_WORDS.get(position, position)})." if position else ".")
            )

    for existing in row.blacklist_players or []:
        if _key(existing) == key:
            raise _conflict(
                f"{existing} is on your avoid list. Take them off it first, "
                "or the optimiser has been told two opposite things."
            )

    known = _pool_by_name()
    entry = known.get(key)

    position = (payload.position or "").upper() or (entry or {}).get("position", "")
    if position not in SQUAD_LIMITS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Couldn't work out what position {name} plays, so there's nowhere "
            "to force them into. Add them from the Setup page instead.",
        )

    forced = {pos: list(names or []) for pos, names in (row.forced_selections or {}).items()}
    current = forced.get(position, [])
    limit = SQUAD_LIMITS[position]
    if len(current) >= limit:
        raise _conflict(
            f"You've already forced {len(current)} {POSITION_WORDS[position]}, "
            f"and a squad only has {limit}. Drop one first: {', '.join(current)}."
        )

    # Four forced from one club can never solve, so it's worth stopping here
    # rather than at the end of a run.
    if entry and entry.get("team"):
        club = entry["team"]
        same_club = [
            existing
            for _, existing in _forced_pairs(row)
            if (known.get(_key(existing)) or {}).get("team") == club
        ]
        if len(same_club) >= MAX_PER_CLUB:
            raise _conflict(
                f"You've already forced {len(same_club)} players from {club} "
                f"({', '.join(same_club)}), and FPL allows {MAX_PER_CLUB} per club."
            )

    forced[position] = current + [entry["name"] if entry else name]
    row.forced_selections = forced
    session.commit()
    session.refresh(row)
    return row


@router.post("/lists/unforce", response_model=SettingsOut)
def unforce_player(
    payload: PlayerRefIn,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> UserSettings:
    row = _settings_row(session, user)
    key = _key(payload.name)

    forced = {pos: list(names or []) for pos, names in (row.forced_selections or {}).items()}
    removed = False
    for position, names in forced.items():
        kept = [n for n in names if _key(n) != key]
        if len(kept) != len(names):
            removed = True
        forced[position] = kept

    if not removed:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"{payload.name} isn't a forced pick."
        )

    row.forced_selections = forced
    session.commit()
    session.refresh(row)
    return row


@router.post("/lists/blacklist", response_model=SettingsOut)
def blacklist_player(
    payload: PlayerRefIn,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> UserSettings:
    """Add one player to the avoid list, or explain why they can't go on it."""
    row = _settings_row(session, user)
    name = payload.name.strip()
    key = _key(name)

    if user.is_guest and len(row.blacklist_players or []) >= (
        guest.MAX_BLACKLIST
    ):
        raise guest.limit("players to avoid", guest.MAX_BLACKLIST)

    for existing in row.blacklist_players or []:
        if _key(existing) == key:
            raise _conflict(f"{existing} is already on your avoid list.")

    for position, existing in _forced_pairs(row):
        if _key(existing) == key:
            raise _conflict(
                f"{existing} is a forced pick"
                + (f" ({POSITION_WORDS.get(position, position)})" if position else "")
                + ". Remove them from your forced picks first."
            )

    entry = _pool_by_name().get(key)
    row.blacklist_players = list(row.blacklist_players or []) + [
        entry["name"] if entry else name
    ]
    session.commit()
    session.refresh(row)
    return row


@router.post("/lists/unblacklist", response_model=SettingsOut)
def unblacklist_player(
    payload: PlayerRefIn,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> UserSettings:
    row = _settings_row(session, user)
    key = _key(payload.name)

    kept = [n for n in (row.blacklist_players or []) if _key(n) != key]
    if len(kept) == len(row.blacklist_players or []):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"{payload.name} isn't on your avoid list."
        )

    row.blacklist_players = kept
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
    if user.is_guest:
        raise guest.members_only("Linking an FPL team")

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


@router.post("/manual-squad")
def manual_squad(
    payload: ManualSquadIn,
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> dict:
    """Save a hand-picked squad as a past gameweek.

    The same job as importing from FPL, for anyone who hasn't linked their
    team or whose real side isn't what they want the optimiser to transfer
    from.
    """
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
            "There's no gameweek before the first one to save a squad for.",
        )
    if gameweek >= upcoming:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Gameweek {gameweek} hasn't been played yet. The latest you can "
            f"enter is gameweek {upcoming - 1}.",
        )

    try:
        result = fpl_import.build_from_ids(
            payload.player_ids, payload.starting_ids, gameweek, payload.bank
        )
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
    squad.transfers_made = 0
    squad.penalty_points = 0
    squad.chip = ""
    squad.payload = data
    squad.engine_rows = result["engine_rows"]

    applied = []
    if payload.apply_budget:
        row = _settings_row(session, user)
        row.budget = result["budget"]
        applied.append(f"budget set to £{result['budget']}m")

    session.commit()

    return {
        "gameweek": gameweek,
        "formation": data["formation"],
        "players": len(result["engine_rows"]),
        "squad_value": result["squad_value"],
        "bank": result["bank"],
        "budget": result["budget"],
        "applied": applied,
    }


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
    if user.is_guest:
        raise guest.members_only("Importing a squad from FPL")

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
