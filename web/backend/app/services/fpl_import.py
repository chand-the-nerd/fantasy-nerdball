"""Imports a manager's real FPL side and reads their budget off the API.

The optimiser reasons about transfers by comparing this week's best squad
against the one it saved last week. Starting mid-season, there is no saved
squad, so it treats you as a new team with a free hand. Pulling the real side
in as the previous gameweek gives it the right starting point.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from ..config import settings
from . import fpl

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Chips that leave your saved free transfers untouched, and during which
# transfers made don't consume any.
FT_NEUTRAL_CHIPS = {"wildcard", "freehit"}

# FPL banks up to five unused free transfers.
MAX_FREE_TRANSFERS = int(os.getenv("FPL_MAX_FREE_TRANSFERS", "5"))


class SquadImportError(Exception):
    """Raised with a message meant for the person, not a stack trace."""


def fetch_picks(entry_id: int, gameweek: int) -> dict:
    response = requests.get(
        f"{fpl.BASE_URL}/entry/{entry_id}/event/{gameweek}/picks/",
        timeout=fpl.TIMEOUT,
    )
    if response.status_code == 404:
        raise SquadImportError(
            f"FPL has no squad for team {entry_id} in gameweek {gameweek}. "
            "That gameweek may not have started yet."
        )
    response.raise_for_status()
    return response.json()


def fetch_history(entry_id: int) -> dict:
    response = requests.get(
        f"{fpl.BASE_URL}/entry/{entry_id}/history/", timeout=fpl.TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def infer_free_transfers(history: dict, upcoming_gameweek: int) -> tuple[int, str]:
    """Work out how many free transfers should be available.

    The API never states this directly, so it is reconstructed from the
    transfers made each week: one accrues per gameweek, unused ones bank up to
    five, and a Wildcard or Free Hit week consumes none. It is an inference
    rather than a fact, so the caller should let the manager correct it.
    """
    chips_by_event = {
        int(chip["event"]): str(chip.get("name", "")).lower()
        for chip in history.get("chips", [])
    }

    free_transfers = 0
    for entry in history.get("current", []):
        gameweek = int(entry["event"])
        if gameweek >= upcoming_gameweek:
            break

        chip = chips_by_event.get(gameweek, "")
        used = 0 if (gameweek == 1 or chip in FT_NEUTRAL_CHIPS) else int(
            entry.get("event_transfers") or 0
        )
        free_transfers = min(MAX_FREE_TRANSFERS, max(0, free_transfers - used) + 1)

    note = (
        f"Worked out from your transfer history, capped at "
        f"{MAX_FREE_TRANSFERS}. Correct it here if it looks wrong."
    )
    return free_transfers, note


def build_squad(entry_id: int, gameweek: int) -> dict:
    """Turn an FPL side into the squad shape this app stores."""
    picks_data = fetch_picks(entry_id, gameweek)
    bootstrap = fpl.bootstrap()

    elements = {int(e["id"]): e for e in bootstrap.get("elements", [])}
    teams = {int(t["id"]): t["name"] for t in bootstrap.get("teams", [])}

    starting: list[dict] = []
    bench: list[dict] = []
    engine_rows: list[dict] = []

    for pick in picks_data.get("picks", []):
        element = elements.get(int(pick["element"]))
        if element is None:
            continue

        position = POSITIONS.get(element.get("element_type"), "MID")
        team = teams.get(element.get("team"), "")
        price = round(element.get("now_cost", 0) / 10, 1)
        slot = int(pick.get("position", 0))
        on_bench = slot > 11
        role = "Bench" if on_bench else "Starting XI"

        player = {
            "id": element["id"],
            "code": element.get("code"),
            "name": element["web_name"],
            "position": position,
            "team": team,
            "team_short": "",
            "price": price,
            # Nothing is projected here: these are the real picks, not the
            # model's. The next run fills in projections.
            "projected_points": 0.0,
            "form": float(element.get("form") or 0),
            "historic_ppg": None,
            "fixture_difficulty": None,
            "start_rate": None,
            "minutes_per_game": None,
            "xg_modifier": None,
            "next_opponent": "",
            "venue": "",
            "status": element.get("status", "a"),
            "news": element.get("news", "") or "",
            "is_captain": bool(pick.get("is_captain")),
            "is_vice": bool(pick.get("is_vice_captain")),
            "is_double_gameweek": False,
            "on_bench": on_bench,
            "bench_order": (slot - 11) if on_bench else None,
        }
        (bench if on_bench else starting).append(player)

        # The optimiser matches a saved squad on player_code first, which is
        # stable across seasons, so an imported squad matches exactly.
        engine_rows.append(
            {
                "id": element["id"],
                "player_code": element.get("code"),
                "display_name": element["web_name"],
                "position": position,
                "team": team,
                "now_cost_m": price,
                "projected_points": 0.0,
                "squad_role": role,
            }
        )

    if len(engine_rows) < 15:
        raise SquadImportError(
            f"FPL returned only {len(engine_rows)} players for gameweek "
            f"{gameweek}. The import was stopped rather than save a partial squad."
        )

    entry_history = picks_data.get("entry_history", {}) or {}
    squad_value = round(float(entry_history.get("value", 0)) / 10, 1)
    bank = round(float(entry_history.get("bank", 0)) / 10, 1)
    active_chip = (picks_data.get("active_chip") or "").lower()

    counts = {"DEF": 0, "MID": 0, "FWD": 0}
    for player in starting:
        if player["position"] in counts:
            counts[player["position"]] += 1

    bench.sort(key=lambda p: p["bench_order"] or 0)

    payload = {
        "starting": starting,
        "bench": bench,
        "formation": f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}",
        "gameweek": gameweek,
        "season": settings.current_season,
        "projected_points": 0.0,
        "squad_value": squad_value,
        "bank": bank,
        "chip": active_chip.replace("freehit", "Free Hit").replace(
            "wildcard", "Wildcard"
        ).replace("bboost", "Bench Boost").replace("3xc", "Triple Captain"),
        "transfers_made": int(entry_history.get("event_transfers") or 0),
        "penalty_points": int(entry_history.get("event_transfers_cost") or 0),
        "made_transfers": bool(entry_history.get("event_transfers")),
        "transfer_reason": "",
        "points_gain_per_gw": None,
        "transfers": {"out": [], "in": []},
        "model_xi": None,
        "imported": True,
    }

    return {
        "payload": payload,
        "engine_rows": engine_rows,
        "squad_value": squad_value,
        "bank": bank,
        "budget": round(squad_value + bank, 1),
        "active_chip": active_chip,
        "actual_points": entry_history.get("points"),
        "overall_rank": entry_history.get("overall_rank"),
    }


def import_summary(entry_id: int, gameweek: int, upcoming: int) -> dict[str, Any]:
    """Everything the endpoint needs, in one place."""
    squad = build_squad(entry_id, gameweek)
    history = fetch_history(entry_id)
    free_transfers, note = infer_free_transfers(history, upcoming)

    squad["free_transfers"] = free_transfers
    squad["free_transfers_note"] = note
    # A Free Hit side reverts the following week, so the optimiser has to be
    # told to look one gameweek further back.
    squad["free_hit_used"] = squad["active_chip"] == "freehit"
    return squad
