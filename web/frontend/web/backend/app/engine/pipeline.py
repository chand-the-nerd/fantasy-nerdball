"""Runs the fantasy-nerdball optimiser and returns structured results.

Nothing here reimplements the model. It calls the same functions ``main.py``
calls, in the same order, and turns the dataframes at the end into JSON the
browser can draw a pitch from.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..config import settings
from .runtime_config import build_config, ensure_engine_on_path
from .workspace import read_saved_squad, run_in_workspace, seed_previous_squad, user_workspace

POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
CAPTAIN_MARK = re.compile(r"\s*\((C|V)\)\s*$")
DGW_MARK = re.compile(r"\s*\*+\s*$")


def _strip_markers(raw: str) -> tuple[str, bool, bool, bool]:
    """Pull the armband and double-gameweek marks back off a display name.

    The engine appends both, and which lands last depends on the order the
    display helpers ran in, so peel them off in a loop rather than assuming.
    """
    name = raw.strip()
    captain = vice = double = False

    while True:
        armband = CAPTAIN_MARK.search(name)
        if armband:
            captain = captain or armband.group(1) == "C"
            vice = vice or armband.group(1) == "V"
            name = CAPTAIN_MARK.sub("", name)
            continue
        if DGW_MARK.search(name):
            double = True
            name = DGW_MARK.sub("", name)
            continue
        break

    return name.strip(), captain, vice, double


def _clean(value: Any) -> Any:
    """Make numpy/pandas values JSON-safe."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else round(value, 3)
    if hasattr(value, "item"):
        try:
            return _clean(value.item())
        except (AttributeError, ValueError):
            return str(value)
    return str(value)


def _player_row(row: pd.Series, *, on_bench: bool, bench_order: int | None) -> dict:
    raw_name = str(row.get("display_name", ""))
    name, is_captain, is_vice, marked_double = _strip_markers(raw_name)
    is_double = marked_double or bool(row.get("has_dgw_next", False))

    def get(*keys, default=None):
        for key in keys:
            if key in row.index:
                cleaned = _clean(row[key])
                if cleaned is not None:
                    return cleaned
        return default

    return {
        "id": get("id"),
        "code": get("player_code"),
        "name": name,
        "position": get("position", default="MID"),
        "team": get("team", default=""),
        "team_short": get("team_short", "short_name", default=""),
        "price": get("now_cost_m", default=0.0),
        "projected_points": get("proj_pts", "projected_points", default=0.0),
        "form": get("form", default=0.0),
        "historic_ppg": get("historic_ppg", "avg_ppg_past2", default=None),
        "fixture_difficulty": get("fixture_diff", "diff", default=None),
        "start_rate": get("reliability", default=None),
        "minutes_per_game": get("minspg", default=None),
        "xg_modifier": get("xConsistency", default=None),
        "next_opponent": get("next_opponent", default=""),
        "venue": get("venue", default=""),
        "status": get("status", default="a"),
        "news": get("news", default=""),
        "is_captain": is_captain,
        "is_vice": is_vice,
        "is_double_gameweek": is_double,
        "on_bench": on_bench,
        "bench_order": bench_order,
    }


def _formation(starting: list[dict]) -> str:
    counts = {"DEF": 0, "MID": 0, "FWD": 0}
    for player in starting:
        if player["position"] in counts:
            counts[player["position"]] += 1
    return f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}"


def _serialise_squad(starting_df: pd.DataFrame, bench_df: pd.DataFrame) -> dict:
    starting = [_player_row(row, on_bench=False, bench_order=None)
                for _, row in starting_df.iterrows()]
    starting.sort(key=lambda p: (POSITION_ORDER.get(p["position"], 9),
                                 -(p["projected_points"] or 0)))

    bench = []
    for order, (_, row) in enumerate(bench_df.iterrows(), start=1):
        bench.append(_player_row(row, on_bench=True, bench_order=order))

    return {"starting": starting, "bench": bench, "formation": _formation(starting)}


def _active_chip(config: Any) -> str:
    if getattr(config, "WILDCARD", False):
        return "Wildcard"
    if getattr(config, "BENCH_BOOST", False):
        return "Bench Boost"
    if getattr(config, "TRIPLE_CAPTAIN", False):
        return "Triple Captain"
    if getattr(config, "FREE_HIT_PREV_GW", False):
        return "Free Hit (last week)"
    return ""


def run_optimisation(
    *,
    user_id: int,
    gameweek: int,
    season: str,
    settings_row: Any,
    previous_squads: dict[int, list[dict]],
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Optimise one gameweek for one manager.

    ``previous_squads`` maps gameweek number to the stored engine rows, so the
    optimiser can see the squad it picked last week and reason about transfers.
    """
    ensure_engine_on_path(settings.engine_dir)

    # Imported here, not at module load: the engine reads relative paths and
    # expects its own directory on sys.path first.
    import main as nerdball  # type: ignore
    from src.utils.token_manager import TokenManager  # type: ignore

    def say(message: str) -> None:
        if on_progress:
            on_progress(message)

    workspace = user_workspace(user_id, season)
    for gw, rows in previous_squads.items():
        seed_previous_squad(workspace, gw, rows)

    config = build_config(gameweek=gameweek, season=season, settings_row=settings_row)
    token_manager = TokenManager(config)

    with run_in_workspace(workspace):
        say(f"Setting up gameweek {gameweek}")
        components = nerdball.initialise_components(config, token_manager)

        say("Loading last week's squad")
        prev_squad_gameweek = token_manager.get_previous_squad_gameweek()
        from src.utils.file_utils import FileUtils  # type: ignore

        prev_squad = FileUtils.load_previous_squad_from_gameweek(
            config.GAMEWEEK, prev_squad_gameweek
        )
        prev_squad_ids = None
        if prev_squad is not None:
            processor = components["player_processor"]
            prev_squad_ids = processor.match_players_to_current(
                prev_squad, processor.fetch_current_players()
            )

        say("Reading form, fixtures and expected goals")
        players, scored, available_budget = nerdball.process_player_data(components, config)

        say("Picking the model's ideal side for the week")
        theoretical_starting, theoretical_points, theoretical_cost = (
            nerdball.generate_theoretical_squad(components, config, players, available_budget)
        )

        say("Searching the transfer market")
        (
            starting_wt,
            bench_wt,
            transfers_made,
            penalty_points,
        ) = nerdball.optimise_squad(
            components, config, scored, prev_squad_ids, available_budget
        )

        transfer_analysis: dict = {}
        evaluator = components["transfer_evaluator"]
        if config.ACCEPT_TRANSFER_PENALTY and hasattr(evaluator, "_last_best_scenario"):
            scenario = evaluator._last_best_scenario or {}
            transfer_analysis = {
                "reason": "Transfers not worth it",
                "points_improvement_ppgw": scenario.get("points_improvement_ppgw", 0),
                "gameweeks_analysed": scenario.get(
                    "gameweeks_analysed", config.FIRST_N_GAMEWEEKS
                ),
            }

        say("Weighing the transfers against a hit")
        should_transfer, evaluator_analysis = nerdball.evaluate_transfer_strategy(
            components, config, scored, prev_squad_ids, starting_wt, transfers_made
        )
        if evaluator_analysis and "points_improvement_ppgw" in evaluator_analysis:
            transfer_analysis = evaluator_analysis
        elif not transfer_analysis:
            transfer_analysis = evaluator_analysis or {}

        starting, bench = nerdball.finalise_squad_selection(
            components, config, should_transfer, starting_wt, bench_wt,
            scored, prev_squad_ids, transfers_made, penalty_points,
        )

        say("Choosing the starting eleven")
        starting, bench = nerdball.optimise_starting_xi(
            components, config, starting, bench, players, available_budget
        )

        transfer_details = nerdball.extract_transfer_details(
            prev_squad_ids, starting_wt, bench_wt, players
        )

        say("Adding fixtures and picking a captain")
        try:
            starting = components["fixture_manager"].add_next_fixture(starting, config.GAMEWEEK)
            bench = components["fixture_manager"].add_next_fixture(bench, config.GAMEWEEK)
        except Exception as error:  # fixtures are cosmetic; never fail the run
            say(f"Fixture lookup skipped: {error}")

        calculator = components["points_calculator"]
        display_utils = components["display_utils"]

        starting_display = calculator.add_points_analysis_to_display(starting)
        bench_display = calculator.add_points_analysis_to_display(bench)
        starting_display = display_utils.sort_and_format_starting_xi(starting_display)
        starting_display = display_utils.apply_captain_and_vice(starting_display)
        bench_display = display_utils.sort_and_format_bench(bench_display)

        your_points = nerdball.calculate_your_points(
            starting_display, bench_display, token_manager
        )
        squad_value = float(
            starting_display["now_cost_m"].sum() + bench_display["now_cost_m"].sum()
        )

        theoretical: dict | None = None
        if theoretical_starting is not None and not theoretical_starting.empty:
            theoretical = {
                "projected_points": _clean(theoretical_points),
                "cost": _clean(theoretical_cost),
                "starting": [
                    _player_row(row, on_bench=False, bench_order=None)
                    for _, row in theoretical_starting.iterrows()
                ],
            }

        FileUtils.save_squad_data(config.GAMEWEEK, starting_display, bench_display)
        engine_rows = read_saved_squad(workspace, config.GAMEWEEK)

    squad = _serialise_squad(starting_display, bench_display)
    squad.update(
        {
            "gameweek": gameweek,
            "season": season,
            "projected_points": _clean(your_points) or 0.0,
            "squad_value": round(squad_value, 1),
            "bank": round(float(config.BUDGET) - squad_value, 1),
            "chip": _active_chip(config),
            "transfers_made": int(transfers_made) if should_transfer else 0,
            "penalty_points": int(penalty_points) if should_transfer else 0,
            "made_transfers": bool(should_transfer),
            "transfer_reason": (transfer_analysis or {}).get("reason", ""),
            "points_gain_per_gw": _clean((transfer_analysis or {}).get("points_improvement_ppgw")),
            "transfers": {
                "out": (transfer_details or {}).get("players_out", []),
                "in": (transfer_details or {}).get("players_in", []),
            }
            if should_transfer and transfer_details
            else {"out": [], "in": []},
            "model_xi": theoretical,
        }
    )
    return {"squad": squad, "engine_rows": engine_rows}
