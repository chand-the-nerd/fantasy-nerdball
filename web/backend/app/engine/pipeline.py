"""Runs the fantasy-nerdball optimiser and returns structured results.

Nothing here reimplements the model. It calls the same functions ``main.py``
calls, in the same order, and turns the dataframes at the end into JSON the
browser can draw a pitch from.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
import sys
import time
from typing import Any, Callable

import pandas as pd

from ..config import settings
from . import scoring_cache
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


# Source column, then target key. Several display columns — fixture_diff,
# historic_ppg, reliability, minspg — are created by PointsCalculator on the
# final squad and do NOT exist on the scored pool, so each falls back to the
# raw column it is derived from. The first source that has a value wins.
SCORE_FIELDS = [
    ("id", "id"),
    ("player_code", "code"),
    ("display_name", "name"),
    ("position", "position"),
    ("team", "team"),
    ("now_cost_m", "price"),
    ("fpl_score", "score"),
    ("projected_points", "projected_points"),
    ("form", "form"),
    ("historic_ppg", "historic_ppg"),
    ("avg_ppg_past2", "historic_ppg"),
    ("fixture_diff", "fixture_difficulty"),
    ("diff", "fixture_difficulty"),
    ("fixture_multiplier", "fixture_multiplier"),
    ("reliability", "start_rate"),
    ("current_reliability", "start_rate_fraction"),
    ("minspg", "minutes_per_game"),
    ("minutes", "minutes_total"),
    ("xConsistency", "xg_modifier"),
    ("team_modifier", "team_modifier"),
    ("selected_by_percent", "ownership"),
    ("status", "status"),
    ("news", "news"),
]


def _serialise_scores(
    scored: pd.DataFrame, gameweek: int, per_position: int = 40
) -> list[dict]:
    """Keep the best few dozen per position, not the whole pool.

    Six hundred players is a lot of JSON for a page that only ever shows the
    top of each list, and the tail is players nobody would pick anyway.
    """
    if scored is None or scored.empty or "fpl_score" not in scored.columns:
        return []

    frame = scored.copy()
    if "id" in frame.columns:
        frame = frame.drop_duplicates(subset=["id"])

    kept = []
    for position in ("GK", "DEF", "MID", "FWD"):
        subset = frame[frame["position"] == position]
        if subset.empty:
            continue
        kept.append(subset.nlargest(per_position, "fpl_score"))

    if not kept:
        return []

    targets = []
    for _, target in SCORE_FIELDS:
        if target not in targets:
            targets.append(target)

    rows = []
    for _, row in pd.concat(kept).iterrows():
        record: dict = {key: None for key in targets}
        for source, target in SCORE_FIELDS:
            if source in row.index and record.get(target) is None:
                record[target] = _clean(row[source])
        # Round the raw fixture difficulty the way the display would.
        if record.get("fixture_difficulty") is not None:
            try:
                record["fixture_difficulty"] = round(
                    float(record["fixture_difficulty"]), 1
                )
            except (TypeError, ValueError):
                record["fixture_difficulty"] = None

        # current_reliability is a 0-1 fraction; the display shows a percentage.
        if record.get("start_rate") is None and record.get("start_rate_fraction") is not None:
            try:
                record["start_rate"] = round(
                    float(record["start_rate_fraction"]) * 100
                )
            except (TypeError, ValueError):
                pass
        record.pop("start_rate_fraction", None)

        # minspg is a display column. From the raw frame only the season total
        # is available, so derive the per-gameweek figure here.
        if record.get("minutes_per_game") is None and record.get("minutes_total") is not None:
            try:
                record["minutes_per_game"] = round(
                    float(record["minutes_total"]) / max(1, gameweek - 1)
                )
            except (TypeError, ValueError):
                pass
        record.pop("minutes_total", None)

        # FPL hands ownership back as a string. Coerce once here rather than
        # leaving every consumer to remember to.
        raw_ownership = record.get("ownership")
        if raw_ownership is not None:
            try:
                record["ownership"] = round(float(raw_ownership), 1)
            except (TypeError, ValueError):
                record["ownership"] = None

        rows.append(record)
    return rows


def _option_entry(
    *,
    key: str,
    label: str,
    kind: str,
    starting_display: pd.DataFrame,
    bench_display: pd.DataFrame,
    projected_points: float,
    budget: float,
    player_ids: list,
    prev_squad_ids: list | None,
    free_transfers: int,
    transfer_details: dict | None,
    bench_weight: float = 0.2,
) -> dict:
    """One selectable squad, dressed the same way the main one is.

    Everything the pitch and the scoreline read is precomputed here
    rather than on activation, so switching option is a database write
    and a repaint rather than another four-minute run.
    """
    squad = _serialise_squad(starting_display, bench_display)
    value = float(
        starting_display["now_cost_m"].sum()
        + bench_display["now_cost_m"].sum()
    )

    # The model's own rating of the squad, on the same terms the selector
    # maximised it: the eleven in full, the bench at whatever weight it was
    # given. Reported alongside the projection because they answer different
    # questions and the recommendation follows this one.
    nerdball_score = None
    if "fpl_score" in starting_display.columns:
        try:
            nerdball_score = float(
                starting_display["fpl_score"].sum()
                + float(bench_weight) * bench_display["fpl_score"].sum()
            )
        except (TypeError, ValueError):
            nerdball_score = None

    transfers_made = 0
    if prev_squad_ids is not None:
        transfers_made = len(set(prev_squad_ids) - set(player_ids))

    # Four points an extra transfer, as FPL charges it. The engine's
    # own penalty machinery weighs a hit over a horizon; here the
    # figure is only being reported, so the plain rule is the honest
    # one.
    extra = max(0, transfers_made - max(0, int(free_transfers or 0)))

    squad.update(
        {
            "key": key,
            "label": label,
            "kind": kind,
            "projected_points": _clean(projected_points) or 0.0,
            "nerdball_score": _clean(nerdball_score),
            "squad_value": round(value, 1),
            "bank": round(float(budget) - value, 1),
            "transfers_made": transfers_made,
            "penalty_points": extra * 4,
            "player_ids": [int(pid) for pid in player_ids],
            "transfers": {
                "out": (transfer_details or {}).get("players_out", []),
                "in": (transfer_details or {}).get("players_in", []),
            },
            # Filled in once the whole list is known.
            "recommended": False,
            "differs_by": 0,
        }
    )
    return squad


def _previous_gameweek_option(
    components,
    scored: pd.DataFrame,
    prev_squad_ids: list | None,
    dress: Callable,
    *,
    budget: float,
    free_transfers: int,
    frames: dict[str, tuple],
    bench_weight: float = 0.2,
) -> dict | None:
    """Last week's fifteen, kept whole.

    The one option here that isn't an output of the optimiser — the point of
    it is that the optimiser gets overruled. The eleven are still picked
    properly from those fifteen, so what's on offer is holding the squad, not
    holding the teamsheet.

    Returns None when there is nothing to hold: no previous squad, or too few
    of last week's players still selectable to field one.
    """
    if not prev_squad_ids:
        return None

    try:
        held_starting, held_bench = components[
            "transfer_evaluator"
        ].get_no_transfer_squad_optimised(scored, prev_squad_ids)
    except Exception:
        return None

    if held_starting is None or held_starting.empty:
        return None

    ids = [
        int(pid)
        for pid in pd.concat([held_starting, held_bench])["id"].tolist()
    ]
    starting_display, bench_display, points = dress(held_starting, held_bench)
    frames["previous"] = (starting_display, bench_display)

    return _option_entry(
        key="previous",
        label="Previous Gameweek",
        kind="previous",
        starting_display=starting_display,
        bench_display=bench_display,
        projected_points=points,
        budget=budget,
        player_ids=ids,
        prev_squad_ids=prev_squad_ids,
        free_transfers=free_transfers,
        bench_weight=bench_weight,
        transfer_details=None,
    )


def _held_squad_gain(evaluator, scored, prev_squad_ids, starting_with_transfers):
    """Projected points of the new eleven, minus keeping last week's.

    Gross of any hit: the penalty is reported separately, and showing it
    twice in one sentence would be worse than showing it once.
    """
    if prev_squad_ids is None or starting_with_transfers is None:
        return None
    try:
        held = evaluator.get_no_transfer_squad(scored, prev_squad_ids)
        if held is None or held.empty:
            return None
        return float(
            starting_with_transfers["projected_points"].sum()
            - held["projected_points"].sum()
        )
    except Exception:
        # A missing figure is a hidden line; a raised one is a failed run.
        return None


def _active_chip(config: Any) -> str:
    # Free Hit is checked first: it sets WILDCARD too, and it's the one that
    # was actually played.
    if getattr(config, "FREE_HIT", False):
        return "Free Hit"
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
    scratch: str | None = None,
    option_count: int = 5,
) -> dict:
    """Optimise one gameweek for one manager.

    ``previous_squads`` maps gameweek number to the stored engine rows, so the
    optimiser can see the squad it picked last week and reason about transfers.

    ``option_count`` is how many squads to offer, the recommendation included
    as the first of them. Set it to zero for speculative work — a multi-week
    plan wants one answer per week, not five, and paying for the runners-up
    eight times over would be most of the plan's cost.
    """
    ensure_engine_on_path(settings.engine_dir)

    # Imported here, not at module load: the engine reads relative paths and
    # expects its own directory on sys.path first.
    import main as nerdball  # type: ignore
    from src.utils.token_manager import TokenManager  # type: ignore

    # Each stage is timed, and the breakdown is logged when the run finishes.
    # Guessing which part of a four-minute run is the slow one is how you end
    # up optimising the wrong thing.
    stage: dict = {"name": None, "at": time.monotonic()}
    timings: list[tuple[str, float]] = []

    def say(message: str) -> None:
        now = time.monotonic()
        if stage["name"] is not None:
            timings.append((stage["name"], now - stage["at"]))
        stage["name"] = message
        stage["at"] = now
        if on_progress:
            on_progress(message)

    def say_timings() -> None:
        if stage["name"] is not None:
            timings.append((stage["name"], time.monotonic() - stage["at"]))
        if not timings:
            return
        slowest = sorted(timings, key=lambda pair: pair[1], reverse=True)[:3]
        total = sum(seconds for _name, seconds in timings)
        parts = ", ".join(f"{name.rstrip('.…')} {seconds:.0f}s" for name, seconds in slowest)
        summary = f"Done in {total:.0f}s. Slowest: {parts}."
        if on_progress:
            on_progress(summary)
        # Deliberately stderr: the worker runs the whole optimisation inside
        # redirect_stdout, which swallows the engine's chatter — and would
        # swallow this with it. stderr is left alone and Railway captures it
        # just the same.
        print(
            f"[timings] gw{gameweek} user{user_id} {summary}",
            file=sys.stderr,
            flush=True,
        )

    workspace = user_workspace(user_id, season, scratch=scratch)
    for gw, rows in previous_squads.items():
        seed_previous_squad(workspace, gw, rows)

    config = build_config(gameweek=gameweek, season=season, settings_row=settings_row)
    token_manager = TokenManager(config)

    with run_in_workspace(workspace):
        say(f"Right, gameweek {gameweek}. Let's have a look.")
        components = nerdball.initialise_components(config, token_manager)

        say("Digging out last week's squad")
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

        say("Checking form, fixtures and expected goals")
        # Reused between runs whose scoring settings match — which is
        # every guest run for a gameweek, since their model is fixed.
        players, scored, available_budget = scoring_cache.load(
            config,
            lambda: nerdball.process_player_data(components, config),
        )

        scored_players = _serialise_scores(scored, config.GAMEWEEK)

        say("Sketching out my ideal side")
        theoretical_starting, theoretical_points, theoretical_cost = (
            nerdball.generate_theoretical_squad(components, config, players, available_budget)
        )

        say("Browsing the transfer market")
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

        say("Working out whether a hit is worth it")
        should_transfer, evaluator_analysis = nerdball.evaluate_transfer_strategy(
            components, config, scored, prev_squad_ids, starting_wt, transfers_made
        )
        if evaluator_analysis and "points_improvement_ppgw" in evaluator_analysis:
            transfer_analysis = evaluator_analysis
        elif not transfer_analysis:
            transfer_analysis = evaluator_analysis or {}

        # The two routes above measure different things. With hits enabled the
        # evaluator reports the gain of the chosen scenario over the
        # free-transfer baseline, which is zero whenever no hit was taken — so
        # the figure was usually missing on exactly the weeks a transfer was
        # being recommended. This is the like-for-like number instead: what the
        # new eleven projects, against keeping last week's.
        if should_transfer and transfers_made and "no_transfer_ppgw" not in (
            transfer_analysis or {}
        ):
            gain = _held_squad_gain(evaluator, scored, prev_squad_ids, starting_wt)
            if gain is not None:
                transfer_analysis = {
                    **(transfer_analysis or {}),
                    "points_improvement_ppgw": gain,
                }

        starting, bench = nerdball.finalise_squad_selection(
            components, config, should_transfer, starting_wt, bench_wt,
            scored, prev_squad_ids, transfers_made, penalty_points,
        )

        calculator = components["points_calculator"]
        display_utils = components["display_utils"]
        fixtures_reachable = {"ok": True}

        def dress(raw_starting, raw_bench):
            """Turn a solved squad into the frames the UI reads.

            Every option goes through exactly this, so the pitch, the
            armband and the projected total mean the same thing on all of
            them. Pulled out of the main flow rather than duplicated: an
            option dressed even slightly differently would be comparing
            itself against the recommendation on unequal terms.
            """
            picked, benched = nerdball.optimise_starting_xi(
                components, config, raw_starting, raw_bench, players,
                available_budget,
            )

            if fixtures_reachable["ok"]:
                try:
                    manager = components["fixture_manager"]
                    picked = manager.add_next_fixture(picked, config.GAMEWEEK)
                    benched = manager.add_next_fixture(benched, config.GAMEWEEK)
                except Exception:
                    # Fixtures are cosmetic; never fail the run over them.
                    # Flagged once so five options don't say it five times.
                    fixtures_reachable["ok"] = False
                    say("Couldn't reach the fixture list, carrying on without it")

            picked = calculator.add_points_analysis_to_display(picked)
            benched = calculator.add_points_analysis_to_display(benched)
            picked = display_utils.sort_and_format_starting_xi(picked)
            picked = display_utils.apply_captain_and_vice(picked)
            benched = display_utils.sort_and_format_bench(benched)

            points = nerdball.calculate_your_points(picked, benched, token_manager)
            return picked, benched, points

        say("Deciding who starts and who sits")
        transfer_details = nerdball.extract_transfer_details(
            prev_squad_ids, starting_wt, bench_wt, players
        )

        say("Handing out the armband")
        recommended_ids = [
            int(pid) for pid in pd.concat([starting, bench])["id"].tolist()
        ]
        starting_display, bench_display, your_points = dress(starting, bench)

        squad_value = float(
            starting_display["now_cost_m"].sum() + bench_display["now_cost_m"].sum()
        )

        # The allowance the options are ranked under. Using the config's free
        # transfers alone would rule out the recommendation itself whenever it
        # took a hit, leaving option one showing something the engine didn't
        # pick.
        allowance = max(int(config.FREE_TRANSFERS or 0), int(transfers_made or 0))

        options: list[dict] = []
        option_rows: dict[str, list] = {}
        option_frames: dict[str, tuple] = {}

        if option_count > 0:
            say("Lining up the alternatives")
            # How far apart the options have to be. Counting players alone
            # is a weak measure — a fourth-choice keeper counts the same as
            # a captain — so the default also requires a starter to leave.
            # All three are settable per manager through the overrides blob.
            #
            # None of it can beat the transfer allowance: with one free
            # transfer every legal squad keeps fourteen of last week's, so
            # two options can be at most two players apart whatever is asked
            # for here. The selector clamps to that rather than going
            # infeasible.
            spacing = int(getattr(config, "OPTION_MIN_CHANGES", 1))
            starter_spacing = int(
                getattr(config, "OPTION_MIN_STARTER_CHANGES", 1)
            )
            spend_spacing = float(
                getattr(config, "OPTION_MIN_SPEND_CHANGE", 0.0)
            )

            # Holding is a recommendation the optimiser can make, and it
            # has its own button, so the numbered options are always squads
            # that change something. Judged on the fifteen rather than on
            # should_transfer: a recommendation that matches last week's
            # squad is a hold whatever flag came back with it.
            holding = (
                prev_squad_ids is not None
                and set(recommended_ids) == set(prev_squad_ids)
            )

            def squad_ids(starting_frame, bench_frame):
                return [
                    int(pid)
                    for pid in pd.concat(
                        [starting_frame, bench_frame]
                    )["id"].tolist()
                ]

            picked = []
            seen = set()

            if prev_squad_ids:
                seen.add(frozenset(int(pid) for pid in prev_squad_ids))

            def take(ids, raw, dressed=None):
                """Add a squad to the list unless it is already on it."""
                if len(picked) >= option_count or frozenset(ids) in seen:
                    return False
                seen.add(frozenset(ids))
                picked.append(
                    {"ids": ids, "raw": raw, "dressed": dressed}
                )
                return True

            if not holding:
                take(
                    recommended_ids,
                    (starting, bench),
                    (starting_display, bench_display, your_points),
                )

            # Options never take a hit of their own. A -4 is worth paying
            # only when the gain covers it, which the ladder has already
            # judged; manufacturing one to fill a slot offers a squad that
            # starts four points down. If the recommendation itself took a
            # hit it is still option one, because that one was earned.
            spendable = min(
                max(0, int(config.FREE_TRANSFERS or 0)), option_count
            )

            # One squad at each number of transfers, fewest first. The
            # ladder already solved these on its way to deciding how many to
            # make, and used to throw all but the winner away.
            for rung in sorted(
                (
                    rung
                    for rung in (evaluator._last_ladder or [])
                    if 0 < rung.get("actual_transfers", 0) <= spendable
                    and rung.get("starting") is not None
                ),
                key=lambda rung: rung["actual_transfers"],
            ):
                take(
                    squad_ids(rung["starting"], rung["bench"]),
                    (rung["starting"], rung["bench"]),
                )

            # Slots left over go to runners-up at the counts already on
            # offer, rather than to a bigger move nobody wants. One free
            # transfer means five different single transfers, which is the
            # useful thing to see when only one is available.
            if len(picked) < option_count and spendable > 0:
                pools = {}
                blocked = [sorted(ids) for ids in seen]

                for count in range(1, spendable + 1):
                    pools[count] = list(
                        nerdball.generate_squad_options(
                            components, config, scored, prev_squad_ids,
                            available_budget,
                            count=option_count,
                            free_transfers=count,
                            min_changes=spacing,
                            min_starter_changes=starter_spacing,
                            min_spend_change=spend_spacing,
                            exclude_squads=blocked or None,
                        )
                    )

                # Round robin, so the spare slots spread across the counts
                # instead of stacking five alternatives onto the smallest.
                while len(picked) < option_count and any(pools.values()):
                    added = False
                    for count in sorted(pools):
                        if len(picked) >= option_count:
                            break
                        while pools[count]:
                            alt_starting, alt_bench = pools[count].pop(0)
                            if take(
                                squad_ids(alt_starting, alt_bench),
                                (alt_starting, alt_bench),
                            ):
                                added = True
                                break
                    if not added:
                        break

            # No previous squad, or a wildcard: there is no transfer count
            # to vary, so the ranking supplies the lot.
            if len(picked) < option_count and not spendable:
                for alt_starting, alt_bench in nerdball.generate_squad_options(
                    components, config, scored, prev_squad_ids,
                    available_budget,
                    count=option_count + 1,
                    free_transfers=allowance,
                    min_changes=spacing,
                    min_starter_changes=starter_spacing,
                    min_spend_change=spend_spacing,
                    exclude_squads=[sorted(ids) for ids in seen] or None,
                ):
                    take(
                        squad_ids(alt_starting, alt_bench),
                        (alt_starting, alt_bench),
                    )

            for rank, item in enumerate(picked, start=1):
                key = f"option-{rank}"
                if item["dressed"] is None:
                    item["dressed"] = dress(*item["raw"])
                item_starting, item_bench, item_points = item["dressed"]
                option_frames[key] = (item_starting, item_bench)
                options.append(
                    _option_entry(
                        key=key,
                        label=f"Option {rank}",
                        kind="alternative",
                        starting_display=item_starting,
                        bench_display=item_bench,
                        projected_points=item_points,
                        budget=float(config.BUDGET),
                        player_ids=item["ids"],
                        prev_squad_ids=prev_squad_ids,
                        free_transfers=config.FREE_TRANSFERS,
                        bench_weight=getattr(config, "BENCH_WEIGHT", 0.2),
                        # Recomputed per option rather than reusing the run's
                        # transfer_details, which describe only the squad the
                        # optimiser proposed.
                        transfer_details=nerdball.extract_transfer_details(
                            prev_squad_ids, item["raw"][0], item["raw"][1],
                            players,
                        ),
                    )
                )

            if not holding and options:
                options[0]["recommended"] = True

            held = _previous_gameweek_option(
                components, scored, prev_squad_ids, dress,
                budget=float(config.BUDGET),
                free_transfers=config.FREE_TRANSFERS,
                frames=option_frames,
                bench_weight=getattr(config, "BENCH_WEIGHT", 0.2),
            )
            if held is not None:
                if holding:
                    held["recommended"] = True
                options.append(held)

            # Everything is measured against the recommendation, since that
            # is the squad a manager is deciding whether to depart from.
            reference = next(
                (entry for entry in options if entry["recommended"]),
                options[0] if options else None,
            )
            if reference is not None:
                for entry in options:
                    entry["differs_by"] = len(
                        set(reference["player_ids"]) - set(entry["player_ids"])
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

        # Each option needs its own engine rows. They are what next week's run
        # reads back as "the squad you had", so an option activated without
        # them would be shown on the pitch and then quietly transferred from
        # the wrong fifteen seven days later.
        for entry in options:
            frames = option_frames.get(entry["key"])
            if frames is None:
                continue
            FileUtils.save_squad_data(config.GAMEWEEK, frames[0], frames[1])
            option_rows[entry["key"]] = read_saved_squad(workspace, config.GAMEWEEK)

        # Written last so what is left on disk is the squad that is active.
        FileUtils.save_squad_data(config.GAMEWEEK, starting_display, bench_display)
        engine_rows = read_saved_squad(workspace, config.GAMEWEEK)

    say("Finalising my thoughts")

    squad = _serialise_squad(starting_display, bench_display)
    squad.update(
        {
            "gameweek": gameweek,
            "season": season,
            "projected_points": _clean(your_points) or 0.0,
            "squad_value": round(squad_value, 1),
            "bank": round(float(config.BUDGET) - squad_value, 1),
            "chip": _active_chip(config),
            # Every squad on offer this week, the recommendation first. The
            # payload's own starting/bench stay as the active one's, so
            # anything reading a squad without knowing about options — the
            # planner, the performance chart, an older client — sees exactly
            # what it saw before.
            "options": options,
            # The recommendation is what opens, wherever it sits in the list.
            # On a week the optimiser wants to hold, that is the previous
            # gameweek rather than any of the numbered options.
            "active_option": next(
                (entry["key"] for entry in options if entry["recommended"]),
                options[0]["key"] if options else "",
            ),
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
            # What the evaluator actually weighed up, so the recommendation
            # can be read as reasoning rather than an oracle.
            "explored": [
                {
                    "player_out": s.get("unavailable_player"),
                    "position": s.get("position"),
                    "out_score": _clean(s.get("unavailable_score")),
                    "replacement": s.get("best_substitute"),
                    "replacement_score": _clean(s.get("substitute_score")),
                    "points_lost": _clean(s.get("score_loss")),
                    "verdict": s.get("recommendation", ""),
                }
                for s in (transfer_analysis or {}).get("scenarios", []) or []
            ],
        }
    )
    say_timings()
    return {
        "squad": squad,
        "engine_rows": engine_rows,
        "option_rows": option_rows,
        "scored_players": scored_players,
        "look_ahead": int(getattr(config, "FIRST_N_GAMEWEEKS", 1)),
    }
