"""Running the optimiser forward, one gameweek at a time.

Each week is a normal optimisation, with three things threaded through from
the week before: the squad it produced, the budget it left, and the free
transfers it didn't use. That last one is the whole point — a plan that
ignores rollover would happily suggest one transfer a week when saving two
weeks and making a double move was better.

What it can't do is see the future. Prices move, form turns, players get
injured. Every week beyond the next is a projection built on today's numbers,
and the further out it goes the less it is worth.
"""

from __future__ import annotations

from typing import Any, Callable

from .pipeline import run_optimisation
from .workspace import clear_saved_squads, user_workspace

# Everything a plan writes goes here rather than in the manager's live
# directory: it is speculative, and the engine can't tell a planned squad from
# a real one once both are CSVs in the same place.
SCRATCH = "plan"

# FPL banks up to five unused free transfers.
MAX_FREE_TRANSFERS = 5

# Chips that don't consume a free transfer, and during which any number of
# transfers can be made.
UNLIMITED_CHIPS = {"wildcard", "free_hit"}

CHIP_FIELDS = ("wildcard", "free_hit", "bench_boost", "triple_captain")


class SettingsOverlay:
    """The manager's saved settings, with a few values swapped per week.

    A plain object rather than a copied ORM row: the row belongs to a session
    that closed long before the worker got here, and nothing downstream needs
    it to be a real model.
    """

    def __init__(self, base: Any, **overrides: Any) -> None:
        self._base = base
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        overrides = self.__dict__["_overrides"]
        if name in overrides:
            return overrides[name]
        return getattr(self.__dict__["_base"], name)


def roll_free_transfers(available: int, used: int, chip: str) -> int:
    """How many free transfers the following gameweek starts with.

    One accrues each week, unused ones bank up to five, and a Wildcard or
    Free Hit week consumes none however many moves were made.
    """
    if chip in UNLIMITED_CHIPS:
        remaining = available
    else:
        remaining = max(0, available - used)
    return min(MAX_FREE_TRANSFERS, remaining + 1)


def _week_settings(base: Any, budget: float, free_transfers: int, chip: str,
                   previous_chip: str) -> SettingsOverlay:
    overrides: dict[str, Any] = {
        "budget": round(float(budget), 1),
        "free_transfers": int(free_transfers),
        # A Free Hit side reverts, so the week after one transfers from the
        # squad two gameweeks back. The engine already knows how to do this.
        "free_hit_prev_gw": previous_chip == "free_hit",
    }
    for field in CHIP_FIELDS:
        overrides[field] = chip == field
    return SettingsOverlay(base, **overrides)


def _summarise(squad: dict, gameweek: int, chip: str, free_transfers: int) -> dict:
    """What the timeline needs, without the whole engine payload per week."""
    def slim(player: dict, on_bench: bool) -> dict:
        return {
            "id": player.get("id"),
            "name": player.get("name", ""),
            "position": player.get("position", ""),
            "team": player.get("team", ""),
            "price": player.get("price"),
            "projected_points": player.get("projected_points"),
            "is_captain": bool(player.get("is_captain")),
            "on_bench": on_bench,
        }

    starting = [slim(p, False) for p in squad.get("starting", [])]
    bench = [slim(p, True) for p in squad.get("bench", [])]

    return {
        "gameweek": gameweek,
        "chip": chip,
        "chip_label": squad.get("chip", ""),
        "formation": squad.get("formation", ""),
        "projected_points": squad.get("projected_points"),
        "squad_value": squad.get("squad_value"),
        "bank": squad.get("bank"),
        "free_transfers_available": free_transfers,
        "transfers_made": int(squad.get("transfers_made") or 0),
        "penalty_points": int(squad.get("penalty_points") or 0),
        "points_gain_per_gw": squad.get("points_gain_per_gw"),
        "transfers": squad.get("transfers", {"in": [], "out": []}),
        "starting": starting,
        "bench": bench,
    }


def run_plan(
    *,
    user_id: int,
    season: str,
    start_gameweek: int,
    weeks: int,
    chips: dict[int, str],
    settings_row: Any,
    previous_squads: dict[int, list[dict]],
    on_progress: Callable[[str], None] | None = None,
    on_week: Callable[[int, dict], None] | None = None,
) -> list[dict]:
    """Optimise a run of gameweeks, feeding each week's result into the next."""
    say = on_progress or (lambda _message: None)

    # A fresh slate each time, so last month's plan can't be read as this
    # month's starting point.
    clear_saved_squads(user_workspace(user_id, season, scratch=SCRATCH))

    budget = float(settings_row.budget)
    free_transfers = int(settings_row.free_transfers)
    squads = dict(previous_squads)
    previous_chip = "free_hit" if getattr(settings_row, "free_hit_prev_gw", False) else ""

    results: list[dict] = []

    for offset in range(weeks):
        gameweek = start_gameweek + offset
        chip = chips.get(gameweek, "")

        label = f"Gameweek {gameweek}"
        if chip:
            label += f" ({chip.replace('_', ' ')})"
        say(f"{label} — {offset + 1} of {weeks}")

        overlay = _week_settings(settings_row, budget, free_transfers, chip, previous_chip)
        result = run_optimisation(
            user_id=user_id,
            gameweek=gameweek,
            season=season,
            settings_row=overlay,
            previous_squads=squads,
            # The engine's own milestones would be eight times over; the week
            # heading above is the useful granularity here.
            on_progress=None,
            scratch=SCRATCH,
            # A plan is one line through the season, and every week of it is
            # speculative anyway. Ranking five squads a week would multiply
            # the work for alternatives nobody can act on.
            option_count=0,
        )

        squad = result["squad"]
        week = _summarise(squad, gameweek, chip, free_transfers)
        results.append(week)
        if on_week:
            on_week(offset + 1, week)

        used = week["transfers_made"]
        say(
            f"Gameweek {gameweek}: {used} transfer{'' if used == 1 else 's'}, "
            f"{week['projected_points']:.1f} points projected"
            if isinstance(week["projected_points"], (int, float))
            else f"Gameweek {gameweek}: {used} transfers"
        )

        # Carry the week's outcome into the next one.
        free_transfers = roll_free_transfers(free_transfers, used, chip)
        # A Free Hit squad is handed back at the end of the week, so the next
        # gameweek plans from the side that existed before it.
        if chip != "free_hit":
            squads = {**squads, gameweek: result.get("engine_rows", [])}
        budget = round(float(week["squad_value"] or budget) + float(week["bank"] or 0), 1)
        previous_chip = chip

    return results
