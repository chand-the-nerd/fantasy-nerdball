"""Builds a fantasy-nerdball Config object from a manager's saved settings.

The optimiser expects a module-level ``Config`` class edited by hand. Here it
is constructed per run instead, so five managers can hold five different sets
of preferences against one deployment.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

# Settings a manager may override from the web UI. Anything outside this set
# is ignored, so a stray key in the overrides blob can't reach into the engine.
ALLOWED_OVERRIDES = {
    "BUDGET",
    "FREE_TRANSFERS",
    "ACCEPT_TRANSFER_PENALTY",
    "EXCLUDE_UNAVAILABLE",
    "WILDCARD",
    "FREE_HIT_PREV_GW",
    "BENCH_BOOST",
    "TRIPLE_CAPTAIN",
    "MIN_TRANSFER_VALUE",
    # How far apart the squad options on the Squad page must be. Player
    # count, how many of them were starting, and how much money changes
    # hands. All three trade a little of the true ranking for options that
    # differ enough to be worth looking at.
    "OPTION_MIN_CHANGES",
    "OPTION_MIN_STARTER_CHANGES",
    "OPTION_MIN_SPEND_CHANGE",
    "FIRST_N_GAMEWEEKS",
    "FIXTURE_DECAY_FACTOR",
    "USE_ML_WEIGHTS",
    "POSITION_SCORING_WEIGHTS",
    "EARLY_SEASON_PENALTY_INITIAL",
    "EARLY_SEASON_DECAY_FACTOR",
    "EARLY_SEASON_PENALTY_GAMEWEEKS",
    "TEAM_MODIFIER_SCALE",
    "XCONSISTENCY_SCALE",
    "FORM_CONSISTENCY_SCALE",
    "TRANSFER_SHRINK",
    "TRANSFER_RECOVERY_GWS",
    "PROMOTED_PENALTY",
    "BENCH_GK_MAX_COST",
    "BENCH_WEIGHT",
    "PENALISE_ZERO_FORM",
    "ZERO_FORM_PENALTY",
    "MAX_PER_TEAM",
}


def _base_config_class():
    """Load the engine's Config class from the cloned repo."""
    for module_name in ("config", "config_example"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        cls = getattr(module, "Config", None)
        if cls is not None:
            return cls
    raise RuntimeError(
        "Could not import Config from the fantasy-nerdball engine. Check "
        "NERDBALL_ENGINE_DIR and that config.py exists in the clone."
    )


def build_config(
    *,
    gameweek: int,
    season: str,
    settings_row: Any,
) -> Any:
    """Return an instantiated engine Config for this run."""
    base = _base_config_class()

    class RunConfig(base):  # type: ignore[misc, valid-type]
        pass

    RunConfig.GAMEWEEK = int(gameweek)
    RunConfig.CURRENT_SEASON = season
    # The engine never prompts or prints a dashboard here; the web layer
    # renders the result. Clean mode also skips the interactive prompts.
    RunConfig.GRANULAR_OUTPUT = False
    RunConfig.DETAILED_CALCULATION = False
    # A stale hardcoded club list breaks startup validation every August.
    RunConfig.AUTO_DETECT_PROMOTED_TEAMS = True
    RunConfig.VALIDATE_TEAM_NAMES = False

    if settings_row is not None:
        RunConfig.BUDGET = float(settings_row.budget)
        RunConfig.FREE_TRANSFERS = int(settings_row.free_transfers)
        RunConfig.ACCEPT_TRANSFER_PENALTY = bool(settings_row.accept_transfer_penalty)
        RunConfig.EXCLUDE_UNAVAILABLE = bool(settings_row.exclude_unavailable)
        # A Free Hit is a Wildcard for one week only: transfers are free and
        # unlimited, but the squad reverts afterwards, so planning beyond this
        # gameweek would be planning for a side you won't have.
        free_hit = bool(getattr(settings_row, "free_hit", False))
        RunConfig.FREE_HIT = free_hit
        RunConfig.WILDCARD = bool(settings_row.wildcard) or free_hit
        RunConfig.FREE_HIT_PREV_GW = bool(settings_row.free_hit_prev_gw)
        RunConfig.BENCH_BOOST = bool(settings_row.bench_boost)
        RunConfig.TRIPLE_CAPTAIN = bool(settings_row.triple_captain)
        # A Bench Boost scores all fifteen, so on that week the bench is
        # not a reserve to be traded down — it is half the return. Picking
        # it at a fifth of a starter's weight would spend the chip on four
        # players chosen to be cheap.
        RunConfig.BENCH_WEIGHT = (
            1.0 if RunConfig.BENCH_BOOST
            else float(getattr(settings_row, "bench_weight", 0.2))
        )
        RunConfig.USE_ML_WEIGHTS = bool(settings_row.use_ml_weights)
        RunConfig.FIRST_N_GAMEWEEKS = (
            1 if free_hit else int(settings_row.first_n_gameweeks)
        )
        # A threshold is a price on a transfer, and on a Wildcard or Free
        # Hit transfers are free and unlimited. Holding a move back because
        # it only gains a point costs nothing to make and gains nothing to
        # refuse, so the strategy setting is ignored for the week.
        RunConfig.MIN_TRANSFER_VALUE = (
            0.0 if RunConfig.WILDCARD
            else float(settings_row.min_transfer_value)
        )

        if settings_row.team_modifiers:
            merged = dict(base.TEAM_MODIFIERS)
            for club, value in settings_row.team_modifiers.items():
                merged[club] = float(value)
            RunConfig.TEAM_MODIFIERS = merged

        forced = settings_row.forced_selections or {}
        RunConfig.FORCED_SELECTIONS = {
            pos: [str(n).lower() for n in forced.get(pos, []) or []]
            for pos in ("GK", "DEF", "MID", "FWD")
        }
        RunConfig.BLACKLIST_PLAYERS = [
            str(n).lower() for n in (settings_row.blacklist_players or [])
        ]

        for key, value in (settings_row.overrides or {}).items():
            if key in ALLOWED_OVERRIDES:
                setattr(RunConfig, key, value)

    # The engine raises if the current season also appears in PAST_SEASONS.
    RunConfig.PAST_SEASONS = [s for s in base.PAST_SEASONS if s != season]
    weights = list(base.HISTORIC_SEASON_WEIGHTS)[: len(RunConfig.PAST_SEASONS)]
    if weights:
        total = sum(weights)
        RunConfig.HISTORIC_SEASON_WEIGHTS = [w / total for w in weights]

    return RunConfig()


def ensure_engine_on_path(engine_dir) -> None:
    """Put the cloned optimiser at the front of sys.path exactly once."""
    path = str(engine_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
