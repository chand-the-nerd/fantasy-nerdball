"""
Configuration settings for Fantasy Nerdball FPL optimisation tool.
Edit this file to customise your optimisation preferences.
"""


class Config:
    """Configuration class containing all settings for the FPL
    optimisation."""

    # === BASIC SETTINGS ===
    GAMEWEEK = 1
    BUDGET = 100
    FREE_TRANSFERS = 1
    ACCEPT_TRANSFER_PENALTY = True
    # Set to False to include unavailable players in optimisation
    EXCLUDE_UNAVAILABLE = True
    # Set to True for detailed technical output, False for clean summary
    GRANULAR_OUTPUT = True
    # Set to True to see how players scores are calculated,
    # GRANULAR_OUTPUT must also be set to True
    DETAILED_CALCULATION = False

    # === SEASON ===
    # The season currently being played. This must NOT also appear in
    # PAST_SEASONS: HistoricalDataManager raises on startup if it does,
    # because the two would produce identically named columns and
    # silently misalign the season weights once current-season
    # integration begins at GW8.
    CURRENT_SEASON = "2026-27"

    # === TOKENS ===
    WILDCARD = False  # True when Wildcard or Free Hit is used
    FREE_HIT_PREV_GW = False  # True if you used Free Hit prev gameweek
    BENCH_BOOST = False
    TRIPLE_CAPTAIN = False

    # === TRANSFER EFFICIENCY SETTINGS ===
    # Minimum FPL score improvement per transfer required
    MIN_TRANSFER_VALUE = 2.0
    # Gameweeks a transfer is assumed to be held for. A -4 hit is a
    # one-off cost while the benefit accrues for as long as you hold the
    # incoming player, so the penalty is spread across this many
    # gameweeks rather than charged entirely to the coming week.
    TRANSFER_HORIZON_GWS = 4

    # === POINTS PROJECTION SETTINGS ===
    # Average points for a decent player by position when they play
    BASELINE_POINTS_PER_GAME = {
        "GK": 3.0,
        "DEF": 3.0,
        "MID": 3.5,
        "FWD": 3.5,
    }
    # How much 1 FPL score unit translates to points
    FPL_SCORE_TO_POINTS_MULTIPLIER = 1.0

    # === HISTORICAL DATA SETTINGS (Enhanced for xG Analysis) ===
    # Only include seasons with xG data available (2022-23 onwards)
    PAST_SEASONS = ["2025-26", "2024-25", "2023-24"]
    # Recent seasons weighted more heavily for xG consistency
    # detection. These must sum to 1.0.
    HISTORIC_SEASON_WEIGHTS = [0.5, 0.3, 0.2]  # Sums to 1.0
    # Games needed for a season_reliability of 1.0
    RELIABILITY_GAMES_BASELINE = 30
    # Minimum appearances before a season counts toward a player's
    # historic average. Appearances, not starts: dividing points by
    # starts counted substitute points in the numerator but not the
    # denominator, which inflated rotation players several times over.
    MIN_SEASON_APPEARANCES = 10
    # Strength of the pull toward the positional mean, expressed in
    # appearances. A season of 10 appearances is weighted equally
    # against the prior; a full season barely moves. Raise to distrust
    # small samples more, set to 0 to disable.
    PPG_SHRINKAGE_APPEARANCES = 10
    # How many upcoming fixtures' difficulty to consider.
    # Safe to raise above 1 now that fixture_multiplier is a per-gameweek
    # count rather than a sum across the window. 4 to 6 is a better match
    # for transfer and wildcard decisions.
    FIRST_N_GAMEWEEKS = 1

    # === FIXTURE DIFFICULTY DECAY SETTINGS ===
    # Controls how much future fixtures are discounted relative to
    # immediate ones. 0.6 means each subsequent gameweek is weighted 60%
    # of the previous. Lower values put more emphasis on the next match.
    FIXTURE_DECAY_FACTOR = 0.9

    # === SCORING WEIGHTS BY POSITION ===
    # Use weights optimised by ML (set to True to load from CSV files)
    USE_ML_WEIGHTS = False

    # Manual weights - used if ML weights disabled or loading fails
    # These should total 1.0 for each position
    POSITION_SCORING_WEIGHTS = {
        "GK": {
            "form": 0.5,        # Importance of current season average
            "historic": 0.2,    # Importance of historic seasons
            "difficulty": 0.3   # Importance of upcoming fixtures
        },
        "DEF": {
            "form": 0.5,
            "historic": 0.25,
            "difficulty": 0.25
        },
        "MID": {
            "form": 0.5,
            "historic": 0.25,
            "difficulty": 0.25
        },
        "FWD": {
            "form": 0.5,
            "historic": 0.25,
            "difficulty": 0.25
        }
    }

    # === EARLY SEASON PENALTY SETTINGS ===
    EARLY_SEASON_PENALTY_INITIAL = 4.0  # Divide form by this initially
    EARLY_SEASON_DECAY_FACTOR = 0.6     # Decay factor per gameweek
    EARLY_SEASON_PENALTY_GAMEWEEKS = 4  # Number of GWs penalty applies

    # === CURRENT SEASON INTEGRATION ===
    # After GW8, current season data will be integrated into historical
    # analysis to preserve information not captured in 'form'
    CURRENT_SEASON_INTEGRATION_GW = 8    # GW when integration begins
    CURRENT_SEASON_MAX_WEIGHT = 0.6      # Max weight for current season

    # === MODIFIER SCALES ===
    # Modifiers are applied additively, in z-score units, so a 1.1 team
    # modifier adds 0.1 * TEAM_MODIFIER_SCALE whether the player's base
    # quality is positive or negative. Multiplying a negative score by a
    # modifier above 1.0 made it worse, so every modifier previously did
    # the opposite of its intent for below-average players.
    TEAM_MODIFIER_SCALE = 2.0
    XCONSISTENCY_SCALE = 2.0
    FORM_CONSISTENCY_SCALE = 1.0

    # The xG regression modifier is weighted by how much evidence it
    # rests on, so it is close to neutral in the opening weeks and
    # opens up as the season progresses. Raise this floor only if you
    # want early-season xG deviations to carry weight regardless of
    # sample size.
    XG_MODIFIER_CONFIDENCE_FLOOR = 0.0

    # === CLUB CHANGES ===
    # Historic output earned at a different club is weaker evidence, so
    # it is shrunk toward the positional mean by this factor at zero
    # gameweeks, recovering to full weight over TRANSFER_RECOVERY_GWS.
    # Lower TRANSFER_SHRINK = distrust a mover's old numbers more.
    TRANSFER_SHRINK = 0.55
    TRANSFER_RECOVERY_GWS = 6

    # === MID-GAMEWEEK RUNS ===
    # Divide cumulative stats (starts, minutes) by each club's own
    # matches played rather than by a single global GAMEWEEK - 1. Run
    # midway through a gameweek, the global count assumes every club is
    # level, so the clubs that have already played read as more reliable
    # and the optimiser piles into them. Leave on to run at any point in
    # the week.
    NORMALISE_PARTIAL_GAMEWEEK = True

    # Treat a club whose fixture has already kicked off as a blank for
    # the gameweek being optimised. Those points can no longer be bought,
    # so scoring the fixture would credit a projection the player cannot
    # deliver. Set False to score the full gameweek retrospectively.
    EXCLUDE_STARTED_FIXTURES = True

    # === SQUAD COMPOSITION ===
    SQUAD_SIZE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    MAX_PER_TEAM = 3

    # Cap the ILP pool to the top N players per position by fpl_score.
    # None means the whole pool, which is the safe default: a cheap
    # enabler the budget constraint wants can sit well down the score
    # order, so trimming is not guaranteed to return the same squad.
    # Everyone in the previous squad and every forced selection is kept
    # regardless. Try {"GK": 20, "DEF": 60, "MID": 60, "FWD": 40} only
    # if solve time is still a problem.
    POOL_LIMITS = None
    # Price cap for the goalkeeper who takes the bench slot
    BENCH_GK_MAX_COST = 4.0
    # Weight given to bench players in the squad objective
    BENCH_WEIGHT = 0.2
    # Players on zero form are penalised rather than barred from the XI.
    # FPL form is points per game over the last 30 days, so a player
    # returning from a knock is legitimately on zero.
    PENALISE_ZERO_FORM = True
    ZERO_FORM_PENALTY = 0.5

    # === TEAM ADJUSTMENTS ===
    # Derive promoted clubs from last season's team list rather than
    # trusting a hardcoded list that goes stale every summer. The list
    # below is only a fallback if the lookup fails.
    AUTO_DETECT_PROMOTED_TEAMS = True
    PROMOTED_TEAMS = ["Coventry City", "Hull City", "Ipswich Town"]

    # Flat penalty applied to players at newly promoted clubs
    PROMOTED_PENALTY = -0.3

    # Raise on startup if TEAM_MODIFIERS does not match the clubs the
    # API returns. Set False to downgrade this to a warning.
    VALIDATE_TEAM_NAMES = True

    # Team performance modifiers (adjust for over/under-performing
    # teams). Teams that have overperformed should be under 1.0 and vice
    # versa. Names must match the FPL API exactly, or VALIDATE_TEAM_NAMES
    # will raise on startup. A mismatched key silently does nothing.
    TEAM_MODIFIERS = {
        "Arsenal": 1.0,
        "Aston Villa": 1.0,
        "Bournemouth": 1.0,
        "Brentford": 1.0,
        "Brighton": 1.0,
        "Chelsea": 1.0,
        "Coventry City": 1.0,
        "Crystal Palace": 1.0,
        "Everton": 1.0,
        "Fulham": 1.0,
        "Hull City": 1.0,
        "Ipswich Town": 1.0,
        "Leeds": 1.0,
        "Liverpool": 1.0,
        "Man City": 1.0,
        "Man Utd": 1.0,
        "Newcastle": 1.0,
        "Nott'm Forest": 1.0,
        "Sunderland": 1.0,
        "Spurs": 1.0,
    }

    # === PLAYER SELECTIONS ===
    # Force specific players to be selected (use lowercase names)
    FORCED_SELECTIONS = {
        "GK": [],
        "DEF": [],
        "MID": [],
        "FWD": []
    }

    # Players that should not be considered (use lowercase names).
    # Matching is on display name, so an entry that several players
    # share removes all of them.
    BLACKLIST_PLAYERS = []

    def __init__(self):
        """Initialise config and load ML weights if enabled."""
        import copy

        # Make a deep copy of POSITION_SCORING_WEIGHTS to avoid
        # modifying the class variable
        self.POSITION_SCORING_WEIGHTS = copy.deepcopy(
            self.__class__.POSITION_SCORING_WEIGHTS
        )

        # PROMOTED_TEAMS is rewritten at runtime when
        # AUTO_DETECT_PROMOTED_TEAMS is on, so copy it off the class to
        # avoid mutating shared state.
        self.PROMOTED_TEAMS = list(self.__class__.PROMOTED_TEAMS)

        # Load ML weights if enabled
        if self.USE_ML_WEIGHTS:
            try:
                from src.utils.ml_weight_loader import MLWeightLoader
                loader = MLWeightLoader(self)
                self.POSITION_SCORING_WEIGHTS = loader.load_all_weights(
                    self.POSITION_SCORING_WEIGHTS
                )
            except ImportError:
                if self.GRANULAR_OUTPUT:
                    print(
                        "⚠ Could not import ml_weight_loader module. "
                        "Using manual weights."
                    )
            except Exception as error:
                if self.GRANULAR_OUTPUT:
                    print(
                        f"⚠ Error loading ML weights: {error}. "
                        f"Using manual weights."
                    )