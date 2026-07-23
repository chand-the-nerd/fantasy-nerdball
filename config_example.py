"""
Configuration settings for Fantasy Nerdball FPL optimisation tool.
Edit this file to customise your optimisation preferences.
"""


class Config:
    """Configuration class containing all settings for the FPL optimisation."""
    
    # === BASIC SETTINGS ===
    GAMEWEEK = 1
    BUDGET = 100
    FREE_TRANSFERS = 1
    ACCEPT_TRANSFER_PENALTY = True
    # Set to False to include unavailable players in optimisation
    EXCLUDE_UNAVAILABLE = True
    # Set to True for detailed technical output, False for clean summary
    GRANULAR_OUTPUT = True
    # Set to True to see how players scores are calculated, GRANULAR_OUTPUT
    # must also be set to True
    DETAILED_CALCULATION = False

    # === TOKENS ===
    WILDCARD = False # Set to True when either Wildcard or Free Hit is used
    FREE_HIT_PREV_GW = False # Set to True if you used Free Hit prev gameweek
    BENCH_BOOST = False
    TRIPLE_CAPTAIN = False

    # === TRANSFER EFFICIENCY SETTINGS ===
    # Minimum FPL score improvement per transfer required
    MIN_TRANSFER_VALUE = 2.0

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
    # Recent seasons weighted more heavily for xG consistency detection
    # Redistributed weights to maintain same relative importance
    HISTORIC_SEASON_WEIGHTS = [0.5, 0.3, 0.2]  # Sums to 1.0
    # How many upcoming fixtures' difficulty to consider
    FIRST_N_GAMEWEEKS = 1

    # === FIXTURE DIFFICULTY DECAY SETTINGS ===
    # Controls how much future fixtures are discounted relative to immediate ones
    # 0.6 means each subsequent gameweek is weighted 60% of the previous
    # Lower values = more emphasis on immediate fixtures
    # Higher values = more balanced weighting across all fixtures
    FIXTURE_DECAY_FACTOR = 0.9

    # === SCORING WEIGHTS BY POSITION ===
    # Use weights optimised by ML (set to True to load from CSV files)
    USE_ML_WEIGHTS = False
    
    # Manual weights - used if ML weights disabled or loading fails
    # These should total 1.0 for each position
    POSITION_SCORING_WEIGHTS = {
        "GK": {
            "form": 0.5,        # Importance of current season average
            "historic": 0.2,    # Importance of historic seasons' average
            "difficulty": 0.3   # Importance of upcoming fixture difficulty
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
    EARLY_SEASON_DECAY_FACTOR = 0.6    # Decay factor per gameweek
    EARLY_SEASON_PENALTY_GAMEWEEKS = 4   # Number of GWs penalty applies

    # === CURRENT SEASON INTEGRATION ===
    # After GW8, current season data will be integrated into historical
    # analysis to preserve information not captured in 'form'
    CURRENT_SEASON_INTEGRATION_GW = 8    # GW when integration begins
    CURRENT_SEASON_MAX_WEIGHT = 0.6      # Maximum weight for current season

    # === SQUAD COMPOSITION ===
    SQUAD_SIZE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    MAX_PER_TEAM = 3

    # === TEAM ADJUSTMENTS ===
    # Teams that will be considered as newly promoted
    PROMOTED_TEAMS = ["Burnley", "Sunderland", "Leeds"]

    # Team performance modifiers (adjust for over/under-performing teams)
    # Teams that have overperformed should be under 1.0 and vice versa
    TEAM_MODIFIERS = {
        "Arsenal": 1.0,
        "Aston Villa": 1.0,
        "Bournemouth": 1.0,
        "Brentford": 1.0,
        "Brighton": 1.0,
        "Chelsea": 1.0,
        "Coventry": 1.0,
        "Crystal Palace": 1.0,
        "Everton": 1.0,
        "Fulham": 1.0,
        "Hull": 1.0,
        "Ipswich": 1.0,
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

    # Players that should not be considered (use lowercase names)
    BLACKLIST_PLAYERS = []

    def __init__(self):
        """Initialise config and load ML weights if enabled."""
        import copy
        
        # Make a deep copy of POSITION_SCORING_WEIGHTS to avoid modifying
        # the class variable
        self.POSITION_SCORING_WEIGHTS = copy.deepcopy(
            self.__class__.POSITION_SCORING_WEIGHTS
        )
        
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
            except Exception as e:
                if self.GRANULAR_OUTPUT:
                    print(
                        f"⚠ Error loading ML weights: {e}. "
                        f"Using manual weights."
                    )