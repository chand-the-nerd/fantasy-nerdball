"""
Configuration settings for Fantasy Nerdball FPL optimisation tool.
Edit this file to customise your optimisation preferences.
"""


class Config:
    """Configuration class containing all settings for the FPL optimisation."""
    
    # === BASIC SETTINGS ===
    GAMEWEEK = 2
    BUDGET = 100.0  # Will be overridden by value of squad from prev gameweek
    FREE_TRANSFERS = 1
    ACCEPT_TRANSFER_PENALTY = False
    # Set to False to include unavailable players in optimisation
    EXCLUDE_UNAVAILABLE = True
    WILDCARD = False

    # === TRANSFER EFFICIENCY SETTINGS ===
    # Minimum FPL score improvement per transfer required
    MIN_TRANSFER_VALUE = 5.0 

    # === POINTS PROJECTION SETTINGS ===
    # Average points for a decent player by position when they play
    BASELINE_POINTS_PER_GAME = {
        "GK": 3.0,
        "DEF": 3.5,
        "MID": 4.5,
        "FWD": 4.5,
    }
    # How much 1 FPL score unit translates to points
    FPL_SCORE_TO_POINTS_MULTIPLIER = 1.0

    # === HISTORICAL DATA SETTINGS (Enhanced for xG Analysis) ===
    # Only include seasons with xG data available (2022-23 onwards)
    PAST_SEASONS = ["2024-25", "2023-24", "2022-23"]
    # Recent seasons weighted more heavily for xG consistency detection
    # Redistributed weights to maintain same relative importance
    HISTORIC_SEASON_WEIGHTS = [0.5, 0.3, 0.2]  # Sums to 1.0
    # How many upcoming fixtures' difficulty to consider
    FIRST_N_GAMEWEEKS = 5

    # === SCORING WEIGHTS ===
    # These should total 1.0
    FORM_WEIGHT = 0.15      # Importance of current season average
    HISTORIC_WEIGHT = 0.50   # Importance of historic seasons' average
    DIFFICULTY_WEIGHT = 0.35 # Importance of upcoming fixture difficulty

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
        "Bournemouth": 0.8,
        "Brentford": 0.8,
        "Brighton": 1.0,
        "Burnley": 1.0,
        "Chelsea": 1.0,
        "Crystal Palace": 1.0,
        "Everton": 1.0,
        "Fulham": 1.0,
        "Leeds": 1.1,
        "Liverpool": 1.0,
        "Man City": 1.0,
        "Man Utd": 1.1,
        "Newcastle": 0.9,
        "Nott'm Forest": 0.8,
        "Sunderland": 1.0,
        "Spurs": 1.1,
        "West Ham": 0.8,
        "Wolves": 0.7,
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
    BLACKLIST_PLAYERS = ["isak"]