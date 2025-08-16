# === CONFIG ===
GAMEWEEK = 1
BUDGET = 100.0  # million
FREE_TRANSFERS = 0
WILDCARD = False

# Transfer efficiency settings
MIN_TRANSFER_VALUE = 0.5  # Minimum FPL score improvement per transfer required
CONSERVATIVE_MODE = False  # If True, be more cautious about making transfers
TRANSFER_ROLLOVER_VALUE = 0.3  # Value of keeping a transfer for next week

# Points projection settings
BASELINE_POINTS_PER_GAME = {
    "GK": 4.0,  # Average points for a decent GK
    "DEF": 4.5,  # Average points for a decent defender
    "MID": 5.0,  # Average points for a decent midfielder
    "FWD": 5.5,  # Average points for a decent forward
}

# Adjusts FPL Score to Projected Points
# Reduce if model is overestimating points, increase if underestimating.
FPL_SCORE_TO_POINTS_MULTIPLIER = 1.5

# Previous season config
PAST_SEASONS = ["2024-25", "2023-24"]  # Historic seasons to consider
HISTORIC_SEASON_WEIGHTS = [0.7, 0.3]  # Relative to seasons provided above
FIRST_N_GAMEWEEKS = 5  # How many upcoming fixtures' difficulty to consider

# These settings should be set based on how you perceive the importance of
# each factor. They should total 1.0.
FORM_WEIGHT = 0.33  # Importance of current season average
HISTORIC_WEIGHT = 0.33  # Importance of historic seasons' average
DIFFICULTY_WEIGHT = 0.34  # Importance of upcoming fixture difficulty

SQUAD_SIZE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_TEAM = 3

# These teams will be considered as shit-kickers
PROMOTED_TEAMS = ["Burnley", "Sunderland", "Leeds"]

# This can be used to adjust players from under/over-performing teams.
# Teams that have overperformed in previous seasons should be under 1.0 and
# vice versa.
TEAM_MODIFIERS = {
    "Arsenal": 1.0,
    "Aston Villa": 1.0,
    "Bournemouth": 1.0,
    "Brentford": 1.0,
    "Brighton": 1.0,
    "Burnley": 1.0,
    "Chelsea": 1.0,
    "Crystal Palace": 1.0,
    "Everton": 1.0,
    "Fulham": 1.0,
    "Leeds": 1.0,
    "Liverpool": 1.0,
    "Man City": 1.0,
    "Man Utd": 1.0,
    "Newcastle": 1.0,
    "Nott'm Forest": 1.0,
    "Sunderland": 1.0,
    "Spurs": 1.0,
    "West Ham": 1.0,
    "Wolves": 1.0,
}

FORCED_SELECTIONS = {
    "GK": [],
    "DEF": [],
    "MID": [],
    "FWD": []
    }

# Players that should not be considered as part of the selection criteria.
BLACKLIST_PLAYERS = []