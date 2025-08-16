import requests
import pandas as pd
import pulp
import os
import unicodedata

# === CONFIG ===
GAMEWEEK = 2
BUDGET = 100.0  # million
FREE_TRANSFERS = 1

# Transfer efficiency settings
MIN_TRANSFER_VALUE = 0.3  # Minimum FPL score improvement per transfer required
CONSERVATIVE_MODE = False   # If True, be more cautious about making transfers
TRANSFER_ROLLOVER_VALUE = 0.3  # Value of keeping a transfer for next week

# Points projection settings
BASELINE_POINTS_PER_GAME = {
    "GK": 4.0,   # Average points for a decent GK
    "DEF": 4.5,  # Average points for a decent defender
    "MID": 5.0,  # Average points for a decent midfielder
    "FWD": 5.5   # Average points for a decent forward
}
FPL_SCORE_TO_POINTS_MULTIPLIER = 1.5  # How much 1 FPL score unit translates to points

# Previous season config
PAST_SEASONS = ["2024-25", "2023-24"]  # Historic seasons to consider
HISTORIC_SEASON_WEIGHTS = [0.7, 0.3]  # Relative to seasons provided above
FIRST_N_GAMEWEEKS = 5  # How many upcoming fixtures' difficulty to consider

# These settings should be set based on how you perceive the importance of
# each factor. They should total 1.0.
FORM_WEIGHT = 0.0  # Importance of current season average
HISTORIC_WEIGHT = 0.7  # Importance of historic seasons' average
DIFFICULTY_WEIGHT = 0.3  # Importance of upcoming fixture difficulty

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
    "Bournemouth": 0.8,
    "Brentford": 1.0,
    "Brighton": 1.0,
    "Burnley": 1.0,
    "Chelsea": 1.0,
    "Crystal Palace": 1.0,
    "Everton": 1.0,
    "Fulham": 1.0,
    "Leeds": 1.0,
    "Liverpool": 1.0,
    "Man City": 1.05,
    "Man Utd": 1.1,
    "Newcastle": 1.0,
    "Nott'm Forest": 0.7,
    "Sunderland": 1.0,
    "Spurs": 1.1,
    "West Ham": 0.8,
    "Wolves": 0.85,
}

FORCED_SELECTIONS = {
    "GK": ["dúbravka"],  # Force Dubravka into any GK slot
    "DEF": [],           # No forced defenders
    "MID": ["baleba"],
    "FWD": []
}

# Players that should not be considered as part of the selection criteria.
BLACKLIST_PLAYERS = [
    "isak"
]


def get_json(url):
    """
    Fetch JSON data from a URL.
    
    Args:
        url (str): The URL to fetch JSON data from.
        
    Returns:
        dict: The JSON response as a dictionary.
    """
    return requests.get(url).json()


def normalize_name(s):
    """
    Normalize a name string for comparison by converting to lowercase.
    
    Args:
        s (str or pd.NA): The name string to normalize.
        
    Returns:
        str: The normalized name string, or empty string if input is NA.
    """
    return "" if pd.isna(s) else str(s).strip().lower()


def normalize_for_matching(text):
    """
    Normalize text for matching by removing accents and converting to 
    lowercase.
    
    Args:
        text (str): The text to normalize.
        
    Returns:
        str: The normalized text with accents removed and lowercase.
    """
    normalized = unicodedata.normalize('NFD', text)
    ascii_text = ''.join(c for c in normalized 
                        if unicodedata.category(c) != 'Mn')
    return ascii_text.lower().strip()


def calculate_projected_points(df):
    """
    Calculate projected points for each player based on their base quality 
    (when they actually play), not influenced by reliability.
    
    Args:
        df (pd.DataFrame): Player data with base_quality and position.
        
    Returns:
        pd.DataFrame: DataFrame with added projected_points column.
    """
    df = df.copy()
    
    # Convert position to string if it's categorical
    if df['position'].dtype.name == 'category':
        position_values = df['position'].astype(str)
    else:
        position_values = df['position']
    
    # Calculate baseline points for each position
    df['baseline_points'] = position_values.map(BASELINE_POINTS_PER_GAME)
    
    # Convert base quality to points adjustment (not FPL score)
    # This represents how good they are WHEN they play
    df['points_adjustment'] = df['base_quality'] * FPL_SCORE_TO_POINTS_MULTIPLIER
    
    # Calculate projected points
    df['projected_points'] = df['baseline_points'] + df['points_adjustment']
    
    # Ensure minimum of 1 point (no player should project negative)
    df['projected_points'] = df['projected_points'].clip(lower=1.0)
    
    return df


def add_points_analysis_to_display(df):
    """
    Add projected points columns and score components for better display.
    
    Args:
        df (pd.DataFrame): Squad data with FPL scores and base quality.
        
    Returns:
        pd.DataFrame: DataFrame with projected points and score components added.
    """
    df = calculate_projected_points(df)
    
    # Calculate minutes per game (total minutes / games with >0 minutes)
    games_played = (df['minutes'] > 0).astype(int)
    df['minspg'] = 0.0
    played_mask = games_played > 0
    df.loc[played_mask, 'minspg'] = df.loc[played_mask, 'minutes'] / games_played[played_mask]
    df['minspg'] = df['minspg'].round(0).astype(int)
    
    # Prepare display columns with exact names requested
    df['form'] = df['form'].round(1)
    df['historic_ppg'] = df['avg_ppg_past2'].round(1)
    
    # Use the fixture difficulty average from fetch_player_fixture_difficulty
    # This shows average difficulty over FIRST_N_GAMEWEEKS (same as algorithm uses)
    df['fixture_diff'] = (6 - df['fixture_bonus']).round(1)
    
    df['reliability'] = (df['current_reliability'] * 100).round(0).astype(int)
    df['proj_pts'] = df['projected_points'].round(1)
    
    return df


def fetch_current_players():
    """
    Fetch current FPL player data from the API and prepare it for analysis.
    
    Returns:
        pd.DataFrame: DataFrame containing current player data with calculated
                     fields for cost, position, team info, and name keys.
    """
    data = get_json("https://fantasy.premierleague.com/api/bootstrap-static/")
    players = pd.DataFrame(data["elements"])
    
    teams = pd.DataFrame(data["teams"])[["id", "name"]].rename(
        columns={"id": "team_id", "name": "team"}
    )
    
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    
    players = players.rename(columns={"team": "team_id", 
                                     "element_type": "pos_id"})
    players["position"] = players["pos_id"].map(pos_map)
    players = players.merge(teams, on="team_id", how="left")
    players["form"] = pd.to_numeric(players["form"], 
                                   errors="coerce").fillna(0.0)
    players["now_cost_m"] = players["now_cost"] / 10.0
    players["display_name"] = players["web_name"]
    players["name_key"] = players["web_name"].map(normalize_name)
    
    # Save to CSV
    os.makedirs("data", exist_ok=True)
    players.to_csv("data/players.csv", index=False)
    print("Player data saved to data/players.csv")
    
    return players


def load_previous_squad(gameweek):
    """
    Load the previous gameweek's squad from CSV file.
    
    Args:
        gameweek (int): Current gameweek (will load gameweek-1's squad).
        
    Returns:
        pd.DataFrame or None: Previous squad data or None if file doesnt exist.
    """
    if gameweek <= 1:
        print("No previous squad to load (this is GW1 or earlier)")
        return None
    
    prev_gw = gameweek - 1
    squad_file = f"squads/gw{prev_gw}/full_squad.csv"
    
    if not os.path.exists(squad_file):
        print(f"Warning: Previous squad file not found at {squad_file}")
        print("Proceeding without transfer constraints (assuming new team)")
        return None
    
    try:
        prev_squad = pd.read_csv(squad_file)
        print(f"Loaded previous squad from {squad_file}")
        print(f"Previous squad had {len(prev_squad)} players")
        return prev_squad
    except Exception as e:
        print(f"Error loading previous squad: {e}")
        return None


def match_players_to_current(prev_squad, current_players):
    """
    Match previous squad players to current player database.
    
    Args:
        prev_squad (pd.DataFrame): Previous gameweek's squad.
        current_players (pd.DataFrame): Current player database.
        
    Returns:
        list: List of player IDs from previous squad that are still available.
    """
    prev_player_ids = []
    
    for _, prev_player in prev_squad.iterrows():
        prev_name = prev_player['display_name'].strip()
        prev_pos = prev_player['position']
        prev_team = prev_player['team']
        
        # Try to find matching player in current database
        matches = current_players[
            (current_players['position'] == prev_pos) &
            (current_players['team'] == prev_team) &
            (current_players['display_name'].str.strip() == prev_name)
        ]
        
        if len(matches) == 1:
            prev_player_ids.append(matches.iloc[0]['id'])
        elif len(matches) > 1:
            # Multiple matches, take the first one
            print(
                f"Warning: Multiple matches for {prev_name},"
                " taking first match")
            prev_player_ids.append(matches.iloc[0]['id'])
        else:
            # Try fuzzy matching on name
            fuzzy_matches = current_players[
                (current_players['position'] == prev_pos) &
                (current_players['team'] == prev_team) &
                (current_players['display_name'].str.contains(
                prev_name.split()[0], case=False, na=False
                ))
            ]
            
            if len(fuzzy_matches) > 0:
                print(
                    f"Fuzzy match for {prev_name}: "
                    f"{fuzzy_matches.iloc[0]['display_name']}"
                      )
                prev_player_ids.append(fuzzy_matches.iloc[0]['id'])
            else:
                print(
                    f"Warning: Could not find current match for "
                    f"{prev_name} ({prev_pos}, {prev_team})"
                    )
    
    print(
        f"Successfully matched {len(prev_player_ids)} "
        "players from previous squad"
        )
    return prev_player_ids


def fetch_past_season_points(season_folder):
    """
    Fetch historical points per game data for a specific season, with 
    reliability filtering.
    
    Args:
        season_folder (str): The season folder name (e.g., "2023-24").
        
    Returns:
        pd.DataFrame: DataFrame containing web_name, points_per_game, 
                     reliability_factor, and name_key for the specified season.
    """
    url = (
        f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
        f"master/data/{season_folder}/players_raw.csv"
    )
    df = pd.read_csv(url)
    
    # Calculate games played and reliability
    df['minutes_played'] = pd.to_numeric(df['minutes'], errors='coerce').fillna(0)
    df['total_points'] = pd.to_numeric(df['total_points'], errors='coerce').fillna(0)
    
    # Calculate games played (assuming 90 minutes = 1 game)
    df['games_played'] = (df['minutes_played'] / 90).round().clip(lower=0)
    
    # Calculate season reliability (games played / max possible games ~38)
    # Use 30 as reasonable threshold for "full season" to account for injuries
    df['season_reliability'] = (df['games_played'] / 30).clip(upper=1.0)
    
    # Calculate points per game, only for players who actually played
    df['points_per_game'] = 0.0
    played_mask = df['games_played'] >= 1
    df.loc[played_mask, 'points_per_game'] = (
        df.loc[played_mask, 'total_points'] / df.loc[played_mask, 'games_played']
    )
    
    # Apply reliability penalty for players who weren't regular starters
    # Players who played <80% of games get their PPG penalized
    reliability_threshold = 0.8
    unreliable_mask = df['season_reliability'] < reliability_threshold
    
    # Penalty factor: if you played 50% of games, your PPG gets reduced
    penalty_factor = df['season_reliability'].clip(lower=0.3)  # Minimum 30% value
    df.loc[unreliable_mask, 'points_per_game'] *= penalty_factor[unreliable_mask]
    
    # Select relevant columns
    result_df = df[["web_name", "points_per_game", "games_played", "season_reliability"]].copy()
    result_df["name_key"] = result_df["web_name"].map(normalize_name)
    result_df = result_df.rename(columns={
        "points_per_game": f"ppg_{season_folder}",
        "games_played": f"games_{season_folder}",
        "season_reliability": f"reliability_{season_folder}"
    })
    
    return result_df


def merge_past_two_seasons(current, past_seasons, weights):
    """
    Merge historical points per game data from multiple seasons with the 
    current player data, calculating reliability based on games played.
    
    Args:
        current (pd.DataFrame): Current season player data.
        past_seasons (list): List of season folder names to fetch data for.
        weights (list): List of weights to apply to each season's PPG.
        
    Returns:
        pd.DataFrame: Current player data merged with weighted average of
                     historical points per game, adjusted for reliability.
    """
    hist_frames = [fetch_past_season_points(s) for s in past_seasons]
    hist = hist_frames[0]
    for extra in hist_frames[1:]:
        hist = hist.merge(extra, on="name_key", how="outer")
    
    # Get PPG columns, games columns, and reliability columns
    ppg_cols = [c for c in hist.columns if c.startswith("ppg_")]
    games_cols = [c for c in hist.columns if c.startswith("games_")]
    reliability_cols = [c for c in hist.columns if c.startswith("reliability_")]
    
    # Calculate weighted average PPG and overall reliability
    hist["avg_ppg_past2"] = 0.0
    hist["total_games_past2"] = 0
    hist["avg_reliability"] = 0.0
    
    for ppg_col, games_col, reliability_col, weight in zip(ppg_cols, games_cols, reliability_cols, weights):
        # Fill NaN values
        hist[ppg_col] = hist[ppg_col].fillna(0)
        hist[games_col] = hist[games_col].fillna(0)
        hist[reliability_col] = hist[reliability_col].fillna(0)
        
        # Only include seasons where player had meaningful game time (>=8 games)
        # This is roughly 20% of a season - minimum for any consideration
        sufficient_games_mask = hist[games_col] >= 8
        
        # Add to weighted averages
        hist.loc[sufficient_games_mask, "avg_ppg_past2"] += (
            hist.loc[sufficient_games_mask, ppg_col] * weight
        )
        hist.loc[sufficient_games_mask, "avg_reliability"] += (
            hist.loc[sufficient_games_mask, reliability_col] * weight
        )
        hist["total_games_past2"] += hist[games_col]
    
    # Calculate current season reliability based on actual games played
    # Get current gameweek from FPL API to calculate current season reliability
    try:
        current_gw_data = get_json("https://fantasy.premierleague.com/api/bootstrap-static/")
        current_gw = None
        for event in current_gw_data['events']:
            if event['is_current']:
                current_gw = event['id']
                break
        
        # If no current gameweek found, use the highest finished gameweek
        if current_gw is None:
            finished_events = [e for e in current_gw_data['events'] if e['finished']]
            if finished_events:
                current_gw = max(e['id'] for e in finished_events) + 1
            else:
                current_gw = 1  # Fallback to GW1
        
        # Calculate current season reliability: games played / gameweeks elapsed
        # Use minutes > 0 as indicator of "appeared in match"
        current_games_played = (current['minutes'] > 0).astype(int)
        gameweeks_elapsed = max(1, current_gw - 1)  # At least 1 to avoid division by zero
        
        current_reliability = current_games_played / gameweeks_elapsed
        current_reliability = current_reliability.clip(upper=1.0)  # Cap at 100%
        
    except Exception as e:
        print(f"Warning: Could not calculate current season reliability: {e}")
        # Fallback: use a simple heuristic based on minutes played
        # Assume ~10 gameweeks have passed (adjust this based on when you run it)
        estimated_gws = 10
        current_games_played = (current['minutes'] > 0).astype(int)
        current_reliability = (current_games_played / estimated_gws).clip(upper=1.0)
    
    return current.merge(
        hist[["name_key", "avg_ppg_past2", "total_games_past2", "avg_reliability"]], 
        on="name_key", 
        how="left"
    ).assign(current_reliability=current_reliability)


def fetch_player_fixture_difficulty(first_n_gws, players, starting_gameweek):
    """
    Calculate average fixture difficulty for each player over the next 
    N gameweeks from a starting gameweek and save fixture data to CSV.
    
    Args:
        first_n_gws (int): Number of gameweeks to consider for fixture 
                          difficulty.
        players (pd.DataFrame): Player data containing team_id and name_key.
        starting_gameweek (int): The gameweek to start calculating from.
        
    Returns:
        pd.DataFrame: DataFrame with name_key, average difficulty, and 
                     fixture bonus (6 - difficulty).
    """
    fixtures = pd.DataFrame(
        get_json("https://fantasy.premierleague.com/api/fixtures/")
    )
    
    # Get team names for fixture CSV
    teams_data = get_json(
        "https://fantasy.premierleague.com/api/bootstrap-static/"
        )
    teams_df = pd.DataFrame(teams_data["teams"])[["id", "name", "short_name"]]
    
    # Create a copy of fixtures for saving to CSV
    fixtures_for_csv = fixtures.copy()
    
    # Add team names to fixtures
    fixtures_for_csv = fixtures_for_csv.merge(
        teams_df.rename(columns={
        "id": "team_h",
        "name": "home_team",
        "short_name": "home_team_short"
        }),
        on="team_h", how="left"
    )
    fixtures_for_csv = fixtures_for_csv.merge(
        teams_df.rename(columns={
        "id": "team_a",
        "name": "away_team",
        "short_name": "away_team_short"
        }),
        on="team_a", how="left"
    )
    
    # Select and reorder columns for the CSV
    fixtures_csv = fixtures_for_csv[[
        "id", "event", "kickoff_time", 
        "home_team", "away_team", 
        "home_team_short", "away_team_short",
        "team_h_difficulty", "team_a_difficulty",
        "team_h_score", "team_a_score", "finished"
    ]].copy()
    
    # Rename columns for clarity
    fixtures_csv = fixtures_csv.rename(columns={
        "id": "fixture_id",
        "event": "gameweek",
        "team_h_difficulty": "home_difficulty",
        "team_a_difficulty": "away_difficulty",
        "team_h_score": "home_score",
        "team_a_score": "away_score"
    })
    
    # Sort by gameweek and kickoff time
    fixtures_csv = fixtures_csv.sort_values(["gameweek", "kickoff_time"])
    
    # Save fixtures to CSV
    os.makedirs("data", exist_ok=True)
    fixtures_csv.to_csv("data/fixtures.csv", index=False)
    print(f"Fixture data saved to data/fixtures.csv")
    
    # Filter fixtures for the specified gameweek range
    end_gameweek = starting_gameweek + first_n_gws - 1
    fixtures = fixtures[
        (pd.to_numeric(fixtures["event"],
                       errors="coerce") >= starting_gameweek) &
        (pd.to_numeric(fixtures["event"], errors="coerce") <= end_gameweek)
    ]
    
    print(f"Calculating fixture difficulty for gameweeks {starting_gameweek} "
          f"to {end_gameweek}")
    
    player_diffs = []

    for _, row in fixtures.iterrows():
        for home_away, team_col, diff_col in [
            ("home", "team_h", "team_h_difficulty"),
            ("away", "team_a", "team_a_difficulty"),
        ]:
            team_id = row[team_col]
            diff = row[diff_col]
            for idx, p in players[players["team_id"] == team_id].iterrows():
                player_diffs.append(
                    {
                        "name_key": p["name_key"],
                        "gw": row["event"],
                        "diff": diff
                    }
                )

    df = pd.DataFrame(player_diffs)
    if df.empty:
        print(f"Warning: No fixtures found for gameweeks {starting_gameweek} "
              f"to {end_gameweek}")
        # Return empty dataframe with correct structure
        return pd.DataFrame(columns=["name_key", "diff", "fixture_bonus"])
    
    avg_diff = df.groupby("name_key", as_index=False)["diff"].mean()
    avg_diff["fixture_bonus"] = 6 - avg_diff["diff"]  # higher is better
    return avg_diff


def build_scores(players, fixture_scores, form_weight, historic_weight,
                diff_weight):
    """
    Calculate FPL scores and base quality scores for each player based on form, 
    historical performance, and fixture difficulty, with reliability considerations.
    
    Args:
        players (pd.DataFrame): Player data with form and historical PPG.
        fixture_scores (pd.DataFrame): Fixture difficulty data.
        form_weight (float): Weight for current season form.
        historic_weight (float): Weight for historical performance.
        diff_weight (float): Weight for fixture difficulty.
        
    Returns:
        pd.DataFrame: Player data with calculated fpl_score and base_quality 
                     for each player.
    """
    df = players.merge(fixture_scores, on="name_key", how="left")
    
    # Fill NaN values to prevent PuLP errors
    df["avg_ppg_past2"] = df["avg_ppg_past2"].fillna(0)
    df["avg_reliability"] = df["avg_reliability"].fillna(0)
    df["current_reliability"] = df["current_reliability"].fillna(0)
    df["fixture_bonus"] = df["fixture_bonus"].fillna(0)
    
    # Calculate team and promotion adjustments
    df["promoted_penalty"] = df["team"].apply(
        lambda x: -0.3 if x in PROMOTED_TEAMS else 0
    )
    df["team_modifier"] = df["team"].map(
        lambda t: TEAM_MODIFIERS.get(t, 1.0)
    )

    def z(s):
        """Z-score normalization with NaN/inf handling."""
        s_clean = s.fillna(0)  # Replace NaN with 0
        s_mean = s_clean.mean()
        s_std = s_clean.std(ddof=0)
        
        # Prevent division by zero
        if s_std == 0 or pd.isna(s_std):
            return pd.Series([0.0] * len(s_clean), index=s_clean.index)
        
        z_scores = (s_clean - s_mean) / s_std
        
        # Replace any remaining NaN/inf values
        z_scores = z_scores.fillna(0)
        z_scores = z_scores.replace([float('inf'), float('-inf')], 0)
        
        return z_scores

    # Calculate base quality components (no reliability factored in)
    form_component = form_weight * z(df["form"])
    historic_component = historic_weight * z(df["avg_ppg_past2"])
    fixture_component = diff_weight * z(df["fixture_bonus"])
    
    # Base quality score (for projected points when they DO play)
    df["base_quality"] = (
        form_component
        + historic_component
        + fixture_component
        + df["promoted_penalty"]
    ) * df["team_modifier"]
    
    # Apply reliability adjustments for squad selection
    # Current season reliability is 5x more important than historical
    reliability_bonus = (
        df["current_reliability"] * 1.5 +     # Current season: games/GWs elapsed
        df["avg_reliability"] * 0.3           # Historical: average games/season ratio
    ) - 0.75  # Center around 0 (0.75 would be 50% current + 25% historical)
    
    # Penalty for historically unreliable players (rotation risks)
    df["historically_unreliable_penalty"] = 0.0  # Initialize column
    unreliable_mask = df["avg_reliability"] < 0.6  # Less than 60% historical games
    df.loc[unreliable_mask, "historically_unreliable_penalty"] = -0.15
    
    # Extra penalty for current season rotation risks
    current_unreliable_mask = df["current_reliability"] < 0.7  # <70% games this season
    df.loc[current_unreliable_mask, "historically_unreliable_penalty"] -= 0.2
    
    # FPL score (for squad selection - includes reliability)
    df["fpl_score"] = (
        df["base_quality"]
        + reliability_bonus
        + df["historically_unreliable_penalty"]
    )
    
    # Final safety check - replace any NaN/inf in both scores
    df["base_quality"] = df["base_quality"].fillna(0)
    df["base_quality"] = df["base_quality"].replace([float('inf'), float('-inf')], 0)
    df["fpl_score"] = df["fpl_score"].fillna(0)
    df["fpl_score"] = df["fpl_score"].replace([float('inf'), float('-inf')], 0)
    
    return df


def add_next_fixture(df, target_gameweek):
    """
    Add fixture information for a specific gameweek for each player.
    
    Args:
        df (pd.DataFrame): Player data to add fixture info to.
        target_gameweek (int): The gameweek number to get fixtures for.
        
    Returns:
        pd.DataFrame: Player data with opponent, venue, and fixture 
                     difficulty columns added for the target gameweek.
    """
    fixtures = pd.DataFrame(
        get_json("https://fantasy.premierleague.com/api/fixtures/")
    )
    fixtures = fixtures[fixtures["event"] == target_gameweek]
    
    if fixtures.empty:
        print(f"Warning: No fixtures found for gameweek {target_gameweek}")
        df["next_opponent"] = "No fixture"
        df["venue"] = "N/A"
        df["fixture_difficulty"] = None
        return df
    
    next_fixtures = []

    for _, p in df.iterrows():
        f = fixtures[
            (fixtures["team_h"] == p["team_id"])
            | (fixtures["team_a"] == p["team_id"])
        ]
        if not f.empty:
            row = f.iloc[0]
            if row["team_h"] == p["team_id"]:
                opponent = row["team_a"]
                venue = "Home"
                difficulty = row["team_h_difficulty"]
            else:
                opponent = row["team_h"]
                venue = "Away"
                difficulty = row["team_a_difficulty"]
            next_fixtures.append(
                {
                    "name_key": p["name_key"],
                    "next_opponent_id": opponent,
                    "venue": venue,
                    "fixture_difficulty": difficulty,
                }
            )
        else:
            next_fixtures.append(
                {
                    "name_key": p["name_key"],
                    "next_opponent_id": None,
                    "venue": "No fixture",
                    "fixture_difficulty": None,
                }
            )

    nf_df = pd.DataFrame(next_fixtures)
    teams = pd.DataFrame(
        get_json(
            "https://fantasy.premierleague.com/api/bootstrap-static/"
        )["teams"]
    )
    nf_df = nf_df.merge(
        teams[["id", "name"]],
        left_on="next_opponent_id",
        right_on="id",
        how="left"
    )
    nf_df = nf_df.rename(columns={"name": "next_opponent"})
    nf_df["next_opponent"] = nf_df["next_opponent"].fillna("No fixture")
    
    df = df.merge(
        nf_df[["name_key", "next_opponent", "venue", "fixture_difficulty"]],
        on="name_key",
        how="left",
    )
    return df


def get_no_transfer_squad(df, prev_squad_ids):
    """
    Get the optimal squad using only players from the previous gameweek 
    (no transfers).
    
    Args:
        df (pd.DataFrame): Current player database with scores.
        prev_squad_ids (list): Player IDs from previous squad.
        
    Returns:
        pd.DataFrame: Best possible squad using only previous players.
    """
    if prev_squad_ids is None:
        return pd.DataFrame()
    
    # Filter to only previous squad players that are still available
    id_to_index = {df.iloc[i]['id']: i for i in range(len(df))}
    available_prev_players = [
        pid for pid in prev_squad_ids if pid in id_to_index
        ]
    
    if len(available_prev_players) < 15:
        print(f"Warning: Only {len(available_prev_players)} "
              "previous players available")
        return pd.DataFrame()
    
    prev_squad_df = df[df['id'].isin(available_prev_players)].copy()
    
    # Simple optimization for starting XI from these 15 players
    n = len(prev_squad_df)
    y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]
    
    prob = pulp.LpProblem("No_Transfer_Squad", pulp.LpMaximize)
    prob += pulp.lpSum(y[i] * prev_squad_df.iloc[i]["fpl_score"] for i in range(n))
    
    # Starting XI constraints
    prob += pulp.lpSum(y[i] for i in range(n)) == 11
    prob += pulp.lpSum(
        y[i] for i in range(n) if prev_squad_df.iloc[i]["position"] == "GK"
        ) == 1
    prob += pulp.lpSum(
        y[i] for i in range(n) if prev_squad_df.iloc[i]["position"] == "DEF"
        ) >= 3
    prob += pulp.lpSum(
        y[i] for i in range(n) if prev_squad_df.iloc[i]["position"] == "DEF"
        ) <= 5
    prob += pulp.lpSum(
        y[i] for i in range(n) if prev_squad_df.iloc[i]["position"] == "MID"
        ) >= 3
    prob += pulp.lpSum(
        y[i] for i in range(n) if prev_squad_df.iloc[i]["position"] == "MID"
        ) <= 5
    prob += pulp.lpSum(
        y[i] for i in range(n) if prev_squad_df.iloc[i]["position"] == "FWD"
        ) >= 1
    prob += pulp.lpSum(
        y[i] for i in range(n) if prev_squad_df.iloc[i]["position"] == "FWD"
        ) <= 3
    
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if status != pulp.LpStatusOptimal:
        return pd.DataFrame()
    
    starting_mask = [pulp.value(y[i]) == 1 for i in range(n)]
    starting_xi = prev_squad_df.iloc[starting_mask].copy()
    
    return starting_xi


def update_forced_selections_from_squad(starting, bench):
    """
    Create forced selections dictionary from squad players.
    
    Args:
        starting (pd.DataFrame): Starting XI players.
        bench (pd.DataFrame): Bench players.
        
    Returns:
        dict: Forced selections dictionary with all squad players.
    """
    forced_selections = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    full_squad = pd.concat([starting, bench], ignore_index=True)
    
    for _, player in full_squad.iterrows():
        pos = player['position']
        name = player['display_name']
        forced_selections[pos].append(name)
    
    return forced_selections


def evaluate_transfer_value(
        current_squad,
        potential_squad,
        transfers_made,
        free_transfers
        ):
    """
    Evaluate whether the transfers provide sufficient value to justify making 
    them.
    
    Args:
        current_squad (pd.DataFrame): Current squad if no transfers were made.
        potential_squad (pd.DataFrame): Potential new squad with transfers.
        transfers_made (int): Number of transfers that would be made.
        free_transfers (int): Number of free transfers available.
        
    Returns:
        tuple: (should_make_transfers, value_analysis_dict)
    """
    if transfers_made == 0:
        return True, {"reason": "No transfers needed", "value": 0}
    
    # Calculate score improvement
    current_score = current_squad['fpl_score'].sum()
    new_score = potential_squad['fpl_score'].sum()
    score_improvement = new_score - current_score
    
    # Calculate value per transfer
    value_per_transfer = (score_improvement / transfers_made 
                          if transfers_made > 0 else 0)
    
    # Calculate opportunity cost of using transfers
    # Each unused transfer has value for future weeks
    transfers_remaining_after = free_transfers - transfers_made
    rollover_value_lost = (min(transfers_made, free_transfers)
                           * TRANSFER_ROLLOVER_VALUE)
    
    # Adjust for conservative mode
    min_threshold = MIN_TRANSFER_VALUE
    if CONSERVATIVE_MODE:
        min_threshold *= 1.5  # Be more cautious
    
    # Net value considering rollover opportunity cost
    net_value = score_improvement - rollover_value_lost
    
    analysis = {
        "transfers_made": transfers_made,
        "score_improvement": score_improvement,
        "value_per_transfer": value_per_transfer,
        "min_threshold": min_threshold,
        "rollover_value_lost": rollover_value_lost,
        "net_value": net_value,
        "transfers_remaining": transfers_remaining_after
    }
    
    # Decision logic
    if score_improvement < 0:
        return False, {**analysis, "reason": "Transfers would decrease "
                       "team value"}
    
    if value_per_transfer < min_threshold:
        return False, {**analysis, "reason": f"Value per transfer "
                       f"({value_per_transfer:.3f}) below threshold "
                       f"({min_threshold:.3f})"}
    
    if net_value < 0 and free_transfers > 1:
        return False, {**analysis, "reason":
                       "Better to save transfers for future weeks"}
    
    # Special case: If we have many transfers (4+), be more willing to use some
    if free_transfers >= 4:
        min_threshold *= 0.7  # Lower threshold when we have many transfers
        if value_per_transfer >= min_threshold:
            return True, {**analysis, "reason":
                          "Many transfers available, using some is beneficial"}
    
    # Must use transfers if we're at the cap (5)
    if free_transfers >= 5:
        return True, {**analysis, "reason":
                      "At transfer cap, must use to avoid waste"}
    
    return True, {**analysis, "reason":
                  "Transfers provide sufficient value"}


def select_squad_ilp(
        df,
        forced_selections,
        prev_squad_ids=None,
        free_transfers=None,
        show_transfer_summary=True
        ):
    """
    Select optimal FPL squad using Integer Linear Programming with 
    forced player selections and transfer constraints.
    
    Args:
        df (pd.DataFrame): Player data with fpl_scores and all required fields.
        forced_selections (dict): Dictionary of forced player selections.
        prev_squad_ids (list, optional): List of player IDs from prev squad.
        free_transfers (int, optional): Number of free transfers available.
        
    Returns:
        tuple: (starting_xi_dataframe, bench_dataframe, forced_selections_str)
               containing the optimal squad selection and info about forced
               players.
    """
    # Process forced selections before DataFrame modifications
    forced_player_ids = []
    forced_players_info = []
    
    for pos, players_to_force in forced_selections.items():
        if not players_to_force:
            continue
            
        for name in players_to_force:
            # Search for player in original dataframe
            for idx, row in df.iterrows():
                if (row['position'] == pos and 
                    (row['display_name'].lower() == name.lower() or
                     normalize_for_matching(row['display_name']) == 
                     normalize_for_matching(name))):
                    forced_player_ids.append(row['id'])
                    forced_players_info.append(
                        f"{row['display_name']} ({row['position']}, "
                        f"{row['team']})"
                    )
                    break
    
    # Store forced selections info for later display
    forced_selections_display = (', '.join(forced_players_info) 
                                if forced_players_info else None)
    
    # Clean the dataframe
    df = df[
        (df["status"] == "a")
        | (df["chance_of_playing_next_round"].fillna(100) >= 75)
    ].copy()
    df = df.reset_index(drop=True).drop_duplicates(subset="name_key")
    df = df[~df["display_name"].str.lower().isin(BLACKLIST_PLAYERS)].copy()
    n = len(df)

    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]

    prob = pulp.LpProblem("FPL_Squad_Selection", pulp.LpMaximize)
    prob += pulp.lpSum(
        y[i] * df.iloc[i]["fpl_score"]
        + 0.2 * (x[i] - y[i]) * df.iloc[i]["fpl_score"]
        for i in range(n)
    )
    prob += pulp.lpSum(x[i] for i in range(n)) == 15  # Total squad
    prob += pulp.lpSum(y[i] for i in range(n)) == 11  # Starting XI
    for i in range(n):
        prob += y[i] <= x[i]  # XI <= squad

    # Position constraints for squad
    for pos, count in SQUAD_SIZE.items():
        prob += (
            pulp.lpSum(
                x[i] for i in range(n) if df.iloc[i]["position"] == pos
            ) == count
        )

    # Starting XI position constraints
    prob += pulp.lpSum(
        y[i] for i in range(n) if df.iloc[i]["position"] == "GK"
    ) == 1
    prob += pulp.lpSum(
        y[i] for i in range(n) if df.iloc[i]["position"] == "DEF"
    ) >= 3
    prob += pulp.lpSum(
        y[i] for i in range(n) if df.iloc[i]["position"] == "DEF"
    ) <= 5
    prob += pulp.lpSum(
        y[i] for i in range(n) if df.iloc[i]["position"] == "MID"
    ) >= 3
    prob += pulp.lpSum(
        y[i] for i in range(n) if df.iloc[i]["position"] == "MID"
    ) <= 5
    prob += pulp.lpSum(
        y[i] for i in range(n) if df.iloc[i]["position"] == "FWD"
    ) >= 1
    prob += pulp.lpSum(
        y[i] for i in range(n) if df.iloc[i]["position"] == "FWD"
    ) <= 3

    # Apply forced selections using player IDs
    for player_id in forced_player_ids:
        # Find player in cleaned dataframe by searching through all rows
        for i in range(len(df)):
            if df.iloc[i]['id'] == player_id:
                prob += x[i] == 1  # Force into squad
                break

    # Transfer constraint
    if prev_squad_ids is not None and free_transfers is not None:
        print(f"Applying transfer constraint: max {free_transfers} transfers")
        
        # Create mapping of player IDs to dataframe indices
        id_to_index = {df.iloc[i]['id']: i for i in range(n)}
        
        # Count how many previous squad players are kept
        prev_players_kept = pulp.lpSum(
            x[id_to_index[player_id]] 
            for player_id in prev_squad_ids 
            if player_id in id_to_index
        )
        
        # Number of transfers = 15 - number of players kept from previous squad
        # This must be <= free_transfers
        # So: 15 - prev_players_kept <= free_transfers
        # Therefore: prev_players_kept >= 15 - free_transfers
        min_players_to_keep = 15 - free_transfers
        prob += prev_players_kept >= min_players_to_keep
        
        print(f"Must keep at least {min_players_to_keep} "
              "players from previous squad")

    # Bench constraints
    prob += (
        pulp.lpSum(
            (x[i] - y[i])
            for i in range(n)
            if df.iloc[i]["position"] == "GK" 
            and df.iloc[i]["now_cost_m"] == 4.0
        )
        == 1
    )

    # Allow up to 2 of DEF, MID, FWD on bench
    for pos in ["DEF", "MID", "FWD"]:
        prob += (
            pulp.lpSum(
                (x[i] - y[i]) for i in range(n) 
                if df.iloc[i]["position"] == pos
            )
            <= 2
        )

    # Max per team
    for team in df["team_id"].unique():
        prob += (
            pulp.lpSum(x[i] for i in range(n) 
                      if df.iloc[i]["team_id"] == team)
            <= MAX_PER_TEAM
        )

    # Budget
    prob += pulp.lpSum(
        x[i] * df.iloc[i]["now_cost_m"] for i in range(n)
    ) <= BUDGET

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if status != pulp.LpStatusOptimal:
        print(f"Optimization failed with status: {pulp.LpStatus[status]}")
        return pd.DataFrame(), pd.DataFrame(), None

    selected_mask = [pulp.value(x[i]) == 1 for i in range(n)]
    squad = df.iloc[selected_mask].copy()
    squad["starting_XI"] = [pulp.value(y[i])
                           for i in range(n) if pulp.value(x[i]) == 1]

    # Calculate and display transfer information
    if prev_squad_ids is not None and show_transfer_summary:
        current_squad_ids = set(squad['id'].tolist())
        prev_squad_ids_set = set(prev_squad_ids)
        
        players_kept = current_squad_ids.intersection(prev_squad_ids_set)
        players_out = prev_squad_ids_set - current_squad_ids
        players_in = current_squad_ids - prev_squad_ids_set
        
        transfers_made = len(players_out)
        
        print(f"\n=== Proposed Transfer Summary ===")
        print(f"Players to keep from previous squad: {len(players_kept)}")
        print(f"Proposed transfers: {transfers_made} "
              f"(out of {free_transfers} free transfers)")
        
        if players_out:
            print("Players to transfer OUT:")
            for player_id in players_out:
                # Find player name from previous squad or current database
                prev_player = df[df['id'] == player_id]
                if not prev_player.empty:
                    player_name = prev_player.iloc[0]['display_name']
                    print(f"  - {player_name}")
        
        if players_in:
            print("Players to transfer IN:")
            for player_id in players_in:
                player = squad[squad['id'] == player_id].iloc[0]
                print(f"  + {player['display_name']} "
                      f"({player['position']}, {player['team']})")
    elif prev_squad_ids is not None and not show_transfer_summary:
        current_squad_ids = set(squad['id'].tolist())
        prev_squad_ids_set = set(prev_squad_ids)
        transfers_made = len(prev_squad_ids_set - current_squad_ids)

    squad_starting = squad[squad["starting_XI"] == 1].copy()
    squad_bench = squad[squad["starting_XI"] == 0].copy()

    # Order bench: GK first, then by descending FPL score
    gk_bench = squad_bench[squad_bench["position"] == "GK"].copy()
    non_gk_bench = squad_bench[squad_bench["position"] != "GK"].copy()
    non_gk_bench = non_gk_bench.sort_values("fpl_score", ascending=False)
    squad_bench = pd.concat([gk_bench, non_gk_bench], ignore_index=True)

    return squad_starting, squad_bench, forced_selections_display


def main():
    """
    Main function to run the FPL optimization process.
    """
    print(f"Planning for Gameweek {GAMEWEEK}")
    print(f"Free transfers available: {FREE_TRANSFERS}")
    
    # Load previous squad if available
    prev_squad = load_previous_squad(GAMEWEEK)
    
    print("Fetching current players...")
    players = fetch_current_players()
    
    # Match previous squad to current players if we have a previous squad
    prev_squad_ids = None
    if prev_squad is not None:
        prev_squad_ids = match_players_to_current(prev_squad, players)
    
    print("Merging historical points...")
    players = merge_past_two_seasons(
        players,
        PAST_SEASONS,
        HISTORIC_SEASON_WEIGHTS
    )
    print(f"Fetching player-level fixture difficulty from GW{GAMEWEEK}...")
    fixture_scores = fetch_player_fixture_difficulty(
        FIRST_N_GAMEWEEKS, players, GAMEWEEK
    )
    print("Scoring players...")
    scored = build_scores(
        players,
        fixture_scores,
        FORM_WEIGHT,
        HISTORIC_WEIGHT,
        DIFFICULTY_WEIGHT
    )
    print("Optimizing squad using PuLP...")
    
    # First, get the best squad with transfers
    (starting_with_transfers,
     bench_with_transfers,
     forced_selections_display) = select_squad_ilp(
        scored,
        FORCED_SELECTIONS,
        prev_squad_ids,
        FREE_TRANSFERS
    )

    if starting_with_transfers.empty:
        print("No valid solution found!")
        return
    
    # Calculate transfers that would be made
    transfers_made = 0
    if prev_squad_ids is not None:
        current_squad_ids = set(
            pd.concat(
            [starting_with_transfers,
             bench_with_transfers]
             )['id'].tolist())
        prev_squad_ids_set = set(prev_squad_ids)
        transfers_made = len(prev_squad_ids_set - current_squad_ids)
    
    # If we have previous squad, evaluate whether transfers are worth it
    should_make_transfers = True
    transfer_analysis = {}
    
    if prev_squad_ids is not None and transfers_made > 0:
        print(f"\nEvaluating transfer value...")
        
        # Get best squad with no transfers for comparison
        no_transfer_starting = get_no_transfer_squad(scored, prev_squad_ids)
        
        if not no_transfer_starting.empty:
            # Find the full no-transfer squad (all 15 players)
            no_transfer_full_squad = scored[
                scored['id'].isin(prev_squad_ids)
                ].copy()
            
            should_make_transfers, transfer_analysis = evaluate_transfer_value(
                no_transfer_starting,  # Starting XI with no transfers
                starting_with_transfers,  # Starting XI with transfers
                transfers_made,
                FREE_TRANSFERS
            )
            
            print(f"\n=== Transfer Value Analysis ===")
            print(f"Transfers to be made: "
                  f"{transfer_analysis['transfers_made']}")
            print(f"Score improvement: "
                  f"{transfer_analysis['score_improvement']:.3f}")
            print(f"Value per transfer: "
                  f"{transfer_analysis['value_per_transfer']:.3f}")
            print(f"Minimum threshold: "
                  f"{transfer_analysis['min_threshold']:.3f}")
            print(f"Rollover value lost: "
                  f"{transfer_analysis['rollover_value_lost']:.3f}")
            print(f"Net value: {transfer_analysis['net_value']:.3f}")
            print(f"Decision: {transfer_analysis['reason']}")
    
    # Choose final squad based on transfer analysis
    if should_make_transfers:
        print(f"\n✅ Making {transfers_made} transfer(s)")
        starting = starting_with_transfers
        bench = bench_with_transfers
    else:
        print(f"\n❌ Transfers not worth it - keeping current squad")
        if prev_squad_ids is not None:
            # Use the no-transfer squad
            full_no_transfer_squad = scored[
                scored['id'].isin(prev_squad_ids)
                ].copy()
            starting = get_no_transfer_squad(scored, prev_squad_ids)
            bench = full_no_transfer_squad[
                ~full_no_transfer_squad['id'].isin(starting['id'])
                ].copy()
            
            # Order bench properly
            gk_bench = bench[bench["position"] == "GK"].copy()
            non_gk_bench = bench[bench["position"] != "GK"].copy()
            non_gk_bench = non_gk_bench.sort_values(
                "fpl_score",
                ascending=False
                )
            bench = pd.concat([gk_bench, non_gk_bench], ignore_index=True)
        else:
            # Fallback to transfer squad if no previous squad
            starting = starting_with_transfers
            bench = bench_with_transfers
    
    # Create updated forced selections with all squad players
    updated_forced_selections = update_forced_selections_from_squad(
        starting, bench
    )
    
    # Show squad table before next match optimization
    full_squad = pd.concat([starting, bench], ignore_index=True)
    full_squad = add_next_fixture(full_squad, GAMEWEEK)
    
    # Sort by position for display
    position_order = ["GK", "DEF", "MID", "FWD"]
    full_squad["position"] = pd.Categorical(
        full_squad["position"], categories=position_order, ordered=True
    )
    full_squad = full_squad.sort_values("position")

    # Show forced selections if any were made
    if forced_selections_display:
        print(f"\nForced selections: {forced_selections_display}")

    # Print full squad with projected points
    full_squad_display = add_points_analysis_to_display(full_squad)
    print(f"\n=== Full Squad for GW{GAMEWEEK} ===")
    print(
        full_squad_display[
            [
                "display_name",
                "position",
                "team",
                "form",
                "historic_ppg",
                "fixture_diff",
                "reliability",
                "minspg",
                "proj_pts",
                "next_opponent",
            ]
        ]
    )

    print(f"\nFetching difficulty for GW{GAMEWEEK} fixture...")
    fixture_scores_next = fetch_player_fixture_difficulty(1, players, GAMEWEEK)
    scored_next = build_scores(
        players,
        fixture_scores_next,
        FORM_WEIGHT,
        HISTORIC_WEIGHT,
        DIFFICULTY_WEIGHT
    )

    print(f"Optimizing Starting XI for GW{GAMEWEEK}...")
    starting, bench, _ = select_squad_ilp(
        scored_next,
        updated_forced_selections,
        prev_squad_ids,
        FREE_TRANSFERS,
        show_transfer_summary=False
    )

    starting = add_next_fixture(starting, GAMEWEEK)
    bench = add_next_fixture(bench, GAMEWEEK)

    # Add projected points to starting XI and bench
    starting_display = add_points_analysis_to_display(starting)
    bench_display = add_points_analysis_to_display(bench)

    # Define position order
    position_order = ["GK", "DEF", "MID", "FWD"]

    # Convert 'position' to categorical with the desired order
    starting_display["position"] = pd.Categorical(
        starting_display["position"], categories=position_order, ordered=True
    )
    bench_display["position"] = pd.Categorical(
        bench_display["position"], categories=position_order, ordered=True
    )

    # Sort by position
    starting_display = starting_display.sort_values("position")
    bench_display = bench_display.sort_values("position")

    # Mark captain and vice-captain
    if not starting_display.empty:
        top_two_idx = starting_display["fpl_score"].nlargest(2).index
        if len(top_two_idx) > 0:
            starting_display.loc[top_two_idx[0], "display_name"] += " (C)"
        if len(top_two_idx) > 1:
            starting_display.loc[top_two_idx[1], "display_name"] += " (V)"

    print(f"\n=== Starting XI for GW{GAMEWEEK} ===")
    print(
        starting_display[
            [
                "display_name",
                "position",
                "team",
                "form",
                "historic_ppg",
                "fixture_diff",
                "reliability",
                "minspg",
                "proj_pts",
                "next_opponent",
            ]
        ]
    )
    print(f"\n=== Bench (in order) for GW{GAMEWEEK} ===")
    print(
        bench_display[
            [
                "display_name",
                "position",
                "team",
                "form",
                "historic_ppg",
                "fixture_diff",
                "reliability",
                "minspg",
                "proj_pts",
                "next_opponent",
            ]
        ]
    )

    total_cost = starting["now_cost_m"].sum() + bench["now_cost_m"].sum()
    total_projected_points = starting_display["projected_points"].sum()
    print(f"\nTotal Squad Cost: {total_cost:.1f}m")
    print(f"Projected Starting XI Points: {total_projected_points:.1f}")

    # Save squad data to CSV files with projected points
    squad_dir = f"squads/gw{GAMEWEEK}"
    os.makedirs(squad_dir, exist_ok=True)
    
    # Save combined squad with all details including projected points
    squad_combined = pd.concat([starting_display, bench_display], 
                               ignore_index=True)
    squad_combined["squad_role"] = (["Starting XI"] * len(starting_display) + 
                                   ["Bench"] * len(bench_display))
    combined_file = f"{squad_dir}/full_squad.csv"
    squad_combined.to_csv(combined_file, index=False)
    
    # Save simple squad overview
    simple_squad = squad_combined[["display_name", "position", "now_cost_m", 
                                  "team", "proj_pts", "squad_role"]].copy()
    simple_squad = simple_squad.rename(columns={
        "display_name": "player",
        "now_cost_m": "price",
        "team": "club",
        "proj_pts": "projected_points"
    })
    simple_file = f"{squad_dir}/full_squad_simple.csv"
    simple_squad.to_csv(simple_file, index=False)
    
    print(f"\nSquad saved to {squad_dir}/")
    print(f"  - {combined_file}")
    print(f"  - {simple_file}")


if __name__ == "__main__":
    main()