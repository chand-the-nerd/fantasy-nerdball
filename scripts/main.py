import requests
import pandas as pd
import pulp
import os
import unicodedata

# === CONFIG ===
GAMEWEEK = 1
BUDGET = 100.0  # million

# Previous season config
PAST_SEASONS = ["2024-25", "2023-24"]  # Historic seasons to consider
HISTORIC_SEASON_WEIGHTS = [0.7, 0.3]  # Relative to seasons provided above
FIRST_N_GAMEWEEKS = 5  # How many upcoming fixtures' difficulty to be considered

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
    "MID": [],
    "FWD": []
}

# Players that should not be considered as part of the selection criteria.
BLACKLIST_PLAYERS = [
    "isak"
]


def get_json(url):
    """Fetch JSON data from a URL.
    
    Args:
        url (str): The URL to fetch JSON data from.
        
    Returns:
        dict: The JSON response as a dictionary.
    """
    return requests.get(url).json()


def normalize_name(s):
    """Normalize a name string for comparison by converting to lowercase.
    
    Args:
        s (str or pd.NA): The name string to normalize.
        
    Returns:
        str: The normalized name string, or empty string if input is NA.
    """
    return "" if pd.isna(s) else str(s).strip().lower()


def normalize_for_matching(text):
    """Normalize text for matching by removing accents and converting to 
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


def fetch_current_players():
    """Fetch current FPL player data from the API and prepare it for analysis.
    
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


def fetch_past_season_points(season_folder):
    """Fetch historical points data for a specific season.
    
    Args:
        season_folder (str): The season folder name (e.g., "2023-24").
        
    Returns:
        pd.DataFrame: DataFrame containing web_name, total_points, and 
                     name_key for the specified season.
    """
    url = (
        f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
        f"master/data/{season_folder}/players_raw.csv"
    )
    df = pd.read_csv(url)
    df = df[["web_name", "total_points"]].copy()
    df["name_key"] = df["web_name"].map(normalize_name)
    df = df.rename(columns={"total_points": f"total_points_{season_folder}"})
    return df


def merge_past_two_seasons(current, past_seasons, weights):
    """Merge historical points data from multiple seasons with the current 
    player data.
    
    Args:
        current (pd.DataFrame): Current season player data.
        past_seasons (list): List of season folder names to fetch data for.
        weights (list): List of weights to apply to each season's points.
        
    Returns:
        pd.DataFrame: Current player data merged with weighted average of
                     historical points.
    """
    hist_frames = [fetch_past_season_points(s) for s in past_seasons]
    hist = hist_frames[0]
    for extra in hist_frames[1:]:
        hist = hist.merge(extra, on="name_key", how="outer")
    point_cols = [c for c in hist.columns if c.startswith("total_points_")]
    hist["avg_points_past2"] = 0
    for col, w in zip(point_cols, weights):
        hist["avg_points_past2"] += hist[col].fillna(0) * w
    return current.merge(
        hist[["name_key", "avg_points_past2"]], on="name_key", how="left"
    )


def fetch_player_fixture_difficulty(first_n_gws, players):
    """Calculate average fixture difficulty for each player over the next 
    N gameweeks.
    
    Args:
        first_n_gws (int): Number of gameweeks to consider for fixture 
                          difficulty.
        players (pd.DataFrame): Player data containing team_id and name_key.
        
    Returns:
        pd.DataFrame: DataFrame with name_key, average difficulty, and 
                     fixture bonus (6 - difficulty).
    """
    fixtures = pd.DataFrame(
        get_json("https://fantasy.premierleague.com/api/fixtures/")
    )
    fixtures = fixtures[
        pd.to_numeric(fixtures["event"], errors="coerce") <= first_n_gws
    ]
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
    avg_diff = df.groupby("name_key", as_index=False)["diff"].mean()
    avg_diff["fixture_bonus"] = 6 - avg_diff["diff"]  # higher is better
    return avg_diff


def build_scores(players, fixture_scores, form_weight, historic_weight,
                diff_weight):
    """Calculate FPL scores for each player based on form, historical 
    performance, and fixture difficulty.
    
    Args:
        players (pd.DataFrame): Player data with form and historical points.
        fixture_scores (pd.DataFrame): Fixture difficulty data.
        form_weight (float): Weight for current season form.
        historic_weight (float): Weight for historical performance.
        diff_weight (float): Weight for fixture difficulty.
        
    Returns:
        pd.DataFrame: Player data with calculated fpl_score for each player.
    """
    df = players.merge(fixture_scores, on="name_key", how="left")
    df["avg_points_past2"] = df["avg_points_past2"].fillna(0)
    df["promoted_penalty"] = df["team"].apply(
        lambda x: -0.3 if x in PROMOTED_TEAMS else 0
    )
    df["team_modifier"] = df["team"].map(
        lambda t: TEAM_MODIFIERS.get(t, 1.0)
    )

    def z(s):
        """Z-score normalization."""
        return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

    df["fpl_score"] = (
        form_weight * z(df["form"])
        + historic_weight * z(df["avg_points_past2"])
        + diff_weight * z(df["fixture_bonus"])
        + df["promoted_penalty"]
    ) * df["team_modifier"]
    return df


def add_next_fixture(df, next_gw=1):
    """Add next fixture information for each player.
    
    Args:
        df (pd.DataFrame): Player data to add fixture info to.
        next_gw (int): The gameweek number to get fixtures for.
        
    Returns:
        pd.DataFrame: Player data with next opponent, venue, and fixture 
                     difficulty columns added.
    """
    fixtures = pd.DataFrame(
        get_json("https://fantasy.premierleague.com/api/fixtures/")
    )
    fixtures = fixtures[fixtures["event"] == next_gw]
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
                    "venue": None,
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
    df = df.merge(
        nf_df[["name_key", "next_opponent", "venue", "fixture_difficulty"]],
        on="name_key",
        how="left",
    )
    return df


def update_forced_selections_from_squad(starting, bench):
    """Create forced selections dictionary from squad players.
    
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


def select_squad_ilp(df, forced_selections):
    """Select optimal FPL squad using Integer Linear Programming with 
    forced player selections.
    
    Args:
        df (pd.DataFrame): Player data with fpl_scores and all required fields.
        forced_selections (dict): Dictionary of forced player selections.
        
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

    squad_starting = squad[squad["starting_XI"] == 1].copy()
    squad_bench = squad[squad["starting_XI"] == 0].copy()

    # Order bench: GK first, then by descending FPL score
    gk_bench = squad_bench[squad_bench["position"] == "GK"].copy()
    non_gk_bench = squad_bench[squad_bench["position"] != "GK"].copy()
    non_gk_bench = non_gk_bench.sort_values("fpl_score", ascending=False)
    squad_bench = pd.concat([gk_bench, non_gk_bench], ignore_index=True)

    return squad_starting, squad_bench, forced_selections_display


def main():
    """Main function to run the FPL optimization process."""
    print("Fetching current players...")
    players = fetch_current_players()
    print("Merging historical points...")
    players = merge_past_two_seasons(
        players,
        PAST_SEASONS,
        HISTORIC_SEASON_WEIGHTS
    )
    print("Fetching player-level fixture difficulty...")
    fixture_scores = fetch_player_fixture_difficulty(
        FIRST_N_GAMEWEEKS, players
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
    starting, bench, forced_selections_display = select_squad_ilp(
        scored,
        FORCED_SELECTIONS
    )

    if starting.empty:
        print("No valid solution found!")
        return
    
    # Create updated forced selections with all squad players
    updated_forced_selections = update_forced_selections_from_squad(
        starting, bench
    )
    
    # Show squad table before next match optimization
    full_squad = pd.concat([starting, bench], ignore_index=True)
    full_squad = add_next_fixture(full_squad)
    
    # Sort by position for display
    position_order = ["GK", "DEF", "MID", "FWD"]
    full_squad["position"] = pd.Categorical(
        full_squad["position"], categories=position_order, ordered=True
    )
    full_squad = full_squad.sort_values("position")

    # Show forced selections if any were made
    if forced_selections_display:
        print(f"\nForced selections: {forced_selections_display}")

    # Print full squad
    print("\n=== Full Squad for Next GW ===")
    print(
        full_squad[
            [
                "display_name",
                "position",
                "team",
                "now_cost_m",
                "fpl_score",
                "next_opponent",
                "venue",
                "fixture_difficulty",
            ]
        ]
    )

    print("\nFetching difficulty for next fixture...")
    fixture_scores_next = fetch_player_fixture_difficulty(1, players)
    scored_next = build_scores(
        players,
        fixture_scores_next,
        FORM_WEIGHT,
        HISTORIC_WEIGHT,
        DIFFICULTY_WEIGHT
    )

    print("Optimizing Starting XI for next match...")
    starting, bench, _ = select_squad_ilp(
        scored_next,
        updated_forced_selections
    )

    starting = add_next_fixture(starting)
    bench = add_next_fixture(bench)

    # Define position order
    position_order = ["GK", "DEF", "MID", "FWD"]

    # Convert 'position' to categorical with the desired order
    starting["position"] = pd.Categorical(
        starting["position"], categories=position_order, ordered=True
    )
    bench["position"] = pd.Categorical(
        bench["position"], categories=position_order, ordered=True
    )

    # Sort by position
    starting = starting.sort_values("position")
    bench = bench.sort_values("position")

    # Mark captain and vice-captain
    if not starting.empty:
        top_two_idx = starting["fpl_score"].nlargest(2).index
        if len(top_two_idx) > 0:
            starting.loc[top_two_idx[0], "display_name"] += " (C)"
        if len(top_two_idx) > 1:
            starting.loc[top_two_idx[1], "display_name"] += " (V)"

    print("\n=== Starting XI ===")
    print(
        starting[
            [
                "display_name",
                "position",
                "team",
                "now_cost_m",
                "fpl_score",
                "next_opponent",
                "venue",
                "fixture_difficulty",
            ]
        ]
    )
    print("\n=== Bench (in order) ===")
    print(
        bench[
            [
                "display_name",
                "position",
                "team",
                "now_cost_m",
                "fpl_score",
                "next_opponent",
                "venue",
                "fixture_difficulty",
            ]
        ]
    )

    total_cost = starting["now_cost_m"].sum() + bench["now_cost_m"].sum()
    total_points = starting["fpl_score"].sum()
    print(f"\nTotal Squad Cost: {total_cost:.1f}m")
    print(f"Expected Starting XI Points: {total_points:.2f}")

    # Save squad data to CSV files
    squad_dir = f"squads/gw{GAMEWEEK}"
    os.makedirs(squad_dir, exist_ok=True)
    
    # Save combined squad with all details
    squad_combined = pd.concat([starting, bench], ignore_index=True)
    squad_combined["squad_role"] = (["Starting XI"] * len(starting) + 
                                   ["Bench"] * len(bench))
    combined_file = f"{squad_dir}/full_squad.csv"
    squad_combined.to_csv(combined_file, index=False)
    
    # Save simple squad overview
    simple_squad = squad_combined[["display_name", "position", "now_cost_m", 
                                  "team", "squad_role"]].copy()
    simple_squad = simple_squad.rename(columns={
        "display_name": "player",
        "now_cost_m": "price",
        "team": "club"
    })
    simple_file = f"{squad_dir}/full_squad_simple.csv"
    simple_squad.to_csv(simple_file, index=False)
    
    print(f"\nSquad saved to {squad_dir}/")
    print(f"  - {combined_file}")
    print(f"  - {simple_file}")


if __name__ == "__main__":
    main()