import requests
import pandas as pd
import pulp

# === CONFIG ===
BUDGET = 100.0  # million
PAST_SEASONS = ["2024-25", "2023-24"]
WEIGHTS = [0.7, 0.3]  # More weight to most recent
FIRST_N_GAMEWEEKS = 5

SQUAD_SIZE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_TEAM = 3
PROMOTED_TEAMS = ["Burnley", "Sunderland", "Leeds"]

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
    "Nott'm Forest": 0.8,
    "Sunderland": 1.0,
    "Spurs": 1.1,
    "West Ham": 0.9,
    "Wolves": 0.85,
}

BLACKLIST_PLAYERS = ["isak"]

#Helpers
def get_json(url):
    return requests.get(url).json()


def normalize_name(s):
    return "" if pd.isna(s) else str(s).strip().lower()


#Fetch current players
def fetch_current_players():
    data = get_json("https://fantasy.premierleague.com/api/bootstrap-static/")
    players = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])[["id", "name"]].rename(
        columns={"id": "team_id", "name": "team"}
    )
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    players = players.rename(
        columns={"team": "team_id", "element_type": "pos_id"}
        )
    players["position"] = players["pos_id"].map(pos_map)
    players = players.merge(teams, on="team_id", how="left")
    players["form"] = pd.to_numeric(
        players["form"], errors="coerce"
        ).fillna(0.0)
    players["now_cost_m"] = players["now_cost"] / 10.0
    players["display_name"] = players["web_name"]
    players["name_key"] = players["web_name"].map(normalize_name)
    return players


#Fetch past season points
def fetch_past_season_points(season_folder):
    url = (
        f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
        f"master/data/{season_folder}/players_raw.csv"
    )
    df = pd.read_csv(url)
    df = df[["web_name", "total_points"]].copy()
    df["name_key"] = df["web_name"].map(normalize_name)
    df = df.rename(columns={"total_points": f"total_points_{season_folder}"})
    return df


#Merge historical points
def merge_past_two_seasons(current, past_seasons, weights):
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


#Player-level fixture difficulty
def fetch_player_fixture_difficulty(first_n_gws, players):
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


# --- Score players with team modifier ---
def build_scores(players, fixture_scores):
    df = players.merge(fixture_scores, on="name_key", how="left")
    df["avg_points_past2"] = df["avg_points_past2"].fillna(0)
    df["promoted_penalty"] = df["team"].apply(
        lambda x: -0.3 if x in PROMOTED_TEAMS else 0
    )
    df["team_modifier"] = df["team"].map(lambda t: TEAM_MODIFIERS.get(t, 1.0))

    def z(s):
        return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

    df["fpl_score"] = (
        0.00 * z(df["form"])
        + 0.7 * z(df["avg_points_past2"])
        + 0.3 * z(df["fixture_bonus"])
        + df["promoted_penalty"]
    ) * df["team_modifier"]
    return df


# --- Add next fixture info ---
def add_next_fixture(df, next_gw=1):
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


# --- PuLP optimizer ---
def select_squad_ilp(df):
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

    # Bench constraints

    # 1 GK must be on bench and must be 4.0m
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
            (x[i] - y[i]) for i in range(n) if df.iloc[i]["position"] == pos
            )
            <= 2
        )

    # Max per team
    for team in df["team_id"].unique():
        prob += (
            pulp.lpSum(x[i] for i in range(n) if df.iloc[i]["team_id"] == team)
            <= MAX_PER_TEAM
        )

    # Budget
    prob += pulp.lpSum(
        x[i] * df.iloc[i]["now_cost_m"] for i in range(n)
        ) <= BUDGET

    prob.solve()

    selected_mask = [pulp.value(x[i]) == 1 for i in range(n)]
    squad = df.iloc[selected_mask].copy()
    squad["starting_XI"] = [pulp.value(y[i])
                            for i in range(n) if pulp.value(x[i]) == 1]

    squad_starting = squad[squad["starting_XI"] == 1].copy()
    squad_bench = squad[squad["starting_XI"] == 0].copy()

    # Order bench: GK last
    bench_order = {"DEF": 1, "MID": 2, "FWD": 3, "GK": 4}
    squad_bench["bench_order"] = squad_bench["position"].map(bench_order)
    squad_bench = squad_bench.sort_values("bench_order")

    return squad_starting, squad_bench


# --- MAIN ---
def main():
    print("Fetching current players...")
    players = fetch_current_players()
    print("Merging historical points...")
    players = merge_past_two_seasons(players, PAST_SEASONS, WEIGHTS)
    print("Fetching player-level fixture difficulty...")
    fixture_scores = fetch_player_fixture_difficulty(
        FIRST_N_GAMEWEEKS, players
        )
    print("Scoring players...")
    scored = build_scores(players, fixture_scores)
    print("Optimizing squad using PuLP...")
    starting, bench = select_squad_ilp(scored)

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


if __name__ == "__main__":
    main()
