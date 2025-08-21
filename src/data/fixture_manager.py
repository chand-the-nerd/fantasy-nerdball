"""Module for managing fixture data and difficulty calculations with 
decaying weights."""

import os
import pandas as pd
from ..api.fpl_client import FPLClient


class FixtureManager:
    """Handles fixture data and difficulty calculations."""
    
    def __init__(self, config):
        self.config = config
        self.fpl_client = FPLClient()
    
    def _calculate_decay_weights(self, num_gameweeks: int) -> list:
        """
        Calculate decaying weights for fixtures.
        
        Args:
            num_gameweeks (int): Number of gameweeks to weight
            
        Returns:
            list: Weights that decay exponentially, summing to 1.0
        """
        if num_gameweeks <= 0:
            return []
        
        if num_gameweeks == 1:
            return [1.0]
        
        # Use exponential decay: weight = decay_factor^(position-1)
        # where position starts at 1 for immediate gameweek
        decay_factor = getattr(self.config, 'FIXTURE_DECAY_FACTOR', 0.6)
        
        raw_weights = []
        for i in range(num_gameweeks):
            weight = decay_factor ** i
            raw_weights.append(weight)
        
        # Normalise to sum to 1.0
        total_weight = sum(raw_weights)
        normalised_weights = [w / total_weight for w in raw_weights]
        
        return normalised_weights
    
    def fetch_player_fixture_difficulty(self, first_n_gws: int, 
                                      players: pd.DataFrame, 
                                      starting_gameweek: int) -> pd.DataFrame:
        """
        Calculate fixture difficulty for each player over the next N 
        gameweeks, with proper DGW/BGW detection and decaying weights for 
        future fixtures.

        Args:
            first_n_gws (int): Number of gameweeks to consider for fixture 
                              difficulty.
            players (pd.DataFrame): Player data containing team_id and 
                                  name_key
            starting_gameweek (int): The gameweek to start calculating from.

        Returns:
            pd.DataFrame: DataFrame with name_key, fixture info, and DGW 
                         flags.
        """
        fixtures = pd.DataFrame(self.fpl_client.get_fixtures())

        # Get team names for fixture CSV
        teams_data = self.fpl_client.get_bootstrap_static()
        teams_df = pd.DataFrame(
            teams_data["teams"])[["id", "name", "short_name"]]

        # Process and save fixture data
        self._save_fixture_data(fixtures, teams_df)

        # Filter fixtures for the specified gameweek range
        end_gameweek = starting_gameweek + first_n_gws - 1
        fixtures = fixtures[
            (pd.to_numeric(
            fixtures["event"], errors="coerce") >= starting_gameweek)
            & (pd.to_numeric(
            fixtures["event"], errors="coerce") <= end_gameweek)
        ]

        # Calculate player difficulties with DGW/BGW detection and decay 
        # weights
        return self._calculate_player_difficulties_with_dgw_detection_and_decay(
            fixtures, players, teams_df, starting_gameweek, end_gameweek, 
            first_n_gws
        )
    
    def _save_fixture_data(
            self, fixtures: pd.DataFrame, teams_df: pd.DataFrame):
        """Process and save fixture data to CSV."""
        # Create a copy of fixtures for saving to CSV
        fixtures_for_csv = fixtures.copy()

        # Add team names to fixtures
        fixtures_for_csv = fixtures_for_csv.merge(
            teams_df.rename(
                columns={
                    "id": "team_h",
                    "name": "home_team",
                    "short_name": "home_team_short",
                }
            ),
            on="team_h",
            how="left",
        )
        fixtures_for_csv = fixtures_for_csv.merge(
            teams_df.rename(
                columns={
                    "id": "team_a",
                    "name": "away_team",
                    "short_name": "away_team_short",
                }
            ),
            on="team_a",
            how="left",
        )

        # Select and reorder columns for the CSV
        fixtures_csv = fixtures_for_csv[
            [
                "id",
                "event",
                "kickoff_time",
                "home_team",
                "away_team",
                "home_team_short",
                "away_team_short",
                "team_h_difficulty",
                "team_a_difficulty",
                "team_h_score",
                "team_a_score",
                "finished",
            ]
        ].copy()

        # Rename columns for clarity
        fixtures_csv = fixtures_csv.rename(
            columns={
                "id": "fixture_id",
                "event": "gameweek",
                "team_h_difficulty": "home_difficulty",
                "team_a_difficulty": "away_difficulty",
                "team_h_score": "home_score",
                "team_a_score": "away_score",
            }
        )

        # Sort by gameweek and kickoff time
        fixtures_csv = fixtures_csv.sort_values(["gameweek", "kickoff_time"])

        # Save fixtures to CSV
        os.makedirs("data", exist_ok=True)
        fixtures_csv.to_csv("data/fixtures.csv", index=False)
    
    def _calculate_player_difficulties_with_dgw_detection_and_decay(
            self, fixtures: pd.DataFrame, players: pd.DataFrame, 
            teams_df: pd.DataFrame, starting_gameweek: int, 
            end_gameweek: int, num_gameweeks: int) -> pd.DataFrame:
        """
        Calculate player difficulties using clean DGW/BGW detection logic 
        and decaying weights for future fixtures.
        """
        # Calculate decay weights
        decay_weights = self._calculate_decay_weights(num_gameweeks)
        
        # Detect DGW/BGW teams across the gameweek range
        dgw_bgw_info = self._detect_dgw_bgw_teams(
            fixtures, teams_df, starting_gameweek, end_gameweek
        )
        
        # Calculate weighted fixture difficulties
        player_diffs = []
        
        # Group fixtures by gameweek for weight application
        for gw_offset, weight in enumerate(decay_weights):
            current_gw = starting_gameweek + gw_offset
            gw_fixtures = fixtures[fixtures["event"] == current_gw]
            
            for _, fixture_row in gw_fixtures.iterrows():
                for home_away, team_col, diff_col in [
                    ("home", "team_h", "team_h_difficulty"),
                    ("away", "team_a", "team_a_difficulty"),
                ]:
                    team_id = fixture_row[team_col]
                    diff = fixture_row[diff_col]
                    
                    # Get players for this team
                    team_players = players[players["team_id"] == team_id]
                    for _, player_row in team_players.iterrows():
                        player_diffs.append({
                            "name_key": player_row["name_key"],
                            "team_id": team_id,
                            "gameweek": current_gw,
                            "diff": diff,
                            "weight": weight
                        })

        if not player_diffs:
            print(f"Warning: No fixtures found for gameweeks "
                  f"{starting_gameweek} to {end_gameweek}")
            return pd.DataFrame(columns=[
                "name_key", "diff", "fixture_bonus", "has_dgw", "has_bgw"
            ])

        df = pd.DataFrame(player_diffs)
        
        # Calculate weighted average difficulty per player
        weighted_avg = df.groupby("name_key", group_keys=False).apply(
            lambda group: pd.Series({
                "diff": (group["diff"] * group["weight"]).sum() / 
                        group["weight"].sum(),
                "team_id": group["team_id"].iloc[0]  # All should be same team
            }), include_groups=False
        ).reset_index()
        
        # Add DGW/BGW information
        weighted_avg = weighted_avg.merge(dgw_bgw_info, on="team_id", 
                                        how="left")
        weighted_avg["has_dgw"] = weighted_avg["has_dgw"].fillna(False)
        weighted_avg["has_bgw"] = weighted_avg["has_bgw"].fillna(False)
        weighted_avg["fixture_multiplier"] = weighted_avg[
            "fixture_multiplier"].fillna(1.0)
        
        # Calculate fixture bonus accounting for DGW/BGW
        weighted_avg["fixture_bonus"] = (
            (6 - weighted_avg["diff"]) * weighted_avg["fixture_multiplier"]
        )
        
        return weighted_avg[["name_key", "diff", "fixture_bonus", "has_dgw", 
                            "has_bgw", "fixture_multiplier"]]
    
    def _detect_dgw_bgw_teams(self, fixtures: pd.DataFrame, teams_df: pd.DataFrame,
                             starting_gameweek: int, end_gameweek: int) -> pd.DataFrame:
        """
        Detect DGW/BGW teams using the clean counting method.
        
        Returns:
            pd.DataFrame: Team analysis with DGW/BGW flags and multipliers
        """
        team_analysis = []
        
        for gw in range(starting_gameweek, end_gameweek + 1):
            gw_fixtures = fixtures[fixtures["event"] == gw]
            
            if gw_fixtures.empty:
                continue
            
            # Get all teams in this gameweek
            home_teams = gw_fixtures["team_h"].tolist()
            away_teams = gw_fixtures["team_a"].tolist()
            all_teams_in_gw = home_teams + away_teams
            
            # Count appearances for each team
            team_counts = pd.Series(all_teams_in_gw).value_counts()
            
            # Analyze each team
            for team_id in teams_df["id"]:
                appearances = team_counts.get(team_id, 0)
                
                if appearances == 2:
                    # Double gameweek (team appears twice = 2 fixtures)
                    team_analysis.append({
                        "team_id": team_id,
                        "gameweek": gw,
                        "appearances": appearances,
                        "has_dgw": True,
                        "has_bgw": False,
                        "fixture_multiplier": 2.0
                    })
                elif appearances == 1:
                    # Normal gameweek (team appears once = 1 fixture)
                    team_analysis.append({
                        "team_id": team_id,
                        "gameweek": gw,
                        "appearances": appearances,
                        "has_dgw": False,
                        "has_bgw": False,
                        "fixture_multiplier": 1.0
                    })
                elif appearances == 0:
                    # Blank gameweek (team doesn't appear = 0 fixtures)
                    team_analysis.append({
                        "team_id": team_id,
                        "gameweek": gw,
                        "appearances": appearances,
                        "has_dgw": False,
                        "has_bgw": True,
                        "fixture_multiplier": 0.0
                    })
                else:
                    # Unexpected count - log for debugging
                    print(f"Warning: Team {team_id} has {appearances} "
                          f"appearances in GW{gw} (expected 0, 1, or 2)")
        
        if not team_analysis:
            # No teams found - return empty with correct structure
            return pd.DataFrame(columns=[
                "team_id", "has_dgw", "has_bgw", "fixture_multiplier"
            ])
        
        # Aggregate across gameweeks for overall DGW/BGW status
        analysis_df = pd.DataFrame(team_analysis)
        team_summary = analysis_df.groupby("team_id").agg({
            "has_dgw": "any",  # True if any DGW in the range
            "has_bgw": "any",  # True if any BGW in the range
            "fixture_multiplier": "sum"  # Sum multipliers across gameweeks
        }).reset_index()
        
        # Debug output
        dgw_teams = team_summary[team_summary["has_dgw"] == True]
        bgw_teams = team_summary[team_summary["has_bgw"] == True]
        
        if len(dgw_teams) > 0:
            team_names = teams_df[teams_df["id"].isin(
                dgw_teams["team_id"])]["name"].tolist()
            print(f"DGW teams in GW{starting_gameweek}-{end_gameweek}: "
                  f"{', '.join(team_names)}")
        
        if len(bgw_teams) > 0:
            team_names = teams_df[teams_df["id"].isin(
                bgw_teams["team_id"])]["name"].tolist()
            print(f"BGW teams in GW{starting_gameweek}-{end_gameweek}: "
                  f"{', '.join(team_names)}")
        
        return team_summary
    
    def add_next_fixture(self, df: pd.DataFrame, 
                        target_gameweek: int) -> pd.DataFrame:
        """
        Add fixture information for a specific gameweek for each player,
        handling multiple fixtures (DGW) and blank gameweeks (BGW).

        Args:
            df (pd.DataFrame): Player data to add fixture info to.
            target_gameweek (int): The gameweek number to get fixtures for.

        Returns:
            pd.DataFrame: Player data with opponent, venue, and fixture
                         difficulty columns added for the target gameweek.
        """
        fixtures = pd.DataFrame(self.fpl_client.get_fixtures())
        teams_data = self.fpl_client.get_bootstrap_static()
        teams_df = pd.DataFrame(teams_data["teams"])
        
        # Filter for target gameweek
        gw_fixtures = fixtures[fixtures["event"] == target_gameweek]
        
        if gw_fixtures.empty:
            print(f"Warning: No fixtures found for gameweek "
                  f"{target_gameweek}")
            return self._add_empty_fixture_info(df)

        # Detect DGW/BGW for this specific gameweek
        dgw_bgw_info = self._detect_dgw_bgw_teams(
            gw_fixtures, teams_df, target_gameweek, target_gameweek
        )

        # Process fixtures for each player
        next_fixtures = self._process_gameweek_fixtures(
            df, gw_fixtures, dgw_bgw_info
        )
        
        # Add team names to fixtures
        return self._add_team_names_to_fixtures(df, next_fixtures, teams_df)
    
    def _add_empty_fixture_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add empty fixture information when no fixtures found."""
        df["next_opponent"] = "No fixture"
        df["venue"] = "N/A"
        df["fixture_difficulty"] = None
        df["has_dgw_next"] = False
        df["has_bgw_next"] = True
        return df
    
    def _process_gameweek_fixtures(self, df: pd.DataFrame, 
                                  gw_fixtures: pd.DataFrame,
                                  dgw_bgw_info: pd.DataFrame) -> list:
        """Process fixtures for each player in the target gameweek."""
        next_fixtures = []

        for _, player_row in df.iterrows():
            team_id = player_row["team_id"]
            
            # Get DGW/BGW status for this team
            team_info = dgw_bgw_info[dgw_bgw_info["team_id"] == team_id]
            has_dgw = team_info["has_dgw"].iloc[0] if len(team_info) > 0 else False
            has_bgw = team_info["has_bgw"].iloc[0] if len(team_info) > 0 else False
            
            # Find fixtures for this team in this gameweek
            team_fixtures = gw_fixtures[
                (gw_fixtures["team_h"] == team_id) | 
                (gw_fixtures["team_a"] == team_id)
            ]
            
            if len(team_fixtures) == 0 or has_bgw:
                # Blank gameweek
                next_fixtures.append({
                    "name_key": player_row["name_key"],
                    "next_opponent_ids": [],
                    "venues": [],
                    "fixture_difficulties": [],
                    "has_dgw_next": False,
                    "has_bgw_next": True,
                })
            else:
                # Process fixtures (single or multiple)
                opponent_ids = []
                venues = []
                difficulties = []
                
                for _, fixture in team_fixtures.iterrows():
                    if fixture["team_h"] == team_id:
                        # Home game
                        opponent_ids.append(fixture["team_a"])
                        venues.append("Home")
                        difficulties.append(fixture["team_h_difficulty"])
                    else:
                        # Away game
                        opponent_ids.append(fixture["team_h"])
                        venues.append("Away")
                        difficulties.append(fixture["team_a_difficulty"])
                
                next_fixtures.append({
                    "name_key": player_row["name_key"],
                    "next_opponent_ids": opponent_ids,
                    "venues": venues,
                    "fixture_difficulties": difficulties,
                    "has_dgw_next": has_dgw,
                    "has_bgw_next": False,
                })
        
        return next_fixtures
    
    def _add_team_names_to_fixtures(self, df: pd.DataFrame, 
                                   next_fixtures: list,
                                   teams_df: pd.DataFrame) -> pd.DataFrame:
        """Add team names to fixture information."""
        nf_df = pd.DataFrame(next_fixtures)
        
        # Process each player's fixtures
        for idx, row in nf_df.iterrows():
            if not row["next_opponent_ids"] or row.get("has_bgw_next", False):
                # Blank gameweek
                nf_df.at[idx, "next_opponent"] = "Blank GW"
                nf_df.at[idx, "venue"] = "N/A"
                nf_df.at[idx, "fixture_difficulty"] = None
            elif len(row["next_opponent_ids"]) == 1:
                # Single gameweek
                opponent_id = row["next_opponent_ids"][0]
                opponent_name = teams_df[teams_df["id"] == opponent_id][
                    "name"
                ].iloc[0]
                nf_df.at[idx, "next_opponent"] = opponent_name
                nf_df.at[idx, "venue"] = row["venues"][0]
                nf_df.at[idx, "fixture_difficulty"] = row[
                    "fixture_difficulties"
                ][0]
            else:
                # Double gameweek - combine opponent names
                opponent_names = []
                for opponent_id in row["next_opponent_ids"]:
                    opponent_name = teams_df[teams_df["id"] == opponent_id][
                        "name"
                    ].iloc[0]
                    opponent_names.append(opponent_name)
                
                nf_df.at[idx, "next_opponent"] = " & ".join(opponent_names)
                nf_df.at[idx, "venue"] = " & ".join(row["venues"])
                nf_df.at[idx, "fixture_difficulty"] = sum(
                    row["fixture_difficulties"]
                ) / len(row["fixture_difficulties"])

        # Merge with original dataframe
        df = df.merge(
            nf_df[["name_key", "next_opponent", "venue", 
                   "fixture_difficulty", "has_dgw_next", "has_bgw_next"]],
            on="name_key",
            how="left",
        )
        
        # Fill any missing values
        df["next_opponent"] = df["next_opponent"].fillna("No fixture")
        df["venue"] = df["venue"].fillna("N/A")
        df["has_dgw_next"] = df["has_dgw_next"].fillna(False)
        df["has_bgw_next"] = df["has_bgw_next"].fillna(False)
        
        return df