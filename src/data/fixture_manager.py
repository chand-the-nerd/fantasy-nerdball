"""Module for managing fixture data and difficulty calculations."""

import os
import pandas as pd
from ..api.fpl_client import FPLClient


class FixtureManager:
    """Handles fixture data and difficulty calculations."""
    
    def __init__(self, config):
        self.config = config
        self.fpl_client = FPLClient()
    
    def fetch_player_fixture_difficulty(self, first_n_gws: int, players: pd.DataFrame, starting_gameweek: int) -> pd.DataFrame:
        """
        Calculate average fixture difficulty for each player over the next
        N gameweeks from a starting gameweek and save fixture data to CSV.

        Args:
            first_n_gws (int): Number of gameweeks to consider for fixture difficulty.
            players (pd.DataFrame): Player data containing team_id and name_key.
            starting_gameweek (int): The gameweek to start calculating from.

        Returns:
            pd.DataFrame: DataFrame with name_key, average difficulty, and fixture bonus (6 - difficulty).
        """
        fixtures = pd.DataFrame(self.fpl_client.get_fixtures())

        # Get team names for fixture CSV
        teams_data = self.fpl_client.get_bootstrap_static()
        teams_df = pd.DataFrame(teams_data["teams"])[["id", "name", "short_name"]]

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
        print(f"Fixture data saved to data/fixtures.csv")

        # Filter fixtures for the specified gameweek range
        end_gameweek = starting_gameweek + first_n_gws - 1
        fixtures = fixtures[
            (pd.to_numeric(fixtures["event"], errors="coerce") >= starting_gameweek)
            & (pd.to_numeric(fixtures["event"], errors="coerce") <= end_gameweek)
        ]

        print(
            f"Calculating fixture difficulty for gameweeks {starting_gameweek} "
            f"to {end_gameweek}"
        )

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
                        {"name_key": p["name_key"], "gw": row["event"], "diff": diff}
                    )

        df = pd.DataFrame(player_diffs)
        if df.empty:
            print(
                f"Warning: No fixtures found for gameweeks {starting_gameweek} "
                f"to {end_gameweek}"
            )
            # Return empty dataframe with correct structure
            return pd.DataFrame(columns=["name_key", "diff", "fixture_bonus"])

        avg_diff = df.groupby("name_key", as_index=False)["diff"].mean()
        avg_diff["fixture_bonus"] = 6 - avg_diff["diff"]  # higher is better
        return avg_diff
    
    def add_next_fixture(self, df: pd.DataFrame, target_gameweek: int) -> pd.DataFrame:
        """
        Add fixture information for a specific gameweek for each player.

        Args:
            df (pd.DataFrame): Player data to add fixture info to.
            target_gameweek (int): The gameweek number to get fixtures for.

        Returns:
            pd.DataFrame: Player data with opponent, venue, and fixture
                         difficulty columns added for the target gameweek.
        """
        fixtures = pd.DataFrame(self.fpl_client.get_fixtures())
        
        # Filter for target gameweek
        gw_fixtures = fixtures[fixtures["event"] == target_gameweek]
        
        if gw_fixtures.empty:
            print(f"Warning: No fixtures found for gameweek {target_gameweek}")
            df["next_opponent"] = "No fixture"
            df["venue"] = "N/A"
            df["fixture_difficulty"] = None
            return df

        next_fixtures = []

        for _, p in df.iterrows():
            team_id = p["team_id"]
            
            # Find fixtures for this team
            team_fixtures = gw_fixtures[
                (gw_fixtures["team_h"] == team_id) | (gw_fixtures["team_a"] == team_id)
            ]
            
            if not team_fixtures.empty:
                fixture = team_fixtures.iloc[0]  # Take first fixture if multiple
                
                if fixture["team_h"] == team_id:
                    # Home game
                    opponent_id = fixture["team_a"]
                    venue = "Home"
                    difficulty = fixture["team_h_difficulty"]
                else:
                    # Away game
                    opponent_id = fixture["team_h"]
                    venue = "Away"
                    difficulty = fixture["team_a_difficulty"]
                
                next_fixtures.append(
                    {
                        "name_key": p["name_key"],
                        "next_opponent_id": opponent_id,
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
            self.fpl_client.get_bootstrap_static()["teams"]
        )
        
        # Only merge for players who have fixtures (next_opponent_id is not None)
        has_fixture_mask = nf_df["next_opponent_id"].notna()
        
        if has_fixture_mask.any():
            # Merge only the rows with fixtures
            nf_with_fixtures = nf_df[has_fixture_mask].copy()
            nf_with_fixtures = nf_with_fixtures.merge(
                teams[["id", "name"]], left_on="next_opponent_id", right_on="id", how="left"
            )
            nf_with_fixtures = nf_with_fixtures.rename(columns={"name": "next_opponent"})
            
            # Combine back with rows without fixtures
            nf_without_fixtures = nf_df[~has_fixture_mask].copy()
            nf_without_fixtures["next_opponent"] = "No fixture"
            
            # Concatenate the results
            nf_df = pd.concat([nf_with_fixtures, nf_without_fixtures], ignore_index=True)
        else:
            # No fixtures at all for this gameweek
            nf_df["next_opponent"] = "No fixture"
        
        # Ensure next_opponent column exists and fill any remaining NaN values
        nf_df["next_opponent"] = nf_df["next_opponent"].fillna("No fixture")

        df = df.merge(
            nf_df[["name_key", "next_opponent", "venue", "fixture_difficulty"]],
            on="name_key",
            how="left",
        )
        return df