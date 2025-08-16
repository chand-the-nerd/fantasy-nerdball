"""Module for processing player data from the FPL API."""

import os
import pandas as pd
from ..api.fpl_client import FPLClient
from ..utils.text_utils import normalize_name


class PlayerProcessor:
    """Handles fetching and processing of player data."""
    
    def __init__(self, config):
        self.config = config
        self.fpl_client = FPLClient()
    
    def fetch_current_players(self) -> pd.DataFrame:
        """
        Fetch current FPL player data from the API and prepare it for analysis.

        Returns:
            pd.DataFrame: DataFrame containing current player data with calculated
                         fields for cost, position, team info, and name keys.
        """
        data = self.fpl_client.get_bootstrap_static()
        players = pd.DataFrame(data["elements"])

        teams = pd.DataFrame(data["teams"])[["id", "name"]].rename(
            columns={"id": "team_id", "name": "team"}
        )

        pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

        players = players.rename(columns={"team": "team_id", "element_type": "pos_id"})
        players["position"] = players["pos_id"].map(pos_map)
        players = players.merge(teams, on="team_id", how="left")
        players["form"] = pd.to_numeric(players["form"], errors="coerce").fillna(0.0)
        players["now_cost_m"] = players["now_cost"] / 10.0
        players["display_name"] = players["web_name"]
        players["name_key"] = players["web_name"].map(normalize_name)

        # Save to CSV
        os.makedirs("data", exist_ok=True)
        players.to_csv("data/players.csv", index=False)
        print("Player data saved to data/players.csv")

        return players
    
    def calculate_budget_from_previous_squad(self, gameweek: int, current_players: pd.DataFrame) -> float:
        """
        Calculate available budget based on previous gameweek's squad value.

        Args:
            gameweek (int): Current gameweek number
            current_players (pd.DataFrame): Current player database with prices

        Returns:
            float: Available budget in millions, or default BUDGET if no previous squad
        """
        if gameweek <= 1:
            print(f"Using default budget: £{self.config.BUDGET:.1f}m (no previous squad)")
            return self.config.BUDGET

        prev_gw = gameweek - 1
        prev_squad_file = f"squads/gw{prev_gw}/full_squad.csv"

        if not os.path.exists(prev_squad_file):
            print(f"Using default budget: £{self.config.BUDGET:.1f}m (no previous squad file)")
            return self.config.BUDGET

        try:
            prev_squad = pd.read_csv(prev_squad_file)
            prev_squad_ids = self.match_players_to_current(prev_squad, current_players)

            if not prev_squad_ids:
                print(
                    f"Using default budget: £{self.config.BUDGET:.1f}m (could not match previous squad)"
                )
                return self.config.BUDGET

            # Calculate total value of previous squad using current prices
            prev_squad_current_df = current_players[
                current_players["id"].isin(prev_squad_ids)
            ]
            total_value = prev_squad_current_df["now_cost_m"].sum()

            print(f"Previous squad value: £{total_value:.1f}m")
            return total_value

        except Exception as e:
            print(f"Error calculating budget from previous squad: {e}")
            print(f"Using default budget: £{self.config.BUDGET:.1f}m")
            return self.config.BUDGET
    
    def match_players_to_current(self, prev_squad: pd.DataFrame, current_players: pd.DataFrame) -> list:
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
            prev_name = prev_player["display_name"].strip()
            prev_pos = prev_player["position"]
            prev_team = prev_player["team"]

            # Try to find matching player in current database
            matches = current_players[
                (current_players["position"] == prev_pos)
                & (current_players["team"] == prev_team)
                & (current_players["display_name"].str.strip() == prev_name)
            ]

            if len(matches) == 1:
                prev_player_ids.append(matches.iloc[0]["id"])
            elif len(matches) > 1:
                # Multiple matches, take the first one
                print(f"Warning: Multiple matches for {prev_name}, taking first match")
                prev_player_ids.append(matches.iloc[0]["id"])
            else:
                # Try fuzzy matching on name
                fuzzy_matches = current_players[
                    (current_players["position"] == prev_pos)
                    & (current_players["team"] == prev_team)
                    & (
                        current_players["display_name"].str.contains(
                            prev_name.split()[0], case=False, na=False
                        )
                    )
                ]

                if len(fuzzy_matches) > 0:
                    prev_player_ids.append(fuzzy_matches.iloc[0]["id"])
                else:
                    print(
                        f"Warning: Could not find current match for "
                        f"{prev_name} ({prev_pos}, {prev_team})"
                    )

        return prev_player_ids