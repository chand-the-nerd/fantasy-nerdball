"""Module for processing player data from the FPL API."""

import os
import pandas as pd
import numpy as np
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
        Enhanced with current season xG performance analysis.

        Returns:
            pd.DataFrame: DataFrame containing current player data with calculated
                         fields for cost, position, team info, name keys, and xG metrics.
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

        # Basic metrics - calculate these FIRST before xG analysis
        players["minutes_played"] = pd.to_numeric(players["minutes"], errors="coerce").fillna(0)
        players["games_played"] = (players["minutes_played"] / 90).round().clip(lower=0)

        # === ENHANCED xG ANALYSIS FOR CURRENT SEASON ===
        players = self._calculate_current_season_xg_performance(players)

        # Save to CSV
        os.makedirs("data", exist_ok=True)
        players.to_csv("data/players.csv", index=False)
        print("Player data saved to data/players.csv")

        return players
    
    def _calculate_current_season_xg_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate current season xG performance ratios for all players with position-weighted analysis.
        Uses per-game metrics to match historical calculations.
        
        Args:
            df (pd.DataFrame): Player dataframe with xG stats
            
        Returns:
            pd.DataFrame: Enhanced dataframe with current season xG performance metrics
        """
        # Extract xG metrics with safe conversion
        xg_columns = {
            'expected_goals': 'expected_goals',
            'expected_assists': 'expected_assists',
            'expected_goal_involvements': 'expected_goal_involvements', 
            'expected_goals_conceded': 'expected_goals_conceded',
            'goals_scored': 'goals_scored',
            'assists': 'assists',
            'goals_conceded': 'goals_conceded'
        }
        
        for col_name, df_col in xg_columns.items():
            if df_col in df.columns:
                df[col_name] = pd.to_numeric(df[df_col], errors="coerce").fillna(0)
            else:
                df[col_name] = 0.0
                print(f"Warning: {df_col} not found in current season data")
        
        # Initialize granular xG performance columns
        df["current_xOP"] = 1.0  # Current season Expected Overperformance ratio
        df["current_xg_context"] = "insufficient_data"
        df["attacking_xOP"] = 1.0  # Separate tracking for debugging
        df["defensive_xOP"] = 1.0  # Separate tracking for debugging
        
        # Current season thresholds based on gameweeks completed
        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        
        # Adjust thresholds based on games played so far
        if gameweeks_completed == 1:
            # After 1 GW: Just need any xG data and player to have started
            min_xg_threshold = 0.01  # Almost any xG involvement per game
            min_xgc_threshold = 0.01  # Almost any defensive involvement per game
            min_games_started = 1  # Must have started at least 1 game
        elif gameweeks_completed <= 3:
            # After 2-3 GWs: Slightly higher thresholds
            min_xg_threshold = 0.05
            min_xgc_threshold = 0.05
            min_games_started = 1
        else:
            # After 4+ GWs: More meaningful sample size needed
            min_xg_threshold = 0.15
            min_xgc_threshold = 0.15
            min_games_started = 2
        
        # Calculate per-game metrics for current season to match historical approach
        df["current_xgi_per_game"] = 0.0
        df["current_gi_per_game"] = 0.0  
        df["current_xgc_per_game"] = 0.0
        df["current_gc_per_game"] = 0.0

        # Only calculate for players who have started games
        current_played_mask = df["starts"] >= 1

        if current_played_mask.any():
            df.loc[current_played_mask, "current_xgi_per_game"] = (
                df.loc[current_played_mask, "expected_goal_involvements"] / 
                df.loc[current_played_mask, "starts"]
            )
            df.loc[current_played_mask, "current_gi_per_game"] = (
                (df.loc[current_played_mask, "goals_scored"] + df.loc[current_played_mask, "assists"]) / 
                df.loc[current_played_mask, "starts"]
            )
            df.loc[current_played_mask, "current_xgc_per_game"] = (
                df.loc[current_played_mask, "expected_goals_conceded"] / 
                df.loc[current_played_mask, "starts"]
            )
            df.loc[current_played_mask, "current_gc_per_game"] = (
                df.loc[current_played_mask, "goals_conceded"] / 
                df.loc[current_played_mask, "starts"]
            )
        
        # Position-based weighting system
        position_weights = {
            "FWD": {"attacking": 1.0, "defensive": 0.0},
            "MID": {"attacking": 0.75, "defensive": 0.25},
            "DEF": {"attacking": 0.25, "defensive": 0.75},
            "GK": {"attacking": 0.0, "defensive": 1.0}
        }
        
        # Calculate attacking xG performance using per-game metrics
        attacking_mask = (
            (df["current_xgi_per_game"] > min_xg_threshold) & 
            (df["starts"] >= min_games_started) &
            (df["position"].isin(["FWD", "MID", "DEF"]))  # All except GK can have attacking component
        )
        
        attacking_count = attacking_mask.sum()
        if attacking_count > 0:
            df.loc[attacking_mask, "attacking_xOP"] = (
                df.loc[attacking_mask, "current_gi_per_game"] / 
                df.loc[attacking_mask, "current_xgi_per_game"]
            ).clip(0.2, 3.0).round(2)
        
        # Calculate defensive xG performance using per-game metrics
        defensive_mask = (
            (df["current_xgc_per_game"] > min_xgc_threshold) & 
            (df["starts"] >= min_games_started) &
            (df["position"].isin(["GK", "DEF", "MID"]))  # All except FWD can have defensive component
        )
        
        defensive_count = defensive_mask.sum()
        if defensive_count > 0:
            # For GC, higher ratio = better performance (conceding less than expected)
            df.loc[defensive_mask, "defensive_xOP"] = (
                df.loc[defensive_mask, "current_xgc_per_game"] / 
                df.loc[defensive_mask, "current_gc_per_game"].clip(lower=0.01)  # Avoid division by zero
            ).clip(0.2, 3.0).round(2)
        
        # Calculate weighted current_xOP for each player based on their position
        players_with_data = 0
        
        for idx, row in df.iterrows():
            position = row["position"]
            if position not in position_weights:
                continue
                
            weights = position_weights[position]
            attacking_weight = weights["attacking"]
            defensive_weight = weights["defensive"]
            
            attacking_xop = row["attacking_xOP"]
            defensive_xop = row["defensive_xOP"]
            
            # Check if we have sufficient per-game data for each component
            has_attacking_data = (attacking_weight > 0 and 
                                row["current_xgi_per_game"] > min_xg_threshold and 
                                row["starts"] >= min_games_started)
            
            has_defensive_data = (defensive_weight > 0 and 
                                row["current_xgc_per_game"] > min_xgc_threshold and 
                                row["starts"] >= min_games_started)
            
            # Calculate weighted xOP based on available data
            if has_attacking_data and has_defensive_data:
                # Both components available - use full weighting
                weighted_xop = (attacking_xop * attacking_weight) + (defensive_xop * defensive_weight)
                context = f"mixed_{int(attacking_weight*100)}att_{int(defensive_weight*100)}def"
                players_with_data += 1
                
            elif has_attacking_data and attacking_weight > 0:
                # Only attacking data available - use attacking component only
                weighted_xop = attacking_xop
                context = "attacking_only"
                players_with_data += 1
                
            elif has_defensive_data and defensive_weight > 0:
                # Only defensive data available - use defensive component only
                weighted_xop = defensive_xop
                context = "defensive_only"
                players_with_data += 1
                
            else:
                # Insufficient data
                weighted_xop = 1.0
                context = "insufficient_data"
            
            df.loc[idx, "current_xOP"] = round(weighted_xop, 2)
            df.loc[idx, "current_xg_context"] = context
        
        # Add xG trend display for easy interpretation
        df["xg_trend"] = df.apply(lambda row: self._format_xg_trend(row), axis=1)
        
        # Summary output with position breakdown
        print(f"   ✅ {players_with_data} players have current season xG analysis")
        
        # Show breakdown by position and context
        context_summary = df[df["current_xg_context"] != "insufficient_data"].groupby(
            ["position", "current_xg_context"]
        ).size().reset_index(name="count")
        
        
        # Clean up temporary columns
        df = df.drop(columns=["attacking_xOP", "defensive_xOP", "current_xgi_per_game", 
                             "current_gi_per_game", "current_xgc_per_game", "current_gc_per_game"])
        
        return df
        
    def _format_xg_trend(self, row):
        """Format xG trend for display with position-aware interpretation."""
        if row.get('current_xg_context') == 'insufficient_data':
            return "N/A"
        
        current_ratio = row.get('current_xOP', 1.0)
        position = row.get('position', 'MID')
        context = row.get('current_xg_context', '')
        
        # Interpretation depends on position and context
        if 'att' in context or context == 'attacking_only':
            # Attacking performance interpretation
            if current_ratio > 1.2:
                return f"🔥{current_ratio:.2f}"  # Hot attacking form
            elif current_ratio > 1.1:
                return f"↗️{current_ratio:.2f}"  # Good attacking form
            elif current_ratio < 0.8:
                return f"📈{current_ratio:.2f}"  # Due attacking regression
            elif current_ratio < 0.9:
                return f"↘️{current_ratio:.2f}"  # Poor attacking form
            else:
                return f"➡️{current_ratio:.2f}"  # Normal attacking
                
        elif 'def' in context or context == 'defensive_only':
            # Defensive performance interpretation
            if current_ratio > 1.2:
                return f"🛡️{current_ratio:.2f}"  # Excellent defense
            elif current_ratio > 1.1:
                return f"↗️{current_ratio:.2f}"  # Good defense
            elif current_ratio < 0.8:
                return f"📈{current_ratio:.2f}"  # Due defensive improvement
            elif current_ratio < 0.9:
                return f"↘️{current_ratio:.2f}"  # Poor defense
            else:
                return f"➡️{current_ratio:.2f}"  # Normal defense
                
        elif 'mixed' in context:
            # Mixed performance - general interpretation
            if current_ratio > 1.15:
                return f"⭐{current_ratio:.2f}"  # Excellent overall
            elif current_ratio > 1.05:
                return f"↗️{current_ratio:.2f}"  # Good overall
            elif current_ratio < 0.85:
                return f"📈{current_ratio:.2f}"  # Due overall improvement
            elif current_ratio < 0.95:
                return f"↘️{current_ratio:.2f}"  # Poor overall
            else:
                return f"➡️{current_ratio:.2f}"  # Normal overall
        
        return f"➡️{current_ratio:.2f}"  # Default
    
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