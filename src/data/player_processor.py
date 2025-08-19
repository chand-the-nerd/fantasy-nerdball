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
        Calculate current season xG performance ratios for all players.
        
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
        
        # Current season thresholds based on gameweeks completed
        # For early season (GW1-3), we need very low thresholds
        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        
        # Adjust thresholds based on games played so far
        if gameweeks_completed == 1:
            # After 1 GW: Just need any xG data and player to have started
            min_xg_threshold = 0.01  # Almost any xG involvement
            min_xgc_threshold = 0.01  # Almost any defensive involvement
            min_games_started = 1  # Must have started at least 1 game
        elif gameweeks_completed <= 3:
            # After 2-3 GWs: Slightly higher thresholds
            min_xg_threshold = 0.1
            min_xgc_threshold = 0.1
            min_games_started = 1
        else:
            # After 4+ GWs: More meaningful sample size needed
            min_xg_threshold = 0.3
            min_xgc_threshold = 0.3
            min_games_started = 2
        
        print(f"🔍 Analyzing current season xG performance (after {gameweeks_completed} GW)...")
        print(f"   Thresholds: xGI≥{min_xg_threshold}, xGC≥{min_xgc_threshold}, starts≥{min_games_started}")
        
        # Goal involvement performance (for forwards and midfielders)
        xgi_mask = (
            (df["expected_goal_involvements"] > min_xg_threshold) & 
            (df["starts"] >= min_games_started)
        )
        xgi_count = xgi_mask.sum()
        
        if xgi_count > 0:
            print(f"   Found {xgi_count} players with sufficient goal involvement data")
            actual_gi = df.loc[xgi_mask, "goals_scored"] + df.loc[xgi_mask, "assists"]
            expected_gi = df.loc[xgi_mask, "expected_goal_involvements"]
            
            # Debug output for first few players
            debug_players = df.loc[xgi_mask].head(3)
            for idx, player in debug_players.iterrows():
                actual = player["goals_scored"] + player["assists"]
                expected = player["expected_goal_involvements"]
                ratio = actual / expected if expected > 0 else 1.0
                print(f"   {player['display_name']}: {actual:.1f} actual vs {expected:.2f} expected = {ratio:.2f}")
            
            df.loc[xgi_mask, "current_xOP"] = (actual_gi / expected_gi).clip(0.2, 3.0).round(2)
            df.loc[xgi_mask, "current_xg_context"] = "goal_involvement"
        else:
            print(f"   No players found with sufficient goal involvement data (threshold: {min_xg_threshold} xGI, starts≥{min_games_started})")
        
        # Goals conceded performance (for goalkeepers and defenders)  
        xgc_mask = (
            (df["expected_goals_conceded"] > min_xgc_threshold) & 
            (df["starts"] >= min_games_started)
        )
        xgc_count = xgc_mask.sum()
        
        if xgc_count > 0:
            print(f"   Found {xgc_count} players with sufficient defensive data")
            
            # Debug output for first few players
            debug_players = df.loc[xgc_mask].head(3)
            for idx, player in debug_players.iterrows():
                expected_gc = player["expected_goals_conceded"]
                actual_gc = max(0.1, player["goals_conceded"])
                ratio = expected_gc / actual_gc
                print(f"   {player['display_name']}: {actual_gc:.1f} goals conceded vs {expected_gc:.2f} expected = {ratio:.2f}")
            
            # For GC, higher ratio = better performance (conceding less than expected)
            df.loc[xgc_mask, "current_xOP"] = (
                df.loc[xgc_mask, "expected_goals_conceded"] / 
                df.loc[xgc_mask, "goals_conceded"].clip(lower=0.1)  # Avoid division by zero
            ).clip(0.2, 3.0).round(2)
            df.loc[xgc_mask, "current_xg_context"] = "defensive"
        else:
            print(f"   No players found with sufficient defensive data (threshold: {min_xgc_threshold} xGC, starts≥{min_games_started})")
        
        # Add xG trend display for easy interpretation
        df["xg_trend"] = df.apply(lambda row: self._format_xg_trend(row), axis=1)
        
        # Summary output
        sufficient_data_count = (df["current_xg_context"] != "insufficient_data").sum()
        print(f"   ✅ {sufficient_data_count} players have current season xG analysis")
        
        return df
        
        # Goals conceded performance (for goalkeepers and defenders)  
        xgc_mask = (df["expected_goals_conceded"] > min_xgc_threshold) & (df["minutes_played"] > min_minutes)
        xgc_count = xgc_mask.sum()
        
        if xgc_count > 0:
            print(f"   Found {xgc_count} players with sufficient defensive data")
            
            # Debug output for first few players
            debug_players = df.loc[xgc_mask].head(3)
            for idx, player in debug_players.iterrows():
                expected_gc = player["expected_goals_conceded"]
                actual_gc = max(0.1, player["goals_conceded"])  # Avoid division by zero
                ratio = expected_gc / actual_gc
                print(f"   {player['display_name']}: {actual_gc:.1f} goals conceded vs {expected_gc:.2f} expected = {ratio:.2f}")
            
            # For GC, higher ratio = better performance (conceding less than expected)
            df.loc[xgc_mask, "current_xOP"] = (
                df.loc[xgc_mask, "expected_goals_conceded"] / 
                df.loc[xgc_mask, "goals_conceded"].clip(lower=0.1)  # Avoid division by zero
            ).clip(0.2, 3.0)
            df.loc[xgc_mask, "current_xg_context"] = "defensive"
        else:
            print(f"   No players found with sufficient defensive data (threshold: {min_xgc_threshold} xGC, {min_minutes} mins)")
        
        # Add xG trend display for easy interpretation
        df["xg_trend"] = df.apply(lambda row: self._format_xg_trend(row), axis=1)
        
        # Summary output
        sufficient_data_count = (df["current_xg_context"] != "insufficient_data").sum()
        print(f"   ✅ {sufficient_data_count} players have current season xG analysis")
        
        return df
    
    def _format_xg_trend(self, row):
        """Format xG trend for display."""
        if row.get('current_xg_context') == 'insufficient_data':
            return "N/A"
        
        current_ratio = row.get('current_xOP', 1.0)
        position = row.get('position', 'MID')
        
        if position in ['FWD', 'MID'] and row.get('current_xg_context') == 'goal_involvement':
            if current_ratio > 1.2:
                return f"🔥{current_ratio:.2f}"  # Hot streak
            elif current_ratio > 1.1:
                return f"↗️{current_ratio:.2f}"  # Over-performing
            elif current_ratio < 0.8:
                return f"📈{current_ratio:.2f}"  # Due regression
            elif current_ratio < 0.9:
                return f"↘️{current_ratio:.2f}"  # Under-performing
            else:
                return f"➡️{current_ratio:.2f}"  # Normal
        
        elif position in ['DEF', 'GK'] and row.get('current_xg_context') == 'defensive':
            if current_ratio > 1.2:
                return f"🛡️{current_ratio:.2f}"  # Excellent defense
            elif current_ratio > 1.1:
                return f"↗️{current_ratio:.2f}"  # Good defense
            elif current_ratio < 0.8:
                return f"📈{current_ratio:.2f}"  # Due improvement
            elif current_ratio < 0.9:
                return f"↘️{current_ratio:.2f}"  # Poor defense
            else:
                return f"➡️{current_ratio:.2f}"  # Normal
        
        return "N/A"
    
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