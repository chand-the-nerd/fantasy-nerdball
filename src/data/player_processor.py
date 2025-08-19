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
        
        # Position-based weighting system for xG analysis
        self.position_weights = {
            "FWD": {"attacking": 1.0, "defensive": 0.0},
            "MID": {"attacking": 0.75, "defensive": 0.25},
            "DEF": {"attacking": 0.25, "defensive": 0.75},
            "GK": {"attacking": 0.0, "defensive": 1.0}
        }
    
    def fetch_current_players(self) -> pd.DataFrame:
        """
        Fetch current FPL player data from the API and prepare it for analysis.
        Enhanced with current season xG performance analysis.

        Returns:
            pd.DataFrame: DataFrame containing current player data with 
                         calculated fields for cost, position, team info, 
                         name keys, and xG metrics.
        """
        data = self.fpl_client.get_bootstrap_static()
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

        # Basic metrics - calculate these FIRST before xG analysis
        players["minutes_played"] = pd.to_numeric(
            players["minutes"], errors="coerce"
        ).fillna(0)
        players["games_played"] = (
            players["minutes_played"] / 90
        ).round().clip(lower=0)

        # Enhanced xG analysis for current season
        players = self._calculate_current_season_xg_performance(players)

        # Save to CSV
        os.makedirs("data", exist_ok=True)
        players.to_csv("data/players.csv", index=False)
        print("Player data saved to data/players.csv")

        return players
    
    def _calculate_current_season_xg_performance(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate current season xG performance ratios for all players 
        with position-weighted analysis.
        
        Args:
            df (pd.DataFrame): Player dataframe with xG stats
            
        Returns:
            pd.DataFrame: Enhanced dataframe with current season xG 
                         performance metrics
        """
        # Extract and validate xG metrics
        df = self._extract_xg_metrics(df)
        
        # Initialise xG performance columns
        df = self._initialise_xg_columns(df)
        
        # Calculate thresholds based on gameweeks completed
        thresholds = self._calculate_xg_thresholds()
        
        # Calculate per-game metrics for current season
        df = self._calculate_per_game_metrics(df)
        
        # Calculate position-weighted xG performance
        df = self._calculate_position_weighted_xop(df, thresholds)
        
        # Add xG trend display for easy interpretation
        df["xg_trend"] = df.apply(
            lambda row: self._format_xg_trend(row), axis=1)
        
        # Print summary
        players_with_data = len(
            df[df["current_xg_context"] != "insufficient_data"])
        print(f"   ✅ {players_with_data} players have current season xG "
              "analysis")
        
        # Clean up temporary columns
        df = self._cleanup_temporary_columns(df)
        
        return df
    
    def _extract_xg_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract xG metrics with safe conversion."""
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
                df[col_name] = pd.to_numeric(
                    df[df_col], errors="coerce").fillna(0)
            else:
                df[col_name] = 0.0
                print(f"Warning: {df_col} not found in current season data")
        
        return df
    
    def _initialise_xg_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initialise granular xG performance columns."""
        df["current_xOP"] = 1.0  # Current season Expected Overperformance 
        df["current_xg_context"] = "insufficient_data"
        df["attacking_xOP"] = 1.0  # Separate tracking for debugging
        df["defensive_xOP"] = 1.0  # Separate tracking for debugging
        return df
    
    def _calculate_xg_thresholds(self) -> dict:
        """Calculate xG thresholds based on gameweeks completed."""
        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        
        if gameweeks_completed == 1:
            return {
                'min_xg_threshold': 0.01,
                'min_xgc_threshold': 0.01,
                'min_games_started': 1
            }
        elif gameweeks_completed <= 3:
            return {
                'min_xg_threshold': 0.05,
                'min_xgc_threshold': 0.05,
                'min_games_started': 1
            }
        else:
            return {
                'min_xg_threshold': 0.15,
                'min_xgc_threshold': 0.15,
                'min_games_started': 2
            }
    
    def _calculate_per_game_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate per-game metrics for current season."""
        # Initialise per-game columns
        per_game_cols = [
            "current_xgi_per_game", "current_gi_per_game", 
            "current_xgc_per_game", "current_gc_per_game"
        ]
        for col in per_game_cols:
            df[col] = 0.0

        # Only calculate for players who have started games
        current_played_mask = df["starts"] >= 1

        if current_played_mask.any():
            df.loc[current_played_mask, "current_xgi_per_game"] = (
                df.loc[current_played_mask, "expected_goal_involvements"] / 
                df.loc[current_played_mask, "starts"]
            )
            df.loc[current_played_mask, "current_gi_per_game"] = (
                (df.loc[current_played_mask, "goals_scored"] + 
                 df.loc[current_played_mask, "assists"]) / 
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
        
        return df
    
    def _calculate_position_weighted_xop(self, df: pd.DataFrame, 
                                       thresholds: dict) -> pd.DataFrame:
        """Calculate position-weighted current xOP for each player."""
        # Calculate attacking xG performance
        attacking_mask = (
            (df["current_xgi_per_game"] > thresholds['min_xg_threshold']) & 
            (df["starts"] >= thresholds['min_games_started']) &
            (df["position"].isin(["FWD", "MID", "DEF"]))
        )
        
        if attacking_mask.any():
            df.loc[attacking_mask, "attacking_xOP"] = (
                df.loc[attacking_mask, "current_gi_per_game"] / 
                df.loc[attacking_mask, "current_xgi_per_game"]
            ).clip(0.2, 3.0).round(2)
        
        # Calculate defensive xG performance
        defensive_mask = (
            (df["current_xgc_per_game"] > thresholds['min_xgc_threshold']) & 
            (df["starts"] >= thresholds['min_games_started']) &
            (df["position"].isin(["GK", "DEF", "MID"]))
        )
        
        if defensive_mask.any():
            # For GC, higher ratio = better performance
            df.loc[defensive_mask, "defensive_xOP"] = (
                df.loc[defensive_mask, "current_xgc_per_game"] / 
                df.loc[defensive_mask, "current_gc_per_game"].clip(lower=0.01)
            ).clip(0.2, 3.0).round(2)
        
        # Calculate weighted current_xOP for each player
        for idx, row in df.iterrows():
            position = row["position"]
            if position not in self.position_weights:
                continue
                
            weights = self.position_weights[position]
            weighted_xop, context = self._calculate_weighted_xop(
                row, weights, thresholds
            )
            
            df.loc[idx, "current_xOP"] = round(weighted_xop, 2)
            df.loc[idx, "current_xg_context"] = context
        
        return df
    
    def _calculate_weighted_xop(self, row: pd.Series, weights: dict, 
                              thresholds: dict) -> tuple:
        """Calculate weighted xOP for a single player."""
        attacking_weight = weights["attacking"]
        defensive_weight = weights["defensive"]
        
        attacking_xop = row["attacking_xOP"]
        defensive_xop = row["defensive_xOP"]
        
        # Check if we have sufficient per-game data for each component
        has_attacking_data = (
            attacking_weight > 0 and 
            row["current_xgi_per_game"] > thresholds['min_xg_threshold'] and 
            row["starts"] >= thresholds['min_games_started']
        )
        
        has_defensive_data = (
            defensive_weight > 0 and 
            row["current_xgc_per_game"] > thresholds['min_xgc_threshold'] and 
            row["starts"] >= thresholds['min_games_started']
        )
        
        # Calculate weighted xOP based on available data
        if has_attacking_data and has_defensive_data:
            # Both components available - use full weighting
            weighted_xop = (
                (attacking_xop * attacking_weight) + 
                (defensive_xop * defensive_weight)
            )
            context = (f"mixed_{int(attacking_weight*100)}att_"
                      f"{int(defensive_weight*100)}def")
            
        elif has_attacking_data and attacking_weight > 0:
            # Only attacking data available
            weighted_xop = attacking_xop
            context = "attacking_only"
            
        elif has_defensive_data and defensive_weight > 0:
            # Only defensive data available
            weighted_xop = defensive_xop
            context = "defensive_only"
            
        else:
            # Insufficient data
            weighted_xop = 1.0
            context = "insufficient_data"
        
        return weighted_xop, context
    
    def _format_xg_trend(self, row: pd.Series) -> str:
        """Format xG trend for display with position-aware interpretation."""
        if row.get('current_xg_context') == 'insufficient_data':
            return "N/A"
        
        current_ratio = row.get('current_xOP', 1.0)
        context = row.get('current_xg_context', '')
        
        # Interpretation depends on context
        if 'att' in context or context == 'attacking_only':
            return self._format_attacking_trend(current_ratio)
        elif 'def' in context or context == 'defensive_only':
            return self._format_defensive_trend(current_ratio)
        elif 'mixed' in context:
            return self._format_mixed_trend(current_ratio)
        
        return f"➡️{current_ratio:.2f}"  # Default
    
    def _format_attacking_trend(self, ratio: float) -> str:
        """Format attacking performance trend."""
        if ratio > 1.2:
            return f"🔥{ratio:.2f}"  # Hot attacking form
        elif ratio > 1.1:
            return f"↗️{ratio:.2f}"  # Good attacking form
        elif ratio < 0.8:
            return f"📈{ratio:.2f}"  # Due attacking regression
        elif ratio < 0.9:
            return f"↘️{ratio:.2f}"  # Poor attacking form
        else:
            return f"➡️{ratio:.2f}"  # Normal attacking
    
    def _format_defensive_trend(self, ratio: float) -> str:
        """Format defensive performance trend."""
        if ratio > 1.2:
            return f"🛡️{ratio:.2f}"  # Excellent defence
        elif ratio > 1.1:
            return f"↗️{ratio:.2f}"  # Good defence
        elif ratio < 0.8:
            return f"📈{ratio:.2f}"  # Due defensive improvement
        elif ratio < 0.9:
            return f"↘️{ratio:.2f}"  # Poor defence
        else:
            return f"➡️{ratio:.2f}"  # Normal defence
    
    def _format_mixed_trend(self, ratio: float) -> str:
        """Format mixed performance trend."""
        if ratio > 1.15:
            return f"⭐{ratio:.2f}"  # Excellent overall
        elif ratio > 1.05:
            return f"↗️{ratio:.2f}"  # Good overall
        elif ratio < 0.85:
            return f"📈{ratio:.2f}"  # Due overall improvement
        elif ratio < 0.95:
            return f"↘️{ratio:.2f}"  # Poor overall
        else:
            return f"➡️{ratio:.2f}"  # Normal overall
    
    def _cleanup_temporary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up temporary columns used in xG analysis."""
        temp_columns = [
            "attacking_xOP",
            "defensive_xOP",
            "current_xgi_per_game", 
            "current_gi_per_game",
            "current_xgc_per_game",
            "current_gc_per_game"
        ]
        
        existing_temp_cols = [col for col in temp_columns if col in df.columns]
        return df.drop(columns=existing_temp_cols)
    
    def calculate_budget_from_previous_squad(
            self, gameweek: int, current_players: pd.DataFrame) -> float:
        """
        Calculate available budget based on previous gameweek's squad value.

        Args:
            gameweek (int): Current gameweek number
            current_players (pd.DataFrame): Current player database with prices

        Returns:
            float: Available budget in millions, or default BUDGET if no
                   previous squad
        """
        if gameweek <= 1:
            print(f"Using default budget: £{self.config.BUDGET:.1f}m "
                  f"(no previous squad)")
            return self.config.BUDGET

        prev_gw = gameweek - 1
        prev_squad_file = f"squads/gw{prev_gw}/full_squad.csv"

        if not os.path.exists(prev_squad_file):
            print(f"Using default budget: £{self.config.BUDGET:.1f}m "
                  f"(no previous squad file)")
            return self.config.BUDGET

        try:
            prev_squad = pd.read_csv(prev_squad_file)
            prev_squad_ids = self.match_players_to_current(
                prev_squad, current_players
            )

            if not prev_squad_ids:
                print(f"Using default budget: £{self.config.BUDGET:.1f}m "
                      f"(could not match previous squad)")
                return self.config.BUDGET

            # Calculate total value using current prices
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
    
    def match_players_to_current(self, prev_squad: pd.DataFrame, 
                               current_players: pd.DataFrame) -> list:
        """
        Match previous squad players to current player database.

        Args:
            prev_squad (pd.DataFrame): Previous gameweek's squad.
            current_players (pd.DataFrame): Current player database.

        Returns:
            list: List of player IDs from previous squad that are still
                  available.
        """
        prev_player_ids = []

        for _, prev_player in prev_squad.iterrows():
            player_id = self._find_matching_player(
                prev_player, current_players)
            if player_id:
                prev_player_ids.append(player_id)

        return prev_player_ids
    
    def _find_matching_player(self, prev_player: pd.Series, 
                            current_players: pd.DataFrame) -> int:
        """Find matching player ID in current database."""
        prev_name = prev_player["display_name"].strip()
        prev_pos = prev_player["position"]
        prev_team = prev_player["team"]

        # Try exact match first
        exact_matches = current_players[
            (current_players["position"] == prev_pos)
            & (current_players["team"] == prev_team)
            & (current_players["display_name"].str.strip() == prev_name)
        ]

        if len(exact_matches) == 1:
            return exact_matches.iloc[0]["id"]
        elif len(exact_matches) > 1:
            print(f"Warning: Multiple matches for {prev_name}, taking first "
                  "match")
            return exact_matches.iloc[0]["id"]
        
        # Try fuzzy matching on name
        fuzzy_matches = current_players[
            (current_players["position"] == prev_pos)
            & (current_players["team"] == prev_team)
            & (current_players["display_name"].str.contains(
                prev_name.split()[0], case=False, na=False
            ))
        ]

        if len(fuzzy_matches) > 0:
            return fuzzy_matches.iloc[0]["id"]
        
        print(f"Warning: Could not find current match for "
              f"{prev_name} ({prev_pos}, {prev_team})")
        return None