"""Module for managing historical player performance data with xG performance analysis."""

import pandas as pd
import numpy as np
from ..utils.text_utils import normalize_name


class HistoricalDataManager:
    """Handles fetching and processing of historical player data with xG performance metrics."""
    
    def __init__(self, config):
        self.config = config
    
    def fetch_past_season_points(self, season_folder: str) -> pd.DataFrame:
        """
        Fetch historical performance data including xG metrics for a specific season.

        Args:
            season_folder (str): The season folder name (e.g., "2023-24").

        Returns:
            pd.DataFrame: DataFrame containing performance metrics including xG analysis.
        """
        url = (
            f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
            f"master/data/{season_folder}/players_raw.csv"
        )
        df = pd.read_csv(url)

        # Calculate basic metrics
        df["minutes_played"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
        df["total_points"] = pd.to_numeric(df["total_points"], errors="coerce").fillna(0)
        df["games_played"] = (df["minutes_played"] / 90).round().clip(lower=0)
        df["season_reliability"] = (df["games_played"] / 30).clip(upper=1.0)

        # xG and performance metrics (with error handling for missing columns)
        xg_columns = {
            'expected_goals': 'expected_goals',
            'expected_assists': 'expected_assists', 
            'expected_goal_involvements': 'expected_goal_involvements',
            'expected_goals_conceded': 'expected_goals_conceded',
            'goals_scored': 'goals_scored',
            'assists': 'assists',
            'goals_conceded': 'goals_conceded'
        }
        
        # Safely extract xG metrics
        for col_name, df_col in xg_columns.items():
            if df_col in df.columns:
                df[col_name] = pd.to_numeric(df[df_col], errors="coerce").fillna(0)
            else:
                df[col_name] = 0.0
                print(f"Warning: {df_col} not found in {season_folder}, using 0")

        # Calculate per-game metrics (only for players who played)
        played_mask = df["games_played"] >= 1
        
        # Initialize per-game columns
        per_game_cols = ['points_per_game', 'goals_per_game', 'assists_per_game', 
                        'goal_involvements_per_game', 'xg_per_game', 'xa_per_game', 
                        'xgi_per_game', 'goals_conceded_per_game', 'xgc_per_game']
        
        for col in per_game_cols:
            df[col] = 0.0

        if played_mask.any():
            df.loc[played_mask, "points_per_game"] = (
                df.loc[played_mask, "total_points"] / df.loc[played_mask, "games_played"]
            )
            df.loc[played_mask, "goals_per_game"] = (
                df.loc[played_mask, "goals_scored"] / df.loc[played_mask, "games_played"]
            )
            df.loc[played_mask, "assists_per_game"] = (
                df.loc[played_mask, "assists"] / df.loc[played_mask, "games_played"]
            )
            df.loc[played_mask, "goal_involvements_per_game"] = (
                (df.loc[played_mask, "goals_scored"] + df.loc[played_mask, "assists"]) / 
                df.loc[played_mask, "games_played"]
            )
            df.loc[played_mask, "xg_per_game"] = (
                df.loc[played_mask, "expected_goals"] / df.loc[played_mask, "games_played"]
            )
            df.loc[played_mask, "xa_per_game"] = (
                df.loc[played_mask, "expected_assists"] / df.loc[played_mask, "games_played"]
            )
            df.loc[played_mask, "xgi_per_game"] = (
                (df.loc[played_mask, "expected_goals"] + df.loc[played_mask, "expected_assists"]) / 
                df.loc[played_mask, "games_played"]
            )
            df.loc[played_mask, "goals_conceded_per_game"] = (
                df.loc[played_mask, "goals_conceded"] / df.loc[played_mask, "games_played"]
            )
            df.loc[played_mask, "xgc_per_game"] = (
                df.loc[played_mask, "expected_goals_conceded"] / df.loc[played_mask, "games_played"]
            )

        # Calculate xG performance ratios (avoiding division by zero)
        df["historical_xOP"] = 1.0  # Historical Expected Overperformance ratio
        
        # Goal involvement performance (for forwards and midfielders)
        xgi_mask = (df["xgi_per_game"] > 0.05) & played_mask  # Minimum threshold
        if xgi_mask.any():
            df.loc[xgi_mask, "historical_xOP"] = (
                df.loc[xgi_mask, "goal_involvements_per_game"] / df.loc[xgi_mask, "xgi_per_game"]
            ).clip(0.2, 3.0)  # Reasonable bounds
        
        # Goals conceded performance (for goalkeepers and defenders)
        xgc_mask = (df["xgc_per_game"] > 0.05) & played_mask  # Minimum threshold
        if xgc_mask.any():
            # For GC, higher ratio = better performance (conceding less than expected)
            df.loc[xgc_mask, "historical_xOP"] = (
                df.loc[xgc_mask, "xgc_per_game"] / df.loc[xgc_mask, "goals_conceded_per_game"]
            ).clip(0.2, 3.0)  # Reasonable bounds
        
        # Apply reliability penalty
        reliability_threshold = 0.8
        unreliable_mask = df["season_reliability"] < reliability_threshold
        penalty_factor = df["season_reliability"].clip(lower=0.3)
        df.loc[unreliable_mask, "points_per_game"] *= penalty_factor[unreliable_mask]

        # Select relevant columns for output
        output_cols = [
            "web_name", "element_type", "points_per_game", "games_played", "season_reliability",
            "historical_xOP", "goal_involvements_per_game",
            "goals_conceded_per_game", "xgi_per_game", "xgc_per_game"
        ]
        
        # Only include columns that exist
        available_cols = [col for col in output_cols if col in df.columns]
        result_df = df[available_cols].copy()
        
        result_df["name_key"] = result_df["web_name"].map(normalize_name)
        
        # Rename columns with season suffix
        rename_map = {
            "points_per_game": f"ppg_{season_folder}",
            "games_played": f"games_{season_folder}",
            "season_reliability": f"reliability_{season_folder}",
            "historical_xOP": f"historical_xOP_{season_folder}",
            "element_type": f"position_{season_folder}"
        }
        
        result_df = result_df.rename(columns=rename_map)
        return result_df
    
    def calculate_xg_performance_modifier(self, player_data: dict) -> float:
        """
        Calculate xG performance modifier based on position, historical data, and current season performance.
        
        Args:
            player_data (dict): Player's historical and current xG performance data
            
        Returns:
            float: Performance modifier (1.0 = neutral, >1.0 = over-performer, <1.0 = under-performer)
        """
        position = player_data.get('current_position', 'MID')
        
        # Get historical xG performance ratios across seasons
        historical_xop_values = []
        weights = []
        
        for i, season in enumerate(self.config.PAST_SEASONS):
            season_weight = self.config.HISTORIC_SEASON_WEIGHTS[i]
            xop_key = f"historical_xOP_{season}"
            
            if xop_key in player_data and player_data[xop_key] > 0:
                historical_xop_values.append(player_data[xop_key])
                weights.append(season_weight)
        
        # Get current season xG performance
        current_xop = player_data.get('current_xOP', 1.0)
        current_xg_context = player_data.get('current_xg_context', 'insufficient_data')
        
        # Calculate historical baseline modifier
        historical_modifier = self._calculate_historical_xg_modifier(
            position, historical_xop_values, weights
        )
        
        # Calculate current season impact based on deviation from historical trend
        current_season_impact = self._calculate_current_season_xg_impact(
            position, historical_modifier, current_xop, current_xg_context
        )
        
        # Combine historical and current season analysis
        final_modifier = historical_modifier + current_season_impact
        
        return max(0.7, min(1.4, final_modifier))  # Reasonable bounds

    def _calculate_historical_xg_modifier(self, position: str, historical_xop_values: list, 
                                        weights: list) -> float:
        """Calculate historical xG performance baseline."""
        if not historical_xop_values:
            return 1.0  # Neutral if no historical data
            
        weighted_avg = np.average(historical_xop_values, weights=weights[:len(historical_xop_values)])
        
        if position == 'FWD':
            # Forwards: Full impact of goal involvement performance
            return 0.9 + (weighted_avg * 0.1)
        elif position == 'MID':
            # Midfielders: Moderate impact
            return 0.95 + (weighted_avg * 0.05)
        elif position in ['DEF', 'GK']:
            # Defenders/GKs: Full impact of defensive performance
            return 0.9 + (weighted_avg * 0.1)
        
        return 1.0

    def _calculate_current_season_xg_impact(self, position: str, historical_modifier: float,
                                          current_xop: float, current_xg_context: str) -> float:
        """
        Calculate current season xG impact based on deviation from historical trend.
        """
        if current_xg_context == 'insufficient_data':
            return 0.0  # No current season adjustment
        
        # Calculate expected performance based on historical trend
        historical_baseline = (historical_modifier - 0.9) / 0.1  # Convert back to ratio scale
        if historical_baseline == 0:
            historical_baseline = 1.0
        
        current_season_impact = 0.0
        deviation = current_xop - historical_baseline
        
        if position == 'FWD':
            # Forwards: Strong regression signals
            if deviation < -0.2:  # Significantly underperforming
                current_season_impact = 0.03  # +3% bonus for regression opportunity
            elif deviation < -0.1:  # Slightly underperforming  
                current_season_impact = 0.015  # +1.5% bonus
            elif deviation > 0.3:  # Very hot streak vs historical norm
                current_season_impact = -0.02  # -2% penalty for unsustainable form
                
        elif position == 'MID':
            # Midfielders: Moderate regression signals
            if deviation < -0.2:
                current_season_impact = 0.02  # +2% bonus
            elif deviation > 0.3:
                current_season_impact = -0.015  # -1.5% penalty
                    
        elif position in ['DEF', 'GK']:
            # Defensive players: Strong regression signals for defensive performance
            if deviation < -0.2:  # Conceding more than expected vs historical
                current_season_impact = 0.025  # +2.5% bonus for regression opportunity
            elif deviation > 0.3:  # Unsustainably good defensive form
                current_season_impact = -0.02  # -2% penalty
        
        return current_season_impact
    
    def calculate_data_availability_factor(self, player_data: dict) -> float:
        """
        Calculate factor to avoid over-penalizing players with limited historical data.
        
        Args:
            player_data (dict): Player's historical data
            
        Returns:
            float: Factor to reduce impact of xG modifier for players with limited data
        """
        seasons_with_data = 0
        total_games = 0
        
        for season in self.config.PAST_SEASONS:
            games_key = f"games_{season}"
            if games_key in player_data and player_data[games_key] > 8:  # Minimum 8 games
                seasons_with_data += 1
                total_games += player_data[games_key]
        
        # Factor based on data availability
        # 0 seasons = 0.1 (minimal impact)
        # 1 season = 0.5 (moderate impact) 
        # 2+ seasons = 1.0 (full impact)
        if seasons_with_data == 0:
            return 0.1
        elif seasons_with_data == 1:
            return 0.5
        else:
            # Additional bonus for more total games
            games_factor = min(total_games / 60, 1.0)  # Cap at 60 games
            return 0.7 + (0.3 * games_factor)
    
    def merge_past_seasons(self, current: pd.DataFrame) -> pd.DataFrame:
        """
        Merge historical data with xG performance analysis.

        Args:
            current (pd.DataFrame): Current season player data.

        Returns:
            pd.DataFrame: Enhanced player data with xG performance modifiers.
        """
        # Fetch historical data
        hist_frames = [self.fetch_past_season_points(s) for s in self.config.PAST_SEASONS]
        
        # Start with the first season's data
        hist = hist_frames[0].copy()
        
        # Merge additional seasons, keeping only season-specific columns to avoid conflicts
        for i, extra in enumerate(hist_frames[1:], 1):
            # Get the season name for this frame
            season = self.config.PAST_SEASONS[i]
            
            # Only keep season-specific columns and name_key from extra dataframe
            season_specific_cols = ["name_key"]
            for col in extra.columns:
                if col.endswith(f"_{season}"):
                    season_specific_cols.append(col)
            
            # Merge only the season-specific columns
            extra_filtered = extra[season_specific_cols].copy()
            hist = hist.merge(extra_filtered, on="name_key", how="outer")

        # Get relevant columns for calculations
        ppg_cols = [c for c in hist.columns if c.startswith("ppg_")]
        games_cols = [c for c in hist.columns if c.startswith("games_")]
        reliability_cols = [c for c in hist.columns if c.startswith("reliability_")]
        historical_xop_cols = [c for c in hist.columns if c.startswith("historical_xOP_")]

        # Calculate traditional weighted averages
        hist["avg_ppg_past2"] = 0.0
        hist["total_games_past2"] = 0
        hist["avg_reliability"] = 0.0
        hist["historical_xOP"] = 1.0  # Weighted historical Expected Overperformance

        for ppg_col, games_col, reliability_col, weight in zip(
            ppg_cols, games_cols, reliability_cols, self.config.HISTORIC_SEASON_WEIGHTS
        ):
            hist[ppg_col] = hist[ppg_col].fillna(0)
            hist[games_col] = hist[games_col].fillna(0)
            hist[reliability_col] = hist[reliability_col].fillna(0)

            sufficient_games_mask = hist[games_col] >= 8
            hist.loc[sufficient_games_mask, "avg_ppg_past2"] += (
                hist.loc[sufficient_games_mask, ppg_col] * weight
            )
            hist.loc[sufficient_games_mask, "avg_reliability"] += (
                hist.loc[sufficient_games_mask, reliability_col] * weight
            )
            hist["total_games_past2"] += hist[games_col]

        # Calculate weighted historical xOP
        for xop_col, weight in zip(historical_xop_cols, self.config.HISTORIC_SEASON_WEIGHTS):
            if xop_col in hist.columns:
                hist[xop_col] = hist[xop_col].fillna(1.0)  # Default to neutral
                # Only include seasons where player had sufficient games
                season_suffix = xop_col.split('_')[-1]
                games_col = f"games_{season_suffix}"
                if games_col in hist.columns:
                    sufficient_games_mask = hist[games_col] >= 8
                    # Initialize historical_xOP to 1.0 for first calculation
                    if xop_col == historical_xop_cols[0]:
                        hist["historical_xOP"] = 1.0
                    # Add weighted contribution (subtract 1.0 to get deviation, then add back)
                    hist.loc[sufficient_games_mask, "historical_xOP"] += (
                        (hist.loc[sufficient_games_mask, xop_col] - 1.0) * weight
                    )

        # Round historical_xOP to 2 decimal places
        hist["historical_xOP"] = hist["historical_xOP"].round(2)

        # Calculate xG performance modifiers for each player
        hist["xConsistency"] = 1.0  # Final Expected modifier
        
        for idx, row in hist.iterrows():
            player_data = row.to_dict()
            player_data['current_position'] = self._get_player_position(row)
            
            # Calculate xG modifier
            xg_modifier = self.calculate_xg_performance_modifier(player_data)
            
            # Apply data availability factor
            availability_factor = self.calculate_data_availability_factor(player_data)
            
            # Blend with neutral (1.0) based on data availability
            final_modifier = 1.0 + ((xg_modifier - 1.0) * availability_factor)
            
            hist.loc[idx, "xConsistency"] = round(final_modifier, 2)

        # Calculate current season reliability
        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        current_reliability = current["starts"] / gameweeks_completed
        current_reliability = current_reliability.clip(upper=1.0)
        
        print(f"Calculated reliability based on starts over {gameweeks_completed} completed gameweek(s)")
        print(f"Applied xG performance analysis across {len(self.config.PAST_SEASONS)} seasons")

        current = current.assign(current_reliability=current_reliability)
        
        # Merge historical data with current
        merge_cols = ["name_key", "avg_ppg_past2", "total_games_past2", "avg_reliability", "historical_xOP", "xConsistency"]
        return current.merge(hist[merge_cols], on="name_key", how="left")
    
    def _get_player_position(self, player_row: pd.Series) -> str:
        """
        Determine player's position from historical data.
        
        Args:
            player_row (pd.Series): Player's historical data row
            
        Returns:
            str: Player position (GK, DEF, MID, FWD)
        """
        # Look for position in any season
        for season in self.config.PAST_SEASONS:
            pos_key = f"position_{season}"
            if pos_key in player_row and pd.notna(player_row[pos_key]):
                pos_num = player_row[pos_key]
                pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
                return pos_map.get(pos_num, "MID")
        
        return "MID"  # Default fallback