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
        Fetch historical performance data including weighted xG metrics for a specific season.

        Args:
            season_folder (str): The season folder name (e.g., "2023-24").

        Returns:
            pd.DataFrame: DataFrame containing performance metrics including weighted xG analysis.
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

        # Calculate position-weighted xG performance ratios
        df["historical_xOP"] = 1.0  # Historical Expected Overperformance ratio
        df["attacking_xOP_hist"] = 1.0  # Separate components for debugging
        df["defensive_xOP_hist"] = 1.0
        
        # Position-based weighting system (same as current season)
        position_weights = {
            1: {"attacking": 0.0, "defensive": 1.0},   # GK
            2: {"attacking": 0.25, "defensive": 0.75}, # DEF  
            3: {"attacking": 0.75, "defensive": 0.25}, # MID
            4: {"attacking": 1.0, "defensive": 0.0}    # FWD
        }
        
        # CONSERVATIVE THRESHOLDS: Use same approach as current season analysis
        # Players need meaningful per-game involvement to be included
        min_xgi_per_game_hist = 0.05  # Minimum 0.05 xGI per game for attacking analysis
        min_xgc_per_game_hist = 0.05  # Minimum 0.05 xGC per game for defensive analysis
        min_games_hist = 8           # Minimum 8 games for historical analysis
        
        # Calculate attacking component ONLY for positions with attacking weight
        attacking_hist_mask = (
            (df["xgi_per_game"] > min_xgi_per_game_hist) & 
            (df["games_played"] >= min_games_hist) & 
            df["element_type"].isin([2, 3, 4])  # DEF, MID, FWD can have attacking
        )
        if attacking_hist_mask.any():
            df.loc[attacking_hist_mask, "attacking_xOP_hist"] = (
                df.loc[attacking_hist_mask, "goal_involvements_per_game"] / 
                df.loc[attacking_hist_mask, "xgi_per_game"]
            ).clip(0.2, 3.0)
        
        # Calculate defensive component ONLY for positions with defensive weight
        defensive_hist_mask = (
            (df["xgc_per_game"] > min_xgc_per_game_hist) & 
            (df["games_played"] >= min_games_hist) & 
            df["element_type"].isin([1, 2, 3])  # GK, DEF, MID can have defensive
        )
        if defensive_hist_mask.any():
            # For GC, higher ratio = better performance (conceding less than expected)
            df.loc[defensive_hist_mask, "defensive_xOP_hist"] = (
                df.loc[defensive_hist_mask, "xgc_per_game"] / 
                df.loc[defensive_hist_mask, "goals_conceded_per_game"].clip(lower=0.01)
            ).clip(0.2, 3.0)
        
        # Calculate weighted historical xOP for each player - EXACTLY like current season
        for idx, row in df.iterrows():
            position_id = row["element_type"]
            if position_id not in position_weights:
                continue
                
            weights = position_weights[position_id]
            attacking_weight = weights["attacking"]
            defensive_weight = weights["defensive"]
            
            attacking_xop = row["attacking_xOP_hist"]
            defensive_xop = row["defensive_xOP_hist"]
            
            # Check if we have sufficient data for each component using SAME logic as current
            has_attacking_data = (
                attacking_weight > 0 and 
                row["xgi_per_game"] > min_xgi_per_game_hist and 
                row["games_played"] >= min_games_hist
            )
            
            has_defensive_data = (
                defensive_weight > 0 and 
                row["xgc_per_game"] > min_xgc_per_game_hist and 
                row["games_played"] >= min_games_hist
            )
            
            # Calculate weighted xOP based on available data - IDENTICAL logic to current season
            if has_attacking_data and has_defensive_data:
                # Both components available - use full weighting
                weighted_xop = (attacking_xop * attacking_weight) + (defensive_xop * defensive_weight)
                
            elif has_attacking_data and attacking_weight > 0:
                # Only attacking data available - use attacking component only
                weighted_xop = attacking_xop
                
            elif has_defensive_data and defensive_weight > 0:
                # Only defensive data available - use defensive component only
                weighted_xop = defensive_xop
                
            else:
                # Insufficient data
                weighted_xop = 1.0
            
            df.loc[idx, "historical_xOP"] = round(weighted_xop, 2)

        # Apply reliability penalty
        reliability_threshold = 0.8
        unreliable_mask = df["season_reliability"] < reliability_threshold
        penalty_factor = df["season_reliability"].clip(lower=0.3)
        df.loc[unreliable_mask, "points_per_game"] *= penalty_factor[unreliable_mask]

        # Clean up temporary columns
        df = df.drop(columns=["attacking_xOP_hist", "defensive_xOP_hist"])

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
        Enhanced xG performance modifier that properly handles regression and volatility.
        
        Args:
            player_data (dict): Player's historical and current xG performance data
            
        Returns:
            float: Performance modifier (1.0 = neutral, >1.0 = expected improvement, <1.0 = expected decline)
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
        
        # Calculate weighted historical baseline (or None if no data)
        historical_baseline = None
        if historical_xop_values:
            historical_baseline = np.average(historical_xop_values, weights=weights[:len(historical_xop_values)])
        
        # Get current season performance
        current_xop = player_data.get('current_xOP', 1.0)
        current_xg_context = player_data.get('current_xg_context', 'insufficient_data')
        
        # Calculate season progression factor (how reliable current xOP is)
        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        season_progression = self._calculate_season_progression_factor(gameweeks_completed)
        
        # Calculate the xG modifier based on available data
        if historical_baseline is not None:
            # Player has historical data - compare current to historical
            modifier, volatility = self._calculate_regression_modifier(
                historical_baseline, current_xop, season_progression, position
            )
        else:
            # New player - evaluate current performance with high volatility
            modifier, volatility = self._calculate_new_player_modifier(
                current_xop, season_progression, current_xg_context
            )
        
        # Apply conservative volatility penalty to reduce FPL score for uncertain players
        volatility_penalty = self._calculate_volatility_penalty(volatility)
        final_modifier = 1.0 + (modifier - 1.0) * (1.0 - volatility_penalty)

        
        return max(0.6, min(1.5, final_modifier))  # Reasonable bounds

    def _calculate_season_progression_factor(self, gameweeks_completed: int) -> float:
        """
        Calculate how reliable current season stats are based on games played.
        
        Returns:
            float: 0.0 = very early season (high volatility), 1.0 = late season (reliable)
        """
        if gameweeks_completed <= 3:
            return 0.1  # Very early season - current stats highly volatile
        elif gameweeks_completed <= 6:
            return 0.3  # Early season - still quite volatile
        elif gameweeks_completed <= 10:
            return 0.6  # Mid-early season - becoming more reliable
        elif gameweeks_completed <= 15:
            return 0.8  # Mid season - quite reliable
        else:
            return 1.0  # Late season - very reliable

    def _calculate_regression_modifier(self, historical_baseline: float, current_xop: float, 
                                     season_progression: float, position: str) -> tuple:
        """
        Calculate regression-based modifier for players with historical data.
        
        Returns:
            tuple: (modifier, volatility_score)
        """
        deviation = current_xop - historical_baseline
        abs_deviation = abs(deviation)
        
        # Position-based sensitivity to xG regression
        position_sensitivity = {
            'FWD': 1.0,   # Forwards most sensitive to xG regression
            'MID': 0.7,   # Midfielders moderately sensitive
            'DEF': 0.8,   # Defenders quite sensitive (clean sheets)
            'GK': 0.9     # Goalkeepers very sensitive (save performance)
        }
        
        sensitivity = position_sensitivity.get(position, 0.7)
        
        # Early season: treat extreme deviations as volatile, expect regression to mean
        # Late season: if deviation persists, it might be more permanent
        
        if deviation > 0:  # Currently overperforming historical baseline
            if season_progression < 0.6:
                # Early season overperformance - expect significant regression
                regression_strength = abs_deviation * sensitivity * 0.15
                modifier = 1.0 - regression_strength  # Penalty for unsustainable performance
                volatility = abs_deviation * (1.0 - season_progression) * 0.3  # High volatility early
            else:
                # Late season overperformance - might be genuine improvement, but still some regression
                regression_strength = abs_deviation * sensitivity * 0.08
                modifier = 1.0 - regression_strength
                volatility = abs_deviation * 0.1  # Lower volatility late season
                
        else:  # Currently underperforming historical baseline  
            if season_progression < 0.6:
                # Early season underperformance - expect positive regression
                regression_strength = abs_deviation * sensitivity * 0.12
                modifier = 1.0 + regression_strength  # Bonus for expected improvement
                volatility = abs_deviation * (1.0 - season_progression) * 0.25  # High volatility early
            else:
                # Late season underperformance - concerning, might be decline
                regression_strength = abs_deviation * sensitivity * 0.05  # Small bonus only
                modifier = 1.0 + regression_strength
                volatility = abs_deviation * 0.15  # Moderate volatility - uncertain if temporary
        
        return modifier, min(volatility, 0.4)  # Cap volatility at 40%

    def _calculate_new_player_modifier(self, current_xop: float, season_progression: float, 
                                     current_xg_context: str) -> tuple:
        """
        Calculate modifier for new players without historical data.
        
        Returns:
            tuple: (modifier, volatility_score)
        """
        if current_xg_context == 'insufficient_data':
            return 1.0, 0.2  # Neutral with moderate volatility
        
        # For new players, current xOP is all we have
        # But treat it with high volatility early season, moderate volatility late season
        
        deviation_from_neutral = current_xop - 1.0
        abs_deviation = abs(deviation_from_neutral)
        
        if season_progression < 0.6:
            # Early season: high volatility, limited trust in current performance
            if current_xop > 1.3:  # Very high performance
                modifier = 1.0 + (deviation_from_neutral * 0.1)  # Small bonus
                volatility = 0.35  # High volatility
            elif current_xop < 0.7:  # Very poor performance  
                modifier = 1.0 + (deviation_from_neutral * 0.1)  # Small penalty
                volatility = 0.3   # High volatility
            else:  # Reasonable performance
                modifier = 1.0 + (deviation_from_neutral * 0.15)
                volatility = 0.2   # Moderate volatility
        else:
            # Late season: current performance more reliable for new players
            if current_xop > 1.2:  # High performance
                modifier = 1.0 + (deviation_from_neutral * 0.2)  # Moderate bonus
                volatility = 0.15  # Lower volatility
            elif current_xop < 0.8:  # Poor performance
                modifier = 1.0 + (deviation_from_neutral * 0.2)  # Moderate penalty  
                volatility = 0.2   # Moderate volatility
            else:  # Normal performance
                modifier = 1.0 + (deviation_from_neutral * 0.25)
                volatility = 0.1   # Low volatility
        
        return modifier, volatility

    def _calculate_volatility_penalty(self, volatility: float) -> float:
        """
        Conservative volatility penalty - punish uncertainty heavily.
        
        Args:
            volatility (float): Volatility score (0.0 to 0.4)
            
        Returns:
            float: Penalty factor (0.0 to 0.35) to reduce modifier
        """
        # More aggressive scaling: 0.4 volatility = 35% penalty
        base_penalty = volatility * 0.875  # Linear scaling
        
        # Add exponential component for very high volatility
        if volatility > 0.25:
            exponential_bonus = (volatility - 0.25) * 2.0  # Extra penalty for extreme volatility
            base_penalty += exponential_bonus
        
        return min(base_penalty, 0.35)  # Cap at 35% penalty
    
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
        Enhanced merge with proper NA handling for missing historical data and conservative volatility penalties.

        Args:
            current (pd.DataFrame): Current season player data.

        Returns:
            pd.DataFrame: Enhanced player data with weighted xG performance modifiers.
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

        # Calculate weighted historical xOP (now using position-weighted values)
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

        # Calculate xG performance modifiers for each player with enhanced volatility handling
        hist["xConsistency"] = 1.0  # Final Expected modifier
        hist["xOP_historical_baseline"] = np.nan  # Track if player has historical baseline
        
        for idx, row in hist.iterrows():
            player_data = row.to_dict()
            player_data['current_position'] = self._get_player_position(row)
            
            # Check if player has historical xOP data
            has_historical_data = False
            for season in self.config.PAST_SEASONS:
                xop_key = f"historical_xOP_{season}"
                games_key = f"games_{season}"
                if (xop_key in player_data and 
                    games_key in player_data and 
                    player_data[games_key] >= 8 and
                    player_data[xop_key] > 0):
                    has_historical_data = True
                    break
            
            if has_historical_data:
                # Calculate normal xG modifier
                xg_modifier = self.calculate_xg_performance_modifier(player_data)
                hist.loc[idx, "xOP_historical_baseline"] = row.get("historical_xOP", 1.0)
            else:
                # New player - calculate modifier but mark baseline as NA
                xg_modifier = self.calculate_xg_performance_modifier(player_data)
                hist.loc[idx, "xOP_historical_baseline"] = np.nan
            
            # Apply data availability factor (but less aggressive for new players)
            if has_historical_data:
                availability_factor = self.calculate_data_availability_factor(player_data)
            else:
                availability_factor = 0.7  # Give new players moderate impact
            
            # Blend with neutral (1.0) based on data availability
            final_modifier = 1.0 + ((xg_modifier - 1.0) * availability_factor)
            hist.loc[idx, "xConsistency"] = round(final_modifier, 2)

        # Calculate current season reliability
        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        current_reliability = current["starts"] / gameweeks_completed
        current_reliability = current_reliability.clip(upper=1.0)
        
        print(f"Calculated reliability based on starts over {gameweeks_completed} completed gameweek(s)")
        print(f"Applied consistent per-game weighted xG analysis with conservative volatility penalties across {len(self.config.PAST_SEASONS)} seasons")

        current = current.assign(current_reliability=current_reliability)
        
        # Enhanced merge columns including historical baseline tracking
        merge_cols = ["name_key", "avg_ppg_past2", "total_games_past2", "avg_reliability", 
                      "historical_xOP", "xConsistency", "xOP_historical_baseline"]
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