"""Module for managing historical player performance data with xG analysis."""

import pandas as pd
import numpy as np
from ..utils.text_utils import normalize_name


class HistoricalDataManager:
    """Handles fetching and processing of historical player data with xG 
    performance metrics."""
    
    def __init__(self, config):
        self.config = config
        
        # Position-based weighting system (same as current season)
        self.position_weights = {
            1: {"attacking": 0.0, "defensive": 1.0},   # GK
            2: {"attacking": 0.25, "defensive": 0.75}, # DEF  
            3: {"attacking": 0.75, "defensive": 0.25}, # MID
            4: {"attacking": 1.0, "defensive": 0.0}    # FWD
        }
        
        # Thresholds for historical analysis
        self.historical_thresholds = {
            'min_xgi_per_game': 0.05,  # Minimum 0.05 xGI per game
            'min_xgc_per_game': 0.05,  # Minimum 0.05 xGC per game
            'min_games': 8             # Minimum 8 games for analysis
        }
    
    def fetch_past_season_points(self, season_folder: str) -> pd.DataFrame:
        """
        Fetch historical performance data including weighted xG metrics 
        for a specific season.

        Args:
            season_folder (str): The season folder name (e.g., "2023-24").

        Returns:
            pd.DataFrame: DataFrame containing performance metrics including 
                         weighted xG analysis.
        """
        url = (
            f"https://raw.githubusercontent.com/vaastav/"
            f"Fantasy-Premier-League/master/data/"
            f"{season_folder}/players_raw.csv"
        )
        df = pd.read_csv(url)

        # Calculate basic metrics
        df = self._calculate_basic_metrics(df)

        # Extract and validate xG metrics
        df = self._extract_historical_xg_metrics(df, season_folder)

        # Calculate per-game metrics
        df = self._calculate_historical_per_game_metrics(df)

        # Calculate position-weighted xG performance ratios
        df = self._calculate_historical_xg_performance(df)

        # Apply reliability penalty
        df = self._apply_reliability_penalty(df)

        # Select relevant columns for output
        df = self._prepare_output_dataframe(df, season_folder)

        return df
    
    def _calculate_basic_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic performance metrics."""
        df["minutes_played"] = pd.to_numeric(
            df["minutes"], errors="coerce"
        ).fillna(0)
        df["total_points"] = pd.to_numeric(
            df["total_points"], errors="coerce"
        ).fillna(0)
        df["games_played"] = (df["minutes_played"] / 90).round().clip(lower=0)
        df["season_reliability"] = (df["games_played"] / 30).clip(upper=1.0)
        
        return df
    
    def _extract_historical_xg_metrics(self, df: pd.DataFrame, 
                                     season_folder: str) -> pd.DataFrame:
        """Extract xG metrics with error handling for missing columns."""
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
                df[col_name] = pd.to_numeric(
                    df[df_col], errors="coerce").fillna(0)
            else:
                df[col_name] = 0.0
                if self.config.GRANULAR_OUTPUT:
                    print(f"Warning: {df_col} not found in {season_folder}, "
                          "using 0")
        
        return df
    
    def _calculate_historical_per_game_metrics(
            self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate per-game metrics for players who played."""
        played_mask = df["games_played"] >= 1
        
        # Initialise per-game columns
        per_game_cols = [
            'points_per_game', 'goals_per_game', 'assists_per_game', 
            'goal_involvements_per_game', 'xg_per_game', 'xa_per_game', 
            'xgi_per_game', 'goals_conceded_per_game', 'xgc_per_game'
        ]
        
        for col in per_game_cols:
            df[col] = 0.0

        if played_mask.any():
            # Calculate all per-game metrics
            df.loc[played_mask, "points_per_game"] = (
                df.loc[played_mask, "total_points"] / 
                df.loc[played_mask, "games_played"]
            )
            df.loc[played_mask, "goals_per_game"] = (
                df.loc[played_mask, "goals_scored"] / 
                df.loc[played_mask, "games_played"]
            )
            df.loc[played_mask, "assists_per_game"] = (
                df.loc[played_mask, "assists"] / 
                df.loc[played_mask, "games_played"]
            )
            df.loc[played_mask, "goal_involvements_per_game"] = (
                (df.loc[played_mask, "goals_scored"] + 
                 df.loc[played_mask, "assists"]) / 
                df.loc[played_mask, "games_played"]
            )
            df.loc[played_mask, "xg_per_game"] = (
                df.loc[played_mask, "expected_goals"] / 
                df.loc[played_mask, "games_played"]
            )
            df.loc[played_mask, "xa_per_game"] = (
                df.loc[played_mask, "expected_assists"] / 
                df.loc[played_mask, "games_played"]
            )
            df.loc[played_mask, "xgi_per_game"] = (
                (df.loc[played_mask, "expected_goals"] + 
                 df.loc[played_mask, "expected_assists"]) / 
                df.loc[played_mask, "games_played"]
            )
            df.loc[played_mask, "goals_conceded_per_game"] = (
                df.loc[played_mask, "goals_conceded"] / 
                df.loc[played_mask, "games_played"]
            )
            df.loc[played_mask, "xgc_per_game"] = (
                df.loc[played_mask, "expected_goals_conceded"] / 
                df.loc[played_mask, "games_played"]
            )

        return df
    
    def _calculate_historical_xg_performance(
            self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate position-weighted xG performance ratios."""
        # Initialise xG performance columns
        df["historical_xOP"] = 1.0  # Historical Expected Overperformance ratio
        df["attacking_xOP_hist"] = 1.0  # Separate components for debugging
        df["defensive_xOP_hist"] = 1.0
        
        # Calculate attacking component for relevant positions
        attacking_hist_mask = (
            (df["xgi_per_game"] >
             self.historical_thresholds['min_xgi_per_game']) & 
            (df["games_played"] >= self.historical_thresholds['min_games']) & 
            df["element_type"].isin([2, 3, 4])  # DEF, MID, FWD 
        )
        
        if attacking_hist_mask.any():
            df.loc[attacking_hist_mask, "attacking_xOP_hist"] = (
                df.loc[attacking_hist_mask, "goal_involvements_per_game"] / 
                df.loc[attacking_hist_mask, "xgi_per_game"]
            ).clip(0.2, 3.0)
        
        # Calculate defensive component for relevant positions
        defensive_hist_mask = (
            (df["xgc_per_game"] >
             self.historical_thresholds['min_xgc_per_game']) & 
            (df["games_played"] >= self.historical_thresholds['min_games']) & 
            df["element_type"].isin([1, 2, 3])  # GK, DEF, MID
        )
        
        if defensive_hist_mask.any():
            # For GC, higher ratio = 
            # better performance (conceding less than expected)
            df.loc[defensive_hist_mask, "defensive_xOP_hist"] = (
                df.loc[defensive_hist_mask,"xgc_per_game"] / 
                df.loc[defensive_hist_mask, "goals_conceded_per_game"].clip(
                lower=0.01
                )
            ).clip(0.2, 3.0)
        
        # Calculate weighted historical xOP for each player
        for idx, row in df.iterrows():
            position_id = row["element_type"]
            if position_id not in self.position_weights:
                continue
                
            weighted_xop = self._calculate_weighted_historical_xop(
                row, position_id)
            df.loc[idx, "historical_xOP"] = round(weighted_xop, 2)

        # Clean up temporary columns
        df = df.drop(columns=["attacking_xOP_hist", "defensive_xOP_hist"])
        
        return df
    
    def _calculate_weighted_historical_xop(self, row: pd.Series, 
                                         position_id: int) -> float:
        """Calculate weighted historical xOP for a single player."""
        weights = self.position_weights[position_id]
        attacking_weight = weights["attacking"]
        defensive_weight = weights["defensive"]
        
        attacking_xop = row["attacking_xOP_hist"]
        defensive_xop = row["defensive_xOP_hist"]
        
        # Check if we have sufficient data for each component
        has_attacking_data = (
            attacking_weight > 0 and 
            row["xgi_per_game"] >
            self.historical_thresholds['min_xgi_per_game'] 
            and row["games_played"] >= self.historical_thresholds['min_games']
        )
        
        has_defensive_data = (
            defensive_weight > 0 and 
            row["xgc_per_game"] >
            self.historical_thresholds['min_xgc_per_game'] and 
            row["games_played"] >= self.historical_thresholds['min_games']
        )
        
        # Calculate weighted xOP based on available data
        if has_attacking_data and has_defensive_data:
            # Both components available - use full weighting
            return ((attacking_xop * attacking_weight) +
                    (defensive_xop * defensive_weight))
            
        elif has_attacking_data and attacking_weight > 0:
            # Only attacking data available
            return attacking_xop
            
        elif has_defensive_data and defensive_weight > 0:
            # Only defensive data available
            return defensive_xop
            
        else:
            # Insufficient data
            return 1.0
    
    def _apply_reliability_penalty(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply reliability penalty to unreliable players."""
        reliability_threshold = 0.8
        unreliable_mask = df["season_reliability"] < reliability_threshold
        penalty_factor = df["season_reliability"].clip(lower=0.3)
        df.loc[
            unreliable_mask,
            "points_per_game"
            ] *= penalty_factor[unreliable_mask]
        
        return df
    
    def _prepare_output_dataframe(self, df: pd.DataFrame, 
                                season_folder: str) -> pd.DataFrame:
        """Prepare output dataframe with relevant columns."""
        output_cols = [
            "web_name",
            "element_type",
            "points_per_game",
            "games_played", 
            "season_reliability",
            "historical_xOP",
            "goal_involvements_per_game",
            "goals_conceded_per_game",
            "xgi_per_game",
            "xgc_per_game"
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
        Enhanced xG performance modifier that properly handles regression 
        and volatility.
        
        Args:
            player_data (dict): Player's historical and current xG performance 
            data
            
        Returns:
            float: Performance modifier (1.0 = neutral, >1.0 = expected 
                  improvement, <1.0 = expected decline)
        """
        position = player_data.get('current_position', 'MID')
        
        # Get historical xG performance ratios across seasons
        historical_baseline = self._calculate_historical_baseline(player_data)
        
        # Get current season performance
        current_xop = player_data.get('current_xOP', 1.0)
        current_xg_context = player_data.get(
            'current_xg_context',
            'insufficient_data'
        )
        
        # Calculate season progression factor
        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        season_progression = self._calculate_season_progression_factor(
            gameweeks_completed
        )
        
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
        
        # Apply conservative volatility penalty
        volatility_penalty = self._calculate_volatility_penalty(volatility)
        final_modifier = 1.0 + (modifier - 1.0) * (1.0 - volatility_penalty)

        return max(0.6, min(1.5, final_modifier))  # Reasonable bounds
    
    def _calculate_historical_baseline(self, player_data: dict) -> float:
        """Calculate weighted historical baseline from available seasons."""
        historical_xop_values = []
        weights = []
        
        for i, season in enumerate(self.config.PAST_SEASONS):
            season_weight = self.config.HISTORIC_SEASON_WEIGHTS[i]
            xop_key = f"historical_xOP_{season}"
            
            if xop_key in player_data and player_data[xop_key] > 0:
                historical_xop_values.append(player_data[xop_key])
                weights.append(season_weight)
        
        if historical_xop_values:
            return np.average(
                historical_xop_values, 
                weights=weights[:len(historical_xop_values)]
            )
        
        return None

    def _calculate_season_progression_factor(
            self, gameweeks_completed: int) -> float:
        """
        Calculate how reliable current season stats are based on games played.
        
        Returns:
            float: 0.0 = very early season (high volatility), 
                  1.0 = late season (reliable)
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

    def _calculate_regression_modifier(self, historical_baseline: float, 
                                     current_xop: float, 
                                     season_progression: float, 
                                     position: str) -> tuple:
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
        
        if deviation > 0:  # Currently overperforming historical baseline
            if season_progression < 0.6:
                # Early season overperformance - expect significant regression
                regression_strength = abs_deviation * sensitivity * 0.15
                modifier = 1.0 - regression_strength
                volatility = abs_deviation * (1.0 - season_progression) * 0.3
            else:
                # Late season overperformance - might be genuine improvement
                regression_strength = abs_deviation * sensitivity * 0.08
                modifier = 1.0 - regression_strength
                volatility = abs_deviation * 0.1
                
        else:  # Currently underperforming historical baseline  
            if season_progression < 0.6:
                # Early season underperformance - expect positive regression
                regression_strength = abs_deviation * sensitivity * 0.12
                modifier = 1.0 + regression_strength
                volatility = abs_deviation * (1.0 - season_progression) * 0.25
            else:
                # Late season underperformance - concerning, might be decline
                regression_strength = abs_deviation * sensitivity * 0.05
                modifier = 1.0 + regression_strength
                volatility = abs_deviation * 0.15
        
        return modifier, min(volatility, 0.4)  # Cap volatility at 40%

    def _calculate_new_player_modifier(self, current_xop: float, 
                                     season_progression: float, 
                                     current_xg_context: str) -> tuple:
        """
        Calculate modifier for new players without historical data.
        
        Returns:
            tuple: (modifier, volatility_score)
        """
        if current_xg_context == 'insufficient_data':
            return 1.0, 0.2  # Neutral with moderate volatility
        
        deviation_from_neutral = current_xop - 1.0
        
        if season_progression < 0.6:
            # Early season: high volatility,
            # limited trust in current performance
            if current_xop > 1.3:  # Very high performance
                modifier = 1.0 + (deviation_from_neutral * 0.1)
                volatility = 0.35
            elif current_xop < 0.7:  # Very poor performance  
                modifier = 1.0 + (deviation_from_neutral * 0.1)
                volatility = 0.3
            else:  # Reasonable performance
                modifier = 1.0 + (deviation_from_neutral * 0.15)
                volatility = 0.2
        else:
            # Late season: current performance more reliable for new players
            if current_xop > 1.2:  # High performance
                modifier = 1.0 + (deviation_from_neutral * 0.2)
                volatility = 0.15
            elif current_xop < 0.8:  # Poor performance
                modifier = 1.0 + (deviation_from_neutral * 0.2)
                volatility = 0.2
            else:  # Normal performance
                modifier = 1.0 + (deviation_from_neutral * 0.25)
                volatility = 0.1
        
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
            exponential_bonus = (volatility - 0.25) * 2.0
            base_penalty += exponential_bonus
        
        return min(base_penalty, 0.35)  # Cap at 35% penalty
    
    def calculate_data_availability_factor(self, player_data: dict) -> float:
        """
        Calculate factor to avoid over-penalising players with limited 
        historical data.
        
        Args:
            player_data (dict): Player's historical data
            
        Returns:
            float: Factor to reduce impact of xG modifier for players 
                  with limited data
        """
        seasons_with_data = 0
        total_games = 0
        
        for season in self.config.PAST_SEASONS:
            games_key = f"games_{season}"
            if (games_key in player_data and 
                player_data[games_key] > 8):  # Minimum 8 games
                seasons_with_data += 1
                total_games += player_data[games_key]
        
        # Factor based on data availability
        if seasons_with_data == 0:
            return 0.1  # Minimal impact
        elif seasons_with_data == 1:
            return 0.5  # Moderate impact
        else:
            # Additional bonus for more total games
            games_factor = min(total_games / 60, 1.0)  # Cap at 60 games
            return 0.7 + (0.3 * games_factor)
    
    def _get_current_season_data(self, current_players: pd.DataFrame) -> dict:
        """
        Extract current season data in historical format after GW8.
        
        Args:
            current_players (pd.DataFrame): Current season player data
            
        Returns:
            dict: Current season data formatted like historical seasons
        """
        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        
        # Only use current season data after 8 gameweeks
        if gameweeks_completed < 8:
            return {}
        
        current_season_data = {}
        current_season = "2024-25"  # Current season identifier
        
        for _, player in current_players.iterrows():
            name_key = player.get("name_key", "")
            if not name_key:
                continue
            
            # Calculate per-game metrics
            minutes_played = player.get("minutes", 0)
            games_played = max(1, round(minutes_played / 90))
            
            if games_played >= 8:  # Only include players with decent sample
                points_per_game = player.get("total_points", 0) / games_played
                
                # Calculate xG metrics if available
                goals = player.get("goals_scored", 0)
                assists = player.get("assists", 0)
                xg = player.get("expected_goals", 0)
                xa = player.get("expected_assists", 0)
                
                goal_involvements_per_game = (goals + assists) / games_played
                xgi_per_game = (xg + xa) / games_played
                
                # Calculate xOP if sufficient data
                historical_xop = 1.0
                if xgi_per_game > 0.1:
                    historical_xop = goal_involvements_per_game / xgi_per_game
                    historical_xop = max(0.2, min(3.0, historical_xop))
                
                current_season_data[name_key] = {
                    f"ppg_{current_season}": points_per_game,
                    f"games_{current_season}": games_played,
                    f"reliability_{current_season}": min(1.0, games_played / gameweeks_completed),
                    f"historical_xOP_{current_season}": historical_xop,
                    f"position_{current_season}": player.get("pos_id", 3)
                }
        
        return current_season_data
    
    def merge_past_seasons(self, current: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced merge with proper NA handling for missing historical data,
        conservative volatility penalties, and current season integration 
        after GW8.

        Args:
            current (pd.DataFrame): Current season player data.

        Returns:
            pd.DataFrame: Enhanced player data with weighted xG performance 
                         modifiers and current season integration.
        """
        # Fetch historical data
        hist_frames = [
            self.fetch_past_season_points(s) for s in self.config.PAST_SEASONS
        ]
        
        # Merge historical data
        hist = self._merge_historical_frames(hist_frames)
        
        # After GW8, include current season data in historical calculations
        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        if gameweeks_completed >= 8:
            current_season_data = self._get_current_season_data(current)
            hist = self._integrate_current_season_data(hist, current_season_data)
        
        # Calculate weighted averages (now includes current season if applicable)
        hist = self._calculate_weighted_historical_averages(hist)
        
        # Calculate xG performance modifiers
        hist = self._calculate_xg_consistency_modifiers(hist)
        
        # Calculate current season reliability
        current_reliability = self._calculate_current_reliability(current)
        current = current.assign(current_reliability=current_reliability)
        
        if self.config.GRANULAR_OUTPUT:
            if gameweeks_completed >= 8:
                print("Current season data integrated into historical analysis "
                      f"after {gameweeks_completed} completed gameweeks")
            print("Calculated reliability based on starts over "
                  f"{gameweeks_completed} completed gameweek(s)")
            print("Applied consistent per-game weighted xG analysis with "
                  f"conservative volatility penalties across "
                  f"{len(self.config.PAST_SEASONS)} seasons")

        # Enhanced merge columns including historical baseline tracking
        merge_cols = [
            "name_key", "avg_ppg_past2", "total_games_past2", 
            "avg_reliability", "historical_xOP", "xConsistency", 
            "xOP_historical_baseline"
        ]
        return current.merge(hist[merge_cols], on="name_key", how="left")
    
    def _integrate_current_season_data(self, hist: pd.DataFrame, 
                                     current_season_data: dict) -> pd.DataFrame:
        """
        Integrate current season data into historical dataframe after GW8.
        
        Args:
            hist (pd.DataFrame): Historical data
            current_season_data (dict): Current season data by name_key
            
        Returns:
            pd.DataFrame: Historical data with current season integrated
        """
        if not current_season_data:
            return hist
        
        # Convert current season data to DataFrame
        current_df_data = []
        for name_key, data in current_season_data.items():
            row_data = {"name_key": name_key}
            row_data.update(data)
            current_df_data.append(row_data)
        
        if not current_df_data:
            return hist
        
        current_df = pd.DataFrame(current_df_data)
        
        # Merge with historical data
        hist = hist.merge(current_df, on="name_key", how="outer")
        
        return hist
    
    def _merge_historical_frames(self, hist_frames: list) -> pd.DataFrame:
        """Merge multiple historical dataframes."""
        # Start with the first season's data
        hist = hist_frames[0].copy()
        
        # Merge additional seasons, keeping only season-specific columns
        for i, extra in enumerate(hist_frames[1:], 1):
            season = self.config.PAST_SEASONS[i]
            
            # Only keep season-specific columns and name_key from 
            # extra dataframe
            season_specific_cols = ["name_key"]
            for col in extra.columns:
                if col.endswith(f"_{season}"):
                    season_specific_cols.append(col)
            
            # Merge only the season-specific columns
            extra_filtered = extra[season_specific_cols].copy()
            hist = hist.merge(extra_filtered, on="name_key", how="outer")
        
        return hist
    
    def _calculate_weighted_historical_averages(
            self, hist: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate weighted averages from historical data, now potentially 
        including current season after GW8.
        """
        # Get relevant columns for calculations (including current season)
        ppg_cols = [c for c in hist.columns if c.startswith("ppg_")]
        games_cols = [c for c in hist.columns if c.startswith("games_")]
        reliability_cols = [c for c in hist.columns if c.startswith("reliability_")]
        historical_xop_cols = [c for c in hist.columns if c.startswith("historical_xOP_")]

        # Determine weights (adjust if current season is included)
        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        if gameweeks_completed >= 8 and "ppg_2024-25" in ppg_cols:
            # Include current season with proportional weight
            current_season_weight = min(0.6, gameweeks_completed / 38)
            historical_weights = [w * (1 - current_season_weight) 
                                for w in self.config.HISTORIC_SEASON_WEIGHTS]
            all_weights = [current_season_weight] + historical_weights
            
            # Reorder columns to put current season first
            ppg_cols = [c for c in ppg_cols if "2024-25" in c] + \
                      [c for c in ppg_cols if "2024-25" not in c]
            games_cols = [c for c in games_cols if "2024-25" in c] + \
                        [c for c in games_cols if "2024-25" not in c]
            reliability_cols = [c for c in reliability_cols if "2024-25" in c] + \
                              [c for c in reliability_cols if "2024-25" not in c]
            historical_xop_cols = [c for c in historical_xop_cols if "2024-25" in c] + \
                                 [c for c in historical_xop_cols if "2024-25" not in c]
        else:
            # Use standard historical weights
            all_weights = self.config.HISTORIC_SEASON_WEIGHTS

        # Initialise averages
        hist["avg_ppg_past2"] = 0.0
        hist["total_games_past2"] = 0
        hist["avg_reliability"] = 0.0
        hist["historical_xOP"] = 1.0

        # Calculate traditional weighted averages
        for ppg_col, games_col, reliability_col, weight in zip(
            ppg_cols, games_cols, reliability_cols, all_weights
        ):
            if ppg_col not in hist.columns:
                continue
                
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
        for xop_col, weight in zip(historical_xop_cols, all_weights):
            if xop_col not in hist.columns:
                continue
                
            hist[xop_col] = hist[xop_col].fillna(1.0)  # Default to neutral
            # Only include seasons where player had sufficient games
            season_suffix = xop_col.split('_')[-1]
            games_col = f"games_{season_suffix}"
            if games_col in hist.columns:
                sufficient_games_mask = hist[games_col] >= 8
                # Initialise historical_xOP to 1.0 for first calculation
                if xop_col == historical_xop_cols[0]:
                    hist["historical_xOP"] = 1.0
                # Add weighted contribution
                hist.loc[sufficient_games_mask, "historical_xOP"] += (
                    (hist.loc[sufficient_games_mask, xop_col] - 1.0) * weight
                )

        # Round historical_xOP to 2 decimal places
        hist["historical_xOP"] = hist["historical_xOP"].round(2)
        
        return hist
    
    def _calculate_xg_consistency_modifiers(
            self, hist: pd.DataFrame) -> pd.DataFrame:
        """Calculate xG performance modifiers for each player."""
        hist["xConsistency"] = 1.0  # Final Expected modifier
        hist["xOP_historical_baseline"] = np.nan
        
        for idx, row in hist.iterrows():
            player_data = row.to_dict()
            player_data['current_position'] = self._get_player_position(row)
            
            # Check if player has historical xOP data
            has_historical_data = self._player_has_historical_data(player_data)
            
            if has_historical_data:
                # Calculate normal xG modifier
                xg_modifier = self.calculate_xg_performance_modifier(
                    player_data)
                hist.loc[idx, "xOP_historical_baseline"] = row.get(
                    "historical_xOP", 1.0)
            else:
                # New player - calculate modifier but mark baseline as NA
                xg_modifier = self.calculate_xg_performance_modifier(
                    player_data)
                hist.loc[idx, "xOP_historical_baseline"] = np.nan
            
            # Apply data availability factor
            if has_historical_data:
                availability_factor = self.calculate_data_availability_factor(
                    player_data)
            else:
                availability_factor = 0.7  # Give new players moderate impact
            
            # Blend with neutral (1.0) based on data availability
            final_modifier = 1.0 + ((xg_modifier - 1.0) * availability_factor)
            hist.loc[idx, "xConsistency"] = round(final_modifier, 2)
        
        return hist
    
    def _player_has_historical_data(self, player_data: dict) -> bool:
        """Check if player has historical xOP data."""
        for season in self.config.PAST_SEASONS:
            xop_key = f"historical_xOP_{season}"
            games_key = f"games_{season}"
            if (xop_key in player_data and 
                games_key in player_data and 
                player_data[games_key] >= 8 and
                player_data[xop_key] > 0):
                return True
        return False
    
    def _calculate_current_reliability(
            self, current: pd.DataFrame) -> pd.Series:
        """Calculate current season reliability."""
        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        current_reliability = current["starts"] / gameweeks_completed
        return current_reliability.clip(upper=1.0)
    
    def _get_player_position(self, player_row: pd.Series) -> str:
        """
        Determine player's position from historical data.
        
        Args:
            player_row (pd.Series): Player's historical data row
            
        Returns:
            str: Player position (GK, DEF, MID, FWD)
        """
        # Look for position in any season (including current)
        all_seasons = self.config.PAST_SEASONS + ["2024-25"]
        for season in all_seasons:
            pos_key = f"position_{season}"
            if pos_key in player_row and pd.notna(player_row[pos_key]):
                pos_num = player_row[pos_key]
                pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
                return pos_map.get(pos_num, "MID")
        
        return "MID"  # Default fallback