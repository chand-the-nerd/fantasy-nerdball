"""Module for calculating player scores based on form, history, and 
fixtures."""

import pandas as pd


class ScoringEngine:
    """Handles the calculation of FPL scores for players."""
    
    def __init__(self, config):
        self.config = config
    
    def build_scores(self, players: pd.DataFrame, 
                    fixture_scores: pd.DataFrame) -> tuple:
        """
        Calculate FPL scores and base quality scores for each player based on 
        form, historical performance, and fixture difficulty, with reliability
        considerations.

        Args:
            players (pd.DataFrame): Player data with form and historical PPG.
            fixture_scores (pd.DataFrame): Fixture difficulty data.

        Returns:
            tuple: (scored_dataframe, involvement_stats_dict) containing the 
                   player data with calculated scores and involvement statistics.
        """
        df = players.merge(fixture_scores, on="name_key", how="left")

        # Fill NaN values to prevent PuLP errors
        df = self._fill_missing_values(df)

        # Calculate team and promotion adjustments
        df = self._calculate_team_adjustments(df)

        # Calculate base quality components
        df = self._calculate_base_quality(df)

        # Apply reliability adjustments for squad selection
        df = self._apply_reliability_adjustments(df)

        # Calculate projected points
        df = self._calculate_projected_points(df)

        # Apply availability filter and get involvement stats
        df, involvement_stats = self._apply_availability_filter(df)

        return df, involvement_stats
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values to prevent errors in calculations."""
        fill_values = {
            "avg_ppg_past2": 0,
            "avg_reliability": 0,
            "current_reliability": 0,
            "fixture_bonus": 0,
            "xConsistency": 1.0  # Neutral if no data
        }
        
        for col, fill_val in fill_values.items():
            df[col] = df[col].fillna(fill_val)
        
        return df
    
    def _calculate_team_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate team and promotion adjustments."""
        df["promoted_penalty"] = df["team"].apply(
            lambda x: -0.3 if x in self.config.PROMOTED_TEAMS else 0
        )
        df["team_modifier"] = df["team"].map(
            lambda t: self.config.TEAM_MODIFIERS.get(t, 1.0)
        )
        return df
    
    def _calculate_base_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate base quality components using z-score normalisation."""
        # Calculate z-scores for each component
        form_component = self.config.FORM_WEIGHT * self._z_score(df["form"])
        historic_component = (
            self.config.HISTORIC_WEIGHT * self._z_score(df["avg_ppg_past2"])
        )
        fixture_component = (
            self.config.DIFFICULTY_WEIGHT * self._z_score(df["fixture_bonus"])
        )

        # Base quality score (for projected points when they DO play)
        # Now includes xG performance modifier
        df["base_quality"] = (
            (form_component + historic_component + fixture_component + 
             df["promoted_penalty"]) 
            * df["team_modifier"] 
            * df["xConsistency"]
        )
        
        return df
    
    def _z_score(self, series: pd.Series) -> pd.Series:
        """
        Z-score normalisation with NaN/inf handling.
        
        Args:
            series (pd.Series): Series to normalise
            
        Returns:
            pd.Series: Normalised series
        """
        s_clean = series.fillna(0)  # Replace NaN with 0
        s_mean = s_clean.mean()
        s_std = s_clean.std(ddof=0)

        # Prevent division by zero
        if s_std == 0 or pd.isna(s_std):
            return pd.Series([0.0] * len(s_clean), index=s_clean.index)

        z_scores = (s_clean - s_mean) / s_std

        # Replace any remaining NaN/inf values
        z_scores = z_scores.fillna(0)
        z_scores = z_scores.replace([float("inf"), float("-inf")], 0)

        return z_scores
    
    def _apply_reliability_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply reliability adjustments for squad selection."""
        # Current season reliability is 5x more important than historical
        reliability_bonus = (
            df["current_reliability"] * 1.5  # Current season weighting
            + df["avg_reliability"] * 0.3    # Historical weighting
        ) - 0.75  # Centre around 0

        # Enhanced penalty for historically unreliable players (rotation risks)
        df["historically_unreliable_penalty"] = 0.0  # Initialise column
        
        # Historical reliability penalty
        unreliable_mask = df["avg_reliability"] < 0.6  # <60% historical games
        df.loc[unreliable_mask, "historically_unreliable_penalty"] = -0.15

        # Enhanced current season rotation risk penalty
        if self.config.GAMEWEEK > 1:
            gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
            
            # Calculate current season start percentage
            current_start_pct = df["starts"].fillna(0) / gameweeks_completed
            
            # Very unreliable (rotation risk) - stronger penalty
            very_unreliable_mask = current_start_pct < 0.5  # <50% starts
            df.loc[very_unreliable_mask, 
                   "historically_unreliable_penalty"] -= 0.3
            
            # Moderately unreliable
            moderately_unreliable_mask = ((current_start_pct >= 0.5) & 
                                        (current_start_pct < 0.7))
            df.loc[moderately_unreliable_mask, 
                   "historically_unreliable_penalty"] -= 0.15
        else:
            # First gameweek - use traditional penalty
            current_unreliable_mask = df["current_reliability"] < 0.7
            df.loc[current_unreliable_mask,
                   "historically_unreliable_penalty"] -= 0.2

        # FPL score (for squad selection - includes reliability)
        df["fpl_score"] = (
            df["base_quality"] + reliability_bonus + 
            df["historically_unreliable_penalty"]
        )

        return df
    
    def _calculate_projected_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate projected points for each player based on their base quality.
        """
        # Convert position to string if it's categorical
        if df["position"].dtype.name == "category":
            position_values = df["position"].astype(str)
        else:
            position_values = df["position"]

        # Calculate baseline points for each position
        df["baseline_points"] = position_values.map(
            self.config.BASELINE_POINTS_PER_GAME
        )

        # Convert base quality to points adjustment (not FPL score)
        # This represents how good they are WHEN they play
        df["points_adjustment"] = (
            df["base_quality"] * self.config.FPL_SCORE_TO_POINTS_MULTIPLIER
        )

        # Calculate projected points
        df["projected_points"] = (
            df["baseline_points"] + df["points_adjustment"]
        )

        # Ensure minimum of 1 point (no player should project negative)
        df["projected_points"] = df["projected_points"].clip(lower=1.0)

        return df
    
    def _apply_availability_filter(self, df: pd.DataFrame) -> tuple:
        """
        Apply availability filter to players based on config setting.
        Enhanced to penalize non-playing players after GW1.
        
        Args:
            df (pd.DataFrame): Player dataframe with scores
            
        Returns:
            tuple: (modified_dataframe, involvement_stats_dict)
        """
        # Initialize involvement stats
        involvement_stats = {
            'high_involvement': 0,
            'zero_involvement': 0,
            'low_involvement': 0,
            'unavailable': 0,
            'exclude_unavailable': True
        }
        
        # Check if EXCLUDE_UNAVAILABLE setting exists, default to True
        exclude_unavailable = getattr(self.config, 'EXCLUDE_UNAVAILABLE', True)
        involvement_stats['exclude_unavailable'] = exclude_unavailable
        
        if not exclude_unavailable:
            return df, involvement_stats
        
        # Identify unavailable players (injury/suspension)
        unavailable_mask = (
            (df["status"] != "a") & 
            (df["chance_of_playing_next_round"].fillna(100) < 75)
        )
        
        involvement_stats['unavailable'] = unavailable_mask.sum()
        
        # After GW1, heavily penalize players who haven't been playing
        if self.config.GAMEWEEK > 1:
            gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
            
            # Players who haven't started any games AND have 0 form
            zero_involvement_mask = (
                (df["starts"].fillna(0) == 0) & 
                (df["form"].fillna(0) == 0) &
                (df["minutes"].fillna(0) < 45)  # Less than half a game total
            )
            
            # Players with very low involvement (rotation risks)
            low_involvement_mask = (
                (df["starts"].fillna(0) / gameweeks_completed < 0.3) &  
                (df["minutes"].fillna(0) / gameweeks_completed < 30) &   
                ~zero_involvement_mask  # Don't double-count zero players
            )
            
            # Players with high involvement (reliable starters)
            high_involvement_mask = (
                (df["starts"].fillna(0) / gameweeks_completed >= 0.7) &  
                (df["minutes"].fillna(0) / gameweeks_completed >= 60)
            )
            
            # Store stats
            involvement_stats['high_involvement'] = high_involvement_mask.sum()
            involvement_stats['zero_involvement'] = zero_involvement_mask.sum()
            involvement_stats['low_involvement'] = low_involvement_mask.sum()
            
            # Apply penalties/zeroing
            if zero_involvement_mask.sum() > 0:
                # Set scores to 0 for completely uninvolved players
                df.loc[zero_involvement_mask, "fpl_score"] = 0.0
                df.loc[zero_involvement_mask, "projected_points"] = 0.0
                
            if low_involvement_mask.sum() > 0:
                # Heavy penalty for low involvement players (75% reduction)
                df.loc[low_involvement_mask, "fpl_score"] *= 0.25
                df.loc[low_involvement_mask, "projected_points"] *= 0.25
        
        # Apply original unavailable filter
        if unavailable_mask.sum() > 0:
            # Set scores to 0 for unavailable players
            df.loc[unavailable_mask, "fpl_score"] = 0.0
            df.loc[unavailable_mask, "projected_points"] = 0.0
        
        return df, involvement_stats
    
    def _finalise_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final safety check - replace any remaining NaN/inf values."""
        score_columns = ["base_quality", "fpl_score", "projected_points"]
        
        for col in score_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                df[col] = df[col].replace([float("inf"), float("-inf")], 0)
        
        return df