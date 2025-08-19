"""Module for calculating player scores based on form, history, and fixtures."""

import pandas as pd


class ScoringEngine:
    """Handles the calculation of FPL scores for players."""
    
    def __init__(self, config):
        self.config = config
    
    def build_scores(self, players: pd.DataFrame, fixture_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate FPL scores and base quality scores for each player based on form,
        historical performance, and fixture difficulty, with reliability considerations.

        Args:
            players (pd.DataFrame): Player data with form and historical PPG.
            fixture_scores (pd.DataFrame): Fixture difficulty data.

        Returns:
            pd.DataFrame: Player data with calculated fpl_score and base_quality
                         for each player.
        """
        df = players.merge(fixture_scores, on="name_key", how="left")

        # Fill NaN values to prevent PuLP errors
        df["avg_ppg_past2"] = df["avg_ppg_past2"].fillna(0)
        df["avg_reliability"] = df["avg_reliability"].fillna(0)
        df["current_reliability"] = df["current_reliability"].fillna(0)
        df["fixture_bonus"] = df["fixture_bonus"].fillna(0)
        df["xConsistency"] = df["xConsistency"].fillna(1.0)  # Neutral if no data

        # Calculate team and promotion adjustments
        df["promoted_penalty"] = df["team"].apply(
            lambda x: -0.3 if x in self.config.PROMOTED_TEAMS else 0
        )
        df["team_modifier"] = df["team"].map(
            lambda t: self.config.TEAM_MODIFIERS.get(t, 1.0)
        )

        def z(s):
            """Z-score normalisation with NaN/inf handling."""
            s_clean = s.fillna(0)  # Replace NaN with 0
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

        # Calculate base quality components (no reliability factored in)
        form_component = self.config.FORM_WEIGHT * z(df["form"])
        historic_component = self.config.HISTORIC_WEIGHT * z(df["avg_ppg_past2"])
        fixture_component = self.config.DIFFICULTY_WEIGHT * z(df["fixture_bonus"])

        # Base quality score (for projected points when they DO play)
        # Now includes xG performance modifier
        df["base_quality"] = (
            (form_component + historic_component + fixture_component + df["promoted_penalty"]) 
            * df["team_modifier"] 
            * df["xConsistency"]
        )

        # Apply reliability adjustments for squad selection
        # Current season reliability is 5x more important than historical
        reliability_bonus = (
            df["current_reliability"] * 1.5  # Current season: games/GWs elapsed
            + df["avg_reliability"] * 0.3  # Historical: average games/season ratio
        ) - 0.75  # Centre around 0 (0.75 would be 50% current + 25% historical)

        # Penalty for historically unreliable players (rotation risks)
        df["historically_unreliable_penalty"] = 0.0  # Initialise column
        unreliable_mask = df["avg_reliability"] < 0.6  # Less than 60% historical games
        df.loc[unreliable_mask, "historically_unreliable_penalty"] = -0.15

        # Extra penalty for current season rotation risks
        current_unreliable_mask = df["current_reliability"] < 0.7  # <70% games this season
        df.loc[current_unreliable_mask, "historically_unreliable_penalty"] -= 0.2

        # FPL score (for squad selection - includes reliability)
        df["fpl_score"] = (
            df["base_quality"] + reliability_bonus + df["historically_unreliable_penalty"]
        )

        # Final safety check - replace any NaN/inf in both scores
        df["base_quality"] = df["base_quality"].fillna(0)
        df["base_quality"] = df["base_quality"].replace([float("inf"), float("-inf")], 0)
        df["fpl_score"] = df["fpl_score"].fillna(0)
        df["fpl_score"] = df["fpl_score"].replace([float("inf"), float("-inf")], 0)

        # Calculate projected points here (moved from PointsCalculator)
        df = self._calculate_projected_points(df)

        # Apply availability filter based on config setting
        df = self._apply_availability_filter(df)

        return df
    
    def _apply_availability_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply availability filter to players based on config setting.
        
        Args:
            df (pd.DataFrame): Player dataframe with scores
            
        Returns:
            pd.DataFrame: Modified dataframe with availability considerations
        """
        # Check if EXCLUDE_UNAVAILABLE setting exists, default to True for backwards compatibility
        exclude_unavailable = getattr(self.config, 'EXCLUDE_UNAVAILABLE', True)
        
        if not exclude_unavailable:
            print("📋 EXCLUDE_UNAVAILABLE = False: Including unavailable players in optimization")
            return df
        
        # Identify unavailable players
        unavailable_mask = (
            (df["status"] != "a") & 
            (df["chance_of_playing_next_round"].fillna(100) < 75)
        )
        
        unavailable_count = unavailable_mask.sum()
        
        if unavailable_count > 0:
            print(f"⚠️  Setting FPL scores and projected points to 0 for {unavailable_count} unavailable players")
            df.loc[unavailable_mask, "fpl_score"] = 0.0
            df.loc[unavailable_mask, "projected_points"] = 0.0
        else:
            print("✅ All players are available for selection")
        
        return df
    
    def _calculate_projected_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate projected points for each player based on their base quality.
        This is moved from PointsCalculator to ensure it's available during squad selection.

        Args:
            df (pd.DataFrame): Player data with base_quality and position.

        Returns:
            pd.DataFrame: DataFrame with added projected_points column.
        """
        # Convert position to string if it's categorical
        if df["position"].dtype.name == "category":
            position_values = df["position"].astype(str)
        else:
            position_values = df["position"]

        # Calculate baseline points for each position
        df["baseline_points"] = position_values.map(self.config.BASELINE_POINTS_PER_GAME)

        # Convert base quality to points adjustment (not FPL score)
        # This represents how good they are WHEN they play
        df["points_adjustment"] = df["base_quality"] * self.config.FPL_SCORE_TO_POINTS_MULTIPLIER

        # Calculate projected points
        df["projected_points"] = df["baseline_points"] + df["points_adjustment"]

        # Ensure minimum of 1 point (no player should project negative)
        df["projected_points"] = df["projected_points"].clip(lower=1.0)

        return df