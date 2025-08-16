"""Module for calculating projected points for players."""

import pandas as pd


class PointsCalculator:
    """Handles calculation of projected points for players."""
    
    def __init__(self, config):
        self.config = config
    
    def calculate_projected_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate projected points for each player - now just a pass-through since
        projected points are calculated in ScoringEngine.

        Args:
            df (pd.DataFrame): Player data with projected_points already calculated.

        Returns:
            pd.DataFrame: DataFrame with projected_points column (unchanged).
        """
        # Projected points are now calculated in ScoringEngine.build_scores()
        # This method is kept for compatibility but doesn't need to do the calculation
        return df
    
    def add_points_analysis_to_display(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add projected points columns and score components for better display.

        Args:
            df (pd.DataFrame): Squad data with projected_points already calculated.

        Returns:
            pd.DataFrame: DataFrame with display formatting added.
        """
        # Projected points should already be calculated by ScoringEngine
        df = self.calculate_projected_points(df)

        # Calculate minutes per game (total minutes / games with >0 minutes)
        games_played = (df["minutes"] > 0).astype(int)
        df["minspg"] = 0.0
        played_mask = games_played > 0
        df.loc[played_mask, "minspg"] = (
            df.loc[played_mask, "minutes"] / games_played[played_mask]
        )
        df["minspg"] = df["minspg"].round(0).astype(int)

        # Prepare display columns with exact names requested
        df["form"] = df["form"].round(1)
        df["historic_ppg"] = df["avg_ppg_past2"].round(1)

        # Use the fixture difficulty average from fetch_player_fixture_difficulty
        # This shows average difficulty over FIRST_N_GAMEWEEKS (same as algorithm uses)
        df["fixture_diff"] = (6 - df["fixture_bonus"]).round(1)

        df["reliability"] = (df["current_reliability"] * 100).round(0).astype(int)
        df["proj_pts"] = df["projected_points"].round(1)

        return df