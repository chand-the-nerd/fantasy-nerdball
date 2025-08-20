"""Module for calculating projected points for players."""

import pandas as pd


class PointsCalculator:
    """Handles calculation of projected points for players."""
    
    def __init__(self, config):
        self.config = config
    
    def calculate_projected_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate projected points for each player - now just a pass-through 
        since projected points are calculated in ScoringEngine with proper
        DGW/BGW handling.

        Args:
            df (pd.DataFrame): Player data with projected_points already 
                              calculated.

        Returns:
            pd.DataFrame: DataFrame with projected_points column (unchanged).
        """
        # Projected points are now calculated in ScoringEngine.build_scores()
        # This method is kept for compatibility but doesn't need to do 
        # the calculation
        return df
    
    def add_points_analysis_to_display(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add projected points columns and score components for better display,
        properly handling DGW scenarios.

        Args:
            df (pd.DataFrame): Squad data with projected_points already 
                              calculated.

        Returns:
            pd.DataFrame: DataFrame with display formatting added.
        """
        # Projected points should already be calculated by ScoringEngine
        df = self.calculate_projected_points(df)

        # Calculate minutes per game as total minutes / gameweeks completed
        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        df["minspg"] = (
            df["minutes"] / gameweeks_completed).round(0).astype(int)

        # Prepare display columns with exact names requested
        df = self._prepare_display_columns(df)

        return df
    
    def _prepare_display_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare display columns with proper formatting including DGW 
        handling."""
        # Round form to 1 decimal place
        df["form"] = df["form"].round(1)
        
        # Historic PPG from past seasons
        df["historic_ppg"] = df["avg_ppg_past2"].round(1)

        # Use the fixture difficulty from fetch_player_fixture_difficulty
        df["fixture_diff"] = (6 - df["diff"]).round(1)

        # Reliability as percentage (starts/games)
        df["reliability"] = (
            df["current_reliability"] * 100).round(0).astype(int)
        
        # Show total projected points per gameweek (not per fixture)
        df["proj_pts"] = df["projected_points"].round(1)

        return df