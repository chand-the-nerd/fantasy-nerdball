"""Module for managing historical player performance data."""

import pandas as pd
from ..utils.text_utils import normalize_name


class HistoricalDataManager:
    """Handles fetching and processing of historical player data."""
    
    def __init__(self, config):
        self.config = config
    
    def fetch_past_season_points(self, season_folder: str) -> pd.DataFrame:
        """
        Fetch historical points per game data for a specific season, with
        reliability filtering.

        Args:
            season_folder (str): The season folder name (e.g., "2023-24").

        Returns:
            pd.DataFrame: DataFrame containing web_name, points_per_game,
                         reliability_factor, and name_key for the specified season.
        """
        url = (
            f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
            f"master/data/{season_folder}/players_raw.csv"
        )
        df = pd.read_csv(url)

        # Calculate games played and reliability
        df["minutes_played"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
        df["total_points"] = pd.to_numeric(df["total_points"], errors="coerce").fillna(0)

        # Calculate games played (assuming 90 minutes = 1 game)
        df["games_played"] = (df["minutes_played"] / 90).round().clip(lower=0)

        # Calculate season reliability (games played / max possible games ~38)
        # Use 30 as reasonable threshold for "full season" to account for injuries
        df["season_reliability"] = (df["games_played"] / 30).clip(upper=1.0)

        # Calculate points per game, only for players who actually played
        df["points_per_game"] = 0.0
        played_mask = df["games_played"] >= 1
        df.loc[played_mask, "points_per_game"] = (
            df.loc[played_mask, "total_points"] / df.loc[played_mask, "games_played"]
        )

        # Apply reliability penalty for players who weren't regular starters
        # Players who played <80% of games get their PPG penalised
        reliability_threshold = 0.8
        unreliable_mask = df["season_reliability"] < reliability_threshold

        # Penalty factor: if you played 50% of games, your PPG gets reduced
        penalty_factor = df["season_reliability"].clip(lower=0.3)  # Minimum 30% value
        df.loc[unreliable_mask, "points_per_game"] *= penalty_factor[unreliable_mask]

        # Select relevant columns
        result_df = df[
            ["web_name", "points_per_game", "games_played", "season_reliability"]
        ].copy()
        result_df["name_key"] = result_df["web_name"].map(normalize_name)
        result_df = result_df.rename(
            columns={
                "points_per_game": f"ppg_{season_folder}",
                "games_played": f"games_{season_folder}",
                "season_reliability": f"reliability_{season_folder}",
            }
        )

        return result_df
    
    def merge_past_seasons(self, current: pd.DataFrame) -> pd.DataFrame:
        """
        Merge historical points per game data from multiple seasons with the
        current player data, calculating reliability based on games played.

        Args:
            current (pd.DataFrame): Current season player data.

        Returns:
            pd.DataFrame: Current player data merged with weighted average of
                         historical points per game, adjusted for reliability.
        """
        hist_frames = [self.fetch_past_season_points(s) for s in self.config.PAST_SEASONS]
        hist = hist_frames[0]
        for extra in hist_frames[1:]:
            hist = hist.merge(extra, on="name_key", how="outer")

        # Get PPG columns, games columns, and reliability columns
        ppg_cols = [c for c in hist.columns if c.startswith("ppg_")]
        games_cols = [c for c in hist.columns if c.startswith("games_")]
        reliability_cols = [c for c in hist.columns if c.startswith("reliability_")]

        # Calculate weighted average PPG and overall reliability
        hist["avg_ppg_past2"] = 0.0
        hist["total_games_past2"] = 0
        hist["avg_reliability"] = 0.0

        for ppg_col, games_col, reliability_col, weight in zip(
            ppg_cols, games_cols, reliability_cols, self.config.HISTORIC_SEASON_WEIGHTS
        ):
            # Fill NaN values
            hist[ppg_col] = hist[ppg_col].fillna(0)
            hist[games_col] = hist[games_col].fillna(0)
            hist[reliability_col] = hist[reliability_col].fillna(0)

            # Only include seasons where player had meaningful game time (>=8 games)
            # This is roughly 20% of a season - minimum for any consideration
            sufficient_games_mask = hist[games_col] >= 8

            # Add to weighted averages
            hist.loc[sufficient_games_mask, "avg_ppg_past2"] += (
                hist.loc[sufficient_games_mask, ppg_col] * weight
            )
            hist.loc[sufficient_games_mask, "avg_reliability"] += (
                hist.loc[sufficient_games_mask, reliability_col] * weight
            )
            hist["total_games_past2"] += hist[games_col]

        # Calculate current season reliability based on starts
        # Use the gameweek from config - gameweeks completed = current gameweek - 1
        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        
        # Simple and reliable: use starts field from the bootstrap data
        # Reliability = starts / gameweeks_completed
        current_reliability = current["starts"] / gameweeks_completed
        current_reliability = current_reliability.clip(upper=1.0)  # Cap at 100%
        
        print(f"Calculated reliability based on starts over {gameweeks_completed} completed gameweek(s)")

        # Assign current_reliability to the dataframe BEFORE merging
        current = current.assign(current_reliability=current_reliability)
        
        return current.merge(
            hist[["name_key", "avg_ppg_past2", "total_games_past2", "avg_reliability"]],
            on="name_key",
            how="left",
        )
