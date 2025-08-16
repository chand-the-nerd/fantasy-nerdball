"""Utility functions for file operations and data persistence."""

import os
import pandas as pd


class FileUtils:
    """Utility class for file operations."""
    
    @staticmethod
    def load_previous_squad(gameweek: int) -> pd.DataFrame:
        """
        Load the previous gameweek's squad from CSV file.

        Args:
            gameweek (int): Current gameweek (will load gameweek-1's squad).

        Returns:
            pd.DataFrame or None: Previous squad data or None if file doesn't exist.
        """
        if gameweek <= 1:
            print("No previous squad to load (this is GW1 or earlier)")
            return None

        prev_gw = gameweek - 1
        squad_file = f"squads/gw{prev_gw}/full_squad.csv"

        if not os.path.exists(squad_file):
            print(f"Warning: Previous squad file not found at {squad_file}")
            print("Proceeding without transfer constraints (assuming new team)")
            return None

        try:
            prev_squad = pd.read_csv(squad_file)
            print(f"Loaded previous squad from {squad_file}")
            return prev_squad
        except Exception as e:
            print(f"Error loading previous squad: {e}")
            return None
    
    @staticmethod
    def save_squad_data(gameweek: int, starting_display: pd.DataFrame, bench_display: pd.DataFrame):
        """
        Save squad data to CSV files with projected points.

        Args:
            gameweek (int): Current gameweek number.
            starting_display (pd.DataFrame): Starting XI with display formatting.
            bench_display (pd.DataFrame): Bench with display formatting.
        """
        squad_dir = f"squads/gw{gameweek}"
        os.makedirs(squad_dir, exist_ok=True)

        # Save combined squad with all details including projected points
        squad_combined = pd.concat([starting_display, bench_display], ignore_index=True)
        squad_combined["squad_role"] = ["Starting XI"] * len(starting_display) + [
            "Bench"
        ] * len(bench_display)
        combined_file = f"{squad_dir}/full_squad.csv"
        squad_combined.to_csv(combined_file, index=False)

        # Save simple squad overview with projected points included
        simple_squad = squad_combined[
            [
                "display_name",
                "position",
                "now_cost_m",
                "team",
                "projected_points",
                "squad_role",
            ]
        ].copy()
        simple_squad = simple_squad.rename(
            columns={
                "display_name": "player",
                "now_cost_m": "price",
                "team": "club",
                "projected_points": "projected_points",
            }
        )
        simple_file = f"{squad_dir}/full_squad_simple.csv"
        simple_squad.to_csv(simple_file, index=False)

        print(f"\nSquad saved to {squad_dir}/")
        print(f"  - {combined_file}")
        print(f"  - {simple_file}")