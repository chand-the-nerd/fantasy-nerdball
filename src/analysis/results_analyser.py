"""Module for analysing and creating results from previous gameweeks."""

import os
import pandas as pd
from ..api.fpl_client import FPLClient
from ..utils.file_utils import FileUtils


class ResultsAnalyser:
    """Handles analysis of previous gameweek results."""
    
    def __init__(self, config):
        self.config = config
        self.fpl_client = FPLClient()
    
    def get_actual_points_for_gameweek(self, player_id: int, gameweek: int) -> int:
        """
        Fetch actual points scored by a player in a specific gameweek.

        Args:
            player_id (int): The FPL player ID.
            gameweek (int): The gameweek number.

        Returns:
            int: Actual points scored, or 0 if no data available.
        """
        try:
            data = self.fpl_client.get_player_summary(player_id)

            for history in data.get("history", []):
                if history["round"] == gameweek:
                    return history["total_points"]
            return 0
        except:
            return 0
    
    def create_previous_gameweek_results(self, current_gameweek: int):
        """
        Create results files for the previous gameweek comparing projected vs actual points.
        Enhanced with better error handling and logging.

        Args:
            current_gameweek (int): Current gameweek number (results will be created for gameweek-1)
        """
        if current_gameweek <= 1:
            print("Cannot create results for GW0 or earlier")
            return

        prev_gw = current_gameweek - 1
        prev_squad_dir = f"squads/gw{prev_gw}"

        # Check if previous gameweek squad file exists
        prev_squad_file = f"{prev_squad_dir}/full_squad.csv"
        if not os.path.exists(prev_squad_file):
            print(f"No previous squad found at {prev_squad_file}")
            return

        print(f"\n=== Creating results for GW{prev_gw} ===")

        try:
            # Load previous squad
            prev_squad = pd.read_csv(prev_squad_file)
            print(f"Loaded squad with {len(prev_squad)} players")

            # Get actual points for each player
            print(f"Fetching actual points for GW{prev_gw}...")
            actual_points = []

            for idx, player in prev_squad.iterrows():
                player_id = player.get("id")
                player_name = player.get("display_name", "Unknown")

                if pd.isna(player_id):
                    print(f"  Warning: No ID found for {player_name}")
                    actual_points.append(0)
                    continue

                try:
                    actual = self.get_actual_points_for_gameweek(int(player_id), prev_gw)
                    actual_points.append(actual)
                    print(f"  {player_name}: {actual} points")
                except Exception as e:
                    print(f"  Error fetching points for {player_name}: {e}")
                    actual_points.append(0)

            # Add actual points to dataframe
            prev_squad["actual_points"] = actual_points
            print(f"Added actual points. Total team points: {sum(actual_points)}")

            # Calculate differences (actual - projected)
            if "projected_points" in prev_squad.columns:
                prev_squad["points_difference"] = (
                    prev_squad["actual_points"] - prev_squad["projected_points"]
                )
                prev_squad["absolute_difference"] = abs(prev_squad["points_difference"])

                total_projected = prev_squad["projected_points"].sum()
                total_actual = sum(actual_points)
                print(
                    f"Total projected: {total_projected:.1f}, Total actual: {total_actual}, Difference: {total_actual - total_projected:+.1f}"
                )
            else:
                print("Warning: No projected_points column found in previous squad")
                prev_squad["projected_points"] = 0  # Add dummy column
                prev_squad["points_difference"] = prev_squad["actual_points"]
                prev_squad["absolute_difference"] = abs(prev_squad["actual_points"])

            # Create full_squad_results.csv
            results_columns = [
                "display_name",
                "position",
                "team",
                "squad_role",
                "projected_points",
                "actual_points",
                "points_difference",
                "absolute_difference",
            ]

            # Only include columns that exist
            available_columns = [
                col for col in results_columns if col in prev_squad.columns
            ]
            results_df = prev_squad[available_columns].copy()

            # Round projected_points and related columns to 1 decimal place in results file
            if "projected_points" in results_df.columns:
                results_df["projected_points"] = results_df["projected_points"].round(1)
            if "points_difference" in results_df.columns:
                results_df["points_difference"] = results_df["points_difference"].round(1)
            if "absolute_difference" in results_df.columns:
                results_df["absolute_difference"] = results_df["absolute_difference"].round(1)

            results_file = f"{prev_squad_dir}/full_squad_results.csv"
            results_df.to_csv(results_file, index=False)
            print(f"✅ Results saved to {results_file}")

            # Create summary.csv
            self.create_summary_analysis(prev_squad, prev_gw, prev_squad_dir)

            print(f"✅ Results creation completed for GW{prev_gw}")

        except Exception as e:
            print(f"❌ Error creating results for GW{prev_gw}: {e}")
            import traceback
            traceback.print_exc()
    
    def create_summary_analysis(self, squad_df: pd.DataFrame, gameweek: int, output_dir: str):
        """
        Create a summary analysis comparing projected vs actual points.
        Enhanced with better error handling.

        Args:
            squad_df (pd.DataFrame): Squad dataframe with projected and actual points
            gameweek (int): Gameweek number
            output_dir (str): Directory to save the summary
        """
        try:
            print(f"Creating summary analysis for GW{gameweek}...")

            # Team totals
            if "projected_points" in squad_df.columns:
                total_projected = squad_df["projected_points"].sum()
            else:
                total_projected = 0

            total_actual = squad_df["actual_points"].sum()
            total_difference = total_actual - total_projected

            # Starting XI vs Bench breakdown
            starting_xi = squad_df[squad_df.get("squad_role", "") == "Starting XI"]
            bench = squad_df[squad_df.get("squad_role", "") == "Bench"]

            if len(starting_xi) == 0:
                print("Warning: No Starting XI players found, using all players")
                starting_xi = squad_df
                bench = pd.DataFrame()

            starting_projected = (
                starting_xi["projected_points"].sum()
                if "projected_points" in starting_xi.columns
                else 0
            )
            starting_actual = starting_xi["actual_points"].sum()
            starting_difference = starting_actual - starting_projected

            bench_projected = (
                bench["projected_points"].sum()
                if "projected_points" in bench.columns
                else 0
            )
            bench_actual = bench["actual_points"].sum()
            bench_difference = bench_actual - bench_projected

            # Position breakdown
            position_summary = []
            for position in ["GK", "DEF", "MID", "FWD"]:
                pos_players = squad_df[squad_df["position"] == position]
                if len(pos_players) > 0:
                    pos_projected = (
                        pos_players["projected_points"].sum()
                        if "projected_points" in pos_players.columns
                        else 0
                    )
                    pos_actual = pos_players["actual_points"].sum()
                    pos_difference = pos_actual - pos_projected
                    pos_count = len(pos_players)

                    position_summary.append(
                        {
                            "position": position,
                            "player_count": pos_count,
                            "projected_points": pos_projected,
                            "actual_points": pos_actual,
                            "difference": pos_difference,
                            "avg_projected": (
                                pos_projected / pos_count if pos_count > 0 else 0
                            ),
                            "avg_actual": pos_actual / pos_count if pos_count > 0 else 0,
                        }
                    )

            # Best and worst performers
            if "points_difference" in squad_df.columns and len(squad_df) > 0:
                best_performer = squad_df.loc[squad_df["points_difference"].idxmax()]
                worst_performer = squad_df.loc[squad_df["points_difference"].idxmin()]
            else:
                best_performer = None
                worst_performer = None

            # Accuracy metrics
            if "absolute_difference" in squad_df.columns:
                mean_absolute_error = squad_df["absolute_difference"].mean()
                players_within_2pts = len(squad_df[squad_df["absolute_difference"] <= 2])
                accuracy_within_2pts = (
                    (players_within_2pts / len(squad_df)) * 100 if len(squad_df) > 0 else 0
                )
            else:
                mean_absolute_error = 0
                accuracy_within_2pts = 0
                players_within_2pts = 0

            # Create summary dataframe
            summary_data = []

            # Team totals
            summary_data.append(
                {
                    "metric": "Total Team Points",
                    "projected": round(total_projected),
                    "actual": total_actual,
                    "difference": round(total_difference),
                    "notes": f"Overall team performance for GW{gameweek}",
                }
            )

            # Starting XI
            summary_data.append(
                {
                    "metric": "Starting XI Points",
                    "projected": round(starting_projected),
                    "actual": starting_actual,
                    "difference": round(starting_difference),
                    "notes": f"Points from {len(starting_xi)} starting players",
                }
            )

            # Bench
            if len(bench) > 0:
                summary_data.append(
                    {
                        "metric": "Bench Points",
                        "projected": round(bench_projected),
                        "actual": bench_actual,
                        "difference": round(bench_difference),
                        "notes": f"Points from {len(bench)} bench players",
                    }
                )

            # Position summaries
            for pos_data in position_summary:
                summary_data.append(
                    {
                        "metric": f'{pos_data["position"]} Total',
                        "projected": round(pos_data["projected_points"]),
                        "actual": pos_data["actual_points"],
                        "difference": round(pos_data["difference"]),
                        "notes": f'{pos_data["player_count"]} players, avg actual: {pos_data["avg_actual"]:.1f}',
                    }
                )

            # Accuracy metrics
            summary_data.append(
                {
                    "metric": "Mean Absolute Error",
                    "projected": "-",
                    "actual": f"{mean_absolute_error:.2f}",
                    "difference": "-",
                    "notes": "Average absolute difference between projected and actual",
                }
            )

            summary_data.append(
                {
                    "metric": "Accuracy (within 2pts)",
                    "projected": "-",
                    "actual": f"{accuracy_within_2pts:.1f}%",
                    "difference": "-",
                    "notes": f"{players_within_2pts}/{len(squad_df)} players within 2 points of projection",
                }
            )

            # Best/worst performers
            if best_performer is not None:
                summary_data.append(
                    {
                        "metric": "Best Performer",
                        "projected": round(best_performer.get("projected_points", 0)),
                        "actual": best_performer["actual_points"],
                        "difference": round(best_performer.get("points_difference", 0)),
                        "notes": f'{best_performer["display_name"]} ({best_performer["position"]})',
                    }
                )

            if worst_performer is not None:
                summary_data.append(
                    {
                        "metric": "Worst Performer",
                        "projected": round(worst_performer.get("projected_points", 0)),
                        "actual": worst_performer["actual_points"],
                        "difference": round(worst_performer.get("points_difference", 0)),
                        "notes": f'{worst_performer["display_name"]} ({worst_performer["position"]})',
                    }
                )

            summary_df = pd.DataFrame(summary_data)

            # Save summary
            summary_file = f"{output_dir}/summary.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"✅ Summary saved to {summary_file}")

            # Print summary to console
            print(f"\n=== GW{gameweek} Performance Summary ===")
            print(
                f"Total Points: {total_actual} (projected: {total_projected:.1f}, difference: {total_difference:+.1f})"
            )
            print(
                f"Starting XI: {starting_actual} (projected: {starting_projected:.1f}, difference: {starting_difference:+.1f})"
            )
            if len(bench) > 0:
                print(
                    f"Bench: {bench_actual} (projected: {bench_projected:.1f}, difference: {bench_difference:+.1f})"
                )
            print(f"Mean Absolute Error: {mean_absolute_error:.2f}")
            print(f"Accuracy (within 2pts): {accuracy_within_2pts:.1f}%")

            if best_performer is not None:
                print(
                    f"Best: {best_performer['display_name']} ({best_performer.get('points_difference', 0):+.1f})"
                )
            if worst_performer is not None:
                print(
                    f"Worst: {worst_performer['display_name']} ({worst_performer.get('points_difference', 0):+.1f})"
                )

        except Exception as e:
            print(f"❌ Error creating summary analysis: {e}")
            import traceback
            traceback.print_exc()