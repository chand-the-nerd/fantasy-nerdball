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
    
    def get_actual_points_for_gameweek(self, player_id: int, 
                                     gameweek: int) -> int:
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
        except Exception:
            return 0
    
    def create_previous_gameweek_results(self, current_gameweek: int):
        """
        Create results files for the previous gameweek comparing projected 
        vs actual points. Enhanced with better error handling and logging.

        Args:
            current_gameweek (int): Current gameweek number (results will be 
                                  created for gameweek-1)
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

        print(f"\n=== GW{prev_gw} SUMMARY ===")

        try:
            self._process_previous_squad_results(
                prev_squad_file, prev_gw, prev_squad_dir
            )
            print(f"\n✅ Results creation completed for GW{prev_gw}")

        except Exception as e:
            print(f"❌ Error creating results for GW{prev_gw}: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_previous_squad_results(self, prev_squad_file: str, 
                                      prev_gw: int, prev_squad_dir: str):
        """Process and analyse previous squad results."""
        # Load previous squad
        prev_squad = pd.read_csv(prev_squad_file)
        actual_points = self._fetch_actual_points(prev_squad, prev_gw)
        
        # Add actual points to dataframe
        prev_squad["actual_points"] = actual_points
        total_actual = sum(actual_points)
        print(f"Total squad points: {total_actual}")

        # Calculate differences (actual - projected)
        prev_squad = self._calculate_point_differences(
            prev_squad, total_actual)
        
        # Create results files
        self._create_results_files(prev_squad, prev_gw, prev_squad_dir)
        
        # Create summary analysis
        self.create_summary_analysis(prev_squad, prev_gw, prev_squad_dir)
    
    def _fetch_actual_points(self, prev_squad: pd.DataFrame, 
                           prev_gw: int) -> list:
        """Fetch actual points for all players in the squad."""
        actual_points = []
        
        for idx, player in prev_squad.iterrows():
            player_id = player.get("id")
            player_name = player.get("display_name", "Unknown")

            if pd.isna(player_id):
                print(f"  Warning: No ID found for {player_name}")
                actual_points.append(0)
                continue

            try:
                actual = self.get_actual_points_for_gameweek(
                    int(player_id), prev_gw
                )
                actual_points.append(actual)
                print(f"  {player_name}: {actual} points")
            except Exception as e:
                print(f"  Error fetching points for {player_name}: {e}")
                actual_points.append(0)
        
        return actual_points
    
    def _calculate_point_differences(self, prev_squad: pd.DataFrame, 
                                   total_actual: int) -> pd.DataFrame:
        """Calculate differences between projected and actual points."""
        if "projected_points" in prev_squad.columns:
            prev_squad["points_difference"] = (
                prev_squad["actual_points"] - prev_squad["projected_points"]
            )
            prev_squad["absolute_difference"] = abs(
                prev_squad["points_difference"]
            )

            total_projected = prev_squad["projected_points"].sum()
            difference = total_actual - total_projected
            print(f"Projected (from last week): {total_projected:.1f}, "
                  f"Total Points: {total_actual}, "
                  f"Difference: {difference:+.1f}")
        else:
            print("Warning: No projected_points column found in previous "
                  "squad")
            prev_squad["projected_points"] = 0  # Add dummy column
            prev_squad["points_difference"] = prev_squad["actual_points"]
            prev_squad["absolute_difference"] = abs(
                prev_squad["actual_points"])
        
        return prev_squad
    
    def _create_results_files(self, prev_squad: pd.DataFrame, prev_gw: int,
                            prev_squad_dir: str):
        """Create full squad results CSV file."""
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

        # Round projected_points and related columns to 1 decimal place
        decimal_columns = [
            "projected_points", "points_difference", "absolute_difference"
        ]
        for col in decimal_columns:
            if col in results_df.columns:
                results_df[col] = results_df[col].round(1)

        results_file = f"{prev_squad_dir}/full_squad_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f"✅ Results saved to {results_file}")
    
    def create_summary_analysis(self, squad_df: pd.DataFrame, gameweek: int, 
                              output_dir: str):
        """
        Create a summary analysis comparing projected vs actual points.
        Enhanced to focus only on Starting XI players.

        Args:
            squad_df (pd.DataFrame): Squad dataframe with projected and 
                                   actual points
            gameweek (int): Gameweek number
            output_dir (str): Directory to save the summary
        """
        try:
            # Focus only on Starting XI players
            starting_xi = squad_df[
                squad_df.get("squad_role", "") == "Starting XI"
            ]

            if len(starting_xi) == 0:
                print("Warning: No Starting XI players found, using all "
                      "players")
                starting_xi = squad_df

            # Calculate starting XI totals and metrics
            summary_data = self._calculate_summary_metrics(
                starting_xi, gameweek
            )
            
            # Create and save summary
            summary_df = pd.DataFrame(summary_data)
            summary_file = f"{output_dir}/summary.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"✅ Starting XI summary saved to {summary_file}")

            # Print summary to console
            self._print_console_summary(summary_data, starting_xi, gameweek)

        except Exception as e:
            print(f"❌ Error creating summary analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_summary_metrics(self, starting_xi: pd.DataFrame, 
                                 gameweek: int) -> list:
        """Calculate comprehensive summary metrics for starting XI."""
        summary_data = []
        
        # Starting XI totals
        total_projected = (
            starting_xi["projected_points"].sum()
            if "projected_points" in starting_xi.columns else 0
        )
        total_actual = starting_xi["actual_points"].sum()
        total_difference = total_actual - total_projected

        summary_data.append({
            "metric": "Starting XI Total Points",
            "projected": round(total_projected),
            "actual": total_actual,
            "difference": round(total_difference),
            "notes": (f"Total points from {len(starting_xi)} starting players "
                     f"in GW{gameweek}"),
        })

        # Position breakdown
        position_summary = self._calculate_position_breakdown(starting_xi)
        summary_data.extend(position_summary)

        # Accuracy metrics
        accuracy_metrics = self._calculate_accuracy_metrics(starting_xi)
        summary_data.extend(accuracy_metrics)

        # Best/worst performers
        performance_metrics = self._calculate_performance_metrics(starting_xi)
        summary_data.extend(performance_metrics)

        # Captain analysis
        captain_metrics = self._calculate_captain_metrics(starting_xi)
        summary_data.extend(captain_metrics)

        return summary_data
    
    def _calculate_position_breakdown(self, starting_xi: pd.DataFrame) -> list:
        """Calculate position-by-position breakdown."""
        position_summary = []
        
        for position in ["GK", "DEF", "MID", "FWD"]:
            pos_players = starting_xi[starting_xi["position"] == position]
            if len(pos_players) > 0:
                pos_projected = (
                    pos_players["projected_points"].sum()
                    if "projected_points" in pos_players.columns else 0
                )
                pos_actual = pos_players["actual_points"].sum()
                pos_difference = pos_actual - pos_projected
                pos_count = len(pos_players)

                position_summary.append({
                    "metric": f'{position} Starting XI',
                    "projected": round(pos_projected),
                    "actual": pos_actual,
                    "difference": round(pos_difference),
                    "notes": (f'{pos_count} players, '
                             f'avg actual: {pos_actual/pos_count:.1f}'),
                })
        
        return position_summary
    
    def _calculate_accuracy_metrics(self, starting_xi: pd.DataFrame) -> list:
        """Calculate prediction accuracy metrics."""
        accuracy_metrics = []
        
        if "absolute_difference" in starting_xi.columns:
            mean_absolute_error = starting_xi["absolute_difference"].mean()
            players_within_2pts = len(
                starting_xi[starting_xi["absolute_difference"] <= 2]
            )
            accuracy_within_2pts = (
                (players_within_2pts / len(starting_xi)) * 100 
                if len(starting_xi) > 0 else 0
            )
        else:
            mean_absolute_error = 0
            accuracy_within_2pts = 0
            players_within_2pts = 0

        accuracy_metrics.extend([
            {
                "metric": "Mean Absolute Error (XI)",
                "projected": "-",
                "actual": f"{mean_absolute_error:.2f}",
                "difference": "-",
                "notes": "Average abs. difference for Starting XI projections",
            },
            {
                "metric": "Accuracy within 2pts (XI)",
                "projected": "-",
                "actual": f"{accuracy_within_2pts:.1f}%",
                "difference": "-",
                "notes": (f"{players_within_2pts}/{len(starting_xi)} "
                         f"Starting XI players within 2 points"),
            }
        ])
        
        return accuracy_metrics
    
    def _calculate_performance_metrics(
            self, starting_xi: pd.DataFrame) -> list:
        """Calculate best and worst performer metrics."""
        performance_metrics = []
        
        if ("points_difference" in starting_xi.columns and 
            len(starting_xi) > 0):
            best_performer = starting_xi.loc[
                starting_xi["points_difference"].idxmax()
            ]
            worst_performer = starting_xi.loc[
                starting_xi["points_difference"].idxmin()
            ]
            
            performance_metrics.extend([
                {
                    "metric": "Best Starting XI Performer",
                    "projected": round(best_performer.get(
                        "projected_points", 0)),
                    "actual": best_performer["actual_points"],
                    "difference": round(best_performer.get(
                        "points_difference", 0)),
                    "notes": (f'{best_performer["display_name"]} '
                             f'({best_performer["position"]})'),
                },
                {
                    "metric": "Worst Starting XI Performer",
                    "projected": round(worst_performer.get(
                        "projected_points", 0)),
                    "actual": worst_performer["actual_points"],
                    "difference": round(worst_performer.get(
                        "points_difference", 0)),
                    "notes": (f'{worst_performer["display_name"]} '
                             f'({worst_performer["position"]})'),
                }
            ])
        
        return performance_metrics
    
    def _calculate_captain_metrics(self, starting_xi: pd.DataFrame) -> list:
        """Calculate captain performance metrics."""
        captain_metrics = []
        
        # Captain analysis (if captain data is available)
        captain_players = starting_xi[
            starting_xi["display_name"].str.contains(r"\(C\)", na=False)
        ]
        
        if len(captain_players) > 0:
            captain = captain_players.iloc[0]
            captain_projected = captain.get("projected_points", 0)
            captain_actual = captain["actual_points"]
            # Account for captain double points in actual score
            captain_actual_doubled = captain_actual * 2
            captain_difference = (
                captain_actual_doubled - (captain_projected * 2)
            )
            
            captain_metrics.append({
                "metric": "Captain Performance",
                "projected": round(captain_projected * 2),
                "actual": captain_actual_doubled,
                "difference": round(captain_difference),
                "notes": (f'{captain["display_name"].replace(" (C)", "")} - '
                         f'doubled points included'),
            })
        
        return captain_metrics
    
    def _print_console_summary(self, summary_data: list, 
                             starting_xi: pd.DataFrame, gameweek: int):
        """Print summary information to console."""
        # Find key metrics from summary data
        total_metric = next(
            (item for item in summary_data 
             if item["metric"] == "Starting XI Total Points"), None
        )
        mae_metric = next(
            (item for item in summary_data 
             if item["metric"] == "Mean Absolute Error (XI)"), None
        )
        accuracy_metric = next(
            (item for item in summary_data 
             if item["metric"] == "Accuracy within 2pts (XI)"), None
        )
        best_metric = next(
            (item for item in summary_data 
             if item["metric"] == "Best Starting XI Performer"), None
        )
        worst_metric = next(
            (item for item in summary_data 
             if item["metric"] == "Worst Starting XI Performer"), None
        )

        print(f"\n=== GW{gameweek} Starting XI Performance Summary ===")
        
        if total_metric:
            print(f"Starting XI Points: {total_metric['actual']} "
                  f"(projected: {total_metric['projected']:.1f}, "
                  f"difference: {total_metric['difference']:+.1f})")
        
        if mae_metric:
            print(f"Mean Absolute Error: {mae_metric['actual']}")
        
        if accuracy_metric:
            print(f"Accuracy (within 2pts): {accuracy_metric['actual']}")

        if best_metric:
            best_name = best_metric['notes'].split(' (')[0]
            print(f"Best XI Performer: {best_name} "
                  f"({best_metric['difference']:+.1f})")
        
        if worst_metric:
            worst_name = worst_metric['notes'].split(' (')[0]
            print(f"Worst XI Performer: {worst_name} "
                  f"({worst_metric['difference']:+.1f})")

        # Position breakdown in console
        print(f"\nPosition Breakdown (Starting XI):")
        for item in summary_data:
            if (
                " Starting XI" in item["metric"]
                and item["metric"] != "Starting XI Total Points"
                ):
                pos = item["metric"].split(" ")[0]
                print(f"  {pos}: {item['actual']} pts "
                      f"(projected: {item['projected']:.1f}, "
                      f"diff: {item['difference']:+.1f})")