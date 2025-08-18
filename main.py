#!/usr/bin/env python3
"""
Fantasy Nerdball - FPL Squad Optimisation Tool
Main entry point for the application.
"""

import sys
import subprocess
import pandas as pd
from config import Config
from src.api.fpl_client import FPLClient
from src.data.player_processor import PlayerProcessor
from src.data.fixture_manager import FixtureManager
from src.data.historical_data import HistoricalDataManager
from src.analysis.scoring_engine import ScoringEngine
from src.analysis.points_calculator import PointsCalculator
from src.analysis.results_analyser import ResultsAnalyser
from src.optimisation.squad_selector import SquadSelector
from src.optimisation.transfer_evaluator import TransferEvaluator
from src.utils.file_utils import FileUtils


def prompt_player_history_update(config):
    """Prompt user to update player history data."""
    if config.GAMEWEEK <= 1:
        # No previous gameweek data to update for GW1
        return
        
    print(f"\n=== Player History Update ===")
    print(f"Would you like to update player history data for GW{config.GAMEWEEK - 1}?")
    print("This captures the previous gameweek's performance data.")
    
    while True:
        response = input("Update player history? (y/n/s for stats): ").lower().strip()
        
        if response in ['y', 'yes']:
            print("Updating player history data...")
            try:
                # Run the update_player_history.py script with 'update' command
                result = subprocess.run([
                    sys.executable, 'update_player_history.py', 'update'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Player history updated successfully!")
                    if result.stdout:
                        print(result.stdout)
                else:
                    print("❌ Error updating player history:")
                    if result.stderr:
                        print(result.stderr)
                    # Continue anyway - don't stop the main script
                    
            except Exception as e:
                print(f"❌ Failed to run player history update: {e}")
                print("Continuing with main analysis...")
            break
            
        elif response in ['n', 'no']:
            print("Skipping player history update.")
            break
            
        elif response in ['s', 'stats']:
            print("Showing player history statistics...")
            try:
                result = subprocess.run([
                    sys.executable, 'update_player_history.py', 'stats'
                ], capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout:
                    print(result.stdout)
                else:
                    print("No statistics available or error occurred.")
                    
            except Exception as e:
                print(f"Error getting statistics: {e}")
            # Ask again after showing stats
            continue
            
        else:
            print("Please enter 'y' for yes, 'n' for no, or 's' for stats.")


def main():
    """Main function to run the FPL optimisation process."""
    config = Config()
    
    print(f"\n=== WELCOME TO FANTASY NERDBALL ===")
    print(f"\nPlanning for Gameweek {config.GAMEWEEK}")
    if config.WILDCARD:
        print("🃏 WILDCARD ACTIVE - No transfer limits!")
    elif config.ACCEPT_TRANSFER_PENALTY:
        print(f"💰 TRANSFER PENALTY MODE - Can exceed {config.FREE_TRANSFERS} free transfers (4 pts penalty each)")
    else:
        print(f"Free transfers available: {config.FREE_TRANSFERS}")

    # Prompt for player history update before starting main analysis
    prompt_player_history_update(config)

    # Initialise components
    fpl_client = FPLClient()
    player_processor = PlayerProcessor(config)
    fixture_manager = FixtureManager(config)
    historical_manager = HistoricalDataManager(config)
    scoring_engine = ScoringEngine(config)
    points_calculator = PointsCalculator(config)
    results_analyser = ResultsAnalyser(config)
    squad_selector = SquadSelector(config)
    transfer_evaluator = TransferEvaluator(config)

    # Create results for previous gameweek if this is GW2+
    if config.GAMEWEEK >= 2:
        results_analyser.create_previous_gameweek_results(config.GAMEWEEK)

    # Load previous squad if available
    prev_squad = FileUtils.load_previous_squad(config.GAMEWEEK)

    print("Fetching current players...")
    players = player_processor.fetch_current_players()

    # Calculate available budget based on previous squad value
    available_budget = player_processor.calculate_budget_from_previous_squad(
        config.GAMEWEEK, players
    )

    # Match previous squad to current players if we have a previous squad
    prev_squad_ids = None
    if prev_squad is not None:
        prev_squad_ids = player_processor.match_players_to_current(prev_squad, players)

    # Merge historical data
    players = historical_manager.merge_past_seasons(players)
    
    print(f"Fetching player-level fixture difficulty from GW{config.GAMEWEEK}...")
    fixture_scores = fixture_manager.fetch_player_fixture_difficulty(
        config.FIRST_N_GAMEWEEKS, players, config.GAMEWEEK
    )
    
    print("Scoring players...")
    scored = scoring_engine.build_scores(players, fixture_scores)

    # === NEW STAGE: Show theoretical best squad for comparison ===
    print(f"\n=== 🏆 NERDBALL PICKS GW{config.GAMEWEEK} ===")
    print("Limitless pick for the week (within budget)")
    
    # Get single gameweek fixture scores for comparison
    fixture_scores_comparison = fixture_manager.fetch_player_fixture_difficulty(1, players, config.GAMEWEEK)
    scoring_engine_comparison = ScoringEngine(config)
    scored_comparison = scoring_engine_comparison.build_scores(players, fixture_scores_comparison)
    
    # Get theoretical best squad (no transfer constraints, no forced selections)
    # Use the same budget as calculated for your actual optimization
    budget_for_comparison = available_budget if available_budget is not None else config.BUDGET
    print(f"Available budget for comparison: {budget_for_comparison:.1f}m")
    
    theoretical_starting, theoretical_bench, _ = squad_selector.select_squad_ilp(
        scored_comparison,
        {"GK": [], "DEF": [], "MID": [], "FWD": []},  # No forced selections
        prev_squad_ids=None,  # No transfer constraints
        free_transfers=None,
        show_transfer_summary=False,
        available_budget=budget_for_comparison,  # Use same budget as your optimization
    )
    
    if not theoretical_starting.empty:
        # Add fixture and points information
        theoretical_starting = fixture_manager.add_next_fixture(theoretical_starting, config.GAMEWEEK)
        theoretical_bench = fixture_manager.add_next_fixture(theoretical_bench, config.GAMEWEEK)
        
        # Add projected points analysis
        theoretical_starting_display = points_calculator.add_points_analysis_to_display(theoretical_starting)
        theoretical_bench_display = points_calculator.add_points_analysis_to_display(theoretical_bench)
        
        # Sort starting XI by position for display
        position_order = ["GK", "DEF", "MID", "FWD"]
        theoretical_starting_display["position"] = pd.Categorical(
            theoretical_starting_display["position"], categories=position_order, ordered=True
        )
        theoretical_starting_display = theoretical_starting_display.sort_values("position")
        
        # Order bench: GK first, then remaining 3 by descending projected points
        gk_bench = theoretical_bench_display[theoretical_bench_display["position"] == "GK"].copy()
        non_gk_bench = theoretical_bench_display[theoretical_bench_display["position"] != "GK"].copy()
        non_gk_bench = non_gk_bench.sort_values("projected_points", ascending=False)
        theoretical_bench_display = pd.concat([gk_bench, non_gk_bench], ignore_index=True)
        
        # Mark captain and vice-captain based on projected points
        theoretical_captain_multiplier_applied = False
        if not theoretical_starting_display.empty:
            top_two_idx = theoretical_starting_display["proj_pts"].nlargest(2).index
            if len(top_two_idx) > 0:
                captain_idx = top_two_idx[0]
                theoretical_starting_display.loc[captain_idx, "display_name"] += " (C)"
                # Create a display column for projected points with captain multiplier
                captain_original_points = theoretical_starting_display.loc[captain_idx, "proj_pts"]
                theoretical_starting_display.loc[captain_idx, "proj_pts_display"] = f"{captain_original_points:.1f} (x2)"
                # Update the actual projected_points column for total calculation
                theoretical_starting_display.loc[captain_idx, "projected_points"] = captain_original_points * 2
                theoretical_captain_multiplier_applied = True
                
            if len(top_two_idx) > 1:
                theoretical_starting_display.loc[top_two_idx[1], "display_name"] += " (V)"

        # Create display column for non-captain players
        if "proj_pts_display" not in theoretical_starting_display.columns:
            theoretical_starting_display["proj_pts_display"] = theoretical_starting_display["proj_pts"].round(1).astype(str)
        else:
            # Fill in non-captain players
            mask = theoretical_starting_display["proj_pts_display"].isna()
            theoretical_starting_display.loc[mask, "proj_pts_display"] = theoretical_starting_display.loc[mask, "proj_pts"].round(1).astype(str)

        # Also create display column for theoretical bench
        if "proj_pts_display" not in theoretical_bench_display.columns:
            theoretical_bench_display["proj_pts_display"] = theoretical_bench_display["proj_pts"].round(1).astype(str)
        
        # Display columns to match your format
        columns_to_show = [
            "display_name", "position", "team", "form", "historic_ppg", 
            "fixture_diff", "reliability", "minspg", "proj_pts_display", "next_opponent"  # Use display column
        ]
        
        print(f"\n=== NERDBALL XI for GW{config.GAMEWEEK} ===")
        print(theoretical_starting_display[columns_to_show])
        
        print(f"\n=== SUBS (in order) for GW{config.GAMEWEEK} ===")
        print(theoretical_bench_display[columns_to_show])
        
        theoretical_cost = theoretical_starting["now_cost_m"].sum() + theoretical_bench["now_cost_m"].sum()
        theoretical_points_with_captain = theoretical_starting_display["projected_points"].sum()
        print(f"\nNerdball Squad Cost: {theoretical_cost:.1f}m")
        print(f"Nerdball Starting XI Projected Points: {theoretical_points_with_captain:.1f}")
    else:
        print("❌ Could not generate theoretical best squad")
        theoretical_points_with_captain = 0
        theoretical_cost = 0

    # Check for unavailable players and analyse substitute vs transfer options
    if prev_squad_ids and not config.WILDCARD:
        unavailable_players = transfer_evaluator.get_unavailable_players(
            scored, prev_squad_ids
        )
        if unavailable_players:
            substitute_analysis = transfer_evaluator.evaluate_substitute_vs_transfer(
                scored, prev_squad_ids, unavailable_players, config.FREE_TRANSFERS
            )

            # Update strategy based on analysis
            if substitute_analysis["recommendation"] == "wildcard_needed":
                print(f"\n⚠️  RECOMMENDATION: Consider activating wildcard")
                print(f"   Too many unavailable players for available transfers")
            elif substitute_analysis["recommendation"] == "use_substitutes":
                print(f"\n💡 RECOMMENDATION: Use substitutes, save transfers")
                print(f"   Substitution score loss acceptable")
        else:
            print(f"\n✅ All previous squad players are available")
    
    print("\n🧠 Thinking...")

    # Get the best squad - either with penalty consideration or standard approach
    if config.ACCEPT_TRANSFER_PENALTY and prev_squad_ids is not None and not config.WILDCARD:
        # Use transfer penalty optimization
        (starting_with_transfers, bench_with_transfers, forced_selections_display, 
         transfers_made, penalty_points) = transfer_evaluator.get_optimal_squad_with_penalties(
            scored, config.FORCED_SELECTIONS, prev_squad_ids, 
            config.FREE_TRANSFERS, available_budget, squad_selector
        )
    else:
        # Use standard optimization
        (starting_with_transfers, bench_with_transfers, forced_selections_display) = (
            squad_selector.select_squad_ilp(
                scored,
                config.FORCED_SELECTIONS,
                prev_squad_ids,
                config.FREE_TRANSFERS,
                show_transfer_summary=True,
                available_budget=available_budget,
            )
        )
        
        # Calculate transfers that would be made
        transfers_made = 0
        penalty_points = 0
        if prev_squad_ids is not None:
            current_squad_ids = set(
                pd.concat([starting_with_transfers, bench_with_transfers])["id"].tolist()
            )
            prev_squad_ids_set = set(prev_squad_ids)
            transfers_made = len(prev_squad_ids_set - current_squad_ids)

    if starting_with_transfers.empty:
        print("No valid solution found!")
        return

    # Evaluate transfer value (skip if penalty mode already optimized)
    if config.ACCEPT_TRANSFER_PENALTY:
        should_make_transfers = True
        transfer_analysis = {"reason": "Transfer penalty mode - already optimized"}
    else:
        should_make_transfers, transfer_analysis = transfer_evaluator.evaluate_transfer_strategy(
            scored, prev_squad_ids, starting_with_transfers, transfers_made, 
            config.FREE_TRANSFERS, config.WILDCARD
        )

    # Choose final squad based on transfer analysis
    if should_make_transfers:
        if penalty_points > 0:
            print(f"\n✅ Making {transfers_made} transfer(s) with {penalty_points} penalty points")
        else:
            print(f"\n✅ Making {transfers_made} transfer(s)")
        starting = starting_with_transfers
        bench = bench_with_transfers
    else:
        print(f"\n❌ Transfers not worth it - keeping current squad")
        if prev_squad_ids is not None:
            starting, bench = transfer_evaluator.get_no_transfer_squad_optimised(scored, prev_squad_ids)
        else:
            starting = starting_with_transfers
            bench = bench_with_transfers

    # Finalise and display results
    starting_display, bench_display, captain_multiplier_applied = _finalise_and_display_results(
        config, starting, bench, fixture_manager, points_calculator, scored, 
        players, prev_squad_ids, available_budget, squad_selector
    )
    
    # === COMPARISON: Show how your squad compares to theoretical best ===
    if theoretical_points_with_captain > 0 and not starting.empty:
        # Calculate points with captain multiplier from the returned displays
        your_points_with_captain = starting_display["projected_points"].sum()
        
        your_cost = (starting["now_cost_m"].sum() + bench["now_cost_m"].sum()) if not bench.empty else starting["now_cost_m"].sum()
        
        if 'penalty_points' in locals() and penalty_points > 0:
            net_your_points = your_points_with_captain - penalty_points
            print(f"\n🎯 SQUAD COMPARISON (including captain multiplier)")
            print(f"📊 Nerdball Pick of the Week:     {theoretical_points_with_captain:.1f} pts  ({theoretical_cost:.1f}m)")
            print(f"🏈 Your Squad (gross):   {your_points_with_captain:.1f} pts  ({your_cost:.1f}m)")
            print(f"💰 Transfer Penalty:     -{penalty_points:.1f} pts")
            print(f"🎯 Your Squad (net):     {net_your_points:.1f} pts")
            print(f"📈 Gap to theoretical:   {theoretical_points_with_captain - net_your_points:.1f} pts ({((theoretical_points_with_captain - net_your_points) / theoretical_points_with_captain * 100):.1f}%)")
        else:
            print(f"\n🎯 SQUAD COMPARISON (including captain multiplier)")
            print(f"📊 Theoretical Best:   {theoretical_points_with_captain:.1f} pts  ({theoretical_cost:.1f}m)")
            print(f"🏈 Your Squad:         {your_points_with_captain:.1f} pts  ({your_cost:.1f}m)")
            print(f"📈 Gap to theoretical: {theoretical_points_with_captain - your_points_with_captain:.1f} pts ({((theoretical_points_with_captain - your_points_with_captain) / theoretical_points_with_captain * 100):.1f}%)")
            
        gap = theoretical_points_with_captain - (your_points_with_captain - (penalty_points if 'penalty_points' in locals() else 0))
        if gap < 2:
            print("🏆 Pretty sweet squad.")
        elif gap < 5:
            print("👍 Nice effort.")
        else:
            print("💡 Not bad, could be better.")


def _finalise_and_display_results(
    config, starting, bench, fixture_manager, points_calculator, scored, 
    players, prev_squad_ids, available_budget, squad_selector
):
    """Finalise squad selection and display results."""
    # Create updated forced selections with all squad players
    updated_forced_selections = squad_selector.update_forced_selections_from_squad(starting, bench)

    # Show squad table before next match optimisation
    full_squad = pd.concat([starting, bench], ignore_index=True)
    full_squad = fixture_manager.add_next_fixture(full_squad, config.GAMEWEEK)

    # Sort by position for display - apply to full squad only
    position_order = ["GK", "DEF", "MID", "FWD"]
    full_squad["position"] = pd.Categorical(
        full_squad["position"], categories=position_order, ordered=True
    )
    full_squad = full_squad.sort_values("position")

    # Print full squad with projected points
    full_squad_display = points_calculator.add_points_analysis_to_display(full_squad)
    print(f"\n=== Full Squad for GW{config.GAMEWEEK} ===")
    print(
        full_squad_display[
            [
                "display_name",
                "position",
                "team",
                "form",
                "historic_ppg",
                "fixture_diff",
                "reliability",
                "minspg",
                "proj_pts",
                "next_opponent",
            ]
        ]
    )

    # Optimise starting XI for the specific gameweek
    print(f"\nFetching difficulty for GW{config.GAMEWEEK} fixture...")
    fixture_scores_next = fixture_manager.fetch_player_fixture_difficulty(1, players, config.GAMEWEEK)
    
    scoring_engine_next = ScoringEngine(config)
    scored_next = scoring_engine_next.build_scores(players, fixture_scores_next)

    print(f"Optimising Starting XI for GW{config.GAMEWEEK}...")
    
    # Try to optimize starting XI with form constraint first
    # NOTE: Don't pass prev_squad_ids or free_transfers here since we're just selecting 
    # the starting XI from our already-chosen 15-player squad
    starting_optimized, bench_optimized, _ = squad_selector.select_squad_ilp(
        scored_next,
        updated_forced_selections,
        prev_squad_ids=None,  # No transfer constraints for starting XI selection
        free_transfers=None,  # No transfer constraints for starting XI selection
        show_transfer_summary=False,
        available_budget=available_budget,
    )

    # If optimization failed (infeasible), fall back to simple selection without form constraint
    if starting_optimized.empty:
        print("⚠️  Starting XI optimization infeasible with form constraint. Using fallback selection...")
        
        # Get the squad from our forced selections and pick best 11 without form constraint
        squad_players = scored_next[scored_next["display_name"].isin(
            full_squad["display_name"]
        )].copy()
        
        if len(squad_players) >= 11:
            # Simple selection: pick highest projected points respecting position constraints
            starting_optimized = _select_starting_xi_fallback(squad_players)
            bench_optimized = squad_players[~squad_players["id"].isin(starting_optimized["id"])].copy()
            
            # Order bench properly
            gk_bench = bench_optimized[bench_optimized["position"] == "GK"].copy()
            non_gk_bench = bench_optimized[bench_optimized["position"] != "GK"].copy()
            non_gk_bench = non_gk_bench.sort_values("projected_points", ascending=False)
            bench_optimized = pd.concat([gk_bench, non_gk_bench], ignore_index=True)
        else:
            print("❌ Unable to create starting XI. Using original squad selection.")
            starting_optimized = starting
            bench_optimized = bench

    # Use optimized selection if successful, otherwise use original
    if not starting_optimized.empty:
        starting = starting_optimized
        bench = bench_optimized

    # Add fixture and points information - with error handling
    try:
        starting = fixture_manager.add_next_fixture(starting, config.GAMEWEEK)
        bench = fixture_manager.add_next_fixture(bench, config.GAMEWEEK)
    except Exception as e:
        print(f"⚠️  Warning: Could not add fixture information - {e}")
        # Continue without fixture info if there's an error

    # Add projected points to starting XI and bench
    starting_display = points_calculator.add_points_analysis_to_display(starting)
    bench_display = points_calculator.add_points_analysis_to_display(bench)

    # Sort starting XI by position for display
    starting_display["position"] = pd.Categorical(
        starting_display["position"], categories=position_order, ordered=True
    )
    starting_display = starting_display.sort_values("position")

    # DO NOT sort bench - preserve the order from squad_selector
    # bench_display keeps its original order (GK first, then by descending projected points)

    # Mark captain and vice-captain based on projected points
    captain_multiplier_applied = False
    if not starting_display.empty:
        top_two_idx = starting_display["proj_pts"].nlargest(2).index
        if len(top_two_idx) > 0:
            captain_idx = top_two_idx[0]
            starting_display.loc[captain_idx, "display_name"] += " (C)"
            # Create a display column for projected points with captain multiplier
            captain_original_points = starting_display.loc[captain_idx, "proj_pts"]
            starting_display.loc[captain_idx, "proj_pts_display"] = f"{captain_original_points:.1f} (x2)"
            # Update the actual projected_points column for total calculation
            starting_display.loc[captain_idx, "projected_points"] = captain_original_points * 2
            captain_multiplier_applied = True
            
        if len(top_two_idx) > 1:
            starting_display.loc[top_two_idx[1], "display_name"] += " (V)"

    # Create display column for non-captain players
    if "proj_pts_display" not in starting_display.columns:
        starting_display["proj_pts_display"] = starting_display["proj_pts"].round(1).astype(str)
    else:
        # Fill in non-captain players
        mask = starting_display["proj_pts_display"].isna()
        starting_display.loc[mask, "proj_pts_display"] = starting_display.loc[mask, "proj_pts"].round(1).astype(str)

    # Display final results
    print(f"\n=== Starting XI for GW{config.GAMEWEEK} ===")
    
    # Check if we have the next_opponent column before displaying
    columns_to_display = [
        "display_name",
        "position",
        "team",
        "form",
        "historic_ppg",
        "fixture_diff",
        "reliability",
        "minspg",
        "proj_pts_display",  # Use display column instead of proj_pts
    ]
    
    if "next_opponent" in starting_display.columns:
        columns_to_display.append("next_opponent")
    
    print(starting_display[columns_to_display])

    print(f"\n=== Bench (in order) for GW{config.GAMEWEEK} ===")
    
    # Also create display column for bench
    if "proj_pts_display" not in bench_display.columns:
        bench_display["proj_pts_display"] = bench_display["proj_pts"].round(1).astype(str)
    
    print(bench_display[columns_to_display])

    total_cost = starting["now_cost_m"].sum() + bench["now_cost_m"].sum()
    total_projected_points = starting_display["projected_points"].sum()
    print(f"\nTotal Squad Cost: {total_cost:.1f}m")
    print(f"Projected Starting XI Points: {total_projected_points:.1f}")

    # Save squad data
    FileUtils.save_squad_data(config.GAMEWEEK, starting_display, bench_display)
    
    # Return the displays and captain status for comparison
    return starting_display, bench_display, captain_multiplier_applied


def _select_starting_xi_fallback(squad_players: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback function to select starting XI when optimization fails.
    Uses simple greedy selection based on projected points.
    """
    starting_xi = []
    remaining_players = squad_players.copy()
    
    # Required positions for starting XI
    required_positions = {
        "GK": 1,
        "DEF": 3,  # minimum
        "MID": 3,  # minimum  
        "FWD": 1   # minimum
    }
    
    # First, ensure minimum requirements
    for pos, min_count in required_positions.items():
        pos_players = remaining_players[remaining_players["position"] == pos].copy()
        pos_players = pos_players.sort_values("projected_points", ascending=False)
        
        selected_count = min(min_count, len(pos_players))
        for i in range(selected_count):
            starting_xi.append(pos_players.iloc[i])
            remaining_players = remaining_players[remaining_players["id"] != pos_players.iloc[i]["id"]]
    
    # Fill remaining spots (up to 11 total) with highest projected points
    # Respect maximum constraints: max 5 DEF, max 5 MID, max 3 FWD
    max_positions = {"DEF": 5, "MID": 5, "FWD": 3, "GK": 1}
    current_counts = {"GK": 1, "DEF": 3, "MID": 3, "FWD": 1}
    
    remaining_players = remaining_players.sort_values("projected_points", ascending=False)
    
    for _, player in remaining_players.iterrows():
        if len(starting_xi) >= 11:
            break
            
        pos = player["position"]
        if current_counts.get(pos, 0) < max_positions.get(pos, 0):
            starting_xi.append(player)
            current_counts[pos] = current_counts.get(pos, 0) + 1
    
    return pd.DataFrame(starting_xi)

if __name__ == "__main__":
    main()