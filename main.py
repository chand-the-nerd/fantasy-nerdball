#!/usr/bin/env python3
"""
Fantasy Nerdball - FPL Squad Optimisation Tool
Main entry point for the application.
"""

import os
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


def main():
    """Main function to run the FPL optimisation process."""
    config = Config()
    
    print(f"\n=== WELCOME TO FANTASY NERDBALL ===")
    print(f"\nPlanning for Gameweek {config.GAMEWEEK}")
    if config.WILDCARD:
        print("🃏 WILDCARD ACTIVE - No transfer limits!")
    else:
        print(f"Free transfers available: {config.FREE_TRANSFERS}")

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

    # Check for unavailable players and analyse substitute vs transfer options
    # (Now we can do this after scoring since we need the fpl_score column)
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
    
    print("Optimising squad using PuLP...")

    # First, get the best squad with transfers
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

    if starting_with_transfers.empty:
        print("No valid solution found!")
        return

    # Calculate transfers that would be made
    transfers_made = 0
    if prev_squad_ids is not None:
        current_squad_ids = set(
            pd.concat([starting_with_transfers, bench_with_transfers])["id"].tolist()
        )
        prev_squad_ids_set = set(prev_squad_ids)
        transfers_made = len(prev_squad_ids_set - current_squad_ids)

    # Evaluate transfer value
    should_make_transfers, transfer_analysis = transfer_evaluator.evaluate_transfer_strategy(
        scored, prev_squad_ids, starting_with_transfers, transfers_made, config.FREE_TRANSFERS, config.WILDCARD
    )

    # Choose final squad based on transfer analysis
    if should_make_transfers:
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
    _finalise_and_display_results(
        config, starting, bench, fixture_manager, points_calculator, scored, 
        players, prev_squad_ids, available_budget, squad_selector
    )


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
    starting, bench, _ = squad_selector.select_squad_ilp(
        scored_next,
        updated_forced_selections,
        prev_squad_ids,
        config.FREE_TRANSFERS,
        show_transfer_summary=False,
        available_budget=available_budget,
    )

    # Add fixture and points information
    starting = fixture_manager.add_next_fixture(starting, config.GAMEWEEK)
    bench = fixture_manager.add_next_fixture(bench, config.GAMEWEEK)

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
    if not starting_display.empty:
        top_two_idx = starting_display["proj_pts"].nlargest(2).index
        if len(top_two_idx) > 0:
            starting_display.loc[top_two_idx[0], "display_name"] += " (C)"
        if len(top_two_idx) > 1:
            starting_display.loc[top_two_idx[1], "display_name"] += " (V)"

    # Display final results
    print(f"\n=== Starting XI for GW{config.GAMEWEEK} ===")
    print(
        starting_display[
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

    print(f"\n=== Bench (in order) for GW{config.GAMEWEEK} ===")
    print(
        bench_display[
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

    total_cost = starting["now_cost_m"].sum() + bench["now_cost_m"].sum()
    total_projected_points = starting_display["projected_points"].sum()
    print(f"\nTotal Squad Cost: {total_cost:.1f}m")
    print(f"Projected Starting XI Points: {total_projected_points:.1f}")

    # Save squad data
    FileUtils.save_squad_data(config.GAMEWEEK, starting_display, bench_display)


if __name__ == "__main__":
    main()