#!/usr/bin/env python3
"""
Fantasy Nerdball - FPL Squad Optimisation Tool
Main entry point for the application with enhanced xG analysis.
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
from src.utils.squad_display_utils import SquadDisplayUtils
from src.utils.token_manager import TokenManager

RENAME_MAP = {
    "display_name": "name",
    "position": "pos",
    "team": "team",
    "now_cost_m": "cost",
    "form": "form",
    "historic_ppg": "his_ppg",
    "fixture_diff": "fix_diff",
    "reliability": "start_pct",
    "historical_xOP": "hist_xOP",
    "current_xOP": "cur_xOP",
    "xConsistency": "xMod",
    "minspg": "minspg",
    "proj_pts": "proj_pts",
    "next_opponent": "next_fix"
}


def prompt_player_history_update(config):
    """Prompt user to update player history data."""
    if config.GAMEWEEK <= 1:
        return
        
    print(f"\n=== Player History Update ===")
    print(f"Would you like to update player history data for "
          f"GW{config.GAMEWEEK - 1}?")
    print("This captures the previous gameweek's performance data.")
    
    while True:
        response = input(
            "Update player history? (y/n/s for stats): "
            ).lower().strip()
        
        if response in ['y', 'yes']:
            print("Updating player history data...")
            try:
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
            continue
            
        else:
            print("Please enter 'y' for yes, 'n' for no, or 's' for stats.")


def initialise_components(config, token_manager):
    """Initialise all required components."""
    return {
        'fpl_client': FPLClient(),
        'player_processor': PlayerProcessor(config),
        'fixture_manager': FixtureManager(config),
        'historical_manager': HistoricalDataManager(config),
        'scoring_engine': ScoringEngine(config),
        'points_calculator': PointsCalculator(config),
        'results_analyser': ResultsAnalyser(config),
        'squad_selector': SquadSelector(config),
        'transfer_evaluator': TransferEvaluator(config),
        'display_utils': SquadDisplayUtils(config),
        'token_manager': token_manager
    }


def print_header(config):
    """Print application header with current settings."""
    print(f"\n=== WELCOME TO FANTASY NERDBALL ===")
    print(f"\nPlanning for Gameweek {config.GAMEWEEK}")
    
    if config.WILDCARD:
        print("🃏 WILDCARD ACTIVE - No transfer limits!")
    elif config.ACCEPT_TRANSFER_PENALTY:
        print(f"💰 TRANSFER PENALTY MODE - Can exceed "
              f"{config.FREE_TRANSFERS} free transfers (4 pts penalty each)")
    else:
        print(f"Free transfers available: {config.FREE_TRANSFERS}")


def _display_availability_and_dgw_stats(involvement_stats, dgw_stats, 
                                       gameweek):
    """Display player availability and DGW/BGW statistics."""
    if not involvement_stats['exclude_unavailable']:
        print("📋 EXCLUDE_UNAVAILABLE = False: Including unavailable "
              "players in optimisation")
        return
    
    print("\nChecking availability...")
    if gameweek > 1:
        if involvement_stats['high_involvement'] > 0:
            print(f"✅ {involvement_stats['high_involvement']} players with "
                  "high involvement (reliable starters)")
            
        if involvement_stats['low_involvement'] > 0:
            print(f"⚠️  {involvement_stats['low_involvement']} players with "
                  "low involvement heavily penalised")
            
        if involvement_stats['zero_involvement'] > 0:
            print(f"🛑 {involvement_stats['zero_involvement']} players with "
                  "zero involvement will have scores set to 0")
    
    if involvement_stats['unavailable'] > 0:
        print(f"🛑 {involvement_stats['unavailable']} players unavailable "
              "due to injury/suspension")
    else:
        print("✅ All players are available for selection")
    
    print("\nChecking double/blank gameweeks...")
    print(f"▶️  {dgw_stats['regular_gw']} players with a regular gameweek")
    
    if dgw_stats['double_gw'] > 0:
        print(f"🔥 {dgw_stats['double_gw']} players with a double gameweek")
    
    if dgw_stats['blank_gw'] > 0:
        print(f"💔 {dgw_stats['blank_gw']} players with a blank gameweek")
    
    if dgw_stats['double_gw'] == 0 and dgw_stats['blank_gw'] == 0:
        print("▶️  All players have regular gameweeks")


def process_player_data(components, config):
    """Process player data with historical and fixture analysis."""
    print("Fetching current players with xG analysis...")
    players = components['player_processor'].fetch_current_players()

    # Calculate available budget based on previous squad value
    available_budget = (
        components['player_processor']
        .calculate_budget_from_previous_squad(config.GAMEWEEK, players)
    )

    print("Analysing historical xG performance...")
    players = components['historical_manager'].merge_past_seasons(players)
    
    print(f"Fetching player-level fixture difficulty from "
          f"GW{config.GAMEWEEK} with DGW/BGW handling...")
    fixture_scores = (
        components['fixture_manager']
        .fetch_player_fixture_difficulty(
            config.FIRST_N_GAMEWEEKS, players, config.GAMEWEEK
        )
    )
    
    print("Scoring players with xG performance modifiers...")
    scored, involvement_stats = components['scoring_engine'].build_scores(
        players, fixture_scores
    )
    
    # Get DGW/BGW statistics
    dgw_stats = _calculate_dgw_stats(scored, config.GAMEWEEK)
    
    # Display availability and DGW statistics (only once, here)
    _display_availability_and_dgw_stats(involvement_stats, dgw_stats, 
                                       config.GAMEWEEK)
    
    return players, scored, available_budget


def _calculate_dgw_stats(players_df, gameweek):
    """Calculate double/blank gameweek statistics."""
    # For now, since your code doesn't seem to have DGW/BGW logic yet,
    # we'll return stats showing all players have regular gameweeks
    # You can enhance this when you add DGW/BGW functionality
    
    dgw_stats = {
        'regular_gw': len(players_df),
        'double_gw': 0,
        'blank_gw': 0
    }
    
    return dgw_stats


def generate_theoretical_squad(components, config, players, available_budget):
    """Generate theoretical best squad for comparison."""
    print(f"\n=== 🏆 NERDBALL PICKS GW{config.GAMEWEEK} ===")
    print("Limitless pick for the week (within budget)")
    
    # Get single gameweek fixture scores for comparison
    fixture_scores_comparison = (
        components['fixture_manager']
        .fetch_player_fixture_difficulty(1, players, config.GAMEWEEK)
    )
    
    scoring_engine_comparison = ScoringEngine(config)
    scored_comparison = scoring_engine_comparison.build_scores(
        players, fixture_scores_comparison
    )[0]  # Take only the dataframe, ignore stats
    
    budget_for_comparison = (
        available_budget if available_budget is not None else config.BUDGET
    )
    print(f"Available budget for comparison: {budget_for_comparison:.1f}m")
    
    # No forced selections or transfer constraints for theoretical squad
    theoretical_starting, theoretical_bench, _ = (
        components['squad_selector'].select_squad_ilp(
            scored_comparison,
            {"GK": [], "DEF": [], "MID": [], "FWD": []},
            prev_squad_ids=None,
            free_transfers=None,
            show_transfer_summary=False,
            available_budget=budget_for_comparison,
        )
    )
    
    if theoretical_starting.empty:
        print("❌ Could not generate theoretical best squad")
        return None, 0, 0
    
    # Process theoretical squad for display
    theoretical_starting = (
        components['fixture_manager']
        .add_next_fixture(theoretical_starting, config.GAMEWEEK)
    )
    theoretical_bench = (
        components['fixture_manager']
        .add_next_fixture(theoretical_bench, config.GAMEWEEK)
    )
    
    # Apply display formatting and captain selection
    theoretical_starting_display = (
        components['points_calculator']
        .add_points_analysis_to_display(theoretical_starting)
    )
    theoretical_bench_display = (
        components['points_calculator']
        .add_points_analysis_to_display(theoretical_bench)
    )
    
    # Use display utils for consistent formatting
    theoretical_starting_display = (
        components['display_utils']
        .sort_and_format_starting_xi(theoretical_starting_display)
    )
    theoretical_bench_display = (
        components['display_utils']
        .sort_and_format_bench(theoretical_bench_display)
    )
    
    # Apply captain and vice-captain
    theoretical_starting_display = (
        components['display_utils']
        .apply_captain_and_vice(theoretical_starting_display)
    )
    
    # Display theoretical squad
    print(f"\n=== NERDBALL XI for GW{config.GAMEWEEK} ===")
    components['display_utils'].print_squad_table(
        theoretical_starting_display, RENAME_MAP
    )
    
    print(f"\n=== SUBS (in order) for GW{config.GAMEWEEK} ===")
    components['display_utils'].print_squad_table(
        theoretical_bench_display, RENAME_MAP
    )
    
    theoretical_cost = (
        theoretical_starting["now_cost_m"].sum() + 
        theoretical_bench["now_cost_m"].sum()
    )
    theoretical_points = theoretical_starting_display["projected_points"].sum()
    
    print(f"\nNerdball Squad Cost: {theoretical_cost:.1f}m")
    print(f"Nerdball Starting XI Projected Points: {theoretical_points:.1f}")
    
    return theoretical_starting_display, theoretical_points, theoretical_cost


def analyse_unavailable_players(components, config, scored, prev_squad_ids):
    """Analyse unavailable players and substitution options."""
    if not prev_squad_ids or config.WILDCARD:
        return
    
    unavailable_players = (
        components['transfer_evaluator']
        .get_unavailable_players(scored, prev_squad_ids)
    )
    
    if unavailable_players:
        substitute_analysis = (
            components['transfer_evaluator']
            .evaluate_substitute_vs_transfer(
                scored, prev_squad_ids, unavailable_players, 
                config.FREE_TRANSFERS
            )
        )

        if substitute_analysis["recommendation"] == "wildcard_needed":
            print(f"\n⚠️  RECOMMENDATION: Consider activating wildcard")
            print(f"   Too many unavailable players for available transfers")
        elif substitute_analysis["recommendation"] == "use_substitutes":
            print(f"\n💡 RECOMMENDATION: Use substitutes, save transfers")
            print(f"   Substitution score loss acceptable")
    else:
        print(f"\n✅ All previous squad players are available")


def optimise_squad(
        components,
        config, scored,
        prev_squad_ids,
        available_budget
        ):
    """Optimise squad selection with transfer considerations."""
    print("\n🧠 Thinking...")

    penalty_mode = (
        config.ACCEPT_TRANSFER_PENALTY and 
        prev_squad_ids is not None and 
        not config.WILDCARD
    )
    
    if penalty_mode:
        # Use transfer penalty optimisation
        result = components[
            'transfer_evaluator'
            ].get_optimal_squad_with_penalties(
            scored,
            config.FORCED_SELECTIONS,
            prev_squad_ids, 
            config.FREE_TRANSFERS,
            available_budget, 
            components['squad_selector']
        )
        (starting_with_transfers,
        bench_with_transfers,
        _,
        transfers_made,
        penalty_points) = result
    else:
        # Use standard optimisation
        starting_with_transfers, bench_with_transfers, _ = (
            components['squad_selector'].select_squad_ilp(
                scored, config.FORCED_SELECTIONS, prev_squad_ids,
                config.FREE_TRANSFERS, show_transfer_summary=True,
                available_budget=available_budget,
            )
        )
        
        # Calculate transfers that would be made
        transfers_made = 0
        penalty_points = 0
        if prev_squad_ids is not None:
            current_squad_ids = set(
                pd.concat(
                [starting_with_transfers, bench_with_transfers]
                )["id"]
            )
            prev_squad_ids_set = set(prev_squad_ids)
            transfers_made = len(prev_squad_ids_set - current_squad_ids)

    if starting_with_transfers.empty:
        raise ValueError("No valid solution found!")

    return (starting_with_transfers, bench_with_transfers, 
            transfers_made, penalty_points)


def evaluate_transfer_strategy(components, config, scored, prev_squad_ids, 
                             starting_with_transfers, transfers_made):
    """Evaluate whether transfers should be made."""
    if config.ACCEPT_TRANSFER_PENALTY:
        return True, {"reason": "Transfer penalty mode - already optimised"}
    
    should_make_transfers, transfer_analysis = (
        components['transfer_evaluator'].evaluate_transfer_strategy(
            scored, prev_squad_ids, starting_with_transfers, transfers_made, 
            config.FREE_TRANSFERS, config.WILDCARD
        )
    )
    
    return should_make_transfers, transfer_analysis


def finalise_squad_selection(components, config, should_make_transfers, 
                           starting_with_transfers, bench_with_transfers,
                           scored, prev_squad_ids, transfers_made, 
                           penalty_points):
    """Finalise the squad selection based on transfer analysis."""
    if should_make_transfers:
        if penalty_points > 0:
            print(f"\n✅ Making {transfers_made} transfer(s) with "
                  f"{penalty_points} penalty points")
        else:
            print(f"\n✅ Making {transfers_made} transfer(s)")
        return starting_with_transfers, bench_with_transfers
    else:
        print(f"\n❌ Transfers not worth it - keeping current squad")
        if prev_squad_ids is not None:
            return components[
                'transfer_evaluator'
                ].get_no_transfer_squad_optimised(
                scored, prev_squad_ids
            )
        else:
            return starting_with_transfers, bench_with_transfers


def optimise_starting_xi(components, config, starting, bench, players, 
                        available_budget):
    """Optimise starting XI for the specific gameweek."""
    print(f"Optimising Starting XI for GW{config.GAMEWEEK}...")
    
    fixture_scores_next = (
        components['fixture_manager']
        .fetch_player_fixture_difficulty(1, players, config.GAMEWEEK)
    )
    
    scoring_engine_next = ScoringEngine(config)
    scored_next = scoring_engine_next.build_scores(
        players,
        fixture_scores_next
        )[0]  # Take only the dataframe, ignore stats

    # Create forced selections from current squad
    updated_forced_selections = (
        components['squad_selector']
        .update_forced_selections_from_squad(starting, bench)
    )
    
    # Try to optimise starting XI
    starting_optimised, bench_optimised, _ = (
        components['squad_selector'].select_squad_ilp(
            scored_next, updated_forced_selections,
            prev_squad_ids=None, free_transfers=None,
            show_transfer_summary=False, available_budget=available_budget,
        )
    )

    # Fallback if optimisation failed
    if starting_optimised.empty:
        print("⚠️  Starting XI optimisation infeasible with form constraint. "
              "Using fallback selection...")
        
        full_squad = pd.concat([starting, bench], ignore_index=True)
        squad_players = scored_next[
            scored_next["display_name"].isin(full_squad["display_name"])
        ].copy()
        
        if len(squad_players) >= 11:
            starting_optimised = components[
                'display_utils'
                ].select_starting_xi_fallback(
                squad_players
            )
            bench_optimised = squad_players[
                ~squad_players["id"].isin(starting_optimised["id"])
            ].copy()
            bench_optimised = components[
                'display_utils'
                ].sort_and_format_bench(
                bench_optimised
            )
        else:
            print("❌ Unable to create starting XI. Using original selection.")
            return starting, bench

    return starting_optimised, bench_optimised


def calculate_your_points(starting_display: pd.DataFrame, 
                        bench_display: pd.DataFrame, 
                        token_manager: TokenManager) -> float:
    """Calculate total points based on active chip."""
    if token_manager.should_include_bench_points():
        # Bench Boost: All 15 players count
        return (starting_display["projected_points"].sum() + 
               bench_display["projected_points"].sum())
    else:
        # Normal: Only starting XI counts
        return starting_display["projected_points"].sum()


def display_final_results(components, config, starting, bench):
    """Display final squad results with proper formatting."""
    token_manager = components['token_manager']
    
    # Add fixture information
    try:
        starting = components['fixture_manager'].add_next_fixture(
            starting, config.GAMEWEEK
        )
        bench = components['fixture_manager'].add_next_fixture(
            bench, config.GAMEWEEK
        )
    except Exception as e:
        print(f"⚠️  Warning: Could not add fixture information - {e}")

    # Add projected points analysis
    starting_display = (
        components['points_calculator']
        .add_points_analysis_to_display(starting)
    )
    bench_display = (
        components['points_calculator']
        .add_points_analysis_to_display(bench)
    )

    # Apply consistent formatting
    starting_display = components['display_utils'].sort_and_format_starting_xi(
        starting_display
    )
    starting_display = components['display_utils'].apply_captain_and_vice(
        starting_display
    )

    # Display results
    print(f"\n=== Starting XI for GW{config.GAMEWEEK} ===")
    components['display_utils'].print_squad_table(starting_display, RENAME_MAP)

    print(f"\n=== Bench (in order) for GW{config.GAMEWEEK} ===")
    components['display_utils'].print_squad_table(bench_display, RENAME_MAP)

    total_cost = starting["now_cost_m"].sum() + bench["now_cost_m"].sum()
    
    # Calculate projected points based on chip
    if token_manager.should_include_bench_points():
        # Bench Boost: All 15 players count
        total_projected_points = (
            starting_display["projected_points"].sum() + 
            bench_display["projected_points"].sum()
        )
    else:
        # Normal: Only starting XI counts
        total_projected_points = starting_display["projected_points"].sum()
    
    points_label = token_manager.get_points_label()
    
    print(f"\nTotal Squad Cost: {total_cost:.1f}m")
    print(f"{points_label}: {total_projected_points:.1f}")

    # Save squad data
    FileUtils.save_squad_data(config.GAMEWEEK, starting_display, bench_display)
    
    return starting_display, bench_display


def display_squad_comparison(theoretical_points, theoretical_cost, 
                           your_points, your_cost, penalty_points=0, 
                           token_manager=None, starting_display=None):
    """Display comparison between theoretical and actual squad."""
    if theoretical_points <= 0:
        return
    
    net_your_points = your_points - penalty_points
    
    # Get appropriate points label
    points_label = (token_manager.get_points_label() 
                   if token_manager else "Starting XI Points")
    
    print(f"\n🎯 SQUAD COMPARISON (including captain multiplier)")
    print(f"📊 Nerdball XI: {theoretical_points:.1f} pts")
    
    # Show captain information
    if starting_display is not None and not starting_display.empty:
        captain_info = _get_captain_info(starting_display, token_manager)
        if captain_info:
            print(captain_info)
    
    if penalty_points > 0:
        print(f"🏈 Your Squad (gross): {your_points:.1f} pts")
        print(f"💰 Transfer Penalty: -{penalty_points:.1f} pts")
        print(f"🎯 Your {points_label}: {net_your_points:.1f} pts")
    else:
        print(f"🏈 Your {points_label}: {your_points:.1f} pts")
    
    gap = theoretical_points - net_your_points
    gap_pct = (gap / theoretical_points * 100) if theoretical_points > 0 else 0
    print(f"📈 Gap to theoretical: {gap:.1f} pts ({gap_pct:.1f}%)")
    
    # Performance assessment
    if gap < 2:
        print("🏆 Pretty sweet squad.")
    elif gap < 5:
        print("👍 Nice effort.")
    else:
        print("💡 Not bad, could be better.")


def _get_captain_info(starting_display, token_manager):
    """Extract captain information for display."""
    # Find captain (player with (C) in name)
    captain_mask = starting_display["display_name"].str.contains(
        r"\(C\)", na=False
    )
    
    if not captain_mask.any():
        return None
    
    captain = starting_display[captain_mask].iloc[0]
    captain_name = captain["display_name"].replace(" (C)", "")
    
    # Get base points (before captain multiplier)
    if "proj_pts" in captain:
        base_points = captain["proj_pts"]
    else:
        # Fallback: reverse engineer from projected_points
        multiplier = 3 if (token_manager and 
                         token_manager.config.TRIPLE_CAPTAIN) else 2
        base_points = captain["projected_points"] / multiplier
    
    # Calculate captain bonus
    multiplier = 3 if (token_manager and 
                     token_manager.config.TRIPLE_CAPTAIN) else 2
    captain_bonus = base_points * (multiplier - 1)
    
    # Format display
    if token_manager and token_manager.config.TRIPLE_CAPTAIN:
        return f"🔥 Triple Captain: {captain_name} (+{captain_bonus:.1f}) 🔥"
    else:
        return f"💪 Captain: {captain_name} (+{captain_bonus:.1f})"


def main():
    """Main function to run the FPL optimisation process."""
    config = Config()
    token_manager = TokenManager(config)  # Handle token logic separately
    
    print_header(config)
    prompt_player_history_update(config)

    # Initialise all components
    components = initialise_components(config, token_manager)

    # Create results for previous gameweek if this is GW2+
    if config.GAMEWEEK >= 2:
        components['results_analyser'].create_previous_gameweek_results(
            config.GAMEWEEK
        )

    # Load previous squad - use correct gameweek based on tokens
    prev_squad_gameweek = token_manager.get_previous_squad_gameweek()
    prev_squad = FileUtils.load_previous_squad_from_gameweek(
        config.GAMEWEEK, prev_squad_gameweek
    )
    prev_squad_ids = None
    if prev_squad is not None:
        prev_squad_ids = (
            components['player_processor']
            .match_players_to_current(prev_squad, 
                                    components['player_processor']
                                    .fetch_current_players())
        )

    # Process player data - this now displays all stats in correct order
    players, scored, available_budget = process_player_data(components, config)

    # Generate theoretical best squad for comparison
    _, theoretical_points, theoretical_cost = (
        generate_theoretical_squad(
            components,
            config,
            players,
            available_budget
        )
    )

    # Analyse unavailable players
    analyse_unavailable_players(components, config, scored, prev_squad_ids)

    # Optimise squad
    (starting_with_transfers,
    bench_with_transfers,
    transfers_made,
    penalty_points) = (
        optimise_squad(
            components,
            config,
            scored,
            prev_squad_ids,
            available_budget
        )
    )

    # Evaluate transfer strategy
    should_make_transfers, transfer_analysis = evaluate_transfer_strategy(
        components, config, scored, prev_squad_ids, 
        starting_with_transfers, transfers_made
    )

    # Finalise squad selection
    starting, bench = finalise_squad_selection(
        components, config, should_make_transfers, 
        starting_with_transfers, bench_with_transfers,
        scored, prev_squad_ids, transfers_made, penalty_points
    )

    # Optimise starting XI for specific gameweek
    starting, bench = optimise_starting_xi(
        components, config, starting, bench, players, available_budget
    )

    # Display final results
    starting_display, bench_display = display_final_results(
        components, config, starting, bench
    )

    # Display comparison with theoretical squad
    if theoretical_points > 0:
        your_points = calculate_your_points(
            starting_display, bench_display, token_manager
        )
        your_cost = (
            starting["now_cost_m"].sum() + bench["now_cost_m"].sum()
        )
        display_squad_comparison(
            theoretical_points, theoretical_cost, 
            your_points, your_cost, penalty_points, token_manager,
            starting_display
        )


if __name__ == "__main__":
    main()