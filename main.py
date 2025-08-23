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
from src.analysis.differential_analyser import DifferentialAnalyser
from src.optimisation.squad_selector import SquadSelector
from src.optimisation.transfer_evaluator import TransferEvaluator
from src.utils.file_utils import FileUtils
from src.utils.squad_display_utils import SquadDisplayUtils
from src.utils.token_manager import TokenManager
from src.utils.calculation_display_utils import CalculationDisplayUtils
from src.utils.enhanced_main_display import (
    display_final_results_enhanced, 
    enhance_squad_comparison_display
)


def extract_transfer_details(prev_squad_ids, starting_with_transfers, 
                            bench_with_transfers, players_df):
    """
    Extract specific transfer details (who goes out, who comes in).
    
    Args:
        prev_squad_ids: List of previous squad player IDs
        starting_with_transfers: New starting XI dataframe
        bench_with_transfers: New bench dataframe  
        players_df: Full players dataframe with names
        
    Returns:
        Dict with 'players_out', 'players_in' lists
    """
    if prev_squad_ids is None:
        return None
    
    # Get current squad IDs
    current_squad_ids = set(
        pd.concat([starting_with_transfers, bench_with_transfers])["id"]
    )
    prev_squad_ids_set = set(prev_squad_ids)
    
    # Find transferred players
    players_out_ids = prev_squad_ids_set - current_squad_ids
    players_in_ids = current_squad_ids - prev_squad_ids_set
    
    if not players_out_ids and not players_in_ids:
        return None
    
    # Get player names
    players_out_names = []
    players_in_names = []
    
    # Create a mapping of id to display_name from players dataframe
    id_to_name = players_df.set_index('id')['display_name'].to_dict()
    
    # Also check the new squad dataframes for names
    new_squad_df = pd.concat([starting_with_transfers, bench_with_transfers])
    new_id_to_name = new_squad_df.set_index('id')['display_name'].to_dict()
    id_to_name.update(new_id_to_name)
    
    for player_id in players_out_ids:
        name = id_to_name.get(player_id, f"Player {player_id}")
        players_out_names.append(name)
    
    for player_id in players_in_ids:
        name = id_to_name.get(player_id, f"Player {player_id}")
        players_in_names.append(name)
    
    return {
        'players_out': players_out_names,
        'players_in': players_in_names,
        'players_out_ids': list(players_out_ids),
        'players_in_ids': list(players_in_ids)
    }


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
        
    if not config.GRANULAR_OUTPUT:
        return  # Skip prompt in clean mode
        
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
                    print("Player history updated successfully!")
                    if result.stdout:
                        print(result.stdout)
                else:
                    print("Error updating player history:")
                    if result.stderr:
                        print(result.stderr)
                    
            except Exception as e:
                print(f"Failed to run player history update: {e}")
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
        'differential_analyser': DifferentialAnalyser(config),
        'squad_selector': SquadSelector(config),
        'transfer_evaluator': TransferEvaluator(config),
        'display_utils': SquadDisplayUtils(config),
        'calculation_display': CalculationDisplayUtils(config),
        'token_manager': token_manager
    }


def print_header(config):
    """Print application header with current settings."""
    if not config.GRANULAR_OUTPUT:
        return  # Skip header in clean mode
        
    print(f"\n=== WELCOME TO FANTASY NERDBALL ===")
    print(f"\nPlanning for Gameweek {config.GAMEWEEK}")
    
    if config.WILDCARD:
        print("WILDCARD ACTIVE - No transfer limits!")
    elif config.ACCEPT_TRANSFER_PENALTY:
        print(f"TRANSFER PENALTY MODE - Can exceed "
              f"{config.FREE_TRANSFERS} free transfers (4 pts penalty each)")
    else:
        print(f"Free transfers available: {config.FREE_TRANSFERS}")


def _display_availability_and_dgw_stats(involvement_stats, dgw_stats, 
                                       gameweek, config):
    """Display player availability and DGW/BGW statistics."""
    if not involvement_stats['exclude_unavailable']:
        if config.GRANULAR_OUTPUT:
            print("EXCLUDE_UNAVAILABLE = False: Including unavailable "
                  "players in optimisation")
        return
    
    if config.GRANULAR_OUTPUT:
        print("\nChecking availability...")
        if gameweek > 1:
            if involvement_stats['high_involvement'] > 0:
                print(f"{involvement_stats['high_involvement']} players with "
                      "high involvement (reliable starters)")
                
            if involvement_stats['low_involvement'] > 0:
                print(f"{involvement_stats['low_involvement']} players with "
                      "low involvement heavily penalised")
                
            if involvement_stats['zero_involvement'] > 0:
                print(f"{involvement_stats['zero_involvement']} players with "
                      "zero involvement will have scores set to 0")
        
        if involvement_stats['unavailable'] > 0:
            print(f"{involvement_stats['unavailable']} players unavailable "
                  "due to injury/suspension")
        else:
            print("All players are available for selection")
        
        print("\nChecking double/blank gameweeks...")
        print(f"{dgw_stats['regular_gw']} players with a regular gameweek")
        
        if dgw_stats['double_gw'] > 0:
            print(f"{dgw_stats['double_gw']} players with a double gameweek")
        
        if dgw_stats['blank_gw'] > 0:
            print(f"{dgw_stats['blank_gw']} players with a blank gameweek")
        
        if dgw_stats['double_gw'] == 0 and dgw_stats['blank_gw'] == 0:
            print("All players have regular gameweeks")


def process_player_data(components, config):
    """Process player data with historical and fixture analysis."""
    if config.GRANULAR_OUTPUT:
        print("Fetching current players with xG analysis...")
    
    players = components['player_processor'].fetch_current_players()

    # Calculate available budget based on previous squad value
    available_budget = (
        components['player_processor']
        .calculate_budget_from_previous_squad(config.GAMEWEEK, players)
    )

    if config.GRANULAR_OUTPUT:
        print("Analysing historical xG performance...")
    
    players = components['historical_manager'].merge_past_seasons(players)
    
    if config.GRANULAR_OUTPUT:
        print(f"Fetching player-level fixture difficulty from "
              f"GW{config.GAMEWEEK} with DGW/BGW handling...")
    
    fixture_scores = (
        components['fixture_manager']
        .fetch_player_fixture_difficulty(
            config.FIRST_N_GAMEWEEKS, players, config.GAMEWEEK
        )
    )
    
    if config.GRANULAR_OUTPUT:
        print("Scoring players with xG performance modifiers...")
    
    scored, involvement_stats = components['scoring_engine'].build_scores(
        players, fixture_scores
    )
    
    # Get DGW/BGW statistics
    dgw_stats = _calculate_dgw_stats(scored, config.GAMEWEEK)
    
    # Display availability and DGW statistics
    _display_availability_and_dgw_stats(involvement_stats, dgw_stats, 
                                       config.GAMEWEEK, config)
    
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


def generate_theoretical_squad(components, config, players, 
                              available_budget):
    """Generate theoretical best squad for comparison."""
    # Always generate theoretical squad regardless of output mode
    if config.GRANULAR_OUTPUT:
        print(f"\n=== NERDBALL PICKS GW{config.GAMEWEEK} ===")
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
    
    if config.GRANULAR_OUTPUT:
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
        if config.GRANULAR_OUTPUT:
            print("Could not generate theoretical best squad")
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
    
    theoretical_cost = (
        theoretical_starting["now_cost_m"].sum() + 
        theoretical_bench["now_cost_m"].sum()
    )
    theoretical_points = theoretical_starting_display["projected_points"].sum()
    
    # Store for clean display
    components['theoretical_starting'] = theoretical_starting_display
    components['theoretical_bench'] = theoretical_bench_display
    components['theoretical_points'] = theoretical_points

    # Only print detailed output in granular mode
    if config.GRANULAR_OUTPUT:
        # Display detailed calculations for theoretical squad if enabled
        if config.DETAILED_CALCULATION:
            print(f"\n=== THEORETICAL SQUAD CALCULATIONS ===")
            components['calculation_display'].display_squad_calculations(
                theoretical_starting_display, theoretical_bench_display
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
        
        print(f"\nNerdball Squad Cost: {theoretical_cost:.1f}m")
        print(f"Nerdball Starting XI Projected Points: "
              f"{theoretical_points:.1f}")
        
        # For differentials, use the full fixture analysis (same as main squad)
        differential_scored = components['scoring_engine'].build_scores(
            players, 
            components['fixture_manager'].fetch_player_fixture_difficulty(
                config.FIRST_N_GAMEWEEKS, players, config.GAMEWEEK
            )
        )[0]
        
        # Add multi-fixture information for differentials
        differential_scored_with_fixtures = (
            components['differential_analyser']
            .add_multi_fixture_info(differential_scored, config.GAMEWEEK,
                                   config.FIRST_N_GAMEWEEKS, 
                                   components['fixture_manager'])
        )
        
        # Add differential suggestions
        differentials = components[
            'differential_analyser'].get_differential_suggestions(
            differential_scored_with_fixtures,
            config.GAMEWEEK,
            config.FIRST_N_GAMEWEEKS
        )
        components['differential_analyser'].print_differential_suggestions(
            differentials,
            config.GAMEWEEK,
            config.FIRST_N_GAMEWEEKS
        )
    
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
            print(f"\nRECOMMENDATION: Consider activating wildcard")
            print(f"   Too many unavailable players for available transfers")
        elif substitute_analysis["recommendation"] == "use_substitutes":
            print(f"\nRECOMMENDATION: Use substitutes, save transfers")
            print(f"   Substitution score loss acceptable")
    else:
        if config.GRANULAR_OUTPUT:
            print(f"\nAll previous squad players are available")


def optimise_squad(
        components,
        config, scored,
        prev_squad_ids,
        available_budget
        ):
    """Optimise squad selection with transfer considerations."""
    if config.GRANULAR_OUTPUT:
        print("\nThinking...")

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
        if config.GRANULAR_OUTPUT:
            if penalty_points > 0:
                print(f"\nMaking {transfers_made} transfer(s) with "
                      f"{penalty_points} penalty points")
            else:
                print(f"\nMaking {transfers_made} transfer(s)")
        return starting_with_transfers, bench_with_transfers
    else:
        if config.GRANULAR_OUTPUT:
            print(f"\nTransfers not worth it - keeping current squad")
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
    if config.GRANULAR_OUTPUT:
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
        if config.GRANULAR_OUTPUT:
            print("Starting XI optimisation infeasible with form constraint. "
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
            if config.GRANULAR_OUTPUT:
                print("Unable to create starting XI. Using original selection.")
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


def get_prev_gw_summary(components, config):
    """Get previous gameweek summary for display."""
    if config.GAMEWEEK <= 1:
        return None
        
    prev_gw = config.GAMEWEEK - 1
    summary_file = f"squads/gw{prev_gw}/summary.csv"
    
    try:
        import os
        if os.path.exists(summary_file):
            summary_df = pd.read_csv(summary_file)
            
            # Extract key metrics
            total_metric = summary_df[
                summary_df['metric'] == 'Starting XI Total Points'
            ]
            mae_metric = summary_df[
                summary_df['metric'] == 'Mean Absolute Error (XI)'
            ]
            accuracy_metric = summary_df[
                summary_df['metric'] == 'Accuracy within 2pts (XI)'
            ]
            best_metric = summary_df[
                summary_df['metric'] == 'Best Starting XI Performer'
            ]
            worst_metric = summary_df[
                summary_df['metric'] == 'Worst Starting XI Performer'
            ]
            
            # Extract position breakdowns
            position_metrics = {}
            for pos in ['GK', 'DEF', 'MID', 'FWD']:
                pos_metric = summary_df[
                    summary_df['metric'] == f'{pos} Starting XI'
                ]
                if len(pos_metric) > 0:
                    position_metrics[pos] = {
                        'actual': int(pos_metric.iloc[0]['actual']),
                        'projected': float(pos_metric.iloc[0]['projected']),
                        'difference': float(pos_metric.iloc[0]['difference'])
                    }
            
            if len(total_metric) > 0:
                accuracy_value = "N/A"
                if len(accuracy_metric) > 0:
                    accuracy_raw = accuracy_metric.iloc[0]['actual']
                    if isinstance(accuracy_raw, str) and '%' in accuracy_raw:
                        accuracy_value = accuracy_raw
                    else:
                        try:
                            accuracy_value = f"{float(accuracy_raw):.1f}%"
                        except (ValueError, TypeError):
                            accuracy_value = "N/A"
                
                # Extract best/worst performer names
                best_performer = "N/A"
                worst_performer = "N/A"
                best_diff = 0
                worst_diff = 0
                
                if len(best_metric) > 0:
                    best_performer = best_metric.iloc[0]['notes'].split(' (')[0]
                    best_diff = float(best_metric.iloc[0]['difference'])
                
                if len(worst_metric) > 0:
                    worst_performer = worst_metric.iloc[0]['notes'].split(' (')[0]
                    worst_diff = float(worst_metric.iloc[0]['difference'])
                
                return {
                    'gameweek': prev_gw,
                    'actual_points': int(total_metric.iloc[0]['actual']),
                    'projected_points': float(total_metric.iloc[0]['projected']),
                    'difference': float(total_metric.iloc[0]['difference']),
                    'mae': float(mae_metric.iloc[0]['actual']) if len(mae_metric) > 0 else 0,
                    'accuracy': accuracy_value,
                    'best_performer': best_performer,
                    'best_diff': best_diff,
                    'worst_performer': worst_performer,
                    'worst_diff': worst_diff,
                    'positions': position_metrics
                }
    except Exception:
        pass
    
    return None


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
        if not config.GRANULAR_OUTPUT:
            print("\n📈 Reviewing previous gameweek...")
            try:
                result = subprocess.run([
                    sys.executable, 'update_player_history.py', 'update'
                ], capture_output=True, text=True)
                
                # Only show output if there's an error
                if result.returncode != 0 and result.stderr:
                    print(f"Warning: Error updating player history: "
                          f"{result.stderr}")
                    
            except Exception as e:
                print(f"Warning: Could not update player history: {e}")

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
    if not config.GRANULAR_OUTPUT:
        print("🤕 Seeing who's available this week...")
    players, scored, available_budget = process_player_data(components, config)

    if not config.GRANULAR_OUTPUT:
        print("⏩ Looking for double gameweeks...")

    # Generate theoretical best squad for comparison
    if not config.GRANULAR_OUTPUT:
        print("🏆 Picking my favourite players for this week...")
    theoretical_starting_display, theoretical_points, theoretical_cost = (
        generate_theoretical_squad(
            components,
            config,
            players,
            available_budget
        )
    )

    # Analyse unavailable players (only for granular output)
    if config.GRANULAR_OUTPUT:
        analyse_unavailable_players(components, config, scored, prev_squad_ids)

    # Optimise squad
    if not config.GRANULAR_OUTPUT:
        print("↔️  Browsing the transfer market...")
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

    # For penalty mode, get the improvement data from the transfer evaluator
    transfer_analysis = {}
    if config.ACCEPT_TRANSFER_PENALTY and hasattr(
        components['transfer_evaluator'], '_last_best_scenario'):
        scenario = components['transfer_evaluator']._last_best_scenario
        transfer_analysis = {
            "reason": "Transfers not worth it",
            "points_improvement_ppgw": scenario.get(
            'points_improvement_ppgw', 0
            ),
            "gameweeks_analysed": scenario.get(
            'gameweeks_analysed', config.FIRST_N_GAMEWEEKS
            )
        }

    # Evaluate transfer strategy 
    (should_make_transfers,
     transfer_analysis_from_evaluator)= evaluate_transfer_strategy(
        components, config, scored, prev_squad_ids, 
        starting_with_transfers, transfers_made
    )
    
    # Use the analysis from evaluator if it has more data,
    # otherwise use penalty mode data
    if (transfer_analysis_from_evaluator
        and 'points_improvement_ppgw'
        in transfer_analysis_from_evaluator):
        transfer_analysis = transfer_analysis_from_evaluator
    elif not transfer_analysis:
        transfer_analysis = transfer_analysis_from_evaluator

    # Finalise squad selection
    starting, bench = finalise_squad_selection(
        components, config, should_make_transfers, 
        starting_with_transfers, bench_with_transfers,
        scored, prev_squad_ids, transfers_made, penalty_points
    )

    # Optimise starting XI for specific gameweek
    if not config.GRANULAR_OUTPUT:
        print("🧠 Making final adjustments...")
    starting, bench = optimise_starting_xi(
        components, config, starting, bench, players, available_budget
    )

    # Extract transfer details for display
    transfer_details = extract_transfer_details(
        prev_squad_ids, starting_with_transfers, bench_with_transfers, players
    )

    # Get previous gameweek summary
    prev_gw_summary = get_prev_gw_summary(components, config)

    # Display final results with enhanced logic - PASS TRANSFER DATA
    starting_display, bench_display = display_final_results_enhanced(
        components, config, starting, bench,
        theoretical_starting_display,  # Pass the actual dataframe
        components.get('theoretical_bench'), 
        theoretical_points,  # Pass the actual points
        should_make_transfers,  # NEW: Pass transfer decision
        transfers_made,        # NEW: Pass number of transfers
        penalty_points,        # NEW: Pass penalty points
        transfer_analysis,     # NEW: Pass transfer analysis
        transfer_details,      # NEW: Pass specific transfer details
        prev_gw_summary        # NEW: Pass previous gameweek summary
    )

    # Display detailed calculations for your actual squad if enabled
    if config.GRANULAR_OUTPUT and config.DETAILED_CALCULATION:
        print(f"\n=== YOUR SQUAD CALCULATIONS ===")
        components['calculation_display'].display_squad_calculations(
            starting_display, bench_display
        )

    # Display comparison with theoretical squad
    if theoretical_points > 0:
        your_points = calculate_your_points(
            starting_display, bench_display, token_manager
        )
        your_cost = (
            starting["now_cost_m"].sum() + bench["now_cost_m"].sum()
        )
        enhance_squad_comparison_display(
            theoretical_points, theoretical_cost, 
            your_points, your_cost, penalty_points, token_manager,
            starting_display, config
        )

    if not starting_display.empty and not bench_display.empty:
        FileUtils.save_squad_data(config.GAMEWEEK, starting_display, bench_display)
        if config.GRANULAR_OUTPUT:
            print(f"✅ Squad data saved for GW{config.GAMEWEEK}")


if __name__ == "__main__":
    main()