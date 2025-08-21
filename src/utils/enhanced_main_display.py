"""Enhanced display functions for main.py integration."""

from src.utils.clean_squad_display_utils import CleanSquadDisplayUtils


def generate_action_data(should_make_transfers, transfers_made, 
                        penalty_points, starting_display, bench_display, 
                        transfer_analysis=None, transfer_details=None):
    """Generate dynamic action recommendations based on actual decisions."""
    
    # Count truly unavailable players (status != 'a')
    unavailable_starting = 0
    unavailable_bench = 0
    
    if starting_display is not None:
        unavailable_starting = len(starting_display[
            starting_display.get('status', 'a') != 'a'
        ])
    
    if bench_display is not None:
        unavailable_bench = len(bench_display[
            bench_display.get('status', 'a') != 'a'
        ])
    
    total_unavailable = unavailable_starting + unavailable_bench
    
    action_data = []
    
    if should_make_transfers and transfers_made > 0:
        # Show specific transfer details if available
        if transfer_details and 'players_out' in transfer_details:
            players_out = transfer_details['players_out']
            players_in = transfer_details.get('players_in', [])
            
            if len(players_out) == len(players_in):
                out_names = ", ".join(players_out)
                in_names = ", ".join(players_in)
                
                # Truncate if too long for display width
                transfer_text = f"OUT: {out_names} → IN: {in_names}"
                if len(transfer_text) > 60:  # Leave room for "- TRANSFER  "
                    # Try shortening names
                    out_short = ", ".join([name.split()[-1] for name in players_out])
                    in_short = ", ".join([name.split()[-1] for name in players_in])
                    transfer_text = f"OUT: {out_short} → IN: {in_short}"
                
                action_data.append(f"- TRANSFER  {transfer_text}")
            else:
                # Fallback if player lists don't match
                action_data.append(f"- TRANSFER  {transfers_made} transfer(s) "
                                 f"recommended")
        else:
            # Fallback without specific details
            if penalty_points > 0:
                action_data.append(f"- TRANSFER  {transfers_made} transfer(s) "
                                 f"with {penalty_points}pt penalty")
            else:
                action_data.append(f"- TRANSFER  {transfers_made} free "
                                 "transfer(s) recommended")
        
        if (transfer_analysis
            and 'points_improvement_ppgw'
            in transfer_analysis):
            points_gain = transfer_analysis['points_improvement_ppgw']
            action_data.append(f"- REASON    Squad optimisation opportunity, "
                               f"gain {points_gain:.1f} projected points")
        elif transfer_analysis and 'points_gain' in transfer_analysis:
            points_gain = transfer_analysis['points_gain']
            action_data.append(f"- REASON    Squad optimisation opportunity, "
                               f"gain {points_gain:.1f} projected points")
        elif transfer_details and 'points_gain' in transfer_details:
            points_gain = transfer_details['points_gain']
            action_data.append(f"- REASON    Squad optimisation opportunity, "
                               f"gain {points_gain:.1f} projected points")
        elif penalty_points > 0:
            action_data.append("- REASON    Improvement worth penalty cost")
        else:
            action_data.append("- REASON    Squad optimisation opportunity")
    else:
        action_data.append("- HOLD      No transfers recommended")
        if transfer_analysis and 'reason' in transfer_analysis:
            reason = transfer_analysis['reason']
            if 'bench coverage' in reason.lower():
                action_data.append("- REASON    Bench coverage sufficient")
            elif 'not worth' in reason.lower():
                action_data.append("- REASON    Transfers not worth penalty")
            else:
                action_data.append(f"- REASON    {reason}")
        else:
            action_data.append("- REASON    Current squad optimal")

    if total_unavailable > 0:
        action_data.append(f"- WARNING   {total_unavailable} player(s) "
                          "unavailable")
    else:
        action_data.append("- STATUS    All players available")
    
    return action_data


def display_final_results_enhanced(components, config, starting, bench,
                                 theoretical_starting=None, 
                                 theoretical_bench=None,
                                 theoretical_points=0,
                                 should_make_transfers=None,
                                 transfers_made=0,
                                 penalty_points=0,
                                 transfer_analysis=None,
                                 transfer_details=None,
                                 prev_gw_summary=None):
    """
    Enhanced display function that chooses between granular and clean output.
    
    Args:
        components: Dictionary of all components
        config: Configuration object
        starting: Your starting XI dataframe
        bench: Your bench dataframe  
        theoretical_starting: Nerdball XI starting dataframe
        theoretical_bench: Nerdball XI bench dataframe
        theoretical_points: Nerdball XI total points
        should_make_transfers: Whether transfers should be made
        transfers_made: Number of transfers made
        penalty_points: Transfer penalty points
        transfer_analysis: Transfer analysis results
        transfer_details: Dict with 'players_out', 'players_in', etc.
        prev_gw_summary: Previous gameweek summary data
    """
    if config.GRANULAR_OUTPUT:
        # Use existing detailed output
        return _display_granular_output(components, config, starting, bench)
    else:
        # Use new clean dashboard output
        return _display_clean_output(components, config, starting, bench,
                                   theoretical_starting, theoretical_bench,
                                   theoretical_points, should_make_transfers,
                                   transfers_made, penalty_points,
                                   transfer_analysis, transfer_details,
                                   prev_gw_summary)


def _display_granular_output(components, config, starting, bench):
    """Display detailed granular output (existing functionality)."""
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
        print(f"Warning: Could not add fixture information - {e}")

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

    # Display results using existing method
    print(f"\n=== Starting XI for GW{config.GAMEWEEK} ===")
    components['display_utils'].print_squad_table(
        starting_display, 
        {
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
    )

    print(f"\n=== Bench (in order) for GW{config.GAMEWEEK} ===")
    components['display_utils'].print_squad_table(bench_display, 
        {
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
    )

    total_cost = starting["now_cost_m"].sum() + bench["now_cost_m"].sum()
    
    # Calculate projected points based on chip
    if token_manager.should_include_bench_points():
        total_projected_points = (
            starting_display["projected_points"].sum() + 
            bench_display["projected_points"].sum()
        )
    else:
        total_projected_points = starting_display["projected_points"].sum()
    
    points_label = token_manager.get_points_label()
    
    print(f"\nTotal Squad Cost: {total_cost:.1f}m")
    print(f"{points_label}: {total_projected_points:.1f}")

    return starting_display, bench_display


def _display_clean_output(components, config, starting, bench,
                        theoretical_starting, theoretical_bench, 
                        theoretical_points, should_make_transfers=None,
                        transfers_made=0, penalty_points=0,
                        transfer_analysis=None, transfer_details=None,
                        prev_gw_summary=None):
    """Display clean dashboard output with dynamic actions."""
    token_manager = components['token_manager']
    
    # Process your squad
    try:
        starting = components['fixture_manager'].add_next_fixture(
            starting, config.GAMEWEEK
        )
        bench = components['fixture_manager'].add_next_fixture(
            bench, config.GAMEWEEK
        )
    except Exception as e:
        print(f"Warning: Could not add fixture information - {e}")

    starting_display = (
        components['points_calculator']
        .add_points_analysis_to_display(starting)
    )
    bench_display = (
        components['points_calculator']
        .add_points_analysis_to_display(bench)
    )

    starting_display = components['display_utils'].sort_and_format_starting_xi(
        starting_display
    )
    starting_display = components['display_utils'].apply_captain_and_vice(
        starting_display
    )

    # Calculate your points
    if token_manager.should_include_bench_points():
        your_points = (
            starting_display["projected_points"].sum() + 
            bench_display["projected_points"].sum()
        )
    else:
        your_points = starting_display["projected_points"].sum()

    # Get captain info
    clean_display = CleanSquadDisplayUtils(config)
    clean_display.set_token_manager(token_manager)  # Pass token manager
    captain_name, captain_points = clean_display.get_captain_info(
        starting_display
    )

    # Use stored data if parameters are None/empty
    if theoretical_starting is None or theoretical_starting.empty:
        theoretical_starting = getattr(components, 'theoretical_starting', 
                                     None)
        theoretical_bench = getattr(components, 'theoretical_bench', None)
        theoretical_points = getattr(components, 'theoretical_points', 0)
    elif theoretical_bench is None:
        theoretical_bench = getattr(components, 'theoretical_bench', None)
    
    # Generate dynamic action data with transfer details
    action_data = generate_action_data(
        should_make_transfers, transfers_made, penalty_points,
        starting_display, bench_display, transfer_analysis, transfer_details
    )

    # Display clean dashboard
    clean_display.print_clean_dashboard(
        theoretical_starting,
        theoretical_bench, 
        theoretical_points,
        starting_display,
        bench_display,
        your_points,
        captain_name,
        captain_points,
        action_data,
        prev_gw_summary
    )

    return starting_display, bench_display


def enhance_squad_comparison_display(theoretical_points, theoretical_cost,
                                   your_points, your_cost, penalty_points=0,
                                   token_manager=None, starting_display=None,
                                   config=None):
    """
    Enhanced squad comparison that adapts to GRANULAR_OUTPUT setting.
    """
    if config and not config.GRANULAR_OUTPUT:
        # Clean output mode - comparison is shown in dashboard
        return
    
    # Granular output mode - show detailed comparison
    if theoretical_points <= 0:
        return
    
    net_your_points = your_points - penalty_points
    
    points_label = (token_manager.get_points_label() 
                   if token_manager else "Starting XI Points")
    
    print(f"\nSQUAD COMPARISON (including captain multiplier)")
    print(f"Nerdball XI: {theoretical_points:.1f} pts")
    
    # Show captain information
    if starting_display is not None and not starting_display.empty:
        captain_info = _get_captain_info(starting_display, token_manager)
        if captain_info:
            print(captain_info)
    
    if penalty_points > 0:
        print(f"Your Squad (gross): {your_points:.1f} pts")
        print(f"Transfer Penalty: -{penalty_points:.1f} pts")
        print(f"Your {points_label}: {net_your_points:.1f} pts")
    else:
        print(f"Your {points_label}: {your_points:.1f} pts")
    
    gap = theoretical_points - net_your_points
    gap_pct = (gap / theoretical_points * 100) if theoretical_points > 0 else 0
    print(f"Gap to theoretical: {gap:.1f} pts ({gap_pct:.1f}%)")
    
    # Performance assessment
    if gap < 2:
        print("Pretty sweet squad.")
    elif gap < 5:
        print("Nice effort.")
    else:
        print("Not bad, could be better.")


def _get_captain_info(starting_display, token_manager):
    """Extract captain information for display."""
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
        multiplier = 3 if (token_manager and 
                         token_manager.config.TRIPLE_CAPTAIN) else 2
        base_points = captain["projected_points"] / multiplier
    
    # Calculate captain bonus
    multiplier = 3 if (token_manager and 
                     token_manager.config.TRIPLE_CAPTAIN) else 2
    captain_bonus = base_points * (multiplier - 1)
    
    # Format display
    if token_manager and token_manager.config.TRIPLE_CAPTAIN:
        return f"Triple Captain: {captain_name} (+{captain_bonus:.1f})"
    else:
        return f"Captain: {captain_name} (+{captain_bonus:.1f})"