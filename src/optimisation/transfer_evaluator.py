"""Module for evaluating transfer strategies and alternatives."""

import pandas as pd
import pulp


class TransferEvaluator:
    """Handles evaluation of transfer strategies and alternatives."""
    
    def __init__(self, config):
        self.config = config
    
    def get_unavailable_players(self, df: pd.DataFrame, 
                              prev_squad_ids: list) -> list:
        """
        Identify players from previous squad who are unavailable 
        for this gameweek.

        Args:
            df (pd.DataFrame): Current player database
            prev_squad_ids (list): Player IDs from previous squad

        Returns:
            list: Player IDs who are unavailable
        """
        if not prev_squad_ids:
            return []

        unavailable_ids = []
        for player_id in prev_squad_ids:
            player = df[df["id"] == player_id]
            if not player.empty:
                player_row = player.iloc[0]
                # Check if player is unavailable (injured, suspended, etc.)
                if (player_row["status"] != "a" and 
                    player_row.get("chance_of_playing_next_round", 100) < 75):
                    unavailable_ids.append(player_id)
        
        return unavailable_ids
    
    def _calculate_hypothetical_projected_points(
            self, player_row: pd.Series) -> float:
        """
        Calculate what a player's projected points would be if they were 
        available. This recreates the calculation from ScoringEngine but skips
        availability filters.
        
        Args:
            player_row (pd.Series): Player data row
            
        Returns:
            float: Hypothetical projected points if player was available
        """
        # Get the player's position for baseline points
        position = player_row.get("position", "MID")
        baseline_points = self.config.BASELINE_POINTS_PER_GAME.get(
            position, 4.0)
        
        # Get the base quality score (this should still be calculated normally)
        base_quality = player_row.get("base_quality", 0.0)
        
        # Convert base quality to points adjustment
        points_adjustment = (
            base_quality * self.config.FPL_SCORE_TO_POINTS_MULTIPLIER
        )
        
        # Calculate hypothetical projected points
        hypothetical_points = baseline_points + points_adjustment
        
        # Ensure minimum of 1 point
        hypothetical_points = max(1.0, hypothetical_points)
        
        # Apply fixture multiplier if present
        fixture_multiplier = player_row.get("fixture_multiplier", 1.0)
        if pd.notna(fixture_multiplier) and fixture_multiplier > 0:
            hypothetical_points *= fixture_multiplier
        
        return hypothetical_points
    
    def evaluate_substitute_vs_transfer(
        self, df: pd.DataFrame, prev_squad_ids: list, 
        unavailable_player_ids: list, free_transfers: int
    ) -> dict:
        """
        Evaluate whether to substitute unavailable players or transfer them 
        out using previous gameweek's squad data.

        Args:
            df (pd.DataFrame): Current player database with scores
            prev_squad_ids (list): Player IDs from previous squad
            unavailable_player_ids (list): IDs of players who can't play this 
                                          GW 
            free_transfers (int): Number of free transfers available

        Returns:
            dict: Analysis of substitute vs transfer options
        """
        if not prev_squad_ids or not unavailable_player_ids:
            return {
                "recommendation": "no_action",
                "reason": "No unavailable players or no previous squad",
            }

        # Load previous gameweek squad CSV
        prev_gw = self.config.GAMEWEEK - 1
        prev_squad_file = f"squads/gw{prev_gw}/full_squad_simple.csv"
        
        try:
            import os
            import pandas as pd
            if not os.path.exists(prev_squad_file):
                return self._fallback_substitute_analysis()
            
            prev_squad_csv = pd.read_csv(prev_squad_file)
            
        except Exception as e:
            return self._fallback_substitute_analysis()

        # Get current player data
        prev_squad_df = df[df["id"].isin(prev_squad_ids)].copy()
        unavailable_df = prev_squad_df[
            prev_squad_df["id"].isin(unavailable_player_ids)
        ]

        if self.config.GRANULAR_OUTPUT:
            print(f"\n=== Substitute vs Transfer Analysis ===")
            print(f"Unavailable players: {len(unavailable_df)}")
            for _, player in unavailable_df.iterrows():
                print(f"  - {player['display_name']} "
                      f"({player['position']}, {player['team']})")

        substitute_scenarios = []
        
        for _, unavailable_player in unavailable_df.iterrows():
            hypothetical_total_points = (
                self._calculate_hypothetical_projected_points(
                unavailable_player)
            )
            unavailable_ppgw = (
                hypothetical_total_points / self.config.FIRST_N_GAMEWEEKS
            )
            
            # Find best substitute from bench
            best_substitute = self._find_best_bench_substitute(
                unavailable_player, prev_squad_csv, prev_squad_df
            )
            
            if best_substitute is None:
                substitute_scenarios.append({
                    "unavailable_player": unavailable_player["display_name"],
                    "position": unavailable_player["position"],
                    "unavailable_score": unavailable_ppgw,
                    "best_substitute": "No suitable substitute",
                    "substitute_score": 0,
                    "score_loss": unavailable_ppgw,
                    "recommendation": "transfer",
                })
            else:
                substitute_ppgw = (best_substitute["projected_points"] / 
                                 self.config.FIRST_N_GAMEWEEKS)
                
                unavailable_rounded = round(unavailable_ppgw, 1)
                substitute_rounded = round(substitute_ppgw, 1)
                score_loss = unavailable_rounded - substitute_rounded
                
                substitute_scenarios.append({
                    "unavailable_player": unavailable_player["display_name"],
                    "position": unavailable_player["position"],
                    "unavailable_score": unavailable_rounded,
                    "best_substitute": (f"{best_substitute['display_name']} "
                                      f"({best_substitute['position']})"),
                    "substitute_score": substitute_rounded,
                    "score_loss": score_loss,
                    "recommendation": (
                        "substitute" if score_loss < 2.0
                        else "consider_transfer"
                    ),
                })

        return self._evaluate_substitution_strategy(
            substitute_scenarios, free_transfers
        )
    
    def _find_best_bench_substitute(self, unavailable_player: pd.Series,
                                  prev_squad_csv: pd.DataFrame,
                                  prev_squad_df: pd.DataFrame) -> pd.Series:
        """Find the best substitute from bench players."""
        
        # Get bench players from CSV
        bench_csv = prev_squad_csv[prev_squad_csv['squad_role'] == 'Bench']
        
        # Find bench players in current data who are available
        available_bench_players = []
        
        for _, bench_player_csv in bench_csv.iterrows():
            # Clean player name (remove captain/vice markers)
            player_name = bench_player_csv[
                'player'].replace(' (C)', '').replace(' (V)', '')
            
            # Find matching player in current data
            matching_players = prev_squad_df[
                prev_squad_df['display_name'].str.lower().str.contains(
                    player_name.lower(), na=False, regex=False
                )
            ]
            
            for _, matching_player in matching_players.iterrows():
                if matching_player["status"] == "a":  # Available to play
                    available_bench_players.append(matching_player)
                    break  # Only add once
        
        if not available_bench_players:
            return None
        
        # Convert to DataFrame and find highest projected points
        bench_df = pd.DataFrame(available_bench_players)
        bench_df = bench_df.drop_duplicates(subset=['id'])
        
        # Return player with highest projected points
        return bench_df.loc[bench_df["projected_points"].idxmax()]
    
    def _fallback_substitute_analysis(self) -> dict:
        """Fallback analysis when CSV data is not available."""
        return {
            "recommendation": "transfer",
            "reason": "Cannot determine previous squad composition, "
                     "recommend transfers",
            "scenarios": []
        }
    
    def _evaluate_substitution_strategy(self, substitute_scenarios: list, 
                                      free_transfers: int) -> dict:
        """
        Evaluate overall substitution strategy based on scenarios.
        
        Args:
            substitute_scenarios (list): List of substitution scenarios
            free_transfers (int): Number of free transfers available
            
        Returns:
            dict: Strategy recommendation
        """
        # Calculate total impact of substitutions (already in ppgw)
        total_score_loss = sum(scenario["score_loss"] 
                             for scenario in substitute_scenarios)
        forced_transfers = len(
            [s for s in substitute_scenarios if s["best_substitute"] is None]
        )

        # Use MIN_TRANSFER_VALUE directly (already per-gameweek)
        threshold_ppgw = self.config.MIN_TRANSFER_VALUE

        # Decision logic based on projected points per gameweek
        if forced_transfers > free_transfers:
            decision = {
                "recommendation": "wildcard_needed",
                "reason": (f"Need {forced_transfers} forced transfers but "
                          f"only have {free_transfers} free"),
                "total_score_loss": total_score_loss,
                "scenarios": substitute_scenarios,
            }
        elif total_score_loss > threshold_ppgw:
            decision = {
                "recommendation": "make_transfers",
                "reason": (f"Score loss ({total_score_loss:.1f}) exceeds "
                          f"transfer threshold ({threshold_ppgw:.1f})"),
                "total_score_loss": total_score_loss,
                "scenarios": substitute_scenarios,
            }
        else:
            decision = {
                "recommendation": "use_substitutes",
                "reason": (f"Score loss ({total_score_loss:.1f}) is "
                           "acceptable, save transfers"),
                "total_score_loss": total_score_loss,
                "scenarios": substitute_scenarios,
            }

        if self.config.GRANULAR_OUTPUT:
            self._print_substitution_analysis(
                substitute_scenarios,
                total_score_loss, 
                decision,
                threshold_ppgw
            )
        return decision
    
    def _print_substitution_analysis(self, substitute_scenarios: list, 
                                   total_score_loss: float, decision: dict,
                                   threshold_ppgw: float):
        """Print substitution analysis to console with ppgw values."""
        print(f"\nSubstitution scenarios:")
        for scenario in substitute_scenarios:
            if scenario["best_substitute"]:
                print(f"  {scenario['unavailable_player']} "
                      f"(projected points {scenario['unavailable_score']:.1f})"
                       f" → {scenario['best_substitute']} "
                      f"(projected points {scenario['substitute_score']:.1f})")
            else:
                print(f"  {scenario['unavailable_player']} → "
                      f"NO SUBSTITUTE AVAILABLE (must transfer)")

        # Display total score loss - show "NONE" if negative (beneficial)
        if total_score_loss < 0:
            print(f"Total score loss from substitutions: NONE")
        else:
            print(f"\nTotal score loss from substitutions: "
                  f"{total_score_loss:.1f}")
            
        print(f"Transfer threshold: {threshold_ppgw:.1f}")

    
    def _calculate_minimum_transfers_needed(self, forced_selections: dict, 
                                          prev_squad_ids: list, 
                                          df: pd.DataFrame) -> int:
        """
        Calculate the minimum number of transfers needed to satisfy 
        forced selections.
        
        Args:
            forced_selections (dict): Dictionary of forced player selections
            prev_squad_ids (list): Player IDs from previous squad
            df (pd.DataFrame): Current player database
            
        Returns:
            int: Minimum transfers needed
        """
        if not prev_squad_ids or not any(forced_selections.values()):
            return 0
            
        prev_squad_ids_set = set(prev_squad_ids)
        forced_player_ids = set()
        
        # Get all forced player IDs
        for position, player_names in forced_selections.items():
            for player_name in player_names:
                player_match = df[
                    df["display_name"].str.lower() == player_name.lower()
                ]
                if not player_match.empty:
                    forced_player_ids.add(player_match.iloc[0]["id"])
        
        # Count how many forced players are not in previous squad
        forced_not_in_prev = forced_player_ids - prev_squad_ids_set
        return len(forced_not_in_prev)
    
    def get_optimal_squad_with_penalties(
        self, df: pd.DataFrame, forced_selections: dict, prev_squad_ids: list, 
        free_transfers: int, available_budget: float, squad_selector
    ) -> tuple:
        """
        Get optimal squad considering transfer penalties when 
        ACCEPT_TRANSFER_PENALTY is True.
        
        Args:
            df (pd.DataFrame): Current player database with scores
            forced_selections (dict): Dictionary of forced player selections
            prev_squad_ids (list): Player IDs from previous squad
            free_transfers (int): Number of free transfers available
            available_budget (float): Available budget for squad
            squad_selector: SquadSelector instance
            
        Returns:
            tuple: (starting_xi, bench, forced_selections_display, 
                   transfers_made, penalty_points)
        """
        if (not self.config.ACCEPT_TRANSFER_PENALTY or 
            prev_squad_ids is None):
            # Use normal squad selection without penalty consideration
            starting, bench, forced_display = (
                squad_selector.select_squad_ilp(
                    df, forced_selections, prev_squad_ids, free_transfers, 
                    show_transfer_summary=True, 
                    available_budget=available_budget,
                    use_projected_points=False
                )
            )
            return starting, bench, forced_display, 0, 0
        
        if self.config.GRANULAR_OUTPUT:
            print(f"\n=== TRANSFER ANALYSIS ===")
            print(f"\nEvaluating all transfer scenarios up to "
                  f"{free_transfers + 3} transfers...")
        
        scenarios = self._evaluate_transfer_scenarios(
            df, forced_selections, prev_squad_ids, free_transfers, 
            available_budget, squad_selector
        )
        
        if not scenarios:
            if self.config.GRANULAR_OUTPUT:
                print("No valid scenarios found")
            return pd.DataFrame(), pd.DataFrame(), None, 0, 0
        
        return self._select_best_scenario(scenarios, free_transfers, df)
    
    def _evaluate_transfer_scenarios(self, df: pd.DataFrame, 
                                   forced_selections: dict, 
                                   prev_squad_ids: list,
                                   free_transfers: int,
                                   available_budget: float,
                                   squad_selector
                                   ) -> list:
        """Evaluate different transfer scenarios and return results."""
        scenarios = []
        
        # Calculate minimum transfers needed for forced selections
        min_transfers_needed = self._calculate_minimum_transfers_needed(
            forced_selections, prev_squad_ids, df
        )
        
        # Test different transfer scenarios - 
        # start from minimum needed transfers
        max_transfers_to_test = free_transfers + 3
        start_transfers = min_transfers_needed
        
        if min_transfers_needed > 0 and self.config.GRANULAR_OUTPUT:
            print(f"Forced selections require minimum "
                  f"{min_transfers_needed} transfer(s)")
        
        for max_transfers_allowed in range(start_transfers, 
                                         max_transfers_to_test + 1):
            scenario = self._evaluate_single_scenario(
                df, forced_selections, prev_squad_ids, max_transfers_allowed,
                available_budget, squad_selector, free_transfers
            )
            
            if scenario:
                scenarios.append(scenario)
                if self.config.GRANULAR_OUTPUT:
                    self._print_scenario_result(scenario)
        
        return scenarios
    
    def _evaluate_single_scenario(self, df: pd.DataFrame, 
                                 forced_selections: dict, 
                                 prev_squad_ids: list,
                                 max_transfers_allowed: int, 
                                 available_budget: float, 
                                 squad_selector,
                                 free_transfers: int) -> dict:
        """Evaluate a single transfer scenario."""
        # Get optimal squad with this transfer limit
        starting, bench, forced_display = squad_selector.select_squad_ilp(
            df, forced_selections, prev_squad_ids, max_transfers_allowed,
            show_transfer_summary=False, available_budget=available_budget,
            use_projected_points=False
        )
        
        if starting.empty:
            return None
            
        # Calculate actual transfers made
        current_squad_ids = set(pd.concat([starting, bench])["id"].tolist())
        prev_squad_ids_set = set(prev_squad_ids)
        actual_transfers = len(prev_squad_ids_set - current_squad_ids)
        
        # Calculate penalty
        extra_transfers = max(0, actual_transfers - free_transfers)
        penalty_points = extra_transfers * 4
        
        # Calculate points per gameweek and apply penalty directly to ppgw
        starting_points_total = starting["projected_points"].sum()
        starting_ppgw = starting_points_total / self.config.FIRST_N_GAMEWEEKS
        net_ppgw = starting_ppgw - penalty_points  # Penalty applied directly
        
        # Get transfer details
        transfer_details = self._format_transfer_details(
            prev_squad_ids_set, current_squad_ids, df, starting, bench
        )
        
        return {
            'max_transfers_allowed': max_transfers_allowed,
            'actual_transfers': actual_transfers,
            'extra_transfers': extra_transfers,
            'penalty_points': penalty_points,
            'starting_points_total': starting_points_total,
            'starting_ppgw': starting_ppgw,
            'net_ppgw': net_ppgw,
            'starting': starting,
            'bench': bench,
            'forced_display': forced_display,
            'transfer_details': transfer_details
        }
    
    def _format_transfer_details(self, prev_squad_ids_set: set, 
                               current_squad_ids: set, df: pd.DataFrame,
                               starting: pd.DataFrame, 
                               bench: pd.DataFrame) -> str:
        """Format transfer details for display."""
        players_out = prev_squad_ids_set - current_squad_ids
        players_in = current_squad_ids - prev_squad_ids_set
        
        if not players_out or not players_in:
            return ""
        
        out_names = []
        for player_id in players_out:
            prev_player = df[df["id"] == player_id]
            if not prev_player.empty:
                out_names.append(prev_player.iloc[0]["display_name"])
        
        in_names = []
        for player_id in players_in:
            player = pd.concat([starting, bench])[
                pd.concat([starting, bench])["id"] == player_id
            ].iloc[0]
            in_names.append(player["display_name"])
        
        if out_names and in_names:
            return (f" (OUT: {', '.join(out_names)} → "
                   f"IN: {', '.join(in_names)})")
        
        return ""
    
    def _print_scenario_result(self, scenario: dict):
        """Print the result of a transfer scenario showing ppgw as 
        'projected points'."""
        actual_transfers = scenario['actual_transfers']
        extra_transfers = scenario['extra_transfers']
        penalty_points = scenario['penalty_points']
        starting_ppgw = scenario['starting_ppgw']
        net_ppgw = scenario['net_ppgw']
        transfer_details = scenario['transfer_details']
        max_transfers_allowed = scenario['max_transfers_allowed']
        
        if actual_transfers == 0:
            prefix = "  Scenario"
        elif extra_transfers == 0:
            prefix = "  Scenario"
        else:
            prefix = "  Scenario"
        
        print(f"{prefix} {max_transfers_allowed}: "
              f"{actual_transfers} transfers, {extra_transfers} extra, "
              f"penalty: -{penalty_points}, projected points: "
              f"{starting_ppgw:.1f}, net: {net_ppgw:.1f}"
              f"{transfer_details}")
    
    def _select_best_scenario(self, scenarios: list, free_transfers: int, 
                            df: pd.DataFrame) -> tuple:
        """Select the best scenario based on net points per gameweek."""
        # Find the best scenario by net points per gameweek (but still use ppgw 
        # for internal comparison)
        best_scenario = max(scenarios, key=lambda x: x['net_ppgw'])
        
        # Get baseline scenario for comparison
        baseline_scenario = self._get_baseline_scenario(scenarios)
        
        if baseline_scenario is None:
            if self.config.GRANULAR_OUTPUT:
                print("No baseline scenario found")
            return pd.DataFrame(), pd.DataFrame(), None, 0, 0
        
        if self.config.GRANULAR_OUTPUT:
            print(f"\nBest scenario analysis:")
            print(f"   Tested {len(scenarios)} different transfer limits")
            
            # Show top 3 scenarios for comparison
            self._print_top_scenarios(scenarios, best_scenario)
        
        # Apply MIN_TRANSFER_VALUE threshold check (per gameweek basis)
        best_scenario = self._apply_value_threshold(
            best_scenario, baseline_scenario
        )
        
        # Store the best scenario for later access
        self._last_best_scenario = best_scenario
        
        # Extract and display final solution
        return self._extract_final_solution(
            best_scenario, free_transfers, df
        )
    
    def _get_baseline_scenario(self, scenarios: list) -> dict:
        """Get baseline scenario for comparison."""
        # Try to find scenario with minimum actual transfers
        if not scenarios:
            return None
        
        min_actual_transfers = min(s['actual_transfers'] for s in scenarios)
        return next(
            (s for s in scenarios if s[
                'actual_transfers'] == min_actual_transfers), 
            None
        )
    
    def _print_top_scenarios(self, scenarios: list, best_scenario: dict):
        """Print top 3 scenarios for comparison showing ppgw as points."""
        top_scenarios = sorted(scenarios, key=lambda x: x['net_ppgw'], 
                             reverse=True)[:3]
        for i, scenario in enumerate(top_scenarios, 1):
            status = " BEST" if scenario == best_scenario else ""
            print(f"   #{i}: {scenario['actual_transfers']} transfers → "
                  f"Net: {scenario['net_ppgw']:.1f} points{status}")
    
    def _apply_value_threshold(self, best_scenario: dict, 
                             baseline_scenario: dict) -> dict:
        """Apply MIN_TRANSFER_VALUE threshold check using ppgw directly."""
        improvement_ppgw = 0  # Default value
        
        if (best_scenario['actual_transfers'] > 
            baseline_scenario['actual_transfers']):
            baseline_ppgw = baseline_scenario['net_ppgw']
            best_ppgw = best_scenario['net_ppgw']
            improvement_ppgw = best_ppgw - baseline_ppgw
            extra_transfers_for_improvement = (
                best_scenario['actual_transfers'] - 
                baseline_scenario['actual_transfers']
            )
            
            # Use MIN_TRANSFER_VALUE directly (already per-gameweek)
            threshold_ppgw = self.config.MIN_TRANSFER_VALUE
            
            if self.config.GRANULAR_OUTPUT:
                print(f"\nTransfer Value Check (per gameweek basis):")
                print(f"   Baseline ({baseline_scenario['actual_transfers']} "
                      f"transfers): {baseline_ppgw:.1f} points")
                print(f"   Best scenario ({best_scenario['actual_transfers']} "
                      f"transfers): {best_ppgw:.1f} points")
                print(
                    f"   Improvement: {improvement_ppgw:.1f} points per gameweek")
                print(f"   Extra transfers for improvement: "
                      f"{extra_transfers_for_improvement}")
                print(f"   Improvement per extra transfer: "
                      f"{improvement_ppgw/extra_transfers_for_improvement:.1f} "
                      f"points per gameweek")
                print(f"   Minimum threshold: {threshold_ppgw:.1f} "
                      f"points per gameweek")
            
            required_improvement_ppgw = (
                threshold_ppgw * extra_transfers_for_improvement
            )
            
            if improvement_ppgw < required_improvement_ppgw:
                if self.config.GRANULAR_OUTPUT:
                    print(f"   INSUFFICIENT VALUE GAINED: Using baseline "
                          f"({baseline_scenario['actual_transfers']} transfers) "
                          "instead")
                best_scenario = baseline_scenario
                if self.config.GRANULAR_OUTPUT:
                    print(f"   SELECTED: {best_scenario['actual_transfers']} "
                          f"transfers → {best_scenario['net_ppgw']:.1f} points")
            else:
                if self.config.GRANULAR_OUTPUT:
                    print(f"   SUFFICIENT VALUE: Extra transfers worthwhile")
                    print(f"   SELECTED: {best_scenario['actual_transfers']} "
                          f"transfers → {best_scenario['net_ppgw']:.1f} points")
        else:
            if self.config.GRANULAR_OUTPUT:
                print(f"\nUsing optimal scenario with "
                      f"{best_scenario['actual_transfers']} transfers")
                print(f"   SELECTED: {best_scenario['actual_transfers']} "
                      f"transfers → {best_scenario['net_ppgw']:.1f} points")
        
        # Store the improvement data in the scenario for later use
        best_scenario['points_improvement_ppgw'] = improvement_ppgw
        best_scenario['gameweeks_analysed'] = self.config.FIRST_N_GAMEWEEKS
        
        return best_scenario
    
    def _extract_final_solution(self, best_scenario: dict, 
                              free_transfers: int, df: pd.DataFrame) -> tuple:
        """Extract and display the final solution."""
        starting = best_scenario['starting']
        bench = best_scenario['bench']
        best_transfers = best_scenario['actual_transfers']
        best_penalty = best_scenario['penalty_points']
        best_forced_display = best_scenario['forced_display']
        
        # Show final transfer summary
        if self.config.GRANULAR_OUTPUT:
            self._print_final_transfer_summary(
                best_transfers, free_transfers, best_penalty, best_scenario, df
            )
        
        return (starting, bench, best_forced_display, 
                best_transfers, best_penalty)
    
    def _print_final_transfer_summary(self, best_transfers: int, 
                                    free_transfers: int, best_penalty: int,
                                    best_scenario: dict, df: pd.DataFrame):
        """Print final transfer summary showing ppgw as points."""
        if best_transfers > 0:
            print(f"\n=== Optimal Transfer Strategy ===")
            print(f"Total transfers: {best_transfers}")
            print(f"Free transfers: {free_transfers}")
            if best_penalty > 0:
                print(f"Extra transfers: {best_transfers - free_transfers}")
                print(f"Transfer penalty: -{best_penalty} points")
            print(f"Projected points: "
                  f"{best_scenario['starting_ppgw']:.1f}")
            print(f"Net points: "
                  f"{best_scenario['net_ppgw']:.1f}")
        else:
            print(f"\n=== Optimal Transfer Strategy ===")
            print(f"Total transfers: 0")
            print(f"Recommended action: Keep current squad")
            print(f"Reason: No transfers needed")
    
    def evaluate_transfer_value(
        self, no_transfer_squad: pd.DataFrame, transfer_squad: pd.DataFrame, 
        transfers_made: int
    ) -> tuple:
        """
        Evaluate whether the transfers provide sufficient projected 
        points improvement using per-gameweek analysis.

        Args:
            no_transfer_squad (pd.DataFrame): Starting XI with no transfers.
            transfer_squad (pd.DataFrame): Starting XI with transfers made.
            transfers_made (int): Number of transfers that would be made.

        Returns:
            tuple: (should_make_transfers, value_analysis_dict)
        """
        if transfers_made == 0:
            return True, {"reason": "No transfers needed", "improvement": 0}

        # Calculate per-gameweek projected points improvement
        no_transfer_points = no_transfer_squad["projected_points"].sum()
        transfer_points = transfer_squad["projected_points"].sum()
        
        no_transfer_ppgw = no_transfer_points / self.config.FIRST_N_GAMEWEEKS
        transfer_ppgw = transfer_points / self.config.FIRST_N_GAMEWEEKS
        points_improvement_ppgw = transfer_ppgw - no_transfer_ppgw

        # Use MIN_TRANSFER_VALUE directly (already per-gameweek)
        threshold_ppgw = self.config.MIN_TRANSFER_VALUE
        min_improvement_needed_ppgw = threshold_ppgw * transfers_made
        
        analysis = {
            "transfers_made": transfers_made,
            "no_transfer_ppgw": no_transfer_ppgw,
            "transfer_ppgw": transfer_ppgw,
            "points_improvement_ppgw": points_improvement_ppgw,
            "improvement_per_transfer_ppgw": (
                points_improvement_ppgw / transfers_made
                if transfers_made > 0 else 0
            ),
            "min_improvement_needed_ppgw": min_improvement_needed_ppgw,
            "threshold_per_transfer_ppgw": threshold_ppgw,
            "gameweeks_analysed": self.config.FIRST_N_GAMEWEEKS,
        }

        # Decision logic based on per-gameweek projected points improvement
        if points_improvement_ppgw < 0:
            return False, {
                **analysis, 
                "reason": "Transfers would decrease projected points per "
                         "gameweek"
            }

        if points_improvement_ppgw < min_improvement_needed_ppgw:
            return False, {
                **analysis,
                "reason": (f"Improvement ({points_improvement_ppgw:.1f} ppgw) "
                          f"below threshold ({min_improvement_needed_ppgw:.1f}"
                          f" ppgw)")
            }

        return True, {
            **analysis, 
            "reason": f"Transfers provide sufficient improvement "
            f"({points_improvement_ppgw:.1f} ppgw)"
        }
    
    def get_no_transfer_squad(self, df: pd.DataFrame, 
                            prev_squad_ids: list) -> pd.DataFrame:
        """
        Get the optimal starting XI using only players from the previous 
        gameweek (no transfers).

        Args:
            df (pd.DataFrame): Current player database with scores.
            prev_squad_ids (list): Player IDs from previous squad.

        Returns:
            pd.DataFrame: Best starting XI using only previous players.
        """
        if prev_squad_ids is None:
            return pd.DataFrame()

        # Filter to only previous squad players that are still available
        id_to_index = {df.iloc[i]["id"]: i for i in range(len(df))}
        available_prev_players = [
            pid for pid in prev_squad_ids if pid in id_to_index
        ]

        if len(available_prev_players) < 15:
            if self.config.GRANULAR_OUTPUT:
                print(f"Warning: Only {len(available_prev_players)} "
                      f"previous players available")
            return pd.DataFrame()

        prev_squad_df = df[df["id"].isin(available_prev_players)].copy()

        # Simple optimisation for starting XI using projected_points
        return self._optimise_starting_xi_from_squad(prev_squad_df)
    
    def _optimise_starting_xi_from_squad(
            self, prev_squad_df: pd.DataFrame) -> pd.DataFrame:
        """Optimise starting XI from a given squad using ILP."""
        n = len(prev_squad_df)
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]

        prob = pulp.LpProblem("No_Transfer_Squad", pulp.LpMaximize)
        prob += pulp.lpSum(
            y[i] * prev_squad_df.iloc[i]["projected_points"] 
            for i in range(n)
        )

        # Starting XI constraints
        prob += pulp.lpSum(y[i] for i in range(n)) == 11
        
        # Position constraints
        position_constraints = [
            ("GK", 1, 1),
            ("DEF", 3, 5),
            ("MID", 3, 5),
            ("FWD", 1, 3)
        ]
        
        for pos, min_count, max_count in position_constraints:
            pos_sum = pulp.lpSum(
                y[i] for i in range(n) 
                if prev_squad_df.iloc[i]["position"] == pos
            )
            prob += pos_sum >= min_count
            prob += pos_sum <= max_count

        # Form constraint - only players with form > 0 can be in starting XI
        for i in range(n):
            if prev_squad_df.iloc[i]["form"] <= 0:
                prob += y[i] == 0

        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if status != pulp.LpStatusOptimal:
            return pd.DataFrame()

        starting_mask = [pulp.value(y[i]) == 1 for i in range(n)]
        return prev_squad_df.iloc[starting_mask].copy()
    
    def get_no_transfer_squad_optimised(self, scored: pd.DataFrame, 
                                      prev_squad_ids: list) -> tuple:
        """
        Get optimised starting XI and bench using only previous squad 
        players.

        Args:
            scored (pd.DataFrame): Current player database with scores.
            prev_squad_ids (list): Player IDs from previous squad.

        Returns:
            tuple: (starting_xi_dataframe, bench_dataframe)
        """
        full_no_transfer_squad = scored[
            scored["id"].isin(prev_squad_ids)].copy()
        starting = self.get_no_transfer_squad(scored, prev_squad_ids)
        bench = full_no_transfer_squad[
            ~full_no_transfer_squad["id"].isin(starting["id"])
        ].copy()

        # Order bench properly: GK first, then by descending projected points
        gk_bench = bench[bench["position"] == "GK"].copy()
        non_gk_bench = bench[bench["position"] != "GK"].copy()
        non_gk_bench = non_gk_bench.sort_values(
            "projected_points", ascending=False
        )
        bench = pd.concat([gk_bench, non_gk_bench], ignore_index=True)
        
        return starting, bench
    
    def evaluate_transfer_strategy(
        self, scored: pd.DataFrame, prev_squad_ids: list, 
        starting_with_transfers: pd.DataFrame, transfers_made: int, 
        free_transfers: int, wildcard_active: bool
    ) -> tuple:
        """
        Evaluate overall transfer strategy and return recommendation.

        Args:
            scored (pd.DataFrame): Current player database with scores.
            prev_squad_ids (list): Player IDs from previous squad.
            starting_with_transfers (pd.DataFrame): Starting XI with transfers.
            transfers_made (int): Number of transfers that would be made.
            free_transfers (int): Number of free transfers available.
            wildcard_active (bool): Whether wildcard is active.

        Returns:
            tuple: (should_make_transfers, transfer_analysis)
        """
        # If wildcard is active, skip transfer value analysis
        if wildcard_active:
            if self.config.GRANULAR_OUTPUT:
                print(f"\nWILDCARD ACTIVE: Making {transfers_made} changes "
                      f"without constraints")
            return True, {"reason": "Wildcard active - no transfer limits"}
        
        if self.config.ACCEPT_TRANSFER_PENALTY:
            extra_transfers = max(0, transfers_made - free_transfers)
            penalty_points = extra_transfers * 4
            
            if self.config.GRANULAR_OUTPUT:
                print(f"\nTRANSFER PENALTY MODE: Making {transfers_made} "
                      "transfers")
                if penalty_points > 0:
                    print(f"   Transfer penalty: -{penalty_points} points "
                          f"(already factored into optimisation)")
            
            # Check if we have stored improvement data from penalty analysis
            if hasattr(self, '_last_best_scenario'):
                scenario = self._last_best_scenario
                return True, {
                    "reason": ("Transfers not worth it"),
                    "points_improvement_ppgw": scenario.get(
                    'points_improvement_ppgw', 0),
                    "gameweeks_analysed": scenario.get(
                    'gameweeks_analysed', self.config.FIRST_N_GAMEWEEKS)
                }
            
            return True, {"reason": "Transfer penalty mode - transfers "
                          "already optimised"}
        
        # If we have previous squad, evaluate whether transfers are worth it
        elif prev_squad_ids is not None and transfers_made > 0:
            if self.config.GRANULAR_OUTPUT:
                print(f"\nEvaluating transfer value...")

            # Get best squad with no transfers for comparison
            no_transfer_starting = self.get_no_transfer_squad(
                scored, prev_squad_ids)

            if not no_transfer_starting.empty:
                (should_make_transfers,
                transfer_analysis) = self.evaluate_transfer_value(
                    no_transfer_starting,  # Starting XI with no transfers
                    starting_with_transfers,  # Starting XI with transfers
                    transfers_made
                )

                if self.config.GRANULAR_OUTPUT:
                    self._print_transfer_value_analysis(transfer_analysis)
                return should_make_transfers, transfer_analysis
        
        return True, {}
    
    def _print_transfer_value_analysis(self, transfer_analysis: dict):
        """Print transfer value analysis results using per-gameweek metrics."""
        print(f"\n=== Transfer Value Analysis ===")
        print(f"Transfers to be made: {transfer_analysis['transfers_made']}")
        print(f"Gameweeks analysed: "
              f"{transfer_analysis.get('gameweeks_analysed', 'N/A')}")
        print(f"No-transfer points per gameweek: "
              f"{transfer_analysis['no_transfer_ppgw']:.1f}")
        print(f"With-transfer points per gameweek: "
              f"{transfer_analysis['transfer_ppgw']:.1f}")
        print(f"Points improvement per gameweek: "
              f"{transfer_analysis['points_improvement_ppgw']:.1f}")
        print(f"Improvement per transfer (ppgw): "
              f"{transfer_analysis['improvement_per_transfer_ppgw']:.1f}")
        print(f"Threshold per transfer (ppgw): "
              f"{transfer_analysis['threshold_per_transfer_ppgw']:.1f}")
        print(f"Minimum improvement needed (ppgw): "
              f"{transfer_analysis['min_improvement_needed_ppgw']:.1f}")
        print(f"Decision: {transfer_analysis['reason']}")