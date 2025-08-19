"""Module for evaluating transfer strategies and alternatives."""

import pandas as pd
import pulp


class TransferEvaluator:
    """Handles evaluation of transfer strategies and alternatives."""
    
    def __init__(self, config):
        self.config = config
    
    def get_unavailable_players(self, df: pd.DataFrame, prev_squad_ids: list) -> list:
        """
        Identify players from previous squad who are unavailable for this gameweek.

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
    
    def evaluate_substitute_vs_transfer(
        self, df: pd.DataFrame, prev_squad_ids: list, unavailable_player_ids: list, free_transfers: int
    ) -> dict:
        """
        Evaluate whether to substitute unavailable players or transfer them out.

        Args:
            df (pd.DataFrame): Current player database with scores
            prev_squad_ids (list): Player IDs from previous squad
            unavailable_player_ids (list): IDs of players who can't play this GW
            free_transfers (int): Number of free transfers available

        Returns:
            dict: Analysis of substitute vs transfer options
        """
        if not prev_squad_ids or not unavailable_player_ids:
            return {
                "recommendation": "no_action",
                "reason": "No unavailable players or no previous squad",
            }

        # Get previous squad dataframe
        prev_squad_df = df[df["id"].isin(prev_squad_ids)].copy()
        unavailable_df = prev_squad_df[prev_squad_df["id"].isin(unavailable_player_ids)]
        available_df = prev_squad_df[~prev_squad_df["id"].isin(unavailable_player_ids)]

        print(f"\n=== Substitute vs Transfer Analysis ===")
        print(f"Unavailable players: {len(unavailable_df)}")
        for _, player in unavailable_df.iterrows():
            print(f"  - {player['display_name']} ({player['position']}, {player['team']})")

        substitute_scenarios = []

        for _, unavailable_player in unavailable_df.iterrows():
            pos = unavailable_player["position"]

            # Find potential substitutes from bench (same squad, available, different position allowed for bench)
            potential_subs = available_df[
                (available_df["id"] != unavailable_player["id"])
                & (available_df["status"] == "a")
            ].copy()

            if len(potential_subs) == 0:
                substitute_scenarios.append(
                    {
                        "unavailable_player": unavailable_player["display_name"],
                        "position": pos,
                        "unavailable_score": unavailable_player["projected_points"],
                        "best_substitute": None,
                        "substitute_score": 0,
                        "score_loss": unavailable_player["projected_points"],
                        "recommendation": "transfer",
                    }
                )
                continue

            # Find best substitute (highest projected points available player)
            best_sub = potential_subs.loc[potential_subs["projected_points"].idxmax()]
            score_loss = unavailable_player["projected_points"] - best_sub["projected_points"]

            substitute_scenarios.append(
                {
                    "unavailable_player": unavailable_player["display_name"],
                    "position": pos,
                    "unavailable_score": unavailable_player["projected_points"],
                    "best_substitute": best_sub["display_name"],
                    "substitute_score": best_sub["projected_points"],
                    "score_loss": score_loss,
                    "recommendation": (
                        "substitute" if score_loss < 2.0 else "consider_transfer"
                    ),
                }
            )

        # Calculate total impact of substitutions
        total_score_loss = sum(scenario["score_loss"] for scenario in substitute_scenarios)
        forced_transfers = len(
            [s for s in substitute_scenarios if s["best_substitute"] is None]
        )

        # Decision logic based on projected points
        if forced_transfers > free_transfers:
            decision = {
                "recommendation": "wildcard_needed",
                "reason": f"Need {forced_transfers} forced transfers but only have {free_transfers} free",
                "total_score_loss": total_score_loss,
                "scenarios": substitute_scenarios,
            }
        elif total_score_loss > self.config.MIN_TRANSFER_VALUE:  # Use MIN_TRANSFER_VALUE threshold
            decision = {
                "recommendation": "make_transfers",
                "reason": f"Score loss ({total_score_loss:.1f}) exceeds transfer threshold ({self.config.MIN_TRANSFER_VALUE})",
                "total_score_loss": total_score_loss,
                "scenarios": substitute_scenarios,
            }
        else:
            decision = {
                "recommendation": "use_substitutes",
                "reason": f"Score loss ({total_score_loss:.1f}) is acceptable, save transfers",
                "total_score_loss": total_score_loss,
                "scenarios": substitute_scenarios,
            }

        # Print analysis
        print(f"\nSubstitution scenarios:")
        for scenario in substitute_scenarios:
            if scenario["best_substitute"]:
                print(
                    f"  {scenario['unavailable_player']} → {scenario['best_substitute']} "
                    f"(score loss: {scenario['score_loss']:.1f} pts)"
                )
            else:
                print(
                    f"  {scenario['unavailable_player']} → NO SUBSTITUTE AVAILABLE (must transfer)"
                )

        print(f"\nTotal score loss from substitutions: {total_score_loss:.1f} pts")
        print("(note that negative score loss is good)")
        print(f"Transfer threshold: {self.config.MIN_TRANSFER_VALUE} pts")
        print(f"Recommendation: {decision['recommendation'].upper()}")
        print(f"Reason: {decision['reason']}")

        return decision
    
    def get_optimal_squad_with_penalties(
        self, df: pd.DataFrame, forced_selections: dict, prev_squad_ids: list, 
        free_transfers: int, available_budget: float, squad_selector
    ) -> tuple:
        """
        Get optimal squad considering transfer penalties when ACCEPT_TRANSFER_PENALTY is True.
        
        Args:
            df (pd.DataFrame): Current player database with scores
            forced_selections (dict): Dictionary of forced player selections
            prev_squad_ids (list): Player IDs from previous squad
            free_transfers (int): Number of free transfers available
            available_budget (float): Available budget for squad
            squad_selector: SquadSelector instance
            
        Returns:
            tuple: (starting_xi, bench, forced_selections_display, transfers_made, penalty_points)
        """
        if not self.config.ACCEPT_TRANSFER_PENALTY or prev_squad_ids is None:
            # Use normal squad selection without penalty consideration
            starting, bench, forced_display = squad_selector.select_squad_ilp(
                df, forced_selections, prev_squad_ids, free_transfers, 
                show_transfer_summary=True, available_budget=available_budget
            )
            return starting, bench, forced_display, 0, 0
        
        print(f"\n=== TRANSFER ANALYSIS ===")
        print(f"\n🔄 Evaluating all transfer scenarios up to {free_transfers + 3} transfers...")
        
        scenarios = []
        
        # Test different transfer scenarios - always test up to free_transfers + 3
        max_transfers_to_test = free_transfers + 3
        
        for max_transfers_allowed in range(0, max_transfers_to_test + 1):
            # Get optimal squad with this transfer limit
            starting, bench, forced_display = squad_selector.select_squad_ilp(
                df, forced_selections, prev_squad_ids, max_transfers_allowed,
                show_transfer_summary=False, available_budget=available_budget
            )
            
            if starting.empty:
                continue
                
            # Calculate actual transfers made
            current_squad_ids = set(pd.concat([starting, bench])["id"].tolist())
            prev_squad_ids_set = set(prev_squad_ids)
            actual_transfers = len(prev_squad_ids_set - current_squad_ids)
            
            # Calculate penalty
            extra_transfers = max(0, actual_transfers - free_transfers)
            penalty_points = extra_transfers * 4
            
            # Calculate net score (projected points for starting XI minus penalty)
            starting_points = starting["projected_points"].sum()
            net_score = starting_points - penalty_points
            
            # Get transfer details for logging
            players_out = prev_squad_ids_set - current_squad_ids
            players_in = current_squad_ids - prev_squad_ids_set
            
            # Format transfer details
            transfer_details = ""
            if actual_transfers > 0:
                out_names = []
                for player_id in players_out:
                    prev_player = df[df["id"] == player_id]
                    if not prev_player.empty:
                        out_names.append(prev_player.iloc[0]["display_name"])
                
                in_names = []
                for player_id in players_in:
                    player = pd.concat([starting, bench])[pd.concat([starting, bench])["id"] == player_id].iloc[0]
                    in_names.append(player["display_name"])
                
                if out_names and in_names:
                    transfer_details = f" (🔄 OUT: {', '.join(out_names)} → IN: {', '.join(in_names)})"
            
            # Store this scenario
            scenario = {
                'max_transfers_allowed': max_transfers_allowed,
                'actual_transfers': actual_transfers,
                'extra_transfers': extra_transfers,
                'penalty_points': penalty_points,
                'starting_points': starting_points,
                'net_score': net_score,
                'starting': starting,
                'bench': bench,
                'forced_display': forced_display,
                'transfer_details': transfer_details
            }
            scenarios.append(scenario)
            
            # Enhanced logging with emojis and transfer details
            if actual_transfers == 0:
                print(f"  🏠 Scenario {max_transfers_allowed}: {actual_transfers} transfers, "
                      f"{extra_transfers} extra, penalty: -{penalty_points}, "
                      f"gross: {starting_points:.1f}, net: {net_score:.1f}")
            elif extra_transfers == 0:
                print(f"  ✅ Scenario {max_transfers_allowed}: {actual_transfers} transfers, "
                      f"{extra_transfers} extra, penalty: -{penalty_points}, "
                      f"gross: {starting_points:.1f}, net: {net_score:.1f}{transfer_details}")
            else:
                print(f"  💰 Scenario {max_transfers_allowed}: {actual_transfers} transfers, "
                      f"{extra_transfers} extra, penalty: -{penalty_points}, "
                      f"gross: {starting_points:.1f}, net: {net_score:.1f}{transfer_details}")
        
        if not scenarios:
            print("❌ No valid scenarios found")
            return pd.DataFrame(), pd.DataFrame(), None, 0, 0
        
        # Find the best scenario by net score
        best_scenario = max(scenarios, key=lambda x: x['net_score'])
        
        # Get the 0-transfer baseline for MIN_TRANSFER_VALUE comparison
        baseline_scenario = next((s for s in scenarios if s['actual_transfers'] == 0), None)
        
        if baseline_scenario is None:
            print("❌ No baseline (0 transfers) scenario found")
            return pd.DataFrame(), pd.DataFrame(), None, 0, 0
        
        print(f"\n🎯 Best scenario analysis:")
        print(f"   Tested {len(scenarios)} different transfer limits")
        
        # Show top 3 scenarios for comparison
        top_scenarios = sorted(scenarios, key=lambda x: x['net_score'], reverse=True)[:3]
        for i, scenario in enumerate(top_scenarios, 1):
            status = " ⭐ BEST" if scenario == best_scenario else ""
            print(f"   #{i}: {scenario['actual_transfers']} transfers → "
                  f"Net: {scenario['net_score']:.1f} points{status}")
        
        # Apply MIN_TRANSFER_VALUE threshold check
        if best_scenario['actual_transfers'] > 0:
            baseline_score = baseline_scenario['net_score']
            best_score = best_scenario['net_score']
            improvement = best_score - baseline_score
            
            print(f"\n📊 Transfer Value Check:")
            print(f"   Baseline (0 transfers): {baseline_score:.1f} points")
            print(f"   Best scenario ({best_scenario['actual_transfers']} transfers): {best_score:.1f} points")
            print(f"   Improvement: {improvement:.1f} points")
            print(f"   Minimum threshold: {self.config.MIN_TRANSFER_VALUE} points")
            
            if improvement < self.config.MIN_TRANSFER_VALUE:
                print(f"   ❌ INSUFFICIENT VALUE GAINED: Using baseline (0 transfers) instead")
                best_scenario = baseline_scenario
                print(f"   ⭐ SELECTED: 0 transfers → {best_scenario['net_score']:.1f} points")
            else:
                print(f"   ✅ SUFFICIENT VALUE: Transfers are worthwhile")
                print(f"   ⭐ SELECTED: {best_scenario['actual_transfers']} transfers → {best_scenario['net_score']:.1f} points")
        
        # Extract best solution (after MIN_TRANSFER_VALUE check)
        starting = best_scenario['starting']
        bench = best_scenario['bench']
        best_transfers = best_scenario['actual_transfers']
        best_penalty = best_scenario['penalty_points']
        best_forced_display = best_scenario['forced_display']
        best_net_score = best_scenario['net_score']
        
        # Show final transfer summary
        if best_transfers > 0:
            current_squad_ids = set(pd.concat([starting, bench])["id"].tolist())
            prev_squad_ids_set = set(prev_squad_ids)
            players_out = prev_squad_ids_set - current_squad_ids
            players_in = current_squad_ids - prev_squad_ids_set
            
            print(f"\n=== Optimal Transfer Strategy ===")
            print(f"Total transfers: {best_transfers}")
            print(f"Free transfers: {free_transfers}")
            if best_penalty > 0:
                print(f"Extra transfers: {best_transfers - free_transfers}")
                print(f"Transfer penalty: -{best_penalty} points")
            print(f"Gross projected points: {best_scenario['starting_points']:.1f}")
            print(f"Net projected points: {best_net_score:.1f}")
            
            if players_out:
                print("Players to transfer OUT:")
                for player_id in players_out:
                    prev_player = df[df["id"] == player_id]
                    if not prev_player.empty:
                        player_name = prev_player.iloc[0]["display_name"]
                        print(f"  - {player_name}")
            
            if players_in:
                print("Players to transfer IN:")
                for player_id in players_in:
                    player = pd.concat([starting, bench])[pd.concat([starting, bench])["id"] == player_id].iloc[0]
                    print(f"  + {player['display_name']} ({player['position']}, {player['team']})")
        else:
            print(f"\n=== Optimal Transfer Strategy ===")
            print(f"Total transfers: 0")
            print(f"Recommended action: Keep current squad")
            print(f"Reason: Transfer improvements below minimum threshold ({self.config.MIN_TRANSFER_VALUE} pts per transfer)")
        
        return starting, bench, best_forced_display, best_transfers, best_penalty
    
    def evaluate_transfer_value(
        self, no_transfer_squad: pd.DataFrame, transfer_squad: pd.DataFrame, transfers_made: int
    ) -> tuple:
        """
        Evaluate whether the transfers provide sufficient projected points improvement.

        Args:
            no_transfer_squad (pd.DataFrame): Starting XI with no transfers made.
            transfer_squad (pd.DataFrame): Starting XI with transfers made.
            transfers_made (int): Number of transfers that would be made.

        Returns:
            tuple: (should_make_transfers, value_analysis_dict)
        """
        if transfers_made == 0:
            return True, {"reason": "No transfers needed", "improvement": 0}

        # Calculate projected points improvement
        no_transfer_points = no_transfer_squad["projected_points"].sum()
        transfer_points = transfer_squad["projected_points"].sum()
        points_improvement = transfer_points - no_transfer_points

        # Check against minimum transfer value threshold
        min_improvement_needed = self.config.MIN_TRANSFER_VALUE * transfers_made
        
        analysis = {
            "transfers_made": transfers_made,
            "no_transfer_points": no_transfer_points,
            "transfer_points": transfer_points,
            "points_improvement": points_improvement,
            "improvement_per_transfer": points_improvement / transfers_made if transfers_made > 0 else 0,
            "min_improvement_needed": min_improvement_needed,
            "threshold_per_transfer": self.config.MIN_TRANSFER_VALUE,
        }

        # Decision logic based on projected points improvement
        if points_improvement < 0:
            return False, {**analysis, "reason": "Transfers would decrease projected points"}

        if points_improvement < min_improvement_needed:
            return False, {
                **analysis,
                "reason": f"Improvement ({points_improvement:.1f} pts) below threshold ({min_improvement_needed:.1f} pts)"
            }

        return True, {**analysis, "reason": f"Transfers provide sufficient improvement ({points_improvement:.1f} pts)"}
    
    def get_no_transfer_squad(self, df: pd.DataFrame, prev_squad_ids: list) -> pd.DataFrame:
        """
        Get the optimal starting XI using only players from the previous gameweek (no transfers).

        Args:
            df (pd.DataFrame): Current player database with scores.
            prev_squad_ids (list): Player IDs from previous squad.

        Returns:
            pd.DataFrame: Best possible starting XI using only previous players.
        """
        if prev_squad_ids is None:
            return pd.DataFrame()

        # Filter to only previous squad players that are still available
        id_to_index = {df.iloc[i]["id"]: i for i in range(len(df))}
        available_prev_players = [pid for pid in prev_squad_ids if pid in id_to_index]

        if len(available_prev_players) < 15:
            print(f"Warning: Only {len(available_prev_players)} previous players available")
            return pd.DataFrame()

        prev_squad_df = df[df["id"].isin(available_prev_players)].copy()

        # Simple optimisation for starting XI from these 15 players using projected_points
        n = len(prev_squad_df)
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]

        prob = pulp.LpProblem("No_Transfer_Squad", pulp.LpMaximize)
        prob += pulp.lpSum(y[i] * prev_squad_df.iloc[i]["projected_points"] for i in range(n))

        # Starting XI constraints
        prob += pulp.lpSum(y[i] for i in range(n)) == 11
        prob += (
            pulp.lpSum(y[i] for i in range(n) if prev_squad_df.iloc[i]["position"] == "GK")
            == 1
        )
        prob += (
            pulp.lpSum(y[i] for i in range(n) if prev_squad_df.iloc[i]["position"] == "DEF")
            >= 3
        )
        prob += (
            pulp.lpSum(y[i] for i in range(n) if prev_squad_df.iloc[i]["position"] == "DEF")
            <= 5
        )
        prob += (
            pulp.lpSum(y[i] for i in range(n) if prev_squad_df.iloc[i]["position"] == "MID")
            >= 3
        )
        prob += (
            pulp.lpSum(y[i] for i in range(n) if prev_squad_df.iloc[i]["position"] == "MID")
            <= 5
        )
        prob += (
            pulp.lpSum(y[i] for i in range(n) if prev_squad_df.iloc[i]["position"] == "FWD")
            >= 1
        )
        prob += (
            pulp.lpSum(y[i] for i in range(n) if prev_squad_df.iloc[i]["position"] == "FWD")
            <= 3
        )

        # Form constraint - only players with form > 0 can be in starting XI
        for i in range(n):
            if prev_squad_df.iloc[i]["form"] <= 0:
                prob += y[i] == 0

        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if status != pulp.LpStatusOptimal:
            return pd.DataFrame()

        starting_mask = [pulp.value(y[i]) == 1 for i in range(n)]
        starting_xi = prev_squad_df.iloc[starting_mask].copy()

        return starting_xi
    
    def get_no_transfer_squad_optimised(self, scored: pd.DataFrame, prev_squad_ids: list) -> tuple:
        """
        Get optimised starting XI and bench using only previous squad players.

        Args:
            scored (pd.DataFrame): Current player database with scores.
            prev_squad_ids (list): Player IDs from previous squad.

        Returns:
            tuple: (starting_xi_dataframe, bench_dataframe)
        """
        full_no_transfer_squad = scored[scored["id"].isin(prev_squad_ids)].copy()
        starting = self.get_no_transfer_squad(scored, prev_squad_ids)
        bench = full_no_transfer_squad[
            ~full_no_transfer_squad["id"].isin(starting["id"])
        ].copy()

        # Order bench properly
        gk_bench = bench[bench["position"] == "GK"].copy()
        non_gk_bench = bench[bench["position"] != "GK"].copy()
        non_gk_bench = non_gk_bench.sort_values("projected_points", ascending=False)
        bench = pd.concat([gk_bench, non_gk_bench], ignore_index=True)
        
        return starting, bench
    
    def evaluate_transfer_strategy(
        self, scored: pd.DataFrame, prev_squad_ids: list, starting_with_transfers: pd.DataFrame, 
        transfers_made: int, free_transfers: int, wildcard_active: bool
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
            print(f"\n🃏 WILDCARD ACTIVE: Making {transfers_made} changes without constraints")
            return True, {"reason": "Wildcard active - no transfer limits"}
        
        # If transfer penalties are accepted, always make the transfers (already optimized)
        if self.config.ACCEPT_TRANSFER_PENALTY:
            extra_transfers = max(0, transfers_made - free_transfers)
            penalty_points = extra_transfers * 4
            print(f"\n✅ TRANSFER PENALTY MODE: Making {transfers_made} transfers")
            if penalty_points > 0:
                print(f"   Transfer penalty: -{penalty_points} points (already factored into optimization)")
            return True, {"reason": "Transfer penalty mode - transfers already optimized"}
        
        # If we have previous squad, evaluate whether transfers are worth it
        elif prev_squad_ids is not None and transfers_made > 0:
            print(f"\nEvaluating transfer value...")

            # Get best squad with no transfers for comparison
            no_transfer_starting = self.get_no_transfer_squad(scored, prev_squad_ids)

            if not no_transfer_starting.empty:
                should_make_transfers, transfer_analysis = self.evaluate_transfer_value(
                    no_transfer_starting,  # Starting XI with no transfers
                    starting_with_transfers,  # Starting XI with transfers
                    transfers_made
                )

                print(f"\n=== Transfer Value Analysis ===")
                print(f"Transfers to be made: {transfer_analysis['transfers_made']}")
                print(f"No-transfer points: {transfer_analysis['no_transfer_points']:.1f}")
                print(f"With-transfer points: {transfer_analysis['transfer_points']:.1f}")
                print(f"Points improvement: {transfer_analysis['points_improvement']:.1f}")
                print(f"Improvement per transfer: {transfer_analysis['improvement_per_transfer']:.1f}")
                print(f"Threshold per transfer: {transfer_analysis['threshold_per_transfer']:.1f}")
                print(f"Minimum improvement needed: {transfer_analysis['min_improvement_needed']:.1f}")
                print(f"Decision: {transfer_analysis['reason']}")
                
                return should_make_transfers, transfer_analysis
        
        return True, {}