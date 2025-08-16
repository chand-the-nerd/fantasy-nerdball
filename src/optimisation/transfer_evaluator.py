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
                        "unavailable_score": unavailable_player["fpl_score"],
                        "best_substitute": None,
                        "substitute_score": 0,
                        "score_loss": unavailable_player["fpl_score"],
                        "recommendation": "transfer",
                    }
                )
                continue

            # Find best substitute (highest scoring available player)
            best_sub = potential_subs.loc[potential_subs["fpl_score"].idxmax()]
            score_loss = unavailable_player["fpl_score"] - best_sub["fpl_score"]

            substitute_scenarios.append(
                {
                    "unavailable_player": unavailable_player["display_name"],
                    "position": pos,
                    "unavailable_score": unavailable_player["fpl_score"],
                    "best_substitute": best_sub["display_name"],
                    "substitute_score": best_sub["fpl_score"],
                    "score_loss": score_loss,
                    "recommendation": (
                        "substitute" if score_loss < 0.5 else "consider_transfer"
                    ),
                }
            )

        # Calculate total impact of substitutions
        total_score_loss = sum(scenario["score_loss"] for scenario in substitute_scenarios)
        forced_transfers = len(
            [s for s in substitute_scenarios if s["best_substitute"] is None]
        )

        # Decision logic
        if forced_transfers > free_transfers:
            decision = {
                "recommendation": "wildcard_needed",
                "reason": f"Need {forced_transfers} forced transfers but only have {free_transfers} free",
                "total_score_loss": total_score_loss,
                "scenarios": substitute_scenarios,
            }
        elif total_score_loss > free_transfers * 0.3:  # If score loss > transfer threshold
            decision = {
                "recommendation": "make_transfers",
                "reason": f"Score loss ({total_score_loss:.2f}) justifies using {min(len(substitute_scenarios), free_transfers)} transfers",
                "total_score_loss": total_score_loss,
                "scenarios": substitute_scenarios,
            }
        else:
            decision = {
                "recommendation": "use_substitutes",
                "reason": f"Score loss ({total_score_loss:.2f}) is acceptable, save transfers",
                "total_score_loss": total_score_loss,
                "scenarios": substitute_scenarios,
            }

        # Print analysis
        print(f"\nSubstitution scenarios:")
        for scenario in substitute_scenarios:
            if scenario["best_substitute"]:
                print(
                    f"  {scenario['unavailable_player']} → {scenario['best_substitute']} "
                    f"(score loss: {scenario['score_loss']:.2f})"
                )
            else:
                print(
                    f"  {scenario['unavailable_player']} → NO SUBSTITUTE AVAILABLE (must transfer)"
                )

        print(f"\nTotal score loss from substitutions: {total_score_loss:.2f}")
        print(f"Recommendation: {decision['recommendation'].upper()}")
        print(f"Reason: {decision['reason']}")

        return decision
    
    def evaluate_transfer_value(
        self, current_squad: pd.DataFrame, potential_squad: pd.DataFrame, transfers_made: int, free_transfers: int
    ) -> tuple:
        """
        Evaluate whether the transfers provide sufficient value to justify making them.

        Args:
            current_squad (pd.DataFrame): Current squad if no transfers were made.
            potential_squad (pd.DataFrame): Potential new squad with transfers.
            transfers_made (int): Number of transfers that would be made.
            free_transfers (int): Number of free transfers available.

        Returns:
            tuple: (should_make_transfers, value_analysis_dict)
        """
        if transfers_made == 0:
            return True, {"reason": "No transfers needed", "value": 0}

        # Calculate score improvement
        current_score = current_squad["fpl_score"].sum()
        new_score = potential_squad["fpl_score"].sum()
        score_improvement = new_score - current_score

        # Calculate value per transfer
        value_per_transfer = score_improvement / transfers_made if transfers_made > 0 else 0

        # Calculate opportunity cost of using transfers
        # Each unused transfer has value for future weeks
        transfers_remaining_after = free_transfers - transfers_made
        rollover_value_lost = min(transfers_made, free_transfers) * self.config.TRANSFER_ROLLOVER_VALUE

        # Adjust for conservative mode
        min_threshold = self.config.MIN_TRANSFER_VALUE
        if self.config.CONSERVATIVE_MODE:
            min_threshold *= 1.5  # Be more cautious

        # Net value considering rollover opportunity cost
        net_value = score_improvement - rollover_value_lost

        analysis = {
            "transfers_made": transfers_made,
            "score_improvement": score_improvement,
            "value_per_transfer": value_per_transfer,
            "min_threshold": min_threshold,
            "rollover_value_lost": rollover_value_lost,
            "net_value": net_value,
            "transfers_remaining": transfers_remaining_after,
        }

        # Decision logic
        if score_improvement < 0:
            return False, {**analysis, "reason": "Transfers would decrease team value"}

        if value_per_transfer < min_threshold:
            return False, {
                **analysis,
                "reason": f"Value per transfer ({value_per_transfer:.3f}) below threshold ({min_threshold:.3f})",
            }

        if net_value < 0 and free_transfers > 1:
            return False, {
                **analysis,
                "reason": "Better to save transfers for future weeks",
            }

        # Special case: If we have many transfers (4+), be more willing to use some
        if free_transfers >= 4:
            min_threshold *= 0.7  # Lower threshold when we have many transfers
            if value_per_transfer >= min_threshold:
                return True, {
                    **analysis,
                    "reason": "Many transfers available, using some is beneficial",
                }

        # Must use transfers if we're at the cap (5)
        if free_transfers >= 5:
            return True, {**analysis, "reason": "At transfer cap, must use to avoid waste"}

        return True, {**analysis, "reason": "Transfers provide sufficient value"}
    
    def get_no_transfer_squad(self, df: pd.DataFrame, prev_squad_ids: list) -> pd.DataFrame:
        """
        Get the optimal squad using only players from the previous gameweek (no transfers).

        Args:
            df (pd.DataFrame): Current player database with scores.
            prev_squad_ids (list): Player IDs from previous squad.

        Returns:
            pd.DataFrame: Best possible squad using only previous players.
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

        # Simple optimisation for starting XI from these 15 players
        n = len(prev_squad_df)
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]

        prob = pulp.LpProblem("No_Transfer_Squad", pulp.LpMaximize)
        prob += pulp.lpSum(y[i] * prev_squad_df.iloc[i]["fpl_score"] for i in range(n))

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
        non_gk_bench = non_gk_bench.sort_values("fpl_score", ascending=False)
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
        
        # If we have previous squad, evaluate whether transfers are worth it
        elif prev_squad_ids is not None and transfers_made > 0:
            print(f"\nEvaluating transfer value...")

            # Get best squad with no transfers for comparison
            no_transfer_starting = self.get_no_transfer_squad(scored, prev_squad_ids)

            if not no_transfer_starting.empty:
                should_make_transfers, transfer_analysis = self.evaluate_transfer_value(
                    no_transfer_starting,  # Starting XI with no transfers
                    starting_with_transfers,  # Starting XI with transfers
                    transfers_made,
                    free_transfers,
                )

                print(f"\n=== Transfer Value Analysis ===")
                print(f"Transfers to be made: {transfer_analysis['transfers_made']}")
                print(f"Score improvement: {transfer_analysis['score_improvement']:.3f}")
                print(f"Value per transfer: {transfer_analysis['value_per_transfer']:.3f}")
                print(f"Minimum threshold: {transfer_analysis['min_threshold']:.3f}")
                print(f"Rollover value lost: {transfer_analysis['rollover_value_lost']:.3f}")
                print(f"Net value: {transfer_analysis['net_value']:.3f}")
                print(f"Decision: {transfer_analysis['reason']}")
                
                return should_make_transfers, transfer_analysis
        
        return True, {}