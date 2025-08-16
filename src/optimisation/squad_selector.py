"""Module for squad selection using integer linear programming."""

import pandas as pd
import pulp
from ..utils.text_utils import normalize_for_matching


class SquadSelector:
    """Handles squad selection optimisation using ILP."""
    
    def __init__(self, config):
        self.config = config
    
    def select_squad_ilp(
        self,
        df: pd.DataFrame,
        forced_selections: dict,
        prev_squad_ids: list = None,
        free_transfers: int = None,
        show_transfer_summary: bool = True,
        available_budget: float = None,
    ) -> tuple:
        """
        Select optimal FPL squad using Integer Linear Programming with
        forced player selections and transfer constraints.

        Args:
            df (pd.DataFrame): Player data with fpl_scores and all required fields.
            forced_selections (dict): Dictionary of forced player selections.
            prev_squad_ids (list, optional): List of player IDs from prev squad.
            free_transfers (int, optional): Number of free transfers available.
            show_transfer_summary (bool): Whether to display transfer information.
            available_budget (float, optional): Available budget for squad.

        Returns:
            tuple: (starting_xi_dataframe, bench_dataframe, forced_selections_str)
                   containing the optimal squad selection and info about forced players.
        """
        # Process forced selections before DataFrame modifications
        forced_player_ids = []
        forced_players_info = []

        for pos, players_to_force in forced_selections.items():
            if not players_to_force:
                continue

            for name in players_to_force:
                # Search for player in original dataframe
                for idx, row in df.iterrows():
                    if row["position"] == pos and (
                        row["display_name"].lower() == name.lower()
                        or normalize_for_matching(row["display_name"])
                        == normalize_for_matching(name)
                    ):
                        forced_player_ids.append(row["id"])
                        forced_players_info.append(
                            f"{row['display_name']} ({row['position']}, {row['team']})"
                        )
                        break

        # Store forced selections info for later display
        forced_selections_display = (
            ", ".join(forced_players_info) if forced_players_info else None
        )

        # Clean the dataframe - remove the availability filter since we handle this with projected points
        df = df.reset_index(drop=True).drop_duplicates(subset="name_key")
        df = df[~df["display_name"].str.lower().isin(self.config.BLACKLIST_PLAYERS)].copy()
        n = len(df)

        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]

        prob = pulp.LpProblem("FPL_Squad_Selection", pulp.LpMaximize)
        prob += pulp.lpSum(
            y[i] * df.iloc[i]["fpl_score"] + 0.2 * (x[i] - y[i]) * df.iloc[i]["fpl_score"]
            for i in range(n)
        )
        prob += pulp.lpSum(x[i] for i in range(n)) == 15  # Total squad
        prob += pulp.lpSum(y[i] for i in range(n)) == 11  # Starting XI
        for i in range(n):
            prob += y[i] <= x[i]  # XI <= squad

        # Position constraints for squad
        for pos, count in self.config.SQUAD_SIZE.items():
            prob += (
                pulp.lpSum(x[i] for i in range(n) if df.iloc[i]["position"] == pos) == count
            )

        # Starting XI position constraints
        prob += pulp.lpSum(y[i] for i in range(n) if df.iloc[i]["position"] == "GK") == 1
        prob += pulp.lpSum(y[i] for i in range(n) if df.iloc[i]["position"] == "DEF") >= 3
        prob += pulp.lpSum(y[i] for i in range(n) if df.iloc[i]["position"] == "DEF") <= 5
        prob += pulp.lpSum(y[i] for i in range(n) if df.iloc[i]["position"] == "MID") >= 3
        prob += pulp.lpSum(y[i] for i in range(n) if df.iloc[i]["position"] == "MID") <= 5
        prob += pulp.lpSum(y[i] for i in range(n) if df.iloc[i]["position"] == "FWD") >= 1
        prob += pulp.lpSum(y[i] for i in range(n) if df.iloc[i]["position"] == "FWD") <= 3

        # Apply forced selections using player IDs
        for player_id in forced_player_ids:
            # Find player in cleaned dataframe by searching through all rows
            for i in range(len(df)):
                if df.iloc[i]["id"] == player_id:
                    prob += x[i] == 1  # Force into squad
                    break

        # Transfer constraint
        if prev_squad_ids is not None and free_transfers is not None and not self.config.WILDCARD:
            print(f"Applying transfer constraint: max {free_transfers} transfers")

            # Create mapping of player IDs to dataframe indices
            id_to_index = {df.iloc[i]["id"]: i for i in range(n)}

            # Count how many previous squad players are kept
            prev_players_kept = pulp.lpSum(
                x[id_to_index[player_id]]
                for player_id in prev_squad_ids
                if player_id in id_to_index
            )

            # Number of transfers = 15 - number of players kept from previous squad
            # This must be <= free_transfers
            # So: 15 - prev_players_kept <= free_transfers
            # Therefore: prev_players_kept >= 15 - free_transfers
            min_players_to_keep = 15 - free_transfers
            prob += prev_players_kept >= min_players_to_keep

            print(f"Must keep at least {min_players_to_keep} players from previous squad")
        elif self.config.WILDCARD and prev_squad_ids is not None:
            print("🃏 WILDCARD ACTIVE: No transfer constraints applied")

        # Bench constraints
        prob += (
            pulp.lpSum(
                (x[i] - y[i])
                for i in range(n)
                if df.iloc[i]["position"] == "GK" and df.iloc[i]["now_cost_m"] == 4.0
            )
            == 1
        )

        # Allow up to 2 of DEF, MID, FWD on bench
        for pos in ["DEF", "MID", "FWD"]:
            prob += (
                pulp.lpSum((x[i] - y[i]) for i in range(n) if df.iloc[i]["position"] == pos)
                <= 2
            )

        # Max per team
        for team in df["team_id"].unique():
            prob += (
                pulp.lpSum(x[i] for i in range(n) if df.iloc[i]["team_id"] == team)
                <= self.config.MAX_PER_TEAM
            )

        # Budget
        budget_to_use = available_budget if available_budget is not None else self.config.BUDGET
        prob += (
            pulp.lpSum(x[i] * df.iloc[i]["now_cost_m"] for i in range(n)) <= budget_to_use
        )

        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if status != pulp.LpStatusOptimal:
            print(f"Optimisation failed with status: {pulp.LpStatus[status]}")
            return pd.DataFrame(), pd.DataFrame(), None

        selected_mask = [pulp.value(x[i]) == 1 for i in range(n)]
        squad = df.iloc[selected_mask].copy()
        squad["starting_XI"] = [pulp.value(y[i]) for i in range(n) if pulp.value(x[i]) == 1]

        # Calculate and display transfer information
        if prev_squad_ids is not None and show_transfer_summary:
            current_squad_ids = set(squad["id"].tolist())
            prev_squad_ids_set = set(prev_squad_ids)

            players_kept = current_squad_ids.intersection(prev_squad_ids_set)
            players_out = prev_squad_ids_set - current_squad_ids
            players_in = current_squad_ids - prev_squad_ids_set

            transfers_made = len(players_out)

            print(f"\n=== Proposed Transfer Summary ===")
            print(f"Players to keep from previous squad: {len(players_kept)}")
            print(
                f"Proposed transfers: {transfers_made} "
                f"(out of {free_transfers} free transfers)"
            )

            if players_out:
                print("Players to transfer OUT:")
                for player_id in players_out:
                    # Find player name from previous squad or current database
                    prev_player = df[df["id"] == player_id]
                    if not prev_player.empty:
                        player_name = prev_player.iloc[0]["display_name"]
                        print(f"  - {player_name}")

            if players_in:
                print("Players to transfer IN:")
                for player_id in players_in:
                    player = squad[squad["id"] == player_id].iloc[0]
                    print(
                        f"  + {player['display_name']} "
                        f"({player['position']}, {player['team']})"
                    )

        squad_starting = squad[squad["starting_XI"] == 1].copy()
        squad_bench = squad[squad["starting_XI"] == 0].copy()

        # Order bench: GK first, then remaining 3 by descending projected points
        gk_bench = squad_bench[squad_bench["position"] == "GK"].copy()
        non_gk_bench = squad_bench[squad_bench["position"] != "GK"].copy()
        non_gk_bench = non_gk_bench.sort_values("projected_points", ascending=False)
        squad_bench = pd.concat([gk_bench, non_gk_bench], ignore_index=True)

        return squad_starting, squad_bench, forced_selections_display
    
    def update_forced_selections_from_squad(self, starting: pd.DataFrame, bench: pd.DataFrame) -> dict:
        """
        Create forced selections dictionary from squad players.

        Args:
            starting (pd.DataFrame): Starting XI players.
            bench (pd.DataFrame): Bench players.

        Returns:
            dict: Forced selections dictionary with all squad players.
        """
        forced_selections = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        full_squad = pd.concat([starting, bench], ignore_index=True)

        for _, player in full_squad.iterrows():
            pos = player["position"]
            name = player["display_name"]
            forced_selections[pos].append(name)

        return forced_selections