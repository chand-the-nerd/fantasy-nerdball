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
            df (pd.DataFrame): Player data with fpl_scores and all required 
                              fields.
            forced_selections (dict): Dictionary of forced player selections.
            prev_squad_ids (list, optional): Lst of player IDs from prev squad.
            free_transfers (int, optional): Number of free transfers available.
            show_transfer_summary (bool): Whether to display transfer 
                                        information.
            available_budget (float, optional): Available budget for squad.

        Returns:
            tuple: (starting_xi_dataframe, bench_dataframe, 
                   forced_selections_str) containing the optimal squad 
                   selection and info about forced players.
        """
        # Process forced selections before DataFrame modifications
        forced_player_ids, forced_selections_display = (
            self._process_forced_selections(df, forced_selections)
        )

        # Clean the dataframe
        df = self._clean_dataframe(df)
        n = len(df)

        # Create decision variables
        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]

        # Set up optimisation problem
        prob = self._setup_optimisation_problem(df, x, y, n)
        
        # Add constraints
        self._add_basic_constraints(prob, x, y, n)
        self._add_position_constraints(prob, x, y, df, n)
        self._add_form_constraints(prob, y, df, n)
        self._add_forced_selection_constraints(prob, x, forced_player_ids, df)
        self._add_transfer_constraints(
            prob, x, prev_squad_ids, free_transfers, df)
        self._add_bench_constraints(prob, x, y, df, n)
        self._add_team_constraints(prob, x, df, n)
        self._add_budget_constraint(prob, x, df, available_budget, n)

        # Solve the problem
        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if status != pulp.LpStatusOptimal:
            print(f"Optimisation failed with status: {pulp.LpStatus[status]}")
            return pd.DataFrame(), pd.DataFrame(), None

        # Extract and process results
        squad = self._extract_solution(df, x, y, n)
        
        # Display transfer information if requested
        if prev_squad_ids is not None and show_transfer_summary:
            self._display_transfer_summary(squad, prev_squad_ids, df, 
                                         free_transfers)

        # Split into starting XI and bench
        starting_xi, bench = self._split_squad(squad)

        return starting_xi, bench, forced_selections_display
    
    def _process_forced_selections(self, df: pd.DataFrame, 
                                 forced_selections: dict) -> tuple:
        """Process forced selections and return player IDs and display info."""
        forced_player_ids = []
        forced_players_info = []

        for pos, players_to_force in forced_selections.items():
            if not players_to_force:
                continue

            for name in players_to_force:
                # Search for player in original dataframe
                for idx, row in df.iterrows():
                    if self._is_matching_player(row, name, pos):
                        forced_player_ids.append(row["id"])
                        forced_players_info.append(
                            f"{row['display_name']} ({row['position']}, "
                            f"{row['team']})"
                        )
                        break

        # Store forced selections info for later display
        forced_selections_display = (
            ", ".join(forced_players_info) if forced_players_info else None
        )
        
        return forced_player_ids, forced_selections_display
    
    def _is_matching_player(self, row: pd.Series, name: str, pos: str) -> bool:
        """Check if a player row matches the given name and position."""
        return (row["position"] == pos and (
            row["display_name"].lower() == name.lower()
            or normalize_for_matching(row["display_name"])
            == normalize_for_matching(name)
        ))
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataframe for optimisation."""
        df = df.reset_index(drop=True).drop_duplicates(subset="name_key")
        df = df[
            ~df["display_name"].str.lower().isin(self.config.BLACKLIST_PLAYERS)
        ].copy()
        return df
    
    def _setup_optimisation_problem(self, df: pd.DataFrame, x: list, y: list, 
                                  n: int) -> pulp.LpProblem:
        """Set up the main optimisation problem with objective function."""
        prob = pulp.LpProblem("FPL_Squad_Selection", pulp.LpMaximize)
        
        # Use projected_points for starting XI optimisation,
        # fpl_score for squad
        prob += pulp.lpSum(
            y[i] * df.iloc[i]["projected_points"] + 
            0.2 * (x[i] - y[i]) * df.iloc[i]["fpl_score"]
            for i in range(n)
        )
        
        return prob
    
    def _add_basic_constraints(self, prob: pulp.LpProblem, x: list, y: list, 
                             n: int):
        """Add basic squad size and starting XI constraints."""
        prob += pulp.lpSum(x[i] for i in range(n)) == 15  # Total squad
        prob += pulp.lpSum(y[i] for i in range(n)) == 11  # Starting XI
        
        # Starting XI must be subset of squad
        for i in range(n):
            prob += y[i] <= x[i]
    
    def _add_position_constraints(self, prob: pulp.LpProblem, x: list, y: list,
                                df: pd.DataFrame, n: int):
        """Add position constraints for squad and starting XI."""
        # Squad position constraints
        for pos, count in self.config.SQUAD_SIZE.items():
            prob += (
                pulp.lpSum(x[i] for i in range(n) 
                          if df.iloc[i]["position"] == pos) == count
            )

        # Starting XI position constraints
        position_constraints = [
            ("GK", 1, 1),
            ("DEF", 3, 5),
            ("MID", 3, 5),
            ("FWD", 1, 3)
        ]
        
        for pos, min_count, max_count in position_constraints:
            pos_sum = pulp.lpSum(y[i] for i in range(n) 
                               if df.iloc[i]["position"] == pos)
            prob += pos_sum >= min_count
            prob += pos_sum <= max_count
    
    def _add_form_constraints(self, prob: pulp.LpProblem, y: list, 
                            df: pd.DataFrame, n: int):
        """Add form constraint - only players with form > 0 in starting XI."""
        for i in range(n):
            if df.iloc[i]["form"] <= 0:
                prob += y[i] == 0
    
    def _add_forced_selection_constraints(self, prob: pulp.LpProblem, x: list,
                                        forced_player_ids: list, 
                                        df: pd.DataFrame):
        """Add constraints for forced player selections."""
        for player_id in forced_player_ids:
            # Find player in cleaned dataframe
            for i in range(len(df)):
                if df.iloc[i]["id"] == player_id:
                    prob += x[i] == 1  # Force into squad
                    break
    
    def _add_transfer_constraints(self, prob: pulp.LpProblem, x: list,
                                prev_squad_ids: list, free_transfers: int,
                                df: pd.DataFrame):
        """Add transfer constraints based on previous squad."""
        if (prev_squad_ids is not None and free_transfers is not None and 
            not self.config.WILDCARD):

            # Create mapping of player IDs to dataframe indices
            id_to_index = {df.iloc[i]["id"]: i for i in range(len(df))}

            # Count how many previous squad players are kept
            prev_players_kept = pulp.lpSum(
                x[id_to_index[player_id]]
                for player_id in prev_squad_ids
                if player_id in id_to_index
            )

            # Number of transfers = 15 - number of players kept from prev squad
            # This must be <= free_transfers
            # So: 15 - prev_players_kept <= free_transfers
            # Therefore: prev_players_kept >= 15 - free_transfers
            min_players_to_keep = 15 - free_transfers
            prob += prev_players_kept >= min_players_to_keep

        elif self.config.WILDCARD and prev_squad_ids is not None:
            print("🃏 WILDCARD ACTIVE: No transfer constraints applied")
    
    def _add_bench_constraints(self, prob: pulp.LpProblem, x: list, y: list,
                             df: pd.DataFrame, n: int):
        """Add bench-specific constraints."""
        # Bench must have exactly one £4.0m GK
        prob += (
            pulp.lpSum(
                (x[i] - y[i])
                for i in range(n)
                if (df.iloc[i]["position"] == "GK" and 
                    df.iloc[i]["now_cost_m"] == 4.0)
            ) == 1
        )

        # Allow up to 2 of DEF, MID, FWD on bench
        for pos in ["DEF", "MID", "FWD"]:
            prob += (
                pulp.lpSum((x[i] - y[i]) for i in range(n) 
                          if df.iloc[i]["position"] == pos) <= 2
            )
    
    def _add_team_constraints(self, prob: pulp.LpProblem, x: list, 
                            df: pd.DataFrame, n: int):
        """Add maximum players per team constraint."""
        for team in df["team_id"].unique():
            prob += (
                pulp.lpSum(x[i] for i in range(n) 
                          if df.iloc[i]["team_id"] == team)
                <= self.config.MAX_PER_TEAM
            )
    
    def _add_budget_constraint(
            self,
            prob: pulp.LpProblem,
            x: list, 
            df: pd.DataFrame,
            available_budget: float,
            n: int
            ):
        """Add budget constraint."""
        budget_to_use = (
            available_budget if available_budget is not None 
            else self.config.BUDGET
        )
        prob += (
            pulp.lpSum(x[i] * df.iloc[i]["now_cost_m"] for i in range(n)) 
            <= budget_to_use
        )
    
    def _extract_solution(self, df: pd.DataFrame, x: list, y: list, 
                        n: int) -> pd.DataFrame:
        """Extract the solution from the optimisation result."""
        selected_mask = [pulp.value(x[i]) == 1 for i in range(n)]
        squad = df.iloc[selected_mask].copy()
        squad["starting_XI"] = [
            pulp.value(y[i]) for i in range(n) if pulp.value(x[i]) == 1
        ]
        return squad
    
    def _display_transfer_summary(self, squad: pd.DataFrame, 
                                prev_squad_ids: list, df: pd.DataFrame,
                                free_transfers: int):
        """Display transfer summary information."""
        current_squad_ids = set(squad["id"].tolist())
        prev_squad_ids_set = set(prev_squad_ids)

        players_kept = current_squad_ids.intersection(prev_squad_ids_set)
        players_out = prev_squad_ids_set - current_squad_ids
        players_in = current_squad_ids - prev_squad_ids_set

        transfers_made = len(players_out)

        print(f"\n=== Proposed Transfer Summary ===")
        print(f"Players to keep from previous squad: {len(players_kept)}")
        print(f"Proposed transfers: {transfers_made} "
              f"(out of {free_transfers} free transfers)")

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
                print(f"  + {player['display_name']} "
                      f"({player['position']}, {player['team']})")
    
    def _split_squad(self, squad: pd.DataFrame) -> tuple:
        """Split squad into starting XI and bench with proper ordering."""
        squad_starting = squad[squad["starting_XI"] == 1].copy()
        squad_bench = squad[squad["starting_XI"] == 0].copy()

        # Order bench: GK first, then remaining by descending projected points
        gk_bench = squad_bench[squad_bench["position"] == "GK"].copy()
        non_gk_bench = squad_bench[squad_bench["position"] != "GK"].copy()
        non_gk_bench = non_gk_bench.sort_values(
            "projected_points", ascending=False
        )
        squad_bench = pd.concat([gk_bench, non_gk_bench], ignore_index=True)

        return squad_starting, squad_bench
    
    def update_forced_selections_from_squad(self, starting: pd.DataFrame, 
                                          bench: pd.DataFrame) -> dict:
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