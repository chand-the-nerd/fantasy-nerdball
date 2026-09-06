"""Module for squad selection using integer linear programming with
enhanced constraints."""

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
        use_projected_points: bool = False,
    ) -> tuple:
        """
        Select the optimal FPL squad using integer linear programming
        with forced player selections and transfer constraints.

        Args:
            df (pd.DataFrame): Player data with scores and all required
                               fields.
            forced_selections (dict): Forced player selections.
            prev_squad_ids (list, optional): Player IDs from the
                                             previous squad.
            free_transfers (int, optional): Free transfers available.
            show_transfer_summary (bool): Whether to display transfer
                                          information.
            available_budget (float, optional): Available budget.
            use_projected_points (bool): Optimise on projected points
                                         rather than FPL score.

        Returns:
            tuple: (starting_xi, bench, forced_selections_display)
        """
        forced_player_ids, forced_selections_display = (
            self._process_forced_selections(df, forced_selections)
        )

        df = self._clean_dataframe(df)
        n = len(df)

        if n == 0:
            print("Optimisation aborted: no players available after "
                  "filtering.")
            return pd.DataFrame(), pd.DataFrame(), None

        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]

        prob = self._setup_optimisation_problem(
            df, x, y, n, use_projected_points
        )

        self._add_basic_constraints(prob, x, y, n)
        self._add_position_constraints(prob, x, y, df, n)
        self._add_forced_selection_constraints(
            prob, x, forced_player_ids, df
        )
        self._add_transfer_constraints(
            prob, x, prev_squad_ids, free_transfers, df
        )
        self._add_bench_constraints(prob, x, y, df, n, forced_player_ids)
        self._add_team_constraints(prob, x, df, n)
        self._add_same_team_position_constraints(prob, x, df, n)
        self._add_budget_constraint(prob, x, df, available_budget, n)

        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if status != pulp.LpStatusOptimal:
            # Always reported, not only under GRANULAR_OUTPUT: an
            # infeasible problem returns empty frames, and a silent
            # empty squad is indistinguishable from "no transfers".
            print(f"Optimisation failed with status: "
                  f"{pulp.LpStatus[status]}. Check the budget, forced "
                  "selections and free transfer count.")
            return pd.DataFrame(), pd.DataFrame(), None

        squad = self._extract_solution(df, x, y, n)

        if (prev_squad_ids is not None and show_transfer_summary
                and self.config.GRANULAR_OUTPUT):
            self._display_transfer_summary(
                squad, prev_squad_ids, df, free_transfers
            )

        starting_xi, bench = self._split_squad(squad)

        return starting_xi, bench, forced_selections_display

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def _process_forced_selections(self, df: pd.DataFrame,
                                   forced_selections: dict) -> tuple:
        """Process forced selections and return IDs and display info."""
        forced_player_ids = []
        forced_players_info = []

        for pos, players_to_force in forced_selections.items():
            if not players_to_force:
                continue

            for name in players_to_force:
                matches = df[
                    df.apply(
                        lambda row: self._is_matching_player(
                            row, name, pos
                        ),
                        axis=1
                    )
                ]

                if matches.empty:
                    print(f"Warning: forced selection '{name}' ({pos}) "
                          "was not found in the player pool.")
                    continue

                if len(matches) > 1:
                    print(f"Warning: forced selection '{name}' ({pos}) "
                          f"matches {len(matches)} players "
                          f"({', '.join(matches['team'].tolist())}). "
                          "Using the first.")

                row = matches.iloc[0]
                forced_player_ids.append(row["id"])
                forced_players_info.append(
                    f"{row['display_name']} ({row['position']}, "
                    f"{row['team']})"
                )

        forced_selections_display = (
            ", ".join(forced_players_info)
            if forced_players_info else None
        )

        return forced_player_ids, forced_selections_display

    def _is_matching_player(self, row: pd.Series, name: str,
                            pos: str) -> bool:
        """Check if a player row matches the given name and position."""
        return (row["position"] == pos and (
            row["display_name"].lower() == name.lower()
            or normalize_for_matching(row["display_name"])
            == normalize_for_matching(name)
        ))

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataframe for optimisation.

        Deduplication is on the FPL element id. The previous key of
        (name_key, position) silently deleted one of any two players
        sharing a display surname in the same position - for example the
        two midfielders named Kamara in the current season - removing
        them from the optimiser's universe entirely.
        """
        df = df.reset_index(drop=True)

        dedupe_key = "id" if "id" in df.columns else "player_code"
        before = len(df)
        df = df.drop_duplicates(subset=[dedupe_key])

        if len(df) != before:
            print(f"Warning: dropped {before - len(df)} duplicate rows "
                  f"on '{dedupe_key}' before optimisation. Duplicates "
                  "indicate a bad merge upstream.")

        blacklist = {
            str(name).lower()
            for name in self.config.BLACKLIST_PLAYERS
        }
        if blacklist:
            df = df[
                ~df["display_name"].str.lower().isin(blacklist)
            ].copy()

        return df.reset_index(drop=True)

    def _objective_scores(self, df: pd.DataFrame,
                          use_projected_points: bool) -> list:
        """
        Build the objective coefficients, applying a soft penalty to
        players with no recent form rather than excluding them.

        The original implementation forbade any player with form of zero
        from the starting XI. FPL form is points per game over the last
        30 days, so a player returning from a knock, or one whose last
        appearance falls outside the window after an international
        break, is legitimately on zero and was being barred from
        selection outright.
        """
        column = (
            "projected_points" if use_projected_points else "fpl_score"
        )
        scores = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

        if not getattr(self.config, "PENALISE_ZERO_FORM", True):
            return scores.tolist()

        form = pd.to_numeric(df["form"], errors="coerce").fillna(0.0)

        # In GW1 nobody has form, so the penalty would apply to
        # everyone and carry no information.
        if not (form > 0).any():
            return scores.tolist()

        penalty = getattr(self.config, "ZERO_FORM_PENALTY", 0.5)
        zero_form = form <= 0

        scores = scores.copy()
        # Reduce positive scores and leave negative ones alone, so the
        # penalty can never improve a player's standing.
        scores.loc[zero_form & (scores > 0)] *= (1.0 - penalty)

        return scores.tolist()

    def _setup_optimisation_problem(self, df: pd.DataFrame, x: list,
                                    y: list, n: int,
                                    use_projected_points: bool = False
                                    ) -> pulp.LpProblem:
        """Set up the main optimisation problem with its objective."""
        prob = pulp.LpProblem("FPL_Squad_Selection", pulp.LpMaximize)

        scores = self._objective_scores(df, use_projected_points)

        if use_projected_points:
            # Starting XI selection: pure projected points
            prob += pulp.lpSum(y[i] * scores[i] for i in range(n))
        else:
            # Squad selection: FPL score with bench weighting
            bench_weight = getattr(self.config, "BENCH_WEIGHT", 0.2)
            prob += pulp.lpSum(
                y[i] * scores[i]
                + bench_weight * (x[i] - y[i]) * scores[i]
                for i in range(n)
            )

        return prob

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def _add_basic_constraints(self, prob: pulp.LpProblem, x: list,
                               y: list, n: int):
        """Add basic squad size and starting XI constraints."""
        prob += pulp.lpSum(x[i] for i in range(n)) == 15
        prob += pulp.lpSum(y[i] for i in range(n)) == 11

        for i in range(n):
            prob += y[i] <= x[i]

    def _add_position_constraints(self, prob: pulp.LpProblem, x: list,
                                  y: list, df: pd.DataFrame, n: int):
        """Add position constraints for the squad and starting XI."""
        for pos, count in self.config.SQUAD_SIZE.items():
            prob += (
                pulp.lpSum(x[i] for i in range(n)
                           if df.iloc[i]["position"] == pos) == count
            )

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

    def _add_forced_selection_constraints(self, prob: pulp.LpProblem,
                                          x: list,
                                          forced_player_ids: list,
                                          df: pd.DataFrame):
        """Add constraints for forced player selections."""
        id_to_index = {
            df.iloc[i]["id"]: i for i in range(len(df))
        }

        for player_id in forced_player_ids:
            index = id_to_index.get(player_id)
            if index is not None:
                prob += x[index] == 1

    def _add_transfer_constraints(self, prob: pulp.LpProblem, x: list,
                                  prev_squad_ids: list,
                                  free_transfers: int,
                                  df: pd.DataFrame):
        """
        Add transfer constraints based on the previous squad.

        Players from the previous squad who are no longer in the player
        pool - because they left the league, were blacklisted, or were
        dropped upstream - are counted as transfers that have already
        happened. Without this the constraint can become unsatisfiable
        and the whole optimisation returns an empty squad.
        """
        if prev_squad_ids is None or free_transfers is None:
            return

        if self.config.WILDCARD:
            if self.config.GRANULAR_OUTPUT:
                print("WILDCARD ACTIVE: no transfer constraints applied")
            return

        id_to_index = {df.iloc[i]["id"]: i for i in range(len(df))}

        available = [
            pid for pid in prev_squad_ids if pid in id_to_index
        ]
        missing = len(prev_squad_ids) - len(available)

        if missing:
            print(f"Warning: {missing} previous squad player(s) are no "
                  "longer selectable and are counted as forced "
                  "transfers.")

        remaining_transfers = max(0, free_transfers - missing)

        prev_players_kept = pulp.lpSum(
            x[id_to_index[pid]] for pid in available
        )

        min_players_to_keep = len(available) - remaining_transfers
        min_players_to_keep = max(0, min_players_to_keep)

        prob += prev_players_kept >= min_players_to_keep

    def _add_bench_constraints(self, prob: pulp.LpProblem, x: list,
                               y: list, df: pd.DataFrame, n: int,
                               forced_player_ids: list = None):
        """Add bench-specific constraints."""
        forced_player_ids = forced_player_ids or []
        cap = getattr(self.config, "BENCH_GK_MAX_COST", 4.0)

        # The benched keeper should be a cheap one so the budget is not
        # spent on a goalkeeper who never plays. The cap is relaxed when
        # honouring it is impossible.
        forced_gk_costs = [
            df.iloc[i]["now_cost_m"]
            for i in range(n)
            if (df.iloc[i]["position"] == "GK"
                and df.iloc[i]["id"] in forced_player_ids)
        ]
        cap_conflicts = (
            len(forced_gk_costs) >= 2
            and all(cost > cap for cost in forced_gk_costs)
        )

        cheap_gks = [
            i for i in range(n)
            if (df.iloc[i]["position"] == "GK"
                and df.iloc[i]["now_cost_m"] <= cap)
        ]

        if not cheap_gks:
            print(f"Bench GK price cap relaxed: no goalkeeper at or "
                  f"below £{cap}m is available.")
        elif cap_conflicts:
            if self.config.GRANULAR_OUTPUT:
                print("Bench GK price cap relaxed: both GKs are forced "
                      f"above £{cap}m")
        else:
            prob += (
                pulp.lpSum((x[i] - y[i]) for i in cheap_gks) == 1
            )

        for pos in ["DEF", "MID", "FWD"]:
            prob += (
                pulp.lpSum((x[i] - y[i]) for i in range(n)
                           if df.iloc[i]["position"] == pos) <= 2
            )

    def _add_team_constraints(self, prob: pulp.LpProblem, x: list,
                              df: pd.DataFrame, n: int):
        """Add the maximum players per team constraint."""
        for team in df["team_id"].unique():
            prob += (
                pulp.lpSum(x[i] for i in range(n)
                           if df.iloc[i]["team_id"] == team)
                <= self.config.MAX_PER_TEAM
            )

    def _add_same_team_position_constraints(self, prob: pulp.LpProblem,
                                            x: list, df: pd.DataFrame,
                                            n: int):
        """
        Allow at most two players from the same team in the same
        position.
        """
        positions = ["GK", "DEF", "MID", "FWD"]
        constraint_count = 0

        for position in positions:
            teams_in_position = df[
                df["position"] == position
            ]["team_id"].unique()

            for team_id in teams_in_position:
                team_pos_players = pulp.lpSum(
                    x[i] for i in range(n)
                    if (df.iloc[i]["position"] == position and
                        df.iloc[i]["team_id"] == team_id)
                )
                prob += team_pos_players <= 2
                constraint_count += 1

        if self.config.GRANULAR_OUTPUT:
            print(f"Added {constraint_count} same team-position "
                  "constraints (max 2 per team-position)")

    def _add_budget_constraint(self, prob: pulp.LpProblem, x: list,
                               df: pd.DataFrame,
                               available_budget: float, n: int):
        """Add the budget constraint."""
        budget_to_use = (
            available_budget if available_budget is not None
            else self.config.BUDGET
        )
        prob += (
            pulp.lpSum(x[i] * df.iloc[i]["now_cost_m"]
                       for i in range(n))
            <= budget_to_use
        )

    # ------------------------------------------------------------------
    # Solution handling
    # ------------------------------------------------------------------

    def _extract_solution(self, df: pd.DataFrame, x: list, y: list,
                          n: int) -> pd.DataFrame:
        """
        Extract the solution from the optimisation result.

        Binary variables are read with a 0.5 threshold. CBC returns
        floating point values, so comparing against 1 exactly can drop a
        selected player whose value comes back as 0.9999999996.
        """
        def is_set(variable):
            value = pulp.value(variable)
            return value is not None and value > 0.5

        selected_mask = [is_set(x[i]) for i in range(n)]
        squad = df.iloc[selected_mask].copy()

        squad["starting_XI"] = [
            int(is_set(y[i])) for i in range(n) if is_set(x[i])
        ]

        if len(squad) != 15:
            print(f"Warning: solver returned {len(squad)} players "
                  "rather than 15.")

        return squad

    def _display_transfer_summary(self, squad: pd.DataFrame,
                                  prev_squad_ids: list,
                                  df: pd.DataFrame,
                                  free_transfers: int):
        """Display transfer summary information."""
        current_squad_ids = set(squad["id"].tolist())
        prev_squad_ids_set = set(prev_squad_ids)

        players_kept = current_squad_ids.intersection(prev_squad_ids_set)
        players_out = prev_squad_ids_set - current_squad_ids
        players_in = current_squad_ids - prev_squad_ids_set

        transfers_made = len(players_out)

        print("\n=== Proposed Transfer Summary ===")
        print(f"Players to keep from previous squad: {len(players_kept)}")
        print(f"Proposed transfers: {transfers_made} "
              f"(out of {free_transfers} free transfers)")

        if players_out:
            print("Players to transfer OUT:")
            for player_id in players_out:
                prev_player = df[df["id"] == player_id]
                if not prev_player.empty:
                    print(f"  - {prev_player.iloc[0]['display_name']}")
                else:
                    print(f"  - player id {player_id} (no longer "
                          "selectable)")

        if players_in:
            print("Players to transfer IN:")
            for player_id in players_in:
                player = squad[squad["id"] == player_id].iloc[0]
                print(f"  + {player['display_name']} "
                      f"({player['position']}, {player['team']})")

    def _split_squad(self, squad: pd.DataFrame) -> tuple:
        """Split the squad into starting XI and bench."""
        squad_starting = squad[squad["starting_XI"] == 1].copy()
        squad_bench = squad[squad["starting_XI"] == 0].copy()

        gk_bench = squad_bench[squad_bench["position"] == "GK"].copy()
        non_gk_bench = squad_bench[
            squad_bench["position"] != "GK"
        ].copy()
        non_gk_bench = non_gk_bench.sort_values(
            "projected_points", ascending=False
        )
        squad_bench = pd.concat(
            [gk_bench, non_gk_bench], ignore_index=True
        )

        return squad_starting, squad_bench

    def update_forced_selections_from_squad(self, starting: pd.DataFrame,
                                            bench: pd.DataFrame) -> dict:
        """
        Create a forced selections dictionary from squad players.

        Args:
            starting (pd.DataFrame): Starting XI players.
            bench (pd.DataFrame): Bench players.

        Returns:
            dict: Forced selections covering the whole squad.
        """
        forced_selections = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        full_squad = pd.concat([starting, bench], ignore_index=True)

        for _, player in full_squad.iterrows():
            pos = player["position"]
            forced_selections[pos].append(player["display_name"])

        return forced_selections