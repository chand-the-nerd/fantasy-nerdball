"""Module for squad selection using integer linear programming with
enhanced constraints.

The model is built in two steps. ``build_model`` assembles the problem
and hands back a handle; ``solve_model`` runs the solver against it.
``select_squad_ilp`` does both and keeps the signature everything else
already used, so callers that only want one squad are unchanged.

Splitting the two matters because the transfer scenarios in
TransferEvaluator differ only in the right-hand side of a single
constraint. Building once and moving that bound turns five builds into
one.
"""

import numpy as np
import pandas as pd
import pulp

from ..utils.text_utils import normalize_for_matching

POSITIONS = ("GK", "DEF", "MID", "FWD")


class PoolIndex:
    """Column arrays and row groupings for one player pool.

    Every constraint builder used to filter the pool by writing
    ``df.iloc[i]["position"]`` inside a comprehension, which builds a
    pandas Series for each row it looks at. At roughly 700 players the
    same team-position block alone did that over 100,000 times, and
    model construction measured 3.4 seconds against 0.13 seconds to
    actually solve. Hoisting the columns to numpy once brings the build
    down to about 0.05 seconds.

    Groupings preserve first-appearance order, which is what
    ``Series.unique()`` gave before, so constraints reach the problem in
    the order they always did and the solver's path is unchanged.
    """

    def __init__(self, df: pd.DataFrame):
        self.n = len(df)
        self.position = df["position"].to_numpy()
        self.team_id = df["team_id"].to_numpy()
        self.cost = (
            pd.to_numeric(df["now_cost_m"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        self.ids = df["id"].to_numpy() if "id" in df.columns else None

        self.by_position = {
            position: np.flatnonzero(self.position == position).tolist()
            for position in POSITIONS
        }

        self.team_order = []
        self.by_team = {}
        for row, team in enumerate(self.team_id):
            if team not in self.by_team:
                self.by_team[team] = []
                self.team_order.append(team)
            self.by_team[team].append(row)

        self.team_order_by_position = {}
        self.by_team_position = {}
        for position in POSITIONS:
            order = []
            for row in self.by_position[position]:
                key = (position, self.team_id[row])
                if key not in self.by_team_position:
                    self.by_team_position[key] = []
                    order.append(self.team_id[row])
                self.by_team_position[key].append(row)
            self.team_order_by_position[position] = order

        self.id_to_index = {}
        if self.ids is not None:
            for row, player_id in enumerate(self.ids):
                self.id_to_index[player_id] = row

    def rows(self, position: str) -> list:
        """Row numbers holding the given position."""
        return self.by_position.get(position, [])


class SquadModel:
    """A built ILP, retained so it can be solved more than once."""

    def __init__(self, prob, x, y, df, index, transfer,
                 forced_display, forced_player_ids):
        self.prob = prob
        self.x = x
        self.y = y
        self.df = df
        self.index = index
        # None when no transfer constraint applies: no previous squad,
        # or a wildcard week. Otherwise the constraint object plus the
        # figures needed to recompute its bound.
        self.transfer = transfer
        self.forced_display = forced_display
        self.forced_player_ids = forced_player_ids


class SquadSelector:
    """Handles squad selection optimisation using ILP."""

    def __init__(self, config):
        self.config = config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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
        model = self.build_model(
            df,
            forced_selections,
            prev_squad_ids=prev_squad_ids,
            free_transfers=free_transfers,
            available_budget=available_budget,
            use_projected_points=use_projected_points,
        )

        if model is None:
            return pd.DataFrame(), pd.DataFrame(), None

        squad = self.solve_model(model)

        if squad is None:
            return pd.DataFrame(), pd.DataFrame(), None

        if (prev_squad_ids is not None and show_transfer_summary
                and self.config.GRANULAR_OUTPUT):
            self._display_transfer_summary(
                squad, prev_squad_ids, model.df, free_transfers
            )

        starting_xi, bench = self.split_squad(squad)

        return starting_xi, bench, model.forced_display

    def build_model(
        self,
        df: pd.DataFrame,
        forced_selections: dict,
        prev_squad_ids: list = None,
        free_transfers: int = None,
        available_budget: float = None,
        use_projected_points: bool = False,
    ):
        """
        Assemble the ILP without solving it.

        Returns:
            SquadModel: The built problem, or None if the pool is empty.
        """
        forced_player_ids, forced_display = (
            self._process_forced_selections(df, forced_selections)
        )

        df = self._clean_dataframe(df)
        df = self._limit_pool(df, forced_player_ids, prev_squad_ids)
        n = len(df)

        if n == 0:
            print("Optimisation aborted: no players available after "
                  "filtering.")
            return None

        index = PoolIndex(df)

        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]

        prob = self._setup_optimisation_problem(
            df, x, y, n, use_projected_points
        )

        self._add_basic_constraints(prob, x, y, n)
        self._add_position_constraints(prob, x, y, index)
        self._add_forced_selection_constraints(
            prob, x, forced_player_ids, index
        )
        transfer = self._add_transfer_constraints(
            prob, x, prev_squad_ids, free_transfers, index
        )
        self._add_bench_constraints(prob, x, y, index, forced_player_ids)
        self._add_team_constraints(prob, x, index)
        self._add_same_team_position_constraints(prob, x, index)
        self._add_budget_constraint(prob, x, index, available_budget)

        return SquadModel(
            prob=prob,
            x=x,
            y=y,
            df=df,
            index=index,
            transfer=transfer,
            forced_display=forced_display,
            forced_player_ids=forced_player_ids,
        )

    def solve_model(self, model: SquadModel):
        """
        Solve a built model and return the selected squad.

        Returns:
            pd.DataFrame: The fifteen selected players, or None if the
                          solver did not reach an optimal solution.
        """
        status = model.prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if status != pulp.LpStatusOptimal:
            # Always reported, not only under GRANULAR_OUTPUT: an
            # infeasible problem returns empty frames, and a silent
            # empty squad is indistinguishable from "no transfers".
            print(f"Optimisation failed with status: "
                  f"{pulp.LpStatus[status]}. Check the budget, forced "
                  "selections and free transfer count.")
            return None

        return self._extract_solution(
            model.df, model.x, model.y, model.index.n
        )

    def solve_scenarios(self, model: SquadModel, transfer_limits):
        """
        Re-solve one built model at each transfer limit in turn.

        The scenarios differ only in how many of last week's players
        must be kept, which is the right-hand side of a single
        constraint. Moving that bound between solves avoids rebuilding
        an identical model each time.

        Yields:
            tuple: (transfer_limit, squad_dataframe_or_None)
        """
        for limit in transfer_limits:
            self.set_transfer_limit(model, limit)
            yield limit, self.solve_model(model)

    def set_transfer_limit(self, model: SquadModel,
                           free_transfers: int) -> None:
        """Move the transfer bound on an already-built model."""
        transfer = model.transfer

        if transfer is None or free_transfers is None:
            return

        remaining = max(0, free_transfers - transfer["missing"])
        minimum = max(0, len(transfer["available"]) - remaining)

        # PuLP holds ``expr >= k`` as ``expr - k >= 0``, so the bound
        # sits in the constraint's constant with its sign flipped.
        transfer["constraint"].constant = -minimum

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def _process_forced_selections(self, df: pd.DataFrame,
                                   forced_selections: dict) -> tuple:
        """
        Process forced selections and return IDs and display info.

        Matching is vectorised rather than a row-wise ``apply``.
        Starting XI optimisation forces all fifteen squad members, so
        the previous version walked the whole pool fifteen times over.
        """
        forced_player_ids = []
        forced_players_info = []

        names_wanted = [
            (pos, name)
            for pos, players_to_force in forced_selections.items()
            for name in (players_to_force or [])
        ]

        if not names_wanted:
            return forced_player_ids, None

        display = df["display_name"].astype(str)
        display_lower = display.str.lower()
        display_normalised = display.map(normalize_for_matching)
        position = df["position"]

        for pos, name in names_wanted:
            matched = (position == pos) & (
                (display_lower == str(name).lower())
                | (display_normalised == normalize_for_matching(name))
            )
            matches = df[matched]

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

    def _limit_pool(self, df: pd.DataFrame, forced_player_ids: list,
                    prev_squad_ids: list) -> pd.DataFrame:
        """
        Optionally cut the pool to the strongest players per position.

        Off unless ``POOL_LIMITS`` is set in the config, for example
        ``{"GK": 20, "DEF": 60, "MID": 60, "FWD": 40}``. Solve time
        falls roughly threefold at those sizes, but the tail is not
        always dead weight - a cheap enabler the budget constraint
        wants can sit well down the score order - so the answer is no
        longer guaranteed to match the full pool. A knob to reach for
        only if the vectorised build somehow is not enough.

        Everyone in the previous squad and every forced selection is
        kept whatever their score, or the transfer and forced-selection
        constraints go infeasible.
        """
        limits = getattr(self.config, "POOL_LIMITS", None)

        if not limits or "fpl_score" not in df.columns:
            return df

        keep_ids = set(forced_player_ids or [])
        keep_ids |= set(prev_squad_ids or [])

        if "id" in df.columns and keep_ids:
            mask = df["id"].isin(keep_ids)
        else:
            mask = pd.Series(False, index=df.index)

        # A position with no limit named against it is kept whole.
        mask |= ~df["position"].isin(list(limits))

        for position, limit in limits.items():
            subset = df[df["position"] == position]

            if limit is None or len(subset) <= int(limit):
                mask |= df["position"] == position
                continue

            top = subset.nlargest(int(limit), "fpl_score").index
            mask |= pd.Series(df.index.isin(top), index=df.index)

        limited = df[mask].reset_index(drop=True)

        if self.config.GRANULAR_OUTPUT:
            print(f"Pool limited to {len(limited)} of {len(df)} "
                  "players by POOL_LIMITS")

        return limited

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
        """
        Set up the main optimisation problem with its objective.

        Left as a term-by-term sum on purpose: it measures at about
        12ms for a full pool, so there is nothing here to win, and
        rewriting it would reorder the columns the solver sees.
        """
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
                                  y: list, index: PoolIndex):
        """Add position constraints for the squad and starting XI."""
        for pos, count in self.config.SQUAD_SIZE.items():
            prob += (
                pulp.lpSum([x[i] for i in index.rows(pos)]) == count
            )

        position_constraints = [
            ("GK", 1, 1),
            ("DEF", 3, 5),
            ("MID", 3, 5),
            ("FWD", 1, 3)
        ]

        for pos, min_count, max_count in position_constraints:
            pos_sum = pulp.lpSum([y[i] for i in index.rows(pos)])
            prob += pos_sum >= min_count
            prob += pos_sum <= max_count

    def _add_forced_selection_constraints(self, prob: pulp.LpProblem,
                                          x: list,
                                          forced_player_ids: list,
                                          index: PoolIndex):
        """Add constraints for forced player selections."""
        for player_id in forced_player_ids:
            row = index.id_to_index.get(player_id)
            if row is not None:
                prob += x[row] == 1

    def _add_transfer_constraints(self, prob: pulp.LpProblem, x: list,
                                  prev_squad_ids: list,
                                  free_transfers: int,
                                  index: PoolIndex):
        """
        Add transfer constraints based on the previous squad.

        Players from the previous squad who are no longer in the player
        pool - because they left the league, were blacklisted, or were
        dropped upstream - are counted as transfers that have already
        happened. Without this the constraint can become unsatisfiable
        and the whole optimisation returns an empty squad.

        Returns:
            dict: The constraint object and the figures needed to move
                  its bound later, or None if no constraint applies.
        """
        if prev_squad_ids is None or free_transfers is None:
            return None

        if self.config.WILDCARD:
            if self.config.GRANULAR_OUTPUT:
                print("WILDCARD ACTIVE: no transfer constraints applied")
            return None

        available = [
            pid for pid in prev_squad_ids if pid in index.id_to_index
        ]
        missing = len(prev_squad_ids) - len(available)

        if missing:
            print(f"Warning: {missing} previous squad player(s) are no "
                  "longer selectable and are counted as forced "
                  "transfers.")

        remaining_transfers = max(0, free_transfers - missing)

        prev_players_kept = pulp.lpSum(
            [x[index.id_to_index[pid]] for pid in available]
        )

        min_players_to_keep = len(available) - remaining_transfers
        min_players_to_keep = max(0, min_players_to_keep)

        constraint = prev_players_kept >= min_players_to_keep
        prob += constraint

        return {
            "constraint": constraint,
            "available": available,
            "missing": missing,
        }

    def _add_bench_constraints(self, prob: pulp.LpProblem, x: list,
                               y: list, index: PoolIndex,
                               forced_player_ids: list = None):
        """Add bench-specific constraints."""
        forced = set(forced_player_ids or [])
        cap = getattr(self.config, "BENCH_GK_MAX_COST", 4.0)

        goalkeepers = index.rows("GK")

        # The benched keeper should be a cheap one so the budget is not
        # spent on a goalkeeper who never plays. The cap is relaxed when
        # honouring it is impossible.
        forced_gk_costs = [
            index.cost[i]
            for i in goalkeepers
            if index.ids is not None and index.ids[i] in forced
        ]
        cap_conflicts = (
            len(forced_gk_costs) >= 2
            and all(cost > cap for cost in forced_gk_costs)
        )

        cheap_gks = [i for i in goalkeepers if index.cost[i] <= cap]

        if not cheap_gks:
            print(f"Bench GK price cap relaxed: no goalkeeper at or "
                  f"below £{cap}m is available.")
        elif cap_conflicts:
            if self.config.GRANULAR_OUTPUT:
                print("Bench GK price cap relaxed: both GKs are forced "
                      f"above £{cap}m")
        else:
            prob += (
                pulp.lpSum([(x[i] - y[i]) for i in cheap_gks]) == 1
            )

        for pos in ["DEF", "MID", "FWD"]:
            prob += (
                pulp.lpSum([(x[i] - y[i]) for i in index.rows(pos)])
                <= 2
            )

    def _add_team_constraints(self, prob: pulp.LpProblem, x: list,
                              index: PoolIndex):
        """Add the maximum players per team constraint."""
        for team in index.team_order:
            prob += (
                pulp.lpSum([x[i] for i in index.by_team[team]])
                <= self.config.MAX_PER_TEAM
            )

    def _add_same_team_position_constraints(self, prob: pulp.LpProblem,
                                            x: list,
                                            index: PoolIndex):
        """
        Allow at most two players from the same team in the same
        position.
        """
        constraint_count = 0

        for position in POSITIONS:
            for team_id in index.team_order_by_position[position]:
                rows = index.by_team_position[(position, team_id)]
                prob += pulp.lpSum([x[i] for i in rows]) <= 2
                constraint_count += 1

        if self.config.GRANULAR_OUTPUT:
            print(f"Added {constraint_count} same team-position "
                  "constraints (max 2 per team-position)")

    def _add_budget_constraint(self, prob: pulp.LpProblem, x: list,
                               index: PoolIndex,
                               available_budget: float):
        """Add the budget constraint."""
        budget_to_use = (
            available_budget if available_budget is not None
            else self.config.BUDGET
        )
        prob += (
            pulp.lpSum([x[i] * index.cost[i] for i in range(index.n)])
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
            int(is_set(y[i])) for i in range(n) if selected_mask[i]
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

    def split_squad(self, squad: pd.DataFrame) -> tuple:
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

    def _split_squad(self, squad: pd.DataFrame) -> tuple:
        """Retained for anything still calling the private name."""
        return self.split_squad(squad)

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

        for position, name in zip(full_squad["position"],
                                  full_squad["display_name"]):
            forced_selections[position].append(name)

        return forced_selections
