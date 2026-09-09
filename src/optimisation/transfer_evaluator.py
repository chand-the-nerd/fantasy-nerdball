"""Module for evaluating transfer strategies and alternatives."""

import os

import pandas as pd
import pulp


class TransferEvaluator:
    """Handles evaluation of transfer strategies and alternatives."""

    def __init__(self, config):
        self.config = config
        # The held-squad comparison is asked for up to three times in a
        # run - by evaluate_transfer_strategy, by its optimised variant
        # and again by the web pipeline - always with the same inputs.
        self._no_transfer_cache = None
        # Set once the transfer ladder has run, so the later strategy
        # check defers to it rather than deciding the same thing twice
        # on a coarser basis.
        self._last_best_scenario = None

    @property
    def horizon(self) -> int:
        """Gameweeks over which a transfer's benefit is counted.

        The same window the scores are built over, deliberately. This
        used to be its own setting, and the two drifting apart was the
        cause of a real misjudgement: with the look-ahead at one
        gameweek and the horizon at four, a squad's advantage was
        measured over a single fixture while the four-point hit that
        bought it was spread across four, so a one-point edge outbid a
        cost four times its size.

        There is no reading of the model on which they should differ.
        The look-ahead is the window over which a projection means
        anything, so it is also the longest window over which a hit
        can honestly be recouped. Counting the benefit further out
        than the fixtures were scored is inventing evidence.
        """
        return max(1, int(getattr(self.config, "FIRST_N_GAMEWEEKS", 1)))

    def _amortised_penalty(self, penalty_points: float) -> float:
        """Spread a one-off points hit across the holding horizon."""
        return penalty_points / self.horizon

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def get_unavailable_players(self, df: pd.DataFrame,
                                prev_squad_ids: list) -> list:
        """
        Identify players from the previous squad who are unavailable for
        this gameweek.

        Args:
            df (pd.DataFrame): Current player database
            prev_squad_ids (list): Player IDs from the previous squad

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
                if (player_row["status"] != "a" and
                        player_row.get(
                            "chance_of_playing_next_round", 100) < 75):
                    unavailable_ids.append(player_id)

        return unavailable_ids

    def _calculate_hypothetical_projected_points(
            self, player_row: pd.Series) -> float:
        """
        Calculate a player's projected points as if they were available,
        skipping the availability filters.

        Args:
            player_row (pd.Series): Player data row

        Returns:
            float: Hypothetical projected points
        """
        position = player_row.get("position", "MID")
        baseline_points = self.config.BASELINE_POINTS_PER_GAME.get(
            position, 4.0)

        base_quality = player_row.get("base_quality", 0.0)

        points_adjustment = (
            base_quality * self.config.FPL_SCORE_TO_POINTS_MULTIPLIER
        )

        hypothetical_points = max(
            1.0, baseline_points + points_adjustment
        )

        fixture_multiplier = player_row.get("fixture_multiplier", 1.0)
        if pd.notna(fixture_multiplier) and fixture_multiplier > 0:
            hypothetical_points *= fixture_multiplier

        return hypothetical_points

    # ------------------------------------------------------------------
    # Substitute versus transfer
    # ------------------------------------------------------------------

    def evaluate_substitute_vs_transfer(
        self, df: pd.DataFrame, prev_squad_ids: list,
        unavailable_player_ids: list, free_transfers: int
    ) -> dict:
        """
        Evaluate whether to substitute unavailable players or transfer
        them out, using the previous gameweek's saved squad.

        Args:
            df (pd.DataFrame): Current player database with scores
            prev_squad_ids (list): Player IDs from the previous squad
            unavailable_player_ids (list): IDs who cannot play this week
            free_transfers (int): Free transfers available

        Returns:
            dict: Analysis of substitute versus transfer options
        """
        if not prev_squad_ids or not unavailable_player_ids:
            return {
                "recommendation": "no_action",
                "reason": "No unavailable players or no previous squad",
            }

        prev_gw = self.config.GAMEWEEK - 1
        prev_squad_file = f"squads/gw{prev_gw}/full_squad_simple.csv"

        try:
            if not os.path.exists(prev_squad_file):
                return self._fallback_substitute_analysis()

            prev_squad_csv = pd.read_csv(prev_squad_file)

        except Exception:
            return self._fallback_substitute_analysis()

        prev_squad_df = df[df["id"].isin(prev_squad_ids)].copy()
        unavailable_df = prev_squad_df[
            prev_squad_df["id"].isin(unavailable_player_ids)
        ]

        if self.config.GRANULAR_OUTPUT:
            print("\n=== Substitute vs Transfer Analysis ===")
            print(f"Unavailable players: {len(unavailable_df)}")
            for _, player in unavailable_df.iterrows():
                print(f"  - {player['display_name']} "
                      f"({player['position']}, {player['team']})")

        substitute_scenarios = []

        for _, unavailable_player in unavailable_df.iterrows():
            unavailable_points = (
                self._calculate_hypothetical_projected_points(
                    unavailable_player)
            )

            best_substitute = self._find_best_bench_substitute(
                unavailable_player, prev_squad_csv, prev_squad_df
            )

            if best_substitute is None:
                substitute_scenarios.append({
                    "unavailable_player":
                        unavailable_player["display_name"],
                    "position": unavailable_player["position"],
                    "unavailable_score": round(unavailable_points, 1),
                    "best_substitute": None,
                    "substitute_score": 0,
                    "score_loss": round(unavailable_points, 1),
                    "recommendation": "transfer",
                })
            else:
                substitute_points = best_substitute["projected_points"]

                unavailable_rounded = round(unavailable_points, 1)
                substitute_rounded = round(substitute_points, 1)
                score_loss = unavailable_rounded - substitute_rounded

                substitute_label = (
                    f"{best_substitute['display_name']} "
                    f"({best_substitute['position']})"
                )
                recommendation = (
                    "substitute" if score_loss < 2.0
                    else "consider_transfer"
                )

                substitute_scenarios.append({
                    "unavailable_player":
                        unavailable_player["display_name"],
                    "position": unavailable_player["position"],
                    "unavailable_score": unavailable_rounded,
                    "best_substitute": substitute_label,
                    "substitute_score": substitute_rounded,
                    "score_loss": score_loss,
                    "recommendation": recommendation,
                })

        return self._evaluate_substitution_strategy(
            substitute_scenarios, free_transfers
        )

    def _find_best_bench_substitute(self, unavailable_player: pd.Series,
                                    prev_squad_csv: pd.DataFrame,
                                    prev_squad_df: pd.DataFrame
                                    ) -> pd.Series:
        """
        Find the best available substitute from the saved bench.

        Matching prefers a saved player id or player code. The substring
        name match is a fallback for older squad files and can pair the
        wrong player when two squad members share a surname.
        """
        bench_csv = prev_squad_csv[
            prev_squad_csv['squad_role'] == 'Bench'
        ]

        available_bench_players = []

        for _, bench_player_csv in bench_csv.iterrows():
            match = self._match_bench_row(
                bench_player_csv, prev_squad_df
            )
            if match is not None and match["status"] == "a":
                available_bench_players.append(match)

        if not available_bench_players:
            return None

        bench_df = pd.DataFrame(available_bench_players)
        bench_df = bench_df.drop_duplicates(subset=['id'])

        return bench_df.loc[bench_df["projected_points"].idxmax()]

    def _match_bench_row(self, bench_player_csv: pd.Series,
                         prev_squad_df: pd.DataFrame) -> pd.Series:
        """Resolve one saved bench row to a current player row."""
        for column, frame_column in [
            ("id", "id"), ("player_code", "player_code")
        ]:
            if column in bench_player_csv.index:
                value = bench_player_csv[column]
                if pd.notna(value):
                    match = prev_squad_df[
                        prev_squad_df[frame_column] == value
                    ]
                    if len(match) == 1:
                        return match.iloc[0]

        player_name = str(bench_player_csv['player']).replace(
            ' (C)', '').replace(' (V)', '').strip()

        matches = prev_squad_df[
            prev_squad_df['display_name'].str.strip().str.lower()
            == player_name.lower()
        ]

        if len(matches) == 1:
            return matches.iloc[0]

        if len(matches) > 1 and self.config.GRANULAR_OUTPUT:
            print(f"Warning: bench entry '{player_name}' matches "
                  f"{len(matches)} squad players. Using the first.")

        if len(matches) >= 1:
            return matches.iloc[0]

        return None

    def _fallback_substitute_analysis(self) -> dict:
        """Fallback analysis when saved squad data is not available."""
        return {
            "recommendation": "transfer",
            "reason": "Cannot determine previous squad composition, "
                      "recommend transfers",
            "scenarios": []
        }

    def _evaluate_substitution_strategy(self, substitute_scenarios: list,
                                        free_transfers: int) -> dict:
        """
        Evaluate the overall substitution strategy.

        Args:
            substitute_scenarios (list): Substitution scenarios
            free_transfers (int): Free transfers available

        Returns:
            dict: Strategy recommendation
        """
        total_score_loss = sum(
            scenario["score_loss"] for scenario in substitute_scenarios
        )
        forced_transfers = len([
            s for s in substitute_scenarios
            if s["best_substitute"] is None
        ])

        threshold_total = self.config.MIN_TRANSFER_VALUE

        if forced_transfers > free_transfers:
            decision = {
                "recommendation": "wildcard_needed",
                "reason": (f"Need {forced_transfers} forced transfers "
                           f"but only have {free_transfers} free"),
                "total_score_loss": total_score_loss,
                "scenarios": substitute_scenarios,
            }
        elif total_score_loss > threshold_total:
            decision = {
                "recommendation": "make_transfers",
                "reason": (f"Score loss ({total_score_loss:.1f}) "
                           f"exceeds transfer threshold "
                           f"({threshold_total:.1f})"),
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
                substitute_scenarios, total_score_loss, decision,
                threshold_total
            )
        return decision

    def _print_substitution_analysis(self, substitute_scenarios: list,
                                     total_score_loss: float,
                                     decision: dict,
                                     threshold: float):
        """Print the substitution analysis to the console."""
        print("\nSubstitution scenarios:")
        for scenario in substitute_scenarios:
            if scenario["best_substitute"]:
                print(f"  {scenario['unavailable_player']} "
                      f"(projected points "
                      f"{scenario['unavailable_score']:.1f})"
                      f" → {scenario['best_substitute']} "
                      f"(projected points "
                      f"{scenario['substitute_score']:.1f})")
            else:
                print(f"  {scenario['unavailable_player']} → "
                      "NO SUBSTITUTE AVAILABLE (must transfer)")

        if total_score_loss < 0:
            print("Total score loss from substitutions: NONE")
        else:
            print(f"\nTotal score loss from substitutions: "
                  f"{total_score_loss:.1f}")

        print(f"Transfer threshold: {threshold:.1f}")

    def _calculate_minimum_transfers_needed(self,
                                            forced_selections: dict,
                                            prev_squad_ids: list,
                                            df: pd.DataFrame) -> int:
        """
        Calculate the minimum transfers needed to satisfy the forced
        selections.

        Args:
            forced_selections (dict): Forced player selections
            prev_squad_ids (list): Player IDs from the previous squad
            df (pd.DataFrame): Current player database

        Returns:
            int: Minimum transfers needed
        """
        if not prev_squad_ids or not any(forced_selections.values()):
            return 0

        prev_squad_ids_set = set(prev_squad_ids)
        forced_player_ids = set()

        for position, player_names in forced_selections.items():
            for player_name in player_names:
                player_match = df[
                    (df["display_name"].str.lower()
                     == player_name.lower())
                    & (df["position"] == position)
                ]
                if not player_match.empty:
                    forced_player_ids.add(player_match.iloc[0]["id"])

        forced_not_in_prev = forced_player_ids - prev_squad_ids_set
        return len(forced_not_in_prev)

    # ------------------------------------------------------------------
    # Scenario evaluation
    # ------------------------------------------------------------------

    def get_optimal_squad_with_penalties(
        self, df: pd.DataFrame, forced_selections: dict,
        prev_squad_ids: list, free_transfers: int,
        available_budget: float, squad_selector
    ) -> tuple:
        """
        Get the optimal squad taking transfer penalties into account.

        Args:
            df (pd.DataFrame): Current player database with scores
            forced_selections (dict): Forced player selections
            prev_squad_ids (list): Player IDs from the previous squad
            free_transfers (int): Free transfers available
            available_budget (float): Available budget
            squad_selector: SquadSelector instance

        Returns:
            tuple: (starting_xi, bench, forced_display,
                    transfers_made, penalty_points)
        """
        if prev_squad_ids is None:
            starting, bench, forced_display = (
                squad_selector.select_squad_ilp(
                    df, forced_selections, prev_squad_ids,
                    free_transfers, show_transfer_summary=True,
                    available_budget=available_budget,
                    use_projected_points=False
                )
            )
            return starting, bench, forced_display, 0, 0

        # With hits switched off the sweep stops at the free transfers
        # available, so no scenario carries a penalty. The ladder still
        # runs: deciding how many of the free transfers are worth using
        # is the same question, and it was previously not being asked.
        allow_penalties = bool(self.config.ACCEPT_TRANSFER_PENALTY)

        if self.config.GRANULAR_OUTPUT:
            print("\n=== TRANSFER ANALYSIS ===")
            print(f"\nEvaluating all transfer scenarios up to "
                  f"{free_transfers + 3} transfers, costing hits over a "
                  f"{self.horizon} gameweek horizon...")

        scenarios = self._evaluate_transfer_scenarios(
            df, forced_selections, prev_squad_ids, free_transfers,
            available_budget, squad_selector,
            allow_penalties=allow_penalties
        )

        if not scenarios:
            print("No valid transfer scenarios found.")
            return pd.DataFrame(), pd.DataFrame(), None, 0, 0

        return self._select_best_scenario(scenarios, free_transfers, df)

    def _evaluate_transfer_scenarios(self, df: pd.DataFrame,
                                     forced_selections: dict,
                                     prev_squad_ids: list,
                                     free_transfers: int,
                                     available_budget: float,
                                     squad_selector,
                                     allow_penalties: bool = True
                                     ) -> list:
        """
        Evaluate transfer scenarios and return the results.

        Every scenario is the same model with a different bound on how
        many of last week's players must be kept, so it is built once
        and re-solved. Rebuilding it per scenario was the single
        largest cost in a run: five builds at roughly 4 seconds each,
        against 0.13 seconds to solve.
        """
        scenarios = []

        min_transfers_needed = self._calculate_minimum_transfers_needed(
            forced_selections, prev_squad_ids, df
        )

        max_transfers_to_test = (
            free_transfers + 3 if allow_penalties else free_transfers
        )
        max_transfers_to_test = max(
            max_transfers_to_test, min_transfers_needed
        )
        start_transfers = min_transfers_needed

        if min_transfers_needed > 0 and self.config.GRANULAR_OUTPUT:
            print(f"Forced selections require minimum "
                  f"{min_transfers_needed} transfer(s)")

        limits = range(start_transfers, max_transfers_to_test + 1)

        model = squad_selector.build_model(
            df,
            forced_selections,
            prev_squad_ids=prev_squad_ids,
            free_transfers=start_transfers,
            available_budget=available_budget,
            use_projected_points=False,
        )

        if model is None:
            return scenarios

        solved = squad_selector.solve_scenarios(model, limits)

        for max_transfers_allowed, squad in solved:
            scenario = self._scenario_from_squad(
                squad_selector, model, squad, df, prev_squad_ids,
                max_transfers_allowed, free_transfers
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
        """
        Evaluate a single transfer scenario from scratch.

        Kept for one-off use. The scenario sweep goes through
        ``_evaluate_transfer_scenarios`` instead, which shares one
        model across every limit.
        """
        starting, bench, forced_display = (
            squad_selector.select_squad_ilp(
                df, forced_selections, prev_squad_ids,
                max_transfers_allowed, show_transfer_summary=False,
                available_budget=available_budget,
                use_projected_points=False
            )
        )

        if starting.empty:
            return None

        return self._scenario_fields(
            starting, bench, forced_display, df, prev_squad_ids,
            max_transfers_allowed, free_transfers
        )

    def _scenario_from_squad(self, squad_selector, model, squad,
                             df: pd.DataFrame, prev_squad_ids: list,
                             max_transfers_allowed: int,
                             free_transfers: int) -> dict:
        """Turn one solved squad into a scenario record."""
        if squad is None or squad.empty:
            return None

        starting, bench = squad_selector.split_squad(squad)

        if starting.empty:
            return None

        return self._scenario_fields(
            starting, bench, model.forced_display, df, prev_squad_ids,
            max_transfers_allowed, free_transfers
        )

    def _scenario_fields(self, starting: pd.DataFrame,
                         bench: pd.DataFrame, forced_display,
                         df: pd.DataFrame, prev_squad_ids: list,
                         max_transfers_allowed: int,
                         free_transfers: int) -> dict:
        """Score one candidate squad against the previous one."""
        current_squad_ids = set(
            pd.concat([starting, bench])["id"].tolist()
        )
        prev_squad_ids_set = set(prev_squad_ids)
        actual_transfers = len(prev_squad_ids_set - current_squad_ids)

        extra_transfers = max(0, actual_transfers - free_transfers)
        penalty_points = extra_transfers * 4

        # projected_points is a single gameweek's projection, so the
        # starting total is already the per-gameweek figure. The one-off
        # hit is spread across the holding horizon.
        starting_ppgw = starting["projected_points"].sum()
        amortised = self._amortised_penalty(penalty_points)
        net_ppgw = starting_ppgw - amortised

        transfer_details = self._format_transfer_details(
            prev_squad_ids_set, current_squad_ids, df, starting, bench
        )

        return {
            'max_transfers_allowed': max_transfers_allowed,
            'actual_transfers': actual_transfers,
            'extra_transfers': extra_transfers,
            'penalty_points': penalty_points,
            'amortised_penalty': amortised,
            # What this squad actually scores this week with the whole
            # hit paid, which is the number the app puts on screen.
            'gw_net': starting_ppgw - penalty_points,
            'starting_points_total': starting_ppgw * self.horizon,
            'starting_ppgw': starting_ppgw,
            'net_ppgw': net_ppgw,
            'starting': starting,
            'bench': bench,
            'forced_display': forced_display,
            'transfer_details': transfer_details
        }

    def _format_transfer_details(self, prev_squad_ids_set: set,
                                 current_squad_ids: set,
                                 df: pd.DataFrame,
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

        full_squad = pd.concat([starting, bench])
        in_names = []
        for player_id in players_in:
            match = full_squad[full_squad["id"] == player_id]
            if not match.empty:
                in_names.append(match.iloc[0]["display_name"])

        if out_names and in_names:
            return (f" (OUT: {', '.join(out_names)} → "
                    f"IN: {', '.join(in_names)})")

        return ""

    def _print_scenario_result(self, scenario: dict):
        """Print the result of a transfer scenario."""
        print(f"  Scenario {scenario['max_transfers_allowed']}: "
              f"{scenario['actual_transfers']} transfers, "
              f"{scenario['extra_transfers']} extra, "
              f"penalty: -{scenario['penalty_points']} "
              f"(-{scenario['amortised_penalty']:.1f} per GW over "
              f"{self.horizon}), projected points: "
              f"{scenario['starting_ppgw']:.1f}, net: "
              f"{scenario['net_ppgw']:.1f}"
              f"{scenario['transfer_details']}")

    def _select_best_scenario(self, scenarios: list,
                              free_transfers: int,
                              df: pd.DataFrame) -> tuple:
        """Select the best scenario on net points per gameweek."""
        ladder = self._best_by_transfer_count(scenarios)

        if not ladder:
            print("No baseline transfer scenario found.")
            return pd.DataFrame(), pd.DataFrame(), None, 0, 0

        baseline_scenario = ladder[0]

        if self.config.GRANULAR_OUTPUT:
            print("\nBest scenario analysis:")
            print(f"   Tested {len(scenarios)} different transfer "
                  f"limits, {len(ladder)} distinct outcomes")

        # Climbed a rung at a time rather than taking the highest
        # scoring scenario outright. The top scorer is almost always
        # the one that spends every transfer available, and most of
        # those transfers are not worth making.
        best_scenario = self._climb_transfer_ladder(ladder)
        best_scenario = self._apply_value_threshold(
            best_scenario, baseline_scenario
        )

        self._last_best_scenario = best_scenario

        return self._extract_final_solution(
            best_scenario, free_transfers, df
        )

    def _get_baseline_scenario(self, scenarios: list) -> dict:
        """Get the baseline scenario for comparison."""
        if not scenarios:
            return None

        min_actual_transfers = min(
            s['actual_transfers'] for s in scenarios
        )
        return next(
            (s for s in scenarios
             if s['actual_transfers'] == min_actual_transfers),
            None
        )

    def _print_top_scenarios(self, scenarios: list,
                             best_scenario: dict):
        """Print the top three scenarios for comparison."""
        top_scenarios = sorted(
            scenarios, key=lambda x: x['net_ppgw'], reverse=True
        )[:3]
        for i, scenario in enumerate(top_scenarios, 1):
            status = " BEST" if scenario is best_scenario else ""
            print(f"   #{i}: {scenario['actual_transfers']} transfers "
                  f"→ Net: {scenario['net_ppgw']:.1f} points{status}")

    def _best_by_transfer_count(self, scenarios: list) -> list:
        """One scenario per transfer count, ordered fewest first.

        Several limits collapse to the same number of transfers
        actually made - once the model has taken every move worth
        taking, raising the bound changes nothing - so the sweep
        returns duplicates. The ladder needs one rung per count.
        """
        best = {}

        for scenario in scenarios:
            count = scenario['actual_transfers']
            if (count not in best
                    or scenario['net_ppgw'] > best[count]['net_ppgw']):
                best[count] = scenario

        return [best[count] for count in sorted(best)]

    def _climb_transfer_ladder(self, ladder: list) -> dict:
        """Take transfers one at a time, while each one pays for itself.

        MIN_TRANSFER_VALUE used to be tested against the total gain of
        the best scenario over the baseline, however many transfers
        produced it. That made the decision all or nothing: a first
        transfer worth three points carried a second worth a tenth of
        one, because between them they cleared the bar. The threshold
        is a per-transfer standard, so it is applied per transfer.

        Scanning continues past a rung that fails rather than stopping
        at it. Gains are lumpy - two transfers can be worth little
        separately and a lot together, when one funds the other - and
        stopping at the first failure would miss that.

        The second test is about hits. A scenario costing four points
        is compared over the holding horizon, which quietly assumes
        this week's edge repeats every week until the horizon runs
        out, and that the same move could not have been made next week
        for free. Neither is safe. So a scenario that pays a bigger
        hit must also beat the one it displaces on this week's points
        with the whole hit paid - the figure on screen. Anything else
        recommends a squad that visibly scores less.
        """
        threshold = self.config.MIN_TRANSFER_VALUE
        accepted = ladder[0]

        if self.config.GRANULAR_OUTPUT:
            print("\nTransfer ladder (each transfer must earn "
                  f"{threshold:.1f} points per gameweek):")
            print(f"   {accepted['actual_transfers']} transfers: "
                  f"{accepted['net_ppgw']:.1f} net — baseline")

        for candidate in ladder[1:]:
            extra = (candidate['actual_transfers']
                     - accepted['actual_transfers'])

            if extra <= 0:
                continue

            gain = candidate['net_ppgw'] - accepted['net_ppgw']
            required = threshold * extra
            takes_bigger_hit = (
                candidate['penalty_points'] > accepted['penalty_points']
            )
            hit_pays = candidate['gw_net'] >= accepted['gw_net']

            if gain < required:
                verdict = (f"rejected, {gain:.1f} gained against "
                           f"{required:.1f} needed")
            elif takes_bigger_hit and not hit_pays:
                verdict = (
                    f"rejected, the hit leaves it "
                    f"{accepted['gw_net'] - candidate['gw_net']:.1f} "
                    "points worse off this week"
                )
            else:
                accepted = candidate
                verdict = f"taken, {gain:.1f} gained for {extra} more"

            if self.config.GRANULAR_OUTPUT:
                print(f"   {candidate['actual_transfers']} transfers: "
                      f"{candidate['net_ppgw']:.1f} net "
                      f"({candidate['gw_net']:.1f} this week) — "
                      f"{verdict}")

        return accepted

    def _apply_value_threshold(self, best_scenario: dict,
                               baseline_scenario: dict) -> dict:
        """Record what the ladder gained over making no transfers."""
        improvement_ppgw = (
            best_scenario['net_ppgw'] - baseline_scenario['net_ppgw']
        )

        if self.config.GRANULAR_OUTPUT:
            print(f"   SELECTED: "
                  f"{best_scenario['actual_transfers']} transfers "
                  f"→ {best_scenario['net_ppgw']:.1f} points per "
                  f"gameweek, {improvement_ppgw:+.1f} on the baseline")

        best_scenario['points_improvement_ppgw'] = improvement_ppgw
        best_scenario['gameweeks_analysed'] = self.horizon

        return best_scenario

    def _extract_final_solution(self, best_scenario: dict,
                                free_transfers: int,
                                df: pd.DataFrame) -> tuple:
        """Extract and display the final solution."""
        starting = best_scenario['starting']
        bench = best_scenario['bench']
        best_transfers = best_scenario['actual_transfers']
        best_penalty = best_scenario['penalty_points']
        best_forced_display = best_scenario['forced_display']

        if self.config.GRANULAR_OUTPUT:
            self._print_final_transfer_summary(
                best_transfers, free_transfers, best_penalty,
                best_scenario
            )

        return (starting, bench, best_forced_display,
                best_transfers, best_penalty)

    def _print_final_transfer_summary(self, best_transfers: int,
                                      free_transfers: int,
                                      best_penalty: int,
                                      best_scenario: dict):
        """Print the final transfer summary."""
        print("\n=== Optimal Transfer Strategy ===")

        if best_transfers > 0:
            print(f"Total transfers: {best_transfers}")
            print(f"Free transfers: {free_transfers}")
            if best_penalty > 0:
                print(f"Extra transfers: "
                      f"{best_transfers - free_transfers}")
                print(f"Transfer penalty: -{best_penalty} points, "
                      f"costed as "
                      f"-{best_scenario['amortised_penalty']:.1f} per "
                      f"gameweek over {self.horizon} gameweeks")
            print(f"Projected points: "
                  f"{best_scenario['starting_ppgw']:.1f}")
            print(f"Net points: {best_scenario['net_ppgw']:.1f}")
        else:
            print("Total transfers: 0")
            print("Recommended action: keep current squad")
            print("Reason: no transfers needed")

    # ------------------------------------------------------------------
    # No-transfer comparison
    # ------------------------------------------------------------------

    def evaluate_transfer_value(
        self, no_transfer_squad: pd.DataFrame,
        transfer_squad: pd.DataFrame, transfers_made: int
    ) -> tuple:
        """
        Evaluate whether transfers provide a sufficient improvement.

        Args:
            no_transfer_squad (pd.DataFrame): Starting XI, no transfers.
            transfer_squad (pd.DataFrame): Starting XI with transfers.
            transfers_made (int): Number of transfers.

        Returns:
            tuple: (should_make_transfers, value_analysis_dict)
        """
        if transfers_made == 0:
            return True, {
                "reason": "No transfers needed", "improvement": 0
            }

        # Both frames hold a single gameweek's projection.
        no_transfer_ppgw = no_transfer_squad["projected_points"].sum()
        transfer_ppgw = transfer_squad["projected_points"].sum()
        points_improvement_ppgw = transfer_ppgw - no_transfer_ppgw
        total_improvement = points_improvement_ppgw * self.horizon

        threshold_total = self.config.MIN_TRANSFER_VALUE

        analysis = {
            "transfers_made": transfers_made,
            "no_transfer_ppgw": no_transfer_ppgw,
            "transfer_ppgw": transfer_ppgw,
            "points_improvement_ppgw": points_improvement_ppgw,
            "total_improvement": total_improvement,
            "improvement_per_transfer": (
                total_improvement / transfers_made
                if transfers_made > 0 else 0
            ),
            "threshold_total": threshold_total,
            "gameweeks_analysed": self.horizon,
        }

        if total_improvement < 0:
            return False, {
                **analysis,
                "reason": "Transfers would decrease projected points"
            }

        if total_improvement < threshold_total:
            return False, {
                **analysis,
                "reason": (f"Total improvement "
                           f"({total_improvement:.1f}) below threshold "
                           f"({threshold_total:.1f})")
            }

        return True, {
            **analysis,
            "reason": (f"Transfers provide sufficient improvement "
                       f"({total_improvement:.1f})")
        }

    def get_no_transfer_squad(self, df: pd.DataFrame,
                              prev_squad_ids: list) -> pd.DataFrame:
        """
        Get the optimal starting XI using only players from the previous
        gameweek.

        Args:
            df (pd.DataFrame): Current player database with scores.
            prev_squad_ids (list): Player IDs from the previous squad.

        Returns:
            pd.DataFrame: Best starting XI from the previous squad.
        """
        if prev_squad_ids is None:
            return pd.DataFrame()

        cache = self._no_transfer_cache
        cache_key = tuple(prev_squad_ids)
        if (cache is not None and cache["frame"] is df
                and cache["key"] == cache_key):
            return cache["result"].copy()

        available_ids = set(df["id"].tolist())
        available_prev_players = [
            pid for pid in prev_squad_ids if pid in available_ids
        ]

        if len(available_prev_players) < 15:
            print(f"Warning: only {len(available_prev_players)} of 15 "
                  "previous squad players are still selectable, so a "
                  "no-transfer comparison is not possible.")
            return pd.DataFrame()

        prev_squad_df = df[
            df["id"].isin(available_prev_players)
        ].copy().reset_index(drop=True)

        result = self._optimise_starting_xi_from_squad(prev_squad_df)
        self._no_transfer_cache = {
            "frame": df,
            "key": cache_key,
            "result": result,
        }

        return result.copy()

    def _optimise_starting_xi_from_squad(
            self, prev_squad_df: pd.DataFrame) -> pd.DataFrame:
        """Optimise the starting XI from a given squad using ILP."""
        n = len(prev_squad_df)
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]

        prob = pulp.LpProblem("No_Transfer_Squad", pulp.LpMaximize)
        prob += pulp.lpSum(
            y[i] * prev_squad_df.iloc[i]["projected_points"]
            for i in range(n)
        )

        prob += pulp.lpSum(y[i] for i in range(n)) == 11

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

        # Zero-form players are not excluded outright here. Players who
        # genuinely cannot contribute already carry a projection of zero
        # from the availability filter, so the objective handles them.

        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if status != pulp.LpStatusOptimal:
            print("Warning: could not build a valid no-transfer XI "
                  f"({pulp.LpStatus[status]}).")
            return pd.DataFrame()

        starting_mask = [
            pulp.value(y[i]) is not None and pulp.value(y[i]) > 0.5
            for i in range(n)
        ]
        return prev_squad_df.iloc[starting_mask].copy()

    def get_no_transfer_squad_optimised(self, scored: pd.DataFrame,
                                        prev_squad_ids: list) -> tuple:
        """
        Get the optimised starting XI and bench using only previous
        squad players.

        Args:
            scored (pd.DataFrame): Current player database with scores.
            prev_squad_ids (list): Player IDs from the previous squad.

        Returns:
            tuple: (starting_xi_dataframe, bench_dataframe)
        """
        full_no_transfer_squad = scored[
            scored["id"].isin(prev_squad_ids)
        ].copy()
        starting = self.get_no_transfer_squad(scored, prev_squad_ids)

        if starting.empty:
            return starting, pd.DataFrame()

        bench = full_no_transfer_squad[
            ~full_no_transfer_squad["id"].isin(starting["id"])
        ].copy()

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
        Evaluate the overall transfer strategy.

        Args:
            scored (pd.DataFrame): Current player database with scores.
            prev_squad_ids (list): Player IDs from the previous squad.
            starting_with_transfers (pd.DataFrame): XI with transfers.
            transfers_made (int): Number of transfers.
            free_transfers (int): Free transfers available.
            wildcard_active (bool): Whether the wildcard is active.

        Returns:
            tuple: (should_make_transfers, transfer_analysis)
        """
        if wildcard_active:
            if self.config.GRANULAR_OUTPUT:
                print(f"\nWILDCARD ACTIVE: making {transfers_made} "
                      "changes without constraints")
            return True, {"reason": "Wildcard active - no limits"}

        # The ladder has already weighed every transfer count against
        # this one, per transfer and against the hit. Re-running the
        # coarse all-or-nothing check below would only overrule a
        # better decision with a worse one.
        scenario = self._last_best_scenario

        if scenario is not None:
            penalty_points = max(
                0, scenario['actual_transfers'] - free_transfers
            ) * 4

            if self.config.GRANULAR_OUTPUT:
                print(f"\nTransfer ladder settled on "
                      f"{scenario['actual_transfers']} transfers")
                if penalty_points > 0:
                    print(f"   Transfer penalty: -{penalty_points} "
                          "points, already costed")

            return True, {
                "reason": (
                    "Transfers chosen one at a time, each worth at "
                    f"least {self.config.MIN_TRANSFER_VALUE:.1f} points "
                    "a gameweek"
                ),
                "points_improvement_ppgw": scenario.get(
                    'points_improvement_ppgw', 0),
                "gameweeks_analysed": scenario.get(
                    'gameweeks_analysed', self.horizon)
            }

        if self.config.ACCEPT_TRANSFER_PENALTY:
            return True, {
                "reason": "Transfer penalty mode - already optimised"
            }

        elif prev_squad_ids is not None and transfers_made > 0:
            if self.config.GRANULAR_OUTPUT:
                print("\nEvaluating transfer value...")

            no_transfer_starting = self.get_no_transfer_squad(
                scored, prev_squad_ids)

            if not no_transfer_starting.empty:
                (should_make_transfers,
                 transfer_analysis) = self.evaluate_transfer_value(
                    no_transfer_starting,
                    starting_with_transfers,
                    transfers_made
                )

                if self.config.GRANULAR_OUTPUT:
                    self._print_transfer_value_analysis(
                        transfer_analysis
                    )
                return should_make_transfers, transfer_analysis

        return True, {}

    def _print_transfer_value_analysis(self, transfer_analysis: dict):
        """Print the transfer value analysis results."""
        print("\n=== Transfer Value Analysis ===")
        print(f"Transfers to be made: "
              f"{transfer_analysis['transfers_made']}")
        print(f"Holding horizon (gameweeks): "
              f"{transfer_analysis.get('gameweeks_analysed', 'N/A')}")
        print(f"No-transfer points per gameweek: "
              f"{transfer_analysis['no_transfer_ppgw']:.1f}")
        print(f"With-transfer points per gameweek: "
              f"{transfer_analysis['transfer_ppgw']:.1f}")
        print(f"Points improvement per gameweek: "
              f"{transfer_analysis['points_improvement_ppgw']:.1f}")
        print(f"Total improvement over "
              f"{transfer_analysis['gameweeks_analysed']} gameweeks: "
              f"{transfer_analysis['total_improvement']:.1f}")
        print(f"Improvement per transfer: "
              f"{transfer_analysis['improvement_per_transfer']:.1f}")
        print(f"Total threshold: "
              f"{transfer_analysis['threshold_total']:.1f}")