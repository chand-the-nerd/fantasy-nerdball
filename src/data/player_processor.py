"""Module for processing player data from the FPL API."""

import os
import pandas as pd
import numpy as np
from ..api.fpl_client import FPLClient
from ..utils.text_utils import normalize_name


class PlayerProcessor:
    """Handles fetching and processing of player data."""

    def __init__(self, config):
        self.config = config
        self.fpl_client = FPLClient()

        # Position-based weighting system for xG analysis
        self.position_weights = {
            "FWD": {"attacking": 1.0, "defensive": 0.0},
            "MID": {"attacking": 0.75, "defensive": 0.25},
            "DEF": {"attacking": 0.25, "defensive": 0.75},
            "GK": {"attacking": 0.0, "defensive": 1.0}
        }

    def fetch_current_players(self) -> pd.DataFrame:
        """
        Fetch current FPL player data from the API and prepare it for
        analysis, including current season xG performance.

        Returns:
            pd.DataFrame: Current player data with cost, position, team
                          info, join keys and xG metrics.
        """
        data = self.fpl_client.get_bootstrap_static()
        players = pd.DataFrame(data["elements"])

        teams = pd.DataFrame(data["teams"])[["id", "name", "code"]].rename(
            columns={"id": "team_id", "name": "team", "code": "team_code"}
        )

        # Validate config team names and refresh the promoted list before
        # any scoring happens, so a stale config fails loudly rather than
        # silently applying modifiers to clubs that are not in the league.
        self._validate_team_configuration(teams)
        self._resolve_promoted_teams(teams)

        teams = self._add_team_matches_played(teams)

        pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

        players = players.rename(
            columns={"team": "team_id", "element_type": "pos_id"}
        )
        players["position"] = players["pos_id"].map(pos_map)

        # bootstrap-static elements already carry team_code; drop it so
        # the teams merge is the single source of truth.
        if "team_code" in players.columns:
            players = players.drop(columns=["team_code"])

        players = players.merge(teams, on="team_id", how="left")

        players["form"] = pd.to_numeric(
            players["form"], errors="coerce"
        ).fillna(0.0)
        players["now_cost_m"] = players["now_cost"] / 10.0
        players["display_name"] = players["web_name"]

        # player_code is the permanent FPL identifier and is the join key
        # for every cross-season and cross-frame merge in this project.
        # name_key is retained for display and legacy lookups only.
        players["player_code"] = pd.to_numeric(
            players["code"], errors="coerce"
        ).astype("Int64")
        players["name_key"] = players["web_name"].map(normalize_name)

        self._check_player_codes(players)

        players["minutes_played"] = pd.to_numeric(
            players["minutes"], errors="coerce"
        ).fillna(0)
        players["starts"] = pd.to_numeric(
            players.get("starts"), errors="coerce"
        ).fillna(0)

        # Prefer starts over minutes/90 so that per-game rates are not
        # inflated for players who are routinely substituted.
        minutes_games = (
            players["minutes_played"] / 90
        ).round().clip(lower=0)
        players["games_played"] = players["starts"].where(
            players["starts"] > 0, minutes_games
        )

        players = self._calculate_current_season_xg_performance(players)

        players = self._calculate_form_consistency(players)

        os.makedirs("data", exist_ok=True)
        players.to_csv("data/players.csv", index=False)

        if self.config.GRANULAR_OUTPUT:
            print("Player data saved to data/players.csv")

        return players

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------

    def _add_team_matches_played(self,
                                 teams: pd.DataFrame) -> pd.DataFrame:
        """
        Count how many matches each club has actually played.

        Cumulative statistics such as starts and minutes were previously
        divided by a single global ``GAMEWEEK - 1``. Run midway through
        a gameweek that assumes every club is level, so the two or four
        clubs that have already played look like they start more often
        and play more minutes than everyone else, and the optimiser
        piles into them. Counting per club removes the skew and lets the
        tool be run at any point in the week.
        """
        fallback = max(1, self.config.GAMEWEEK - 1)

        if not getattr(self.config, "NORMALISE_PARTIAL_GAMEWEEK", True):
            teams["team_matches_played"] = fallback
            return teams

        try:
            fixtures = pd.DataFrame(self.fpl_client.get_fixtures())
        except Exception as error:
            print(f"Could not fetch fixtures to count matches played "
                  f"({error}). Falling back to GW{fallback} for every "
                  "club.")
            teams["team_matches_played"] = fallback
            return teams

        if fixtures.empty or "started" not in fixtures.columns:
            teams["team_matches_played"] = fallback
            return teams

        # A match that has kicked off is already contributing minutes
        # and points to the bootstrap totals, so it counts as played.
        started = fixtures["started"].fillna(False).astype(bool)
        played = fixtures[started]

        appearances = pd.Series(
            played["team_h"].tolist() + played["team_a"].tolist()
        ).value_counts()

        teams["team_matches_played"] = (
            teams["team_id"].map(appearances).fillna(0).astype(int)
        )

        self._report_partial_gameweek(teams)

        return teams

    def _report_partial_gameweek(self, teams: pd.DataFrame):
        """Tell the user when clubs are not level on matches played."""
        counts = teams["team_matches_played"]
        if counts.empty or counts.min() == counts.max():
            return

        ahead = teams.loc[counts == counts.max(), "team"].tolist()

        print(f"\nPartial gameweek detected: clubs have played between "
              f"{int(counts.min())} and {int(counts.max())} matches.")
        print(f"  Ahead of the rest: {', '.join(sorted(ahead))}")
        print("  Per-club match counts are being used for every "
              "rate calculation, so these clubs get no advantage from "
              "having played more.")

    def _validate_team_configuration(self, teams: pd.DataFrame):
        """
        Check that TEAM_MODIFIERS covers exactly the clubs the API is
        returning.

        A club named differently in config (for example "Ipswich" where
        the API says "Ipswich Town") silently falls through the
        ``.get(team, 1.0)`` default, so the modifier never applies and
        the team ratings fallback never creates an entry for it.
        """
        if not getattr(self.config, "VALIDATE_TEAM_NAMES", True):
            return

        api_teams = set(teams["team"])
        config_teams = set(self.config.TEAM_MODIFIERS)

        unknown = config_teams - api_teams
        missing = api_teams - config_teams

        if unknown or missing:
            message = ["TEAM_MODIFIERS is out of sync with the FPL API."]
            if missing:
                message.append(
                    f"  Missing from config: {sorted(missing)}"
                )
            if unknown:
                message.append(
                    f"  Not in the league: {sorted(unknown)}"
                )
            message.append(
                "  Set VALIDATE_TEAM_NAMES = False to downgrade this to "
                "a warning."
            )
            raise ValueError("\n".join(message))

    def _resolve_promoted_teams(self, teams: pd.DataFrame):
        """
        Work out which clubs are newly promoted from last season's data
        rather than relying on a hardcoded list that goes stale every
        summer.

        A club whose team code did not appear in the most recent past
        season has just come up.
        """
        if not getattr(self.config, "AUTO_DETECT_PROMOTED_TEAMS", True):
            return

        past_seasons = getattr(self.config, "PAST_SEASONS", [])
        if not past_seasons:
            return

        latest_season = max(past_seasons)
        url = (
            f"https://raw.githubusercontent.com/vaastav/"
            f"Fantasy-Premier-League/master/data/{latest_season}/"
            f"teams.csv"
        )

        try:
            previous = pd.read_csv(url)
        except Exception as error:
            if self.config.GRANULAR_OUTPUT:
                print(f"Could not fetch {latest_season} teams to detect "
                      f"promoted clubs ({error}). Keeping the configured "
                      "PROMOTED_TEAMS list.")
            return

        if "code" not in previous.columns:
            return

        previous_codes = set(previous["code"].astype(int))
        promoted = sorted(
            teams.loc[
                ~teams["team_code"].astype(int).isin(previous_codes),
                "team"
            ].tolist()
        )

        if not promoted:
            return

        configured = sorted(getattr(self.config, "PROMOTED_TEAMS", []))
        if configured != promoted and self.config.GRANULAR_OUTPUT:
            print(f"Promoted teams detected from {latest_season} data: "
                  f"{promoted} (config had {configured})")

        self.config.PROMOTED_TEAMS = promoted

    def _check_player_codes(self, players: pd.DataFrame):
        """Fail loudly if player codes are missing or not unique."""
        missing = int(players["player_code"].isna().sum())
        if missing:
            raise ValueError(
                f"{missing} players have no 'code' value - cannot join "
                "reliably across seasons."
            )

        duplicates = int(players["player_code"].duplicated().sum())
        if duplicates:
            raise ValueError(
                f"{duplicates} duplicate player codes in the current "
                "player set."
            )

        if self.config.GRANULAR_OUTPUT:
            colliding = players["name_key"].duplicated(keep=False).sum()
            if colliding:
                print(f"Note: {colliding} players share a display name "
                      "with another player. Joins use player_code, so "
                      "this is informational only.")

    # ------------------------------------------------------------------
    # Current season xG analysis
    # ------------------------------------------------------------------

    def _calculate_current_season_xg_performance(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate current season xG performance ratios for all players
        with position-weighted analysis.

        Args:
            df (pd.DataFrame): Player dataframe with xG stats

        Returns:
            pd.DataFrame: Dataframe with current season xG metrics
        """
        df = self._extract_xg_metrics(df)
        df = self._initialise_xg_columns(df)
        thresholds = self._calculate_xg_thresholds()
        df = self._calculate_per_game_metrics(df)
        df = self._calculate_position_weighted_xop(df, thresholds)

        df["xg_trend"] = df.apply(
            lambda row: self._format_xg_trend(row), axis=1)

        players_with_data = len(
            df[df["current_xg_context"] != "insufficient_data"])
        if self.config.GRANULAR_OUTPUT:
            print(f"   {players_with_data} players have current season "
                  "xG analysis")

        df = self._cleanup_temporary_columns(df)

        return df

    def _extract_xg_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract xG metrics with safe conversion."""
        xg_columns = {
            'expected_goals': 'expected_goals',
            'expected_assists': 'expected_assists',
            'expected_goal_involvements': 'expected_goal_involvements',
            'expected_goals_conceded': 'expected_goals_conceded',
            'goals_scored': 'goals_scored',
            'assists': 'assists',
            'goals_conceded': 'goals_conceded'
        }

        for col_name, df_col in xg_columns.items():
            if df_col in df.columns:
                df[col_name] = pd.to_numeric(
                    df[df_col], errors="coerce").fillna(0)
            else:
                df[col_name] = 0.0
                if self.config.GRANULAR_OUTPUT:
                    print(f"Warning: {df_col} not found in current "
                          "season data")

        return df

    def _initialise_xg_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initialise granular xG performance columns."""
        df["current_xOP"] = 1.0
        df["current_xg_context"] = "insufficient_data"
        df["attacking_xOP"] = 1.0
        df["defensive_xOP"] = 1.0
        return df

    def _calculate_xg_thresholds(self) -> dict:
        """Calculate xG thresholds based on gameweeks completed."""
        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)

        if gameweeks_completed == 1:
            return {
                'min_xg_threshold': 0.01,
                'min_xgc_threshold': 0.01,
                'min_games_started': 1
            }
        elif gameweeks_completed <= 3:
            return {
                'min_xg_threshold': 0.05,
                'min_xgc_threshold': 0.05,
                'min_games_started': 1
            }
        return {
            'min_xg_threshold': 0.15,
            'min_xgc_threshold': 0.15,
            'min_games_started': 2
        }

    def _calculate_per_game_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate per-game metrics for the current season."""
        per_game_cols = [
            "current_xgi_per_game", "current_gi_per_game",
            "current_xgc_per_game", "current_gc_per_game"
        ]
        for col in per_game_cols:
            df[col] = 0.0

        current_played_mask = df["starts"] >= 1

        if current_played_mask.any():
            starts = df.loc[current_played_mask, "starts"]

            df.loc[current_played_mask, "current_xgi_per_game"] = (
                df.loc[current_played_mask,
                       "expected_goal_involvements"] / starts
            )
            df.loc[current_played_mask, "current_gi_per_game"] = (
                (df.loc[current_played_mask, "goals_scored"] +
                 df.loc[current_played_mask, "assists"]) / starts
            )
            df.loc[current_played_mask, "current_xgc_per_game"] = (
                df.loc[current_played_mask,
                       "expected_goals_conceded"] / starts
            )
            df.loc[current_played_mask, "current_gc_per_game"] = (
                df.loc[current_played_mask, "goals_conceded"] / starts
            )

        return df

    def _calculate_position_weighted_xop(self, df: pd.DataFrame,
                                         thresholds: dict) -> pd.DataFrame:
        """Calculate position-weighted current xOP for each player."""
        attacking_mask = (
            (df["current_xgi_per_game"] >
             thresholds['min_xg_threshold']) &
            (df["starts"] >= thresholds['min_games_started']) &
            (df["position"].isin(["FWD", "MID", "DEF"]))
        )

        if attacking_mask.any():
            df.loc[attacking_mask, "attacking_xOP"] = (
                df.loc[attacking_mask, "current_gi_per_game"] /
                df.loc[attacking_mask, "current_xgi_per_game"]
            ).clip(0.2, 3.0).round(2)

        defensive_mask = (
            (df["current_xgc_per_game"] >
             thresholds['min_xgc_threshold']) &
            (df["starts"] >= thresholds['min_games_started']) &
            (df["position"].isin(["GK", "DEF", "MID"]))
        )

        if defensive_mask.any():
            # For GC, a higher ratio means better performance
            df.loc[defensive_mask, "defensive_xOP"] = (
                df.loc[defensive_mask, "current_xgc_per_game"] /
                df.loc[defensive_mask,
                       "current_gc_per_game"].clip(lower=0.01)
            ).clip(0.2, 3.0).round(2)

        attacking_weight = df["position"].map(
            lambda p: self.position_weights.get(
                p, {"attacking": 0.0}
            )["attacking"]
        ).astype(float)
        defensive_weight = df["position"].map(
            lambda p: self.position_weights.get(
                p, {"defensive": 0.0}
            )["defensive"]
        ).astype(float)

        use_attacking = attacking_mask & (attacking_weight > 0)
        use_defensive = defensive_mask & (defensive_weight > 0)

        both = use_attacking & use_defensive
        attacking_only = use_attacking & ~use_defensive
        defensive_only = use_defensive & ~use_attacking

        weighted = pd.Series(1.0, index=df.index)
        weighted[both] = (
            df.loc[both, "attacking_xOP"] * attacking_weight[both]
            + df.loc[both, "defensive_xOP"] * defensive_weight[both]
        )
        weighted[attacking_only] = df.loc[
            attacking_only, "attacking_xOP"
        ]
        weighted[defensive_only] = df.loc[
            defensive_only, "defensive_xOP"
        ]

        context = pd.Series("insufficient_data", index=df.index)
        context[both] = (
            "mixed_"
            + (attacking_weight[both] * 100).astype(int).astype(str)
            + "att_"
            + (defensive_weight[both] * 100).astype(int).astype(str)
            + "def"
        )
        context[attacking_only] = "attacking_only"
        context[defensive_only] = "defensive_only"

        df["current_xOP"] = weighted.round(2)
        df["current_xg_context"] = context

        return df.copy()

    def _calculate_weighted_xop(self, row: pd.Series, weights: dict,
                                thresholds: dict) -> tuple:
        """Calculate weighted xOP for a single player."""
        attacking_weight = weights["attacking"]
        defensive_weight = weights["defensive"]

        attacking_xop = row["attacking_xOP"]
        defensive_xop = row["defensive_xOP"]

        has_attacking_data = (
            attacking_weight > 0 and
            row["current_xgi_per_game"] >
            thresholds['min_xg_threshold'] and
            row["starts"] >= thresholds['min_games_started']
        )

        has_defensive_data = (
            defensive_weight > 0 and
            row["current_xgc_per_game"] >
            thresholds['min_xgc_threshold'] and
            row["starts"] >= thresholds['min_games_started']
        )

        if has_attacking_data and has_defensive_data:
            weighted_xop = (
                (attacking_xop * attacking_weight) +
                (defensive_xop * defensive_weight)
            )
            context = (f"mixed_{int(attacking_weight * 100)}att_"
                       f"{int(defensive_weight * 100)}def")

        elif has_attacking_data and attacking_weight > 0:
            weighted_xop = attacking_xop
            context = "attacking_only"

        elif has_defensive_data and defensive_weight > 0:
            weighted_xop = defensive_xop
            context = "defensive_only"

        else:
            weighted_xop = 1.0
            context = "insufficient_data"

        return weighted_xop, context

    def _format_xg_trend(self, row: pd.Series) -> str:
        """Format xG trend for display with position-aware reading."""
        if row.get('current_xg_context') == 'insufficient_data':
            return "N/A"

        current_ratio = row.get('current_xOP', 1.0)
        context = row.get('current_xg_context', '')

        if 'att' in context or context == 'attacking_only':
            return self._format_attacking_trend(current_ratio)
        elif 'def' in context or context == 'defensive_only':
            return self._format_defensive_trend(current_ratio)
        elif 'mixed' in context:
            return self._format_mixed_trend(current_ratio)

        return f"➡️{current_ratio:.2f}"

    def _format_attacking_trend(self, ratio: float) -> str:
        """Format attacking performance trend."""
        if ratio > 1.2:
            return f"🔥{ratio:.2f}"
        elif ratio > 1.1:
            return f"↗️{ratio:.2f}"
        elif ratio < 0.8:
            return f"📈{ratio:.2f}"
        elif ratio < 0.9:
            return f"↘️{ratio:.2f}"
        return f"➡️{ratio:.2f}"

    def _format_defensive_trend(self, ratio: float) -> str:
        """Format defensive performance trend."""
        if ratio > 1.2:
            return f"🛡️{ratio:.2f}"
        elif ratio > 1.1:
            return f"↗️{ratio:.2f}"
        elif ratio < 0.8:
            return f"📈{ratio:.2f}"
        elif ratio < 0.9:
            return f"↘️{ratio:.2f}"
        return f"➡️{ratio:.2f}"

    def _format_mixed_trend(self, ratio: float) -> str:
        """Format mixed performance trend."""
        if ratio > 1.15:
            return f"⭐{ratio:.2f}"
        elif ratio > 1.05:
            return f"↗️{ratio:.2f}"
        elif ratio < 0.85:
            return f"📈{ratio:.2f}"
        elif ratio < 0.95:
            return f"↘️{ratio:.2f}"
        return f"➡️{ratio:.2f}"

    def _cleanup_temporary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up temporary columns used in xG analysis."""
        temp_columns = [
            "attacking_xOP",
            "defensive_xOP",
            "current_xgi_per_game",
            "current_gi_per_game",
            "current_xgc_per_game",
            "current_gc_per_game"
        ]

        existing = [col for col in temp_columns if col in df.columns]
        return df.drop(columns=existing)

    # ------------------------------------------------------------------
    # Previous squad matching
    # ------------------------------------------------------------------

    def match_players_to_current(self, prev_squad: pd.DataFrame,
                                 current_players: pd.DataFrame) -> list:
        """
        Match previous squad players to the current player database.

        Args:
            prev_squad (pd.DataFrame): Previous gameweek's squad.
            current_players (pd.DataFrame): Current player database.

        Returns:
            list: Player IDs from the previous squad that are still
                  available.
        """
        prev_player_ids = []

        for _, prev_player in prev_squad.iterrows():
            player_id = self._find_matching_player(
                prev_player, current_players)
            if player_id:
                prev_player_ids.append(player_id)

        return prev_player_ids

    def _find_matching_player(self, prev_player: pd.Series,
                              current_players: pd.DataFrame) -> int:
        """
        Find the matching player ID in the current database.

        Matching prefers the permanent player code where the saved squad
        has one. Name matching is kept only as a fallback for squad files
        written before player_code was recorded, and no longer filters on
        team, so a player who has moved club is still found.
        """
        if "player_code" in prev_player.index:
            code = prev_player["player_code"]
            if pd.notna(code):
                match = current_players[
                    current_players["player_code"] == int(code)
                ]
                if len(match) == 1:
                    return match.iloc[0]["id"]

        prev_name = str(prev_player["display_name"]).strip()
        prev_pos = prev_player["position"]
        prev_team = prev_player.get("team")

        exact_matches = current_players[
            (current_players["position"] == prev_pos)
            & (current_players["display_name"].str.strip() == prev_name)
        ]

        if len(exact_matches) == 1:
            return exact_matches.iloc[0]["id"]

        if len(exact_matches) > 1:
            # Same name and position: disambiguate on club if we can.
            same_team = exact_matches[
                exact_matches["team"] == prev_team
            ]
            if len(same_team) == 1:
                return same_team.iloc[0]["id"]

            print(f"Warning: {len(exact_matches)} players match "
                  f"'{prev_name}' ({prev_pos}) and the club no longer "
                  "identifies them. Save squads with player_code to "
                  "resolve this.")
            return exact_matches.iloc[0]["id"]

        print(f"Warning: could not find a current match for {prev_name} "
              f"({prev_pos}, {prev_team}). It will be treated as a "
              "forced transfer.")
        return None

    # ------------------------------------------------------------------
    # Form consistency
    # ------------------------------------------------------------------

    def _calculate_form_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate form consistency using stored per-gameweek history.
        Penalises volatile, spiky form and high blank rates.

        History is looked up by FPL element id rather than by name and
        club. The previous name-and-club lookup returned nothing for any
        player who had moved, so every transferred player silently kept
        the neutral 1.0 modifier.
        """
        from ..data.player_history_tracker import PlayerHistoryTracker

        tracker = PlayerHistoryTracker(self.config)

        df['form_consistency'] = 1.0

        current_gw = self.config.GAMEWEEK
        n_weeks = min(8, max(5, current_gw - 1))

        if current_gw <= 2:
            return df

        resolved = 0

        for idx, player in df.iterrows():
            try:
                player_history = tracker.get_player_history(
                    player['web_name'],
                    player['team'],
                    player_id=player.get('id')
                )
                if player_history.empty:
                    continue

                recent_gws = player_history.head(n_weeks)
                if len(recent_gws) < 3:
                    continue

                resolved += 1

                recent_points = recent_gws['total_points'].to_numpy(
                    dtype=float
                )

                mean_points = recent_points.mean()
                if mean_points <= 0:
                    df.loc[idx, 'form_consistency'] = 1.0
                    continue

                std_points = recent_points.std(ddof=0)
                cv = std_points / mean_points

                # "Blank" = fewer than 3 points
                blank_rate = (recent_points < 3).mean()

                sorted_points = np.sort(recent_points)[::-1]
                if len(sorted_points) >= 2:
                    second_best = max(sorted_points[1], 1e-6)
                    spike_ratio = sorted_points[0] / second_best
                else:
                    spike_ratio = 1.0

                if cv <= 0.4:
                    base = 1.1 + (0.4 - cv) * 0.5
                elif cv <= 1.0:
                    base = 1.0 + (1.0 - cv) * 0.1
                elif cv <= 1.6:
                    base = 1.0 - (cv - 1.0) * 0.3
                else:
                    base = 0.82 - (cv - 1.6) * 0.2

                blank_mult = 1.0 - blank_rate * 0.5
                blank_mult = np.clip(blank_mult, 0.5, 1.1)

                if spike_ratio > 2.5:
                    spike_mult = (
                        1.0 - min(spike_ratio - 2.5, 2.0) * 0.25
                    )
                else:
                    spike_mult = 1.0

                consistency_modifier = base * blank_mult * spike_mult
                consistency_modifier = float(
                    np.clip(consistency_modifier, 0.4, 1.4)
                )

                df.loc[idx, 'form_consistency'] = consistency_modifier

            except Exception as error:
                if self.config.GRANULAR_OUTPUT:
                    print(
                        f"Warning: could not calculate consistency for "
                        f"{player.get('web_name', 'unknown')}: {error}"
                    )
                continue

        if self.config.GRANULAR_OUTPUT:
            print(f"   Form consistency resolved for {resolved} players")

        return df