"""Module for managing historical player performance data with xG analysis.
"""

import os
import pandas as pd
import numpy as np

RECALCULATED_POINTS_DIR = "data/historic"


class HistoricalDataManager:
    """Handles fetching and processing of historical player data with xG
    performance metrics."""

    def __init__(self, config):
        self.config = config

        # Position-based weighting system (same as current season)
        self.position_weights = {
            1: {"attacking": 0.0, "defensive": 1.0},   # GK
            2: {"attacking": 0.25, "defensive": 0.75},  # DEF
            3: {"attacking": 0.75, "defensive": 0.25},  # MID
            4: {"attacking": 1.0, "defensive": 0.0}    # FWD
        }

        # Thresholds for historical analysis
        self.historical_thresholds = {
            'min_xgi_per_game': 0.05,  # Minimum 0.05 xGI per game
            'min_xgc_per_game': 0.05,  # Minimum 0.05 xGC per game
            'min_games': getattr(
                config, "MIN_SEASON_APPEARANCES", 10
            )
        }

        self.current_season = getattr(
            self.config, "CURRENT_SEASON", "2026-27"
        )

        # The current season must never also appear in PAST_SEASONS, or
        # the current-season integration below would create duplicate
        # column names and silently misalign the season weights.
        if self.current_season in self.config.PAST_SEASONS:
            raise ValueError(
                f"CURRENT_SEASON ({self.current_season}) also appears in "
                "PAST_SEASONS. Remove it from PAST_SEASONS."
            )

    # ------------------------------------------------------------------
    # Per-season loading
    # ------------------------------------------------------------------

    def fetch_past_season_points(self, season_folder: str) -> pd.DataFrame:
        """
        Fetch historical performance data including weighted xG metrics
        for a specific season.

        Args:
            season_folder (str): The season folder name (e.g. "2023-24").

        Returns:
            pd.DataFrame: DataFrame containing performance metrics
                          including weighted xG analysis.
        """
        url = (
            f"https://raw.githubusercontent.com/vaastav/"
            f"Fantasy-Premier-League/master/data/"
            f"{season_folder}/players_raw.csv"
        )
        df = pd.read_csv(url)

        # Appearances must come from the raw file. FPL's own
        # points_per_game is total points divided by appearances, so it
        # is the only place the appearance count can be recovered from -
        # and the recalculated file overwrites total_points while
        # leaving points_per_game at its original value, which would
        # make the division meaningless there.
        df["appearances"] = self._derive_appearances(df)

        recalculated_path = os.path.join(
            RECALCULATED_POINTS_DIR, f"{season_folder}_updated_points.csv"
        )
        if os.path.exists(recalculated_path):
            # Points recalculated under current_rules.py (see
            # HistoricPointsRecalculator) take precedence over raw
            # historic points, so DEFCON reflects this season's rules.
            if self.config.GRANULAR_OUTPUT:
                print(f"Using recalculated points for {season_folder} "
                      f"from {recalculated_path}")
            recalculated = pd.read_csv(recalculated_path)
            if {"code", "total_points"} <= set(recalculated.columns):
                replacement = recalculated[
                    ["code", "total_points"]
                ].drop_duplicates(subset=["code"])
                df = df.drop(columns=["total_points"]).merge(
                    replacement, on="code", how="left"
                )
                df["total_points"] = df["total_points"].fillna(0)

        if "code" not in df.columns:
            raise ValueError(
                f"{season_folder} data has no 'code' column - cannot "
                "join players reliably across seasons."
            )

        # Calculate basic metrics
        df = self._calculate_basic_metrics(df)

        # Extract and validate xG metrics
        df = self._extract_historical_xg_metrics(df, season_folder)

        # Calculate per-game metrics
        df = self._calculate_historical_per_game_metrics(df)

        # Damp small-sample points per game toward the positional mean
        df = self._shrink_ppg_to_position_mean(df)

        # Calculate position-weighted xG performance ratios
        df = self._calculate_historical_xg_performance(df)

        # Select relevant columns for output
        df = self._prepare_output_dataframe(df, season_folder)

        return df

    def _derive_appearances(self, df: pd.DataFrame) -> pd.Series:
        """
        Recover each player's appearance count for the season.

        FPL's points_per_game is total points over appearances, so
        dividing one by the other returns the appearance count. This
        matters more than it sounds: dividing points by starts counts
        every substitute appearance in the numerator but not the
        denominator, which inflated rotation players and impact subs by
        a factor of two to six. Nketiah's 23 points from 12 appearances
        scored as 11.5 per game rather than 1.9.
        """
        total_points = pd.to_numeric(
            df.get("total_points"), errors="coerce"
        ).fillna(0)
        fpl_ppg = pd.to_numeric(
            df.get("points_per_game"), errors="coerce"
        ).fillna(0)

        appearances = pd.Series(0.0, index=df.index)
        derivable = fpl_ppg > 0
        appearances[derivable] = (
            total_points[derivable] / fpl_ppg[derivable]
        ).round()

        # Players on zero points have no usable ratio, so fall back to
        # the best available proxy.
        starts = pd.to_numeric(
            df.get("starts"), errors="coerce"
        ).fillna(0)
        minutes = pd.to_numeric(
            df.get("minutes"), errors="coerce"
        ).fillna(0)
        fallback = pd.concat(
            [starts, (minutes / 90).round()], axis=1
        ).max(axis=1)

        appearances[~derivable] = fallback[~derivable]

        return appearances.clip(lower=0, upper=38)

    def _calculate_basic_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic performance metrics.

        Games played is the appearance count, matching FPL's own
        definition of points per game. Points earned as a substitute
        belong to the numerator, so substitute appearances have to
        belong to the denominator.
        """
        df["minutes_played"] = pd.to_numeric(
            df["minutes"], errors="coerce"
        ).fillna(0)
        df["total_points"] = pd.to_numeric(
            df["total_points"], errors="coerce"
        ).fillna(0)

        if "appearances" not in df.columns:
            df["appearances"] = self._derive_appearances(df)

        df["games_played"] = df["appearances"]

        reliability_baseline = getattr(
            self.config, "RELIABILITY_GAMES_BASELINE", 30
        )
        df["season_reliability"] = (
            df["games_played"] / reliability_baseline
        ).clip(upper=1.0)

        return df

    def _shrink_ppg_to_position_mean(
            self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pull each player's points per game toward the mean for their
        position, in proportion to how few appearances it rests on.

        This replaces the original habit of multiplying points per game
        by an availability factor, which changed the meaning of a
        per-game rate. Shrinkage leaves a full season's rate untouched
        while stopping a ten-appearance sample from outranking it.
        """
        k = getattr(self.config, "PPG_SHRINKAGE_APPEARANCES", 10)
        if k <= 0:
            return df

        min_games = self.historical_thresholds['min_games']
        qualified = df["games_played"] >= min_games

        if qualified.sum() < 4:
            return df

        position_mean = df.loc[qualified].groupby(
            "element_type"
        )["points_per_game"].mean()

        prior = df["element_type"].map(position_mean)
        prior = prior.fillna(df.loc[qualified, "points_per_game"].mean())

        games = df["games_played"]
        df["points_per_game"] = (
            (games * df["points_per_game"] + k * prior) / (games + k)
        )

        return df

    def _extract_historical_xg_metrics(self, df: pd.DataFrame,
                                       season_folder: str) -> pd.DataFrame:
        """Extract xG metrics with error handling for missing columns."""
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
                    print(f"Warning: {df_col} not found in {season_folder}"
                          f", using 0")

        return df

    def _calculate_historical_per_game_metrics(
            self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate per-game metrics for players who played."""
        played_mask = df["games_played"] >= 1

        per_game_cols = [
            'points_per_game', 'goals_per_game', 'assists_per_game',
            'goal_involvements_per_game', 'xg_per_game', 'xa_per_game',
            'xgi_per_game', 'goals_conceded_per_game', 'xgc_per_game'
        ]

        for col in per_game_cols:
            df[col] = 0.0

        if played_mask.any():
            games = df.loc[played_mask, "games_played"]

            df.loc[played_mask, "points_per_game"] = (
                df.loc[played_mask, "total_points"] / games
            )
            df.loc[played_mask, "goals_per_game"] = (
                df.loc[played_mask, "goals_scored"] / games
            )
            df.loc[played_mask, "assists_per_game"] = (
                df.loc[played_mask, "assists"] / games
            )
            df.loc[played_mask, "goal_involvements_per_game"] = (
                (df.loc[played_mask, "goals_scored"] +
                 df.loc[played_mask, "assists"]) / games
            )
            df.loc[played_mask, "xg_per_game"] = (
                df.loc[played_mask, "expected_goals"] / games
            )
            df.loc[played_mask, "xa_per_game"] = (
                df.loc[played_mask, "expected_assists"] / games
            )
            df.loc[played_mask, "xgi_per_game"] = (
                (df.loc[played_mask, "expected_goals"] +
                 df.loc[played_mask, "expected_assists"]) / games
            )
            df.loc[played_mask, "goals_conceded_per_game"] = (
                df.loc[played_mask, "goals_conceded"] / games
            )
            df.loc[played_mask, "xgc_per_game"] = (
                df.loc[played_mask, "expected_goals_conceded"] / games
            )

        return df

    def _calculate_historical_xg_performance(
            self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate position-weighted xG performance ratios."""
        df = df.copy()
        df["historical_xOP"] = 1.0
        df["attacking_xOP_hist"] = 1.0
        df["defensive_xOP_hist"] = 1.0

        attacking_hist_mask = (
            (df["xgi_per_game"] >
             self.historical_thresholds['min_xgi_per_game']) &
            (df["games_played"] >= self.historical_thresholds['min_games'])
            & df["element_type"].isin([2, 3, 4])  # DEF, MID, FWD
        )

        if attacking_hist_mask.any():
            df.loc[attacking_hist_mask, "attacking_xOP_hist"] = (
                df.loc[attacking_hist_mask, "goal_involvements_per_game"] /
                df.loc[attacking_hist_mask, "xgi_per_game"]
            ).clip(0.2, 3.0)

        defensive_hist_mask = (
            (df["xgc_per_game"] >
             self.historical_thresholds['min_xgc_per_game']) &
            (df["games_played"] >= self.historical_thresholds['min_games'])
            & df["element_type"].isin([1, 2, 3])  # GK, DEF, MID
        )

        if defensive_hist_mask.any():
            # For GC, higher ratio = better (conceding less than expected)
            df.loc[defensive_hist_mask, "defensive_xOP_hist"] = (
                df.loc[defensive_hist_mask, "xgc_per_game"] /
                df.loc[defensive_hist_mask,
                       "goals_conceded_per_game"].clip(lower=0.01)
            ).clip(0.2, 3.0)

        has_attacking = attacking_hist_mask
        has_defensive = defensive_hist_mask

        attacking_weight = df["element_type"].map(
            lambda p: self.position_weights.get(
                p, {"attacking": 0.0}
            )["attacking"]
        ).astype(float)
        defensive_weight = df["element_type"].map(
            lambda p: self.position_weights.get(
                p, {"defensive": 0.0}
            )["defensive"]
        ).astype(float)

        use_attacking = has_attacking & (attacking_weight > 0)
        use_defensive = has_defensive & (defensive_weight > 0)

        both = use_attacking & use_defensive
        attacking_only = use_attacking & ~use_defensive
        defensive_only = use_defensive & ~use_attacking

        weighted = pd.Series(1.0, index=df.index)
        weighted[both] = (
            df.loc[both, "attacking_xOP_hist"]
            * attacking_weight[both]
            + df.loc[both, "defensive_xOP_hist"]
            * defensive_weight[both]
        )
        weighted[attacking_only] = df.loc[
            attacking_only, "attacking_xOP_hist"
        ]
        weighted[defensive_only] = df.loc[
            defensive_only, "defensive_xOP_hist"
        ]

        df["historical_xOP"] = weighted.round(2)

        df = df.drop(
            columns=["attacking_xOP_hist", "defensive_xOP_hist"]
        )

        return df.copy()

    def _calculate_weighted_historical_xop(self, row: pd.Series,
                                           position_id: int) -> float:
        """Calculate weighted historical xOP for a single player."""
        weights = self.position_weights[position_id]
        attacking_weight = weights["attacking"]
        defensive_weight = weights["defensive"]

        attacking_xop = row["attacking_xOP_hist"]
        defensive_xop = row["defensive_xOP_hist"]

        has_attacking_data = (
            attacking_weight > 0 and
            row["xgi_per_game"] >
            self.historical_thresholds['min_xgi_per_game']
            and row["games_played"] >=
            self.historical_thresholds['min_games']
        )

        has_defensive_data = (
            defensive_weight > 0 and
            row["xgc_per_game"] >
            self.historical_thresholds['min_xgc_per_game'] and
            row["games_played"] >=
            self.historical_thresholds['min_games']
        )

        if has_attacking_data and has_defensive_data:
            return ((attacking_xop * attacking_weight) +
                    (defensive_xop * defensive_weight))

        elif has_attacking_data and attacking_weight > 0:
            return attacking_xop

        elif has_defensive_data and defensive_weight > 0:
            return defensive_xop

        return 1.0

    def _prepare_output_dataframe(self, df: pd.DataFrame,
                                  season_folder: str) -> pd.DataFrame:
        """Prepare output dataframe with relevant columns.

        ``team_code`` is carried through so that the scoring engine can
        tell whether a player has changed club since the season the
        history was earned in.
        """
        output_cols = [
            "code",
            "web_name",
            "team_code",
            "element_type",
            "points_per_game",
            "games_played",
            "season_reliability",
            "historical_xOP",
            "goal_involvements_per_game",
            "goals_conceded_per_game",
            "xgi_per_game",
            "xgc_per_game"
        ]

        available_cols = [col for col in output_cols if col in df.columns]
        result_df = df[available_cols].copy()

        result_df["player_code"] = pd.to_numeric(
            result_df["code"], errors="coerce"
        ).astype("Int64")
        result_df = result_df.drop(columns=["code"])

        # One row per player per season. Duplicate codes would silently
        # multiply rows in the outer merges below.
        before = len(result_df)
        result_df = result_df.drop_duplicates(subset=["player_code"])
        if len(result_df) != before and self.config.GRANULAR_OUTPUT:
            print(f"Warning: dropped {before - len(result_df)} duplicate "
                  f"player codes from {season_folder}")

        rename_map = {
            "points_per_game": f"ppg_{season_folder}",
            "games_played": f"games_{season_folder}",
            "season_reliability": f"reliability_{season_folder}",
            "historical_xOP": f"historical_xOP_{season_folder}",
            "element_type": f"position_{season_folder}",
            "team_code": f"team_code_{season_folder}",
            "web_name": f"web_name_{season_folder}",
        }

        result_df = result_df.rename(columns=rename_map)
        return result_df

    # ------------------------------------------------------------------
    # xG regression modifiers
    # ------------------------------------------------------------------

    def calculate_xg_performance_modifier(self,
                                          player_data: dict) -> float:
        """
        xG performance modifier with strong penalties for defensive
        overperformance and proper regression logic.

        Args:
            player_data (dict): Player's historical and current xG
                                performance data

        Returns:
            float: Performance modifier (1.0 = neutral, >1.0 = expected
                   improvement, <1.0 = expected decline)
        """
        position = player_data.get('current_position', 'MID')

        historical_baseline = self._calculate_historical_baseline(
            player_data)

        current_xop = player_data.get('current_xOP', 1.0)
        current_xg_context = player_data.get(
            'current_xg_context',
            'insufficient_data'
        )

        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        season_progression = self._calculate_season_progression_factor(
            gameweeks_completed
        )

        if historical_baseline is not None:
            modifier, volatility = self._calculate_regression_modifier(
                historical_baseline, current_xop, season_progression,
                position
            )
        else:
            modifier, volatility = self._calculate_new_player_modifier(
                current_xop, season_progression, current_xg_context
            )

        volatility_penalty = self._calculate_volatility_penalty(volatility)
        final_modifier = 1.0 + (modifier - 1.0) * (
            1.0 - volatility_penalty)

        # 0.4 = very strong penalty, 1.6 = very strong bonus
        return max(0.4, min(1.6, final_modifier))

    def _calculate_historical_baseline(self,
                                       player_data: dict) -> float:
        """Calculate weighted historical baseline from available
        seasons."""
        historical_xop_values = []
        weights = []

        for i, season in enumerate(self.config.PAST_SEASONS):
            season_weight = self.config.HISTORIC_SEASON_WEIGHTS[i]
            xop_key = f"historical_xOP_{season}"

            value = player_data.get(xop_key)
            if value is not None and pd.notna(value) and value > 0:
                historical_xop_values.append(value)
                weights.append(season_weight)

        if historical_xop_values:
            return np.average(historical_xop_values, weights=weights)

        return None

    def _calculate_season_progression_factor(
            self, gameweeks_completed: int) -> float:
        """
        Calculate how reliable current season stats are based on games.

        Returns:
            float: 0.0 = very early (high volatility), 1.0 = late
        """
        if gameweeks_completed <= 3:
            return 0.1
        elif gameweeks_completed <= 6:
            return 0.3
        elif gameweeks_completed <= 10:
            return 0.6
        elif gameweeks_completed <= 15:
            return 0.8
        return 1.0

    def _calculate_regression_modifier(self, historical_baseline: float,
                                       current_xop: float,
                                       season_progression: float,
                                       position: str) -> tuple:
        """
        Calculate regression-based modifier.

        Key principle: extreme deviations from historical baseline are
        almost always unsustainable and require strong regression.

        Returns:
            tuple: (modifier, volatility_score)
        """
        deviation = current_xop - historical_baseline
        abs_deviation = abs(deviation)

        position_sensitivity = {
            'FWD': 1.0,
            'MID': 1.0,
            'DEF': 0.7,
            'GK': 0.7
        }

        sensitivity = position_sensitivity.get(position, 0.9)

        is_defensive_position = position in ['DEF', 'GK']

        if deviation > 0:  # Overperforming historical baseline
            if season_progression < 0.6:
                if is_defensive_position:
                    regression_strength = min(
                        abs_deviation * sensitivity * 0.50, 0.6
                    )
                else:
                    regression_strength = min(
                        abs_deviation * sensitivity * 0.45, 0.55
                    )

                modifier = 1.0 - regression_strength
                volatility = (
                    abs_deviation * (1.0 - season_progression) * 0.3
                )
            else:
                if is_defensive_position:
                    regression_strength = min(
                        abs_deviation * sensitivity * 0.35, 0.45
                    )
                else:
                    # Might be genuine skill for attackers later on
                    regression_strength = min(
                        abs_deviation * sensitivity * 0.25, 0.35
                    )

                modifier = 1.0 - regression_strength
                volatility = abs_deviation * 0.15

        else:
            if season_progression < 0.6:
                regression_strength = min(
                    abs_deviation * sensitivity * 0.40, 0.35
                )
                modifier = 1.0 + regression_strength
                volatility = (
                    abs_deviation * (1.0 - season_progression) * 0.2
                )
            else:
                # Late season - player might genuinely be declining
                regression_strength = min(
                    abs_deviation * sensitivity * 0.25, 0.25
                )
                modifier = 1.0 + regression_strength
                volatility = abs_deviation * 0.12

        return modifier, min(volatility, 0.4)

    def _calculate_new_player_modifier(self, current_xop: float,
                                       season_progression: float,
                                       current_xg_context: str) -> tuple:
        """
        Calculate modifier for new players without historical data.

        Returns:
            tuple: (modifier, volatility_score)
        """
        if current_xg_context == 'insufficient_data':
            return 1.0, 0.2

        deviation_from_neutral = current_xop - 1.0

        if season_progression < 0.6:
            if current_xop > 1.3:
                modifier = 1.0 + (deviation_from_neutral * 0.1)
                volatility = 0.35
            elif current_xop < 0.7:
                modifier = 1.0 + (deviation_from_neutral * 0.1)
                volatility = 0.3
            else:
                modifier = 1.0 + (deviation_from_neutral * 0.15)
                volatility = 0.2
        else:
            if current_xop > 1.2:
                modifier = 1.0 + (deviation_from_neutral * 0.2)
                volatility = 0.15
            elif current_xop < 0.8:
                modifier = 1.0 + (deviation_from_neutral * 0.2)
                volatility = 0.2
            else:
                modifier = 1.0 + (deviation_from_neutral * 0.25)
                volatility = 0.1

        return modifier, volatility

    def _calculate_volatility_penalty(self, volatility: float) -> float:
        """
        Apply a volatility penalty to pull the modifier toward neutral.
        Higher volatility means less confidence in the deviation.

        Args:
            volatility (float): Volatility score (0.0 to 0.4)

        Returns:
            float: Penalty factor (0.0 to 0.4)
        """
        base_penalty = volatility * 0.8

        if volatility > 0.25:
            exponential_bonus = (volatility - 0.25) ** 1.3 * 0.5
            base_penalty += exponential_bonus

        return min(base_penalty, 0.4)

    def calculate_data_availability_factor(self,
                                           player_data: dict) -> float:
        """
        Calculate factor to avoid over-penalising players with limited
        historical data.

        Args:
            player_data (dict): Player's historical data

        Returns:
            float: Factor to reduce the impact of the xG modifier
        """
        seasons_with_data = 0
        total_games = 0

        for season in self.config.PAST_SEASONS:
            games_key = f"games_{season}"
            games = player_data.get(games_key)
            if (games is not None and pd.notna(games)
                    and games >= self.historical_thresholds[
                        'min_games']):
                seasons_with_data += 1
                total_games += games

        if seasons_with_data == 0:
            return 0.1
        elif seasons_with_data == 1:
            return 0.5

        games_factor = min(total_games / 60, 1.0)
        return 0.7 + (0.3 * games_factor)

    # ------------------------------------------------------------------
    # Current season integration
    # ------------------------------------------------------------------

    def _get_current_season_data(self,
                                 current_players: pd.DataFrame) -> dict:
        """
        Extract current season data in historical format once enough
        gameweeks have been played.

        Args:
            current_players (pd.DataFrame): Current season player data

        Returns:
            dict: Current season data keyed by player_code
        """
        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        integration_gw = getattr(
            self.config, "CURRENT_SEASON_INTEGRATION_GW", 8
        )

        if gameweeks_completed < integration_gw:
            return {}

        current_season_data = {}
        season = self.current_season

        for _, player in current_players.iterrows():
            player_code = player.get("player_code")
            if pd.isna(player_code):
                continue

            total_points = player.get("total_points", 0) or 0
            fpl_ppg = pd.to_numeric(
                player.get("points_per_game"), errors="coerce"
            )

            if pd.notna(fpl_ppg) and fpl_ppg > 0:
                games_played = round(total_points / fpl_ppg)
            else:
                minutes_played = player.get("minutes", 0) or 0
                starts = player.get("starts", 0) or 0
                games_played = max(starts, round(minutes_played / 90))

            if games_played >= self.historical_thresholds['min_games']:
                points_per_game = total_points / games_played

                goals = player.get("goals_scored", 0)
                assists = player.get("assists", 0)
                xg = player.get("expected_goals", 0)
                xa = player.get("expected_assists", 0)

                goal_involvements_per_game = (
                    (goals + assists) / games_played
                )
                xgi_per_game = (xg + xa) / games_played

                historical_xop = 1.0
                if xgi_per_game > 0.1:
                    historical_xop = (goal_involvements_per_game /
                                      xgi_per_game)
                    historical_xop = max(0.2, min(3.0, historical_xop))

                reliability_baseline = getattr(
                    self.config, "RELIABILITY_GAMES_BASELINE", 30
                )

                current_season_data[int(player_code)] = {
                    f"ppg_{season}": points_per_game,
                    f"games_{season}": games_played,
                    f"reliability_{season}": min(
                        1.0, games_played / reliability_baseline),
                    f"historical_xOP_{season}": historical_xop,
                    f"position_{season}": player.get("pos_id", 3),
                    f"team_code_{season}": player.get("team_code"),
                }

        return current_season_data

    def _integrate_current_season_data(self, hist: pd.DataFrame,
                                       current_season_data: dict
                                       ) -> pd.DataFrame:
        """
        Integrate current season data into the historical dataframe.

        Args:
            hist (pd.DataFrame): Historical data
            current_season_data (dict): Current season data by code

        Returns:
            pd.DataFrame: Historical data with current season integrated
        """
        if not current_season_data:
            return hist

        current_df_data = []
        for player_code, data in current_season_data.items():
            row_data = {"player_code": player_code}
            row_data.update(data)
            current_df_data.append(row_data)

        if not current_df_data:
            return hist

        current_df = pd.DataFrame(current_df_data)
        current_df["player_code"] = current_df["player_code"].astype(
            "Int64"
        )

        hist = hist.merge(
            current_df, on="player_code", how="outer", validate="1:1"
        )

        return hist

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def merge_past_seasons(self, current: pd.DataFrame) -> pd.DataFrame:
        """
        Merge historical seasons onto the current player set, joining on
        the permanent FPL player code.

        Args:
            current (pd.DataFrame): Current season player data.

        Returns:
            pd.DataFrame: Current player data enriched with weighted
                          historical averages, xG modifiers and a
                          ``changed_club`` flag.
        """
        if "player_code" not in current.columns:
            raise ValueError(
                "Current player data has no 'player_code' column. "
                "PlayerProcessor.fetch_current_players() must add it."
            )

        hist_frames = [
            self.fetch_past_season_points(s)
            for s in self.config.PAST_SEASONS
        ]

        hist = self._merge_historical_frames(hist_frames)

        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        integration_gw = getattr(
            self.config, "CURRENT_SEASON_INTEGRATION_GW", 8
        )
        include_current = gameweeks_completed >= integration_gw

        if include_current:
            current_season_data = self._get_current_season_data(current)
            hist = self._integrate_current_season_data(
                hist, current_season_data
            )

        hist = self._calculate_weighted_historical_averages(
            hist, include_current
        )

        hist = self._resolve_historical_team(hist, include_current)

        hist = self._calculate_xg_consistency_modifiers(hist, current)

        current_reliability = self._calculate_current_reliability(current)
        current = current.assign(current_reliability=current_reliability)

        merge_cols = [
            "player_code", "avg_ppg_past2", "total_games_past2",
            "avg_reliability", "historical_xOP", "xConsistency",
            "xOP_historical_baseline", "hist_team_code",
            "seasons_with_history"
        ]

        before = len(current)
        merged = current.merge(
            hist[merge_cols], on="player_code", how="left", validate="1:1"
        )

        if len(merged) != before:
            raise ValueError(
                f"Historical merge changed row count from {before} to "
                f"{len(merged)} - player codes are not unique."
            )

        # A player whose recorded historical club differs from their
        # current club earned that history somewhere else.
        merged["changed_club"] = (
            merged["hist_team_code"].notna()
            & (merged["hist_team_code"] != merged["team_code"])
        )

        matched = merged["avg_ppg_past2"].notna().sum()
        if self.config.GRANULAR_OUTPUT:
            if include_current:
                print("Current season data integrated into historical "
                      f"analysis after {gameweeks_completed} completed "
                      "gameweeks")
            print("Calculated reliability based on starts over "
                  f"{gameweeks_completed} completed gameweek(s)")
            print(f"Historical join: {matched}/{len(merged)} players "
                  "matched on player code")
            print(f"Players who have changed club since their last "
                  f"recorded season: {int(merged['changed_club'].sum())}")

        return merged

    def _merge_historical_frames(self, hist_frames: list) -> pd.DataFrame:
        """Merge multiple historical dataframes on player_code."""
        hist = hist_frames[0].copy()

        for i, extra in enumerate(hist_frames[1:], 1):
            season = self.config.PAST_SEASONS[i]

            season_specific_cols = ["player_code"]
            for col in extra.columns:
                if col.endswith(f"_{season}"):
                    season_specific_cols.append(col)

            extra_filtered = extra[season_specific_cols].copy()
            hist = hist.merge(
                extra_filtered, on="player_code", how="outer",
                validate="1:1"
            )

        return hist

    def _season_weighting(self, include_current: bool) -> tuple:
        """
        Build the ordered season list and matching weights.

        Returns:
            tuple: (seasons, weights) with weights summing to 1.0
        """
        seasons = list(self.config.PAST_SEASONS)
        weights = list(self.config.HISTORIC_SEASON_WEIGHTS)

        if include_current:
            gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
            max_weight = getattr(
                self.config, "CURRENT_SEASON_MAX_WEIGHT", 0.6
            )
            current_weight = min(max_weight, gameweeks_completed / 38)

            seasons = [self.current_season] + seasons
            weights = (
                [current_weight]
                + [w * (1 - current_weight) for w in weights]
            )

        return seasons, weights

    def _calculate_weighted_historical_averages(
            self, hist: pd.DataFrame,
            include_current: bool) -> pd.DataFrame:
        """
        Calculate weighted historical averages, renormalising the season
        weights across whichever seasons a player actually has data for.

        Without renormalisation a player with one qualifying season
        received their points per game multiplied by that season's raw
        weight (0.5) and nothing else, which caps every recent arrival at
        roughly half the score of an established player regardless of how
        well they played.
        """
        seasons, weights = self._season_weighting(include_current)

        hist["avg_ppg_past2"] = 0.0
        hist["total_games_past2"] = 0.0
        hist["avg_reliability"] = 0.0
        hist["historical_xOP"] = 0.0
        hist["seasons_with_history"] = 0

        weight_used = pd.Series(0.0, index=hist.index)
        xop_weight_used = pd.Series(0.0, index=hist.index)

        min_games = self.historical_thresholds['min_games']

        for season, weight in zip(seasons, weights):
            ppg_col = f"ppg_{season}"
            games_col = f"games_{season}"
            reliability_col = f"reliability_{season}"
            xop_col = f"historical_xOP_{season}"

            if games_col not in hist.columns:
                continue

            games = pd.to_numeric(
                hist[games_col], errors="coerce"
            ).fillna(0)
            hist["total_games_past2"] += games

            qualifies = games >= min_games
            if not qualifies.any():
                continue

            hist["seasons_with_history"] += qualifies.astype(int)

            if ppg_col in hist.columns:
                ppg = pd.to_numeric(
                    hist[ppg_col], errors="coerce"
                ).fillna(0)
                hist.loc[qualifies, "avg_ppg_past2"] += (
                    ppg[qualifies] * weight
                )

            if reliability_col in hist.columns:
                reliability = pd.to_numeric(
                    hist[reliability_col], errors="coerce"
                ).fillna(0)
                hist.loc[qualifies, "avg_reliability"] += (
                    reliability[qualifies] * weight
                )

            weight_used.loc[qualifies] += weight

            if xop_col in hist.columns:
                xop = pd.to_numeric(
                    hist[xop_col], errors="coerce"
                ).fillna(1.0)
                valid_xop = qualifies & (xop > 0)
                hist.loc[valid_xop, "historical_xOP"] += (
                    xop[valid_xop] * weight
                )
                xop_weight_used.loc[valid_xop] += weight

        # Renormalise so the averages are true weighted means over the
        # seasons each player actually has.
        used = weight_used > 0
        hist.loc[used, "avg_ppg_past2"] /= weight_used[used]
        hist.loc[used, "avg_reliability"] /= weight_used[used]

        xop_used = xop_weight_used > 0
        hist.loc[xop_used, "historical_xOP"] /= xop_weight_used[xop_used]
        hist.loc[~xop_used, "historical_xOP"] = 1.0

        # Players with no qualifying season have no historical estimate.
        # NaN keeps them out of the historic z-score population rather
        # than entering it as a genuine average of zero points per game.
        hist.loc[~used, "avg_ppg_past2"] = np.nan
        hist.loc[~used, "avg_reliability"] = np.nan

        hist["historical_xOP"] = hist["historical_xOP"].round(2)
        hist["seasons_with_history"] = (
            hist["seasons_with_history"].fillna(0).astype(int)
        )

        return hist

    def _resolve_historical_team(self, hist: pd.DataFrame,
                                 include_current: bool) -> pd.DataFrame:
        """
        Record the club a player's most recent qualifying season was
        played for, so club changes can be detected downstream.
        """
        seasons, _ = self._season_weighting(include_current)

        # Most recent season first.
        ordered = sorted(seasons, reverse=True)

        hist["hist_team_code"] = pd.NA
        min_games = self.historical_thresholds['min_games']

        for season in ordered:
            team_col = f"team_code_{season}"
            games_col = f"games_{season}"

            if team_col not in hist.columns:
                continue

            games = (
                pd.to_numeric(hist[games_col], errors="coerce").fillna(0)
                if games_col in hist.columns
                else pd.Series(0.0, index=hist.index)
            )

            candidate = pd.to_numeric(
                hist[team_col], errors="coerce"
            ).astype("Int64")

            fill = (
                hist["hist_team_code"].isna()
                & candidate.notna()
                & (games >= min_games)
            )
            hist.loc[fill, "hist_team_code"] = candidate[fill]

        hist["hist_team_code"] = hist["hist_team_code"].astype("Int64")

        return hist

    def _calculate_xg_consistency_modifiers(
            self, hist: pd.DataFrame,
            current: pd.DataFrame) -> pd.DataFrame:
        """Calculate xG performance modifiers for each player.

        Args:
            hist: Historical dataframe with historical_xOP
            current: Current season dataframe with current_xOP
        """
        modifiers = []
        baselines = []

        current_xop_lookup = {}
        current_xg_context_lookup = {}
        if ('current_xOP' in current.columns
                and 'player_code' in current.columns):
            for _, row in current.iterrows():
                code = row['player_code']
                if pd.isna(code):
                    continue
                code = int(code)
                current_xop_lookup[code] = row.get('current_xOP', 1.0)
                current_xg_context_lookup[code] = row.get(
                    'current_xg_context', 'insufficient_data')

        for idx, row in hist.iterrows():
            player_data = row.to_dict()
            player_data['current_position'] = self._get_player_position(
                row)

            code = row.get('player_code')
            code = int(code) if pd.notna(code) else None

            if code in current_xop_lookup:
                player_data['current_xOP'] = current_xop_lookup[code]
                player_data['current_xg_context'] = (
                    current_xg_context_lookup[code])
            else:
                player_data['current_xOP'] = 1.0
                player_data['current_xg_context'] = 'insufficient_data'

            has_historical_data = self._player_has_historical_data(
                player_data)

            xg_modifier = self.calculate_xg_performance_modifier(
                player_data)

            if has_historical_data:
                baselines.append(row.get("historical_xOP", 1.0))
            else:
                baselines.append(np.nan)

            confidence = self._modifier_confidence(
                player_data, has_historical_data
            )
            final_modifier = 1.0 + (xg_modifier - 1.0) * confidence

            modifiers.append(round(final_modifier, 2))

        hist["xConsistency"] = modifiers
        hist["xOP_historical_baseline"] = baselines

        return hist.copy()

    def _modifier_confidence(self, player_data: dict,
                             has_historical_data: bool) -> float:
        """
        Weight the xG regression modifier by how much evidence it
        actually rests on.

        The modifier compares this season's conversion rate against a
        historical baseline. Three gameweeks of expected goals is noise,
        so at that point the comparison should barely move a player's
        score. Previously the modifier ran at full strength from GW2 and
        was the single largest term in base quality, handing large
        bonuses to anyone whose finishing had been unlucky in two games.

        Returns:
            float: 0.0 (ignore the modifier) to 1.0 (trust it fully)
        """
        gameweeks_completed = max(1, self.config.GAMEWEEK - 1)
        season_progression = self._calculate_season_progression_factor(
            gameweeks_completed
        )

        availability = self.calculate_data_availability_factor(
            player_data
        )

        confidence = season_progression * availability

        if not has_historical_data:
            confidence *= 0.7

        floor = getattr(
            self.config, "XG_MODIFIER_CONFIDENCE_FLOOR", 0.0
        )
        return float(min(1.0, max(floor, confidence)))

    def _player_has_historical_data(self, player_data: dict) -> bool:
        """Check if player has historical xOP data."""
        min_games = self.historical_thresholds['min_games']

        for season in self.config.PAST_SEASONS:
            xop = player_data.get(f"historical_xOP_{season}")
            games = player_data.get(f"games_{season}")

            if (xop is not None and games is not None
                    and pd.notna(xop) and pd.notna(games)
                    and games >= min_games and xop > 0):
                return True
        return False

    def _calculate_current_reliability(
            self, current: pd.DataFrame) -> pd.Series:
        """
        Calculate current season reliability.

        Divided by each club's own matches played rather than a global
        gameweek count, so a club that has already played this week does
        not read as more reliable than one that has not.
        """
        fallback = float(max(1, self.config.GAMEWEEK - 1))

        if ("team_matches_played" in current.columns
                and getattr(self.config, "NORMALISE_PARTIAL_GAMEWEEK",
                            True)):
            matches = pd.to_numeric(
                current["team_matches_played"], errors="coerce"
            ).fillna(fallback).clip(lower=1)
        else:
            matches = pd.Series(fallback, index=current.index)

        starts = pd.to_numeric(
            current["starts"], errors="coerce"
        ).fillna(0)
        return (starts / matches).clip(upper=1.0)

    def _get_player_position(self, player_row: pd.Series) -> str:
        """
        Determine a player's position from historical data.

        Args:
            player_row (pd.Series): Player's historical data row

        Returns:
            str: Player position (GK, DEF, MID, FWD)
        """
        all_seasons = (
            [self.current_season] + list(self.config.PAST_SEASONS)
        )
        pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

        for season in sorted(all_seasons, reverse=True):
            pos_key = f"position_{season}"
            if pos_key in player_row and pd.notna(player_row[pos_key]):
                try:
                    pos_num = int(player_row[pos_key])
                except (TypeError, ValueError):
                    continue
                return pos_map.get(pos_num, "MID")

        return "MID"