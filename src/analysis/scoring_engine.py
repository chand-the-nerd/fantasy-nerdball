"""Module for calculating player scores based on form, history and
fixtures.
"""

import numpy as np
import pandas as pd


class ScoringEngine:
    """Handles the calculation of FPL scores for players."""

    def __init__(self, config):
        self.config = config

    def build_scores(self, players: pd.DataFrame,
                     fixture_scores: pd.DataFrame) -> tuple:
        """
        Calculate FPL scores and base quality scores for each player
        based on form, historical performance and fixture difficulty,
        with reliability considerations and early season penalties.

        Args:
            players (pd.DataFrame): Player data with form and historic
                                    points per game.
            fixture_scores (pd.DataFrame): Fixture difficulty data.

        Returns:
            tuple: (scored_dataframe, involvement_stats_dict)
        """
        before = len(players)
        df = players.merge(
            fixture_scores, on="player_code", how="left", validate="1:1"
        )

        if len(df) != before:
            raise ValueError(
                f"Fixture merge changed row count from {before} to "
                f"{len(df)} - fixture scores are not unique per player."
            )

        df = self._fill_missing_values(df)

        df = self._calculate_team_adjustments(df)

        # Discount historical output earned at a previous club.
        df = self._apply_club_change_adjustment(df)

        df = self._calculate_base_quality_adaptive(df)

        df = self._apply_reliability_adjustments(df)

        df = self._calculate_projected_points(df)

        df, involvement_stats = self._apply_availability_filter(df)

        df = self._finalise_scores(df)

        return df, involvement_stats

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values to prevent errors in calculations.

        ``avg_ppg_past2`` is deliberately left as NaN where a player has
        no qualifying history, so those players stay out of the historic
        z-score population instead of entering it as a genuine average of
        zero points per game.
        """
        fill_values = {
            "avg_reliability": 0.0,
            "current_reliability": 0.0,
            "fixture_bonus": 0.0,
            "xConsistency": 1.0,
            "form_consistency": 1.0,
            "changed_club": False,
        }

        for col, fill_val in fill_values.items():
            if col not in df.columns:
                df[col] = fill_val
            else:
                df[col] = df[col].fillna(fill_val)

        if "avg_ppg_past2" not in df.columns:
            df["avg_ppg_past2"] = np.nan

        df["changed_club"] = df["changed_club"].astype(bool)

        return df

    def _calculate_team_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate team and promotion adjustments."""
        promoted_penalty = getattr(
            self.config, "PROMOTED_PENALTY", -0.3
        )

        df["promoted_penalty"] = df["team"].apply(
            lambda x: promoted_penalty
            if x in self.config.PROMOTED_TEAMS else 0.0
        )
        df["team_modifier"] = df["team"].map(
            lambda t: self.config.TEAM_MODIFIERS.get(t, 1.0)
        )
        return df

    def _apply_club_change_adjustment(
            self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Shrink historical points per game toward the positional mean for
        players who have changed club.

        Historic output is a joint product of the player and the side
        they played in, so it is weaker evidence once they move. The
        shrinkage relaxes as gameweeks at the new club accumulate, at
        which point current-season form carries the signal instead.
        """
        shrink_floor = getattr(self.config, "TRANSFER_SHRINK", 0.55)
        recovery_gws = getattr(self.config, "TRANSFER_RECOVERY_GWS", 6)

        df["club_change_shrink"] = 1.0

        movers = df["changed_club"] & df["avg_ppg_past2"].notna()
        if not movers.any():
            return df

        gameweeks_completed = max(0, self.config.GAMEWEEK - 1)
        recovery = min(1.0, gameweeks_completed / max(1, recovery_gws))
        shrink = shrink_floor + (1.0 - shrink_floor) * recovery

        position_mean = df.groupby("position")[
            "avg_ppg_past2"
        ].transform("mean")

        df.loc[movers, "club_change_shrink"] = shrink
        df.loc[movers, "avg_ppg_past2"] = (
            position_mean[movers]
            + (df.loc[movers, "avg_ppg_past2"] - position_mean[movers])
            * shrink
        )

        if self.config.GRANULAR_OUTPUT:
            print(f"   Applied a {shrink:.2f} historic weighting to "
                  f"{int(movers.sum())} players who changed club")

        return df

    # ------------------------------------------------------------------
    # Base quality
    # ------------------------------------------------------------------

    def _matches_played(self, df: pd.DataFrame) -> pd.Series:
        """
        Matches actually played by each player's club.

        Used as the denominator for every rate built from a cumulative
        total. Falls back to a global ``GAMEWEEK - 1`` if the column is
        absent, which keeps older callers working but reintroduces the
        mid-gameweek skew, so PlayerProcessor should always supply it.
        """
        fallback = float(max(1, self.config.GAMEWEEK - 1))

        if not getattr(self.config, "NORMALISE_PARTIAL_GAMEWEEK", True):
            return pd.Series(fallback, index=df.index)

        if "team_matches_played" not in df.columns:
            return pd.Series(fallback, index=df.index)

        matches = pd.to_numeric(
            df["team_matches_played"], errors="coerce"
        )
        return matches.fillna(fallback).clip(lower=1).astype(float)

    def _scoring_pool_mask(self, df: pd.DataFrame) -> pd.Series:
        """
        Identify the players who should define the z-score distribution.

        Normalising across the whole pool - which includes several
        hundred squad players with no minutes - compresses the genuine
        contenders into a narrow band and shifts the mean week to week as
        the pool changes.
        """
        if self.config.GAMEWEEK <= 1:
            return pd.Series(True, index=df.index)

        minutes = pd.to_numeric(
            df.get("minutes", 0), errors="coerce"
        ).fillna(0)
        pool = minutes > 0

        # Fall back to the whole pool if the filter leaves too little to
        # estimate a distribution from.
        if pool.sum() < 30:
            return pd.Series(True, index=df.index)

        return pool

    def _z_score(self, series: pd.Series,
                 valid_mask: pd.Series = None) -> pd.Series:
        """
        Z-score normalisation with NaN and infinity handling.

        Args:
            series (pd.Series): Series to normalise
            valid_mask (pd.Series): Rows that define the distribution.
                                    Rows outside it are still scored, but
                                    do not influence the mean or standard
                                    deviation.

        Returns:
            pd.Series: Normalised series
        """
        values = pd.to_numeric(series, errors="coerce")

        if valid_mask is None:
            valid_mask = values.notna()
        else:
            valid_mask = valid_mask & values.notna()

        if valid_mask.sum() < 2:
            return pd.Series(0.0, index=series.index)

        reference = values[valid_mask]
        mean = reference.mean()
        std = reference.std(ddof=0)

        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=series.index)

        z_scores = (values - mean) / std
        z_scores = z_scores.replace(
            [float("inf"), float("-inf")], np.nan
        )
        return z_scores.fillna(0.0)

    def _z_score_by_position(self, df: pd.DataFrame, column: str,
                             valid_mask: pd.Series) -> pd.Series:
        """Z-score a column separately within each position."""
        result = pd.Series(0.0, index=df.index)

        for position in df["position"].dropna().unique():
            position_mask = df["position"] == position
            if not position_mask.any():
                continue

            subset = df.loc[position_mask, column]
            subset_valid = valid_mask.loc[position_mask]

            result.loc[position_mask] = self._z_score(
                subset, subset_valid
            )

        return result

    def _calculate_early_season_penalty(self) -> float:
        """
        Calculate the early season divisor applied to the form
        contribution, decaying to 1.0 over the first few gameweeks.

        Returns:
            float: Divisor for the form contribution
        """
        current_gameweek = self.config.GAMEWEEK

        if current_gameweek <= 1:
            return 1.0  # No penalty in GW1 (form is 0 anyway)

        penalty_end_gw = (
            2 + self.config.EARLY_SEASON_PENALTY_GAMEWEEKS
        )
        if current_gameweek >= penalty_end_gw:
            return 1.0

        initial_divisor = self.config.EARLY_SEASON_PENALTY_INITIAL
        decay_factor = self.config.EARLY_SEASON_DECAY_FACTOR

        penalty_steps = current_gameweek - 2
        current_divisor = initial_divisor * (
            decay_factor ** penalty_steps
        )

        return max(1.0, current_divisor)

    def _position_weight_series(self, df: pd.DataFrame,
                                key: str) -> pd.Series:
        """Map a position-specific scoring weight onto every row."""
        weights = self.config.POSITION_SCORING_WEIGHTS
        default = weights["MID"]

        return df["position"].map(
            lambda p: weights.get(p, default)[key]
        ).astype(float)

    def _calculate_base_quality_adaptive(
            self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate base quality with position-specific weighting, an early
        season penalty on the form contribution, and additive modifiers.
        """
        early_season_penalty = self._calculate_early_season_penalty()
        df["form_adjusted"] = df["form"] / early_season_penalty

        pool = self._scoring_pool_mask(df)

        # Z-scores are computed within position so that, for example, a
        # goalkeeper is compared against goalkeepers rather than against
        # the whole league.
        form_z = self._z_score_by_position(df, "form", pool)
        historic_z = self._z_score_by_position(
            df, "avg_ppg_past2", df["avg_ppg_past2"].notna()
        )
        fixture_z = self._z_score_by_position(df, "fixture_bonus", pool)

        has_history = df["avg_ppg_past2"].notna()

        form_weight = self._position_weight_series(df, "form")
        historic_weight = self._position_weight_series(df, "historic")
        fixture_weight = self._position_weight_series(df, "difficulty")

        # Players with no usable history put the historic weight onto
        # their form instead.
        form_weight = form_weight.where(
            has_history, 1.0 - fixture_weight
        )
        historic_weight = historic_weight.where(has_history, 0.0)

        form_consistency_scale = getattr(
            self.config, "FORM_CONSISTENCY_SCALE", 1.0
        )
        team_modifier_scale = getattr(
            self.config, "TEAM_MODIFIER_SCALE", 2.0
        )
        xconsistency_scale = getattr(
            self.config, "XCONSISTENCY_SCALE", 2.0
        )

        form_contribution = (
            form_z * form_weight
        ) / early_season_penalty

        # Additive so that a consistency bonus always raises the score
        # and a penalty always lowers it, whatever the sign of form_z.
        form_contribution = form_contribution + (
            (df["form_consistency"] - 1.0)
            * form_consistency_scale
            * form_weight
        )

        base_quality = (
            form_contribution
            + (historic_z * historic_weight)
            + (fixture_z * fixture_weight)
        )

        base_quality = base_quality + (
            (df["team_modifier"] - 1.0) * team_modifier_scale
        )
        base_quality = base_quality + (
            (df["xConsistency"] - 1.0) * xconsistency_scale
        )

        # The promotion penalty is added last so it is not scaled by any
        # of the modifiers above.
        base_quality = base_quality + df["promoted_penalty"]

        df["form_z"] = form_z
        df["historic_z"] = historic_z
        df["fixture_z"] = fixture_z
        df["base_quality"] = base_quality

        return df

    # ------------------------------------------------------------------
    # Reliability and points
    # ------------------------------------------------------------------

    def _apply_reliability_adjustments(
            self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply reliability adjustments for squad selection."""
        avg_reliability = df["avg_reliability"].fillna(0.0)

        reliability_bonus = (
            df["current_reliability"] * 1.5
            + avg_reliability * 0.3
        ) - 0.75  # Centre around 0

        df["historically_unreliable_penalty"] = 0.0

        unreliable_mask = avg_reliability < 0.6
        df.loc[unreliable_mask, "historically_unreliable_penalty"] = -0.15

        if self.config.GAMEWEEK > 1:
            matches_played = self._matches_played(df)

            current_start_pct = (
                df["starts"].fillna(0) / matches_played
            ).clip(upper=1.0)

            very_unreliable_mask = current_start_pct < 0.5
            df.loc[very_unreliable_mask,
                   "historically_unreliable_penalty"] -= 0.3

            moderately_unreliable_mask = (
                (current_start_pct >= 0.5) & (current_start_pct < 0.7)
            )
            df.loc[moderately_unreliable_mask,
                   "historically_unreliable_penalty"] -= 0.15
        else:
            current_unreliable_mask = df["current_reliability"] < 0.7
            df.loc[current_unreliable_mask,
                   "historically_unreliable_penalty"] -= 0.2

        df["fpl_score"] = (
            df["base_quality"] + reliability_bonus +
            df["historically_unreliable_penalty"]
        )

        return df

    def _calculate_projected_points(
            self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate projected points for each player based on their base
        quality.
        """
        if df["position"].dtype.name == "category":
            position_values = df["position"].astype(str)
        else:
            position_values = df["position"]

        df["baseline_points"] = position_values.map(
            self.config.BASELINE_POINTS_PER_GAME
        )

        df["points_adjustment"] = (
            df["base_quality"] * self.config.FPL_SCORE_TO_POINTS_MULTIPLIER
        )

        df["projected_points"] = (
            df["baseline_points"] + df["points_adjustment"]
        )

        df["projected_points"] = df["projected_points"].clip(lower=1.0)

        # Blank gameweek handling. fixture_multiplier is per gameweek and
        # is 0.0 for a blank, 1.0 for a normal week and 2.0 for a double.
        if "fixture_multiplier" in df.columns:
            df["fixture_multiplier"] = df["fixture_multiplier"].fillna(1.0)

            df["projected_points"] = (
                df["projected_points"] * df["fixture_multiplier"]
            )

            df["projected_points"] = df["projected_points"].fillna(0.0)
            df["projected_points"] = df["projected_points"].replace(
                [float("inf"), float("-inf")], 0.0
            )

            bgw_mask = df["fixture_multiplier"] == 0.0
            df.loc[~bgw_mask, "projected_points"] = (
                df.loc[~bgw_mask, "projected_points"].clip(lower=1.0)
            )
            df.loc[bgw_mask, "projected_points"] = 0.0

        return df

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def _apply_availability_filter(self, df: pd.DataFrame) -> tuple:
        """
        Apply the availability filter to players based on config.

        Args:
            df (pd.DataFrame): Player dataframe with scores

        Returns:
            tuple: (modified_dataframe, involvement_stats_dict)
        """
        involvement_stats = {
            'high_involvement': 0,
            'zero_involvement': 0,
            'low_involvement': 0,
            'unavailable': 0,
            'exclude_unavailable': True
        }

        exclude_unavailable = getattr(
            self.config, 'EXCLUDE_UNAVAILABLE', True
        )
        involvement_stats['exclude_unavailable'] = exclude_unavailable

        if not exclude_unavailable:
            return df, involvement_stats

        unavailable_mask = (
            (df["status"] != "a") &
            (df["chance_of_playing_next_round"].fillna(100) < 75)
        )

        involvement_stats['unavailable'] = int(unavailable_mask.sum())

        if self.config.GAMEWEEK > 1:
            matches_played = self._matches_played(df)

            zero_involvement_mask = (
                (df["starts"].fillna(0) == 0) &
                (df["form"].fillna(0) == 0) &
                (df["minutes"].fillna(0) < 45)
            )

            low_involvement_mask = (
                (df["starts"].fillna(0) / matches_played < 0.3) &
                (df["minutes"].fillna(0) / matches_played < 30) &
                ~zero_involvement_mask
            )

            high_involvement_mask = (
                (df["starts"].fillna(0) / matches_played >= 0.7) &
                (df["minutes"].fillna(0) / matches_played >= 60)
            )

            involvement_stats['high_involvement'] = int(
                high_involvement_mask.sum()
            )
            involvement_stats['zero_involvement'] = int(
                zero_involvement_mask.sum()
            )
            involvement_stats['low_involvement'] = int(
                low_involvement_mask.sum()
            )

            if zero_involvement_mask.any():
                df.loc[zero_involvement_mask, "fpl_score"] = 0.0
                df.loc[zero_involvement_mask, "projected_points"] = 0.0

            if low_involvement_mask.any():
                # A 75% reduction. Applied to the score rather than to
                # any component, so it cannot flip a sign.
                df.loc[low_involvement_mask, "fpl_score"] = (
                    df.loc[low_involvement_mask, "fpl_score"].clip(
                        lower=0.0
                    ) * 0.25
                )
                df.loc[low_involvement_mask, "projected_points"] *= 0.25

        if unavailable_mask.any():
            df.loc[unavailable_mask, "fpl_score"] = 0.0
            df.loc[unavailable_mask, "projected_points"] = 0.0

        return df, involvement_stats

    def _finalise_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final safety check - replace any remaining NaN or infinity."""
        score_columns = [
            "base_quality", "fpl_score", "projected_points"
        ]

        for col in score_columns:
            if col in df.columns:
                df[col] = df[col].replace(
                    [float("inf"), float("-inf")], 0.0
                )
                df[col] = df[col].fillna(0.0)

        # Restore a displayable value for players with no history now
        # that all z-scoring is complete.
        if "avg_ppg_past2" in df.columns:
            df["avg_ppg_past2"] = df["avg_ppg_past2"].fillna(0.0)
        if "avg_reliability" in df.columns:
            df["avg_reliability"] = df["avg_reliability"].fillna(0.0)

        return df