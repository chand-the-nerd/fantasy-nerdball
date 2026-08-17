"""
Recalculates a historic FPL season's player points using the CURRENT
season's full scoring rules (see current_rules.py) - goals, assists, clean
sheets, cards, saves, penalties and Defensive Contribution (DEFCON) - so
that historic points used elsewhere in this project reflect this season's
rules rather than the rules the points were originally awarded under.

Bonus points are left untouched, since they're derived from BPS rank
within a match, which hasn't changed between seasons.

DEFCON recalculation requires per-match defensive stats
(clearances_blocks_interceptions, tackles, recoveries), which vaastav's
Fantasy-Premier-League data only records from the 2025-26 season onwards.
For earlier seasons this component simply nets to 0 (no data either way),
while every other rule difference (e.g. goalkeeper goal points) is still
recalculated correctly.
"""

import os
import pandas as pd

from current_rules import CurrentRules

OUTPUT_DIR = "data/historic"
PLAYERS_RAW_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    "master/data/{season}/players_raw.csv"
)
GW_DATA_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    "master/data/{season}/gws/gw{gw}.csv"
)
MAX_GAMEWEEKS = 38

# players_raw.csv element_type codes
POSITION_BY_ELEMENT_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


class HistoricPointsRecalculator:
    """Rebuilds a historic season's player totals as if the current
    season's scoring rules had applied throughout."""

    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def output_path(self, season_folder: str) -> str:
        return os.path.join(
            OUTPUT_DIR, f"{season_folder}_updated_points.csv"
        )

    def recalculate_season(self, season_folder: str,
                          force: bool = False) -> pd.DataFrame:
        """
        Recalculate a season's player totals under current_rules.py and
        save them to data/historic/{season_folder}_updated_points.csv.

        Args:
            season_folder (str): Season identifier, e.g. "2025-26".
            force (bool): Recalculate even if an output file already
                exists.

        Returns:
            pd.DataFrame or None: The recalculated season data (in the
                same shape as players_raw.csv), or None if no gameweek
                data could be found for this season at all.
        """
        output_path = self.output_path(season_folder)
        if not force and os.path.exists(output_path):
            print(f"{output_path} already exists, skipping "
                  f"(use force=True to recalculate anyway)")
            return pd.read_csv(output_path)

        gw_data = self._fetch_all_gameweeks(season_folder)
        if gw_data is None:
            return None

        players_raw = pd.read_csv(
            PLAYERS_RAW_URL.format(season=season_folder)
        )

        recalculated_totals = self._recalculate_totals(
            gw_data, season_folder
        )

        updated = players_raw.merge(
            recalculated_totals, on="id", how="left",
            suffixes=("_original", "")
        )
        # Players with no gameweek data (e.g. never played) keep their
        # original total_points (0) rather than becoming NaN.
        updated["total_points"] = updated["total_points"].fillna(
            updated["total_points_original"]
        )
        updated = updated.drop(columns=["total_points_original"])

        updated.to_csv(output_path, index=False)
        print(f"Saved recalculated points for {season_folder} to "
              f"{output_path}")
        return updated

    def _fetch_all_gameweeks(self, season_folder: str) -> pd.DataFrame:
        """Fetch and concatenate every available gameweek file for a
        season. Returns None if no gameweek data could be found at all."""
        gw_frames = []
        for gw in range(1, MAX_GAMEWEEKS + 1):
            url = GW_DATA_URL.format(season=season_folder, gw=gw)
            try:
                gw_frames.append(pd.read_csv(url))
            except Exception:
                continue

        if not gw_frames:
            print(f"No gameweek data found for {season_folder} - cannot "
                  f"recalculate.")
            return None

        all_gws = pd.concat(gw_frames, ignore_index=True)

        if "defensive_contribution" not in all_gws.columns:
            print(
                f"{season_folder} has no defensive contribution data - "
                "DEFCON points can't be recalculated for this season "
                "(will net to 0), but all other rule changes still will "
                "be."
            )

        return all_gws

    def _recalculate_totals(self, gw_data: pd.DataFrame,
                           season_folder: str) -> pd.DataFrame:
        """Recalculate each match's points, then sum to season totals per
        player (keyed by element id, matching players_raw.csv's id)."""
        old_rules = CurrentRules.get_season_rules(season_folder)
        new_rules = CurrentRules.get_season_rules(CurrentRules.RULES_SEASON)

        gw_data = gw_data.copy()
        gw_data["recalculated_total_points"] = gw_data.apply(
            lambda row: self._recalculate_match_points(
                row, old_rules, new_rules
            ),
            axis=1
        )

        totals = (
            gw_data.groupby("element")["recalculated_total_points"]
            .sum()
            .reset_index()
            .rename(columns={
                "element": "id",
                "recalculated_total_points": "total_points"
            })
        )
        return totals

    def _recalculate_match_points(self, row, old_rules: dict,
                                 new_rules: dict) -> float:
        position = row.get("position")
        original_points = row.get("total_points", 0) or 0

        old_points = CurrentRules.calculate_stat_points(
            row, position, old_rules
        )
        new_points = CurrentRules.calculate_stat_points(
            row, position, new_rules
        )

        return original_points - old_points + new_points
