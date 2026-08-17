#!/usr/bin/env python3
"""
Recalculates historic FPL seasons' points under the current season's
scoring rules (see current_rules.py) and saves them to
data/historic/{season}_updated_points.csv.

Usage:
    python utility_scripts/recalculate_historic_points.py [season ...]
    python utility_scripts/recalculate_historic_points.py --force [season ...]

If no seasons are given, recalculates every season in config.PAST_SEASONS.
Seasons without per-match defensive contribution data (anything before
2025-26) cannot be recalculated and will be skipped with a message.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from src.data.historic_points_recalculator import HistoricPointsRecalculator


def main():
    args = sys.argv[1:]
    force = "--force" in args
    seasons = [a for a in args if a != "--force"]

    if not seasons:
        seasons = Config.PAST_SEASONS

    recalculator = HistoricPointsRecalculator()

    for season in seasons:
        print(f"\nRecalculating {season}...")
        recalculator.recalculate_season(season, force=force)


if __name__ == "__main__":
    main()
