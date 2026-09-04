"""
Defines the CURRENT Fantasy Premier League season's scoring rules.

FPL scoring rules change slightly between seasons - most notably the
Defensive Contribution (DEFCON) mechanic (which has changed thresholds,
points-per-threshold and cap behaviour every season since it was
introduced) and goalkeeper goal-scoring points. Because a player's historic
total_points were calculated under that season's own rules, comparing them
directly to the current season's projections under/over-values players
affected by these changes.

This file defines the full current-season scoring rule set in one place,
plus `LEGACY_RULES_BY_SEASON` overrides describing how historic seasons'
rules differed. It's used by HistoricPointsRecalculator (see
src/data/historic_points_recalculator.py) to retrospectively recalculate
what a historic season's points would have been had this season's rules
applied throughout.

Note: retrospective DEFCON recalculation is only possible for seasons where
per-match clearances/blocks/interceptions/tackles/recoveries data is
available. That data was first recorded in the 2025-26 season, so seasons
before that (2024-25 and earlier) will simply score 0 DEFCON points either
way - all other rule differences (e.g. goalkeeper goal points) are still
recalculated correctly for those seasons.
"""


class CurrentRules:
    """Scoring rules for the current (2026-27) FPL season."""

    # Season these rules represent, used for labelling output only.
    RULES_SEASON = "2026-27"

    # === APPEARANCE ===
    MINUTES_POINTS = {"played": 1, "played_60_plus": 2}

    # === GOALS ===
    # GK goal points rose from 6 to 10 from the 2025-26 season onwards.
    GOAL_POINTS = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}

    # === ASSISTS ===
    ASSIST_POINTS = 3

    # === CLEAN SHEETS === (needs 60+ minutes played)
    CLEAN_SHEET_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}

    # === GOALS CONCEDED === (GK/DEF only)
    GOALS_CONCEDED_THRESHOLD = 2
    GOALS_CONCEDED_POINTS = -1

    # === SAVES === (GK only)
    SAVES_THRESHOLD = 3
    SAVE_POINTS = 1

    # === PENALTIES, CARDS, OWN GOALS ===
    PENALTY_SAVE_POINTS = 5
    PENALTY_MISS_POINTS = -2
    OWN_GOAL_POINTS = -2
    YELLOW_CARD_POINTS = -1
    RED_CARD_POINTS = -3

    # === DEFENSIVE CONTRIBUTION (DEFCON) ===
    # Points are awarded per multiple of `threshold` defensive actions
    # reached in a single match. Uncapped - multiple bonuses can be scored.
    DEFCON_UNCAPPED = False
    DEFCON_RULES = {
        "DEF": {
            "threshold": 10,
            "points_per_threshold": 2,
            "stats": ["clearances_blocks_interceptions", "tackles"],
        },
        "MID": {
            "threshold": 12,
            "points_per_threshold": 2,
            "stats": [
                "clearances_blocks_interceptions", "tackles", "recoveries"
            ],
        },
        "FWD": {
            "threshold": 12,
            "points_per_threshold": 2,
            "stats": [
                "clearances_blocks_interceptions", "tackles", "recoveries"
            ],
        },
        # Goalkeepers are not eligible for the DEFCON bonus.
        "GK": None,
    }

    # === LEGACY RULES BY SEASON ===
    # How a historic season's rules differed from the current season's,
    # for the fields above. Only the fields that actually differed need to
    # be listed - anything omitted is assumed unchanged from this season.
    LEGACY_RULES_BY_SEASON = {
        "2025-26": {
            # GK goal points and DEFCON both arrived this season, but the
            # DEFCON mechanic itself was capped and worth fewer points.
            "DEFCON_RULES": {
                "DEF": {
                    "threshold": 10,
                    "points_per_threshold": 2,
                    "cap_multiples": 1,
                    "stats": [
                        "clearances_blocks_interceptions", "tackles"
                    ],
                },
                "MID": {
                    "threshold": 12,
                    "points_per_threshold": 2,
                    "cap_multiples": 1,
                    "stats": [
                        "clearances_blocks_interceptions", "tackles",
                        "recoveries"
                    ],
                },
                "FWD": {
                    "threshold": 12,
                    "points_per_threshold": 2,
                    "cap_multiples": 1,
                    "stats": [
                        "clearances_blocks_interceptions", "tackles",
                        "recoveries"
                    ],
                },
                "GK": None,
            },
        },
        "2024-25": {
            # Pre-DEFCON: GK goals were worth 6, and no defensive
            # contribution data was recorded at all.
            "GOAL_POINTS": {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4},
            "DEFCON_RULES": None,
        },
        "2023-24": {
            "GOAL_POINTS": {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4},
            "DEFCON_RULES": None,
        },
    }

    @classmethod
    def get_season_rules(cls, season_folder: str) -> dict:
        """
        Resolve the full set of scoring rules that applied in a given
        historic season, using the current season's rules as a baseline
        and applying any documented differences for that season.

        Args:
            season_folder (str): Season identifier, e.g. "2024-25". Pass
                cls.RULES_SEASON (or any unlisted season) to get the
                current season's rules unmodified.

        Returns:
            dict: Resolved rules, keyed the same as this class's
                attributes (e.g. "GOAL_POINTS", "DEFCON_RULES").
        """
        base = {
            "MINUTES_POINTS": cls.MINUTES_POINTS,
            "GOAL_POINTS": cls.GOAL_POINTS,
            "ASSIST_POINTS": cls.ASSIST_POINTS,
            "CLEAN_SHEET_POINTS": cls.CLEAN_SHEET_POINTS,
            "GOALS_CONCEDED_THRESHOLD": cls.GOALS_CONCEDED_THRESHOLD,
            "GOALS_CONCEDED_POINTS": cls.GOALS_CONCEDED_POINTS,
            "SAVES_THRESHOLD": cls.SAVES_THRESHOLD,
            "SAVE_POINTS": cls.SAVE_POINTS,
            "PENALTY_SAVE_POINTS": cls.PENALTY_SAVE_POINTS,
            "PENALTY_MISS_POINTS": cls.PENALTY_MISS_POINTS,
            "OWN_GOAL_POINTS": cls.OWN_GOAL_POINTS,
            "YELLOW_CARD_POINTS": cls.YELLOW_CARD_POINTS,
            "RED_CARD_POINTS": cls.RED_CARD_POINTS,
            "DEFCON_RULES": cls.DEFCON_RULES,
        }
        base.update(cls.LEGACY_RULES_BY_SEASON.get(season_folder, {}))
        return base

    @classmethod
    def calculate_defcon_points(cls, stat_values, position: str,
                               rules_by_position: dict) -> int:
        """
        Calculate DEFCON bonus points for a single match.

        Args:
            stat_values (dict or pd.Series): Mapping of stat name (e.g.
                "clearances_blocks_interceptions") to its match value.
            position (str): One of "GK", "DEF", "MID", "FWD".
            rules_by_position (dict): Rules to apply, e.g.
                CurrentRules.DEFCON_RULES, or None/empty if no DEFCON
                mechanic existed that season.

        Returns:
            int: DEFCON points earned in the match.
        """
        if not rules_by_position:
            return 0
        rules = rules_by_position.get(position)
        if not rules:
            return 0

        defcon_value = sum(
            stat_values.get(stat, 0) or 0 for stat in rules["stats"]
        )
        multiples = defcon_value // rules["threshold"]

        cap = rules.get("cap_multiples")
        if cap is not None:
            multiples = min(multiples, cap)

        return int(multiples * rules["points_per_threshold"])

    @classmethod
    def calculate_stat_points(cls, row, position: str, rules: dict) -> float:
        """
        Calculate a single match's points from its raw stats under a given
        resolved rules dict (see get_season_rules). Excludes bonus points,
        since those are kept as originally awarded rather than
        recalculated (bonus is derived from BPS rank within the match,
        which hasn't changed between seasons).

        Args:
            row (dict or pd.Series): Raw per-match stats (minutes,
                goals_scored, assists, clean_sheets, goals_conceded,
                own_goals, penalties_saved, penalties_missed, saves,
                yellow_cards, red_cards, and DEFCON stats).
            position (str): One of "GK", "DEF", "MID", "FWD".
            rules (dict): Resolved rules from get_season_rules().

        Returns:
            float: Points earned in the match, excluding bonus.
        """
        minutes = row.get("minutes", 0) or 0
        if minutes >= 60:
            points = rules["MINUTES_POINTS"]["played_60_plus"]
        elif minutes > 0:
            points = rules["MINUTES_POINTS"]["played"]
        else:
            points = 0

        points += (row.get("goals_scored", 0) or 0) \
            * rules["GOAL_POINTS"].get(position, 0)
        points += (row.get("assists", 0) or 0) * rules["ASSIST_POINTS"]

        if minutes >= 60:
            points += (row.get("clean_sheets", 0) or 0) \
                * rules["CLEAN_SHEET_POINTS"].get(position, 0)

        if position in ("GK", "DEF"):
            goals_conceded = row.get("goals_conceded", 0) or 0
            points += (goals_conceded // rules["GOALS_CONCEDED_THRESHOLD"]) \
                * rules["GOALS_CONCEDED_POINTS"]

        if position == "GK":
            saves = row.get("saves", 0) or 0
            points += (saves // rules["SAVES_THRESHOLD"]) \
                * rules["SAVE_POINTS"]

        points += (row.get("penalties_saved", 0) or 0) \
            * rules["PENALTY_SAVE_POINTS"]
        points += (row.get("penalties_missed", 0) or 0) \
            * rules["PENALTY_MISS_POINTS"]
        points += (row.get("own_goals", 0) or 0) * rules["OWN_GOAL_POINTS"]
        points += (row.get("yellow_cards", 0) or 0) \
            * rules["YELLOW_CARD_POINTS"]
        points += (row.get("red_cards", 0) or 0) * rules["RED_CARD_POINTS"]

        points += cls.calculate_defcon_points(
            row, position, rules["DEFCON_RULES"]
        )

        return points
