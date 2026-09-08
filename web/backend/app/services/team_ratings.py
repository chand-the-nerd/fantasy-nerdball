"""Team strength, from results and the underlying numbers behind them.

The local `calculate_team_ratings.py` compares actual goals against what the
fixture difficulty implied. This does the same job from live data, but anchors
it on expected goals rather than on difficulty alone: a side winning games on
0.6 xG a match is flattered by results, and a side losing them on 2.1 xG is
better than its table position.
"""

from __future__ import annotations

import requests

from . import fpl

# A goalkeeper plays very nearly every minute, so their expected goals conceded
# is a good stand-in for the team's. Summing outfielders would count the same
# shots many times over.
GK_TYPE = 1

# How hard an opponent's own numbers pull their FPL difficulty around. The CLI
# optimiser uses the same 1.5 for the position that cares, and the same 0.3
# home discount, so the two stay comparable.
SWING = 1.5
HOME_DISCOUNT = 0.3
# Ratings are ratios against the league average, clamped so that one freak
# scoreline in August can't send a fixture to 1.0 or 5.0 on its own.
RATING_FLOOR = 0.5
RATING_CEILING = 2.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _adjusted(base, swing_rating: float, at_home: bool) -> float | None:
    """FPL's difficulty, pulled towards what the opponent actually does.

    A rating of 1.00 is league average and leaves the fixture where FPL put
    it. Above average makes it harder, below makes it easier.
    """
    if base in (None, ""):
        return None
    adjustment = (swing_rating - 1.0) * SWING
    if at_home:
        adjustment -= HOME_DISCOUNT
    return round(_clamp(float(base) + adjustment, 1.0, 5.0), 1)


def _number(raw, default: float = 0.0) -> float:
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _fixtures() -> list[dict]:
    response = requests.get(f"{fpl.BASE_URL}/fixtures/", timeout=fpl.TIMEOUT)
    response.raise_for_status()
    return response.json()


def build_ratings(look_ahead: int = 5) -> dict:
    data = fpl.bootstrap()
    fixtures = _fixtures()

    teams = {int(t["id"]): t for t in data.get("teams", [])}
    elements = data.get("elements", [])

    # Attacking expected goals: sum of the squad's, which counts each shot once.
    xg_for: dict[int, float] = {tid: 0.0 for tid in teams}
    # Defensive: take the busiest keeper rather than summing.
    xgc_by_gk: dict[int, float] = {tid: 0.0 for tid in teams}
    gk_minutes: dict[int, int] = {tid: 0 for tid in teams}

    for element in elements:
        team_id = int(element.get("team", 0))
        if team_id not in teams:
            continue
        xg_for[team_id] += _number(element.get("expected_goals"))
        if element.get("element_type") == GK_TYPE:
            minutes = int(element.get("minutes") or 0)
            if minutes > gk_minutes[team_id]:
                gk_minutes[team_id] = minutes
                xgc_by_gk[team_id] = _number(element.get("expected_goals_conceded"))

    played = {tid: 0 for tid in teams}
    goals_for = {tid: 0 for tid in teams}
    goals_against = {tid: 0 for tid in teams}
    upcoming: dict[int, list[dict]] = {tid: [] for tid in teams}

    for fixture in fixtures:
        home, away = fixture.get("team_h"), fixture.get("team_a")
        if home not in teams or away not in teams:
            continue

        if fixture.get("finished") and fixture.get("team_h_score") is not None:
            hs = int(fixture["team_h_score"])
            as_ = int(fixture["team_a_score"])
            played[home] += 1
            played[away] += 1
            goals_for[home] += hs
            goals_against[home] += as_
            goals_for[away] += as_
            goals_against[away] += hs
        elif fixture.get("event") is not None:
            for team_id, opponent_id, at_home, key in (
                (home, away, True, "team_h_difficulty"),
                (away, home, False, "team_a_difficulty"),
            ):
                if len(upcoming[team_id]) < look_ahead:
                    upcoming[team_id].append(
                        {
                            "gameweek": fixture.get("event"),
                            "opponent": teams[opponent_id].get("short_name", ""),
                            "opponent_id": opponent_id,
                            "venue": "Home" if at_home else "Away",
                            "difficulty": fixture.get(key),
                        }
                    )

    # League averages, over the sides that have actually played. Everything
    # below is measured against these rather than against an absolute.
    per_game_xg = {
        tid: xg_for[tid] / played[tid] for tid in teams if played[tid]
    }
    per_game_xgc = {
        tid: xgc_by_gk[tid] / played[tid] for tid in teams if played[tid]
    }
    mean_xg = (
        sum(per_game_xg.values()) / len(per_game_xg) if per_game_xg else 0.0
    )
    mean_xgc = (
        sum(per_game_xgc.values()) / len(per_game_xgc) if per_game_xgc else 0.0
    )

    # Above 1.00 attacking means creating more than the league average does.
    # Above 1.00 defensively means allowing less — so a high defence rating is
    # a good defence, matching the CLI ratings file.
    attack_rating: dict[int, float] = {}
    defence_rating: dict[int, float] = {}
    for team_id in teams:
        own_xg = per_game_xg.get(team_id)
        own_xgc = per_game_xgc.get(team_id)
        attack_rating[team_id] = (
            round(_clamp(own_xg / mean_xg, RATING_FLOOR, RATING_CEILING), 2)
            if own_xg is not None and mean_xg > 0
            else 1.0
        )
        if own_xgc is None or mean_xgc <= 0:
            defence_rating[team_id] = 1.0
        elif own_xgc <= 0:
            defence_rating[team_id] = RATING_CEILING
        else:
            defence_rating[team_id] = round(
                _clamp(mean_xgc / own_xgc, RATING_FLOOR, RATING_CEILING), 2
            )

    # One fixture, two difficulties. Your attackers care about the opponent's
    # defence; your defenders and keeper care about their attack. Chelsea can
    # be a hard fixture on FPL's rating and still a soft one for a striker.
    for team_id, fixture_list in upcoming.items():
        for fixture in fixture_list:
            opponent_id = fixture.pop("opponent_id")
            at_home = fixture["venue"] == "Home"
            fixture["attack_difficulty"] = _adjusted(
                fixture["difficulty"], defence_rating.get(opponent_id, 1.0), at_home
            )
            fixture["defence_difficulty"] = _adjusted(
                fixture["difficulty"], attack_rating.get(opponent_id, 1.0), at_home
            )
            fixture["opponent_attack_rating"] = attack_rating.get(opponent_id, 1.0)
            fixture["opponent_defence_rating"] = defence_rating.get(opponent_id, 1.0)

    def _mean(values: list) -> float | None:
        real = [v for v in values if v is not None]
        return round(sum(real) / len(real), 2) if real else None

    rows = []
    for team_id, team in teams.items():
        games = played[team_id]
        xg = round(xg_for[team_id], 2)
        xgc = round(xgc_by_gk[team_id], 2)

        # Ratios of actual against expected. Above 1.00 attacking means
        # scoring more than the chances deserve; above 1.00 defensively means
        # conceding more than the chances allowed, which is bad.
        attack_ratio = round(goals_for[team_id] / xg, 2) if xg > 0 else None
        defence_ratio = (
            round(goals_against[team_id] / xgc, 2) if xgc > 0 else None
        )

        fixture_list = upcoming[team_id]
        difficulties = [f["difficulty"] for f in fixture_list if f["difficulty"]]
        fixture_score = (
            round(sum(difficulties) / len(difficulties), 2) if difficulties else None
        )

        rows.append(
            {
                "id": team_id,
                "name": team.get("name", ""),
                "short_name": team.get("short_name", ""),
                "played": games,
                "goals_for": goals_for[team_id],
                "goals_against": goals_against[team_id],
                "xg": xg,
                "xgc": xgc,
                "xg_per_game": round(xg / games, 2) if games else None,
                "xgc_per_game": round(xgc / games, 2) if games else None,
                "goal_difference": goals_for[team_id] - goals_against[team_id],
                "xg_difference": round(xg - xgc, 2),
                "xg_difference_per_game": (
                    round((xg - xgc) / games, 2) if games else None
                ),
                "attack_rating": attack_rating[team_id],
                "defence_rating": defence_rating[team_id],
                "attack_overperformance": attack_ratio,
                "defence_overperformance": defence_ratio,
                "fpl_strength": team.get("strength"),
                "attack_strength_home": team.get("strength_attack_home"),
                "attack_strength_away": team.get("strength_attack_away"),
                "defence_strength_home": team.get("strength_defence_home"),
                "defence_strength_away": team.get("strength_defence_away"),
                "next_fixtures": fixture_list,
                "fixture_difficulty": fixture_score,
                "attack_fdr": _mean([f["attack_difficulty"] for f in fixture_list]),
                "defence_fdr": _mean([f["defence_difficulty"] for f in fixture_list]),
            }
        )

    rows.sort(key=lambda r: -(r["xg_difference"]))
    for rank, row in enumerate(rows, start=1):
        row["xg_rank"] = rank

    return {
        "season": fpl.settings.current_season,
        "look_ahead": look_ahead,
        "teams": rows,
    }
