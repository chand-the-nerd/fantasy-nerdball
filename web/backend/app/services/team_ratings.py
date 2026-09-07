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
                            "venue": "Home" if at_home else "Away",
                            "difficulty": fixture.get(key),
                        }
                    )

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
                "attack_overperformance": attack_ratio,
                "defence_overperformance": defence_ratio,
                "fpl_strength": team.get("strength"),
                "attack_strength_home": team.get("strength_attack_home"),
                "attack_strength_away": team.get("strength_attack_away"),
                "defence_strength_home": team.get("strength_defence_home"),
                "defence_strength_away": team.get("strength_defence_away"),
                "next_fixtures": fixture_list,
                "fixture_difficulty": fixture_score,
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
