"""Insert a demo squad so the interface can be worked on without a full run.

An optimisation takes minutes and needs live FPL data. When the thing you're
changing is the pitch, that loop is far too slow, so this writes a realistic
squad straight into the database.

    python web/backend/seed_demo.py

Re-running overwrites the demo squad. It does not touch real ones.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("SECRET_KEY", "dev-local-not-a-secret")
os.environ.setdefault("DEV_MODE", "true")

from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.models import GameweekResult, GameweekStat, Squad, User, UserSettings  # noqa: E402

DEMO_GAMEWEEK = 4

# Deliberately awkward: the longest names in the league, a pale kit that needs
# dark lettering, an accented name, a doubtful player and a double gameweek.
# If the card layout survives this lot it survives anything.
STARTING = [
    ("Raya", "GK", "Arsenal", 5.6, 4.4, "SUN", "Home", "a", "", False),
    ("Alexander-Arnold", "DEF", "Liverpool", 7.1, 5.8, "FUL", "Away", "a", "", True),
    ("Van de Ven", "DEF", "Spurs", 4.9, 4.6, "BUR", "Home", "a", "", False),
    ("Wan-Bissaka", "DEF", "West Ham", 4.6, 4.2, "EVE", "Away", "a", "", False),
    ("Palmer", "MID", "Chelsea", 10.6, 6.9, "BHA", "Home", "a", "", False),
    ("B.Fernandes", "MID", "Man Utd", 9.0, 5.9, "LEE", "Away", "a", "", False),
    ("Saka", "MID", "Arsenal", 10.1, 5.7, "SUN", "Home", "a", "", False),
    ("Kluivert", "MID", "Bournemouth", 6.2, 5.1, "NFO", "Home", "a", "", False),
    ("Haaland", "FWD", "Man City", 14.3, 8.4, "WOL", "Away", "a", "", False),
    ("João Pedro", "FWD", "Chelsea", 7.7, 5.4, "BHA", "Home", "a", "", False),
    ("Calvert-Lewin", "FWD", "Leeds", 5.5, 4.1, "NEW", "Away", "d",
     "Knock — 75% chance of playing", False),
]

BENCH = [
    ("Steele", "GK", "Brighton", 4.0, 2.9, "CHE", "Away", "a", "", False),
    ("Mitchell", "DEF", "Crystal Palace", 4.8, 3.8, "IPS", "Home", "a", "", False),
    ("Stach", "MID", "Leeds", 5.0, 3.6, "NEW", "Away", "a", "", False),
    ("Ajayi", "DEF", "Hull City", 4.2, 3.1, "CHE", "Away", "u",
     "Suspended until 15 Sep", False),
]


def player(index, row, on_bench, bench_order, captain=False, vice=False):
    name, position, team, price, points, opponent, venue, status, news, dgw = row
    return {
        "id": 1000 + index,
        "code": 90000 + index,
        "name": name,
        "position": position,
        "team": team,
        "team_short": "",
        "price": price,
        "projected_points": points,
        "form": round(points * 0.8, 1),
        "historic_ppg": round(points * 0.9, 1),
        "fixture_difficulty": 3,
        "start_rate": 88,
        "minutes_per_game": 79,
        "xg_modifier": 1.04,
        "next_opponent": opponent,
        "venue": venue,
        "status": status,
        "news": news,
        "is_captain": captain,
        "is_vice": vice,
        "is_double_gameweek": dgw,
        "on_bench": on_bench,
        "bench_order": bench_order,
    }


def build_payload() -> dict:
    starting = [
        player(i, row, False, None, captain=(row[0] == "Haaland"),
               vice=(row[0] == "Palmer"))
        for i, row in enumerate(STARTING)
    ]
    bench = [player(100 + i, row, True, i + 1) for i, row in enumerate(BENCH)]

    counts = {"DEF": 0, "MID": 0, "FWD": 0}
    for p in starting:
        if p["position"] in counts:
            counts[p["position"]] += 1

    total = sum(p["projected_points"] for p in starting)
    total += next(p["projected_points"] for p in starting if p["is_captain"])
    value = sum(p["price"] for p in starting + bench)

    return {
        "starting": starting,
        "bench": bench,
        "formation": f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}",
        "gameweek": DEMO_GAMEWEEK,
        "season": settings.current_season,
        "projected_points": round(total, 1),
        "squad_value": round(value, 1),
        "bank": round(100.0 - value, 1),
        "chip": "",
        "transfers_made": 1,
        "penalty_points": 0,
        "made_transfers": True,
        "transfer_reason": "",
        "points_gain_per_gw": 1.8,
        "transfers": {"out": ["Isak (Newcastle)"], "in": ["Haaland (Man City)"]},
        "model_xi": {
            "projected_points": round(total + 4.2, 1),
            "cost": round(value, 1),
            "starting": starting,
        },
    }


def main() -> None:
    init_db()
    session = SessionLocal()
    email = os.getenv("DEV_LOGIN_EMAIL", "you@example.com").lower()

    user = session.scalar(select(User).where(User.email == email))
    if user is None:
        user = User(email=email, name="Local developer", is_admin=True)
        user.settings = UserSettings()
        session.add(user)
        session.commit()
        session.refresh(user)
        print(f"Created demo user {email}")

    payload = build_payload()
    squad = session.scalar(
        select(Squad).where(
            Squad.user_id == user.id,
            Squad.season == settings.current_season,
            Squad.gameweek == DEMO_GAMEWEEK,
        )
    )
    if squad is None:
        squad = Squad(user_id=user.id, season=settings.current_season,
                      gameweek=DEMO_GAMEWEEK)
        session.add(squad)

    squad.formation = payload["formation"]
    squad.projected_points = payload["projected_points"]
    squad.squad_value = payload["squad_value"]
    squad.bank = payload["bank"]
    squad.transfers_made = 1
    squad.penalty_points = 0
    squad.chip = ""
    squad.payload = payload
    squad.engine_rows = []

    # A few finished gameweeks so the Form page has a chart to draw.
    for gw, (avg, high, mine) in enumerate(
        [(57, 128, 64), (49, 115, 44), (61, 131, 72), (54, 122, 58)], start=1
    ):
        stat = session.scalar(
            select(GameweekStat).where(
                GameweekStat.season == settings.current_season,
                GameweekStat.gameweek == gw,
            )
        )
        if stat is None:
            stat = GameweekStat(season=settings.current_season, gameweek=gw)
            session.add(stat)
        stat.average_score, stat.highest_score, stat.finished = avg, high, True

        result = session.scalar(
            select(GameweekResult).where(
                GameweekResult.user_id == user.id,
                GameweekResult.season == settings.current_season,
                GameweekResult.gameweek == gw,
            )
        )
        if result is None:
            result = GameweekResult(user_id=user.id, season=settings.current_season,
                                    gameweek=gw)
            session.add(result)
        result.actual_points = mine
        result.overall_rank = 1_200_000 - gw * 90_000
        result.source = "manual"

    session.commit()
    session.close()

    print(f"Seeded GW{DEMO_GAMEWEEK}: {payload['formation']}, "
          f"{payload['projected_points']} projected, 4 scored gameweeks")
    print("Open http://localhost:5173 and sign in as the local developer.")


if __name__ == "__main__":
    main()
