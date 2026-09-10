"""Migrate a realistic SQLite database into a real Postgres."""
import os, pathlib, subprocess, sys, tempfile

tmp = tempfile.mkdtemp()
SQLITE = f"{tmp}/old.db"
PG = os.getenv(
    "NERDBALL_TEST_PG",
    "postgresql+psycopg://postgres@localhost:5433/nerdball",
)

# A migration test is only meaningful against an empty target, so start
# from one every time.
BIN = "/usr/lib/postgresql/16/bin"
subprocess.run(
    ["su", "postgres", "-c",
     f"{BIN}/dropdb -h localhost -p 5433 -U postgres nerdball --if-exists;"
     f" {BIN}/createdb -h localhost -p 5433 -U postgres nerdball"],
    capture_output=True)

os.environ.update({
    "NERDBALL_DATA_DIR": tmp, "DATABASE_URL": f"sqlite:///{SQLITE}",
    "STATIC_DIR": f"{tmp}/s", "NERDBALL_ENGINE_DIR": f"{tmp}/e",
    "SECRET_KEY": "test", "HISTORY_AUTO_UPDATE": "false",
})
BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from sqlalchemy import create_engine, select, func, text
from sqlalchemy.orm import Session
from app.db import init_db, SessionLocal, _dumps
from app.models import (Base, GameweekResult, InboxItem, Invite,
                        MetricEvent, Run, Squad, User, UserSettings,
                        utcnow)

ok = True
def check(label, cond, extra=""):
    global ok
    if not cond: ok = False
    print(f"{'PASS' if cond else 'FAIL'}  {label} {extra}")

# Build a SQLite database that looks like a live deployment, NaN included.
init_db()
with SessionLocal() as s:
    owner = User(email="owner@x.com", name="Owner", is_admin=True,
                 fpl_entry_id=12345)
    owner.settings = UserSettings(budget=101.2, first_n_gameweeks=5)
    mate = User(email="mate@x.com", name="Mate", dormant_at=utcnow())
    mate.settings = UserSettings()
    guest = User(email="guest-abc@guests.invalid", name="Guest",
                 is_guest=True)
    guest.settings = UserSettings()
    s.add_all([owner, mate, guest])
    s.commit()
    s.add(Squad(user_id=owner.id, season="2026-27", gameweek=4,
                formation="3-5-2", projected_points=56.95,
                payload={"starting": [{"id": 1,
                                       "chance_of_playing_next_round": float("nan")}]},
                engine_rows=[{"can_select": True, "form": float("nan")}],
                option_rows={"option-1": [{"x": float("inf")}]}))
    s.add(Run(user_id=owner.id, season="2026-27", gameweek=4,
              status="complete"))
    s.add(Invite(email="waiting@x.com", invited_by="owner@x.com"))
    s.add(InboxItem(kind="access_request", email="waiting@x.com",
                    body="please"))
    s.add(MetricEvent(kind="session_start", visitor="user:1", user_id=1))
    s.commit()

    # SQLite doesn't enforce foreign keys, so a deployment that has
    # deleted a manager carries orphaned children forever. Made the way
    # it really happens: create them properly, then delete the parent.
    from app.models import GameweekResult
    ghost = User(email="ghost@x.com", name="Ghost")
    ghost.settings = UserSettings()
    s.add(ghost)
    s.commit()
    ghost_id = ghost.id
    s.add(GameweekResult(user_id=ghost_id, season="2026-27", gameweek=1,
                         actual_points=43.0, source="fpl"))
    s.add(GameweekResult(user_id=ghost_id, season="2026-27", gameweek=2,
                         actual_points=118.0, source="fpl"))
    s.add(Run(user_id=ghost_id, season="2026-27", gameweek=4,
              status="complete"))
    s.commit()
    # The manager goes; SQLite leaves the children behind.
    s.execute(text(f"DELETE FROM user_settings WHERE user_id = {ghost_id}"))
    s.execute(text(f"DELETE FROM users WHERE id = {ghost_id}"))
    s.commit()
    before = {
        "users": s.scalar(select(func.count()).select_from(User)),
        "squads": s.scalar(select(func.count()).select_from(Squad)),
        "runs": s.scalar(select(func.count()).select_from(Run)),
        "settings": s.scalar(select(func.count()).select_from(UserSettings)),
    }
print(f"      source: {before}")

# Dry run must change nothing.
dry = subprocess.run([sys.executable, "-m", "app.migrate_to_postgres",
                      "--sqlite", SQLITE, "--target", PG, "--dry-run"],
                     capture_output=True, text=True, cwd=str(BACKEND))
check("dry run succeeds", dry.returncode == 0, dry.stdout[-200:] + dry.stderr[-300:])
check("and says nothing was written", "Nothing was written" in dry.stdout)
target = create_engine(PG, future=True, json_serializer=_dumps)
with Session(target) as t:
    check("target still empty after dry run",
          t.scalar(select(func.count()).select_from(User)) == 0)

# The real thing.
run = subprocess.run([sys.executable, "-m", "app.migrate_to_postgres",
                      "--sqlite", SQLITE, "--target", PG],
                     capture_output=True, text=True, cwd=str(BACKEND))
check("migration succeeds", run.returncode == 0,
      run.stdout[-300:] + run.stderr[-400:])
print("      " + run.stdout.strip().replace("\n", "\n      "))

with Session(target) as t:
    after = {
        "users": t.scalar(select(func.count()).select_from(User)),
        "squads": t.scalar(select(func.count()).select_from(Squad)),
        "runs": t.scalar(select(func.count()).select_from(Run)),
        "settings": t.scalar(select(func.count()).select_from(UserSettings)),
    }
    # One run and two results belonged to the deleted manager, so the
    # target should hold everything except those three.
    expected = dict(before, runs=before["runs"] - 1)
    check("every valid row came across", after == expected,
          f"{before} -> {after}")
    check("orphans were reported",
          "had no matching parent" in run.stdout, run.stdout[-200:])
    check("and left behind",
          t.scalar(select(func.count()).select_from(GameweekResult)) == 0,
          "results referencing a deleted manager should not be copied")

    owner_row = t.scalar(select(User).where(User.email == "owner@x.com"))
    check("the admin flag survived", owner_row.is_admin is True)
    check("the FPL id survived", owner_row.fpl_entry_id == 12345)
    check("dormancy survived",
          t.scalar(select(User).where(User.email == "mate@x.com")).dormant_at
          is not None)
    check("settings came with them",
          abs(owner_row.settings.budget - 101.2) < 0.001,
          owner_row.settings.budget)

    squad = t.scalar(select(Squad))
    check("the squad's NaN became null",
          squad.payload["starting"][0]["chance_of_playing_next_round"] is None,
          squad.payload)
    check("engine rows too", squad.engine_rows[0]["form"] is None)
    check("and the option rows", squad.option_rows["option-1"][0]["x"] is None)
    check("real values intact", abs(squad.projected_points - 56.95) < 0.01)

    # Sequences: the next insert must not collide with a copied id.
    fresh = User(email="new@x.com", name="New")
    t.add(fresh)
    t.commit()
    check("new rows insert cleanly after migration", fresh.id > 3, fresh.id)

    newsquad = Squad(user_id=fresh.id, season="2026-27", gameweek=5,
                     payload={}, engine_rows=[])
    t.add(newsquad)
    t.commit()
    check("and so do child rows", newsquad.id > 1, newsquad.id)

# Running it again must refuse.
again = subprocess.run([sys.executable, "-m", "app.migrate_to_postgres",
                        "--sqlite", SQLITE, "--target", PG],
                       capture_output=True, text=True, cwd=str(BACKEND))
check("a second run is refused", again.returncode == 1, again.stdout[-150:])
check("with an explanation", "Refusing" in again.stdout, again.stdout[-150:])

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
