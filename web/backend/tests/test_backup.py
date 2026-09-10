"""Back up, destroy everything, restore, and check it all came back."""
import datetime as dt, os, pathlib, sys, tempfile

tmp = tempfile.mkdtemp()
PG = os.getenv(
    "NERDBALL_TEST_PG",
    "postgresql+psycopg://postgres@localhost:5433/backuptest",
)
os.environ.update({
    "NERDBALL_DATA_DIR": tmp, "DATABASE_URL": PG,
    "STATIC_DIR": f"{tmp}/s", "NERDBALL_ENGINE_DIR": f"{tmp}/e",
    "SECRET_KEY": "test", "HISTORY_AUTO_UPDATE": "false",
    "HEARTBEAT_MINUTES": "0", "BACKUP_KEEP": "3",
})
BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

# A backup/restore test is only meaningful from a known state, and
# dropping the tables works on whatever database it's pointed at.
def _wipe() -> None:
    from app.db import engine
    from app.models import Base

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

from sqlalchemy import func, select
from app import backup
from app.db import SessionLocal, init_db
from app.models import (GameweekResult, InboxItem, Invite, Run, Squad,
                        User, UserSettings, utcnow)

ok = True
def check(label, cond, extra=""):
    global ok
    if not cond: ok = False
    print(f"{'PASS' if cond else 'FAIL'}  {label} {extra}")

init_db()
_wipe()
with SessionLocal() as s:
    owner = User(email="owner@x.com", name="Owner", is_admin=True,
                 fpl_entry_id=12345)
    owner.settings = UserSettings(budget=101.2, first_n_gameweeks=5)
    mate = User(email="mate@x.com", name="Mate", dormant_at=utcnow())
    mate.settings = UserSettings()
    s.add_all([owner, mate]); s.commit()
    s.add(Squad(user_id=owner.id, season="2026-27", gameweek=4,
                formation="3-5-2", projected_points=56.95,
                payload={"starting": [{"id": 1,
                          "chance": float("nan")}]},
                engine_rows=[{"a": 1}], option_rows={"o1": [{"b": 2}]}))
    s.add(Run(user_id=owner.id, season="2026-27", gameweek=4,
              status="complete"))
    s.add(GameweekResult(user_id=owner.id, season="2026-27", gameweek=1,
                         actual_points=43.0, source="fpl"))
    s.add(Invite(email="waiting@x.com", invited_by="owner@x.com",
                 expires_at=utcnow() + dt.timedelta(hours=72)))
    s.add(InboxItem(kind="access_request", email="waiting@x.com",
                    body="please"))
    s.commit()
    before = {m.__name__: s.scalar(select(func.count()).select_from(m))
              for m in (User, Squad, Run, GameweekResult, Invite,
                        InboxItem, UserSettings)}
print(f"      before: {before}")

path = backup.create()
check("a backup file is written", path.exists(), path.name)
check("it's compressed", path.name.endswith(".json.gz"))
check("and not empty", path.stat().st_size > 200, path.stat().st_size)
check("it shows up in the listing",
      any(i["name"] == path.name for i in backup.listing()))

# Now destroy the lot, exactly as a bad sweep would.
with SessionLocal() as s:
    from sqlalchemy import delete as sqldelete
    from app.models import Base
    for table in reversed(Base.metadata.sorted_tables):
        s.execute(sqldelete(table))
    s.commit()
    check("everything is gone",
          s.scalar(select(func.count()).select_from(User)) == 0)

counts = backup.restore(path, force=True)
with SessionLocal() as s:
    after = {m.__name__: s.scalar(select(func.count()).select_from(m))
             for m in (User, Squad, Run, GameweekResult, Invite,
                       InboxItem, UserSettings)}
check("every row came back", after == before, f"{before} -> {after}")

with SessionLocal() as s:
    owner = s.scalar(select(User).where(User.email == "owner@x.com"))
    check("the admin flag survived", owner.is_admin is True)
    check("the FPL id survived", owner.fpl_entry_id == 12345)
    check("settings came back", abs(owner.settings.budget - 101.2) < 0.01)
    mate = s.scalar(select(User).where(User.email == "mate@x.com"))
    check("dormancy survived", mate.dormant_at is not None)
    check("and is still a datetime, not a string",
          isinstance(mate.dormant_at, dt.datetime), type(mate.dormant_at))
    squad = s.scalar(select(Squad))
    check("squad JSON came back",
          squad.payload["starting"][0]["chance"] is None, squad.payload)
    check("option rows too", squad.option_rows["o1"][0]["b"] == 2)
    check("floats intact", abs(squad.projected_points - 56.95) < 0.01)
    invite = s.scalar(select(Invite))
    check("invite expiry is a datetime",
          isinstance(invite.expires_at, dt.datetime))

    # New inserts must not collide with restored ids.
    fresh = User(email="new@x.com", name="New")
    s.add(fresh); s.commit()
    check("sequences were reset", fresh.id > 2, fresh.id)

# Restoring over live data without force must refuse.
try:
    backup.restore(path)
    check("restoring over live data is refused", False, "it went ahead")
except RuntimeError as error:
    check("restoring over live data is refused", "force" in str(error))

# Retention.
for _ in range(5):
    import time as _t; _t.sleep(1.05); backup.create()
removed = backup.prune(3)
check("old backups are pruned", len(backup.listing()) == 3,
      len(backup.listing()))
check("and it says how many went", removed >= 1, removed)

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
