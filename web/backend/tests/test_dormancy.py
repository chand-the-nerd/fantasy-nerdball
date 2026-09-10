"""Dormancy, restoration, season purge and the two-year deletion."""
import datetime as dt, os, pathlib, sys, tempfile

tmp = tempfile.mkdtemp()
os.environ.update({
    "NERDBALL_DATA_DIR": tmp, "DATABASE_URL": f"sqlite:///{tmp}/t.db",
    "STATIC_DIR": f"{tmp}/s", "NERDBALL_ENGINE_DIR": f"{tmp}/e",
    "SECRET_KEY": "test", "HISTORY_AUTO_UPDATE": "false",
    "HEARTBEAT_MINUTES": "0", "DEV_MODE": "true",
    "DEV_LOGIN_EMAIL": "owner@example.com",
    "ALLOWED_EMAILS": "owner@example.com", "ADMIN_EMAILS": "owner@example.com",
    "MAX_USERS": "3", "INACTIVE_DAYS": "28", "PURGE_AFTER_MONTHS": "24",
    "MAIL_TO": "owner@example.com", "RESEND_API_KEY": "re_test",
    "CURRENT_SEASON": "2025-26",
})
BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from fastapi.testclient import TestClient
from sqlalchemy import select
from app import lifecycle, mailer
from app.auth import seat_count, upsert_user
from app.config import settings
from app.db import SessionLocal, init_db
from app.main import app
from app.models import Squad, User, utcnow

init_db()
posted = []
mailer.send_template = lambda tpl, to="", reply_to="": posted.append(
    {"subject": tpl[0], "text": tpl[2], "to": to})
lifecycle.mailer = mailer
import app.routers.admin as admin_router
admin_router.mailer = mailer

ok = True
def check(label, cond, extra=""):
    global ok
    if not cond: ok = False
    print(f"{'PASS' if cond else 'FAIL'}  {label} {extra}")

owner = TestClient(app, headers={"x-forwarded-for": "203.0.113.1"})
owner.post("/api/auth/dev-login")

# A manager with history who then goes quiet.
with SessionLocal() as s:
    lapsed = User(email="lapsed@x.com", name="Lapsed", fpl_entry_id=98765,
                  last_seen_at=utcnow() - dt.timedelta(days=40))
    s.add(lapsed)
    s.commit()
    for gw in (5, 6, 7):
        s.add(Squad(user_id=lapsed.id, season="2025-26", gameweek=gw,
                    formation="3-4-3", projected_points=50 + gw,
                    payload={"gw": gw}, engine_rows=[{"name": f"p{gw}"}]))
    s.commit()
    lapsed_id = lapsed.id
    check("they take a place while active", seat_count(s) == 2, seat_count(s))

posted.clear()
with SessionLocal() as s:
    n = lifecycle.make_dormant(s)
check("the quiet manager goes dormant", n == 1, n)

with SessionLocal() as s:
    user = s.get(User, lapsed_id)
    check("marked dormant, not deleted", user is not None and user.dormant_at)
    kept = s.scalars(select(Squad).where(Squad.user_id == lapsed_id)).all()
    check("every squad is kept", len(kept) == 3, len(kept))
    check("their place is free again", seat_count(s) == 1, seat_count(s))

check("they were told their data is safe",
      any("still here" in p["text"].lower() for p in posted),
      [p["subject"] for p in posted])
check("and you were told too",
      any(p["to"] == "" for p in posted), [p["to"] for p in posted])

# Signing back in restores everything.
with SessionLocal() as s:
    back = upsert_user(s, email="lapsed@x.com", sub="g1", name="Lapsed",
                       picture="")
    check("signing in wakes the account", back.dormant_at is None)
    check("with their squads intact",
          len(s.scalars(select(Squad).where(Squad.user_id == lapsed_id)).all()) == 3)
    check("and their FPL id", back.fpl_entry_id == 98765)
    check("taking a place again", seat_count(s) == 2, seat_count(s))

# Season rollover: dormant, and nothing left from this season.
with SessionLocal() as s:
    user = s.get(User, lapsed_id)
    user.dormant_at = utcnow()
    s.commit()
    n = lifecycle.purge_season_data(s)
check("this season's data is not purged", n == 0, n)

with SessionLocal() as s:
    for sq in s.scalars(select(Squad).where(Squad.user_id == lapsed_id)):
        sq.season = "2024-25"
    s.commit()
    n = lifecycle.purge_season_data(s)
check("last season's data is purged", n == 1, n)
with SessionLocal() as s:
    user = s.get(User, lapsed_id)
    left = s.scalars(select(Squad).where(Squad.user_id == lapsed_id)).all()
    check("the squads are gone", len(left) == 0, len(left))
    check("the account survives", user is not None)
    check("and the FPL id is what persists", user.fpl_entry_id == 98765)
    check("marked as purged", user.data_purged_at is not None)

# Two years is the only thing that deletes an account.
with SessionLocal() as s:
    old = User(email="ancient@x.com",
               last_seen_at=utcnow() - dt.timedelta(days=800))
    s.add(old)
    s.commit()
    n = lifecycle.delete_long_gone(s)
check("a two-year-old account is deleted", n == 1, n)
with SessionLocal() as s:
    check("but the recently dormant one is not",
          s.get(User, lapsed_id) is not None)
    check("and the owner never is",
          s.scalar(select(User).where(User.email == "owner@example.com")))

# Admin can see and restore.
listing = owner.get("/api/admin/users").json()
statuses = {u["email"]: u["status"] for u in listing["users"]}
print(f"      statuses: {statuses}")
check("dormant managers are listed", "lapsed@x.com" in statuses)
check("with their status", statuses["lapsed@x.com"] in ("dormant", "purged"))
check("seats only count active managers", listing["seats_used"] == 1,
      listing["seats_used"])

# Restore a squad from a given gameweek.
with SessionLocal() as s:
    u = s.get(User, lapsed_id)
    u.dormant_at = None
    s.commit()
    for gw in (5, 6):
        s.add(Squad(user_id=lapsed_id, season="2025-26", gameweek=gw,
                    formation="3-4-3", projected_points=40 + gw,
                    payload={"gw": gw}, engine_rows=[{"n": gw}]))
    s.commit()

snaps = owner.get(f"/api/admin/users/{lapsed_id}/squads").json()
check("their gameweeks are listed", len(snaps["squads"]) == 2, snaps["squads"])

restored = owner.post(f"/api/admin/users/{lapsed_id}/restore-squad",
                      json={"season": "2025-26", "gameweek": 5,
                            "into_gameweek": 6})
check("restore works", restored.status_code == 200, restored.text[:100])
with SessionLocal() as s:
    gw6 = s.scalar(select(Squad).where(Squad.user_id == lapsed_id,
                                       Squad.gameweek == 6))
    gw5 = s.scalar(select(Squad).where(Squad.user_id == lapsed_id,
                                       Squad.gameweek == 5))
    check("gameweek 6 now holds gameweek 5's squad",
          gw6.payload == {"gw": 5}, gw6.payload)
    check("and gameweek 5 is untouched", gw5.payload == {"gw": 5})

missing = owner.post(f"/api/admin/users/{lapsed_id}/restore-squad",
                     json={"season": "2025-26", "gameweek": 33})
check("restoring a gameweek they never had is a 404",
      missing.status_code == 404, missing.status_code)

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
