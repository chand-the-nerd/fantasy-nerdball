"""NaN handling, the schema lock, and the migration script."""
import json, os, pathlib, subprocess, sys, tempfile

tmp = tempfile.mkdtemp()
os.environ.update({
    "NERDBALL_DATA_DIR": tmp, "DATABASE_URL": f"sqlite:///{tmp}/old.db",
    "STATIC_DIR": f"{tmp}/s", "NERDBALL_ENGINE_DIR": f"{tmp}/e",
    "SECRET_KEY": "test", "HISTORY_AUTO_UPDATE": "false",
    "HEARTBEAT_MINUTES": "0",
})
BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.db import _dumps, _finite, init_db, SessionLocal
from app.models import Squad, User, utcnow

ok = True
def check(label, cond, extra=""):
    global ok
    if not cond: ok = False
    print(f"{'PASS' if cond else 'FAIL'}  {label} {extra}")

# The exact shape that broke: NaN inside a nested engine row.
payload = {
    "starting": [
        {"id": 1, "can_select": True, "chance_of_playing_next_round": float("nan")},
        {"id": 2, "form": float("inf"), "name": "Fine"},
    ],
    "value": 100.0,
}
text = _dumps(payload)
check("NaN never reaches the database", "NaN" not in text, text[:80])
check("nor Infinity", "Infinity" not in text, text[:80])
back = json.loads(text)
check("it becomes null",
      back["starting"][0]["chance_of_playing_next_round"] is None, back)
check("infinity too", back["starting"][1]["form"] is None)
check("real values are untouched", back["value"] == 100.0)
check("and so is everything else",
      back["starting"][1]["name"] == "Fine" and back["starting"][0]["can_select"] is True)
check("output is valid JSON", isinstance(back, dict))

# Numpy-ish scalars (what pandas hands back) shouldn't blow up either.
class FakeNumpyFloat:
    def item(self): return float("nan")
class FakeNumpyInt:
    def item(self): return 7
mixed = _dumps({"a": FakeNumpyFloat(), "b": FakeNumpyInt()})
check("numpy scalars are unwrapped", json.loads(mixed) == {"a": None, "b": 7}, mixed)

check("a plain float survives _finite", _finite(1.5) == 1.5)
check("nested lists are walked",
      _finite([[float("nan"), 1]]) == [[None, 1]])

# Write a real row through the ORM with a NaN in it.
init_db()
with SessionLocal() as s:
    s.add(User(email="a@b.com", name="A"))
    s.commit()
    s.add(Squad(user_id=1, season="2026-27", gameweek=4,
                payload=payload, engine_rows=[{"x": float("nan")}]))
    s.commit()
with SessionLocal() as s:
    row = s.query(Squad).first()
    check("the row stores and reads back",
          row.payload["starting"][0]["chance_of_playing_next_round"] is None,
          row.payload["starting"][0])

# Migration: dry run then real, into a second SQLite standing in for PG.
result = subprocess.run(
    [sys.executable, "-m", "app.migrate_to_postgres",
     "--sqlite", f"{tmp}/old.db", "--target", f"sqlite:///{tmp}/new.db"],
    capture_output=True, text=True,
    cwd=str(BACKEND))
check("it refuses a SQLite target",
      "leaving" in result.stdout, result.stdout[-120:])

dry = subprocess.run(
    [sys.executable, "-m", "app.migrate_to_postgres",
     "--sqlite", f"{tmp}/missing.db", "--target", "postgresql://x/y",
     "--dry-run"], capture_output=True, text=True,
    cwd=str(BACKEND))
check("and a missing source file",
      "No SQLite file" in dry.stdout, dry.stdout[-120:])

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
