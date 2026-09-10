"""Exercise the guest flow end to end against a throwaway SQLite file."""

import os
import pathlib
import sys
import tempfile

tmp = tempfile.mkdtemp()
os.environ["NERDBALL_DATA_DIR"] = tmp
os.environ["DATABASE_URL"] = f"sqlite:///{tmp}/test.db"
os.environ["STATIC_DIR"] = f"{tmp}/nostatic"
os.environ["NERDBALL_ENGINE_DIR"] = f"{tmp}/noengine"
os.environ["HISTORY_AUTO_UPDATE"] = "false"
os.environ["SECRET_KEY"] = "test-only"
os.environ["DEV_MODE"] = "true"
os.environ["ALLOWED_EMAILS"] = "owner@example.com"

BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402

from app import guest  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models import User, UserSettings  # noqa: E402

init_db()
client = TestClient(app)

ok = True


def check(label, condition, extra=""):
    global ok
    if not condition:
        ok = False
    print(f"{'PASS' if condition else 'FAIL'}  {label} {extra}")


config = client.get("/api/auth/config").json()
check("auth config advertises guest", config.get("guest") is True, config)

check("no session means 401", client.get("/api/me").status_code == 401)

started = client.post("/api/auth/guest")
check("guest session starts", started.status_code == 200)

me = client.get("/api/me").json()
check("me says guest", me.get("is_guest") is True, me)

settings = client.get("/api/me/settings").json()
check("5 gameweeks", settings["first_n_gameweeks"] == 5)
check("strategy 1.2", abs(settings["min_transfer_value"] - 1.2) < 1e-6)
check("bench 0.2", abs(settings["bench_weight"] - 0.2) < 1e-6)
check("hits on", settings["accept_transfer_penalty"] is True)
check("skip unavailable on", settings["exclude_unavailable"] is True)
check("ml weights off", settings["use_ml_weights"] is False)
check("no clubs adjusted", settings["team_modifiers"] == {})
weights = settings["overrides"]["POSITION_SCORING_WEIGHTS"]
check(
    "40/30/30 on every position",
    all(
        w == {"form": 0.4, "historic": 0.3, "difficulty": 0.3}
        for w in weights.values()
    )
    and set(weights) == {"GK", "DEF", "MID", "FWD"},
    weights,
)

# A guest saving something it IS allowed to change must still succeed,
# and the locked fields must come back untouched.
saved = client.put(
    "/api/me/settings",
    json={
        "budget": 101.5,
        "first_n_gameweeks": 10,
        "min_transfer_value": 0.0,
        "bench_weight": 1.0,
        "accept_transfer_penalty": False,
        "exclude_unavailable": False,
        "team_modifiers": {"Arsenal": 1.5},
        "overrides": {
            "POSITION_SCORING_WEIGHTS": {
                "MID": {"form": 1.0, "historic": 0.0, "difficulty": 0.0}
            }
        },
    },
)
check("save accepted", saved.status_code == 200, saved.text[:120])
body = saved.json()
check("budget kept", abs(body["budget"] - 101.5) < 1e-6)
check("horizon put back", body["first_n_gameweeks"] == 5)
check("strategy put back", abs(body["min_transfer_value"] - 1.2) < 1e-6)
check("bench put back", abs(body["bench_weight"] - 0.2) < 1e-6)
check("hits put back on", body["accept_transfer_penalty"] is True)
check("clubs put back neutral", body["team_modifiers"] == {})
check(
    "weights put back",
    body["overrides"]["POSITION_SCORING_WEIGHTS"]["MID"]
    == {"form": 0.4, "historic": 0.3, "difficulty": 0.3},
)

over = client.put(
    "/api/me/settings",
    json={"forced_selections": {"MID": ["A", "B"], "FWD": ["C"]}},
)
check("three forced picks refused", over.status_code == 403, over.text[:90])

two = client.put(
    "/api/me/settings",
    json={"forced_selections": {"MID": ["A", "B"]}},
)
check("two forced picks allowed", two.status_code == 200, two.text[:90])

over_bl = client.put(
    "/api/me/settings", json={"blacklist_players": ["A", "B", "C"]}
)
check("three avoided refused", over_bl.status_code == 403)

link = client.post("/api/me/fpl-entry", json={"fpl_entry_id": 1234567})
check("FPL link refused", link.status_code == 403, link.text[:90])

imported = client.post(
    "/api/me/import-squad",
    json={"apply_budget": True, "apply_free_transfers": True},
)
check("FPL import refused", imported.status_code == 403)

plan = client.post("/api/plans", json={"weeks": 5, "chips": {}})
check("planner refused", plan.status_code == 403, plan.text[:90])

# Guests must not consume one of the league's seats.
with SessionLocal() as session:
    from app.auth import seat_count

    check("seat count ignores guests", seat_count(session) == 0)
    guest_id = session.query(User).filter(User.is_guest.is_(True)).one().id

client.post("/api/auth/logout")
with SessionLocal() as session:
    gone = session.get(User, guest_id)
    check("guest row deleted on sign-out", gone is None)
    orphan = (
        session.query(UserSettings)
        .filter(UserSettings.user_id == guest_id)
        .first()
    )
    check("settings row went with it", orphan is None)

check("session really ended", client.get("/api/me").status_code == 401)

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
