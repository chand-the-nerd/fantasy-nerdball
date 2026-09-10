"""Drive the app, then check the admin dashboard describes what happened."""

import os
import pathlib
import sys
import tempfile
import time

tmp = tempfile.mkdtemp()
os.environ["NERDBALL_DATA_DIR"] = tmp
os.environ["DATABASE_URL"] = f"sqlite:///{tmp}/test.db"
os.environ["STATIC_DIR"] = f"{tmp}/nostatic"
os.environ["NERDBALL_ENGINE_DIR"] = f"{tmp}/noengine"
os.environ["HISTORY_AUTO_UPDATE"] = "false"
os.environ["HEARTBEAT_MINUTES"] = "0"
os.environ["SECRET_KEY"] = "test-only"
os.environ["DEV_MODE"] = "true"
os.environ["DEV_LOGIN_EMAIL"] = "owner@example.com"
os.environ["ALLOWED_EMAILS"] = "owner@example.com"
os.environ["ADMIN_EMAILS"] = "owner@example.com"

BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402

from app import metrics  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models import MetricEvent  # noqa: E402

init_db()

ok = True


def check(label, condition, extra=""):
    global ok
    if not condition:
        ok = False
    print(f"{'PASS' if condition else 'FAIL'}  {label} {extra}")


def settle():
    """Wait for the background writer to drain."""
    for _ in range(50):
        if metrics._queue.unfinished_tasks == 0:
            time.sleep(0.05)
            return
        time.sleep(0.05)


# Two guests from different addresses, plus one repeat visit from the
# first, so unique counting has something to get wrong.
alice = TestClient(app, headers={"x-forwarded-for": "203.0.113.9"})
bob = TestClient(app, headers={"x-forwarded-for": "198.51.100.4"})

alice.post("/api/auth/guest")
alice.get("/api/me")
alice.put("/api/me/settings", json={"budget": 100.2})
alice.put("/api/me/settings", json={"blacklist_players": ["A", "B", "C"]})

bob.post("/api/auth/guest")
bob.get("/api/me")

owner = TestClient(app, headers={"x-forwarded-for": "203.0.113.200"})
owner.post("/api/auth/dev-login")
owner.get("/api/me")

settle()

with SessionLocal() as session:
    rows = session.query(MetricEvent).all()
    check("rows were written", len(rows) > 4, len(rows))
    kinds = sorted({r.kind for r in rows})
    print(f"      kinds stored: {kinds}")
    check(
        "http_request is not stored",
        "http_request" not in kinds,
        "it belongs in the log, not the table",
    )
    guests = [r for r in rows if r.is_guest]
    check("guests recorded", bool(guests))
    check(
        "guest visitor is a salted hash, not an address",
        all(
            r.visitor.startswith("ip:") and "203.0.113.9" not in r.visitor
            for r in guests
        ),
        guests[0].visitor if guests else "",
    )
    check(
        "guest label is truncated to the network",
        any(r.label == "203.0.113.0/24" for r in guests),
        sorted({r.label for r in guests}),
    )
    member = [r for r in rows if not r.is_guest and r.user_id]
    check(
        "a signed-in manager is keyed by account, not address",
        bool(member) and all(r.visitor.startswith("user:") for r in member),
    )
    check(
        "no full address anywhere in the table",
        not any("203.0.113.9" == r.label for r in rows),
    )

denied = alice.get("/api/admin/metrics")
check("guests can't read the dashboard", denied.status_code == 403, denied.text[:80])

response = owner.get("/api/admin/metrics?window=24h")
check("admin can read it", response.status_code == 200, response.text[:120])
data = response.json()

totals = data["totals"]
print(f"      totals: {totals}")
check("three unique visitors", totals["visitors"] == 3, totals["visitors"])
check("two of them guests", totals["guests"] == 2, totals["guests"])
check("one signed in", totals["members"] == 1, totals["members"])
check("visits counted", totals["sessions"] >= 3, totals["sessions"])

check("series covers the window", len(data["series"]) > 20, len(data["series"]))
check(
    "series buckets carry both measures",
    all({"visitors", "runs", "label"} <= set(p) for p in data["series"]),
)
check(
    "somebody appears in the series",
    any(p["visitors"] > 0 for p in data["series"]),
)

people = data["people"]
check("people listed", len(people) == 3, len(people))
check(
    "the manager is named by account",
    any(p["detail"] == "owner@example.com" for p in people),
    [p["who"] for p in people],
)
check(
    "guests are shown as their network",
    any(p["guest"] and "/24" in p["detail"] for p in people),
    [p["detail"] for p in people if p["guest"]],
)

check(
    "guest_blocked is surfaced",
    any(b["feature"] for b in data["blocked"]),
    data["blocked"],
)

cap = data["capacity"]
print(f"      capacity: {cap}")
check("capacity block present", {"rss_mb", "workers"} <= set(cap), cap)
check("this process reports its memory", cap["rss_mb"] and cap["rss_mb"] > 0)
check("worker count reported", cap["workers"] >= 1)
check(
    "parallel estimate is honest with no runs measured",
    cap["measured_runs"] == 0 and cap["parallel_runs_by_memory"] is None,
    cap,
)

from app.capacity import MemorySampler, parallel_capacity, rss_mb  # noqa: E402

with MemorySampler(interval=0.05) as sampler:
    ballast = [bytearray(4 * 1024 * 1024) for _ in range(8)]
    time.sleep(0.3)
check("sampler saw the allocation", (sampler.cost_mb or 0) > 8, sampler.cost_mb)
del ballast
check("baseline recorded", (sampler.baseline or 0) > 0)
check(
    "no limit reported means no estimate, not infinity",
    parallel_capacity(None) is None,
)
check("a huge run cost leaves no room", parallel_capacity(1e9) in (0, None))

from app.engine.jobs import queue_depth  # noqa: E402
from app.models import Run  # noqa: E402

with SessionLocal() as session:
    session.add(Run(user_id=1, season="2025-26", gameweek=5, status="queued"))
    session.add(Run(user_id=2, season="2025-26", gameweek=5, status="queued"))
    session.add(Run(user_id=3, season="2025-26", gameweek=5, status="running"))
    session.commit()
    check(
        "queue depth counts queued work from the database",
        queue_depth(session) == 2,
        queue_depth(session),
    )
check(
    "queue depth works without a session too",
    queue_depth() == 2,
    queue_depth(),
)

bad = owner.get("/api/admin/metrics?window=decade")
check("an unknown window is refused", bad.status_code == 400)

for window in ("1h", "6h", "7d"):
    got = owner.get(f"/api/admin/metrics?window={window}")
    check(f"window {window} works", got.status_code == 200)

# Deleting a guest must not delete the history of their visit.
before = len(data["people"])
alice.post("/api/auth/logout")
settle()
after = owner.get("/api/admin/metrics?window=24h").json()
check(
    "a departed guest is still in the numbers",
    len(after["people"]) >= before,
    f"{before} then {len(after['people'])}",
)

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
