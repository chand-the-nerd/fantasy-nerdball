"""Drive the app and check the JSON events that come out of it."""

import io
import json
import logging
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
os.environ["HEARTBEAT_MINUTES"] = "0"
os.environ["SECRET_KEY"] = "test-only"
# The session cookie is https-only outside dev mode, which a test client
# on http:// would silently drop.
os.environ["DEV_MODE"] = "true"
os.environ["LOG_JSON"] = "true"

BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402

from app import events  # noqa: E402
from app.db import init_db  # noqa: E402
from app.main import app  # noqa: E402

init_db()

# Capture stdout logging into a buffer we can parse.
buffer = io.StringIO()
handler = logging.StreamHandler(buffer)
handler.setFormatter(events.JsonFormatter())
root = logging.getLogger()
for existing in list(root.handlers):
    root.removeHandler(existing)
root.addHandler(handler)
root.setLevel(logging.INFO)

logging.getLogger("httpx").setLevel(logging.WARNING)

client = TestClient(app)

client.get("/api/auth/config")
client.post("/api/auth/guest")
client.get("/api/me")
client.put("/api/me/settings", json={"budget": 100.5})
client.put("/api/me/settings", json={"blacklist_players": ["A", "B", "C"]})
client.post("/api/plans", json={"weeks": 5, "chips": {}})
client.post("/api/auth/logout")

lines = [ln for ln in buffer.getvalue().splitlines() if ln.strip()]
records = []
bad = []
for line in lines:
    try:
        records.append(json.loads(line))
    except json.JSONDecodeError:
        bad.append(line)

ok = True


def check(label, condition, extra=""):
    global ok
    if not condition:
        ok = False
    print(f"{'PASS' if condition else 'FAIL'}  {label} {extra}")


def first(name):
    return next((r for r in records if r.get("event") == name), None)


check("every line is valid JSON", not bad, bad[:2])
check("something was logged", len(records) > 5, len(records))

seen = sorted({r.get("event") for r in records if r.get("event")})
print(f"      events seen: {seen}")

guest_start = first("guest_start")
check("guest_start emitted", guest_start is not None)

session = first("session_start")
check("session_start emitted", session is not None)
check(
    "session_start marks the guest",
    session and session.get("guest") is True,
    session,
)

saved = first("settings_saved")
check("settings_saved emitted", saved is not None)
check(
    "settings_saved carries the shape",
    saved and "horizon" in saved and "forced" in saved,
    saved,
)

blocked = first("guest_blocked")
check("guest_blocked emitted on a refused limit", blocked is not None, blocked)

http = first("http_request")
check("http_request emitted", http is not None)
check(
    "http_request has route, status and timing",
    http
    and {"route", "status", "ms"} <= set(http)
    and isinstance(http["ms"], int),
    http,
)
check(
    "http_request uses the route template, not the raw path",
    all(
        "{" in r["route"] or r["route"].count("/") <= 4
        for r in records
        if r.get("event") == "http_request"
    ),
)

check(
    "quiet routes stay out of the log",
    not any(
        r.get("route") == "/api/health"
        for r in records
        if r.get("event") == "http_request"
    ),
)

out = first("sign_out")
check("sign_out emitted", out is not None, out)

check(
    "no email or name in any line",
    not any(
        "@" in json.dumps(r) and r.get("event") != "sign_in_denied"
        for r in records
    ),
)

# A second visit inside the dedupe window must not count as a new session.
before = len([r for r in records if r.get("event") == "session_start"])
client.post("/api/auth/guest")
client.get("/api/me")
client.get("/api/me")
client.get("/api/me")
again = [
    json.loads(ln)
    for ln in buffer.getvalue().splitlines()
    if ln.strip() and json.loads(ln).get("event") == "session_start"
]
check(
    "repeat requests don't inflate the session count",
    len(again) == before + 1,
    f"{before} then {len(again)}",
)

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
