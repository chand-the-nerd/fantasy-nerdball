"""Signed-in managers go ahead of guests when the worker is busy."""
import os, pathlib, sys, tempfile

tmp = tempfile.mkdtemp()
os.environ.update({
    "NERDBALL_DATA_DIR": tmp, "DATABASE_URL": f"sqlite:///{tmp}/t.db",
    "STATIC_DIR": f"{tmp}/s", "NERDBALL_ENGINE_DIR": f"{tmp}/e",
    "SECRET_KEY": "test", "HISTORY_AUTO_UPDATE": "false",
    "HEARTBEAT_MINUTES": "0", "GUEST_PAUSE_DEPTH": "3",
    "CURRENT_SEASON": "2026-27",
})
BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app import lifecycle
from app.config import settings
from app.db import init_db
from app.engine import jobs

ok = True
def check(label, cond, extra=""):
    global ok
    if not cond: ok = False
    print(f"{'PASS' if cond else 'FAIL'}  {label} {extra}")

init_db()

# The queue is what's under test, not the worker — and a live worker
# would quietly consume the items before they could be inspected.
jobs.start_worker = lambda: None

# Queue a guest, then a member, then another guest.
jobs._enqueue("run", 101, jobs.GUEST)
jobs._enqueue("run", 102, jobs.MEMBER)
jobs._enqueue("run", 103, jobs.GUEST)
jobs._enqueue("run", 104, jobs.MEMBER)

order = []
while not jobs._queue.empty():
    priority, _seq, _kind, item = jobs._queue.get()
    order.append(item)

check("members come out first", order[:2] == [102, 104], order)
check("guests follow", order[2:] == [101, 103], order)
check("and first-come holds within a priority",
      order == [102, 104, 101, 103], order)

# Guests are asked to wait once the queue is deep.
check("guests aren't held back on an empty queue",
      not jobs.guests_should_wait())
for n in range(3):
    jobs._enqueue("run", 200 + n, jobs.MEMBER)
check("guests are held back once it's busy", jobs.guests_should_wait(),
      jobs._queue.qsize())

settings.guest_pause_depth = 0
check("and the preference can be switched off",
      not jobs.guests_should_wait())
settings.guest_pause_depth = 3

while not jobs._queue.empty():
    jobs._queue.get()

# The queue still refuses work beyond its cap.
for n in range(settings.max_queued_runs):
    jobs._enqueue("run", 900 + n, jobs.MEMBER)
try:
    jobs._enqueue("run", 999, jobs.MEMBER)
    check("a full queue is refused", False, "it accepted one too many")
except RuntimeError as error:
    check("a full queue is refused", "full" in str(error).lower())

# Season staleness.
settings.current_season = "2024-25"
check("an old season is spotted", lifecycle.season_looks_stale())
settings.current_season = "2026-27"
check("the current one isn't", not lifecycle.season_looks_stale())
settings.current_season = "nonsense"
check("and nonsense doesn't crash it",
      lifecycle.season_looks_stale() is False)

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
