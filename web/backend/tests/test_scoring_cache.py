"""The scored pool is reused only when it's genuinely the same pool."""
import os, pathlib, sys, tempfile

tmp = tempfile.mkdtemp()
os.environ.update({
    "NERDBALL_DATA_DIR": tmp, "DATABASE_URL": f"sqlite:///{tmp}/t.db",
    "STATIC_DIR": f"{tmp}/s", "NERDBALL_ENGINE_DIR": f"{tmp}/e",
    "SECRET_KEY": "test", "HISTORY_AUTO_UPDATE": "false",
    "HEARTBEAT_MINUTES": "0", "SCORING_CACHE_MINUTES": "20",
    "SCORING_CACHE_ENTRIES": "2",
})
BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import pandas as pd
from app.config import settings
from app.engine import scoring_cache

ok = True
def check(label, cond, extra=""):
    global ok
    if not cond: ok = False
    print(f"{'PASS' if cond else 'FAIL'}  {label} {extra}")


class Config:
    """Stands in for the engine config, with the fields that matter."""
    def __init__(self, **kw):
        self.GAMEWEEK = 5
        self.CURRENT_SEASON = "2026-27"
        self.FIRST_N_GAMEWEEKS = 5
        self.USE_ML_WEIGHTS = False
        self.EXCLUDE_UNAVAILABLE = True
        self.TEAM_MODIFIERS = {}
        self.POSITION_SCORING_WEIGHTS = {
            "MID": {"form": 0.4, "historic": 0.3, "difficulty": 0.3}}
        # Squad-selection settings, which must NOT affect the key.
        self.BUDGET = 100.0
        self.FREE_TRANSFERS = 1
        self.MIN_TRANSFER_VALUE = 1.2
        self.BENCH_WEIGHT = 0.2
        self.FORCED_SELECTIONS = {"MID": ["salah"]}
        self.BLACKLIST_PLAYERS = []
        self.WILDCARD = False
        self.TRIPLE_CAPTAIN = False
        for k, v in kw.items():
            setattr(self, k, v)


def pool(marker):
    players = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
    scored = pd.DataFrame({"id": [1, 2], "fpl_score": [marker, marker + 1]})
    return players, scored


base = Config()
scoring_cache.clear()

# Two guests: identical model, different budget, free transfers, forced
# picks and chips. Same pool.
guest_a = Config(BUDGET=100.0, FREE_TRANSFERS=1)
guest_b = Config(BUDGET=103.5, FREE_TRANSFERS=2,
                 FORCED_SELECTIONS={"FWD": ["haaland"]},
                 TRIPLE_CAPTAIN=True)
check("squad settings don't change the key",
      scoring_cache.fingerprint(guest_a) == scoring_cache.fingerprint(guest_b),
      "budget and forced picks must not split the cache")

# Anything that feeds the scores must split it.
for field, value in [
    ("GAMEWEEK", 6),
    ("CURRENT_SEASON", "2025-26"),
    ("FIRST_N_GAMEWEEKS", 8),
    ("USE_ML_WEIGHTS", True),
    ("EXCLUDE_UNAVAILABLE", False),
    ("TEAM_MODIFIERS", {"Arsenal": 1.4}),
    ("POSITION_SCORING_WEIGHTS",
     {"MID": {"form": 0.9, "historic": 0.05, "difficulty": 0.05}}),
]:
    other = Config(**{field: value})
    check(f"{field} changes the key",
          scoring_cache.fingerprint(base) != scoring_cache.fingerprint(other))

# A setting nobody has thought of yet must split it too, so the failure
# mode of forgetting to update NOT_SCORING is a missed cache, not a
# wrong answer.
future = Config()
future.SOME_NEW_SCORING_KNOB = 0.7
check("an unknown setting changes the key",
      scoring_cache.fingerprint(base) != scoring_cache.fingerprint(future))

# Round trip.
scoring_cache.clear()
check("nothing cached to begin with", scoring_cache.get(base) is None)

players, scored = pool(10.0)
scoring_cache.put(base, players, scored)
got = scoring_cache.get(guest_b)
check("a matching config gets the pool back", got is not None)
check("and it's the right one", got[1]["fpl_score"].iloc[0] == 10.0)

# Copies, not the cached objects.
got[1].loc[0, "fpl_score"] = 999.0
again = scoring_cache.get(base)
check("callers get copies, not the cached frame",
      again[1]["fpl_score"].iloc[0] == 10.0, again[1]["fpl_score"].iloc[0])

check("a different gameweek misses",
      scoring_cache.get(Config(GAMEWEEK=9)) is None)

# load() computes on a miss and reuses on a hit.
scoring_cache.clear()
calls = []
def compute():
    calls.append(1)
    p, s = pool(20.0)
    return p, s, 101.5

cfg = Config(BUDGET=101.5)
p1, s1, budget1 = scoring_cache.load(cfg, compute)
check("first load computes", len(calls) == 1)
check("and returns the budget", budget1 == 101.5, budget1)

other_manager = Config(BUDGET=97.0, FREE_TRANSFERS=3)
p2, s2, budget2 = scoring_cache.load(other_manager, compute)
check("second load reuses the pool", len(calls) == 1, len(calls))
check("but takes its own budget", budget2 == 97.0, budget2)
check("with the same scores", s2["fpl_score"].iloc[0] == 20.0)

tuned = Config(POSITION_SCORING_WEIGHTS={"MID": {"form": 1.0}})
scoring_cache.load(tuned, compute)
check("a tuned manager recomputes", len(calls) == 2, len(calls))

# Eviction keeps memory bounded.
scoring_cache.clear()
for gw in range(1, 6):
    scoring_cache.put(Config(GAMEWEEK=gw), *pool(float(gw)))
check("only the newest entries are kept",
      len(scoring_cache.state()["entries"]) == 2,
      scoring_cache.state()["entries"])

# Expiry.
settings.scoring_cache_minutes = 0
check("a zero TTL disables it", scoring_cache.get(base) is None)
settings.scoring_cache_minutes = 20

report = scoring_cache.state()
check("hits and misses are counted",
      report["hits"] > 0 and report["misses"] > 0, report)

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
