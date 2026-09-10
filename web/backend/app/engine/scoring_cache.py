"""Reuse the scored player pool between runs that would produce it twice.

Scoring every player — merging past seasons, fetching each one's fixture
difficulty over the horizon, then building the scores — is the expensive
half of a run and depends on none of the things that make one manager's
run different from another's. Two people running the same gameweek with
the same model settings get byte-identical frames out of it.

That matters most for guests, whose settings are locked, so every guest
run for a gameweek recomputes exactly the same numbers. It helps
signed-in managers on the defaults too, and it does nothing at all for
someone who has tuned their weights — which is correct, because their
scores really are different.

The risk in a cache like this is serving one manager numbers computed
for another's settings. Guarded two ways: the key is built by including
everything on the config and excluding a named list of fields known not
to affect scoring, so a setting added later is included automatically and
the failure mode is a missed cache rather than a wrong answer. And the
frames are copied on the way out, because the optimiser mutates them.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import threading
from typing import Any

from .. import events
from ..config import settings

# Fields that change what the optimiser does with the scores, not the
# scores themselves. Everything else on the config goes into the key.
#
# Get this wrong in one direction and the cache misses more often than it
# needs to. Get it wrong in the other and somebody sees the wrong
# numbers, so the list stays short and each entry has to be obviously
# about squad selection rather than player scoring.
NOT_SCORING = frozenset(
    {
        "BUDGET",
        "FREE_TRANSFERS",
        "ACCEPT_TRANSFER_PENALTY",
        "MIN_TRANSFER_VALUE",
        "BENCH_WEIGHT",
        "FORCED_SELECTIONS",
        "BLACKLIST_PLAYERS",
        "WILDCARD",
        "FREE_HIT",
        "FREE_HIT_PREV_GW",
        "BENCH_BOOST",
        "TRIPLE_CAPTAIN",
        "GRANULAR_OUTPUT",
        "DETAILED_CALCULATION",
    }
)

_entries: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()

_hits = 0
_misses = 0


def _encode(value: Any) -> Any:
    """Reduce a config value to something stable to hash."""
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def fingerprint(config: Any) -> str:
    """A key covering everything that could change the scores."""
    material: dict[str, Any] = {}
    for name in dir(config):
        if not name.isupper() or name in NOT_SCORING:
            continue
        try:
            value = getattr(config, name)
        except Exception:
            continue
        if callable(value):
            continue
        material[name] = _encode(value)

    blob = json.dumps(material, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _expired(entry: dict[str, Any]) -> bool:
    age = dt.datetime.now(dt.timezone.utc) - entry["at"]
    return age > dt.timedelta(minutes=settings.scoring_cache_minutes)


def get(config: Any) -> tuple[Any, Any] | None:
    """The cached (players, scored) for this config, or None.

    Returns copies. The optimiser writes into these frames, and handing
    out the cached object itself would let one run corrupt the next.
    """
    global _hits, _misses

    if settings.scoring_cache_minutes <= 0:
        return None

    key = fingerprint(config)
    with _lock:
        entry = _entries.get(key)
        if entry is None or _expired(entry):
            if entry is not None:
                _entries.pop(key, None)
            _misses += 1
            return None
        _hits += 1
        players = entry["players"]
        scored = entry["scored"]

    return players.copy(), scored.copy()


def put(config: Any, players: Any, scored: Any) -> None:
    """Keep this pool for the next run that wants the same one."""
    if settings.scoring_cache_minutes <= 0:
        return

    key = fingerprint(config)
    with _lock:
        _entries[key] = {
            "players": players.copy(),
            "scored": scored.copy(),
            "at": dt.datetime.now(dt.timezone.utc),
            "gameweek": getattr(config, "GAMEWEEK", None),
        }

        # A pool is a few megabytes, and this shares a container with the
        # optimiser itself. Keep the newest handful and drop the rest.
        if len(_entries) > settings.scoring_cache_entries:
            oldest = sorted(_entries.items(), key=lambda kv: kv[1]["at"])
            for stale_key, _ in oldest[: -settings.scoring_cache_entries]:
                _entries.pop(stale_key, None)


def state() -> dict[str, Any]:
    """What's cached, for the admin page."""
    now = dt.datetime.now(dt.timezone.utc)
    with _lock:
        entries = [
            {
                "gameweek": entry["gameweek"],
                "age_seconds": round((now - entry["at"]).total_seconds()),
            }
            for entry in _entries.values()
        ]
    total = _hits + _misses
    return {
        "entries": entries,
        "hits": _hits,
        "misses": _misses,
        "hit_rate": round(_hits / total, 2) if total else None,
    }


def clear() -> None:
    with _lock:
        _entries.clear()


def load(config: Any, compute) -> tuple[Any, Any, float]:
    """The scored pool, from cache if it's there and fresh.

    `compute` is the engine call that produces it. The budget comes back
    from the config rather than the cache: it is the one part of that
    call's result that belongs to the manager rather than the gameweek.
    """
    cached = get(config)
    if cached is not None:
        players, scored = cached
        events.emit(
            "scoring_cache_hit", gameweek=getattr(config, "GAMEWEEK", None)
        )
        return players, scored, float(config.BUDGET)

    players, scored, available_budget = compute()
    try:
        put(config, players, scored)
    except Exception:
        # A cache that can't store is a slow run, not a failed one.
        pass
    return players, scored, available_budget
