"""The queryable half of the telemetry.

events.py writes JSON to stdout, which Railway keeps for thirty days and
charts on its own dashboard. This module keeps the same facts in the
database, because an in-app dashboard needs to ask questions of history
rather than tail a stream, and because a log line about a guest outlives
the guest account while a foreign key would not.

Three things kept it from being a burden on the request path:

Writes happen on a background thread behind a bounded queue. A metric is
never worth making somebody wait for, and if the queue backs up the
events are dropped rather than allowed to slow the app down.

Only meaningful events are stored. `http_request` fires on every call and
belongs in the log, not in a table that has to be aggregated later.

Rows are pruned on a timer. Ninety days is long enough to see a season
take shape and short enough that nothing accumulates quietly.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import ipaddress
import queue
import threading
from contextvars import ContextVar
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .config import settings
from .db import session_scope
from .models import MetricEvent, User, utcnow

# Events worth a row. Everything else still reaches the log.
STORED = {
    "session_start",
    "sign_in",
    "guest_start",
    "sign_out",
    "run_queued",
    "run_started",
    "run_finished",
    "run_failed",
    "run_rejected",
    "plan_queued",
    "plan_finished",
    "plan_failed",
    "settings_saved",
    "squad_entered",
    "option_activated",
    "player_lookup",
    "guest_blocked",
    "fpl_linked",
}

# Who and where the current request is from, put here by the middleware
# and the current-user dependency so an event emitted deep in a handler
# doesn't have to be handed the request to know either.
_client_ip: ContextVar[str] = ContextVar("client_ip", default="")
_actor: ContextVar[tuple[int | None, bool]] = ContextVar(
    "actor", default=(None, False)
)

_queue: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=2000)
_writer: threading.Thread | None = None
_writer_lock = threading.Lock()
_dropped = 0


def set_client_ip(value: str) -> None:
    _client_ip.set(value or "")
    # A fresh request starts with nobody attached; the dependency sets it
    # once the session cookie has been resolved.
    _actor.set((None, False))


def set_actor(user_id: int | None, is_guest: bool) -> None:
    """Remember who this request belongs to.

    Some events are raised from places that never see the user — a guest
    refusal comes out of an exception helper, for instance. Without this
    they would be filed against the bare address and show up as a second
    visitor standing next to the person who caused them.
    """
    _actor.set((user_id, bool(is_guest)))


def client_ip_from(request: Any) -> str:
    """The caller's address, as seen from behind Railway's proxy.

    The first entry in X-Forwarded-For is the client; everything after it
    is the chain of proxies that carried the request.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = getattr(request, "client", None)
    return getattr(client, "host", "") or ""


def _truncate(raw: str) -> str:
    """An address cut down to the network it came from.

    A /24 (or /48 for IPv6) still separates one visitor from another well
    enough to count them, without the table holding a list of addresses.
    """
    try:
        address = ipaddress.ip_address(raw)
    except ValueError:
        return ""
    if address.version == 4:
        network = ipaddress.ip_network(f"{raw}/24", strict=False)
    else:
        network = ipaddress.ip_network(f"{raw}/48", strict=False)
    return str(network)


def _fingerprint(raw: str) -> str:
    """A stable, one-way label for an address.

    Salted with the app's secret so the same visitor is recognisable
    across a week of rows without the value being reversible into an
    address by anybody who reads the table.
    """
    digest = hashlib.sha256(f"{settings.secret_key}:{raw}".encode())
    return digest.hexdigest()[:10]


def identify(user_id: int | None, is_guest: bool) -> tuple[str, str]:
    """The (visitor, label) pair for whoever is making this request."""
    if user_id is None:
        user_id, is_guest = _actor.get()

    if user_id and not is_guest:
        # A signed-in manager is identified by account, not address:
        # their name is resolvable at query time and doesn't go stale.
        return f"user:{user_id}", ""

    raw = _client_ip.get()
    if not raw:
        return "anon", ""

    fingerprint = f"ip:{_fingerprint(raw)}"
    mode = settings.visitor_ip_mode
    if mode == "full":
        return fingerprint, raw
    if mode == "truncated":
        return fingerprint, _truncate(raw)
    return fingerprint, ""


def record(kind: str, fields: dict[str, Any]) -> None:
    """Queue one event for storage. Never blocks, never raises."""
    global _dropped
    if not settings.metrics_enabled or kind not in STORED:
        return

    try:
        user_id = fields.get("user")
        is_guest = bool(fields.get("guest"))
        if user_id is None:
            user_id, is_guest = _actor.get()
        visitor, label = identify(user_id, is_guest)

        meta = {
            k: v
            for k, v in fields.items()
            if k not in {"event", "user", "guest", "seconds"}
        }
        row = {
            "at": utcnow(),
            "kind": kind,
            "visitor": visitor,
            "label": label,
            "user_id": user_id if isinstance(user_id, int) else None,
            "is_guest": is_guest,
            "value": fields.get("seconds"),
            "meta": meta or None,
        }
        _queue.put_nowait(row)
        _start_writer()
    except queue.Full:
        # Metrics are the first thing that should suffer under load.
        _dropped += 1
    except Exception:
        pass


def _start_writer() -> None:
    global _writer
    with _writer_lock:
        if _writer is not None and _writer.is_alive():
            return
        _writer = threading.Thread(
            target=_loop, name="metrics-writer", daemon=True
        )
        _writer.start()


def _loop() -> None:
    """Drain the queue in batches, so a burst is one transaction."""
    while True:
        first = _queue.get()
        batch = [first]
        try:
            while len(batch) < 100:
                batch.append(_queue.get_nowait())
        except queue.Empty:
            pass

        try:
            with session_scope() as session:
                session.add_all(MetricEvent(**row) for row in batch)
        except Exception:
            # A dashboard that misses a few rows is worth more than a
            # thread that dies and misses all of them.
            pass
        finally:
            for _ in batch:
                _queue.task_done()


def prune(session: Session) -> int:
    """Drop rows past the retention window."""
    if settings.metrics_retention_days <= 0:
        return 0
    cutoff = utcnow() - dt.timedelta(days=settings.metrics_retention_days)
    result = session.execute(
        delete(MetricEvent).where(MetricEvent.at < cutoff)
    )
    return int(result.rowcount or 0)


# ── Reading it back ──────────────────────────────────────────────────────

WINDOWS: dict[str, tuple[int, int, str]] = {
    # name: (minutes covered, bucket minutes, label format)
    "1h": (60, 5, "%H:%M"),
    "6h": (360, 30, "%H:%M"),
    "24h": (1440, 60, "%H:%M"),
    "7d": (10080, 360, "%a %H:%M"),
}


def _bucket_start(moment: dt.datetime, minutes: int) -> dt.datetime:
    """Round down to the start of the bucket a moment belongs in."""
    stamp = int(moment.timestamp())
    size = minutes * 60
    return dt.datetime.fromtimestamp(
        stamp - (stamp % size), tz=dt.timezone.utc
    )


def summary(session: Session, window: str = "24h") -> dict[str, Any]:
    """Everything the admin dashboard draws, for one time window.

    Aggregated in Python rather than in SQL. The row counts involved are
    small, and it keeps the same code working on SQLite and Postgres
    without date-truncation dialects diverging.
    """
    span, bucket, fmt = WINDOWS.get(window, WINDOWS["24h"])
    since = utcnow() - dt.timedelta(minutes=span)

    rows = session.scalars(
        select(MetricEvent)
        .where(MetricEvent.at >= since)
        .order_by(MetricEvent.at)
        # A hard ceiling so an unexpectedly busy week can't turn the
        # dashboard into the slowest page in the app.
        .limit(50000)
    ).all()

    visitors: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    run_seconds: list[float] = []
    waits: list[float] = []
    blocked: dict[str, int] = {}

    series_visitors: dict[dt.datetime, set[str]] = {}
    series_runs: dict[dt.datetime, int] = {}

    for row in rows:
        counts[row.kind] = counts.get(row.kind, 0) + 1

        at = row.at
        if at.tzinfo is None:
            at = at.replace(tzinfo=dt.timezone.utc)
        slot = _bucket_start(at, bucket)

        if row.visitor:
            series_visitors.setdefault(slot, set()).add(row.visitor)
            seen = visitors.setdefault(
                row.visitor,
                {
                    "visitor": row.visitor,
                    "user_id": row.user_id,
                    "is_guest": row.is_guest,
                    "label": row.label,
                    "sessions": 0,
                    "runs": 0,
                    "events": 0,
                    "first_seen": at,
                    "last_seen": at,
                },
            )
            seen["events"] += 1
            seen["last_seen"] = max(seen["last_seen"], at)
            seen["first_seen"] = min(seen["first_seen"], at)
            if row.label and not seen["label"]:
                seen["label"] = row.label
            if row.kind == "session_start":
                seen["sessions"] += 1
            if row.kind == "run_queued":
                seen["runs"] += 1

        if row.kind == "run_queued":
            series_runs[slot] = series_runs.get(slot, 0) + 1
        if row.kind == "run_finished" and row.value is not None:
            run_seconds.append(float(row.value))
        if row.kind == "run_started" and (row.meta or {}).get(
            "waited_seconds"
        ) is not None:
            waits.append(float(row.meta["waited_seconds"]))
        if row.kind == "guest_blocked":
            feature = (row.meta or {}).get("feature") or "unknown"
            blocked[feature] = blocked.get(feature, 0) + 1

    # Names for the signed-in visitors. One query, not one per row.
    ids = [
        v["user_id"]
        for v in visitors.values()
        if v["user_id"] and not v["is_guest"]
    ]
    named: dict[int, User] = {}
    if ids:
        for user in session.scalars(select(User).where(User.id.in_(ids))):
            named[user.id] = user

    people = []
    for entry in visitors.values():
        user = named.get(entry["user_id"] or -1)
        if entry["is_guest"]:
            who = "Guest"
            detail = entry["label"] or "address not recorded"
        elif entry["user_id"] is None:
            # Reached the site but never got as far as an account.
            who = "Visitor"
            detail = entry["label"] or "address not recorded"
        elif user is not None:
            who = user.name or user.email
            detail = user.email
        else:
            # Their account has since been removed; the activity stands.
            who = "Former manager"
            detail = ""
        people.append(
            {
                "visitor": entry["visitor"],
                "who": who,
                "detail": detail,
                "guest": entry["is_guest"],
                "sessions": entry["sessions"],
                "runs": entry["runs"],
                "events": entry["events"],
                "first_seen": entry["first_seen"].isoformat(),
                "last_seen": entry["last_seen"].isoformat(),
            }
        )
    people.sort(key=lambda p: p["last_seen"], reverse=True)

    # Every bucket in the window, including the empty ones, so the chart
    # shows a quiet Tuesday rather than skipping it.
    now = utcnow()
    slots: list[dt.datetime] = []
    cursor = _bucket_start(since, bucket)
    end = _bucket_start(now, bucket)
    while cursor <= end:
        slots.append(cursor)
        cursor += dt.timedelta(minutes=bucket)

    series = [
        {
            "at": slot.isoformat(),
            "label": slot.strftime(fmt),
            "visitors": len(series_visitors.get(slot, ())),
            "runs": series_runs.get(slot, 0),
        }
        for slot in slots
    ]

    ordered = sorted(run_seconds)

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        index = min(len(values) - 1, int(len(values) * fraction))
        return round(values[index], 1)

    return {
        "window": window,
        "since": since.isoformat(),
        "bucket_minutes": bucket,
        "totals": {
            "visitors": len(visitors),
            "members": len(
                [
                    v
                    for v in visitors.values()
                    if v["user_id"] and not v["is_guest"]
                ]
            ),
            "guests": len([v for v in visitors.values() if v["is_guest"]]),
            "sessions": counts.get("session_start", 0),
            "sign_ins": counts.get("sign_in", 0),
            "runs": counts.get("run_queued", 0),
            "runs_finished": counts.get("run_finished", 0),
            "runs_failed": counts.get("run_failed", 0),
            "runs_rejected": counts.get("run_rejected", 0),
            "plans": counts.get("plan_queued", 0),
            "run_seconds_median": percentile(ordered, 0.5),
            "run_seconds_p95": percentile(ordered, 0.95),
            "wait_seconds_p95": percentile(sorted(waits), 0.95),
        },
        "series": series,
        "people": people[:100],
        "blocked": sorted(
            ({"feature": k, "count": v} for k, v in blocked.items()),
            key=lambda item: item["count"],
            reverse=True,
        ),
        "activity": sorted(
            ({"kind": k, "count": v} for k, v in counts.items()),
            key=lambda item: item["count"],
            reverse=True,
        ),
        "ip_mode": settings.visitor_ip_mode,
        "dropped": _dropped,
    }
