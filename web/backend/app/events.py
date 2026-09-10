"""One JSON line per thing worth counting.

Railway parses JSON on stdout into filterable attributes, so an event
written here becomes something the Observability dashboard can chart
without any of it leaving the platform. Everything is a single line with
an `event` key, which is what the widget filters on: `@event:run_started`,
`@event:session_start`, and so on. web/OBSERVABILITY.md lists them all.

Two rules hold throughout:

Identities are ids, never names or addresses. A log line is readable by
anyone with dashboard access and is retained for weeks, so `user: 4` is
enough to count people and follow one person's session without the log
becoming a copy of the users table. The one deliberate exception is a
refused sign-in, where the address is the whole point of the record.

Nothing here can break a request. `emit` swallows its own failures: a
logging bug should never be the reason an optimisation doesn't start.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

log = logging.getLogger("nerdball.events")

# How long a user can go quiet before their next request counts as a new
# visit. Thirty minutes is the usual analytics convention and it makes
# `session_start` countable as "sessions today" without the app having to
# track sessions itself.
SESSION_GAP_SECONDS = 30 * 60

_seen: dict[int, float] = {}
_seen_lock = threading.Lock()


class JsonFormatter(logging.Formatter):
    """Renders every log record as one JSON object.

    Applied to the root logger, so warnings and tracebacks from anywhere
    in the app land in the same shape as the events below and can be
    filtered the same way.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }

        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)

        if record.exc_info:
            kind = record.exc_info[0]
            payload["error_type"] = kind.__name__ if kind else "Exception"
            payload["traceback"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure(as_json: bool = True, level: int = logging.INFO) -> None:
    """Point the root logger at stdout in the chosen shape.

    JSON in production so the dashboard can filter on attributes, plain
    text in development where a human is reading the terminal.
    """
    handler = logging.StreamHandler()
    if as_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(levelname)s %(name)s %(message)s")
        )

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    if as_json:
        # Our own http_request event carries the same request with the
        # user attached, so uvicorn's access line is duplicate volume
        # against a log retention measured in weeks.
        logging.getLogger("uvicorn.access").disabled = True


def emit(event: str, **fields: Any) -> None:
    """Record one event. Never raises.

    Goes two places: stdout as a JSON line, which is what Railway charts,
    and — for the events worth keeping — a row in the database, which is
    what the admin dashboard reads. metrics.STORED decides which.
    """
    try:
        clean = {k: v for k, v in fields.items() if v is not None}
        log.info(event, extra={"fields": {"event": event, **clean}})
    except Exception:  # pragma: no cover - logging must never bite
        pass

    try:
        from . import metrics

        metrics.record(event, fields)
    except Exception:
        pass


def actor(user: Any) -> dict[str, Any]:
    """The standard way to attach who did something.

    Guests are marked rather than hidden: nearly every question worth
    asking about usage is really "how does this differ between guests and
    signed-in managers".
    """
    if user is None:
        return {}
    return {"user": user.id, "guest": bool(user.is_guest)}


def note_session(user: Any) -> None:
    """Emit `session_start` the first time a user appears in a while.

    Called from the current-user dependency, so it sees every
    authenticated request without each route having to remember. The
    dedupe window is what turns a stream of requests into a countable
    number of visits.
    """
    try:
        now = time.monotonic()
        with _seen_lock:
            last = _seen.get(user.id)
            if last is not None and now - last < SESSION_GAP_SECONDS:
                _seen[user.id] = now
                return
            _seen[user.id] = now

            # Bounded: guests are throwaway, so this would otherwise grow
            # with every visitor for as long as the process lives.
            if len(_seen) > 5000:
                cutoff = now - SESSION_GAP_SECONDS
                for key in [k for k, v in _seen.items() if v < cutoff]:
                    _seen.pop(key, None)

        emit("session_start", **actor(user))
    except Exception:
        pass


def forget_session(user_id: int) -> None:
    """Drop a signed-out user so their next visit counts as a new one."""
    with _seen_lock:
        _seen.pop(user_id, None)
