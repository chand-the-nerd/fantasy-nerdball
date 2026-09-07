"""Keeps the stored per-gameweek player history topped up.

The optimiser's `_calculate_form_consistency` reads `data/players/` to penalise
spiky, blank-prone form. On a laptop that directory is filled in by an
interactive prompt in `main.py`; a web request has nobody to answer a prompt,
so without this the whole signal silently sits at its neutral value.

The work is driven by which gameweeks have actually finished, not by a clock,
so running this every hour is cheap and running it once a week still catches
up. Whatever triggers it — an external scheduler, the built-in timer, or a
person clicking a button — the outcome is the same.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from ..config import settings
from ..engine.runtime_config import ensure_engine_on_path
from ..engine.workspace import maintenance_workspace, run_in_workspace, shared_data_dir
from . import fpl

# Records the last gameweek written, so five managers running the optimiser on
# the same Friday don't each redo the same work.
MARKER = "history-state.json"

_update_lock = threading.Lock()


def _marker_path():
    return shared_data_dir() / MARKER


def read_state() -> dict[str, Any]:
    path = _marker_path()
    if not path.exists():
        return {"last_gameweek": 0, "updated_at": None}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {"last_gameweek": 0, "updated_at": None}


def _write_state(gameweek: int) -> None:
    from ..models import utcnow

    try:
        _marker_path().write_text(
            json.dumps(
                {"last_gameweek": gameweek, "updated_at": utcnow().isoformat()}
            )
        )
    except OSError:
        pass


def finished_gameweeks() -> list[int]:
    events = fpl.bootstrap(force=True).get("events", [])
    return sorted(int(e["id"]) for e in events if e.get("finished"))


def pending_gameweek() -> int | None:
    """The latest finished gameweek not yet stored, or None if up to date."""
    finished = finished_gameweeks()
    if not finished:
        return None
    latest = finished[-1]
    return latest if latest > int(read_state().get("last_gameweek") or 0) else None


def update(force_gameweek: int | None = None) -> dict[str, Any]:
    """Store the latest finished gameweek's player data.

    Returns a summary rather than raising, so a scheduler polling this gets a
    200 with "nothing to do" instead of an error it might alert on.
    """
    if not _update_lock.acquire(blocking=False):
        return {"status": "busy", "detail": "An update is already running."}

    try:
        target = force_gameweek or pending_gameweek()
        if target is None:
            state = read_state()
            return {
                "status": "up-to-date",
                "last_gameweek": state.get("last_gameweek"),
                "updated_at": state.get("updated_at"),
            }

        ensure_engine_on_path(settings.engine_dir)
        from src.data.player_history_tracker import PlayerHistoryTracker  # type: ignore

        from ..engine.runtime_config import build_config

        # The tracker writes the gameweek *before* the configured one, so aim
        # one past the gameweek being stored.
        config = build_config(
            gameweek=target + 1, season=settings.current_season, settings_row=None
        )

        # The tracker writes to a relative data/players/, so it needs the same
        # symlinked workspace an optimisation run uses, and the same lock —
        # chdir is process-wide.
        import contextlib
        import io

        captured = io.StringIO()
        stored = 0
        with run_in_workspace(maintenance_workspace()):
            tracker = PlayerHistoryTracker(config)
            with contextlib.redirect_stdout(captured):
                tracker.update_all_players()

                # The tracker prints its errors and returns rather than
                # raising, so a failed fetch looks identical to a successful
                # one. Confirm the gameweek actually landed before marking it
                # done, or a transient FPL outage would be recorded as
                # complete and never retried.
                try:
                    written = tracker.get_all_players_gameweek(target)
                    stored = 0 if written is None else len(written)
                except Exception:
                    stored = 0

        tail = captured.getvalue().strip().splitlines()[-5:]

        if stored == 0:
            return {
                "status": "failed",
                "gameweek": target,
                "detail": "Nothing was written, so the gameweek has been left "
                          "outstanding and will be retried.",
                "log": tail,
            }

        _write_state(target)
        return {
            "status": "updated",
            "gameweek": target,
            "players_stored": stored,
            "log": tail,
        }
    except Exception as error:
        return {"status": "failed", "detail": f"{type(error).__name__}: {error}"}
    finally:
        _update_lock.release()
