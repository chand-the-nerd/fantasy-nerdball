"""Background worker for optimisation runs.

A run takes minutes and pins a CPU, so requests return a run id immediately
and the browser polls for progress. One worker thread processes the queue in
order, which also keeps the engine's working-directory switching safe.
"""

from __future__ import annotations

import datetime as dt
import io
import queue
import threading
import time
import traceback
from contextlib import redirect_stdout
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import capacity, events
from ..config import settings
from ..db import session_scope
from ..models import Plan, PlayerScores, Run, Squad, User, utcnow
from ..services import fpl
from .pipeline import run_optimisation
from .planner import run_plan

# Items are (kind, id): single runs and multi-week plans share one worker, so
# they can't fight over the engine's working directory.
_queue: "queue.Queue[tuple[str, int]]" = queue.Queue()
_worker: threading.Thread | None = None
_worker_lock = threading.Lock()
# Whether the single worker is mid-job. Read by the heartbeat, which is
# how utilisation of the one thing that can't be parallelised gets onto a
# dashboard.
_busy = False


def _should_store_scores(cache: PlayerScores | None, gameweek: int) -> bool:
    """Whether this run's scored pool should become the Players tab's data.

    Runs for the gameweek being played always do. A run for a future gameweek
    only does so when there is nothing better to show, or when what is stored
    is older still — otherwise planning three weeks ahead would quietly
    replace the rankings for the week whose deadline is next.
    """
    if cache is None:
        return True
    try:
        current = fpl.current_gameweek()
    except Exception:
        # No way to tell which of the two is the live one; the newer run is
        # the better guess, which is what this did before.
        return True
    if gameweek == current:
        return True
    if cache.gameweek == current:
        return False
    # Neither is the live gameweek: prefer whichever is closer to it.
    return abs(gameweek - current) < abs(cache.gameweek - current)


# ── Plans ────────────────────────────────────────────────────────────────


def append_plan_log(plan_id: int, line: str) -> None:
    with session_scope() as session:
        plan = session.get(Plan, plan_id)
        if plan is None:
            return
        existing = (plan.log or "").splitlines()[-200:]
        existing.append(line)
        plan.log = "\n".join(existing)


def _execute_plan(plan_id: int) -> None:
    started = time.perf_counter()
    with session_scope() as session:
        plan = session.get(Plan, plan_id)
        if plan is None or plan.status != "queued":
            return
        user = session.get(User, plan.user_id)
        if user is None:
            return
        plan.status = "running"
        plan.started_at = utcnow()
        events.emit(
            "plan_started",
            plan=plan.id,
            user=user.id,
            weeks=plan.weeks,
            queue_depth=queue_depth(session),
        )

        squads = session.scalars(
            select(Squad).where(Squad.user_id == user.id, Squad.season == plan.season)
        ).all()
        context = {
            "user_id": user.id,
            "season": plan.season,
            "start_gameweek": plan.start_gameweek,
            "weeks": plan.weeks,
            "chips": {int(gw): chip for gw, chip in (plan.chips or {}).items()},
            "settings_row": user.settings,
            "previous_squads": {s.gameweek: (s.engine_rows or []) for s in squads},
        }

    stream = _LogStream(plan_id)

    def store_week(done: int, _week: dict) -> None:
        with session_scope() as session:
            row = session.get(Plan, plan_id)
            if row is not None:
                row.progress = done

    try:
        with capacity.MemorySampler() as memory, redirect_stdout(stream):
            weeks = run_plan(
                user_id=context["user_id"],
                season=context["season"],
                start_gameweek=context["start_gameweek"],
                weeks=context["weeks"],
                chips=context["chips"],
                settings_row=context["settings_row"],
                previous_squads=context["previous_squads"],
                on_progress=lambda msg: append_plan_log(plan_id, msg),
                on_week=store_week,
            )
        stream.flush()
    except Exception as error:
        stream.flush()
        append_plan_log(plan_id, "That didn't work out. Details below.")
        with session_scope() as session:
            row = session.get(Plan, plan_id)
            if row is not None:
                row.status = "failed"
                row.error = f"{type(error).__name__}: {error}\n\n{stream.tail()}".strip()
                row.finished_at = utcnow()
        events.emit(
            "plan_failed",
            plan=plan_id,
            seconds=round(time.perf_counter() - started, 1),
            error_type=type(error).__name__,
        )
        traceback.print_exc()
        return

    with session_scope() as session:
        row = session.get(Plan, plan_id)
        if row is not None:
            row.payload = weeks
            row.progress = len(weeks)
            row.status = "complete"
            row.finished_at = utcnow()

    events.emit(
        "plan_finished",
        plan=plan_id,
        weeks=len(weeks),
        seconds=round(time.perf_counter() - started, 1),
        rss_cost_mb=memory.cost_mb,
        rss_peak_mb=memory.peak,
    )


class _LogStream(io.TextIOBase):
    """Swallows the engine's stdout, keeping it only for diagnostics.

    The optimiser prints a great deal — per-player scores, solver output,
    intermediate tables. That is the right level for a terminal and the wrong
    level for someone watching a progress panel, so none of it is shown. The
    curated milestones come from the pipeline's own progress callback instead.
    The raw output is retained here so a failed run can still be diagnosed.
    """

    def __init__(self, run_id: int) -> None:
        self.run_id = run_id
        self.captured: list[str] = []
        self._pending = ""

    def write(self, text: str) -> int:  # noqa: D102
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            line = line.rstrip()
            if line:
                # Bounded: a long run can print thousands of lines.
                self.captured.append(line)
                del self.captured[:-400]
        return len(text)

    def flush(self) -> None:  # noqa: D102
        if self._pending.strip():
            self.captured.append(self._pending.strip())
            self._pending = ""

    def tail(self, lines: int = 40) -> str:
        return "\n".join(self.captured[-lines:])


def append_log(run_id: int, line: str) -> None:
    with session_scope() as session:
        run = session.get(Run, run_id)
        if run is None:
            return
        # Keep the tail bounded; nobody scrolls back past a few hundred lines.
        existing = (run.log or "").splitlines()[-400:]
        existing.append(line)
        run.log = "\n".join(existing)


def enqueue(run_id: int) -> None:
    _enqueue(("run", run_id))


def enqueue_plan(plan_id: int) -> None:
    _enqueue(("plan", plan_id))


def _enqueue(item: tuple[str, int]) -> None:
    if _queue.qsize() >= settings.max_queued_runs:
        raise RuntimeError("The optimiser queue is full. Try again in a few minutes.")
    _queue.put(item)
    start_worker()


def queue_depth(session: Session | None = None) -> int:
    """How many jobs are waiting, across the whole deployment.

    Counted from the database rather than from this process's queue.
    The in-memory queue is per worker process, so with more than one it
    would report a fraction of the real backlog — which is exactly the
    moment the number starts to matter.

    Pass an open session when there is one: opening a second connection
    inside a transaction that is midway through a write is asking for
    trouble on SQLite.
    """
    def count(active: Session) -> int:
        runs = active.scalar(
            select(func.count())
            .select_from(Run)
            .where(Run.status == "queued")
        )
        plans = active.scalar(
            select(func.count())
            .select_from(Plan)
            .where(Plan.status == "queued")
        )
        return int(runs or 0) + int(plans or 0)

    try:
        if session is not None:
            return count(session)
        with session_scope() as fresh:
            return count(fresh)
    except Exception:
        # Never let a diagnostic take a run down with it.
        return _queue.qsize()


def is_busy() -> bool:
    return _busy


def recover_orphans() -> int:
    """Put back anything the last process was holding when it stopped.

    The queue lives in memory, so a deploy takes it with it. Rows left
    saying "queued" are re-queued; rows left saying "running" were mid-
    flight when the process went away, and there is no way to resume the
    engine partway, so they are failed with an explanation rather than
    left spinning forever in somebody's browser.

    Runs once at boot, under the same lease as the other shared jobs, so
    several workers don't each re-queue the same backlog.
    """
    recovered = 0
    with session_scope() as session:
        interrupted = session.scalars(
            select(Run).where(Run.status == "running")
        ).all()
        for run in interrupted:
            run.status = "failed"
            run.error = (
                "The server restarted while this was running. Nothing "
                "was saved — start it again."
            )
            run.finished_at = utcnow()
            recovered += 1

        for plan in session.scalars(
            select(Plan).where(Plan.status == "running")
        ):
            plan.status = "failed"
            plan.error = (
                "The server restarted while this was running. Start it "
                "again."
            )
            plan.finished_at = utcnow()
            recovered += 1

        waiting = session.scalars(
            select(Run)
            .where(Run.status == "queued")
            .order_by(Run.created_at)
        ).all()
        queued_plans = session.scalars(
            select(Plan)
            .where(Plan.status == "queued")
            .order_by(Plan.created_at)
        ).all()

    # Re-queued outside the transaction: the worker picks jobs up by id
    # and needs to find them committed.
    for run in waiting:
        try:
            _queue.put_nowait(("run", run.id))
            recovered += 1
        except queue.Full:
            break
    for plan in queued_plans:
        try:
            _queue.put_nowait(("plan", plan.id))
            recovered += 1
        except queue.Full:
            break

    if recovered:
        events.emit(
            "orphans_recovered",
            failed=len(interrupted),
            requeued=len(waiting) + len(queued_plans),
        )
    return recovered


def start_worker() -> None:
    global _worker
    with _worker_lock:
        if _worker is not None and _worker.is_alive():
            return
        _worker = threading.Thread(target=_loop, name="nerdball-worker", daemon=True)
        _worker.start()


def _loop() -> None:
    global _busy
    while True:
        kind, item_id = _queue.get()
        _busy = True
        try:
            if kind == "plan":
                _execute_plan(item_id)
            else:
                _execute(item_id)
        except Exception:  # a worker that dies takes the queue with it
            events.emit("worker_error", kind=kind, id=item_id)
            traceback.print_exc()
        finally:
            _busy = False
            _queue.task_done()


def _load_context(run_id: int) -> dict[str, Any] | None:
    with session_scope() as session:
        run = session.get(Run, run_id)
        if run is None or run.status != "queued":
            return None
        user = session.get(User, run.user_id)
        if user is None:
            return None

        run.status = "running"
        run.started_at = utcnow()

        # How long this run sat in the queue. With one worker for the
        # whole site this is the number that turns into "the app feels
        # slow" long before any request does.
        waited = None
        if run.created_at is not None:
            created = run.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=dt.timezone.utc)
            waited = round((utcnow() - created).total_seconds(), 1)

        events.emit(
            "run_started",
            run=run.id,
            user=user.id,
            guest=bool(user.is_guest),
            gameweek=run.gameweek,
            waited_seconds=waited,
            queue_depth=queue_depth(session),
        )

        # Hand the engine every squad this manager has saved this season. It
        # asks for a specific gameweek, which varies with Free Hit.
        squads = session.scalars(
            select(Squad).where(Squad.user_id == user.id, Squad.season == run.season)
        ).all()

        row = user.settings
        return {
            "user_id": user.id,
            "gameweek": run.gameweek,
            "season": run.season,
            "settings_row": row,
            "previous_squads": {s.gameweek: (s.engine_rows or []) for s in squads},
        }


def _execute(run_id: int) -> None:
    context = _load_context(run_id)
    if context is None:
        return

    started = time.perf_counter()
    stream = _LogStream(run_id)
    try:
        with capacity.MemorySampler() as memory, redirect_stdout(stream):
            result = run_optimisation(
                user_id=context["user_id"],
                gameweek=context["gameweek"],
                season=context["season"],
                settings_row=context["settings_row"],
                previous_squads=context["previous_squads"],
                on_progress=lambda msg: append_log(run_id, msg),
            )
        stream.flush()
    except Exception as error:
        stream.flush()
        detail = f"{type(error).__name__}: {error}"
        append_log(run_id, "That didn't work out. Details below.")
        with session_scope() as session:
            run = session.get(Run, run_id)
            if run is not None:
                run.status = "failed"
                # The engine's own output is the useful part when something
                # breaks, so it goes here rather than in the progress feed.
                run.error = f"{detail}\n\n{stream.tail()}".strip()
                run.finished_at = utcnow()
        events.emit(
            "run_failed",
            run=run_id,
            user=context["user_id"],
            gameweek=context["gameweek"],
            seconds=round(time.perf_counter() - started, 1),
            error_type=type(error).__name__,
            rss_peak_mb=memory.peak,
        )
        traceback.print_exc()
        return

    _store_result(run_id, context, result)
    events.emit(
        "run_finished",
        run=run_id,
        user=context["user_id"],
        gameweek=context["gameweek"],
        seconds=round(time.perf_counter() - started, 1),
        projected_points=round(
            float(result.get("squad", {}).get("projected_points") or 0), 1
        ),
        transfers=int(result.get("squad", {}).get("transfers_made") or 0),
        chip=result.get("squad", {}).get("chip") or None,
        # What the run cost on top of an already-warm process, and where
        # that left the process overall. Together with the container's
        # limit these are what decide how many runs could ever go at
        # once — see web/CAPACITY.md.
        rss_cost_mb=memory.cost_mb,
        rss_peak_mb=memory.peak,
    )


def _store_result(run_id: int, context: dict[str, Any], result: dict) -> None:
    squad_data = result["squad"]
    with session_scope() as session:
        run = session.get(Run, run_id)
        if run is None:
            return

        squad = session.scalar(
            select(Squad).where(
                Squad.user_id == context["user_id"],
                Squad.season == context["season"],
                Squad.gameweek == context["gameweek"],
            )
        )
        if squad is None:
            squad = Squad(
                user_id=context["user_id"],
                season=context["season"],
                gameweek=context["gameweek"],
            )
            session.add(squad)

        squad.formation = squad_data.get("formation", "")
        squad.projected_points = float(squad_data.get("projected_points") or 0.0)
        squad.squad_value = float(squad_data.get("squad_value") or 0.0)
        squad.bank = float(squad_data.get("bank") or 0.0)
        squad.transfers_made = int(squad_data.get("transfers_made") or 0)
        squad.penalty_points = int(squad_data.get("penalty_points") or 0)
        squad.chip = squad_data.get("chip", "")
        squad.payload = squad_data
        squad.engine_rows = result.get("engine_rows", [])
        squad.option_rows = result.get("option_rows") or {}
        # A fresh run replaces the options, so whatever was active last time
        # no longer refers to anything. Back to the recommendation.
        squad.active_option = squad_data.get("active_option") or "option-1"
        session.flush()

        # The scored pool feeds the Players tab. Stored per season and
        # replaced each run, so it always reflects the current settings.
        #
        # A run for a future gameweek does not replace it. Those scores look
        # ahead from the gameweek that was run, so a plan for gameweek 5 would
        # have the rankings skipping over the fixtures you are actually
        # picking for this week.
        scored = result.get("scored_players") or []
        if scored:
            cache = session.scalar(
                select(PlayerScores).where(
                    PlayerScores.user_id == context["user_id"],
                    PlayerScores.season == context["season"],
                )
            )
            if _should_store_scores(cache, context["gameweek"]):
                if cache is None:
                    cache = PlayerScores(
                        user_id=context["user_id"], season=context["season"]
                    )
                    session.add(cache)
                cache.gameweek = context["gameweek"]
                cache.look_ahead = int(result.get("look_ahead") or 1)
                cache.players = scored
                cache.created_at = utcnow()

        run.status = "complete"
        run.squad_id = squad.id
        run.finished_at = utcnow()
        run.result = {"projected_points": squad.projected_points,
                      "formation": squad.formation}
