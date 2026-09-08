"""Fantasy Nerdball web application."""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from .auth import current_user
from .config import settings
from .db import get_session, init_db
from .engine import jobs
from .models import User
from .routers import (
    admin,
    auth,
    cron,
    me,
    performance,
    plans,
    players,
    runs,
    squads,
    teams,
)
from .services import fpl

# The optimiser imports matplotlib for its performance plots. Without a
# headless backend it tries to find a display and the run dies.
os.environ.setdefault("MPLBACKEND", "Agg")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("nerdball")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    init_db()

    if settings.database_is_fallback:
        log.warning(
            "No DATABASE_URL, so storage is SQLite at %s. That file lives on "
            "the mounted volume and survives redeploys, but attach Railway's "
            "Postgres plugin for anything long-lived.",
            settings.data_dir / "nerdball.db",
        )
    jobs.start_worker()
    start_history_scheduler()
    if not settings.engine_dir.exists():
        log.warning(
            "Optimiser not found at %s. Runs will fail until it's cloned.",
            settings.engine_dir,
        )
    else:
        warm_engine()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan, docs_url="/api/docs")

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
    session_cookie="nerdball_session",
    https_only=not settings.dev_mode,
    same_site="lax",
    max_age=60 * 60 * 24 * 30,
)

app.include_router(auth.router)
app.include_router(me.router)
app.include_router(runs.router)
app.include_router(plans.router)
app.include_router(squads.router)
app.include_router(players.router)
app.include_router(teams.router)
app.include_router(cron.router)
app.include_router(performance.router)
app.include_router(admin.router)


def warm_engine() -> None:
    """Import the optimiser and prime the FPL cache at boot.

    The engine is imported lazily inside the pipeline, so the first run
    after a deploy paid for the whole import tree and a cold
    bootstrap-static download on top of its own work. None of that is
    per-user, so it happens here instead and nobody waits for it.

    In a thread, because it costs a few seconds and the health check
    should not wait on it. A run that starts mid-warm blocks on the
    import lock, which is the same work it would have done itself.
    """

    def warm() -> None:
        try:
            from .engine.runtime_config import ensure_engine_on_path

            ensure_engine_on_path(settings.engine_dir)
            import main  # noqa: F401
        except Exception:
            # A failure here costs nothing: the pipeline imports the
            # engine itself and will surface the real error there.
            log.warning("Could not pre-import the optimiser", exc_info=True)
            return

        try:
            fpl.bootstrap()
        except Exception:
            log.warning("Could not pre-fetch FPL bootstrap data")

        try:
            # The engine keeps its own shared cache, separate from the
            # web layer's. It honours FPL_CACHE_SECONDS (600 by
            # default), so raise that if runs are spread far apart.
            from src.api.fpl_client import FPLClient

            client = FPLClient()
            client.get_bootstrap_static()
            client.get_fixtures()
        except Exception:
            log.warning("Could not pre-fetch engine FPL data")

        log.info("Optimiser warmed and FPL data cached")

    threading.Thread(target=warm, name="engine-warm", daemon=True).start()


def start_history_scheduler() -> None:
    """Tops up player history from inside the app, on a timer.

    An external scheduler works just as well and is easier to see, but this
    means the app is correct on its own rather than depending on one being
    wired up. The check is cheap when there's nothing to do.
    """
    if not settings.history_auto_update:
        return

    from .services import history_updater

    def loop() -> None:
        # A short first pass so a redeploy catches up promptly.
        time.sleep(90)
        while True:
            try:
                result = history_updater.update()
                if result.get("updated"):
                    log.info(
                        "Stored player history for GW%s",
                        ", ".join(str(g) for g in result["updated"]),
                    )
                # A backlog is worked through a few gameweeks at a time, so
                # come back sooner while there is more to do.
                if result.get("remaining"):
                    time.sleep(60)
                    continue
            except Exception:
                log.exception("Player history update failed")
            time.sleep(max(5, settings.history_check_minutes) * 60)

    threading.Thread(target=loop, name="history-scheduler", daemon=True).start()


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "engine": settings.engine_dir.exists(),
        # Railway sets this to the deployed commit. Lets you confirm which
        # build is actually live rather than inferring it from behaviour.
        "commit": os.getenv("RAILWAY_GIT_COMMIT_SHA", "unknown")[:8],
        "admin_emails": len(settings.admin_emails) + bool(settings.owner_email),
        # If this says sqlite, no Postgres is attached. Data still persists
        # (the file is on the volume), but Postgres is the intended setup.
        "database": settings.database_backend,
        "database_is_fallback": settings.database_is_fallback,
        "routes": len([r for r in app.routes if getattr(r, "path", "").startswith("/api/")]),
    }


@app.get("/api/gameweek")
def gameweek(user: User = Depends(current_user)) -> dict:
    """The gameweek to plan for, plus its deadline."""
    try:
        data = fpl.bootstrap()
    except Exception:
        return JSONResponse(
            {"detail": "The FPL API isn't responding. Pick a gameweek manually."},
            status_code=503,
        )

    events = data.get("events", [])
    current = fpl.current_gameweek()
    event = next((e for e in events if int(e["id"]) == current), None)
    return {
        "gameweek": current,
        "season": settings.current_season,
        "deadline": event.get("deadline_time") if event else None,
        "average_last_gw": next(
            (e.get("average_entry_score") for e in reversed(events) if e.get("finished")),
            None,
        ),
    }


@app.post("/api/sync")
def sync(
    user: User = Depends(current_user), session: Session = Depends(get_session)
) -> dict:
    """Pull fresh global benchmarks and, if linked, the manager's real points."""
    gameweeks = fpl.sync_global_stats(session)
    entries = fpl.sync_entry_history(session, user)
    return {"gameweeks_synced": gameweeks, "results_synced": entries}


# The built frontend is mounted last so API routes always win.
if settings.static_dir.exists():
    assets = settings.static_dir / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str, request: Request):
        # An unmatched /api/ path is a missing route, not a page. Falling
        # through to index.html here makes a stale deployment look like a
        # working one: you request an endpoint that doesn't exist and get
        # the homepage back instead of a 404.
        if full_path.startswith("api/"):
            return JSONResponse(
                {"detail": f"No such endpoint: /{full_path}"}, status_code=404
            )

        candidate = settings.static_dir / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(settings.static_dir / "index.html")
