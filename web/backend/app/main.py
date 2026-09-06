"""Fantasy Nerdball web application."""

from __future__ import annotations

import logging
import os
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
from .routers import admin, auth, me, performance, runs, squads
from .services import fpl

# The optimiser imports matplotlib for its performance plots. Without a
# headless backend it tries to find a display and the run dies.
os.environ.setdefault("MPLBACKEND", "Agg")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("nerdball")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    jobs.start_worker()
    if not settings.engine_dir.exists():
        log.warning(
            "Optimiser not found at %s. Runs will fail until it's cloned.",
            settings.engine_dir,
        )
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
app.include_router(squads.router)
app.include_router(performance.router)
app.include_router(admin.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "engine": settings.engine_dir.exists()}


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
        candidate = settings.static_dir / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(settings.static_dir / "index.html")
