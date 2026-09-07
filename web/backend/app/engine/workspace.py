"""Filesystem workspace for an optimiser run.

The optimiser reads and writes relative paths (``data/``, ``squads/gw{n}/``)
off the current working directory. That is fine for a laptop and wrong for a
container with an ephemeral disk, so each manager gets a directory on the
mounted volume, and the shared reference data is linked into it rather than
downloaded five times over.

``os.chdir`` is process-wide, so runs are serialised behind ``RUN_LOCK``.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from ..config import settings

RUN_LOCK = threading.Lock()

# Reference data the engine builds up over a season: team ratings, standings
# history, per-player history. Shared across managers, so it lives once.
SHARED_DATA_DIRNAME = "shared-data"


def shared_data_dir() -> Path:
    path = settings.data_dir / SHARED_DATA_DIRNAME
    (path / "historic").mkdir(parents=True, exist_ok=True)
    (path / "players").mkdir(parents=True, exist_ok=True)
    return path


def user_workspace(user_id: int, season: str) -> Path:
    """Directory the engine will treat as its project root."""
    path = settings.data_dir / "managers" / str(user_id) / season
    try:
        (path / "squads").mkdir(parents=True, exist_ok=True)
    except PermissionError as error:
        raise PermissionError(
            f"Can't write to {settings.data_dir}. On Railway this usually means "
            "the mounted volume is owned by root while the app runs as a "
            "non-root user. Redeploy so the entrypoint can take ownership of "
            f"the mount, or check the volume is mounted at {settings.data_dir}."
        ) from error
    return path


def _link_shared_data(workspace: Path) -> None:
    """Point ``<workspace>/data`` at the shared reference directory."""
    target = shared_data_dir()
    link = workspace / "data"

    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        # A real directory from an earlier layout; leave it alone.
        return

    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        # Some filesystems refuse symlinks. Fall back to a private copy.
        link.mkdir(parents=True, exist_ok=True)
        (link / "historic").mkdir(exist_ok=True)
        (link / "players").mkdir(exist_ok=True)


def seed_previous_squad(workspace: Path, gameweek: int, rows: list[dict]) -> None:
    """Write a stored squad back out as the CSV the engine expects to find."""
    if not rows:
        return
    squad_dir = workspace / "squads" / f"gw{gameweek}"
    squad_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(squad_dir / "full_squad.csv", index=False)

    # transfer_evaluator reads the simplified file when judging a hold.
    simple_cols = [
        c
        for c in ("id", "player_code", "display_name", "position", "now_cost_m",
                  "team", "projected_points", "squad_role")
        if c in frame.columns
    ]
    if simple_cols:
        simple = frame[simple_cols].rename(
            columns={"display_name": "player", "now_cost_m": "price", "team": "club"}
        )
        simple.to_csv(squad_dir / "full_squad_simple.csv", index=False)


def read_saved_squad(workspace: Path, gameweek: int) -> list[dict]:
    """Read back what the engine just saved, for storage in the database."""
    path = workspace / "squads" / f"gw{gameweek}" / "full_squad.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return frame.where(pd.notnull(frame), None).to_dict(orient="records")


@contextlib.contextmanager
def run_in_workspace(workspace: Path) -> Iterator[Path]:
    """Serialise runs and point the process at this manager's directory."""
    _link_shared_data(workspace)
    with RUN_LOCK:
        previous = Path.cwd()
        os.chdir(workspace)
        try:
            yield workspace
        finally:
            os.chdir(previous)
