#!/usr/bin/env bash
# Starts the API for local development. Run the Vite dev server alongside it
# in a second terminal:
#
#   Terminal 1:  ./web/dev.sh
#   Terminal 2:  cd web/frontend && npm run dev
#
# Then open http://localhost:5173 — NOT 8000. Vite serves the interface and
# proxies /api through to this process, which is what gives you hot reload.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export SECRET_KEY="${SECRET_KEY:-dev-local-not-a-secret}"
export DEV_MODE=true
export DEV_LOGIN_EMAIL="${DEV_LOGIN_EMAIL:-you@example.com}"
export ALLOWED_EMAILS="${ALLOWED_EMAILS:-$DEV_LOGIN_EMAIL}"
export CURRENT_SEASON="${CURRENT_SEASON:-2026-27}"

# The repository root is the optimiser, so point the engine at ourselves.
export NERDBALL_ENGINE_DIR="$REPO_ROOT"
export NERDBALL_DATA_DIR="${NERDBALL_DATA_DIR:-$REPO_ROOT/localdata}"

# Absolute, deliberately. The optimiser chdirs into each manager's workspace
# while it runs, so a relative SQLite path would resolve somewhere different
# mid-run and produce baffling "no such table" errors.
export DATABASE_URL="${DATABASE_URL:-sqlite:///$REPO_ROOT/localdata/dev.db}"

export MPLBACKEND=Agg

mkdir -p "$NERDBALL_DATA_DIR/shared-data"

# You've likely already got three seasons of history from running main.py.
# Reuse it rather than downloading the lot again.
if [ -d "$REPO_ROOT/data" ] && [ ! -e "$NERDBALL_DATA_DIR/shared-data/players" ]; then
    echo "Seeding cached reference data from ./data"
    cp -R "$REPO_ROOT/data/." "$NERDBALL_DATA_DIR/shared-data/" 2>/dev/null || true
fi

echo "API        http://localhost:8000"
echo "Interface  http://localhost:5173  (start Vite in another terminal)"
echo "Data       $NERDBALL_DATA_DIR"
echo

exec uvicorn app.main:app \
    --app-dir "$REPO_ROOT/web/backend" \
    --reload \
    --reload-dir "$REPO_ROOT/web/backend/app" \
    --port 8000
