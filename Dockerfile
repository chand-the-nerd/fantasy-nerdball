# syntax=docker/dockerfile:1
#
# Builds the web app together with the optimiser that lives alongside it in
# this repository. Build context is the repository root.

# ── Stage 1: build the browser app ───────────────────────────────────────
FROM node:20-slim AS frontend

WORKDIR /build
COPY web/frontend/package.json web/frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY web/frontend/ ./
RUN npm run build


# ── Stage 2: runtime ─────────────────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    NERDBALL_ENGINE_DIR=/app/nerdball \
    STATIC_DIR=/app/static \
    NERDBALL_DATA_DIR=/data

# PuLP ships its own CBC solver binary, which needs libstdc++ and libgomp.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libstdc++6 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY web/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# The web layer.
COPY web/backend/app ./app

# The optimiser. Copied from this repository rather than cloned, so the model
# and the app that serves it are always the same commit.
COPY main.py current_rules.py config_example.py ./nerdball/
COPY src ./nerdball/src
COPY ml ./nerdball/ml
COPY utility_scripts ./nerdball/utility_scripts

# config.py is gitignored here because it's the file you edit locally. The web
# app builds a Config per manager at runtime, but the module still has to be
# importable, so seed it from the example.
RUN cp ./nerdball/config_example.py ./nerdball/config.py

COPY --from=frontend /build/dist ./static

RUN mkdir -p /data && \
    useradd --create-home --uid 10001 nerdball && \
    chown -R nerdball:nerdball /app /data
USER nerdball

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/api/health" || exit 1

# Exactly one worker. The optimiser switches the process working directory and
# holds a lock while it runs, so a second worker would fight it.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
