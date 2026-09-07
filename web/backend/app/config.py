"""Application settings, read from the environment."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _csv_env(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [v.strip().lower() for v in raw.split(",") if v.strip()]


class Settings:
    """Runtime settings. Everything is overridable by environment variable."""

    def __init__(self) -> None:
        self.app_name = "Fantasy Nerdball"

        # Railway injects PORT. Uvicorn is started with it in the Dockerfile.
        self.port = int(os.getenv("PORT", "8000"))

        # Signs the session cookie. Generate with: openssl rand -hex 32
        self.secret_key = os.getenv("SECRET_KEY", "dev-only-do-not-use-in-prod")

        # Railway's Postgres plugin sets DATABASE_URL.
        self.database_url = self._normalise_db_url(
            os.getenv("DATABASE_URL", "sqlite:///./nerdball.db")
        )

        # Google OAuth. Create credentials at console.cloud.google.com.
        self.google_client_id = os.getenv("GOOGLE_CLIENT_ID", "")
        self.google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")

        # Public origin of the deployment, used to build the OAuth callback.
        # Railway sets RAILWAY_PUBLIC_DOMAIN automatically.
        explicit = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
        railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "")
        if explicit:
            self.public_base_url = explicit
        elif railway_domain:
            self.public_base_url = f"https://{railway_domain}"
        else:
            self.public_base_url = f"http://localhost:{self.port}"

        # Who is allowed in. The first address is the owner and is granted
        # admin rights on first sign-in.
        self.allowed_emails = _csv_env("ALLOWED_EMAILS")
        self.max_users = int(os.getenv("MAX_USERS", "6"))

        # Where the optimiser's working files live. On Railway, attach a
        # volume and point this at its mount path so cached reference data
        # survives redeploys.
        self.data_dir = Path(os.getenv("NERDBALL_DATA_DIR", "/data"))

        # Where the fantasy-nerdball engine was cloned to at build time.
        self.engine_dir = Path(
            os.getenv("NERDBALL_ENGINE_DIR", "/app/nerdball")
        )

        # Static frontend build output.
        self.static_dir = Path(os.getenv("STATIC_DIR", "/app/static"))

        self.current_season = os.getenv("CURRENT_SEASON", "2026-27")

        # Gates the admin page. Unset means the admin page stays locked for
        # everyone, which is a safer default than leaving it open.
        self.admin_password = os.getenv("ADMIN_PASSWORD", "")
        # How long an unlock lasts before the password is asked for again.
        self.admin_session_minutes = int(os.getenv("ADMIN_SESSION_MINUTES", "30"))

        # A single optimisation run is CPU-bound and chdir-based, so runs
        # are serialised. This only caps how many can queue up.
        self.max_queued_runs = int(os.getenv("MAX_QUEUED_RUNS", "20"))

        self.dev_mode = os.getenv("DEV_MODE", "").lower() in {"1", "true", "yes"}
        # Lets you work on the UI without Google credentials configured.
        self.dev_login_email = os.getenv("DEV_LOGIN_EMAIL", "")

    @staticmethod
    def _normalise_db_url(url: str) -> str:
        # SQLAlchemy 2 rejects the postgres:// scheme some providers hand out.
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    @property
    def oauth_redirect_uri(self) -> str:
        return f"{self.public_base_url}/api/auth/callback"

    @property
    def admin_configured(self) -> bool:
        return bool(self.admin_password)

    @property
    def google_configured(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
