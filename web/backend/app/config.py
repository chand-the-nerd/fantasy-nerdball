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

        # Read early: the SQLite fallback below is written onto this volume,
        # so it survives a redeploy even without Postgres attached.
        self.data_dir = Path(os.getenv("NERDBALL_DATA_DIR", "/data"))

        # Signs the session cookie. Generate with: openssl rand -hex 32
        self.secret_key = os.getenv("SECRET_KEY", "dev-only-do-not-use-in-prod")

        # Railway's Postgres plugin sets DATABASE_URL. Without it we fall back
        # to SQLite ON THE MOUNTED VOLUME. The old fallback wrote to the
        # container's own filesystem, which is wiped on every redeploy — so
        # squads and settings silently vanished each time you shipped.
        raw_database_url = os.getenv("DATABASE_URL", "")
        self.database_is_fallback = not raw_database_url
        if not raw_database_url:
            raw_database_url = f"sqlite:///{self.data_dir / 'nerdball.db'}"
        self.database_url = self._normalise_db_url(raw_database_url)

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

        # Where the fantasy-nerdball engine was cloned to at build time.
        self.engine_dir = Path(
            os.getenv("NERDBALL_ENGINE_DIR", "/app/nerdball")
        )

        # Static frontend build output.
        self.static_dir = Path(os.getenv("STATIC_DIR", "/app/static"))

        self.current_season = os.getenv("CURRENT_SEASON", "2026-27")

        # Authorises the scheduled-maintenance endpoint. Unset means the
        # endpoint is closed, which is the right default for a public URL.
        self.cron_secret = os.getenv("CRON_SECRET", "")
        # Runs the same maintenance from inside the app, so an external
        # scheduler is optional rather than required.
        self.history_auto_update = os.getenv(
            "HISTORY_AUTO_UPDATE", "true"
        ).lower() in {"1", "true", "yes"}
        self.history_check_minutes = int(os.getenv("HISTORY_CHECK_MINUTES", "60"))

        # The admin page belongs to accounts, not to a password. The owner is
        # the first address in ALLOWED_EMAILS; ADMIN_EMAILS names any others.
        # Both are re-applied on every sign-in, so adding an address here and
        # signing in again is all it takes to grant or keep admin rights.
        self.admin_emails = _csv_env("ADMIN_EMAILS")

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
    def database_backend(self) -> str:
        if self.database_url.startswith("sqlite"):
            return "sqlite"
        return "postgres"

    @property
    def owner_email(self) -> str:
        """The first allowed address. Always an admin."""
        return self.allowed_emails[0] if self.allowed_emails else ""

    def is_admin_email(self, email: str) -> bool:
        email = (email or "").lower().strip()
        if not email:
            return False
        return email == self.owner_email or email in self.admin_emails

    @property
    def google_configured(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
