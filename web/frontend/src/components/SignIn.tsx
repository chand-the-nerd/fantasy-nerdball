import { useEffect, useState } from "react";
import { api } from "../lib/api";

const MESSAGES: Record<string, string> = {
  not_invited:
    "That Google account isn't on the list. Ask whoever runs this deployment to add it.",
  league_full: "Every seat is taken. A manager has to be removed before you can join.",
  signin_failed: "Google sign-in didn't complete. Try again.",
  no_verified_email: "That account has no verified email address on it.",
};

export function SignIn() {
  const [config, setConfig] = useState<{ google: boolean; dev_login: boolean } | null>(null);
  const params = new URLSearchParams(window.location.search);
  const error = params.get("error");

  useEffect(() => {
    api.authConfig().then(setConfig).catch(() => setConfig({ google: false, dev_login: false }));
  }, []);

  return (
    <div className="signin">
      <div className="signin-inner">
        <h1>
          Fantasy
          <em>Nerdball</em>
        </h1>
        <p>
          Taking the fun out of Fantasy Premier League.
          FPL Optimisation using Integer Linear Programming, xG Models and FPL Stats.
        </p>

        {error && MESSAGES[error] && <div className="notice bad">{MESSAGES[error]}</div>}

        {config?.google && (
          <a className="google-btn" href="/api/auth/login">
            <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden="true">
              <path fill="#4285F4" d="M45.1 24.5c0-1.6-.1-3.1-.4-4.5H24v8.5h11.8c-.5 2.7-2 5-4.4 6.5v5.4h7.1c4.2-3.8 6.6-9.5 6.6-15.9z" />
              <path fill="#34A853" d="M24 46c5.9 0 10.9-2 14.5-5.3l-7.1-5.4c-2 1.3-4.5 2.1-7.4 2.1-5.7 0-10.5-3.8-12.2-9H4.5v5.6C8.1 41.2 15.5 46 24 46z" />
              <path fill="#FBBC05" d="M11.8 28.4c-.4-1.3-.7-2.7-.7-4.4s.3-3 .7-4.4v-5.6H4.5C2.9 17.2 2 20.5 2 24s.9 6.8 2.5 10l7.3-5.6z" />
              <path fill="#EA4335" d="M24 10.3c3.2 0 6.1 1.1 8.4 3.3l6.3-6.3C34.9 3.8 29.9 2 24 2 15.5 2 8.1 6.8 4.5 14l7.3 5.6c1.7-5.2 6.5-9.3 12.2-9.3z" />
            </svg>
            Continue with Google
          </a>
        )}

        {config && !config.google && (
          <div className="notice bad">
            Google sign-in isn't configured. Set GOOGLE_CLIENT_ID and
            GOOGLE_CLIENT_SECRET on the service, then redeploy.
          </div>
        )}

        {config?.dev_login && (
          <button
            className="link-button"
            style={{ marginTop: 18, display: "block" }}
            onClick={() => api.devLogin().then(() => window.location.reload())}
            type="button"
          >
            Sign in as the local developer
          </button>
        )}
      </div>
    </div>
  );
}
