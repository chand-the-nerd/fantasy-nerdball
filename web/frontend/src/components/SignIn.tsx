import { useEffect, useState } from "react";
import { GuestFeaturesDialog } from "./GuestFeatures";
import { PrivacyNotice } from "./PrivacyNotice";
import { api, ApiError } from "../lib/api";
import type { AuthConfig } from "../lib/types";

const MESSAGES: Record<string, string> = {
  not_invited:
    "That Google account isn't on the list. Ask whoever runs this " +
    "deployment to add it.",
  league_full:
    "Every seat is taken. A manager has to be removed before you can join.",
  signin_failed: "Google sign-in didn't complete. Try again.",
  no_verified_email: "That account has no verified email address on it.",
};

export function SignIn() {
  const [config, setConfig] = useState<AuthConfig | null>(null);
  const [showGuest, setShowGuest] = useState(false);
  const [starting, setStarting] = useState(false);
  const [guestError, setGuestError] = useState("");
  const [asking, setAsking] = useState(false);
  const [privacy, setPrivacy] = useState(false);
  const [askEmail, setAskEmail] = useState("");
  const [askNote, setAskNote] = useState("");
  const [askState, setAskState] = useState<"" | "sending" | "done">("");
  const [askReply, setAskReply] = useState<{
    status: string;
    message: string;
    queue?: number | null;
  }>({ status: "", message: "" });
  const [askError, setAskError] = useState("");
  const params = new URLSearchParams(window.location.search);
  const error = params.get("error");

  useEffect(() => {
    api
      .authConfig()
      .then(setConfig)
      .catch(() =>
        setConfig({
          google: false,
          dev_login: false,
          guest: false,
          seats_used: 0,
          seats_total: 0,
          seats_free: 0,
        }),
      );
  }, []);

  const requestAccess = async () => {
    setAskError("");
    setAskState("sending");
    try {
      const reply = await api.requestAccess({
        email: askEmail.trim(),
        note: askNote.trim() || undefined,
      });
      setAskReply({
        status: reply.status,
        message: reply.message,
        queue: reply.queue_position,
      });
      setAskState("done");
    } catch (err) {
      setAskError(err instanceof ApiError ? err.message : String(err));
      setAskState("");
    }
  };

  const continueAsGuest = async () => {
    setGuestError("");
    setStarting(true);
    try {
      await api.guestLogin();
      window.location.assign("/");
    } catch (err) {
      setGuestError(err instanceof ApiError ? err.message : String(err));
      setStarting(false);
      setShowGuest(false);
    }
  };

  return (
    <div className="signin">
      <div className="signin-inner">
        <h1>
          Fantasy
          <em>Nerdball</em>
        </h1>
        <p>Fantasy Premier League, reduced to numbers.</p>

        <div className="notice beta-notice">
          <strong>Beta</strong>
          <span>
            The app is still in beta. Full access is by request — ask the
            page admin to add your Google address, then sign in below.
          </span>
        </div>

        {error && MESSAGES[error] && (
          <div className="notice bad">{MESSAGES[error]}</div>
        )}
        {guestError && <div className="notice bad">{guestError}</div>}

        {config?.google && (
          <a className="google-btn" href="/api/auth/login">
            <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden="true">
              <path
                fill="#4285F4"
                d="M45.1 24.5c0-1.6-.1-3.1-.4-4.5H24v8.5h11.8c-.5 2.7-2 5-4.4 6.5v5.4h7.1c4.2-3.8 6.6-9.5 6.6-15.9z"
              />
              <path
                fill="#34A853"
                d="M24 46c5.9 0 10.9-2 14.5-5.3l-7.1-5.4c-2 1.3-4.5 2.1-7.4 2.1-5.7 0-10.5-3.8-12.2-9H4.5v5.6C8.1 41.2 15.5 46 24 46z"
              />
              <path
                fill="#FBBC05"
                d="M11.8 28.4c-.4-1.3-.7-2.7-.7-4.4s.3-3 .7-4.4v-5.6H4.5C2.9 17.2 2 20.5 2 24s.9 6.8 2.5 10l7.3-5.6z"
              />
              <path
                fill="#EA4335"
                d="M24 10.3c3.2 0 6.1 1.1 8.4 3.3l6.3-6.3C34.9 3.8 29.9 2 24 2 15.5 2 8.1 6.8 4.5 14l7.3 5.6c1.7-5.2 6.5-9.3 12.2-9.3z"
              />
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

        {config && config.seats_total === 0 && null}
        {config && config.seats_total > 0 && (
          <p className="seats">
            <strong>{config.seats_used}</strong> active manager
            {config.seats_used === 1 ? "" : "s"} ·{" "}
            <strong>{config.seats_free}</strong>{" "}
            {config.seats_free === 1 ? "space" : "spaces"} left
          </p>
        )}

        {config?.google && !asking && askState !== "done" && (
          <button
            className="link-button request-link"
            type="button"
            onClick={() => setAsking(true)}
          >
            Request access from the admin
          </button>
        )}

        {askState === "done" && (
          <div
            className={
              askReply.status === "already_approved"
                ? "notice"
                : "notice good"
            }
          >
            {askReply.message}
            {askReply.queue != null && (
              <span className="hint">
                Places come free as managers stop using the site, so the
                queue does move. Check your email for confirmation.
              </span>
            )}
          </div>
        )}

        {asking && askState !== "done" && (
          <div className="request-panel">
            <label htmlFor="ask-email">
              Your Google address
              <span className="hint">
                Sign-in is Google-only for now, so this has to be the
                address on a Google account.
              </span>
            </label>
            <input
              id="ask-email"
              type="email"
              value={askEmail}
              placeholder="you@gmail.com"
              onChange={(event) => setAskEmail(event.target.value)}
            />
            <label htmlFor="ask-note">
              Anything to say? <span className="hint">Optional.</span>
            </label>
            <textarea
              id="ask-note"
              rows={2}
              value={askNote}
              placeholder="Who you are, or who sent you."
              onChange={(event) => setAskNote(event.target.value)}
            />
            {askError && <div className="notice bad">{askError}</div>}
            <div className="request-actions">
              <button
                className="btn small"
                type="button"
                disabled={!askEmail.trim() || askState === "sending"}
                onClick={() => void requestAccess()}
              >
                {askState === "sending" ? "Sending…" : "Send request"}
              </button>
              <button
                className="btn quiet small"
                type="button"
                onClick={() => setAsking(false)}
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {config?.guest && (
          <div className="guest-entry">
            <button
              className="btn quiet guest-btn"
              type="button"
              onClick={() => setShowGuest(true)}
            >
              Continue without signing in
            </button>
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

      <p className="disclaimer">
        Fantasy Nerdball is an independent project. It is not affiliated
        with, endorsed by or connected to the Premier League or Fantasy
        Premier League.{" "}
        <button
          className="link-button"
          type="button"
          onClick={() => setPrivacy(true)}
        >
          Privacy
        </button>
      </p>

      {privacy && <PrivacyNotice onClose={() => setPrivacy(false)} />}

      {showGuest && (
        <GuestFeaturesDialog
          onClose={() => setShowGuest(false)}
          onContinue={() => void continueAsGuest()}
          busy={starting}
        />
      )}
    </div>
  );
}
