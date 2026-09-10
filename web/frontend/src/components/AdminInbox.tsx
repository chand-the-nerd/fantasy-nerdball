import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../lib/api";
import type { Inbox } from "../lib/types";

function when(iso: string): string {
  if (!iso) return "";
  const seconds = Math.max(0, (Date.now() - Date.parse(iso)) / 1000);
  if (seconds < 90) return "just now";
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return new Date(iso).toLocaleDateString();
}

export function AdminInbox({ onCount }: { onCount?: (n: number) => void }) {
  const [data, setData] = useState<Inbox | null>(null);
  const [showDone, setShowDone] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{
    ok: boolean;
    detail: string;
  } | null>(null);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");

  const load = useCallback(
    async (done: boolean) => {
      try {
        const fresh = await api.adminInbox(done);
        setData(fresh);
        onCount?.(fresh.unread);
        setError("");
      } catch (err) {
        setError(err instanceof ApiError ? err.message : String(err));
      }
    },
    [onCount],
  );

  useEffect(() => {
    void load(showDone);
  }, [load, showDone]);

  const act = async (run: () => Promise<unknown>, message: string) => {
    setError("");
    setStatus("");
    try {
      await run();
      setStatus(message);
      await load(showDone);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    }
  };

  const testEmail = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      setTestResult(await api.adminTestEmail());
    } catch (err) {
      setTestResult({
        ok: false,
        detail: err instanceof ApiError ? err.message : String(err),
      });
    } finally {
      setTesting(false);
      void load(showDone);
    }
  };

  if (error) return <div className="notice bad">{error}</div>;
  if (!data) return <p className="muted">Opening the inbox…</p>;

  const seatsLeft = data.seats_total - data.seats_used;

  return (
    <>
      {status && <div className="notice good">{status}</div>}

      <div className="mail-status">
        <div className="mail-line">
          <span className="muted">
            {data.email.mode === "off"
              ? "Email isn't set up — this page is the only place these " +
                "appear."
              : `Emailing ${data.email.to || "nobody"} via ` +
                `${data.email.mode}`}
            {data.email.last_sent && ` · last sent ${when(
              data.email.last_sent,
            )}`}
          </span>
          <button
            className="link-button"
            type="button"
            disabled={testing}
            onClick={() => void testEmail()}
          >
            {testing ? "Sending…" : "Send test email"}
          </button>
        </div>

        {data.email.mode === "off" && (
          <p className="hint">
            Set <code>MAIL_TO</code> and either <code>RESEND_API_KEY</code>{" "}
            or the <code>SMTP_*</code> variables on the Railway service.
          </p>
        )}

        {data.email.last_error && !testResult && (
          <div className="notice bad">
            Last attempt failed. {data.email.last_error}
          </div>
        )}

        {testResult && (
          <div className={testResult.ok ? "notice good" : "notice bad"}>
            {testResult.detail}
          </div>
        )}
      </div>

      <div className="metric-head">
        <span className="muted">
          {data.unread === 0
            ? "Nothing waiting"
            : `${data.unread} waiting`}
          {" · "}
          {seatsLeft} {seatsLeft === 1 ? "seat" : "seats"} free
        </span>
        <button
          className="link-button"
          type="button"
          onClick={() => setShowDone(!showDone)}
        >
          {showDone ? "Hide handled" : "Show handled"}
        </button>
      </div>

      {data.items.length === 0 ? (
        <p className="muted">
          Access requests and feedback land here as they arrive.
        </p>
      ) : (
        <ul className="inbox">
          {data.items.map((item) => (
            <li
              key={item.id}
              className={
                item.status === "new" ? "inbox-item" : "inbox-item done"
              }
            >
              <div className="inbox-head">
                <strong>{item.title}</strong>
                <span className="muted">{when(item.created_at)}</span>
              </div>

              {item.body && <p className="inbox-body">{item.body}</p>}

              <div className="inbox-foot">
                <span className="muted small">
                  {item.kind === "access_request"
                    ? item.name || item.email
                    : item.from_guest
                      ? "From a guest"
                      : `From ${item.name || item.email}`}
                  {item.status !== "new" &&
                    item.handled_by &&
                    ` · handled by ${item.handled_by}`}
                </span>

                {item.status === "new" && (
                  <span className="inbox-actions">
                    {item.kind === "access_request" && (
                      <button
                        className="btn small"
                        type="button"
                        disabled={seatsLeft <= 0}
                        title={
                          seatsLeft <= 0
                            ? "No seats free. Remove a manager first."
                            : undefined
                        }
                        onClick={() =>
                          void act(
                            () => api.adminApprove(item.id),
                            `${item.email} can now sign in.`,
                          )
                        }
                      >
                        Approve
                      </button>
                    )}
                    <button
                      className="link-button"
                      type="button"
                      onClick={() =>
                        void act(
                          () => api.adminInboxDone(item.id),
                          "Marked as handled.",
                        )
                      }
                    >
                      {item.kind === "access_request" ? "Dismiss" : "Done"}
                    </button>
                  </span>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}

      <p className="hint">
        Approving adds the address to the allowlist. They still have to
        sign in with the Google account for it.
      </p>
    </>
  );
}
