import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../lib/api";
import type { AdminUsers } from "../lib/types";

function ago(iso: string): string {
  if (!iso) return "never";
  const seconds = Math.max(0, (Date.now() - Date.parse(iso)) / 1000);
  if (seconds < 90) return "just now";
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}

export function AdminManagers() {
  const [data, setData] = useState<AdminUsers | null>(null);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      setData(await api.adminUsers());
      setError("");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (error) return <div className="notice bad">{error}</div>;
  if (!data) return <p className="muted">Counting heads…</p>;

  return (
    <>
      <div className="metric-head">
        <span className="muted">
          {data.seats_used} of {data.seats_total} places taken
          {data.inactive_days > 0 &&
            ` · places freed after ${data.inactive_days} days idle`}
        </span>
        <button
          className="link-button"
          type="button"
          onClick={() => void load()}
        >
          Refresh
        </button>
      </div>

      <div className="calc-scroll">
        <table className="calc-table">
          <thead>
            <tr>
              <th>Manager</th>
              <th className="num">FPL ID</th>
              <th className="num">Runs</th>
              <th className="num">Last seen</th>
              <th className="num">Place freed in</th>
            </tr>
          </thead>
          <tbody>
            {data.users.map((row) => (
              <tr key={row.id}>
                <td>
                  <span className="who-cell">
                    {row.name || row.email}
                    {row.is_admin && <span className="pill">admin</span>}
                  </span>
                  <div className="muted small">{row.email}</div>
                </td>
                <td className="num">
                  {row.fpl_entry_id ?? <span className="muted">—</span>}
                </td>
                <td className="num">{row.runs}</td>
                <td className="num muted">{ago(row.last_seen_at)}</td>
                <td
                  className={
                    row.removal_in_days !== null && row.removal_in_days <= 7
                      ? "num warn"
                      : "num muted"
                  }
                >
                  {row.removal_in_days === null
                    ? "never"
                    : `${row.removal_in_days}d`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="hint">
        Ordered by last activity, so anyone about to lose their place is at
        the bottom. Admins and anyone in ALLOWED_EMAILS are never removed.
      </p>

      {data.invites.length > 0 && (
        <>
          <h4>Invitations out</h4>
          <div className="stat-rows">
            {data.invites.map((invite) => (
              <div key={invite.email}>
                <span>{invite.email}</span>
                <span
                  className={
                    !invite.signed_in &&
                    invite.hours_left !== null &&
                    invite.hours_left < 12
                      ? "warn"
                      : ""
                  }
                >
                  {invite.signed_in
                    ? "signed in"
                    : invite.hours_left === null
                      ? "no deadline"
                      : invite.hours_left <= 0
                        ? "expired, sweeping shortly"
                        : `${Math.round(invite.hours_left)}h left`}
                </span>
              </div>
            ))}
          </div>
          <p className="hint">
            An unused invitation lapses after {data.invite_ttl_hours} hours
            and the place goes back to the queue. They're emailed when it
            does.
          </p>
        </>
      )}
    </>
  );
}
