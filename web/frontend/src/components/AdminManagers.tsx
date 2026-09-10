import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../lib/api";
import type { AdminUsers, SavedSquad } from "../lib/types";

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
  const [status, setStatus] = useState("");
  const [openId, setOpenId] = useState<number | null>(null);
  const [squads, setSquads] = useState<SavedSquad[]>([]);
  const [busy, setBusy] = useState(false);

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

  const openSquads = async (id: number) => {
    if (openId === id) {
      setOpenId(null);
      return;
    }
    setOpenId(id);
    setSquads([]);
    try {
      setSquads((await api.adminUserSquads(id)).squads);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    }
  };

  const act = async (run: () => Promise<unknown>, message: string) => {
    setBusy(true);
    setError("");
    setStatus("");
    try {
      await run();
      setStatus(message);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  if (error) return <div className="notice bad">{error}</div>;
  if (!data) return <p className="muted">Counting heads…</p>;

  return (
    <>
      {status && <div className="notice good">{status}</div>}

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
              <th>Status</th>
              <th className="num">FPL ID</th>
              <th className="num">Runs</th>
              <th className="num">Last seen</th>
              <th className="num">Place freed in</th>
            </tr>
          </thead>
          <tbody>
            {data.users.map((row) => (
              <tr
                key={row.id}
                className={row.status === "active" ? "" : "is-dormant"}
              >
                <td>
                  <span className="who-cell">
                    {row.name || row.email}
                    {row.is_admin && <span className="pill">admin</span>}
                  </span>
                  <div className="muted small">{row.email}</div>
                  <div className="manager-actions">
                    <button
                      className="link-button"
                      type="button"
                      onClick={() => void openSquads(row.id)}
                    >
                      {openId === row.id
                        ? "Hide squads"
                        : `${row.squads_kept} saved gameweek${
                            row.squads_kept === 1 ? "" : "s"
                          }`}
                    </button>
                    {row.status !== "active" && (
                      <button
                        className="link-button"
                        type="button"
                        disabled={busy}
                        onClick={() =>
                          void act(
                            () => api.adminReactivate(row.id),
                            `${row.email} has their place back.`,
                          )
                        }
                      >
                        Give their place back
                      </button>
                    )}
                  </div>

                  {openId === row.id && (
                    <div className="squad-list">
                      {squads.length === 0 ? (
                        <span className="muted small">
                          Nothing saved.
                        </span>
                      ) : (
                        squads.map((squad) => (
                          <div
                            key={`${squad.season}-${squad.gameweek}`}
                            className="squad-line"
                          >
                            <span>
                              {squad.season} · GW{squad.gameweek} ·{" "}
                              {squad.projected_points.toFixed(1)} pts
                              {squad.chip && ` · ${squad.chip}`}
                            </span>
                            <button
                              className="link-button"
                              type="button"
                              disabled={busy}
                              onClick={() =>
                                void act(
                                  () =>
                                    api.adminRestoreSquad(row.id, {
                                      season: squad.season,
                                      gameweek: squad.gameweek,
                                    }),
                                  `Restored GW${squad.gameweek}.`,
                                )
                              }
                            >
                              Restore
                            </button>
                          </div>
                        ))
                      )}
                    </div>
                  )}
                </td>
                <td>
                  {row.status === "active" ? (
                    <span className="muted">active</span>
                  ) : row.status === "purged" ? (
                    <span className="pill">FPL id only</span>
                  ) : (
                    <span className="pill warn">dormant</span>
                  )}
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
        Ordered by last activity. A dormant manager has given their place
        up but kept everything else — their squads come back with them.
        Data is cleared once its season is over, leaving only the FPL id,
        and accounts are deleted outright only after{" "}
        {data.purge_after_months} months. Admins and anyone in
        ALLOWED_EMAILS are never touched.
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
