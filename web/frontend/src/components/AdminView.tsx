import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../lib/api";

interface Member {
  id: number;
  email: string;
  name: string;
  avatar_url: string;
  is_admin: boolean;
  is_you: boolean;
  last_seen_at: string;
}

interface AllowEntry {
  email: string;
  registered: boolean;
}

interface Invite extends AllowEntry {
  id: number;
  invited_by: string;
}

interface Members {
  seats_used: number;
  seats_total: number;
  members: Member[];
  env_allowlist: AllowEntry[];
  invites: Invite[];
}

export function AdminView({ onClose }: { onClose: () => void }) {
  const [configured, setConfigured] = useState<boolean | null>(null);
  const [unlocked, setUnlocked] = useState(false);
  const [password, setPassword] = useState("");
  const [data, setData] = useState<Members | null>(null);
  const [cron, setCron] = useState<any>(null);
  const [newEmail, setNewEmail] = useState("");
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      setData(await api.adminMembers());
      setUnlocked(true);
      api.cronStatus().then(setCron).catch(() => undefined);
    } catch (err) {
      if (err instanceof ApiError && err.status === 403) setUnlocked(false);
      else setError(err instanceof ApiError ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    api
      .adminStatus()
      .then((s) => {
        setConfigured(s.configured);
        if (s.unlocked) void load();
      })
      .catch(() => setConfigured(false));
  }, [load]);

  const unlock = async () => {
    setError("");
    setBusy(true);
    try {
      await api.adminUnlock(password);
      setPassword("");
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const lock = async () => {
    await api.adminLock().catch(() => undefined);
    setUnlocked(false);
    setData(null);
    onClose();
  };

  const act = async (run: () => Promise<unknown>, message: string) => {
    setError("");
    setStatus("");
    try {
      await run();
      setStatus(message);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    }
  };

  if (configured === false) {
    return (
      <div className="admin-panel">
        <h2>Admin</h2>
        <div className="notice bad">
          The admin area is switched off. Set <code>ADMIN_PASSWORD</code> on the
          Railway service and redeploy to enable it.
        </div>
        <button className="btn quiet small" onClick={onClose} type="button">
          Close
        </button>
      </div>
    );
  }

  if (configured === null) return <div className="admin-panel" />;

  if (!unlocked) {
    return (
      <div className="admin-panel">
        <h2>Admin</h2>
        <p className="muted">Enter the admin password to manage access.</p>
        {error && <div className="notice bad">{error}</div>}
        <div className="field">
          <label htmlFor="admin-password">Password</label>
          <input
            id="admin-password"
            type="password"
            autoFocus
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && unlock()}
          />
        </div>
        <div className="admin-actions">
          <button className="btn" onClick={unlock} disabled={busy || !password} type="button">
            {busy ? "Checking…" : "Unlock"}
          </button>
          <button className="btn quiet" onClick={onClose} type="button">
            Cancel
          </button>
        </div>
      </div>
    );
  }

  if (!data) return <div className="admin-panel" />;

  const seatsLeft = data.seats_total - data.seats_used;

  return (
    <div className="admin-panel wide">
      <div className="admin-head">
        <h2>Admin</h2>
        <div className="admin-actions">
          <button className="link-button" onClick={lock} type="button">
            Lock and close
          </button>
        </div>
      </div>

      {error && <div className="notice bad">{error}</div>}
      {status && <div className="notice good">{status}</div>}

      <section className="admin-section">
        <h3>
          Managers <span className="muted">{data.seats_used} of {data.seats_total} seats</span>
        </h3>
        <table>
          <thead>
            <tr>
              <th>Manager</th>
              <th>Email</th>
              <th className="num" />
            </tr>
          </thead>
          <tbody>
            {data.members.map((member) => (
              <tr key={member.id}>
                <td>
                  <span className="who-cell">
                    {member.avatar_url && <img src={member.avatar_url} alt="" />}
                    {member.name || "—"}
                    {member.is_you && <span className="muted">you</span>}
                  </span>
                </td>
                <td className="muted">{member.email}</td>
                <td className="num">
                  {member.is_you ? (
                    <span className="muted">—</span>
                  ) : (
                    <button
                      className="link-button danger"
                      type="button"
                      onClick={() => {
                        if (
                          window.confirm(
                            `Remove ${member.name || member.email}? Their squads, ` +
                              "settings and results are deleted with them.",
                          )
                        ) {
                          void act(
                            () => api.adminRemoveMember(member.id),
                            `${member.email} removed.`,
                          );
                        }
                      }}
                    >
                      Remove
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {cron && (
        <section className="admin-section">
          <h3>Player history</h3>
          <p className="muted">
            The model penalises spiky, blank-prone form using stored per-gameweek
            data. Without it that signal sits at neutral for everyone.
          </p>
          <div className="stat-rows">
            <div>
              <span>Last stored gameweek</span>
              <span>{cron.last_gameweek || "none yet"}</span>
            </div>
            <div>
              <span>Outstanding</span>
              <span
                style={{
                  color: cron.pending_gameweek ? "var(--floodlight)" : "var(--gain)",
                }}
              >
                {cron.pending_gameweek
                  ? `gameweek ${cron.pending_gameweek}`
                  : "up to date"}
              </span>
            </div>
            <div>
              <span>Built-in timer</span>
              <span>{cron.internal_scheduler ? "on" : "off"}</span>
            </div>
            <div>
              <span>External trigger</span>
              <span>{cron.configured ? "enabled" : "no CRON_SECRET set"}</span>
            </div>
          </div>
          <div className="admin-actions">
            <button
              className="btn quiet small"
              type="button"
              onClick={() =>
                act(async () => {
                  const r = await api.cronRunNow();
                  setCron(await api.cronStatus());
                  return r;
                }, "Player history updated.")
              }
            >
              Update now
            </button>
          </div>
        </section>
      )}

      <section className="admin-section">
        <h3>Who can sign in</h3>
        <p className="muted">
          An address gets in if it's listed here. Everyone also has to be a test
          user on the Google consent screen, or Google blocks them first.
        </p>

        <table>
          <tbody>
            {data.env_allowlist.map((entry) => (
              <tr key={`env-${entry.email}`}>
                <td>{entry.email}</td>
                <td className="muted">
                  {entry.registered ? "signed in" : "not yet signed in"}
                </td>
                <td className="num muted" title="Set by the ALLOWED_EMAILS variable on Railway">
                  from ALLOWED_EMAILS
                </td>
              </tr>
            ))}
            {data.invites.map((invite) => (
              <tr key={`inv-${invite.id}`}>
                <td>{invite.email}</td>
                <td className="muted">
                  {invite.registered ? "signed in" : "not yet signed in"}
                </td>
                <td className="num">
                  <button
                    className="link-button danger"
                    type="button"
                    onClick={() =>
                      act(
                        () => api.adminRemoveInvite(invite.id),
                        `${invite.email} can no longer sign in.`,
                      )
                    }
                  >
                    Remove
                  </button>
                </td>
              </tr>
            ))}
            {data.env_allowlist.length === 0 && data.invites.length === 0 && (
              <tr>
                <td colSpan={3} className="muted">
                  Nobody is allowed in yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>

        <p className="hint" style={{ marginTop: 10 }}>
          Addresses from <code>ALLOWED_EMAILS</code> are set on the Railway
          service, so they can only be changed there. Removing someone's account
          above doesn't revoke access on its own — take their address off this
          list too, or the next sign-in recreates it.
        </p>

        <div className="admin-add">
          <div className="field" style={{ flex: 1, marginBottom: 0 }}>
            <label htmlFor="new-email">Allow another Google account</label>
            <input
              id="new-email"
              type="text"
              value={newEmail}
              placeholder="friend@gmail.com"
              disabled={seatsLeft <= 0}
              onChange={(event) => setNewEmail(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && newEmail.trim()) {
                  void act(
                    () => api.adminAddInvite(newEmail.trim()),
                    `${newEmail.trim()} can now sign in.`,
                  ).then(() => setNewEmail(""));
                }
              }}
            />
          </div>
          <button
            className="btn small"
            type="button"
            disabled={!newEmail.trim() || seatsLeft <= 0}
            onClick={() =>
              act(
                () => api.adminAddInvite(newEmail.trim()),
                `${newEmail.trim()} can now sign in.`,
              ).then(() => setNewEmail(""))
            }
          >
            Add
          </button>
        </div>
        {seatsLeft <= 0 && (
          <p className="hint" style={{ color: "var(--flag)" }}>
            All {data.seats_total} seats are taken. Remove a manager before adding
            anyone else.
          </p>
        )}
      </section>
    </div>
  );
}
