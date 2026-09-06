import { useEffect, useState } from "react";
import {
  ForcedEditor,
  TeamModifierEditor,
  WeightsEditor,
  type Reference,
} from "./SettingsEditors";
import { api, ApiError } from "../lib/api";
import type { Me, Settings } from "../lib/types";

interface Member {
  id: number;
  email: string;
  name: string;
  is_admin: boolean;
}

interface MembersResponse {
  seats_used: number;
  seats_total: number;
  members: Member[];
  invites: { id: number; email: string }[];
}

function Toggle({
  checked,
  onChange,
  title,
  hint,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  title: string;
  hint: string;
}) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="copy">
        <strong>{title}</strong>
        <span>{hint}</span>
      </span>
    </label>
  );
}

export function SetupView({ me, onMeChange }: { me: Me; onMeChange: (me: Me) => void }) {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [entryId, setEntryId] = useState(me.fpl_entry_id ? String(me.fpl_entry_id) : "");
  const [blacklist, setBlacklist] = useState("");
  const [reference, setReference] = useState<Reference | null>(null);
  const [members, setMembers] = useState<MembersResponse | null>(null);
  const [inviteEmail, setInviteEmail] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .settings()
      .then((s) => {
        setSettings(s);
        setBlacklist((s.blacklist_players || []).join(", "));
      })
      .catch((err) => setError(err instanceof ApiError ? err.message : String(err)));

    api.reference().then(setReference).catch(() => undefined);

    if (me.is_admin) {
      fetch("/api/admin/members")
        .then((r) => (r.ok ? r.json() : null))
        .then(setMembers)
        .catch(() => undefined);
    }
  }, [me.is_admin]);

  const patch = (changes: Partial<Settings>) =>
    setSettings((current) => (current ? { ...current, ...changes } : current));

  const save = async () => {
    if (!settings) return;
    setError("");
    setStatus("");
    try {
      const saved = await api.saveSettings({
        ...settings,
        blacklist_players: blacklist
          .split(",")
          .map((name) => name.trim())
          .filter(Boolean),
      });
      setSettings(saved);
      setStatus("Settings saved. They apply on your next run.");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    }
  };

  const linkEntry = async () => {
    setError("");
    setStatus("");
    try {
      const value = entryId.trim() ? Number(entryId.trim()) : null;
      const updated = await api.linkEntry(value);
      onMeChange(updated);
      setStatus(value ? "FPL team linked. Your points will sync on the Form page." : "FPL team unlinked.");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    }
  };

  const invite = async () => {
    setError("");
    try {
      const response = await fetch("/api/admin/invites", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: inviteEmail }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || "Couldn't add that address.");
      }
      setInviteEmail("");
      setStatus(`${inviteEmail} can now sign in with Google.`);
      setMembers(await (await fetch("/api/admin/members")).json());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  if (!settings) return <p className="muted">{error || "Loading settings…"}</p>;

  return (
    <>
      <div className="topbar">
        <div>
          <h1>Setup</h1>
          <span className="when">How the optimiser treats your squad</span>
        </div>
        <button className="btn" onClick={save} type="button">
          Save settings
        </button>
      </div>

      {error && <div className="notice bad">{error}</div>}
      {status && <div className="notice good">{status}</div>}

      <div style={{ display: "grid", gap: 16, gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", alignItems: "start" }}>
        <div className="panel">
          <h3>This gameweek</h3>
          <div className="grid-2">
            <div className="field">
              <label htmlFor="budget">Budget</label>
              <input
                id="budget"
                type="number"
                step="0.1"
                value={settings.budget}
                onChange={(e) => patch({ budget: Number(e.target.value) })}
              />
              <span className="hint">Your squad value plus whatever's in the bank.</span>
            </div>
            <div className="field">
              <label htmlFor="fts">Free transfers</label>
              <input
                id="fts"
                type="number"
                min={0}
                max={15}
                value={settings.free_transfers}
                onChange={(e) => patch({ free_transfers: Number(e.target.value) })}
              />
            </div>
          </div>

          <Toggle
            checked={settings.accept_transfer_penalty}
            onChange={(v) => patch({ accept_transfer_penalty: v })}
            title="Consider taking a hit"
            hint="Lets the model spend 4 points on an extra transfer when the gain covers it."
          />
          <Toggle
            checked={settings.exclude_unavailable}
            onChange={(v) => patch({ exclude_unavailable: v })}
            title="Skip injured and suspended players"
            hint="Turn off to see what the model would do if everyone were fit."
          />
        </div>

        <div className="panel">
          <h3>Chips</h3>
          <p className="muted" style={{ marginTop: -6 }}>One at a time. Turn it off again after the deadline.</p>
          <Toggle
            checked={settings.wildcard}
            onChange={(v) => patch({ wildcard: v, bench_boost: false, triple_captain: false })}
            title="Wildcard"
            hint="Removes the transfer limit and rebuilds the squad from scratch."
          />
          <Toggle
            checked={settings.bench_boost}
            onChange={(v) => patch({ bench_boost: v, wildcard: false, triple_captain: false })}
            title="Bench Boost"
            hint="All fifteen players score, so the bench is optimised properly."
          />
          <Toggle
            checked={settings.triple_captain}
            onChange={(v) => patch({ triple_captain: v, wildcard: false, bench_boost: false })}
            title="Triple Captain"
            hint="Your captain returns three times their points."
          />
          <Toggle
            checked={settings.free_hit_prev_gw}
            onChange={(v) => patch({ free_hit_prev_gw: v })}
            title="Free Hit last week"
            hint="Loads the squad from two gameweeks ago, since the Free Hit side has reverted."
          />
        </div>

        <div className="panel">
          <h3>Model</h3>
          <div className="grid-2">
            <div className="field">
              <label htmlFor="horizon">Fixtures ahead</label>
              <input
                id="horizon"
                type="number"
                min={1}
                max={10}
                value={settings.first_n_gameweeks}
                onChange={(e) => patch({ first_n_gameweeks: Number(e.target.value) })}
              />
              <span className="hint">How far the fixture difficulty looks. Raise it for wildcard planning.</span>
            </div>
            <div className="field">
              <label htmlFor="minval">Transfer threshold</label>
              <input
                id="minval"
                type="number"
                step="0.5"
                value={settings.min_transfer_value}
                onChange={(e) => patch({ min_transfer_value: Number(e.target.value) })}
              />
              <span className="hint">Score improvement a transfer must clear before it's worth making.</span>
            </div>
          </div>
          <div className="field">
            <label htmlFor="hold">Gameweeks a transfer is held</label>
            <input
              id="hold"
              type="number"
              min={1}
              max={15}
              value={settings.transfer_horizon_gws}
              onChange={(e) => patch({ transfer_horizon_gws: Number(e.target.value) })}
            />
            <span className="hint">A hit is a one-off cost spread across this many weeks.</span>
          </div>
          <Toggle
            checked={settings.use_ml_weights}
            onChange={(v) => patch({ use_ml_weights: v })}
            title="Use trained position weights"
            hint="Reads the weights produced by the ML training scripts instead of the manual defaults."
          />
        </div>

        <div className="panel">
          <h3>Players to avoid</h3>
          <div className="field">
            <label htmlFor="blacklist">Never pick these</label>
            <textarea
              id="blacklist"
              rows={3}
              value={blacklist}
              onChange={(e) => setBlacklist(e.target.value)}
              placeholder="Separate names with commas"
            />
            <span className="hint">
              Matched on display name, so a shared surname removes everyone who has it.
            </span>
          </div>
        </div>

        {reference && (
          <WeightsEditor
            reference={reference}
            weights={
              (settings.overrides?.POSITION_SCORING_WEIGHTS as Record<
                string,
                Record<string, number>
              >) ?? {}
            }
            onChange={(weights) =>
              patch({
                overrides: {
                  ...settings.overrides,
                  POSITION_SCORING_WEIGHTS: weights,
                },
              })
            }
          />
        )}

        {reference && (
          <ForcedEditor
            reference={reference}
            forced={settings.forced_selections ?? {}}
            onChange={(forced) => patch({ forced_selections: forced })}
          />
        )}

        {reference && (
          <TeamModifierEditor
            reference={reference}
            modifiers={settings.team_modifiers ?? {}}
            onChange={(modifiers) => patch({ team_modifiers: modifiers })}
          />
        )}

        <div className="panel">
          <h3>Your FPL team</h3>
          <div className="field">
            <label htmlFor="entry">Team id</label>
            <input
              id="entry"
              type="text"
              inputMode="numeric"
              value={entryId}
              onChange={(e) => setEntryId(e.target.value)}
              placeholder="e.g. 1234567"
            />
            <span className="hint">
              The number in your team's URL on the FPL site. Linking it pulls your real
              points in each week so you can see them against the global average.
            </span>
          </div>
          <button className="btn quiet small" onClick={linkEntry} type="button">
            {entryId.trim() ? "Link team" : "Unlink team"}
          </button>
        </div>

        {me.is_admin && members && (
          <div className="panel">
            <h3>Managers</h3>
            <p className="muted" style={{ marginTop: -6 }}>
              {members.seats_used} of {members.seats_total} seats taken.
            </p>
            <div className="stat-rows">
              {members.members.map((member) => (
                <div key={member.id}>
                  <span>{member.name || member.email}</span>
                  <span>{member.is_admin ? "owner" : "manager"}</span>
                </div>
              ))}
            </div>
            <div className="field" style={{ marginTop: 16 }}>
              <label htmlFor="invite">Invite a Google account</label>
              <input
                id="invite"
                type="text"
                value={inviteEmail}
                onChange={(e) => setInviteEmail(e.target.value)}
                placeholder="friend@gmail.com"
              />
            </div>
            <button className="btn quiet small" onClick={invite} type="button">
              Add manager
            </button>
          </div>
        )}
      </div>
    </>
  );
}
