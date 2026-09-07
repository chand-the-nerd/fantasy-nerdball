import { useEffect, useMemo, useState } from "react";
import { PlayerPicker, type PoolPlayer } from "./PlayerPicker";
import { TeamSliders } from "./TeamSliders";
import { WeightBar, type Weights } from "./WeightBar";
import { FplTeamPanel } from "./FplTeamPanel";
import { checkConstraints } from "./constraints";
import { api, ApiError } from "../lib/api";
import type { Me, Reference, Settings } from "../lib/types";

const POSITIONS = ["GK", "DEF", "MID", "FWD"];
const POSITION_LABELS: Record<string, string> = {
  GK: "Goalkeepers",
  DEF: "Defenders",
  MID: "Midfielders",
  FWD: "Forwards",
};

function Section({
  title,
  blurb,
  children,
}: {
  title: string;
  blurb: string;
  children: React.ReactNode;
}) {
  return (
    <section className="setup-section">
      <div className="section-head">
        <h2>{title}</h2>
        <p className="muted">{blurb}</p>
      </div>
      <div className="setup-grid">{children}</div>
    </section>
  );
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
  const [reference, setReference] = useState<Reference | null>(null);
  const [pool, setPool] = useState<PoolPlayer[]>([]);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .settings()
      .then(setSettings)
      .catch((err) => setError(err instanceof ApiError ? err.message : String(err)));
    api.reference().then(setReference).catch(() => undefined);
    api.players().then((data) => setPool(data.players)).catch(() => undefined);

  }, []);

  const reloadSettings = () => {
    api.settings().then(setSettings).catch(() => undefined);
  };

  const patch = (changes: Partial<Settings>) =>
    setSettings((current) => (current ? { ...current, ...changes } : current));

  const weightsFor = (position: string): Weights => {
    const stored = (settings?.overrides?.POSITION_SCORING_WEIGHTS as
      | Record<string, Weights>
      | undefined)?.[position];
    if (stored) return stored;

    const fallback = reference?.default_weights?.[position];
    if (fallback && "form" in fallback) {
      return {
        form: fallback.form ?? 0.5,
        historic: fallback.historic ?? 0.25,
        difficulty: fallback.difficulty ?? 0.25,
      };
    }
    return { form: 0.5, historic: 0.25, difficulty: 0.25 };
  };

  const setWeights = (position: string, weights: Weights) => {
    if (!settings) return;
    const current =
      (settings.overrides?.POSITION_SCORING_WEIGHTS as Record<string, Weights>) ?? {};
    patch({
      overrides: {
        ...settings.overrides,
        POSITION_SCORING_WEIGHTS: { ...current, [position]: weights },
      },
    });
  };

  const problems = useMemo(() => {
    if (!settings || !reference) return [];
    return checkConstraints({
      forced: settings.forced_selections ?? {},
      blacklist: settings.blacklist_players ?? [],
      pool,
      budget: settings.budget,
      excludeUnavailable: settings.exclude_unavailable,
      limits: reference.squad_limits,
    });
  }, [settings, reference, pool]);

  const blocking = problems.some((p) => p.level === "error");

  const save = async () => {
    if (!settings) return;
    setError("");
    setStatus("");
    try {
      setSettings(await api.saveSettings(settings));
      setStatus("Settings saved. They apply on your next run.");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    }
  };

  if (!settings) return <p className="muted">{error || "Loading settings…"}</p>;

  const forced = settings.forced_selections ?? {};
  const setForced = (position: string, names: string[]) =>
    patch({ forced_selections: { ...forced, [position]: names } });

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

      {problems.length > 0 && (
        <div className={`notice ${blocking ? "bad" : ""}`}>
          <strong>
            {blocking
              ? "These picks can't produce a valid squad"
              : "Worth knowing before you run"}
          </strong>
          <ul className="problem-list">
            {problems.map((problem, index) => (
              <li key={index} className={problem.level}>
                {problem.text}
              </li>
            ))}
          </ul>
        </div>
      )}

      <Section
        title="Gameweek"
        blurb="What's true of your squad right now. Worth checking every week."
      >
        <div className="panel col-half">
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
              <span className="hint">Squad value plus whatever's in the bank.</span>
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

        <div className="panel col-half">
          <h3>Chips</h3>
          <p className="muted" style={{ marginTop: -6 }}>
            One at a time, and turn it off again after the deadline.
          </p>
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
      </Section>

      <Section
        title="Model tuning"
        blurb="How the optimiser decides. Set these once and leave them unless something isn't working."
      >
        <div className="panel col-half">
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
              <span className="hint">
                How far fixture difficulty looks. Raise it for wildcard planning.
              </span>
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
              <span className="hint">
                Improvement a transfer must clear before it's worth making.
              </span>
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
        </div>

        <div className="panel col-half">
          <h3>Model weighting</h3>
          <p className="muted" style={{ marginTop: -6 }}>
            Drag the handles to divide each position's score between recent form,
            historic points per game and upcoming fixture difficulty. The three
            always total 100%.
          </p>
          {settings.use_ml_weights && (
            <div className="notice" style={{ margin: "14px 0 0" }}>
              Trained weights are switched on, so these are ignored.{" "}
              <button
                className="link-button"
                type="button"
                onClick={() => patch({ use_ml_weights: false })}
              >
                Turn them off
              </button>
            </div>
          )}
          <div className={settings.use_ml_weights ? "weights is-inactive" : "weights"}>
            {POSITIONS.map((position) => (
              <WeightBar
                key={position}
                label={POSITION_LABELS[position]}
                weights={weightsFor(position)}
                onChange={(weights) => setWeights(position, weights)}
              />
            ))}
          </div>
        </div>

        <div className="panel col-half">
          <h3>Forced picks</h3>
          <p className="muted" style={{ marginTop: -6 }}>
            Players the squad is always built around.
          </p>
          {POSITIONS.map((position) => {
            const limit = reference?.squad_limits?.[position] ?? 5;
            const chosen = forced[position] ?? [];
            return (
              <div className="field" key={position}>
                <label htmlFor={`forced-${position}`}>
                  {POSITION_LABELS[position]}{" "}
                  <span className={chosen.length > limit ? "count-over" : "count"}>
                    {chosen.length}/{limit}
                  </span>
                </label>
                <PlayerPicker
                  inputId={`forced-${position}`}
                  pool={pool}
                  position={position}
                  selected={chosen}
                  limit={limit}
                  onChange={(names) => setForced(position, names)}
                />
              </div>
            );
          })}
        </div>

        <div className="panel col-half">
          <h3>Players to avoid</h3>
          <p className="muted" style={{ marginTop: -6 }}>
            Removed from the pool entirely, whatever the numbers say.
          </p>
          <div className="field">
            <PlayerPicker
              inputId="blacklist"
              pool={pool}
              selected={settings.blacklist_players ?? []}
              onChange={(names) => patch({ blacklist_players: names })}
              placeholder="Search any position"
            />
          </div>
        </div>

        <div className="panel">
          <h3>Team adjustments</h3>
          <p className="muted" style={{ marginTop: -6 }}>
            Below 1.00 marks a club down, above marks it up. For what the numbers
            can't know yet — a new manager, a European run, a defence about to
            regress.
          </p>
          {reference && reference.teams.length > 0 ? (
            <TeamSliders
              teams={reference.teams}
              modifiers={settings.team_modifiers ?? {}}
              onChange={(modifiers) => patch({ team_modifiers: modifiers })}
            />
          ) : (
            <p className="muted">
              Club names come from the FPL API, which isn't responding. Reload to
              try again.
            </p>
          )}
        </div>
      </Section>

      <Section title="Users" blurb="Your FPL side. Access is managed from the admin page.">
        <FplTeamPanel
          me={me}
          onMeChange={onMeChange}
          onSettingsChanged={reloadSettings}
        />
      </Section>
    </>
  );
}
