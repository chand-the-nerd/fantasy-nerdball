import { useEffect, useMemo, useState } from "react";
import { GuestLock, GuestNote } from "./GuestLock";
import { PlayerPicker, type PoolPlayer } from "./PlayerPicker";
import { TeamSliders } from "./TeamSliders";
import { WeightBar, type Weights } from "./WeightBar";
import { FplTeamPanel } from "./FplTeamPanel";
import { checkConstraints } from "./constraints";
import { api, ApiError } from "../lib/api";
import { GUEST, useGuest } from "../lib/guest";
import { publishSettings } from "../lib/settingsStore";
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
  locked = false,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  title: string;
  hint: string;
  locked?: boolean;
}) {
  return (
    <label className={`toggle${locked ? " is-locked" : ""}`}>
      <input
        type="checkbox"
        checked={checked}
        disabled={locked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="copy">
        <strong>{title}</strong>
        <span>{hint}</span>
      </span>
    </label>
  );
}

const STRATEGY_MAX = 2.5;

/**
 * Where the transfer threshold sits on the Kneejerk-to-Conservative scale.
 *
 * The old control was a free number and its hint suggested five, so a stored
 * value can sit above the slider's top. Clamped for display rather than
 * quietly rewritten: the setting is still whatever it was until the slider
 * is moved.
 */
function strategyLabel(value: number): string {
  if (value <= 0.05) return "Kneejerk";
  if (value >= STRATEGY_MAX - 0.05) return "Conservative";
  return `${value.toFixed(1)} pts a gameweek`;
}

export function SetupView({
  me,
  onMeChange,
}: {
  me: Me;
  onMeChange: (me: Me) => void;
}) {
  const guest = useGuest();
  const [settings, setSettings] = useState<Settings | null>(null);
  const [reference, setReference] = useState<Reference | null>(null);
  const [pool, setPool] = useState<PoolPlayer[]>([]);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .settings()
      .then(setSettings)
      .catch((err) =>
        setError(err instanceof ApiError ? err.message : String(err)),
      );
    api.reference().then(setReference).catch(() => undefined);
    api.players().then((data) => setPool(data.players)).catch(() => undefined);

  }, []);

  const reloadSettings = () => {
    api
      .settings()
      .then((fresh) => {
        setSettings(fresh);
        publishSettings(fresh);
      })
      .catch(() => undefined);
  };

  const patch = (changes: Partial<Settings>) =>
    setSettings((current) => (current ? { ...current, ...changes } : current));

  const weightsFor = (position: string): Weights => {
    // A guest's weights are the same 40/30/30 everywhere, and the server
    // puts them back on every save, so there is nothing stored to read.
    if (guest) return { ...GUEST.weights };

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
      (settings.overrides?.POSITION_SCORING_WEIGHTS as Record<
        string,
        Weights
      >) ?? {};
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
      const saved = await api.saveSettings(settings);
      setSettings(saved);
      publishSettings(saved);
      setStatus("Settings saved. They apply on your next run.");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    }
  };

  if (!settings) return <p className="muted">{error || "Loading settings…"}</p>;

  // Guests see the locked values rather than whatever happens to be in
  // the row, so the panel and the run always agree.
  const horizon = guest ? GUEST.gameweeks : settings.first_n_gameweeks;
  const strategy = guest
    ? GUEST.transferStrategy
    : Math.min(settings.min_transfer_value, STRATEGY_MAX);
  const benchPercent = Math.round(
    (guest ? GUEST.benchWeight : settings.bench_weight ?? 0.2) * 100,
  );

  const forced = settings.forced_selections ?? {};
  const setForced = (position: string, names: string[]) =>
    patch({ forced_selections: { ...forced, [position]: names } });

  // Two forced picks in total for a guest, not two per position, so each
  // picker's limit is whatever the other three have left.
  const forcedTotal = POSITIONS.reduce(
    (total, position) => total + (forced[position] ?? []).length,
    0,
  );
  const blacklist = settings.blacklist_players ?? [];

  const forcedLimit = (position: string) => {
    const chosen = (forced[position] ?? []).length;
    const squadLimit = reference?.squad_limits?.[position] ?? 5;
    if (!guest) return squadLimit;
    return Math.min(
      squadLimit,
      chosen + Math.max(0, GUEST.maxForced - forcedTotal),
    );
  };

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

      {guest && (
        <GuestNote>
          Most of the tuning below is for signed-in users. As a guest the
          model runs on fixed settings: {GUEST.gameweeks} gameweeks ahead,
          weighted 40/30/30 on form, history and fixtures.
        </GuestNote>
      )}

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
        title="Model tuning"
        blurb="How the optimiser decides. Set these once and leave them unless something isn't working."
      >
        <div className="panel col-half">
          <h3>Model</h3>
          <div className={`field${guest ? " is-locked" : ""}`}>
            <label htmlFor="horizon">
              Fixtures ahead
              <output htmlFor="horizon">
                {horizon} {horizon === 1 ? "gameweek" : "gameweeks"}
              </output>
            </label>
            <div className="setting-slider">
              <input
                id="horizon"
                type="range"
                min={1}
                max={10}
                step={1}
                value={horizon}
                disabled={guest}
                onChange={(e) =>
                  patch({ first_n_gameweeks: Number(e.target.value) })
                }
              />
              <div className="slider-ends">
                <span>This week only</span>
                <span>Ten weeks out</span>
              </div>
            </div>
            <span className="hint">
              {guest
                ? `Fixed at ${GUEST.gameweeks} gameweeks without an account.`
                : "How far ahead the model looks."}
            </span>
          </div>

          <div className={`field${guest ? " is-locked" : ""}`}>
            <label htmlFor="minval">
              Transfer strategy
              <output htmlFor="minval">{strategyLabel(strategy)}</output>
            </label>
            <div className="setting-slider">
              <input
                id="minval"
                type="range"
                min={0}
                max={2.5}
                step={0.1}
                value={strategy}
                disabled={guest}
                onChange={(e) =>
                  patch({ min_transfer_value: Number(e.target.value) })
                }
              />
              <div className="slider-ends">
                <span>Kneejerk</span>
                <span>Conservative</span>
              </div>
            </div>
            <span className="hint">
              {guest
                ? `Fixed at ${GUEST.transferStrategy.toFixed(1)} points ` +
                  "without an account."
                : "How many points-gain a single transfer must deliver to " +
                  "the squad to be considered worth making."}
            </span>
          </div>

          <div className={`field${guest ? " is-locked" : ""}`}>
            <label htmlFor="bench">
              Bench importance
              <output htmlFor="bench">{benchPercent}%</output>
            </label>
            <div className="setting-slider">
              <input
                id="bench"
                type="range"
                min={0}
                max={100}
                step={5}
                value={benchPercent}
                disabled={guest}
                onChange={(e) =>
                  patch({ bench_weight: Number(e.target.value) / 100 })
                }
              />
              <div className="slider-ends">
                <span>Ignore the bench</span>
                <span>Equal to starters</span>
              </div>
            </div>
            <span className="hint">
              {guest
                ? `Fixed at ${Math.round(GUEST.benchWeight * 100)}% ` +
                  "without an account."
                : "How important your bench is."}
            </span>
          </div>
          <Toggle
            checked={guest ? true : settings.accept_transfer_penalty}
            onChange={(v) => patch({ accept_transfer_penalty: v })}
            locked={guest}
            title="Consider taking a hit"
            hint="Lets the model spend 4 points on an extra transfer when the gain covers it."
          />
          <Toggle
            checked={guest ? true : settings.exclude_unavailable}
            onChange={(v) => patch({ exclude_unavailable: v })}
            locked={guest}
            title="Skip injured and suspended players"
            hint="Turn off to see what the model would do if everyone were fit."
          />
        </div>

        <div className="panel col-half">
          <h3>Model weighting</h3>
          <p className="muted" style={{ marginTop: -6 }}>
            How to prioritise form, points from previous seasons, and upcoming fixture difficulty.
          </p>
          {guest && (
            <p className="hint" style={{ marginTop: 8 }}>
              Locked at 40/30/30 for every position without an account.
            </p>
          )}
          {!guest && settings.use_ml_weights && (
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
          <div
            className={
              !guest && settings.use_ml_weights ? "weights is-inactive" : "weights"
            }
          >
            {POSITIONS.map((position) => (
              <WeightBar
                key={position}
                label={POSITION_LABELS[position]}
                weights={weightsFor(position)}
                locked={guest}
                onChange={(weights) => setWeights(position, weights)}
              />
            ))}
          </div>
        </div>

        <div className="panel col-half">
          <h3>Forced picks</h3>
          <p className="muted" style={{ marginTop: -6 }}>
            Any players you must have? Add them here. I'll build the squad around them.
          </p>
          {guest && (
            <GuestNote>
              Guests can force {GUEST.maxForced} players ({forcedTotal} of{" "}
              {GUEST.maxForced} used).
            </GuestNote>
          )}
          {POSITIONS.map((position) => {
            const limit = forcedLimit(position);
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
            Players to omit from your squad, no matter how much the model likes them.
          </p>
          {guest && (
            <GuestNote>
              Guests can avoid {GUEST.maxBlacklist} players ({blacklist.length}{" "}
              of {GUEST.maxBlacklist} used).
            </GuestNote>
          )}
          <div className="field">
            <PlayerPicker
              inputId="blacklist"
              pool={pool}
              selected={blacklist}
              limit={guest ? GUEST.maxBlacklist : undefined}
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
          {guest && (
            <GuestNote>
              Club adjustments are for signed-in users. Every club runs
              neutral at 1.00 as a guest.
            </GuestNote>
          )}
          {reference && reference.teams.length > 0 ? (
            <TeamSliders
              teams={reference.teams}
              modifiers={settings.team_modifiers ?? {}}
              locked={guest}
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

      <Section title="User" blurb="Link your FPL side (recommended).">
        {guest ? (
          <GuestLock note="Links your real team so its squad, value and free transfers come across on their own.">
            <div className="panel col-half">
              <h3>Your FPL team</h3>
              <p className="muted" style={{ marginTop: -6 }}>
                Linking your side lets the app import your real squad and
                track your points week to week.
              </p>
              <div className="field">
                <label htmlFor="entry-locked">Team id</label>
                <input id="entry-locked" type="text" value="" readOnly />
              </div>
            </div>
          </GuestLock>
        ) : (
          <FplTeamPanel
            me={me}
            onMeChange={onMeChange}
            onSettingsChanged={reloadSettings}
          />
        )}
      </Section>
    </>
  );
}
