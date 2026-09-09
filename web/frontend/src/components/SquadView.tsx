import { useEffect, useMemo, useState } from "react";
import { Pitch } from "./Pitch";
import { PlayerActions } from "./PlayerActions";
import { RunConsole } from "./RunConsole";
import { RunSettings } from "./RunSettings";
import { SquadBuilder } from "./SquadBuilder";
import { SquadOptions } from "./SquadOptions";
import { StartingSquadPrompt } from "./StartingSquadPrompt";
import { ExploredTransfers, SquadCalculations } from "./SquadCalculations";
import { api, ApiError } from "../lib/api";
import { useSettings } from "../lib/settingsStore";
import { normalise } from "../lib/text";
import type {
  GameweekInfo,
  Me,
  Player,
  Run,
  Squad,
  SquadOption,
} from "../lib/types";

/**
 * The squad to open on: the gameweek FPL is currently on, or the closest one
 * behind it. Runs can be made for future gameweeks, and the newest squad in
 * the list is often one of those — which isn't the side you're picking now.
 */
function forGameweek(squads: Squad[], gameweek: number | null): Squad | null {
  if (squads.length === 0) return null;
  if (gameweek === null) return squads[0];
  const exact = squads.find((squad) => squad.gameweek === gameweek);
  if (exact) return exact;
  // squads arrive newest first, so the first one at or below is the closest.
  return squads.find((squad) => squad.gameweek < gameweek) ?? squads[0];
}

function deadlineText(iso: string | null): string {
  if (!iso) return "";
  const date = new Date(iso);
  return date.toLocaleString("en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * The numbers on screen. Either the squad as saved, or an option being looked
 * at before it is committed — the two are interchangeable here on purpose, so
 * previewing costs nothing more than swapping this object.
 */
interface SquadShape {
  starting: Player[];
  bench: Player[];
  formation: string;
  projected_points: number;
  squad_value: number;
  bank: number;
  transfers_made: number;
  penalty_points: number;
  made_transfers: boolean;
  transfers: { in: string[]; out: string[] };
}

function Scoreline({ gameweek, chip, view }: { gameweek: number; chip: string; view: SquadShape }) {
  return (
    <div className="scoreline">
      <div className="cell">
        <span className="value">GW{gameweek}</span>
      </div>
      <div className="cell">
        <span className="value accent">{view.projected_points.toFixed(1)}</span>
        <span className="label">projected</span>
      </div>
      <div className="cell">
        <span className="value">{view.formation}</span>
        <span className="label">shape</span>
      </div>
      <div className="cell">
        <span className="value">£{view.squad_value.toFixed(1)}m</span>
        <span className="label">
          {view.bank >= 0.05 ? `£${view.bank.toFixed(1)}m in the bank` : "fully spent"}
        </span>
      </div>
      <div className="cell">
        <span className="value">{view.transfers_made}</span>
        <span className="label">
          {view.penalty_points > 0
            ? `transfers, −${view.penalty_points} pts`
            : "transfers"}
        </span>
      </div>
      {chip && <span className="chip-badge">{chip}</span>}
    </div>
  );
}

/* Direction reads faster as an arrow than as a word, and the word was wider
   than the column it sat in. Colour carries the same meaning, so the label
   stays on for screen readers. */
function TransferArrow({ direction }: { direction: "in" | "out" }) {
  const up = direction === "in";
  return (
    <svg
      className={`dir ${direction}`}
      viewBox="0 0 16 16"
      role="img"
      aria-label={up ? "In" : "Out"}
    >
      <path d="M8 2.6v10.8" />
      <path d={up ? "M3.6 7 8 2.6 12.4 7" : "M3.6 9 8 13.4 12.4 9"} />
    </svg>
  );
}

function TransferPanel({
  view,
  reason,
  gain,
}: {
  view: SquadShape;
  reason: string;
  gain: number | null;
}) {
  const { in: incoming, out: outgoing } = view.transfers;

  if (!view.made_transfers || (incoming.length === 0 && outgoing.length === 0)) {
    return (
      <div className="panel">
        <h3>Hold the squad</h3>
        <p className="muted">
          {reason || "No move clears the improvement threshold this week."}
        </p>
      </div>
    );
  }

  return (
    <div className="panel">
      <h3>Make {outgoing.length === 1 ? "this transfer" : "these transfers"}</h3>
      <ul className="transfer-list">
        {outgoing.map((name, i) => (
          <li key={`out-${i}`}>
            <TransferArrow direction="out" />
            <span>{name}</span>
          </li>
        ))}
        {incoming.map((name, i) => (
          <li key={`in-${i}`}>
            <TransferArrow direction="in" />
            <span>{name}</span>
          </li>
        ))}
      </ul>
      {gain != null && gain > 0 ? (
        <p className="transfer-gain">
          Worth about <strong>{gain.toFixed(1)} points a
          gameweek</strong> more than holding the squad
          {view.penalty_points > 0
            ? `, before the ${view.penalty_points} point hit.`
            : "."}
        </p>
      ) : (
        reason && (
          <p className="muted" style={{ marginTop: 12, marginBottom: 0 }}>
            {reason}
          </p>
        )
      )}
    </div>
  );
}

function ModelXiPanel({ squad }: { squad: Squad }) {
  const model = squad.payload.model_xi;
  if (!model) return null;

  const gap = model.projected_points - squad.projected_points;

  return (
    <div className="panel">
      <h3>If you started again</h3>
      <div className="stat-rows">
        <div>
          <span>Best possible XI</span>
          <span>{model.projected_points.toFixed(1)}</span>
        </div>
        <div>
          <span>Your XI</span>
          <span>{squad.projected_points.toFixed(1)}</span>
        </div>
        <div>
          <span>Gap</span>
          <span style={{ color: gap > 3 ? "var(--flag)" : "var(--gain)" }}>
            {gap >= 0 ? "−" : "+"}
            {Math.abs(gap).toFixed(1)}
          </span>
        </div>
      </div>
      <p className="muted" style={{ marginTop: 12, marginBottom: 0 }}>
        Your squad against the model's squad as if it had a Free Hit.
      </p>
    </div>
  );
}

function PlayerDetail({ player, onClose }: { player: Player; onClose: () => void }) {
  return (
    <div className="panel">
      <h3>{player.name}</h3>
      <p className="muted" style={{ marginTop: -6 }}>
        {player.position} · {player.team} · £{player.price.toFixed(1)}m
      </p>
      <div className="stat-rows">
        <div>
          <span>Projected this week</span>
          <span>{player.projected_points.toFixed(1)}</span>
        </div>
        <div>
          <span>Form</span>
          <span>{player.form ?? "—"}</span>
        </div>
        <div>
          <span>Historic points per game</span>
          <span>{player.historic_ppg ?? "—"}</span>
        </div>
        <div>
          <span>Fixture difficulty</span>
          <span>{player.fixture_difficulty ?? "—"}</span>
        </div>
        <div>
          <span>Start rate</span>
          <span>{player.start_rate != null ? `${player.start_rate}%` : "—"}</span>
        </div>
        <div>
          <span>Minutes per gameweek</span>
          <span>{player.minutes_per_game ?? "—"}</span>
        </div>
        <div>
          <span>Expected goals modifier</span>
          <span>{player.xg_modifier != null ? player.xg_modifier.toFixed(2) : "—"}</span>
        </div>
        <div>
          <span>Next</span>
          <span>
            {player.next_opponent || "—"}
            {player.venue ? ` (${player.venue[0]})` : ""}
          </span>
        </div>
      </div>
      {player.news && <div className="notice bad" style={{ marginTop: 14 }}>{player.news}</div>}
      <PlayerActions name={player.name} position={player.position} />
      <button
        className="link-button"
        style={{ marginTop: 14 }}
        onClick={onClose}
        type="button"
      >
        Close
      </button>
    </div>
  );
}

export function SquadView({
  me,
  onMeChange,
}: {
  me: Me;
  onMeChange: (me: Me) => void;
}) {
  const { settings } = useSettings();
  const [squad, setSquad] = useState<Squad | null>(null);
  const [history, setHistory] = useState<Squad[]>([]);
  const [run, setRun] = useState<Run | null>(null);
  const [info, setInfo] = useState<GameweekInfo | null>(null);
  const [selected, setSelected] = useState<Player | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  // Defaults to the gameweek FPL says is next; overridable for back-testing
  // or for planning ahead of the deadline.
  const [targetGw, setTargetGw] = useState<number | null>(null);
  const [building, setBuilding] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  // The option being looked at. Null means whichever one is in force, so a
  // fresh run or a change of gameweek falls back to the active squad rather
  // than holding a preview of something that no longer exists.
  const [previewKey, setPreviewKey] = useState<string | null>(null);
  const [activating, setActivating] = useState(false);

  const busy = run?.status === "queued" || run?.status === "running";

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [squads, latestRun, gw] = await Promise.all([
          api.squads(),
          api.latestRun().catch(() => null),
          api.gameweek().catch(() => null),
        ]);
        if (cancelled) return;
        setHistory(squads);
        setSquad(forGameweek(squads, gw?.gameweek ?? null));
        setRun(latestRun);
        setInfo(gw);
        if (gw) setTargetGw(gw.gameweek);
      } catch (err) {
        if (!cancelled) setError(err instanceof ApiError ? err.message : String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Poll while a run is in flight, then pull the squad it produced.
  useEffect(() => {
    if (!busy || !run) return;
    const timer = setInterval(async () => {
      try {
        const fresh = await api.run(run.id);
        setRun(fresh);
        if (fresh.status === "complete") {
          const squads = await api.squads();
          setHistory(squads);
          setSquad(squads.find((s) => s.gameweek === fresh.gameweek) ?? squads[0] ?? null);
        }
      } catch {
        /* a dropped poll is not worth surfacing; the next one will land */
      }
    }, 2500);
    return () => clearInterval(timer);
  }, [busy, run?.id]);

  useEffect(() => {
    setPreviewKey(null);
  }, [squad?.id, squad?.payload.active_option]);

  const reloadSquads = async () => {
    const squads = await api.squads();
    setHistory(squads);
    setSquad(forGameweek(squads, info?.gameweek ?? null));
    setBuilding(false);
  };

  const optimise = async () => {
    setError("");
    try {
      setRun(await api.startRun(targetGw ?? undefined));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    }
  };

  const options: SquadOption[] = squad?.payload.options ?? [];
  const activeKey = squad?.payload.active_option ?? "option-1";
  const shownKey = previewKey ?? activeKey;
  const previewing = shownKey !== activeKey;

  // What to draw. An option carries its own eleven, shape and totals, so a
  // preview is a straight swap; without options — an older squad saved before
  // this existed, or an imported one — the payload is the view.
  const view: SquadShape | null = useMemo(() => {
    if (!squad) return null;
    const payload = squad.payload;
    const chosen = options.find((option) => option.key === shownKey);

    if (!chosen) {
      return {
        starting: payload.starting,
        bench: payload.bench,
        formation: squad.formation,
        projected_points: squad.projected_points,
        squad_value: squad.squad_value,
        bank: squad.bank,
        transfers_made: payload.made_transfers ? squad.transfers_made : 0,
        penalty_points: squad.penalty_points,
        made_transfers: payload.made_transfers,
        transfers: payload.transfers,
      };
    }

    return {
      starting: chosen.starting,
      bench: chosen.bench,
      formation: chosen.formation,
      projected_points: chosen.projected_points,
      squad_value: chosen.squad_value,
      bank: chosen.bank,
      transfers_made: chosen.transfers_made,
      penalty_points: chosen.penalty_points,
      made_transfers: chosen.transfers_made > 0,
      transfers: chosen.transfers,
    };
  }, [squad, options, shownKey]);

  const activate = async (key: string) => {
    if (!squad) return;
    setError("");
    setActivating(true);
    try {
      const updated = await api.activateOption(squad.gameweek, key);
      setSquad(updated);
      setHistory((squads) =>
        squads.map((entry) => (entry.id === updated.id ? updated : entry)),
      );
      setPreviewKey(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setActivating(false);
    }
  };

  // Forced picks are stored per position; the pitch only needs the names.
  const forcedNames = useMemo(() => {
    const names = new Set<string>();
    for (const list of Object.values(settings?.forced_selections ?? {})) {
      for (const name of list ?? []) names.add(normalise(name));
    }
    return names;
  }, [settings]);

  // The gameweek before the one being planned for. Null until FPL answers,
  // since guessing it would mean prompting for the wrong week.
  const previousGw = info?.gameweek ? Math.max(0, info.gameweek - 1) || null : null;
  const hasPrevious =
    previousGw !== null && history.some((entry) => entry.gameweek === previousGw);

  const gameweekOptions = useMemo(
    () => history.map((s) => s.gameweek).sort((a, b) => b - a),
    [history],
  );

  if (loading) return <p className="muted">Loading your squad…</p>;

  return (
    <>
      <div className="topbar">
        <div>
          <h1>{squad ? `Gameweek ${squad.gameweek}` : "No squad yet"}</h1>
          {info?.deadline && (
            <span className="when">Deadline {deadlineText(info.deadline)}</span>
          )}
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          {gameweekOptions.length > 1 && (
            <select
              className="btn quiet small"
              value={squad?.gameweek ?? ""}
              onChange={(e) =>
                setSquad(history.find((s) => s.gameweek === Number(e.target.value)) ?? null)
              }
            >
              {gameweekOptions.map((gw) => (
                <option key={gw} value={gw}>
                  Gameweek {gw}
                </option>
              ))}
            </select>
          )}
          <label className="gw-picker">
            <span>Optimise</span>
            <select
              value={targetGw ?? ""}
              onChange={(e) => setTargetGw(Number(e.target.value))}
              disabled={busy}
              aria-label="Gameweek to optimise"
            >
              {Array.from({ length: 38 }, (_, i) => i + 1).map((gw) => (
                <option key={gw} value={gw}>
                  GW{gw}
                  {info?.gameweek === gw ? " (next)" : ""}
                </option>
              ))}
            </select>
          </label>
          <button className="btn" onClick={optimise} disabled={busy} type="button">
            {busy ? "Optimising…" : "Run"}
          </button>
        </div>
      </div>

      {error && <div className="notice bad">{error}</div>}

      {/* The gameweek just gone is what a run transfers from, so a gap there
          is worth resolving before anything else on the page. */}
      {previousGw !== null && !hasPrevious && !building && !dismissed && (
        <StartingSquadPrompt
          gameweek={previousGw}
          me={me}
          onMeChange={onMeChange}
          onImported={() => void reloadSquads()}
          onBuild={() => setBuilding(true)}
          onDismiss={() => setDismissed(true)}
        />
      )}

      {building && previousGw !== null ? (
        <SquadBuilder
          gameweek={previousGw}
          onSaved={() => void reloadSquads()}
          onCancel={() => setBuilding(false)}
        />
      ) : (
        <>
      <RunSettings disabled={busy} />

      <div className="squad-layout">
        <div>
          {squad && view ? (
            <>
              <Scoreline
                gameweek={squad.gameweek}
                chip={squad.chip}
                view={view}
              />
              <Pitch
                starting={view.starting}
                bench={view.bench}
                benchBoost={squad.chip === "Bench Boost"}
                onSelect={setSelected}
                forced={forcedNames}
              />
              <SquadOptions
                options={options}
                activeKey={activeKey}
                previewKey={shownKey}
                onPreview={setPreviewKey}
                onActivate={activate}
                activating={activating}
                disabled={busy}
              />
            </>
          ) : (
            <div className="panel">
              <h3>Build your first squad</h3>
              <p className="muted">
                Check your budget and chips under Setup, then run the optimiser. It
                pulls live form, fixtures and expected goals, so the first run takes a
                few minutes.
              </p>
            </div>
          )}
        </div>

        <div className="side-stack">
          {(busy || run?.status === "failed") && run && <RunConsole run={run} />}
          {selected && (
            <PlayerDetail player={selected} onClose={() => setSelected(null)} />
          )}
          {squad && view && !selected && (
            <TransferPanel
              view={view}
              // The optimiser's reasoning belongs to the squad it recommended.
              // Shown against one you picked instead, it would be arguing for
              // transfers that aren't on screen.
              reason={
                previewing
                  ? "Previewing an option. Activate it to make it your squad."
                  : squad.payload.transfer_reason
              }
              gain={previewing ? null : squad.payload.points_gain_per_gw}
            />
          )}
          {squad && !selected && <ExploredTransfers squad={squad} />}
          {squad && !selected && <ModelXiPanel squad={squad} />}
        </div>
      </div>

      {squad && <SquadCalculations squad={squad} />}
        </>
      )}
    </>
  );
}
