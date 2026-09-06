import { useEffect, useMemo, useState } from "react";
import { Pitch } from "./Pitch";
import { RunConsole } from "./RunConsole";
import { api, ApiError } from "../lib/api";
import type { GameweekInfo, Player, Run, Squad } from "../lib/types";

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

function Scoreline({ squad }: { squad: Squad }) {
  const payload = squad.payload;
  return (
    <div className="scoreline">
      <div className="cell">
        <span className="value">GW{squad.gameweek}</span>
      </div>
      <div className="cell">
        <span className="value accent">{squad.projected_points.toFixed(1)}</span>
        <span className="label">projected</span>
      </div>
      <div className="cell">
        <span className="value">{squad.formation}</span>
        <span className="label">shape</span>
      </div>
      <div className="cell">
        <span className="value">£{squad.squad_value.toFixed(1)}m</span>
        <span className="label">
          {squad.bank >= 0.05 ? `£${squad.bank.toFixed(1)}m in the bank` : "fully spent"}
        </span>
      </div>
      <div className="cell">
        <span className="value">
          {payload.made_transfers ? squad.transfers_made : 0}
        </span>
        <span className="label">
          {squad.penalty_points > 0
            ? `transfers, −${squad.penalty_points} pts`
            : "transfers"}
        </span>
      </div>
      {squad.chip && <span className="chip-badge">{squad.chip}</span>}
    </div>
  );
}

function TransferPanel({ squad }: { squad: Squad }) {
  const payload = squad.payload;
  const { in: incoming, out: outgoing } = payload.transfers;

  if (!payload.made_transfers || (incoming.length === 0 && outgoing.length === 0)) {
    return (
      <div className="panel">
        <h3>Hold the squad</h3>
        <p className="muted">
          {payload.transfer_reason ||
            "No move clears the improvement threshold this week."}
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
            <span className="dir out">OUT</span>
            <span>{name}</span>
          </li>
        ))}
        {incoming.map((name, i) => (
          <li key={`in-${i}`}>
            <span className="dir in">IN</span>
            <span>{name}</span>
          </li>
        ))}
      </ul>
      {payload.points_gain_per_gw != null && payload.points_gain_per_gw > 0 && (
        <p className="muted" style={{ marginTop: 12, marginBottom: 0 }}>
          Worth about {payload.points_gain_per_gw.toFixed(1)} points a gameweek
          {squad.penalty_points > 0
            ? `, after the ${squad.penalty_points} point hit.`
            : "."}
        </p>
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
        What the model would pick with a free hand and the same budget. A small gap
        means your transfer constraints aren't costing you much.
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

export function SquadView() {
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
        setSquad(squads[0] ?? null);
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

  const optimise = async () => {
    setError("");
    try {
      setRun(await api.startRun(targetGw ?? undefined));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    }
  };

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

      <div className="squad-layout">
        <div>
          {squad ? (
            <>
              <Scoreline squad={squad} />
              <Pitch
                starting={squad.payload.starting}
                bench={squad.payload.bench}
                benchBoost={squad.chip === "Bench Boost"}
                onSelect={setSelected}
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
          {squad && !selected && <TransferPanel squad={squad} />}
          {squad && !selected && <ModelXiPanel squad={squad} />}
        </div>
      </div>
    </>
  );
}
