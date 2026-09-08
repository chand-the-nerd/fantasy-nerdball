import { useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError } from "../lib/api";

const CHIPS: { id: string; label: string; short: string }[] = [
  { id: "wildcard", label: "Wildcard", short: "WC" },
  { id: "free_hit", label: "Free Hit", short: "FH" },
  { id: "bench_boost", label: "Bench Boost", short: "BB" },
  { id: "triple_captain", label: "Triple Captain", short: "TC" },
];

const POSITION_ORDER = ["GK", "DEF", "MID", "FWD"];

interface PlanPlayer {
  id: number | null;
  name: string;
  position: string;
  team: string;
  price: number | null;
  projected_points: number | null;
  is_captain: boolean;
  on_bench: boolean;
}

interface PlanWeek {
  gameweek: number;
  chip: string;
  formation: string;
  projected_points: number | null;
  squad_value: number | null;
  bank: number | null;
  free_transfers_available: number;
  transfers_made: number;
  penalty_points: number;
  transfers: { in: string[]; out: string[] };
  starting: PlanPlayer[];
  bench: PlanPlayer[];
}

interface Plan {
  id: number;
  start_gameweek: number;
  weeks: number;
  chips: Record<string, string>;
  status: string;
  progress: number;
  log: string;
  error: string;
  payload: PlanWeek[];
}

type Cell = { state: "start" | "bench"; captain: boolean } | null;

/** One row per player who appears at any point, with their week-by-week state. */
function buildRows(weeks: PlanWeek[]) {
  const rows = new Map<string, { player: PlanPlayer; cells: Cell[] }>();

  weeks.forEach((week, index) => {
    for (const player of [...week.starting, ...week.bench]) {
      const key = String(player.id ?? player.name);
      let row = rows.get(key);
      if (!row) {
        row = { player, cells: new Array(weeks.length).fill(null) };
        rows.set(key, row);
      }
      row.cells[index] = {
        state: player.on_bench ? "bench" : "start",
        captain: player.is_captain,
      };
    }
  });

  return [...rows.values()].sort((a, b) => {
    const byPosition =
      POSITION_ORDER.indexOf(a.player.position) -
      POSITION_ORDER.indexOf(b.player.position);
    if (byPosition !== 0) return byPosition;
    // Then by how long they're kept: the ever-presents group together and the
    // churn sits at the bottom, which is where the eye should go.
    const span = (cells: Cell[]) => cells.filter(Boolean).length;
    return span(b.cells) - span(a.cells) || a.player.name.localeCompare(b.player.name);
  });
}

function Timeline({ plan }: { plan: Plan }) {
  const weeks = plan.payload;
  const rows = useMemo(() => buildRows(weeks), [weeks]);
  const columns = `minmax(112px, 1.4fr) repeat(${weeks.length}, minmax(46px, 1fr))`;

  let lastPosition = "";

  return (
    <div className="plan-scroll">
      <div className="plan-grid" style={{ gridTemplateColumns: columns }}>
        <div className="plan-corner">Squad</div>
        {weeks.map((week) => {
          const chip = CHIPS.find((entry) => entry.id === week.chip);
          return (
            <div key={week.gameweek} className="plan-col-head">
              <strong>GW{week.gameweek}</strong>
              {chip && <span className="plan-chip">{chip.short}</span>}
              <span className="plan-col-sub">
                {week.transfers_made === 0
                  ? "hold"
                  : `${week.transfers_made} in`}
              </span>
              <span className="plan-col-sub">
                {week.projected_points != null
                  ? `${week.projected_points.toFixed(0)} pts`
                  : "—"}
              </span>
            </div>
          );
        })}

        {rows.map((row) => {
          const heading =
            row.player.position !== lastPosition ? row.player.position : "";
          lastPosition = row.player.position;
          return (
            <div className="plan-row" key={row.player.id ?? row.player.name}>
              {heading && (
                <div className="plan-group" style={{ gridColumn: "1 / -1" }}>
                  {heading}
                </div>
              )}
              <div className="plan-name" title={`${row.player.name} · ${row.player.team}`}>
                <span>{row.player.name}</span>
                <span className="plan-team">{row.player.team}</span>
              </div>
              {row.cells.map((cell, index) => {
                const previous = index > 0 ? row.cells[index - 1] : null;
                const next = index < row.cells.length - 1 ? row.cells[index + 1] : null;
                const arriving = cell && !previous;
                const leaving = cell && !next;
                const classes = [
                  "plan-cell",
                  cell ? `is-${cell.state}` : "is-out",
                  arriving ? "is-arriving" : "",
                  leaving ? "is-leaving" : "",
                ]
                  .filter(Boolean)
                  .join(" ");
                const label = !cell
                  ? "not in the squad"
                  : cell.state === "start"
                    ? cell.captain
                      ? "starting, captain"
                      : "starting"
                    : "on the bench";
                return (
                  <div
                    key={index}
                    className={classes}
                    title={`GW${weeks[index].gameweek}: ${row.player.name} — ${label}${
                      arriving && index > 0 ? " (transferred in)" : ""
                    }`}
                  >
                    {cell?.captain && <span className="plan-captain">C</span>}
                    {arriving && index > 0 && <span className="plan-mark in">▲</span>}
                    {leaving && index < row.cells.length - 1 && (
                      <span className="plan-mark out">▼</span>
                    )}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function WeekDetail({ week }: { week: PlanWeek }) {
  const chip = CHIPS.find((entry) => entry.id === week.chip);
  return (
    <div className="plan-week">
      <div className="plan-week-head">
        <strong>GW{week.gameweek}</strong>
        {chip && <span className="plan-chip">{chip.label}</span>}
      </div>
      <ul>
        {week.transfers.out.map((name, i) => (
          <li key={`out-${i}`}>
            <span className="dir-mark out">↓</span>
            {name}
          </li>
        ))}
        {week.transfers.in.map((name, i) => (
          <li key={`in-${i}`}>
            <span className="dir-mark in">↑</span>
            {name}
          </li>
        ))}
        {week.transfers.in.length === 0 && week.transfers.out.length === 0 && (
          <li className="muted">Squad held</li>
        )}
      </ul>
      <p className="muted">
        {week.free_transfers_available} free
        {week.penalty_points > 0 ? ` · −${week.penalty_points} pts` : ""} · £
        {(week.bank ?? 0).toFixed(1)}m in the bank
      </p>
    </div>
  );
}

export function PlannerView() {
  const [plan, setPlan] = useState<Plan | null>(null);
  const [weeks, setWeeks] = useState(5);
  const [start, setStart] = useState<number | null>(null);
  const [chips, setChips] = useState<Record<number, string>>({});
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const poll = useRef<number | null>(null);

  useEffect(() => {
    api
      .gameweek()
      .then((info) => setStart(info.gameweek))
      .catch(() => undefined);
    api
      .latestPlan()
      .then((existing) => existing && setPlan(existing))
      .catch(() => undefined);
  }, []);

  const running = plan?.status === "queued" || plan?.status === "running";

  // Poll while it works. A plan is several runs, so this is minutes, not
  // seconds — the progress count is what makes the wait bearable.
  useEffect(() => {
    if (!running || !plan) return;
    poll.current = window.setInterval(async () => {
      try {
        setPlan(await api.plan(plan.id));
      } catch {
        /* keep the last state and try again */
      }
    }, 3000);
    return () => {
      if (poll.current) window.clearInterval(poll.current);
    };
  }, [running, plan?.id]);

  const covered = useMemo(
    () =>
      start === null
        ? []
        : Array.from({ length: weeks }, (_, index) => start + index),
    [start, weeks],
  );

  const startPlan = async () => {
    setError("");
    setBusy(true);
    try {
      const chipMap: Record<string, string> = {};
      for (const [gw, chip] of Object.entries(chips)) {
        if (chip && covered.includes(Number(gw))) chipMap[gw] = chip;
      }
      setPlan(await api.startPlan({ weeks, chips: chipMap }));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const chipTaken = (chip: string, gameweek: number) =>
    Object.entries(chips).some(
      ([gw, value]) => value === chip && Number(gw) !== gameweek,
    );

  return (
    <>
      <div className="topbar">
        <div>
          <h1>Planner</h1>
          <span className="when">
            The optimiser run forward, week after week
          </span>
        </div>
      </div>

      <div className="notice">
        A plan assumes today's prices and today's form hold for the whole run.
        It can't see price changes, injuries, or a player finding form — so
        treat the first week or two as advice and the rest as a sketch of the
        shape.
      </div>

      {error && <div className="notice bad">{error}</div>}

      <div className="panel">
        <div className="plan-controls">
          <div className="field">
            <label htmlFor="plan-weeks">Gameweeks to plan</label>
            <select
              id="plan-weeks"
              value={weeks}
              disabled={running}
              onChange={(event) => setWeeks(Number(event.target.value))}
            >
              {[3, 4, 5, 6, 7, 8].map((option) => (
                <option key={option} value={option}>
                  {option} weeks
                </option>
              ))}
            </select>
            <span className="hint">
              Each one is a full optimisation, so eight takes a while.
            </span>
          </div>

          <button
            className="btn"
            type="button"
            onClick={startPlan}
            disabled={busy || running || start === null}
          >
            {running ? "Planning…" : "Run the plan"}
          </button>
        </div>

        <div className="chip-plan">
          <span className="chip-plan-label">Chips</span>
          <div className="chip-plan-grid">
            {covered.map((gameweek) => (
              <div className="chip-plan-week" key={gameweek}>
                <span>GW{gameweek}</span>
                <select
                  value={chips[gameweek] ?? ""}
                  disabled={running}
                  aria-label={`Chip for gameweek ${gameweek}`}
                  onChange={(event) =>
                    setChips((current) => ({
                      ...current,
                      [gameweek]: event.target.value,
                    }))
                  }
                >
                  <option value="">—</option>
                  {CHIPS.map((chip) => (
                    <option
                      key={chip.id}
                      value={chip.id}
                      disabled={chipTaken(chip.id, gameweek)}
                    >
                      {chip.label}
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </div>
        </div>
      </div>

      {running && (
        <div className="panel plan-progress">
          <h3>
            Working through the weeks — {plan?.progress ?? 0} of {plan?.weeks}
          </h3>
          <div className="plan-bar">
            <span
              style={{
                width: `${((plan?.progress ?? 0) / (plan?.weeks || 1)) * 100}%`,
              }}
            />
          </div>
          <p className="muted">
            {(plan?.log || "").split("\n").filter(Boolean).slice(-1)[0] ||
              "Queued."}
          </p>
        </div>
      )}

      {plan?.status === "failed" && (
        <div className="notice bad">{plan.error || "The plan didn't finish."}</div>
      )}

      {plan?.status === "complete" && plan.payload.length > 0 && (
        <>
          <div className="panel">
            <h3>Squad over time</h3>
            <p className="muted" style={{ marginTop: -4 }}>
              Solid means starting, faded means benched, empty means not in the
              squad. ▲ is a transfer in, ▼ the last week before one goes.
            </p>
            <Timeline plan={plan} />
          </div>

          <div className="plan-weeks">
            {plan.payload.map((week) => (
              <WeekDetail key={week.gameweek} week={week} />
            ))}
          </div>
        </>
      )}
    </>
  );
}
