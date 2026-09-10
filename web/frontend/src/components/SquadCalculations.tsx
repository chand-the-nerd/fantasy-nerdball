import { useState } from "react";
import { GuestLock } from "./GuestLock";
import { useGuest } from "../lib/guest";
import type { Player, Squad } from "../lib/types";

type SortKey =
  | "name"
  | "position"
  | "price"
  | "projected_points"
  | "form"
  | "historic_ppg"
  | "fixture_difficulty"
  | "start_rate"
  | "minutes_per_game"
  | "xg_modifier";

const COLUMNS: {
  key: SortKey;
  label: string;
  title: string;
  numeric: boolean;
  format?: (player: Player) => string;
}[] = [
  { key: "name", label: "Player", title: "Player name", numeric: false },
  { key: "position", label: "Pos", title: "Position", numeric: false },
  {
    key: "price",
    label: "Price",
    title: "Current FPL price",
    numeric: true,
    format: (p) => `£${p.price.toFixed(1)}m`,
  },
  {
    key: "form",
    label: "Form",
    title: "FPL form: average points over recent gameweeks",
    numeric: true,
  },
  {
    key: "historic_ppg",
    label: "History",
    title: "Points per game across previous seasons, weighted by recency",
    numeric: true,
  },
  {
    key: "fixture_difficulty",
    label: "Fixtures",
    title:
      "Average difficulty of the upcoming fixtures in your look-ahead window. Lower is easier.",
    numeric: true,
  },
  {
    key: "start_rate",
    label: "Starts",
    title: "How often they start when available",
    numeric: true,
    format: (p) => (p.start_rate != null ? `${p.start_rate}%` : "—"),
  },
  {
    key: "minutes_per_game",
    label: "Mins",
    title: "Average minutes per gameweek",
    numeric: true,
  },
  {
    key: "xg_modifier",
    label: "xG mult",
    title:
      "Expected-goals consistency: above 1.00 means underlying numbers back the returns up",
    numeric: true,
    format: (p) => (p.xg_modifier != null ? p.xg_modifier.toFixed(2) : "—"),
  },
  {
    key: "projected_points",
    label: "Projected",
    title: "The model's projected points for this gameweek",
    numeric: true,
    format: (p) => p.projected_points.toFixed(1),
  },
];

function value(player: Player, key: SortKey): string | number {
  const raw = player[key as keyof Player];
  if (raw === null || raw === undefined || raw === "") return key === "name" ? "" : -Infinity;
  return raw as string | number;
}

export function SquadCalculations({ squad }: { squad: Squad }) {
  const [sort, setSort] = useState<SortKey>("projected_points");
  const [ascending, setAscending] = useState(false);
  const guest = useGuest();

  const players = [...squad.payload.starting, ...squad.payload.bench];
  const sorted = [...players].sort((a, b) => {
    const left = value(a, sort);
    const right = value(b, sort);
    const result =
      typeof left === "string" || typeof right === "string"
        ? String(left).localeCompare(String(right))
        : (left as number) - (right as number);
    return ascending ? result : -result;
  });

  const toggle = (key: SortKey) => {
    if (key === sort) setAscending((v) => !v);
    else {
      setSort(key);
      setAscending(key === "name" || key === "position" || key === "fixture_difficulty");
    }
  };

  return (
    <div className="panel calc-panel">
      <h3>Calculations</h3>
      <p className="muted" style={{ marginTop: -6 }}>
        The inputs behind every prediction;
        click to sort.
      </p>

      <Wrap locked={guest}>
      <div className="calc-scroll">
        <table className="calc-table">
          <thead>
            <tr>
              {COLUMNS.map((column) => (
                <th
                  key={column.key}
                  className={column.numeric ? "num" : undefined}
                  title={column.title}
                >
                  <button type="button" onClick={() => toggle(column.key)}>
                    {column.label}
                    {sort === column.key && <span>{ascending ? " ↑" : " ↓"}</span>}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((player, index) => (
              <tr key={`${player.id}-${index}`} className={player.on_bench ? "is-bench" : undefined}>
                {COLUMNS.map((column) => {
                  const cell = column.format
                    ? column.format(player)
                    : (player[column.key as keyof Player] ?? "—");
                  return (
                    <td key={column.key} className={column.numeric ? "num" : undefined}>
                      {column.key === "name" ? (
                        <span className="calc-name">
                          {player.name}
                          {player.is_captain && <span className="calc-tag">C</span>}
                          {player.is_vice && <span className="calc-tag vice">V</span>}
                          {player.on_bench && <span className="calc-tag bench">bench</span>}
                        </span>
                      ) : (
                        String(cell)
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      </Wrap>
    </div>
  );
}

/** The table, veiled for guests and plain for everyone else. */
function Wrap({
  locked,
  children,
}: {
  locked: boolean;
  children: React.ReactNode;
}) {
  if (!locked) return <>{children}</>;
  return (
    <GuestLock
      note={
        "Every number the projection was built from, player by player."
      }
    >
      {children}
    </GuestLock>
  );
}

export function ExploredTransfers({ squad }: { squad: Squad }) {
  const explored = squad.payload.explored ?? [];
  if (explored.length === 0) return null;

  return (
    <div className="panel">
      <h3>What it considered</h3>
      <p className="muted" style={{ marginTop: -6 }}>
        Moves the model weighed up before settling on its recommendation.
      </p>
      <div className="calc-scroll">
        <table className="calc-table">
          <thead>
            <tr>
              <th title="The player the model looked at moving on">Out</th>
              <th title="The best replacement it found">Replacement</th>
              <th className="num" title="Projected points given up by not transferring">
                Cost
              </th>
              <th title="What it decided">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {explored.map((row, index) => (
              <tr key={index}>
                <td>
                  {row.player_out}
                  {row.position && <span className="muted"> {row.position}</span>}
                </td>
                <td>{row.replacement ?? <span className="muted">nothing suitable</span>}</td>
                <td
                  className="num"
                  style={{
                    color:
                      (row.points_lost ?? 0) >= 2 ? "var(--flag)" : "var(--fade)",
                  }}
                >
                  {row.points_lost != null ? row.points_lost.toFixed(1) : "—"}
                </td>
                <td className="muted">
                  {row.verdict === "substitute"
                    ? "Bench cover is enough"
                    : row.verdict === "consider_transfer"
                      ? "Worth a transfer"
                      : row.verdict === "transfer"
                        ? "Must transfer"
                        : row.verdict}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
