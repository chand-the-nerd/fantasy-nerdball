import { useEffect, useMemo, useState } from "react";
import { api, ApiError } from "../lib/api";

interface Fixture {
  gameweek: number;
  opponent: string;
  venue: string;
  difficulty: number;
  attack_difficulty: number | null;
  defence_difficulty: number | null;
  opponent_attack_rating: number;
  opponent_defence_rating: number;
}

interface Team {
  id: number;
  name: string;
  short_name: string;
  played: number;
  goals_for: number;
  goals_against: number;
  xg: number;
  xgc: number;
  xg_per_game: number | null;
  xgc_per_game: number | null;
  goal_difference: number;
  xg_difference: number;
  xg_difference_per_game: number | null;
  attack_rating: number;
  defence_rating: number;
  attack_overperformance: number | null;
  defence_overperformance: number | null;
  fixture_difficulty: number | null;
  attack_fdr: number | null;
  defence_fdr: number | null;
  xg_rank: number;
  next_fixtures: Fixture[];
}

type SortKey =
  | "name"
  | "played"
  | "goals_for"
  | "goals_against"
  | "xg_per_game"
  | "xgc_per_game"
  | "xg_difference_per_game"
  | "attack_overperformance"
  | "defence_overperformance"
  | "attack_fdr"
  | "defence_fdr";

/** Columns where a small number is the good one, so a first click sorts up. */
const ASCENDING_FIRST: SortKey[] = [
  "name",
  "goals_against",
  "xgc_per_game",
  "attack_fdr",
  "defence_fdr",
];

const COLUMNS: { key: SortKey; label: string; title: string; numeric: boolean }[] = [
  { key: "name", label: "Team", title: "Club name", numeric: false },
  { key: "played", label: "P", title: "Games played", numeric: true },
  { key: "goals_for", label: "GF", title: "Goals scored", numeric: true },
  { key: "goals_against", label: "GA", title: "Goals conceded", numeric: true },
  {
    key: "xg_per_game",
    label: "xG/g",
    title: "Expected goals per game — the quality of chances created",
    numeric: true,
  },
  {
    key: "xgc_per_game",
    label: "xGC/g",
    title: "Expected goals conceded per game — chances allowed. Lower is better.",
    numeric: true,
  },
  {
    key: "xg_difference_per_game",
    label: "xGD/g",
    title: "Expected goals minus expected conceded, per game",
    numeric: true,
  },
  {
    key: "attack_overperformance",
    label: "Finishing",
    title:
      "Goals scored divided by expected. Above 1.00 is finishing above the chances, which rarely lasts.",
    numeric: true,
  },
  {
    key: "defence_overperformance",
    label: "Keeping",
    title:
      "Goals conceded divided by expected. Below 1.00 means the keeper and defence are saving them.",
    numeric: true,
  },
  {
    key: "attack_fdr",
    label: "FDR att",
    title:
      "Difficulty for their attackers, adjusted for how freely the opponents concede. Lower is easier.",
    numeric: true,
  },
  {
    key: "defence_fdr",
    label: "FDR def",
    title:
      "Difficulty for their defenders and keeper, adjusted for how much the opponents create. Lower is easier.",
    numeric: true,
  },
];

function difficultyColour(value: number | null): string | undefined {
  if (value == null) return undefined;
  if (value <= 2.4) return "var(--gain)";
  if (value >= 3.6) return "var(--flag)";
  return undefined;
}

function value(team: Team, key: SortKey): string | number {
  if (key === "name") return team.name;
  const raw = team[key];
  if (raw === null || raw === undefined) return -Infinity;
  return raw as number;
}

export function TeamsView() {
  const [data, setData] = useState<{ teams: Team[]; look_ahead: number } | null>(null);
  const [error, setError] = useState("");
  const [sort, setSort] = useState<SortKey>("xg_difference_per_game");
  const [ascending, setAscending] = useState(false);
  const [colourBy, setColourBy] = useState<"attack" | "defence">("attack");
  const [busy, setBusy] = useState(false);

  const load = (refresh = false) => {
    setBusy(true);
    api
      .teams(refresh)
      .then(setData)
      .catch((err) => setError(err instanceof ApiError ? err.message : String(err)))
      .finally(() => setBusy(false));
  };

  useEffect(() => load(), []);

  const sorted = useMemo(() => {
    if (!data) return [];
    return [...data.teams].sort((a, b) => {
      const left = value(a, sort);
      const right = value(b, sort);
      const result =
        typeof left === "string" || typeof right === "string"
          ? String(left).localeCompare(String(right))
          : (left as number) - (right as number);
      return ascending ? result : -result;
    });
  }, [data, sort, ascending]);

  const toggle = (key: SortKey) => {
    if (key === sort) setAscending((current) => !current);
    else {
      setSort(key);
      setAscending(ASCENDING_FIRST.includes(key));
    }
  };

  if (error) return <div className="notice bad">{error}</div>;
  if (!data) return <p className="muted">Working out team strength…</p>;

  return (
    <>
      <div className="topbar">
        <div>
          <h1>Teams</h1>
          <span className="when">
            Strength from the underlying numbers, not just results
          </span>
        </div>
        <button
          className="btn quiet small"
          onClick={() => load(true)}
          disabled={busy}
          type="button"
        >
          {busy ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      <div className="panel">
        <div className="table-controls">
          <p className="muted" style={{ margin: 0 }}>
            Click a heading to sort. The next {data.look_ahead} fixtures are
            coloured by {colourBy === "attack" ? "attacking" : "defensive"}{" "}
            difficulty.
          </p>
          <div className="segmented">
            {(
              [
                ["attack", "Colour: attack"],
                ["defence", "Colour: defence"],
              ] as ["attack" | "defence", string][]
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                className={colourBy === id ? "is-active" : ""}
                onClick={() => setColourBy(id)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="calc-scroll">
          <table className="calc-table teams-table">
            <thead>
              <tr>
                {COLUMNS.map((column) => (
                  <th
                    key={column.key}
                    className={column.numeric ? "num" : undefined}
                    title={column.title}
                    aria-sort={
                      sort === column.key
                        ? ascending
                          ? "ascending"
                          : "descending"
                        : undefined
                    }
                  >
                    <button type="button" onClick={() => toggle(column.key)}>
                      {column.label}
                      {sort === column.key && <span>{ascending ? " ↑" : " ↓"}</span>}
                    </button>
                  </th>
                ))}
                <th>Next fixtures</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((team) => (
                <tr key={team.id}>
                  <td>
                    <span className="calc-name">{team.name}</span>
                  </td>
                  <td className="num muted">{team.played}</td>
                  <td className="num">{team.goals_for}</td>
                  <td className="num">{team.goals_against}</td>
                  <td className="num">{team.xg_per_game ?? "—"}</td>
                  <td className="num">{team.xgc_per_game ?? "—"}</td>
                  <td
                    className="num"
                    title={`${team.xg_difference} across the season so far`}
                    style={{
                      color:
                        (team.xg_difference_per_game ?? 0) >= 0
                          ? "var(--gain)"
                          : "var(--flag)",
                    }}
                  >
                    {team.xg_difference_per_game == null
                      ? "—"
                      : `${team.xg_difference_per_game > 0 ? "+" : ""}${
                          team.xg_difference_per_game
                        }`}
                  </td>
                  <td
                    className="num"
                    style={{
                      color:
                        team.attack_overperformance == null
                          ? undefined
                          : team.attack_overperformance > 1.15
                            ? "var(--flag)"
                            : team.attack_overperformance < 0.85
                              ? "var(--gain)"
                              : undefined,
                    }}
                    title={
                      team.attack_overperformance == null
                        ? undefined
                        : team.attack_overperformance > 1.15
                          ? "Scoring above the chances — expect this to fall back"
                          : team.attack_overperformance < 0.85
                            ? "Scoring below the chances — due a correction upwards"
                            : "Scoring roughly in line with the chances"
                    }
                  >
                    {team.attack_overperformance ?? "—"}
                  </td>
                  <td
                    className="num"
                    style={{
                      color:
                        team.defence_overperformance == null
                          ? undefined
                          : team.defence_overperformance < 0.85
                            ? "var(--gain)"
                            : team.defence_overperformance > 1.15
                              ? "var(--flag)"
                              : undefined,
                    }}
                  >
                    {team.defence_overperformance ?? "—"}
                  </td>
                  <td
                    className="num"
                    style={{ color: difficultyColour(team.attack_fdr) }}
                    title="Average over the fixtures alongside, for their attackers"
                  >
                    {team.attack_fdr ?? "—"}
                  </td>
                  <td
                    className="num"
                    style={{ color: difficultyColour(team.defence_fdr) }}
                    title="Average over the fixtures alongside, for their defence"
                  >
                    {team.defence_fdr ?? "—"}
                  </td>
                  <td>
                    <span className="fdr-run">
                      {team.next_fixtures.map((fixture, index) => {
                        const shown =
                          colourBy === "attack"
                            ? fixture.attack_difficulty
                            : fixture.defence_difficulty;
                        const band = Math.round(shown ?? fixture.difficulty);
                        return (
                          <span
                            key={index}
                            className={`fdr fdr-${band}`}
                            title={
                              `GW${fixture.gameweek} ${fixture.opponent} (${fixture.venue})\n` +
                              `FPL difficulty ${fixture.difficulty}\n` +
                              `Attackers ${fixture.attack_difficulty ?? "—"} · ` +
                              `defenders ${fixture.defence_difficulty ?? "—"}`
                            }
                          >
                            {fixture.opponent}
                            {fixture.venue === "Home" ? "" : "*"}
                          </span>
                        );
                      })}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel glossary">
        <h3>What the columns mean</h3>
        <dl>
          <div>
            <dt>P · GF · GA</dt>
            <dd>Games played, goals scored, goals conceded. Results as they stand.</dd>
          </div>
          <div>
            <dt>xG/g</dt>
            <dd>
              Expected goals per game: the quality of the chances a side creates,
              whether or not they went in. Higher is better.
            </dd>
          </div>
          <div>
            <dt>xGC/g</dt>
            <dd>
              Expected goals conceded per game, taken from the busiest
              goalkeeper so the same shot isn't counted eleven times over. Lower
              is better.
            </dd>
          </div>
          <div>
            <dt>xGD/g</dt>
            <dd>
              xG/g minus xGC/g, and the best one-number summary of how good a
              side actually is. Hover for the season total.
            </dd>
          </div>
          <div>
            <dt>Finishing</dt>
            <dd>
              Goals scored divided by expected goals. Above 1.00 means scoring
              more than the chances merit, which rarely holds; below 1.00 is a
              side due a correction upwards.
            </dd>
          </div>
          <div>
            <dt>Keeping</dt>
            <dd>
              Goals conceded divided by expected conceded. Below 1.00 means the
              keeper and defence are keeping out more than they should.
            </dd>
          </div>
          <div>
            <dt>FDR att</dt>
            <dd>
              Fixture difficulty for this side's attackers over the next{" "}
              {data.look_ahead} games. FPL's own 1–5 rating, moved by how freely
              each opponent concedes expected goals — so a side that's hard to
              beat but still leaks chances rates as a softer fixture for a
              striker than FPL says. Home games get a small discount. Lower is
              easier.
            </dd>
          </div>
          <div>
            <dt>FDR def</dt>
            <dd>
              The same fixtures for this side's defenders and goalkeeper, moved
              instead by how much expected goal threat each opponent creates.
              The two regularly disagree, which is the point: one fixture can be
              worth attacking and worth avoiding at the back.
            </dd>
          </div>
          <div>
            <dt>Next fixtures</dt>
            <dd>
              Opponents in order, with an asterisk marking an away game. Colours
              follow whichever difficulty is selected above the table, so the
              same fixture can read green for attackers and red for defenders.
            </dd>
          </div>
        </dl>
      </div>
    </>
  );
}
