import { useEffect, useMemo, useState } from "react";
import { api, ApiError } from "../lib/api";

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
  attack_overperformance: number | null;
  defence_overperformance: number | null;
  fixture_difficulty: number | null;
  xg_rank: number;
  next_fixtures: {
    gameweek: number;
    opponent: string;
    venue: string;
    difficulty: number;
  }[];
}

type SortKey = "xg_difference" | "xg_per_game" | "xgc_per_game" | "fixture_difficulty" | "name";

export function TeamsView() {
  const [data, setData] = useState<{ teams: Team[]; look_ahead: number } | null>(null);
  const [error, setError] = useState("");
  const [sort, setSort] = useState<SortKey>("xg_difference");
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
    const rows = [...data.teams];
    rows.sort((a, b) => {
      if (sort === "name") return a.name.localeCompare(b.name);
      if (sort === "fixture_difficulty")
        return (a.fixture_difficulty ?? 9) - (b.fixture_difficulty ?? 9);
      if (sort === "xgc_per_game") return (a.xgc_per_game ?? 9) - (b.xgc_per_game ?? 9);
      return (b[sort] ?? -99) - (a[sort] ?? -99);
    });
    return rows;
  }, [data, sort]);

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
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <select
            className="btn quiet small"
            value={sort}
            onChange={(e) => setSort(e.target.value as SortKey)}
            aria-label="Sort teams by"
          >
            <option value="xg_difference">Sort: expected goal difference</option>
            <option value="xg_per_game">Sort: attack</option>
            <option value="xgc_per_game">Sort: defence</option>
            <option value="fixture_difficulty">Sort: easiest fixtures</option>
            <option value="name">Sort: name</option>
          </select>
          <button className="btn quiet small" onClick={() => load(true)} disabled={busy} type="button">
            {busy ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>

      <div className="notice">
        Ratings are built from expected goals rather than results. A side winning
        on 0.6 xG a game is being flattered; one losing on 2.1 xG is better than
        the table says. The overperformance columns show which is which.
      </div>

      <div className="panel">
        <div className="calc-scroll">
          <table className="calc-table teams-table">
            <thead>
              <tr>
                <th>Team</th>
                <th className="num">P</th>
                <th className="num" title="Goals scored">GF</th>
                <th className="num" title="Goals conceded">GA</th>
                <th className="num" title="Expected goals per game — the quality of chances created">
                  xG/g
                </th>
                <th className="num" title="Expected goals conceded per game — chances allowed. Lower is better.">
                  xGC/g
                </th>
                <th className="num" title="Expected goals minus expected conceded, across the season">
                  xGD
                </th>
                <th
                  className="num"
                  title="Goals scored divided by expected. Above 1.00 is finishing above the chances, which rarely lasts."
                >
                  Finishing
                </th>
                <th
                  className="num"
                  title="Goals conceded divided by expected. Below 1.00 means the keeper and defence are saving them."
                >
                  Keeping
                </th>
                <th className="num" title={`Average difficulty of the next ${data.look_ahead} fixtures`}>
                  FDR
                </th>
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
                    style={{
                      color: team.xg_difference >= 0 ? "var(--gain)" : "var(--flag)",
                    }}
                  >
                    {team.xg_difference > 0 ? "+" : ""}
                    {team.xg_difference}
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
                  <td className="num">{team.fixture_difficulty ?? "—"}</td>
                  <td>
                    <span className="fdr-run">
                      {team.next_fixtures.map((fixture, index) => (
                        <span
                          key={index}
                          className={`fdr fdr-${fixture.difficulty}`}
                          title={`GW${fixture.gameweek} ${fixture.opponent} (${fixture.venue})`}
                        >
                          {fixture.opponent}
                          {fixture.venue === "Home" ? "" : "*"}
                        </span>
                      ))}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="hint" style={{ marginTop: 12 }}>
          An asterisk marks an away fixture. Fixture colours run from easy to
          hard using FPL's own difficulty rating.
        </p>
      </div>
    </>
  );
}
