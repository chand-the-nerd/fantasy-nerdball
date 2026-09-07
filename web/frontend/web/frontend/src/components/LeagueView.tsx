import { useEffect, useState } from "react";
import { api, ApiError } from "../lib/api";
import type { League } from "../lib/types";

export function LeagueView() {
  const [league, setLeague] = useState<League | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .league()
      .then(setLeague)
      .catch((err) => setError(err instanceof ApiError ? err.message : String(err)));
  }, []);

  if (error) return <div className="notice bad">{error}</div>;
  if (!league) return <p className="muted">Loading the table…</p>;

  const scored = league.standings.some((row) => row.gameweeks > 0);

  return (
    <>
      <div className="topbar">
        <div>
          <h1>League</h1>
          <span className="when">Season {league.season}</span>
        </div>
      </div>

      {!scored ? (
        <div className="panel">
          <h3>Nothing to compare yet</h3>
          <p className="muted">
            Scores appear here once managers link their FPL teams or record a
            gameweek by hand. The global average is tracked alongside, so you can
            see who is actually beating the game rather than each other.
          </p>
        </div>
      ) : (
        <div className="panel">
          <table>
            <thead>
              <tr>
                <th style={{ width: 36 }}>#</th>
                <th>Manager</th>
                <th className="num">Points</th>
                <th className="num">Gameweeks</th>
                <th className="num">Overall rank</th>
              </tr>
            </thead>
            <tbody>
              {league.standings.map((row) => (
                <tr key={row.user_id} className={row.is_you ? "you" : undefined}>
                  <td>{row.position}</td>
                  <td>
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 9 }}>
                      {row.avatar_url && (
                        <img
                          src={row.avatar_url}
                          alt=""
                          width={22}
                          height={22}
                          style={{ borderRadius: "50%" }}
                        />
                      )}
                      {row.name}
                      {row.is_you && <span className="muted">you</span>}
                    </span>
                  </td>
                  <td className="num">{row.total_points}</td>
                  <td className="num">{row.gameweeks}</td>
                  <td className="num">
                    {row.overall_rank ? row.overall_rank.toLocaleString("en-GB") : "—"}
                  </td>
                </tr>
              ))}
              <tr>
                <td />
                <td className="muted">Global average</td>
                <td className="num muted">{league.global_average_total}</td>
                <td className="num muted">—</td>
                <td className="num muted">—</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
