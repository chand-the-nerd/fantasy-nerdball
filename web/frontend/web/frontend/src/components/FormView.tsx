import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, ApiError } from "../lib/api";
import type { Me, Performance } from "../lib/types";

const AXIS = { stroke: "#7C978D", fontSize: 12, fontFamily: "Archivo" };

export function FormView({ me }: { me: Me }) {
  const [data, setData] = useState<Performance | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [entryGw, setEntryGw] = useState("");
  const [entryPoints, setEntryPoints] = useState("");

  const load = async (refresh = false) => {
    setBusy(true);
    setError("");
    try {
      setData(await api.performance(refresh));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const submitResult = async () => {
    const gw = Number(entryGw);
    const points = Number(entryPoints);
    if (!gw || Number.isNaN(points)) {
      setError("Enter a gameweek and a score.");
      return;
    }
    try {
      await api.recordResult(gw, points);
      setEntryGw("");
      setEntryPoints("");
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    }
  };

  if (!data) {
    return <p className="muted">{error || "Loading your season…"}</p>;
  }

  const s = data.summary;
  const ahead = s.points_above_average >= 0;
  const chartData = data.series.filter(
    (row) => row.actual !== null || row.projected !== null || row.global_average !== null,
  );

  return (
    <>
      <div className="topbar">
        <div>
          <h1>Form</h1>
          <span className="when">Season {data.season}</span>
        </div>
        <button className="btn quiet" onClick={() => load(true)} disabled={busy} type="button">
          {busy ? "Refreshing…" : "Refresh from FPL"}
        </button>
      </div>

      {error && <div className="notice bad">{error}</div>}

      {!s.fpl_entry_linked && (
        <div className="notice">
          Link your FPL team under Setup and your real points appear here
          automatically. Otherwise, enter each week's score below.
        </div>
      )}

      {s.gameweeks_scored > 0 && (
        <div className={`notice ${ahead ? "good" : "bad"}`}>
          {ahead
            ? `You're ${s.points_above_average} points clear of the global average across ${s.gameweeks_scored} scored gameweeks, beating it in ${s.gameweeks_beating_average} of them.`
            : `You're ${Math.abs(s.points_above_average)} points behind the global average across ${s.gameweeks_scored} scored gameweeks, beating it in ${s.gameweeks_beating_average} of them.`}
          {s.latest_overall_rank && ` Overall rank ${s.latest_overall_rank.toLocaleString("en-GB")}.`}
          {s.model_mean_error != null &&
            ` The model's projection has been out by ${s.model_mean_error} points a week on average.`}
        </div>
      )}

      <div className="panel" style={{ paddingTop: 22 }}>
        <div style={{ width: "100%", height: 340 }}>
          <ResponsiveContainer>
            <LineChart data={chartData} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
              <CartesianGrid stroke="#1E3B33" vertical={false} />
              <XAxis
                dataKey="gameweek"
                tick={AXIS}
                axisLine={{ stroke: "#1E3B33" }}
                tickLine={false}
                tickFormatter={(gw) => `GW${gw}`}
              />
              <YAxis tick={AXIS} axisLine={false} tickLine={false} width={44} />
              <Tooltip
                contentStyle={{
                  background: "#10241E",
                  border: "1px solid #1E3B33",
                  borderRadius: 4,
                  fontFamily: "Archivo",
                  fontSize: 13,
                }}
                labelFormatter={(gw) => `Gameweek ${gw}`}
                labelStyle={{ color: "#EAF2EE", fontWeight: 600 }}
              />
              <Legend wrapperStyle={{ fontSize: 13, fontFamily: "Archivo" }} />
              <Line
                name="Your points"
                type="monotone"
                dataKey="actual"
                stroke="#F2C14E"
                strokeWidth={2.5}
                dot={{ r: 3, fill: "#F2C14E", strokeWidth: 0 }}
                connectNulls
              />
              <Line
                name="Global average"
                type="monotone"
                dataKey="global_average"
                stroke="#56C7E8"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
              <Line
                name="Model projection"
                type="monotone"
                dataKey="projected"
                stroke="#7C978D"
                strokeWidth={1.5}
                strokeDasharray="4 3"
                dot={false}
                connectNulls
              />
              <Line
                name="Week's best"
                type="monotone"
                dataKey="global_highest"
                stroke="#2E5A50"
                strokeWidth={1}
                dot={false}
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ display: "grid", gap: 16, gridTemplateColumns: "minmax(0,2fr) minmax(0,1fr)", marginTop: 20, alignItems: "start" }}>
        <div className="panel">
          <h3>Gameweek by gameweek</h3>
          <table>
            <thead>
              <tr>
                <th>GW</th>
                <th className="num">You</th>
                <th className="num">Average</th>
                <th className="num">Difference</th>
                <th className="num">Projected</th>
                <th className="num">Rank</th>
              </tr>
            </thead>
            <tbody>
              {[...chartData].reverse().map((row) => {
                const diff =
                  row.actual != null && row.global_average != null
                    ? row.actual - row.global_average
                    : null;
                return (
                  <tr key={row.gameweek}>
                    <td>
                      {row.gameweek}
                      {row.chip && <span className="muted"> · {row.chip}</span>}
                    </td>
                    <td className="num">{row.actual ?? "—"}</td>
                    <td className="num">{row.global_average ?? "—"}</td>
                    <td
                      className="num"
                      style={{ color: diff == null ? undefined : diff >= 0 ? "var(--gain)" : "var(--flag)" }}
                    >
                      {diff == null ? "—" : `${diff >= 0 ? "+" : ""}${diff.toFixed(0)}`}
                    </td>
                    <td className="num">{row.projected ?? "—"}</td>
                    <td className="num">
                      {row.overall_rank ? row.overall_rank.toLocaleString("en-GB") : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {!me.fpl_entry_id && (
          <div className="panel">
            <h3>Add a score</h3>
            <div className="field">
              <label htmlFor="result-gw">Gameweek</label>
              <input
                id="result-gw"
                type="number"
                min={1}
                max={38}
                value={entryGw}
                onChange={(e) => setEntryGw(e.target.value)}
              />
            </div>
            <div className="field">
              <label htmlFor="result-pts">Points scored</label>
              <input
                id="result-pts"
                type="number"
                min={0}
                value={entryPoints}
                onChange={(e) => setEntryPoints(e.target.value)}
              />
            </div>
            <button className="btn small" onClick={submitResult} type="button">
              Save score
            </button>
          </div>
        )}
      </div>
    </>
  );
}
