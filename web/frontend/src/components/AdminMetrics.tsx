import { useCallback, useEffect, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, ApiError } from "../lib/api";
import type { AdminMetrics } from "../lib/types";

const WINDOWS: { id: string; label: string }[] = [
  { id: "1h", label: "Last hour" },
  { id: "6h", label: "6 hours" },
  { id: "24h", label: "24 hours" },
  { id: "7d", label: "7 days" },
];

/** A number and what it means, with the small print underneath. */
function Tile({
  value,
  label,
  hint,
}: {
  value: string | number;
  label: string;
  hint?: string;
}) {
  return (
    <div className="metric-tile">
      <strong>{value}</strong>
      <span>{label}</span>
      {hint && <span className="hint">{hint}</span>}
    </div>
  );
}

function ago(iso: string): string {
  const seconds = Math.max(0, (Date.now() - Date.parse(iso)) / 1000);
  if (seconds < 90) return "just now";
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}

function seconds(value: number | null): string {
  if (value === null) return "—";
  if (value < 90) return `${value}s`;
  return `${(value / 60).toFixed(1)}m`;
}

export function AdminMetricsPanel() {
  const [window_, setWindow] = useState("24h");
  const [data, setData] = useState<AdminMetrics | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const load = useCallback(async (id: string) => {
    setLoading(true);
    try {
      setData(await api.adminMetrics(id));
      setError("");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(window_);
  }, [load, window_]);

  if (error) return <div className="notice bad">{error}</div>;
  if (!data) return <p className="muted">Reading the numbers…</p>;

  const { totals } = data;
  const failRate =
    totals.runs_finished + totals.runs_failed > 0
      ? Math.round(
          (totals.runs_failed / (totals.runs_finished + totals.runs_failed)) *
            100,
        )
      : 0;

  return (
    <>
      <div className="metric-head">
        <div className="metric-windows">
          {WINDOWS.map((option) => (
            <button
              key={option.id}
              type="button"
              className={option.id === window_ ? "chip on" : "chip"}
              onClick={() => setWindow(option.id)}
            >
              {option.label}
            </button>
          ))}
        </div>
        <button
          className="link-button"
          type="button"
          onClick={() => void load(window_)}
        >
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      <div className="metric-tiles">
        <Tile
          value={totals.visitors}
          label="Unique visitors"
          hint={`${totals.members} signed in, ${totals.guests} guests`}
        />
        <Tile
          value={totals.sessions}
          label="Visits"
          hint="A gap of 30 minutes starts a new one"
        />
        <Tile
          value={totals.runs}
          label="Runs started"
          hint={`${totals.plans} plans`}
        />
        <Tile
          value={seconds(totals.run_seconds_median)}
          label="Typical run"
          hint={`95th percentile ${seconds(totals.run_seconds_p95)}`}
        />
        <Tile
          value={seconds(totals.wait_seconds_p95)}
          label="Worst queue wait"
          hint="Above a minute means the worker is behind"
        />
        <Tile
          value={`${failRate}%`}
          label="Runs failing"
          hint={`${totals.runs_rejected} turned away`}
        />
      </div>

      <div className="metric-chart">
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart
            data={data.series}
            margin={{ top: 8, right: 8, bottom: 0, left: -18 }}
          >
            <defs>
              <linearGradient id="gVisitors" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="0%"
                  stopColor="var(--floodlight)"
                  stopOpacity={0.55}
                />
                <stop
                  offset="100%"
                  stopColor="var(--floodlight)"
                  stopOpacity={0.04}
                />
              </linearGradient>
              <linearGradient id="gRuns" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--gain)" stopOpacity={0.5} />
                <stop
                  offset="100%"
                  stopColor="var(--gain)"
                  stopOpacity={0.04}
                />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="var(--line)" vertical={false} />
            <XAxis
              dataKey="label"
              tick={{ fill: "var(--fade)", fontSize: 11 }}
              stroke="var(--line)"
              minTickGap={28}
            />
            <YAxis
              allowDecimals={false}
              tick={{ fill: "var(--fade)", fontSize: 11 }}
              stroke="var(--line)"
              width={38}
            />
            <Tooltip
              contentStyle={{
                background: "var(--panel)",
                border: "1px solid var(--line)",
                borderRadius: 10,
                fontSize: 12,
              }}
              labelStyle={{ color: "var(--fade)" }}
            />
            <Area
              type="monotone"
              dataKey="visitors"
              name="Visitors"
              stroke="var(--floodlight)"
              fill="url(#gVisitors)"
              strokeWidth={2}
            />
            <Area
              type="monotone"
              dataKey="runs"
              name="Runs"
              stroke="var(--gain)"
              fill="url(#gRuns)"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
        <p className="hint">
          Visitors and runs, in {data.bucket_minutes}-minute buckets.
        </p>
      </div>

      <h4>Who's been here</h4>
      {data.people.length === 0 ? (
        <p className="muted">Nobody in this window.</p>
      ) : (
        <div className="calc-scroll">
          <table className="calc-table">
            <thead>
              <tr>
                <th>Visitor</th>
                <th className="num">Visits</th>
                <th className="num">Runs</th>
                <th className="num">Last seen</th>
              </tr>
            </thead>
            <tbody>
              {data.people.map((person) => (
                <tr key={person.visitor}>
                  <td>
                    <span className="who-cell">
                      {person.who}
                      {person.guest && <span className="pill">guest</span>}
                    </span>
                    {person.detail && (
                      <div className="muted small">{person.detail}</div>
                    )}
                  </td>
                  <td className="num">{person.sessions}</td>
                  <td className="num">{person.runs}</td>
                  <td className="num muted">{ago(person.last_seen)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="hint">
        {data.ip_mode === "none"
          ? "Guests are counted but not identified. Set VISITOR_IP_MODE to " +
            "truncated or full to show where they came from."
          : data.ip_mode === "truncated"
            ? "Guests are shown as the network they came from, not their " +
              "address. VISITOR_IP_MODE=full shows the address itself."
            : "Full addresses are being stored for guests " +
              "(VISITOR_IP_MODE=full)."}
      </p>

      {data.blocked.length > 0 && (
        <>
          <h4>Where guests hit the wall</h4>
          <p className="muted">
            What people wanted and couldn't have — the best guide to what's
            worth opening up.
          </p>
          <div className="stat-rows">
            {data.blocked.slice(0, 6).map((row) => (
              <div key={row.feature}>
                <span>{row.feature}</span>
                <span>{row.count}</span>
              </div>
            ))}
          </div>
        </>
      )}

      <h4>Activity</h4>
      <div className="stat-rows">
        {data.activity.slice(0, 10).map((row) => (
          <div key={row.kind}>
            <span>{row.kind.replace(/_/g, " ")}</span>
            <span>{row.count}</span>
          </div>
        ))}
      </div>
      {data.dropped > 0 && (
        <p className="hint">
          {data.dropped} events dropped since the last restart — the metrics
          queue filled up, which means the app was busier than it could
          record.
        </p>
      )}
    </>
  );
}
