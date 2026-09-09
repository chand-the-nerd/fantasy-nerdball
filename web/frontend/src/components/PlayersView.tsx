import { useEffect, useState } from "react";
import { GuestLock } from "./GuestLock";
import { PlayerActions, PlayerActionsDialog } from "./PlayerActions";
import { PlayerPicker, type PoolPlayer, availability } from "./PlayerPicker";
import { api, ApiError } from "../lib/api";
import { useGuest } from "../lib/guest";

type Mode = "best" | "differentials" | "lookup";

const POSITION_LABELS: Record<string, string> = {
  GK: "Goalkeepers",
  DEF: "Defenders",
  MID: "Midfielders",
  FWD: "Forwards",
};

interface Ranked {
  id: number;
  name: string;
  position: string;
  team: string;
  price: number;
  score: number;
  projected_points: number | null;
  form: number | null;
  fixture_difficulty: number | null;
  ownership: number | null;
  status: string;
}

function RankTable({
  rows,
  onPick,
}: {
  rows: Ranked[];
  onPick: (row: Ranked) => void;
}) {
  if (rows.length === 0) {
    return <p className="muted">Nothing clears the filter here.</p>;
  }
  return (
    <table className="calc-table rank-table">
      <thead>
        <tr>
          <th>Player</th>
          <th>Club</th>
          <th className="num" title="The model's own score, which is what it ranks on">
            Score
          </th>
          <th className="num">Price</th>
          <th className="num" title="Average difficulty over your look-ahead window">
            Fixtures
          </th>
          <th className="num" title="Percentage of FPL managers who own them">
            Owned
          </th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const state = availability(row.status);
          return (
            <tr
              key={row.id}
              className="is-clickable"
              tabIndex={0}
              role="button"
              title={`Force ${row.name} in, or add them to your avoid list`}
              onClick={() => onPick(row)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onPick(row);
                }
              }}
            >
              <td>
                <span className="calc-name">
                  <span className={`dot tone-${state.tone}`} />
                  {row.name}
                </span>
              </td>
              <td className="muted">{row.team}</td>
              <td className="num accent">{row.score?.toFixed(2) ?? "—"}</td>
              <td className="num">£{row.price?.toFixed(1)}m</td>
              <td className="num">{row.fixture_difficulty ?? "—"}</td>
              <td className="num">
                {row.ownership != null ? `${row.ownership}%` : "—"}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/** "(GW4–GW7)", so the window is stated rather than worked out. */
function fixtureWindow(data: any): string {
  const from = Number(data.gameweek);
  const span = Number(data.look_ahead);
  if (!from || !span || span < 2) return "";
  return ` (GW${from}–GW${from + span - 1})`;
}

function Ranked({ mode }: { mode: "best" | "differentials" }) {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState("");
  const [picked, setPicked] = useState<Ranked | null>(null);

  useEffect(() => {
    const call = mode === "best" ? api.bestPlayers() : api.differentialPlayers();
    call
      .then(setData)
      .catch((err) => setError(err instanceof ApiError ? err.message : String(err)));
  }, [mode]);

  if (error) return <div className="notice bad">{error}</div>;
  if (!data) return <p className="muted">Loading…</p>;

  if (!data.available) {
    return (
      <div className="panel">
        <h3>Nothing scored yet</h3>
        <p className="muted">{data.reason}</p>
      </div>
    );
  }

  return (
    <>
      <p className="muted">
        {mode === "best" ? (
          <>
            Ranked by your own weights, looking ahead from gameweek{" "}
            {data.gameweek} over {data.look_ahead} gameweek
            {data.look_ahead === 1 ? "" : "s"}
            {fixtureWindow(data)}.
          </>
        ) : (
          <>
            The same ranking, restricted to players owned by under{" "}
            {data.max_ownership}% of managers.
          </>
        )}
      </p>

      {data.stale && (
        <div className="notice">
          These scores come from your gameweek {data.gameweek} run, so the
          fixtures behind them start at gameweek {data.gameweek} rather than at
          gameweek {data.current_gameweek}, which is the one you're picking for.
          Run the optimiser for gameweek {data.current_gameweek} to rank on the
          right window.
        </div>
      )}

      {mode === "differentials" && data.thin_positions?.length > 0 && (
        <div className="notice">
          Nothing under {data.max_ownership}% ownership scores well enough in:{" "}
          {data.thin_positions.map((p: string) => POSITION_LABELS[p]).join(", ")}.
          That usually means the good options there are already popular.
        </div>
      )}

      <div className="setup-grid">
        {Object.keys(POSITION_LABELS).map((position) => (
          <div className="panel col-half" key={position}>
            <h3>{POSITION_LABELS[position]}</h3>
            <div className="calc-scroll">
              <RankTable
                rows={data.positions[position] ?? []}
                onPick={setPicked}
              />
            </div>
          </div>
        ))}
      </div>

      {picked && (
        <PlayerActionsDialog
          name={picked.name}
          position={picked.position}
          subtitle={`${picked.position} · ${picked.team} · £${picked.price?.toFixed(
            1,
          )}m · score ${picked.score?.toFixed(2) ?? "—"}`}
          onClose={() => setPicked(null)}
        />
      )}
    </>
  );
}

function Stat({ label, value, hint }: { label: string; value: any; hint?: string }) {
  return (
    <div title={hint}>
      <span>{label}</span>
      <span>{value ?? "—"}</span>
    </div>
  );
}

function Lookup() {
  const [pool, setPool] = useState<PoolPlayer[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [detail, setDetail] = useState<any>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.players().then((d) => setPool(d.players)).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (selected.length === 0) {
      setDetail(null);
      return;
    }
    const match = pool.find(
      (p) => p.name.toLowerCase() === selected[selected.length - 1].toLowerCase(),
    );
    if (!match) return;
    api
      .playerDetail(match.id)
      .then(setDetail)
      .catch((err) => setError(err instanceof ApiError ? err.message : String(err)));
  }, [selected, pool]);

  return (
    <>
      <div className="panel">
        <h3>Look up a player</h3>
        <p className="muted" style={{ marginTop: -6 }}>
          Underlying numbers and upcoming fixtures, straight from FPL. Works
          without having run the optimiser.
        </p>
        <PlayerPicker
          pool={pool}
          selected={selected}
          limit={1}
          onChange={(names) => setSelected(names.slice(-1))}
          placeholder="Search any player"
        />
        {detail && (
          <PlayerActions name={detail.name} position={detail.position} />
        )}
      </div>

      {error && <div className="notice bad">{error}</div>}

      {detail && (
        <div className="setup-grid" style={{ marginTop: 16 }}>
          <div className="panel col-half">
            <h3>
              {detail.name}{" "}
              <span className="muted">
                {detail.position} · {detail.team} · £{detail.price.toFixed(1)}m
              </span>
            </h3>
            {detail.news && <div className="notice bad">{detail.news}</div>}
            <div className="stat-rows">
              <Stat label="Total points" value={detail.fpl.total_points} />
              <Stat label="Points per game" value={detail.fpl.points_per_game} />
              <Stat label="Form" value={detail.fpl.form} />
              <Stat label="Minutes" value={detail.fpl.minutes} />
              <Stat label="Starts" value={detail.fpl.starts} />
              <Stat label="Goals" value={detail.fpl.goals} />
              <Stat label="Assists" value={detail.fpl.assists} />
              <Stat label="Clean sheets" value={detail.fpl.clean_sheets} />
              <Stat label="Bonus" value={detail.fpl.bonus} />
              <Stat label="Owned by" value={detail.fpl.ownership ? `${detail.fpl.ownership}%` : "—"} />
            </div>
          </div>

          <div className="panel col-half">
            <h3>Underlying numbers</h3>
            <p className="muted" style={{ marginTop: -6 }}>
              What the chances were worth, regardless of what went in.
            </p>
            <div className="stat-rows">
              <Stat label="Expected goals" value={detail.underlying.xg} hint="Season total xG" />
              <Stat label="Expected assists" value={detail.underlying.xa} />
              <Stat
                label="Expected involvements"
                value={detail.underlying.xgi}
                hint="Goals plus assists, expected"
              />
              <Stat
                label="Expected conceded"
                value={detail.underlying.xgc}
                hint="Expected goals against while they were on the pitch"
              />
              <Stat label="xG per 90" value={detail.underlying.xg_per_90} />
              <Stat label="xA per 90" value={detail.underlying.xa_per_90} />
              <Stat label="xGC per 90" value={detail.underlying.xgc_per_90} />
            </div>
            {detail.fpl.goals != null && detail.underlying.xg != null && detail.underlying.xg > 0 && (
              <p className="hint" style={{ marginTop: 10 }}>
                {detail.fpl.goals > detail.underlying.xg
                  ? `Scoring above the chances (${detail.fpl.goals} from ${detail.underlying.xg} xG), which tends not to last.`
                  : `Scoring below the chances (${detail.fpl.goals} from ${detail.underlying.xg} xG), which often corrects.`}
              </p>
            )}
          </div>

          <div className="panel col-half">
            <h3>Next {detail.fixtures.length} fixtures</h3>
            {detail.fixtures.length === 0 ? (
              <p className="muted">No upcoming fixtures listed.</p>
            ) : (
              <table className="calc-table fixtures-table">
                <thead>
                  <tr>
                    <th>GW</th>
                    <th>Opponent</th>
                    <th>Venue</th>
                    <th className="num">Difficulty</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.fixtures.map((f: any, i: number) => (
                    <tr key={i}>
                      <td>{f.gameweek}</td>
                      <td>{f.opponent}</td>
                      <td className="muted">{f.venue}</td>
                      <td className="num">
                        <span className={`fdr fdr-${f.difficulty}`}>{f.difficulty}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          {detail.model && (
            <div className="panel col-half">
              <h3>How your model rates them</h3>
              <div className="stat-rows">
                <Stat label="Model score" value={detail.model.score} />
                <Stat label="Projected points" value={detail.model.projected_points} />
                <Stat label="Historic points per game" value={detail.model.historic_ppg} />
                <Stat label="Fixture difficulty" value={detail.model.fixture_difficulty} />
                <Stat label="Start rate" value={detail.model.start_rate ? `${detail.model.start_rate}%` : "—"} />
                <Stat label="Minutes per game" value={detail.model.minutes_per_game} />
                <Stat label="xG multiplier" value={detail.model.xg_modifier} />
                <Stat label="Team modifier" value={detail.model.team_modifier} />
              </div>
            </div>
          )}
        </div>
      )}
    </>
  );
}

/** The lookup panel as a guest sees it: present, blurred, inert. */
function LockedLookup() {
  return (
    <GuestLock note="Season totals, expected goals and upcoming fixtures for any player in the game.">
      <div className="panel">
        <h3>Look up a player</h3>
        <p className="muted" style={{ marginTop: -6 }}>
          Underlying numbers and upcoming fixtures, straight from FPL.
        </p>
        <div className="stat-rows">
          <div>
            <span>Total points</span>
            <span>—</span>
          </div>
          <div>
            <span>Expected goals</span>
            <span>—</span>
          </div>
          <div>
            <span>Expected assists</span>
            <span>—</span>
          </div>
          <div>
            <span>Next fixtures</span>
            <span>—</span>
          </div>
        </div>
      </div>
    </GuestLock>
  );
}

export function PlayersView() {
  const [mode, setMode] = useState<Mode>("best");
  const guest = useGuest();

  return (
    <>
      <div className="topbar">
        <div>
          <h1>Players</h1>
          <span className="when">The model's player database. Click a player to add or remove them from your squad.</span>
        </div>
        <div className="segmented">
          {(
            [
              ["best", "Best picks"],
              ["differentials", "Differentials"],
              ["lookup", "Look up"],
            ] as [Mode, string][]
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              className={mode === id ? "is-active" : ""}
              onClick={() => setMode(id)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {mode === "lookup" ? (
        guest ? (
          <LockedLookup />
        ) : (
          <Lookup />
        )
      ) : (
        <Ranked mode={mode} />
      )}
    </>
  );
}
