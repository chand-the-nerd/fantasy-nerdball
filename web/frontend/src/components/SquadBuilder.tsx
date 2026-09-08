import { useEffect, useMemo, useState } from "react";
import { PitchMarkings } from "./Pitch";
import { availability, type PoolPlayer } from "./PlayerPicker";
import { api, ApiError } from "../lib/api";
import { normalise } from "../lib/text";

const POSITIONS = ["GK", "DEF", "MID", "FWD"] as const;
type Position = (typeof POSITIONS)[number];

/** An FPL squad is always two, five, five and three. */
const SQUAD_SHAPE: Record<Position, number> = { GK: 2, DEF: 5, MID: 5, FWD: 3 };

const FORMATIONS = ["3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"];

const POSITION_LABELS: Record<Position, string> = {
  GK: "Goalkeeper",
  DEF: "Defender",
  MID: "Midfielder",
  FWD: "Forward",
};

/** How many of each position start, for a given formation. */
function startingCounts(formation: string): Record<Position, number> {
  const [def, mid, fwd] = formation.split("-").map(Number);
  return { GK: 1, DEF: def, MID: mid, FWD: fwd };
}

type Squad = Record<Position, (PoolPlayer | null)[]>;

const EMPTY: Squad = {
  GK: [null, null],
  DEF: [null, null, null, null, null],
  MID: [null, null, null, null, null],
  FWD: [null, null, null],
};

interface SlotRef {
  position: Position;
  index: number;
}

function Slot({
  player,
  position,
  onPick,
  onClear,
}: {
  player: PoolPlayer | null;
  position: Position;
  onPick: () => void;
  onClear: () => void;
}) {
  if (!player) {
    return (
      <button
        type="button"
        className="shirt is-empty"
        onClick={onPick}
        title={`Add a ${POSITION_LABELS[position].toLowerCase()}`}
      >
        <span className="band">
          <span className="name">{position}</span>
        </span>
        <span className="plus">+</span>
      </button>
    );
  }

  const state = availability(player.status);
  return (
    <button
      type="button"
      className={`shirt is-filled${state.tone === "out" ? " is-doubtful" : ""}`}
      onClick={onPick}
      title={`${player.name} · ${player.team} · £${player.price.toFixed(1)}m · ${
        state.word
      }\nClick to replace`}
    >
      <span className="band">
        <span className="name">{player.name}</span>
      </span>
      <span
        className="slot-clear"
        role="button"
        aria-label={`Remove ${player.name}`}
        onClick={(event) => {
          event.stopPropagation();
          onClear();
        }}
      >
        ×
      </span>
      <span className="meta">{player.team}</span>
      <span className="price">£{player.price.toFixed(1)}m</span>
    </button>
  );
}

function PlayerSearch({
  pool,
  position,
  taken,
  onChoose,
  onClose,
}: {
  pool: PoolPlayer[];
  position: Position;
  taken: Set<number>;
  onChoose: (player: PoolPlayer) => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const matches = useMemo(() => {
    const q = normalise(query.trim());
    return pool
      .filter((player) => player.position === position && !taken.has(player.id))
      .filter(
        (player) =>
          !q ||
          normalise(player.name).includes(q) ||
          normalise(player.full_name).includes(q) ||
          normalise(player.team).includes(q),
      )
      .slice(0, 40);
  }, [pool, position, taken, query]);

  return (
    <div
      className="overlay"
      role="dialog"
      aria-modal="true"
      aria-label={`Pick a ${POSITION_LABELS[position]}`}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="admin-panel player-dialog">
        <h2>Pick a {POSITION_LABELS[position].toLowerCase()}</h2>
        <div className="field">
          <input
            type="text"
            autoFocus
            autoComplete="off"
            placeholder="Search by name or club"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </div>
        <ul className="picker-list is-static" role="listbox">
          {matches.map((player) => {
            const state = availability(player.status);
            return (
              <li key={player.id} role="option" aria-selected={false}>
                <button type="button" onClick={() => onChoose(player)}>
                  <span className={`dot tone-${state.tone}`} />
                  <span className="picker-name">{player.name}</span>
                  <span className="picker-meta">{player.team}</span>
                  <span className="picker-price">£{player.price.toFixed(1)}m</span>
                </button>
              </li>
            );
          })}
          {matches.length === 0 && (
            <li className="muted" style={{ padding: "8px 4px" }}>
              Nobody matches that.
            </li>
          )}
        </ul>
        <div className="admin-actions">
          <button className="btn quiet small" type="button" onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Entering last week's side by hand, for anyone who hasn't linked an FPL team
 * or whose real side isn't what they want the optimiser to transfer from.
 *
 * Laid out on the same pitch as a saved squad, so where a player will sit is
 * obvious before they're picked rather than after.
 */
export function SquadBuilder({
  gameweek,
  onSaved,
  onCancel,
}: {
  gameweek: number;
  onSaved: () => void;
  onCancel: () => void;
}) {
  const [pool, setPool] = useState<PoolPlayer[]>([]);
  const [squad, setSquad] = useState<Squad>(EMPTY);
  const [formation, setFormation] = useState("4-4-2");
  const [picking, setPicking] = useState<SlotRef | null>(null);
  const [bank, setBank] = useState("0.0");
  const [applyBudget, setApplyBudget] = useState(true);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api
      .players()
      .then((data) => setPool(data.players))
      .catch((err) => setError(err instanceof ApiError ? err.message : String(err)));
  }, []);

  const chosen = useMemo(
    () => POSITIONS.flatMap((position) => squad[position]).filter(Boolean) as PoolPlayer[],
    [squad],
  );
  const taken = useMemo(() => new Set(chosen.map((p) => p.id)), [chosen]);
  const value = chosen.reduce((sum, player) => sum + player.price, 0);
  const counts = startingCounts(formation);

  // Three per club is an FPL rule, so it's worth saying before the save fails.
  const clubProblem = useMemo(() => {
    const perClub = new Map<string, number>();
    for (const player of chosen) {
      perClub.set(player.team, (perClub.get(player.team) ?? 0) + 1);
    }
    for (const [team, count] of perClub) {
      if (count > 3) return `${count} players from ${team}. FPL allows 3 per club.`;
    }
    return "";
  }, [chosen]);

  const set = (ref: SlotRef, player: PoolPlayer | null) =>
    setSquad((current) => {
      const next = { ...current, [ref.position]: [...current[ref.position]] };
      next[ref.position][ref.index] = player;
      return next;
    });

  const save = async () => {
    setError("");
    setBusy(true);
    try {
      const starting: number[] = [];
      const all: number[] = [];
      for (const position of POSITIONS) {
        squad[position].forEach((player, index) => {
          if (!player) return;
          all.push(player.id);
          if (index < counts[position]) starting.push(player.id);
        });
      }
      await api.manualSquad({
        gameweek,
        player_ids: all,
        starting_ids: starting,
        bank: Number(bank) || 0,
        apply_budget: applyBudget,
      });
      onSaved();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const row = (position: Position, from: number, to: number) =>
    squad[position].slice(from, to).map((player, offset) => {
      const ref = { position, index: from + offset };
      return (
        <Slot
          key={`${position}-${from + offset}`}
          player={player}
          position={position}
          onPick={() => setPicking(ref)}
          onClear={() => set(ref, null)}
        />
      );
    });

  const complete = chosen.length === 15;

  return (
    <>
      <div className="builder-head">
        <div>
          <h3>Enter your gameweek {gameweek} squad</h3>
          <p className="muted" style={{ margin: "2px 0 0" }}>
            Tap a shirt to pick a player. The optimiser transfers from this side,
            so it should be the fifteen you actually had.
          </p>
        </div>
        <label className="gw-picker">
          <span>Formation</span>
          <select
            value={formation}
            onChange={(event) => setFormation(event.target.value)}
          >
            {FORMATIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
      </div>

      {error && <div className="notice bad">{error}</div>}
      {clubProblem && <div className="notice bad">{clubProblem}</div>}

      <div className="pitch-frame">
        <div className="pitch">
          <PitchMarkings />
          <div className="lines">
            <div className="line-row">{row("GK", 0, 1)}</div>
            <div className="line-row">{row("DEF", 0, counts.DEF)}</div>
            <div className="line-row">{row("MID", 0, counts.MID)}</div>
            <div className="line-row">{row("FWD", 0, counts.FWD)}</div>
          </div>
        </div>

        <div className="bench">
          <div className="bench-head">
            <strong>Bench</strong>
            <span>
              {chosen.length} of 15 picked · £{value.toFixed(1)}m
            </span>
          </div>
          <div className="line-row">
            {row("GK", 1, 2)}
            {row("DEF", counts.DEF, SQUAD_SHAPE.DEF)}
            {row("MID", counts.MID, SQUAD_SHAPE.MID)}
            {row("FWD", counts.FWD, SQUAD_SHAPE.FWD)}
          </div>
        </div>
      </div>

      <div className="panel builder-foot">
        <div className="field">
          <label htmlFor="builder-bank">In the bank</label>
          <input
            id="builder-bank"
            type="number"
            step="0.1"
            min={0}
            inputMode="decimal"
            value={bank}
            onChange={(event) => setBank(event.target.value)}
          />
          <span className="hint">Money not spent, from the FPL site.</span>
        </div>

        <label className="toggle">
          <input
            type="checkbox"
            checked={applyBudget}
            onChange={(event) => setApplyBudget(event.target.checked)}
          />
          <span className="copy">
            <strong>Set my budget from this</strong>
            <span>
              £{(value + (Number(bank) || 0)).toFixed(1)}m — squad value plus the
              bank.
            </span>
          </span>
        </label>

        <div className="builder-actions">
          <button
            className="btn"
            type="button"
            onClick={save}
            disabled={!complete || busy || Boolean(clubProblem)}
          >
            {busy
              ? "Saving…"
              : complete
                ? `Save gameweek ${gameweek} squad`
                : `${15 - chosen.length} still to pick`}
          </button>
          <button className="btn quiet" type="button" onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>

      {picking && (
        <PlayerSearch
          pool={pool}
          position={picking.position}
          taken={taken}
          onChoose={(player) => {
            set(picking, player);
            setPicking(null);
          }}
          onClose={() => setPicking(null)}
        />
      )}
    </>
  );
}
