import { useState } from "react";

export interface Reference {
  positions: string[];
  weight_keys: string[];
  default_weights: Record<string, Record<string, number>>;
  teams: string[];
  squad_limits: Record<string, number>;
}

type Weights = Record<string, Record<string, number>>;

const WEIGHT_LABELS: Record<string, string> = {
  form: "Form",
  historic: "History",
  difficulty: "Fixtures",
};

const POSITION_LABELS: Record<string, string> = {
  GK: "Goalkeepers",
  DEF: "Defenders",
  MID: "Midfielders",
  FWD: "Forwards",
};

/**
 * Editor for POSITION_SCORING_WEIGHTS. Each position splits a player's core
 * score across current form, historic points per game and upcoming fixture
 * difficulty. The three have to total 1.00, so the running total is shown as
 * you type and the save is blocked until it balances.
 */
export function WeightsEditor({
  reference,
  weights,
  onChange,
}: {
  reference: Reference;
  weights: Weights;
  onChange: (weights: Weights) => void;
}) {
  const effective = (position: string) =>
    weights[position] ?? reference.default_weights[position] ?? {
      form: 0.5,
      historic: 0.25,
      difficulty: 0.25,
    };

  const set = (position: string, key: string, value: number) => {
    onChange({
      ...weights,
      [position]: { ...effective(position), [key]: value },
    });
  };

  const balance = (position: string) => {
    const row = effective(position);
    const total = reference.weight_keys.reduce((sum, k) => sum + (row[k] ?? 0), 0);
    onChange({
      ...weights,
      [position]: Object.fromEntries(
        reference.weight_keys.map((k) => [
          k,
          total === 0 ? 1 / reference.weight_keys.length : Number(((row[k] ?? 0) / total).toFixed(3)),
        ]),
      ),
    });
  };

  return (
    <div className="panel">
      <h3>How players are scored</h3>
      <p className="muted" style={{ marginTop: -6 }}>
        Each position splits a player's score across these three. They have to
        total 1.00.
      </p>

      <table style={{ marginTop: 6 }}>
        <thead>
          <tr>
            <th />
            {reference.weight_keys.map((key) => (
              <th key={key} className="num">
                {WEIGHT_LABELS[key] ?? key}
              </th>
            ))}
            <th className="num">Total</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {reference.positions.map((position) => {
            const row = effective(position);
            const total = reference.weight_keys.reduce((s, k) => s + (row[k] ?? 0), 0);
            const balanced = Math.abs(total - 1) <= 0.01;
            return (
              <tr key={position}>
                <td>{POSITION_LABELS[position] ?? position}</td>
                {reference.weight_keys.map((key) => (
                  <td key={key} className="num">
                    <input
                      className="cell-input"
                      type="number"
                      step="0.05"
                      min="0"
                      max="1"
                      value={row[key] ?? 0}
                      aria-label={`${position} ${key} weight`}
                      onChange={(e) => set(position, key, Number(e.target.value))}
                    />
                  </td>
                ))}
                <td
                  className="num"
                  style={{ color: balanced ? "var(--gain)" : "var(--flag)" }}
                >
                  {total.toFixed(2)}
                </td>
                <td className="num">
                  {!balanced && (
                    <button
                      className="link-button"
                      type="button"
                      onClick={() => balance(position)}
                    >
                      Balance
                    </button>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <p className="muted" style={{ marginTop: 12, marginBottom: 0 }}>
        Raising fixtures makes the model chase easy runs; raising history makes
        it trust proven players over hot streaks.
      </p>
    </div>
  );
}

/**
 * Forced picks, per position. The optimiser matches on lowercased display
 * name, so a shared surname forces whoever it finds first.
 */
export function ForcedEditor({
  reference,
  forced,
  onChange,
}: {
  reference: Reference;
  forced: Record<string, string[]>;
  onChange: (forced: Record<string, string[]>) => void;
}) {
  const [drafts, setDrafts] = useState<Record<string, string>>(() =>
    Object.fromEntries(
      reference.positions.map((p) => [p, (forced[p] ?? []).join(", ")]),
    ),
  );

  const commit = (position: string, text: string) => {
    setDrafts({ ...drafts, [position]: text });
    onChange({
      ...forced,
      [position]: text
        .split(",")
        .map((name) => name.trim())
        .filter(Boolean),
    });
  };

  return (
    <div className="panel">
      <h3>Players you always want</h3>
      <p className="muted" style={{ marginTop: -6 }}>
        The optimiser builds the rest of the squad around these.
      </p>
      {reference.positions.map((position) => {
        const count = (forced[position] ?? []).length;
        const limit = reference.squad_limits[position] ?? 5;
        const over = count > limit;
        return (
          <div className="field" key={position}>
            <label htmlFor={`forced-${position}`}>
              {POSITION_LABELS[position] ?? position}{" "}
              <span style={{ color: over ? "var(--flag)" : undefined }}>
                {count}/{limit}
              </span>
            </label>
            <input
              id={`forced-${position}`}
              type="text"
              value={drafts[position] ?? ""}
              placeholder="Separate names with commas"
              onChange={(e) => commit(position, e.target.value)}
            />
          </div>
        );
      })}
      <p className="hint">
        Forcing a whole line leaves the model almost nothing to optimise, and
        can make the budget infeasible.
      </p>
    </div>
  );
}

/**
 * Team modifiers. Below 1.0 marks a club down, above marks it up. This is the
 * hook for judgement the data can't carry — a new manager, a European run,
 * a defence that's about to regress.
 */
export function TeamModifierEditor({
  reference,
  modifiers,
  onChange,
}: {
  reference: Reference;
  modifiers: Record<string, number>;
  onChange: (modifiers: Record<string, number>) => void;
}) {
  const [open, setOpen] = useState(false);
  const adjusted = Object.entries(modifiers).filter(([, v]) => v !== 1);

  if (reference.teams.length === 0) {
    return (
      <div className="panel">
        <h3>Team adjustments</h3>
        <p className="muted">
          Club names load from the FPL API, which isn't responding right now.
          Reload the page to try again.
        </p>
      </div>
    );
  }

  return (
    <div className="panel">
      <h3>Team adjustments</h3>
      <p className="muted" style={{ marginTop: -6 }}>
        Below 1.00 marks a club down, above marks it up. For what the numbers
        can't know yet.
      </p>

      {adjusted.length > 0 && (
        <div className="stat-rows" style={{ marginBottom: 12 }}>
          {adjusted.map(([team, value]) => (
            <div key={team}>
              <span>{team}</span>
              <span style={{ color: value < 1 ? "var(--flag)" : "var(--gain)" }}>
                {value.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      )}

      <button
        className="link-button"
        type="button"
        onClick={() => setOpen((v) => !v)}
      >
        {open ? "Hide all clubs" : `Adjust clubs (${adjusted.length} changed)`}
      </button>

      {open && (
        <table style={{ marginTop: 12 }}>
          <tbody>
            {reference.teams.map((team) => (
              <tr key={team}>
                <td>{team}</td>
                <td className="num">
                  <input
                    className="cell-input"
                    type="number"
                    step="0.05"
                    min="0.5"
                    max="1.5"
                    aria-label={`${team} modifier`}
                    value={modifiers[team] ?? 1}
                    onChange={(e) =>
                      onChange({ ...modifiers, [team]: Number(e.target.value) })
                    }
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
