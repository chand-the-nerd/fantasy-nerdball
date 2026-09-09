import type { SquadOption } from "../lib/types";

/**
 * The squads the optimiser ranked, as a row of buttons under the pitch.
 *
 * Selecting one previews it; the pitch above redraws and an Activate button
 * appears. Nothing is committed until that is pressed, so a manager can look
 * through all six without changing what their squad is.
 */

interface Props {
  options: SquadOption[];
  /** The option currently in force. */
  activeKey: string;
  /** The option being looked at, which may not be the active one. */
  previewKey: string;
  onPreview: (key: string) => void;
  onActivate: (key: string) => void;
  activating: boolean;
  disabled?: boolean;
}

function difference(option: SquadOption, best: number): string {
  const gap = option.projected_points - best;
  if (Math.abs(gap) < 0.05) return "level";
  return `${gap > 0 ? "+" : "−"}${Math.abs(gap).toFixed(1)}`;
}

export function SquadOptions({
  options,
  activeKey,
  previewKey,
  onPreview,
  onActivate,
  activating,
  disabled = false,
}: Props) {
  if (options.length < 2) return null;

  const best = Math.max(...options.map((option) => option.projected_points));
  const preview = options.find((option) => option.key === previewKey);
  const showActivate = preview != null && preview.key !== activeKey;

  return (
    <div className="option-strip">
      <div className="option-head">
        <strong>Other ways to line up</strong>
        <span className="option-key">
          <span className="swatch" aria-hidden="true" />
          Recommended
        </span>
      </div>


      <div className="option-row">
        {options.map((option) => {
          const active = option.key === activeKey;
          const shown = option.key === previewKey;
          const net = option.projected_points - option.penalty_points;
          return (
            <button
              key={option.key}
              type="button"
              className={[
                "option-card",
                option.kind === "previous" ? "hold" : "",
                option.recommended ? "recommended" : "",
                shown ? "shown" : "",
                active ? "active" : "",
              ]
                .filter(Boolean)
                .join(" ")}
              onClick={() => onPreview(option.key)}
              disabled={disabled}
              aria-pressed={shown}
            >
              <span className="option-label">{option.label}</span>
              <span className="option-points">
                {option.projected_points.toFixed(1)}
              </span>
              {option.nerdball_score != null && (
                <span className="option-score">
                  {option.nerdball_score.toFixed(1)} nerdball
                </span>
              )}
              <span className="option-note">
                {option.penalty_points > 0
                  ? `${net.toFixed(1)} after the −${option.penalty_points} hit`
                  : difference(option, best)}
              </span>
              <span className="option-foot">
                {option.transfers_made === 0
                  ? "no transfers"
                  : `${option.transfers_made} transfer${
                      option.transfers_made === 1 ? "" : "s"
                    }`}
                {/* How far this sits from the recommendation is what decides
                    whether it's worth opening, and it isn't the transfer
                    count — two options can each make two transfers and still
                    be the same squad but for one player. */}
              </span>
              {active && <span className="option-flag">Active</span>}
            </button>
          );
        })}
      </div>

      <p className="muted option-legend">
        The big number is <strong>projected points</strong>: what the eleven
        should score this gameweek. The small one is the{" "}
        <strong>nerdball score</strong>, the model&rsquo;s own rating of the
        squad from form, history and fixtures across your look-ahead.
      </p>
      
      <p>
        Squads are picked on nerdball score, because a side has to hold up beyond
        the weekend; the projection only says how the next gameweek looks. The
        recommended option isn't always the highest score — it's the squad where
        every transfer was worth making.
      </p>

      {preview && (
        <div className="option-foot-row">
          <p className="muted">
            {preview.recommended
              ? preview.kind === "previous"
                ? "The optimiser's pick: no move clears the threshold this week."
                : "The optimiser's pick."
              : preview.kind === "previous"
                ? "Last week's fifteen, best eleven starting."
                : `${preview.formation}, £${preview.squad_value.toFixed(1)}m spent.`}
          </p>
          {showActivate ? (
            <button
              className="btn"
              type="button"
              onClick={() => onActivate(preview.key)}
              disabled={activating || disabled}
            >
              {activating ? "Activating…" : `Activate ${preview.label}`}
            </button>
          ) : (
            <span className="option-current">Currently active</span>
          )}
        </div>
      )}
    </div>
  );
}
