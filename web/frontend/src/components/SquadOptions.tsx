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
          Green is the optimiser&rsquo;s recommendation
        </span>
      </div>

      <p className="muted option-blurb">
        Ranked by projected points. Nothing changes until you activate one.
      </p>

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
                {!option.recommended && option.differs_by > 0 &&
                  ` · ${option.differs_by} different`}
              </span>
              {active && <span className="option-flag">Active</span>}
            </button>
          );
        })}
      </div>

      {preview && (
        <div className="option-foot-row">
          <p className="muted">
            {preview.recommended
              ? preview.kind === "previous"
                ? "The optimiser's pick: no move this week clears the improvement threshold."
                : "The optimiser's pick, left to itself."
              : preview.kind === "previous"
                ? "Last week's fifteen kept whole, with the best eleven of them starting."
                : `${preview.formation}, £${preview.squad_value.toFixed(1)}m spent, ` +
                  `${preview.differs_by} different to the recommendation.`}
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
