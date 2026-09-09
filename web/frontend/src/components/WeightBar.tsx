import { useCallback, useEffect, useRef, useState } from "react";

export interface Weights {
  form: number;
  historic: number;
  difficulty: number;
}

const SEGMENTS = [
  { key: "form" as const, label: "Form", colour: "var(--w-form)" },
  { key: "historic" as const, label: "History", colour: "var(--w-history)" },
  { key: "difficulty" as const, label: "Fixtures", colour: "var(--w-fixtures)" },
];

const round2 = (n: number) => Math.round(n * 100) / 100;

/**
 * A single bar split into three proportions by two draggable handles.
 *
 * The three weights always total exactly 1.00 because that is what the bar
 * represents — you're dividing a fixed quantity, not setting three numbers
 * that happen to need to add up. There is nothing to balance and no invalid
 * state to warn about.
 *
 * Internally the handles are cut points: a = form, b = form + historic.
 */
export function WeightBar({
  weights,
  onChange,
  label,
  locked = false,
}: {
  weights: Weights;
  onChange: (weights: Weights) => void;
  label: string;
  /** Read-only: the bar still shows the split, the handles don't move. */
  locked?: boolean;
}) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<0 | 1 | null>(null);

  const a = weights.form;
  const b = weights.form + weights.historic;

  const emit = useCallback(
    (nextA: number, nextB: number) => {
      const lo = Math.max(0, Math.min(1, nextA));
      const hi = Math.max(lo, Math.min(1, nextB));
      onChange({
        form: round2(lo),
        historic: round2(hi - lo),
        difficulty: round2(1 - hi),
      });
    },
    [onChange],
  );

  const fractionFromEvent = (clientX: number) => {
    const track = trackRef.current;
    if (!track) return 0;
    const rect = track.getBoundingClientRect();
    return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  };

  useEffect(() => {
    if (dragging === null) return;

    const move = (event: PointerEvent) => {
      const fraction = fractionFromEvent(event.clientX);
      if (dragging === 0) emit(Math.min(fraction, b), b);
      else emit(a, Math.max(fraction, a));
    };
    const stop = () => setDragging(null);

    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
    window.addEventListener("pointercancel", stop);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
    };
  }, [dragging, a, b, emit]);

  const onKeyDown = (handle: 0 | 1) => (event: React.KeyboardEvent) => {
    const step = event.shiftKey ? 0.1 : 0.01;
    let delta = 0;
    if (event.key === "ArrowLeft" || event.key === "ArrowDown") delta = -step;
    else if (event.key === "ArrowRight" || event.key === "ArrowUp") delta = step;
    else if (event.key === "Home") delta = -1;
    else if (event.key === "End") delta = 1;
    else return;

    event.preventDefault();
    if (handle === 0) emit(Math.min(a + delta, b), b);
    else emit(a, Math.max(b + delta, a));
  };

  const percent = (n: number) => `${Math.round(n * 100)}%`;

  return (
    <div className={`weight-row${locked ? " is-locked" : ""}`}>
      <div className="weight-head">
        <span className="weight-label">{label}</span>
        <span className="weight-readout">
          {SEGMENTS.map((segment) => (
            <span key={segment.key}>
              <i style={{ background: segment.colour }} />
              {segment.label} {percent(weights[segment.key])}
            </span>
          ))}
        </span>
      </div>

      <div className="weight-track" ref={trackRef}>
        <span
          className="weight-fill"
          style={{ left: 0, width: percent(a), background: "var(--w-form)" }}
        />
        <span
          className="weight-fill"
          style={{
            left: percent(a),
            width: percent(b - a),
            background: "var(--w-history)",
          }}
        />
        <span
          className="weight-fill"
          style={{
            left: percent(b),
            width: percent(1 - b),
            background: "var(--w-fixtures)",
          }}
        />

        {!locked &&
          ([0, 1] as const).map((handle) => {
          const value = handle === 0 ? a : b;
          return (
            <span
              key={handle}
              className={`weight-handle${dragging === handle ? " is-dragging" : ""}`}
              style={{ left: percent(value) }}
              role="slider"
              tabIndex={0}
              aria-label={
                handle === 0
                  ? `${label}: boundary between form and history`
                  : `${label}: boundary between history and fixtures`
              }
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(value * 100)}
              aria-valuetext={
                handle === 0
                  ? `Form ${percent(weights.form)}`
                  : `History ${percent(weights.historic)}, fixtures ${percent(
                      weights.difficulty,
                    )}`
              }
              onPointerDown={(event) => {
                event.preventDefault();
                (event.target as HTMLElement).focus();
                setDragging(handle);
              }}
              onKeyDown={onKeyDown(handle)}
            />
          );
        })}
      </div>
    </div>
  );
}
