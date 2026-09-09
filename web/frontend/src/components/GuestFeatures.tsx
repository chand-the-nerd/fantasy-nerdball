import { Fragment, useEffect } from "react";
import { FEATURE_GROUPS, type FeatureRow } from "../lib/guest";

/** A tick, a half, or a dash — read before any of the words are. */
function Mark({ state }: { state: FeatureRow["available"] }) {
  if (state === "yes") return <span className="mark yes">✓</span>;
  if (state === "partial") return <span className="mark partial">◑</span>;
  return <span className="mark no">—</span>;
}

export function GuestFeatureTable() {
  return (
    <div className="calc-scroll">
      <table className="calc-table feature-table">
        <thead>
          <tr>
            <th>Feature</th>
            <th>As a guest</th>
            <th>Signed in</th>
          </tr>
        </thead>
        <tbody>
          {FEATURE_GROUPS.map((group) => (
            <Fragment key={group.title}>
              <tr className="feature-group">
                <th colSpan={3}>{group.title}</th>
              </tr>
              {group.rows.map((row) => (
                <tr key={`${group.title}-${row.feature}`}>
                  <td>{row.feature}</td>
                  <td>
                    <Mark state={row.available} />
                    {row.guest}
                  </td>
                  <td className="muted">{row.member}</td>
                </tr>
              ))}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Shown when someone clicks "Continue without signing in", so the trade
 * is on screen before it's made rather than discovered one greyed-out
 * slider at a time.
 */
export function GuestFeaturesDialog({
  onClose,
  onContinue,
  busy = false,
}: {
  onClose: () => void;
  onContinue?: () => void;
  busy?: boolean;
}) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="overlay"
      role="dialog"
      aria-modal="true"
      aria-label="What you get as a guest"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="admin-panel wide">
        <div className="admin-head">
          <h2>Guest access</h2>
          <button className="link-button" type="button" onClick={onClose}>
            Close
          </button>
        </div>
        <p className="muted">
          You can build a squad and run the optimiser without an account.
          The tuning is fixed, the deeper tables are closed, and nothing
          is saved once you leave.
        </p>

        <GuestFeatureTable />

        {onContinue && (
          <div className="admin-actions">
            <button
              className="btn"
              type="button"
              onClick={onContinue}
              disabled={busy}
            >
              {busy ? "Starting…" : "Continue as a guest"}
            </button>
            <button className="btn quiet" type="button" onClick={onClose}>
              Back
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
