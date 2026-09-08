import { useEffect, useMemo, useState } from "react";
import { message, useSettings } from "../lib/settingsStore";
import { normalise } from "../lib/text";

interface Props {
  name: string;
  /** Saves the server a lookup, and lets it refuse an unknown name cleanly. */
  position?: string;
}

/**
 * The two buttons that appear wherever a player does: force them into the
 * squad, or keep them out of it.
 *
 * Both lists are constrained — two goalkeepers, five midfielders, three from
 * any one club, and nobody on both lists at once — so the server decides and
 * the reason it gives is shown here rather than swallowed.
 */
export function PlayerActions({ name, position }: Props) {
  const { settings, force, unforce, avoid, unavoid } = useSettings();
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [done, setDone] = useState("");

  // A different player means the last message no longer applies.
  useEffect(() => {
    setError("");
    setDone("");
  }, [name]);

  const key = normalise(name);

  const forcedAt = useMemo(() => {
    const forced = settings?.forced_selections ?? {};
    for (const [pos, names] of Object.entries(forced)) {
      if ((names ?? []).some((entry) => normalise(entry) === key)) return pos;
    }
    return null;
  }, [settings, key]);

  const avoided = useMemo(
    () =>
      (settings?.blacklist_players ?? []).some((entry) => normalise(entry) === key),
    [settings, key],
  );

  const run = async (label: string, action: () => Promise<void>, note: string) => {
    setBusy(label);
    setError("");
    setDone("");
    try {
      await action();
      setDone(note);
    } catch (err) {
      setError(message(err));
    } finally {
      setBusy("");
    }
  };

  const working = busy !== "";

  return (
    <div className="player-actions">
      <div className="player-action-row">
        <button
          type="button"
          className={forcedAt ? "btn quiet small" : "btn small"}
          disabled={working || !settings}
          onClick={() =>
            forcedAt
              ? run("force", () => unforce(name), `${name} is no longer forced.`)
              : run(
                  "force",
                  () => force(name, position),
                  `${name} will be in every squad from the next run.`,
                )
          }
        >
          {busy === "force"
            ? "Saving…"
            : forcedAt
              ? "Remove from forced picks"
              : "Force into squad"}
        </button>
        <button
          type="button"
          className={avoided ? "btn quiet small" : "btn quiet small danger"}
          disabled={working || !settings}
          onClick={() =>
            avoided
              ? run("avoid", () => unavoid(name), `${name} is back in the pool.`)
              : run(
                  "avoid",
                  () => avoid(name),
                  `${name} is out of the pool from the next run.`,
                )
          }
        >
          {busy === "avoid"
            ? "Saving…"
            : avoided
              ? "Remove from avoid list"
              : "Add to avoid list"}
        </button>
      </div>

      {(forcedAt || avoided) && !error && !done && (
        <p className="action-note muted">
          {forcedAt
            ? `Currently a forced pick (${forcedAt}).`
            : "Currently on your avoid list."}
        </p>
      )}
      {done && <p className="action-note good">{done}</p>}
      {error && <p className="action-note bad">{error}</p>}
    </div>
  );
}

interface DialogProps {
  name: string;
  position?: string;
  subtitle?: string;
  onClose: () => void;
}

/** The same buttons in a modal, for tables where there's no room for a panel. */
export function PlayerActionsDialog({
  name,
  position,
  subtitle,
  onClose,
}: DialogProps) {
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
      aria-label={name}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="admin-panel player-dialog">
        <h2>{name}</h2>
        {subtitle && <p className="muted">{subtitle}</p>}
        <PlayerActions name={name} position={position} />
        <div className="admin-actions">
          <button className="btn quiet small" type="button" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
