import { useEffect, useState } from "react";
import { FplIdHelp } from "./FplIdHelp";
import { GuestLock } from "./GuestLock";
import { api, ApiError } from "../lib/api";
import { useGuest } from "../lib/guest";
import type { Me } from "../lib/types";

/**
 * Shown when nothing is saved for the gameweek just gone.
 *
 * The optimiser works out transfers by comparing this week's best squad
 * against last week's saved one. With nothing there it treats you as a new
 * team with a free hand, and its first set of transfers is meaningless — so
 * this asks for that squad before a run rather than after.
 */
export function StartingSquadPrompt({
  gameweek,
  me,
  onMeChange,
  onImported,
  onBuild,
  onDismiss,
}: {
  gameweek: number;
  me: Me;
  onMeChange: (me: Me) => void;
  onImported: () => void;
  onBuild: () => void;
  onDismiss: () => void;
}) {
  const guest = useGuest();
  const [entryId, setEntryId] = useState("");
  const [busy, setBusy] = useState<"link" | "import" | null>(null);
  const [error, setError] = useState("");
  const [note, setNote] = useState("");

  useEffect(() => {
    setError("");
  }, [me.fpl_entry_id]);

  const runImport = async () => {
    setError("");
    setBusy("import");
    try {
      const result = await api.importSquad({
        gameweek,
        apply_budget: true,
        apply_free_transfers: true,
      });
      setNote(
        `Imported ${result.players} players, ${result.formation}, ` +
          `£${result.budget}m budget.`,
      );
      onImported();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const linkThenImport = async () => {
    setError("");
    setBusy("link");
    try {
      onMeChange(await api.linkEntry(Number(entryId.trim())));
      setEntryId("");
      await runImport();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
      setBusy(null);
    }
  };

  return (
    <div className="panel start-prompt">
      <div className="start-prompt-head">
        <h3>No squad saved for gameweek {gameweek}</h3>
        <button className="link-button" type="button" onClick={onDismiss}>
          Not now
        </button>
      </div>
      <p className="muted" style={{ marginTop: -4 }}>
        The optimiser decides transfers by comparing against the side you had
        last week. Without it, it builds from scratch and every suggested
        transfer is guesswork.
      </p>

      {error && <div className="notice bad">{error}</div>}
      {note && <div className="notice good">{note}</div>}

      <div className="start-options">
        {guest ? (
          <GuestLock note="Pulls your real side, its value and your free transfers straight from FPL.">
            <div className="start-option">
              <strong>Import from FPL</strong>
              <p className="muted">
                Link your team id once and the squad comes across on its
                own.
              </p>
            </div>
          </GuestLock>
        ) : (
        <div className="start-option">
          <strong>Import from FPL</strong>
          <p className="muted">
            Pulls your real side, its value and your free transfers straight from
            the FPL API.
          </p>
          {me.fpl_entry_id ? (
            <button
              className="btn"
              type="button"
              onClick={runImport}
              disabled={busy !== null}
            >
              {busy === "import" ? "Importing…" : "Import my team"}
            </button>
          ) : (
            <>
              <div className="field">
                <label htmlFor="prompt-entry">FPL team id</label>
                <input
                  id="prompt-entry"
                  type="text"
                  inputMode="numeric"
                  placeholder="1234567"
                  value={entryId}
                  onChange={(event) => setEntryId(event.target.value)}
                  onKeyDown={(event) =>
                    event.key === "Enter" && entryId.trim() && linkThenImport()
                  }
                />
                <span className="hint">
                  The number in your team's URL on the FPL site.
                </span>
              </div>
              <FplIdHelp />
              <button
                className="btn"
                type="button"
                onClick={linkThenImport}
                disabled={!entryId.trim() || busy !== null}
              >
                {busy ? "Working…" : "Link and import"}
              </button>
            </>
          )}
        </div>
        )}

        <div className="start-option">
          <strong>Enter it by hand</strong>
          <p className="muted">
            Pick the fifteen yourself on a blank pitch. Useful if you'd rather
            not link an account, or the side you want to transfer from isn't
            what FPL has.
          </p>
          <button
            className="btn quiet"
            type="button"
            onClick={onBuild}
            disabled={busy !== null}
          >
            Build it manually
          </button>
        </div>
      </div>
    </div>
  );
}
