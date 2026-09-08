import { useCallback, useEffect, useState } from "react";
import { FplIdHelp } from "./FplIdHelp";
import { api, ApiError } from "../lib/api";
import type { Me } from "../lib/types";

interface EntryDetails {
  linked: boolean;
  unreachable?: boolean;
  entry_id?: number;
  team_name?: string;
  manager_name?: string;
  overall_rank?: number | null;
  total_points?: number | null;
  upcoming_gameweek?: number | null;
  importable_gameweek?: number | null;
  existing_squad?: { gameweek: number; from_fpl: boolean } | null;
}

interface ImportResult {
  gameweek: number;
  formation: string;
  players: number;
  squad_value: number;
  bank: number;
  budget: number;
  free_transfers: number;
  free_transfers_note: string;
  chip: string;
  free_hit_used: boolean;
  applied: string[];
}

/**
 * Linking an FPL team and pulling the real squad in as the previous gameweek.
 *
 * The optimiser works out transfers by comparing this week's best squad
 * against the one it saved last week. Starting mid-season there is no saved
 * squad, so it treats you as a new team and its first set of transfers is
 * meaningless. Importing gives it the right starting point.
 */
export function FplTeamPanel({
  me,
  onMeChange,
  onSettingsChanged,
}: {
  me: Me;
  onMeChange: (me: Me) => void;
  onSettingsChanged: () => void;
}) {
  const [details, setDetails] = useState<EntryDetails | null>(null);
  const [draftId, setDraftId] = useState("");
  const [applyBudget, setApplyBudget] = useState(true);
  const [applyTransfers, setApplyTransfers] = useState(true);
  const [result, setResult] = useState<ImportResult | null>(null);
  const [busy, setBusy] = useState<"link" | "unlink" | "import" | null>(null);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      setDetails(await api.fplEntry());
    } catch {
      setDetails({ linked: Boolean(me.fpl_entry_id) });
    }
  }, [me.fpl_entry_id]);

  useEffect(() => {
    void load();
  }, [load]);

  const link = async () => {
    setError("");
    setBusy("link");
    try {
      onMeChange(await api.linkEntry(Number(draftId.trim())));
      setDraftId("");
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const unlink = async () => {
    if (
      !window.confirm(
        "Unlink this FPL team? Your saved squads stay, but your real points " +
          "will stop syncing until you link a team again.",
      )
    ) {
      return;
    }
    setError("");
    setBusy("unlink");
    try {
      onMeChange(await api.linkEntry(null));
      setResult(null);
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const runImport = async () => {
    setError("");
    setResult(null);
    setBusy("import");
    try {
      const imported = await api.importSquad({
        apply_budget: applyBudget,
        apply_free_transfers: applyTransfers,
      });
      setResult(imported);
      if (applyBudget || applyTransfers) onSettingsChanged();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  if (!details) return <div className="panel col-half" />;

  // ── Not linked ─────────────────────────────────────────────────────────
  if (!details.linked) {
    return (
      <div className="panel col-half">
        <h3>Your FPL team</h3>
        <p className="muted" style={{ marginTop: -6 }}>
          Linking your side lets the app import your real squad and track your
          points. It stays linked to this Google account until you unlink it.
        </p>
        {error && <div className="notice bad">{error}</div>}
        <div className="field">
          <label htmlFor="entry">Team id</label>
          <input
            id="entry"
            type="text"
            inputMode="numeric"
            value={draftId}
            placeholder="e.g. 1234567"
            onChange={(event) => setDraftId(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && draftId.trim() && link()}
          />
          <span className="hint">
            The number in your team's URL on the FPL site, between /entry/ and
            /event/.
          </span>
        </div>
        <FplIdHelp />
        <div className="panel-foot">
          <button
            className="btn small"
            onClick={link}
            disabled={!draftId.trim() || busy !== null}
            type="button"
          >
            {busy === "link" ? "Checking…" : "Link team"}
          </button>
        </div>
      </div>
    );
  }

  // ── Linked ─────────────────────────────────────────────────────────────
  const importable = details.importable_gameweek;
  const already = details.existing_squad;

  return (
    <div className="panel col-half">
      <h3>Your FPL team</h3>

      <div className="linked-team">
        <span className="linked-dot" aria-hidden="true" />
        <div>
          <strong>{details.team_name || `Team ${details.entry_id}`}</strong>
          <span className="muted">
            {details.unreachable
              ? `id ${details.entry_id} · FPL unreachable right now`
              : [
                  details.manager_name,
                  `id ${details.entry_id}`,
                  details.total_points != null ? `${details.total_points} pts` : null,
                ]
                  .filter(Boolean)
                  .join(" · ")}
          </span>
        </div>
        <button
          className="link-button danger"
          onClick={unlink}
          disabled={busy !== null}
          type="button"
        >
          {busy === "unlink" ? "Unlinking…" : "Unlink team"}
        </button>
      </div>

      {error && <div className="notice bad">{error}</div>}

      <div className="import-block">
        <h4>Import your squad</h4>
        {importable ? (
          <>
            <p className="muted">
              Saves your real side as gameweek {importable}, so the optimiser has
              something to compare against when it works out transfers for
              gameweek {details.upcoming_gameweek}.
            </p>

            {already && (
              <div className="notice">
                A squad is already saved for gameweek {importable}
                {already.from_fpl ? ", imported from FPL" : ", from an optimiser run"}.
                Importing replaces it.
              </div>
            )}

            <label className="toggle">
              <input
                type="checkbox"
                checked={applyBudget}
                onChange={(event) => setApplyBudget(event.target.checked)}
              />
              <span className="copy">
                <strong>Set my budget from FPL</strong>
                <span>Squad value plus whatever's in the bank.</span>
              </span>
            </label>

            <label className="toggle">
              <input
                type="checkbox"
                checked={applyTransfers}
                onChange={(event) => setApplyTransfers(event.target.checked)}
              />
              <span className="copy">
                <strong>Set my free transfers</strong>
                <span>
                  Worked out from your transfer history — FPL doesn't publish the
                  number directly, so check it afterwards.
                </span>
              </span>
            </label>

            <div className="panel-foot">
              <button
                className="btn small"
                onClick={runImport}
                disabled={busy !== null}
                type="button"
              >
                {busy === "import" ? "Importing…" : `Import gameweek ${importable}`}
              </button>
            </div>
          </>
        ) : (
          <p className="muted">
            There's no completed gameweek to import yet. Come back once gameweek
            1 has been played.
          </p>
        )}
      </div>

      {result && (
        <div className="notice good">
          <strong>
            Imported {result.players} players as gameweek {result.gameweek} (
            {result.formation}).
          </strong>
          <ul className="problem-list">
            <li>
              Squad value £{result.squad_value.toFixed(1)}m, bank £
              {result.bank.toFixed(1)}m.
            </li>
            {result.applied.length > 0 ? (
              <li>Applied: {result.applied.join(", ")}.</li>
            ) : (
              <li>Nothing applied to your settings.</li>
            )}
            {result.free_hit_used && (
              <li>
                You played a Free Hit that week, so "Free Hit last week" has been
                switched on — that side reverts, and the optimiser needs to look
                a gameweek further back.
              </li>
            )}
            <li className="warning">{result.free_transfers_note}</li>
          </ul>
        </div>
      )}
    </div>
  );
}
