import { useEffect, useState } from "react";
import { ChipStrip } from "./ChipStrip";
import { message, useSettings } from "../lib/settingsStore";
import type { Settings } from "../lib/types";

/**
 * What changes week to week, on the page where you run the optimiser.
 *
 * Everything here saves as you go — a run reads the saved row, so a budget
 * typed and left unsaved would have been silently ignored.
 */
export function RunSettings({ disabled = false }: { disabled?: boolean }) {
  const { settings, patch } = useSettings();
  const [budget, setBudget] = useState("");
  const [transfers, setTransfers] = useState("");
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);

  // Track the stored values, except while the field is being edited.
  useEffect(() => {
    if (!settings) return;
    setBudget((current) => (current === "" ? String(settings.budget) : current));
    setTransfers((current) =>
      current === "" ? String(settings.free_transfers) : current,
    );
  }, [settings]);

  if (!settings) return null;

  const save = async (changes: Partial<Settings>) => {
    setError("");
    try {
      await patch(changes);
      setSaved(true);
      window.setTimeout(() => setSaved(false), 2000);
    } catch (err) {
      setError(message(err));
      // Put the fields back to what is actually stored.
      setBudget(String(settings.budget));
      setTransfers(String(settings.free_transfers));
    }
  };

  const commitBudget = () => {
    const value = Number(budget);
    if (!Number.isFinite(value) || value === settings.budget) {
      setBudget(String(settings.budget));
      return;
    }
    void save({ budget: Math.round(value * 10) / 10 });
  };

  const commitTransfers = () => {
    const value = Math.round(Number(transfers));
    if (!Number.isFinite(value) || value === settings.free_transfers) {
      setTransfers(String(settings.free_transfers));
      return;
    }
    void save({ free_transfers: value });
  };

  return (
    <div className="panel run-settings">
      <div className="run-settings-grid">
        <div className="field">
          <label htmlFor="run-budget">Budget</label>
          <input
            id="run-budget"
            type="number"
            step="0.1"
            inputMode="decimal"
            value={budget}
            disabled={disabled}
            onChange={(event) => setBudget(event.target.value)}
            onBlur={commitBudget}
            onKeyDown={(event) => event.key === "Enter" && event.currentTarget.blur()}
          />
          <span className="hint">Squad value plus the bank.</span>
        </div>

        <div className="field">
          <label htmlFor="run-transfers">Free transfers</label>
          <input
            id="run-transfers"
            type="number"
            min={0}
            max={15}
            inputMode="numeric"
            value={transfers}
            disabled={disabled}
            onChange={(event) => setTransfers(event.target.value)}
            onBlur={commitTransfers}
            onKeyDown={(event) => event.key === "Enter" && event.currentTarget.blur()}
          />
          <span className="hint">Before any hit.</span>
        </div>

        <div className="field chips-field">
          <label>
            Chips
            <span className="saved-flash" aria-live="polite">
              {saved ? "saved" : ""}
            </span>
          </label>
          <ChipStrip
            settings={settings}
            onChange={(changes) => void save(changes)}
            disabled={disabled}
          />
        </div>
      </div>

      {error && <div className="notice bad">{error}</div>}
    </div>
  );
}
