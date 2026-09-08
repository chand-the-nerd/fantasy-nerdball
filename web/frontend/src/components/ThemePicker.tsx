import { useState } from "react";
import { THEMES, applyTheme, message, useSettings } from "../lib/settingsStore";
import type { Theme } from "../lib/types";

/** The style dropdown in the footer. Saves to the account, not the browser. */
export function ThemePicker() {
  const { settings, patch } = useSettings();
  const [error, setError] = useState("");

  const choose = async (theme: Theme) => {
    setError("");
    // Switch immediately: waiting on a round trip to change a colour feels
    // broken. If the save fails the old value comes back with the message.
    applyTheme(theme);
    try {
      await patch({ theme });
    } catch (err) {
      applyTheme(settings?.theme);
      setError(message(err));
    }
  };

  return (
    <label className="theme-picker" title={error || "Saved to your account"}>
      <span>Style</span>
      <select
        value={settings?.theme ?? "legacy"}
        disabled={!settings}
        onChange={(event) => void choose(event.target.value as Theme)}
        aria-label="Colour style"
      >
        {THEMES.map((theme) => (
          <option key={theme.id} value={theme.id}>
            {theme.label}
          </option>
        ))}
      </select>
    </label>
  );
}
