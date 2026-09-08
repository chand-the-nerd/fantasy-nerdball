/**
 * One copy of the manager's settings, shared by every view that touches them.
 *
 * Forcing a player from the pitch, avoiding one from the rankings and editing
 * the budget on the optimiser page all write to the same row, so they read
 * from the same place too. Without this each panel would hold its own stale
 * copy and the last one to save would quietly undo the others.
 */
import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "./api";
import type { Settings, Theme } from "./types";

const THEME_KEY = "nerdball-theme";
export const THEMES: { id: Theme; label: string }[] = [
  { id: "legacy", label: "Legacy" },
  { id: "dark", label: "Dark" },
  { id: "light", label: "Light" },
];

/**
 * Legacy is what the stylesheet renders without an attribute, so it is set
 * rather than removed — otherwise a saved Legacy would look like no
 * preference at all and the last theme would linger for a frame.
 */
export function applyTheme(theme: Theme | undefined): void {
  document.documentElement.dataset.theme = theme ?? "legacy";
  try {
    if (theme) window.localStorage.setItem(THEME_KEY, theme);
  } catch {
    /* private browsing: the account copy is the one that matters */
  }
}

/** Applied before the first paint, so the page doesn't flash the old palette
    while the saved preference is still being fetched. */
export function applyStoredTheme(): void {
  try {
    const saved = window.localStorage.getItem(THEME_KEY) as Theme | null;
    if (saved) document.documentElement.dataset.theme = saved;
  } catch {
    /* nothing stored, nothing to do */
  }
}

let cached: Settings | null = null;
let inflight: Promise<Settings> | null = null;
const listeners = new Set<(settings: Settings) => void>();

export function publishSettings(next: Settings): void {
  cached = next;
  applyTheme(next.theme);
  for (const listener of listeners) listener(next);
}

export function loadSettings(force = false): Promise<Settings> {
  if (cached && !force) return Promise.resolve(cached);
  if (!inflight) {
    inflight = api
      .settings()
      .then((settings) => {
        publishSettings(settings);
        return settings;
      })
      .finally(() => {
        inflight = null;
      });
  }
  return inflight;
}

export function message(error: unknown): string {
  return error instanceof ApiError ? error.message : String(error);
}

export function useSettings() {
  const [settings, setSettings] = useState<Settings | null>(cached);
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    listeners.add(setSettings);
    loadSettings(true).catch((error) => setLoadError(message(error)));
    return () => {
      listeners.delete(setSettings);
    };
  }, []);

  /** Save a few fields. Throws, so callers can show why it didn't take. */
  const patch = useCallback(async (changes: Partial<Settings>) => {
    publishSettings(await api.saveSettings(changes));
  }, []);

  const force = useCallback(async (name: string, position?: string) => {
    publishSettings(await api.forcePlayer(name, position));
  }, []);

  const unforce = useCallback(async (name: string) => {
    publishSettings(await api.unforcePlayer(name));
  }, []);

  const avoid = useCallback(async (name: string) => {
    publishSettings(await api.blacklistPlayer(name));
  }, []);

  const unavoid = useCallback(async (name: string) => {
    publishSettings(await api.unblacklistPlayer(name));
  }, []);

  return { settings, loadError, patch, force, unforce, avoid, unavoid };
}
