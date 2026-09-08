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
import type { Settings } from "./types";

let cached: Settings | null = null;
let inflight: Promise<Settings> | null = null;
const listeners = new Set<(settings: Settings) => void>();

export function publishSettings(next: Settings): void {
  cached = next;
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
