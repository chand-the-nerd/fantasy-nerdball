import type {
  GameweekInfo,
  PlayerPool,
  Reference,
  League,
  Me,
  Performance,
  Run,
  Settings,
  Squad,
} from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
    ...init,
  });

  if (response.status === 204) return undefined as T;

  const text = await response.text();
  const data = text ? JSON.parse(text) : null;

  if (!response.ok) {
    const detail =
      (data && (data.detail || data.message)) || `Request failed (${response.status})`;
    throw new ApiError(response.status, String(detail));
  }
  return data as T;
}

export const api = {
  authConfig: () => request<{ google: boolean; dev_login: boolean }>("/api/auth/config"),
  devLogin: () => request<{ ok: boolean }>("/api/auth/dev-login", { method: "POST" }),
  logout: () => request<{ ok: boolean }>("/api/auth/logout", { method: "POST" }),

  me: () => request<Me>("/api/me"),
  settings: () => request<Settings>("/api/me/settings"),
  saveSettings: (body: Partial<Settings>) =>
    request<Settings>("/api/me/settings", {
      method: "PUT",
      body: JSON.stringify(body),
    }),
  fplEntry: () => request<any>("/api/me/fpl-entry"),
  importSquad: (body: {
    gameweek?: number;
    apply_budget: boolean;
    apply_free_transfers: boolean;
  }) =>
    request<any>("/api/me/import-squad", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  linkEntry: (fpl_entry_id: number | null) =>
    request<Me>("/api/me/fpl-entry", {
      method: "POST",
      body: JSON.stringify({ fpl_entry_id }),
    }),

  gameweek: () => request<GameweekInfo>("/api/gameweek"),
  reference: () => request<Reference>("/api/me/reference"),
  players: () => request<PlayerPool>("/api/players"),
  sync: () => request<{ gameweeks_synced: number }>("/api/sync", { method: "POST" }),

  startRun: (gameweek?: number) =>
    request<Run>("/api/runs", {
      method: "POST",
      body: JSON.stringify({ gameweek: gameweek ?? null }),
    }),
  run: (id: number) => request<Run>(`/api/runs/${id}`),
  latestRun: () => request<Run | null>("/api/runs/latest"),

  squads: () => request<Squad[]>("/api/squads"),
  latestSquad: () => request<Squad | null>("/api/squads/latest"),
  squad: (gameweek: number) => request<Squad>(`/api/squads/${gameweek}`),

  performance: (refresh = false) =>
    request<Performance>(`/api/performance${refresh ? "?refresh=true" : ""}`),
  recordResult: (gameweek: number, actual_points: number) =>
    request<{ ok: boolean }>("/api/performance/results", {
      method: "POST",
      body: JSON.stringify({ gameweek, actual_points }),
    }),
  league: () => request<League>("/api/performance/league"),

  adminStatus: () =>
    request<{ configured: boolean; unlocked: boolean; session_minutes: number }>(
      "/api/admin/status",
    ),
  adminUnlock: (password: string) =>
    request<{ unlocked: boolean }>("/api/admin/unlock", {
      method: "POST",
      body: JSON.stringify({ password }),
    }),
  adminLock: () => request<{ unlocked: boolean }>("/api/admin/lock", { method: "POST" }),
  adminMembers: () => request<any>("/api/admin/members"),
  adminAddInvite: (email: string) =>
    request<{ email: string }>("/api/admin/invites", {
      method: "POST",
      body: JSON.stringify({ email }),
    }),
  adminRemoveInvite: (id: number) =>
    request<void>(`/api/admin/invites/${id}`, { method: "DELETE" }),
  adminRemoveMember: (id: number) =>
    request<void>(`/api/admin/members/${id}`, { method: "DELETE" }),
};
