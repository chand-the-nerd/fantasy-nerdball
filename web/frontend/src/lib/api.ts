import type {
  AdminMetrics,
  AuthConfig,
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
  authConfig: () => request<AuthConfig>("/api/auth/config"),
  guestLogin: () =>
    request<{ ok: boolean }>("/api/auth/guest", { method: "POST" }),
  devLogin: () =>
    request<{ ok: boolean }>("/api/auth/dev-login", { method: "POST" }),
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
  forcePlayer: (name: string, position?: string) =>
    request<Settings>("/api/me/lists/force", {
      method: "POST",
      body: JSON.stringify({ name, position: position ?? null }),
    }),
  unforcePlayer: (name: string) =>
    request<Settings>("/api/me/lists/unforce", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  blacklistPlayer: (name: string) =>
    request<Settings>("/api/me/lists/blacklist", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  unblacklistPlayer: (name: string) =>
    request<Settings>("/api/me/lists/unblacklist", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),

  startPlan: (body: { weeks: number; chips: Record<string, string> }) =>
    request<any>("/api/plans", { method: "POST", body: JSON.stringify(body) }),
  plan: (id: number) => request<any>(`/api/plans/${id}`),
  latestPlan: () => request<any>("/api/plans/latest"),

  manualSquad: (body: {
    gameweek?: number;
    player_ids: number[];
    starting_ids: number[];
    bank: number;
    apply_budget: boolean;
  }) =>
    request<any>("/api/me/manual-squad", {
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
  bestPlayers: () => request<any>("/api/players/best"),
  differentialPlayers: () => request<any>("/api/players/differentials"),
  playerDetail: (id: number) => request<any>(`/api/players/${id}`),
  teams: (refresh = false) =>
    request<any>(`/api/teams${refresh ? "?refresh=true" : ""}`),
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
  activateOption: (gameweek: number, option: string) =>
    request<Squad>(`/api/squads/${gameweek}/activate`, {
      method: "POST",
      body: JSON.stringify({ option }),
    }),

  performance: (refresh = false) =>
    request<Performance>(`/api/performance${refresh ? "?refresh=true" : ""}`),
  recordResult: (gameweek: number, actual_points: number) =>
    request<{ ok: boolean }>("/api/performance/results", {
      method: "POST",
      body: JSON.stringify({ gameweek, actual_points }),
    }),
  league: () => request<League>("/api/performance/league"),

  adminStatus: () =>
    request<{ admin: boolean; email: string; owner_email: string }>(
      "/api/admin/status",
    ),
  adminMetrics: (window: string) =>
    request<AdminMetrics>(
      `/api/admin/metrics?window=${encodeURIComponent(window)}`,
    ),
  adminMembers: () => request<any>("/api/admin/members"),
  adminAddInvite: (email: string) =>
    request<{ email: string }>("/api/admin/invites", {
      method: "POST",
      body: JSON.stringify({ email }),
    }),
  adminRemoveInvite: (id: number) =>
    request<void>(`/api/admin/invites/${id}`, { method: "DELETE" }),
  cronStatus: () => request<any>("/api/cron/status"),
  cronRunNow: () => request<any>("/api/cron/run-now", { method: "POST" }),
  adminRemoveMember: (id: number) =>
    request<void>(`/api/admin/members/${id}`, { method: "DELETE" }),
};
