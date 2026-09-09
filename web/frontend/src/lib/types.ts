export type Position = "GK" | "DEF" | "MID" | "FWD";

export interface Player {
  id: number | null;
  code: number | null;
  name: string;
  position: Position;
  team: string;
  team_short: string;
  price: number;
  projected_points: number;
  form: number | null;
  historic_ppg: number | null;
  fixture_difficulty: number | null;
  start_rate: number | null;
  minutes_per_game: number | null;
  xg_modifier: number | null;
  next_opponent: string;
  venue: string;
  status: string;
  news: string;
  is_captain: boolean;
  is_vice: boolean;
  is_double_gameweek: boolean;
  on_bench: boolean;
  bench_order: number | null;
}

/**
 * One squad the optimiser is offering. Everything the pitch and the scoreline
 * need is here, so switching between them is a repaint rather than a run.
 */
export interface SquadOption {
  key: string;
  label: string;
  kind: "recommended" | "alternative" | "previous";
  starting: Player[];
  bench: Player[];
  formation: string;
  projected_points: number;
  squad_value: number;
  bank: number;
  transfers_made: number;
  penalty_points: number;
  player_ids: number[];
  transfers: { in: string[]; out: string[] };
  /** True for the one the optimiser would pick left to itself. */
  recommended: boolean;
  /** Players this option drops relative to option one. Zero on option one. */
  differs_by: number;
}

export interface SquadPayload {
  starting: Player[];
  bench: Player[];
  formation: string;
  gameweek: number;
  season: string;
  projected_points: number;
  squad_value: number;
  bank: number;
  chip: string;
  transfers_made: number;
  penalty_points: number;
  made_transfers: boolean;
  transfer_reason: string;
  points_gain_per_gw: number | null;
  transfers: { in: string[]; out: string[] };
  model_xi: { projected_points: number; cost: number; starting: Player[] } | null;
  options?: SquadOption[];
  active_option?: string;
  imported?: boolean;
  explored?: {
    player_out: string;
    position: string;
    out_score: number | null;
    replacement: string | null;
    replacement_score: number | null;
    points_lost: number | null;
    verdict: string;
  }[];
}

export interface Squad {
  id: number;
  gameweek: number;
  season: string;
  formation: string;
  projected_points: number;
  squad_value: number;
  bank: number;
  transfers_made: number;
  penalty_points: number;
  chip: string;
  active_option?: string;
  payload: SquadPayload;
}

export interface Me {
  id: number;
  email: string;
  name: string;
  avatar_url: string;
  is_admin: boolean;
  fpl_entry_id: number | null;
}

export type Theme = "legacy" | "dark" | "light";

export interface Settings {
  theme: Theme;
  tutorial_seen: boolean;
  budget: number;
  free_transfers: number;
  accept_transfer_penalty: boolean;
  exclude_unavailable: boolean;
  wildcard: boolean;
  free_hit: boolean;
  free_hit_prev_gw: boolean;
  bench_boost: boolean;
  triple_captain: boolean;
  use_ml_weights: boolean;
  first_n_gameweeks: number;
  min_transfer_value: number;
  overrides: Record<string, unknown>;
  team_modifiers: Record<string, number>;
  forced_selections: Record<string, string[]>;
  blacklist_players: string[];
}

export interface Run {
  id: number;
  gameweek: number;
  season: string;
  status: "queued" | "running" | "complete" | "failed" | "cancelled";
  log: string;
  error: string;
  squad_id: number | null;
  result: Record<string, unknown>;
}

export interface PerformancePoint {
  gameweek: number;
  projected: number | null;
  actual: number | null;
  global_average: number | null;
  global_highest: number | null;
  overall_rank: number | null;
  finished: boolean;
  chip: string;
}

export interface Performance {
  season: string;
  series: PerformancePoint[];
  summary: {
    gameweeks_scored: number;
    total_points: number;
    total_global_average: number;
    points_above_average: number;
    gameweeks_beating_average: number;
    latest_overall_rank: number | null;
    model_mean_error: number | null;
    fpl_entry_linked: boolean;
  };
}

export interface LeagueRow {
  user_id: number;
  name: string;
  avatar_url: string;
  total_points: number;
  gameweeks: number;
  overall_rank: number | null;
  is_you: boolean;
  position: number;
}

export interface League {
  season: string;
  global_average_total: number;
  standings: LeagueRow[];
}

export interface GameweekInfo {
  gameweek: number;
  season: string;
  deadline: string | null;
  average_last_gw: number | null;
}

export interface Reference {
  positions: string[];
  weight_keys: string[];
  default_weights: Record<string, Record<string, number>>;
  teams: string[];
  squad_limits: Record<string, number>;
}

export interface PlayerPool {
  season: string;
  players: {
    id: number;
    name: string;
    full_name: string;
    position: string;
    team: string;
    price: number;
    status: string;
    news: string;
    chance_of_playing: number | null;
    total_points: number;
    selected_by: number;
    ambiguous: boolean;
  }[];
}
