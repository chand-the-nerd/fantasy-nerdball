import type { PoolPlayer } from "./PlayerPicker";
import { availability } from "./PlayerPicker";
import { normalise } from "../lib/text";

export interface Problem {
  level: "error" | "warning";
  text: string;
}

const SQUAD_SIZE: Record<string, number> = { GK: 2, DEF: 5, MID: 5, FWD: 3 };
const MAX_PER_CLUB = 3;

/**
 * Checks the forced and blacklisted picks against the rules the optimiser
 * will apply, before a run is started.
 *
 * The point is to catch infeasibility here rather than have PuLP return
 * "No valid solution found" ten minutes in with no explanation of which
 * constraint was at fault.
 */
export function checkConstraints({
  forced,
  blacklist,
  pool,
  budget,
  excludeUnavailable,
  limits,
}: {
  forced: Record<string, string[]>;
  blacklist: string[];
  pool: PoolPlayer[];
  budget: number;
  excludeUnavailable: boolean;
  limits: Record<string, number>;
}): Problem[] {
  const problems: Problem[] = [];
  if (pool.length === 0) return problems;

  const byName = new Map(pool.map((p) => [normalise(p.name), p]));
  const positions = Object.keys(SQUAD_SIZE);

  const allForced: { name: string; position: string }[] = [];
  for (const position of positions) {
    for (const name of forced[position] ?? []) {
      allForced.push({ name, position });
    }
  }

  // Position limits.
  for (const position of positions) {
    const count = (forced[position] ?? []).length;
    const limit = limits[position] ?? SQUAD_SIZE[position];
    if (count > limit) {
      problems.push({
        level: "error",
        text: `${count} ${position} players forced, but a squad only has ${limit}.`,
      });
    }
  }

  // Unknown names. The optimiser prints a warning and carries on, so the
  // constraint is silently dropped rather than failing loudly.
  for (const { name } of allForced) {
    if (!byName.has(normalise(name))) {
      problems.push({
        level: "warning",
        text: `"${name}" isn't in this season's player pool, so it will be ignored.`,
      });
    }
  }
  for (const name of blacklist) {
    if (!byName.has(normalise(name))) {
      problems.push({
        level: "warning",
        text: `"${name}" isn't in this season's player pool, so avoiding them does nothing.`,
      });
    }
  }

  // Forced and avoided at once.
  const avoided = new Set(blacklist.map(normalise));
  for (const { name } of allForced) {
    if (avoided.has(normalise(name))) {
      problems.push({
        level: "error",
        text: `${name} is both forced and avoided. Pick one.`,
      });
    }
  }

  // Three per club is an FPL rule, so four forced from one club can never solve.
  const perClub = new Map<string, string[]>();
  for (const { name } of allForced) {
    const player = byName.get(normalise(name));
    if (!player) continue;
    perClub.set(player.team, [...(perClub.get(player.team) ?? []), player.name]);
  }
  for (const [team, names] of perClub) {
    if (names.length > MAX_PER_CLUB) {
      problems.push({
        level: "error",
        text: `${names.length} players forced from ${team} (${names.join(", ")}). FPL allows ${MAX_PER_CLUB} per club, so there's no valid squad.`,
      });
    }
  }

  // Forcing someone the optimiser has been told to exclude cannot be satisfied.
  if (excludeUnavailable) {
    for (const { name } of allForced) {
      const player = byName.get(normalise(name));
      if (!player) continue;
      const state = availability(player.status);
      if (state.tone === "out") {
        problems.push({
          level: "error",
          text: `${player.name} is ${state.word} and injured players are being skipped, so forcing them leaves no valid squad. Either drop them or turn off "Skip injured and suspended players".`,
        });
      } else if (state.tone === "doubt") {
        problems.push({
          level: "warning",
          text: `${player.name} is doubtful${
            player.chance_of_playing !== null
              ? ` (${player.chance_of_playing}% chance)`
              : ""
          }.`,
        });
      }
    }
  }

  // Budget. Fill the unforced slots with the cheapest player available in
  // each position to get the floor.
  const cheapest: Record<string, number> = {};
  for (const position of positions) {
    const prices = pool
      .filter((p) => p.position === position && (!excludeUnavailable || p.status === "a"))
      .map((p) => p.price);
    cheapest[position] = prices.length ? Math.min(...prices) : 4.0;
  }

  let forcedCost = 0;
  const forcedByPosition: Record<string, number> = { GK: 0, DEF: 0, MID: 0, FWD: 0 };
  for (const { name, position } of allForced) {
    const player = byName.get(normalise(name));
    if (!player) continue;
    forcedCost += player.price;
    forcedByPosition[position] += 1;
  }

  let floor = forcedCost;
  for (const position of positions) {
    const remaining = Math.max(0, SQUAD_SIZE[position] - forcedByPosition[position]);
    floor += remaining * cheapest[position];
  }

  if (allForced.length > 0 && floor > budget + 0.05) {
    problems.push({
      level: "error",
      text: `Your forced picks cost £${forcedCost.toFixed(
        1,
      )}m, and the cheapest possible squad around them is £${floor.toFixed(
        1,
      )}m — more than your £${budget.toFixed(1)}m budget.`,
    });
  } else if (allForced.length > 0 && floor > budget - 4) {
    problems.push({
      level: "warning",
      text: `Forced picks leave only about £${(budget - floor).toFixed(
        1,
      )}m of headroom, so the rest of the squad will be near-minimum price.`,
    });
  }

  // Duplicated web_names: the optimiser warns and takes the first match.
  for (const { name } of allForced) {
    const player = byName.get(normalise(name));
    if (player?.ambiguous) {
      problems.push({
        level: "warning",
        text: `More than one player is listed as "${player.name}". The optimiser will take the first it finds.`,
      });
    }
  }

  return problems;
}
