/**
 * What a guest gets, in one place.
 *
 * The numbers here mirror web/backend/app/guest.py, which is what
 * actually enforces them — these exist so the controls can be greyed out
 * at the right values rather than letting someone move a slider and then
 * telling them off for it. Change one, change the other.
 */
import { createContext, useContext } from "react";
import { api } from "./api";

export const GUEST = {
  gameweeks: 5,
  transferStrategy: 1.2,
  benchWeight: 0.2,
  weights: { form: 0.4, historic: 0.3, difficulty: 0.3 },
  maxForced: 2,
  maxBlacklist: 2,
} as const;

/** True for the rest of the tree when the session is a guest one. */
const GuestContext = createContext(false);

export const GuestProvider = GuestContext.Provider;

export function useGuest(): boolean {
  return useContext(GuestContext);
}

/**
 * Back to the sign-in page.
 *
 * A guest is signed into a throwaway account, so getting back to the
 * sign-in screen means ending that session — which is also what deletes
 * the account and everything it was holding.
 */
export function backToSignIn(): void {
  void api
    .logout()
    .catch(() => undefined)
    .finally(() => window.location.assign("/"));
}

export interface FeatureRow {
  feature: string;
  guest: string;
  member: string;
  /** Drives the tick or dash in the guest column. */
  available: "yes" | "partial" | "no";
}

export interface FeatureGroup {
  title: string;
  rows: FeatureRow[];
}

/** The comparison shown before anyone commits to going in as a guest. */
export const FEATURE_GROUPS: FeatureGroup[] = [
  {
    title: "Your squad",
    rows: [
      {
        feature: "Getting a squad in",
        guest: "Entered by hand",
        member: "By hand, or imported from your FPL team",
        available: "partial",
      },
      {
        feature: "Keeping it",
        guest: "Nothing is saved once you leave",
        member: "Every squad and run kept, week to week",
        available: "no",
      },
      {
        feature: "Squad options",
        guest: "Every option a run produces",
        member: "The same, carried into next week's transfers",
        available: "yes",
      },
      {
        feature: "Calculations table",
        guest: "Locked",
        member: "Every input behind every projection",
        available: "no",
      },
      {
        feature: "Free Hit last week",
        guest: "Not needed — nothing carries over",
        member: "Available",
        available: "no",
      },
      {
        feature: "Planner",
        guest: "Locked",
        member: "Up to eight gameweeks ahead",
        available: "no",
      },
    ],
  },
  {
    title: "Teams and players",
    rows: [
      {
        feature: "Teams page",
        guest: "Everything",
        member: "Everything",
        available: "yes",
      },
      {
        feature: "Best picks and differentials",
        guest: "Everything",
        member: "Everything",
        available: "yes",
      },
      {
        feature: "Player lookup",
        guest: "Locked",
        member: "Full underlying numbers and fixtures",
        available: "no",
      },
    ],
  },
  {
    title: "Setup",
    rows: [
      {
        feature: "Fixtures ahead",
        guest: `Fixed at ${GUEST.gameweeks} gameweeks`,
        member: "Anywhere from 1 to 10",
        available: "partial",
      },
      {
        feature: "Transfer strategy",
        guest: `Fixed at ${GUEST.transferStrategy.toFixed(1)}`,
        member: "Kneejerk through to conservative",
        available: "partial",
      },
      {
        feature: "Bench importance",
        guest: `Fixed at ${Math.round(GUEST.benchWeight * 100)}%`,
        member: "0% to 100%",
        available: "partial",
      },
      {
        feature: "Hits and injured players",
        guest: "Both left switched on",
        member: "Yours to switch",
        available: "partial",
      },
      {
        feature: "Form / history / fixtures",
        guest: "Fixed at 40/30/30",
        member: "Tuned per position",
        available: "partial",
      },
      {
        feature: "Forced picks",
        guest: `${GUEST.maxForced} players`,
        member: "Up to a full squad",
        available: "partial",
      },
      {
        feature: "Players to avoid",
        guest: `${GUEST.maxBlacklist} players`,
        member: "As many as you like",
        available: "partial",
      },
      {
        feature: "Club adjustments",
        guest: "Locked at 1.00",
        member: "0.00 to 2.00, club by club",
        available: "no",
      },
      {
        feature: "Import your FPL team",
        guest: "Locked",
        member: "Link once, import any gameweek",
        available: "no",
      },
    ],
  },
];
