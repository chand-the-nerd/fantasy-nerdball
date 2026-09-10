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
        feature: "Squad Import",
        guest: "Manually",
        member: "Manually, or use FPL ID",
        available: "partial",
      },
      {
        feature: "Squad Save",
        guest: "No",
        member: "Squad linked to account",
        available: "no",
      },
      {
        feature: "Squad Calculations",
        guest: "No",
        member: "Yes",
        available: "no",
      },
      {
        feature: "Planner",
        guest: "No",
        member: "Plan up to 8 gameweeks ahead",
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
        guest: "No",
        member: "See underlying stats for any player",
        available: "no",
      },
    ],
  },
  {
    title: "Setup",
    rows: [
      {
        feature: "Lookahead Period",
        guest: `Fixed at ${GUEST.gameweeks} gameweeks`,
        member: "1 to 10",
        available: "partial",
      },
      {
        feature: "Transfer Sensitivity",
        guest: `Fixed`,
        member: "Adjustable",
        available: "partial",
      },
      {
        feature: "Bench importance",
        guest: `Fixed`,
        member: "Adjustable",
        available: "partial",
      },
      {
        feature: "Hits and injured players",
        guest: `Fixed`,
        member: "Adjustable",
        available: "partial",
      },
      {
        feature: "Model Prioritisation",
        guest: `Fixed`,
        member: "Adjustable",
        available: "partial",
      },
      {
        feature: "Forced picks",
        guest: `Limited`,
        member: "Unlimited",
        available: "partial",
      },
      {
        feature: "Players to avoid",
        guest: `Limited`,
        member: "Unlimited",
        available: "partial",
      },
      {
        feature: "Club adjustments",
        guest: `Fixed`,
        member: "Adjustable",
        available: "partial",
      },
      {
        feature: "Import your FPL team",
        guest: "Locked",
        member: "Link to account",
        available: "no",
      },
    ],
  },
];
