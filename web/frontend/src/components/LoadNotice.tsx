import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { useGuest } from "../lib/guest";
import type { SiteStatus } from "../lib/types";

/**
 * Says so when the site is under load, rather than letting people guess.
 *
 * The honest framing matters here. This isn't a service anybody pays
 * for, running on hardware sized for a handful of friends, and when it
 * runs out of room the useful thing to tell someone is why — and that
 * signing in is what gets them ahead of the queue, because it does.
 *
 * Only shown when it's true. A permanent "we might be slow" banner is
 * wallpaper within a week.
 */
export function LoadNotice() {
  const guest = useGuest();
  const [status, setStatus] = useState<SiteStatus | null>(null);

  useEffect(() => {
    let live = true;

    const check = () => {
      api
        .status()
        .then((next) => {
          if (live) setStatus(next);
        })
        .catch(() => undefined);
    };

    check();
    // A minute is often enough to catch a busy spell and rare enough to
    // cost nothing on a server this size.
    const timer = window.setInterval(check, 60000);
    return () => {
      live = false;
      window.clearInterval(timer);
    };
  }, []);

  if (!status?.busy) return null;

  return (
    <div className="notice load-notice">
      <strong>Busy right now</strong>
      <span>
        {guest ? (
          <>
            {status.queue_depth} optimisation
            {status.queue_depth === 1 ? "" : "s"} are queued, and
            signed-in managers go first while it's like this. Fantasy
            Nerdball is a free beta running on one small server that
            nobody pays for — runs may be turned away until it clears.
          </>
        ) : (
          <>
            {status.queue_depth} optimisation
            {status.queue_depth === 1 ? "" : "s"} are queued ahead of the
            worker, so a run may take a few minutes to start. You're
            ahead of any guests in the queue.
          </>
        )}
      </span>
    </div>
  );
}
