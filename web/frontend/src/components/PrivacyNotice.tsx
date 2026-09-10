import { useEffect } from "react";

/**
 * What the site keeps, why, and for how long.
 *
 * Short and specific on purpose. A notice nobody reads is still worth
 * having, but one somebody can actually read is worth more — and every
 * line here corresponds to something the app genuinely does, so it stays
 * true as long as somebody remembers to change it when the app changes.
 */
export function PrivacyNotice({ onClose }: { onClose: () => void }) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="overlay"
      role="dialog"
      aria-modal="true"
      aria-label="Privacy"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="admin-panel">
        <div className="admin-head">
          <h2>Privacy</h2>
          <button className="link-button" type="button" onClick={onClose}>
            Close
          </button>
        </div>

        <p className="muted">
          Fantasy Nerdball is a personal project run by one person. It
          keeps as little as it can get away with.
        </p>

        <h4>If you sign in</h4>
        <p>
          Your name, email address and profile picture come from Google
          when you sign in, and are used to identify your account and to
          email you about it. Your squads, settings and optimiser runs are
          stored so the app can compare week to week. If you link an FPL
          team id, it's stored to fetch your side and your points.
        </p>

        <h4>If you don't</h4>
        <p>
          Guest sessions create a temporary account with no name or
          address attached. Everything in it is deleted when you leave, or
          within a day if you don't come back.
        </p>

        <h4>Usage</h4>
        <p>
          The site records what happens on it — visits, runs, errors — so
          it can be kept working. For signed-in users that's tied to your
          account id. For anonymous visitors, a one-way fingerprint of
          your IP address is stored, along with the network it came from
          (not the full address), so one visitor can be told from another.
          Nothing is shared with advertisers, and there are no third-party
          trackers.
        </p>

        <h4>How long</h4>
        <p>
          Usage records are deleted after 90 days. Accounts inactive for 28
          days are deleted along with everything in them. Email delivery is
          handled by Resend, who process the messages on our behalf.
        </p>

        <h4>Your data</h4>
        <p>
          Ask and your account and everything attached to it will be
          deleted — use the feedback link in the footer, or reply to any
          email from the site. Signing out of a guest session deletes it
          immediately.
        </p>

        <p className="hint">
          Not affiliated with, endorsed by or connected to the Premier
          League or Fantasy Premier League. Player and fixture data comes
          from the publicly available Fantasy Premier League API.
        </p>
      </div>
    </div>
  );
}
