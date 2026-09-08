/**
 * How to find an FPL team id, next to the box that asks for one.
 *
 * Collapsed by default: anyone who already knows the number doesn't need six
 * lines of instructions in their way, and anyone who doesn't shouldn't have
 * to go and ask.
 */
export function FplIdHelp() {
  return (
    <details className="id-help">
      <summary>Where do I find this?</summary>
      <ol>
        <li>
          Sign in at{" "}
          <a
            href="https://fantasy.premierleague.com"
            target="_blank"
            rel="noreferrer noopener"
          >
            fantasy.premierleague.com
          </a>{" "}
          in a browser — the app won't do, since you need to see the address
          bar.
        </li>
        <li>
          Click the <strong>Points</strong> tab.
        </li>
        <li>
          Read the address bar:{" "}
          <code>fantasy.premierleague.com/entry/1234567/event/4</code>. The
          number between <code>/entry/</code> and <code>/event/</code> is your
          id.
        </li>
      </ol>
      <p className="hint">
        No Points tab yet, before a season starts? Pick Team → Gameweek History
        gives the same number, as <code>/entry/1234567/history</code>. To check
        you have the right one, open{" "}
        <code>fantasy.premierleague.com/api/entry/1234567/</code> — it should
        show your team name back. The id is public, and grants nobody access to
        your account.
      </p>
    </details>
  );
}
