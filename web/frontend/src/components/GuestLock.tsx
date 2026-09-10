import { backToSignIn } from "../lib/guest";

/**
 * A panel a guest can see the shape of but not read.
 *
 * The content underneath is still rendered, blurred and inert, because a
 * blank box tells you nothing about what you're missing — the point is
 * to show that there is a table there, not to hide that one exists.
 */
export function GuestLock({
  children,
  note,
}: {
  children: React.ReactNode;
  note?: string;
}) {
  return (
    <div className="locked">
      <div className="locked-body" aria-hidden="true">
        {children}
      </div>
      <div className="locked-veil">
        <strong>For Signed-In Users Only</strong>
        {note && <p>{note}</p>}
        <button className="btn small" type="button" onClick={backToSignIn}>
          Back to sign in
        </button>
      </div>
    </div>
  );
}

/**
 * A limit a guest has run into, said plainly, with the way out of it.
 */
export function GuestNote({
  children,
  tone = "",
}: {
  children: React.ReactNode;
  tone?: "" | "bad";
}) {
  return (
    <div className={`notice guest-note ${tone}`.trim()}>
      <span>{children}</span>
      <button className="link-button" type="button" onClick={backToSignIn}>
        Sign in
      </button>
    </div>
  );
}
