import { useState } from "react";
import { api, ApiError } from "../lib/api";

const KINDS: { id: string; label: string; placeholder: string }[] = [
  {
    id: "broken",
    label: "Something is broken",
    placeholder:
      "What were you doing, and what happened instead? The gameweek " +
      "and the page help.",
  },
  {
    id: "feature",
    label: "Feature request",
    placeholder: "What would you like it to do?",
  },
  {
    id: "general",
    label: "General feedback",
    placeholder: "Anything at all.",
  },
];

export function FeedbackDialog({ onClose }: { onClose: () => void }) {
  const [kind, setKind] = useState(KINDS[0].id);
  const [body, setBody] = useState("");
  const [state, setState] = useState<"" | "sending" | "sent">("");
  const [error, setError] = useState("");

  const chosen = KINDS.find((option) => option.id === kind) ?? KINDS[0];

  const submit = async () => {
    setError("");
    setState("sending");
    try {
      await api.sendFeedback(kind, body.trim());
      setState("sent");
      // Long enough to read, short enough not to be in the way.
      window.setTimeout(onClose, 1600);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : String(err));
      setState("");
    }
  };

  return (
    <div
      className="overlay"
      role="dialog"
      aria-modal="true"
      aria-label="Send feedback"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="admin-panel">
        <div className="admin-head">
          <h2>Feedback</h2>
          <button className="link-button" type="button" onClick={onClose}>
            Close
          </button>
        </div>

        {state === "sent" ? (
          <div className="notice good">Thanks — that's gone through.</div>
        ) : (
          <>
            <div className="field">
              <label htmlFor="feedback-kind">What kind?</label>
              <select
                id="feedback-kind"
                value={kind}
                onChange={(event) => setKind(event.target.value)}
              >
                {KINDS.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="field">
              <label htmlFor="feedback-body">Tell me about it</label>
              <textarea
                id="feedback-body"
                rows={5}
                value={body}
                placeholder={chosen.placeholder}
                onChange={(event) => setBody(event.target.value)}
              />
            </div>

            {error && <div className="notice bad">{error}</div>}

            <div className="admin-actions">
              <button
                className="btn"
                type="button"
                disabled={!body.trim() || state === "sending"}
                onClick={() => void submit()}
              >
                {state === "sending" ? "Sending…" : "Send"}
              </button>
              <button
                className="btn quiet"
                type="button"
                onClick={onClose}
              >
                Cancel
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
