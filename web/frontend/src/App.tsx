import { useEffect, useState } from "react";
import { AdminView } from "./components/AdminView";
import { FeedbackDialog } from "./components/FeedbackDialog";
import { GuestFeaturesDialog } from "./components/GuestFeatures";
import { PrivacyNotice } from "./components/PrivacyNotice";
import { GuestLock } from "./components/GuestLock";
import { PlannerView } from "./components/PlannerView";
import { PlayersView } from "./components/PlayersView";
import { SetupView } from "./components/SetupView";
import { TeamsView } from "./components/TeamsView";
import { SignIn } from "./components/SignIn";
import { SquadView } from "./components/SquadView";
import { ThemePicker } from "./components/ThemePicker";
import { FirstRunTour, Tutorial } from "./components/Tutorial";
import { api, ApiError } from "./lib/api";
import { GuestProvider, backToSignIn } from "./lib/guest";
import type { Me } from "./lib/types";

// Form and League are built and working, but hidden for now. To bring either
// back, add it to TABS and render it below — the components and their API
// routes are untouched.
type Tab = "squad" | "planner" | "players" | "teams" | "setup";

const TABS: { id: Tab; label: string }[] = [
  { id: "squad", label: "Squad" },
  { id: "planner", label: "Planner" },
  { id: "players", label: "Players" },
  { id: "teams", label: "Teams" },
  { id: "setup", label: "Setup" },
];

export function App() {
  const [me, setMe] = useState<Me | null>(null);
  const [checked, setChecked] = useState(false);
  const [tab, setTab] = useState<Tab>("squad");
  const [adminOpen, setAdminOpen] = useState(false);
  const [tourOpen, setTourOpen] = useState(false);
  const [guestInfo, setGuestInfo] = useState(false);
  const [feedback, setFeedback] = useState(false);
  const [privacy, setPrivacy] = useState(false);

  useEffect(() => {
    api
      .me()
      .then(setMe)
      .catch((err) => {
        if (!(err instanceof ApiError) || err.status !== 401) console.error(err);
      })
      .finally(() => setChecked(true));
  }, []);

  // Escape closes the admin overlay, as with any modal.
  useEffect(() => {
    if (!adminOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setAdminOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [adminOpen]);

  if (!checked) return null;
  if (!me) return <SignIn />;

  const guest = me.is_guest;
  const signOut = () => api.logout().then(() => window.location.reload());

  return (
    <GuestProvider value={guest}>
    <div className="shell">
      <aside className="rail">
        <div className="wordmark">
          Fantasy
          <span>Nerdball</span>
        </div>

        <nav>
          {TABS.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setTab(item.id)}
              aria-current={tab === item.id ? "page" : undefined}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <div className="rail-foot">
          {!guest && me.avatar_url && <img src={me.avatar_url} alt="" />}
          <span className="who">
            <strong>{guest ? "Guest" : me.name}</strong>
            {guest ? (
              <button
                className="link-button"
                onClick={backToSignIn}
                type="button"
              >
                Sign in
              </button>
            ) : (
              <button className="link-button" onClick={signOut} type="button">
                Sign out
              </button>
            )}
          </span>
        </div>
      </aside>

      <main className="main">
        {tab === "squad" && <SquadView me={me} onMeChange={setMe} />}
        {tab === "planner" &&
          (guest ? (
            <>
              <div className="topbar">
                <div>
                  <h1>Planner</h1>
                  <span className="when">
                    Several gameweeks, planned in one go
                  </span>
                </div>
              </div>
              <GuestLock
                note={
                  "A plan is built over several gameweeks and kept " +
                  "between them, so it needs an account."
                }
              >
                <div className="panel">
                  <h3>Plan ahead</h3>
                  <p className="muted">
                    Runs the optimiser forward week after week, with your
                    chips placed where you want them.
                  </p>
                </div>
              </GuestLock>
            </>
          ) : (
            <PlannerView />
          ))}
        {tab === "players" && <PlayersView />}
        {tab === "teams" && <TeamsView />}
        {tab === "setup" && <SetupView me={me} onMeChange={setMe} />}

        <footer className="app-foot">
          <span>Fantasy Nerdball</span>
          <div className="foot-actions">
            <button
              className="admin-link"
              type="button"
              onClick={() => setTourOpen(true)}
            >
              Tutorial
            </button>
            <button
              className="admin-link"
              type="button"
              onClick={() => setFeedback(true)}
            >
              Feedback
            </button>
            <button
              className="admin-link"
              type="button"
              onClick={() => setPrivacy(true)}
            >
              Privacy
            </button>
            <span className="beta-flag" title="Free, unfunded, and small">
              Beta
            </span>
            {guest && (
              <button
                className="admin-link"
                type="button"
                onClick={() => setGuestInfo(true)}
              >
                Guest limits
              </button>
            )}
            <ThemePicker />
            {me.is_admin && (
              <button
                className="admin-link"
                type="button"
                onClick={() => setAdminOpen(true)}
              >
                Admin
              </button>
            )}
          </div>
        </footer>
      </main>

      {/* Mounted inside the signed-in tree, so nothing asks the server for
          settings before there's an account to ask about. */}
      <FirstRunTour onOpen={() => setTourOpen(true)} />
      <Tutorial
        open={tourOpen}
        onClose={() => setTourOpen(false)}
        onTab={setTab}
      />

      {guestInfo && (
        <GuestFeaturesDialog onClose={() => setGuestInfo(false)} />
      )}

      {feedback && <FeedbackDialog onClose={() => setFeedback(false)} />}

      {privacy && <PrivacyNotice onClose={() => setPrivacy(false)} />}

      {adminOpen && (
        <div
          className="overlay"
          role="dialog"
          aria-modal="true"
          aria-label="Admin"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setAdminOpen(false);
          }}
        >
          <AdminView onClose={() => setAdminOpen(false)} />
        </div>
      )}
    </div>
    </GuestProvider>
  );
}
