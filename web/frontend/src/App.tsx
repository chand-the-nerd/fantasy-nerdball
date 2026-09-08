import { useEffect, useState } from "react";
import { AdminView } from "./components/AdminView";
import { PlayersView } from "./components/PlayersView";
import { SetupView } from "./components/SetupView";
import { TeamsView } from "./components/TeamsView";
import { SignIn } from "./components/SignIn";
import { SquadView } from "./components/SquadView";
import { ThemePicker } from "./components/ThemePicker";
import { api, ApiError } from "./lib/api";
import type { Me } from "./lib/types";

// Form and League are built and working, but hidden for now. To bring either
// back, add it to TABS and render it below — the components and their API
// routes are untouched.
type Tab = "squad" | "players" | "teams" | "setup";

const TABS: { id: Tab; label: string }[] = [
  { id: "squad", label: "Squad" },
  { id: "players", label: "Players" },
  { id: "teams", label: "Teams" },
  { id: "setup", label: "Setup" },
];

export function App() {
  const [me, setMe] = useState<Me | null>(null);
  const [checked, setChecked] = useState(false);
  const [tab, setTab] = useState<Tab>("squad");
  const [adminOpen, setAdminOpen] = useState(false);

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

  const signOut = () => api.logout().then(() => window.location.reload());

  return (
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
          {me.avatar_url && <img src={me.avatar_url} alt="" />}
          <span className="who">
            <strong>{me.name}</strong>
            <button className="link-button" onClick={signOut} type="button">
              Sign out
            </button>
          </span>
        </div>
      </aside>

      <main className="main">
        {tab === "squad" && <SquadView />}
        {tab === "players" && <PlayersView />}
        {tab === "teams" && <TeamsView />}
        {tab === "setup" && <SetupView me={me} onMeChange={setMe} />}

        <footer className="app-foot">
          <span>Fantasy Nerdball</span>
          <div className="foot-actions">
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
  );
}
