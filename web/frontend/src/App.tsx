import { Suspense, lazy, useEffect, useState } from "react";
import { LeagueView } from "./components/LeagueView";
import { SetupView } from "./components/SetupView";
import { SignIn } from "./components/SignIn";
import { SquadView } from "./components/SquadView";

// The Form page is the only thing that needs the charting library, so it is
// fetched when someone opens that tab rather than on first paint.
const FormView = lazy(() =>
  import("./components/FormView").then((m) => ({ default: m.FormView })),
);
import { api, ApiError } from "./lib/api";
import type { Me } from "./lib/types";

type Tab = "squad" | "form" | "league" | "setup";

const TABS: { id: Tab; label: string }[] = [
  { id: "squad", label: "Squad" },
  { id: "form", label: "Form" },
  { id: "league", label: "League" },
  { id: "setup", label: "Setup" },
];

export function App() {
  const [me, setMe] = useState<Me | null>(null);
  const [checked, setChecked] = useState(false);
  const [tab, setTab] = useState<Tab>("squad");

  useEffect(() => {
    api
      .me()
      .then(setMe)
      .catch((err) => {
        if (!(err instanceof ApiError) || err.status !== 401) {
          console.error(err);
        }
      })
      .finally(() => setChecked(true));
  }, []);

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
        {tab === "form" && (
          <Suspense fallback={<p className="muted">Loading your season…</p>}>
            <FormView me={me} />
          </Suspense>
        )}
        {tab === "league" && <LeagueView />}
        {tab === "setup" && <SetupView me={me} onMeChange={setMe} />}
      </main>
    </div>
  );
}
