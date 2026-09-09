import { useEffect, useState, type ReactNode } from "react";
import { useSettings } from "../lib/settingsStore";

export type TourTab = "squad" | "planner" | "players" | "teams" | "setup";

/* ── Illustrations ───────────────────────────────────────────────────────
   Drawn here rather than screenshotted: they follow the theme, stay sharp
   at any size, and never go out of date when the UI moves. */

function ArtStart() {
  return (
    <svg viewBox="0 0 220 120" aria-hidden="true">
      {[0, 1, 2].map((i) => (
        <g key={i} transform={`translate(${18 + i * 68} 34)`}>
          <circle cx="20" cy="20" r="18" className="ink-line" />
          <text x="20" y="26" textAnchor="middle" className="ink-num">
            {i + 1}
          </text>
        </g>
      ))}
      <path d="M58 54h22M126 54h22" className="ink-line arrow" />
      <text x="38" y="98" textAnchor="middle" className="ink-cap">
        Last week
      </text>
      <text x="106" y="98" textAnchor="middle" className="ink-cap">
        Settings
      </text>
      <text x="174" y="98" textAnchor="middle" className="ink-cap">
        Run
      </text>
    </svg>
  );
}

function ArtPitch() {
  const rows = [
    [110],
    [40, 87, 133, 180],
    [40, 87, 133, 180],
    [70, 150],
  ];
  return (
    <svg viewBox="0 0 220 120" aria-hidden="true">
      <rect x="4" y="4" width="212" height="88" rx="4" className="ink-turf" />
      <line x1="4" y1="48" x2="216" y2="48" className="ink-line faint" />
      <circle cx="110" cy="48" r="14" className="ink-line faint" />
      {rows.map((row, r) =>
        row.map((x) => (
          <rect
            key={`${r}-${x}`}
            x={x - 9}
            y={12 + r * 19}
            width="18"
            height="13"
            rx="2"
            className={r === 3 ? "ink-fill-accent" : "ink-fill"}
          />
        )),
      )}
      <rect x="72" y="100" width="76" height="16" rx="3" className="ink-btn" />
      <text x="110" y="111" textAnchor="middle" className="ink-btn-text">
        Run optimiser
      </text>
    </svg>
  );
}

function ArtTransfers() {
  return (
    <svg viewBox="0 0 220 120" aria-hidden="true">
      <g transform="translate(26 24)">
        <path d="M8 2v20M3 16l5 6 5-6" className="ink-out" />
        <rect x="26" y="4" width="80" height="14" rx="3" className="ink-fill" />
      </g>
      <g transform="translate(26 62)">
        <path d="M8 22V2M3 8l5-6 5 6" className="ink-in" />
        <rect x="26" y="4" width="96" height="14" rx="3" className="ink-fill" />
      </g>
      <g transform="translate(150 24)">
        <rect width="46" height="72" rx="3" className="ink-panel" />
        <path d="M9 14h28M9 28h28M9 42h20M9 56h24" className="ink-line faint" />
      </g>
      <text x="173" y="112" textAnchor="middle" className="ink-cap">
        Why
      </text>
    </svg>
  );
}

function ArtPlan() {
  return (
    <svg viewBox="0 0 220 120" aria-hidden="true">
      {[0, 1, 2, 3, 4].map((c) => (
        <text key={c} x={54 + c * 34} y="16" textAnchor="middle" className="ink-cap">
          GW{c + 1}
        </text>
      ))}
      {[
        [0, 5],
        [0, 3],
        [2, 5],
        [0, 2],
      ].map(([from, to], r) => (
        <g key={r}>
          <rect x="10" y={26 + r * 22} width="30" height="10" rx="2" className="ink-fill" />
          <rect
            x={40 + from * 34}
            y={24 + r * 22}
            width={(to - from) * 34 - 6}
            height="14"
            rx="3"
            className={r === 1 ? "ink-fill-accent" : "ink-fill-gain"}
          />
        </g>
      ))}
      <text x="110" y="114" textAnchor="middle" className="ink-cap">
        who you keep, and for how long
      </text>
    </svg>
  );
}

function ArtTable() {
  return (
    <svg viewBox="0 0 220 120" aria-hidden="true">
      <rect x="8" y="8" width="204" height="104" rx="4" className="ink-panel" />
      <path d="M8 30h204" className="ink-line faint" />
      {[0, 1, 2, 3].map((i) => (
        <g key={i} transform={`translate(0 ${44 + i * 20})`}>
          <rect
            x="8"
            y="-13"
            width="204"
            height="18"
            className={i === 1 ? "ink-row-on" : "ink-row"}
          />
          <rect x="20" y="-9" width="54" height="9" rx="2" className="ink-fill" />
          <rect x="112" y="-9" width="22" height="9" rx="2" className="ink-line-fill" />
          <rect x="150" y="-9" width="22" height="9" rx="2" className="ink-line-fill" />
          <rect x="182" y="-9" width="20" height="9" rx="2" className="ink-line-fill" />
        </g>
      ))}
      <circle cx="47" cy="51" r="15" className="ink-tap" />
    </svg>
  );
}

function ArtTeams() {
  return (
    <svg viewBox="0 0 220 120" aria-hidden="true">
      <text x="12" y="24" className="ink-cap">
        xG
      </text>
      <rect x="40" y="14" width="120" height="12" rx="2" className="ink-fill-gain" />
      <text x="12" y="48" className="ink-cap">
        xGC
      </text>
      <rect x="40" y="38" width="62" height="12" rx="2" className="ink-fill-flag" />
      <text x="12" y="82" className="ink-cap">
        Att
      </text>
      <text x="12" y="104" className="ink-cap">
        Def
      </text>
      {[0, 1, 2, 3].map((i) => (
        <g key={i}>
          <rect
            x={40 + i * 34}
            y={70}
            width="30"
            height="16"
            rx="2"
            className={i < 2 ? "ink-fill-gain" : "ink-fill-mid"}
          />
          <rect
            x={40 + i * 34}
            y={92}
            width="30"
            height="16"
            rx="2"
            className={i === 0 ? "ink-fill-mid" : "ink-fill-flag"}
          />
        </g>
      ))}
    </svg>
  );
}

function ArtWeights() {
  return (
    <svg viewBox="0 0 220 120" aria-hidden="true">
      <rect x="14" y="30" width="90" height="18" rx="3" className="ink-w-form" />
      <rect x="104" y="30" width="58" height="18" rx="3" className="ink-w-history" />
      <rect x="162" y="30" width="44" height="18" rx="3" className="ink-w-fixtures" />
      <circle cx="104" cy="39" r="7" className="ink-handle" />
      <circle cx="162" cy="39" r="7" className="ink-handle" />
      <text x="14" y="66" className="ink-cap">
        Form
      </text>
      <text x="104" y="66" className="ink-cap">
        History
      </text>
      <text x="162" y="66" className="ink-cap">
        Fixtures
      </text>
      <rect x="14" y="82" width="86" height="26" rx="3" className="ink-panel" />
      <rect x="120" y="82" width="86" height="26" rx="3" className="ink-panel" />
      <text x="57" y="99" textAnchor="middle" className="ink-cap">
        Must pick
      </text>
      <text x="163" y="99" textAnchor="middle" className="ink-cap">
        Avoid
      </text>
    </svg>
  );
}

function ArtScore() {
  return (
    <svg viewBox="0 0 220 120" aria-hidden="true">
      {["Form", "History", "Fixtures"].map((label, i) => (
        <g key={label}>
          <rect
            x="10"
            y={12 + i * 32}
            width="72"
            height="24"
            rx="3"
            className="ink-panel"
          />
          <text x="46" y={28 + i * 32} textAnchor="middle" className="ink-cap">
            {label}
          </text>
          <path
            d={`M86 ${24 + i * 32}H124`}
            className="ink-line arrow"
          />
        </g>
      ))}
      <rect x="128" y="34" width="82" height="46" rx="4" className="ink-btn" />
      <text x="169" y="55" textAnchor="middle" className="ink-btn-text">
        One score
      </text>
      <text x="169" y="70" textAnchor="middle" className="ink-btn-text small">
        per player
      </text>
    </svg>
  );
}

function ArtDone() {
  return (
    <svg viewBox="0 0 220 120" aria-hidden="true">
      <circle cx="110" cy="52" r="30" className="ink-line" />
      <path d="M96 52l10 11 20-24" className="ink-tick" />
      <rect x="46" y="94" width="128" height="18" rx="3" className="ink-panel" />
      <text x="110" y="107" textAnchor="middle" className="ink-cap">
        Tutorial · in the footer
      </text>
    </svg>
  );
}

function ArtScores() {
  return (
    <svg viewBox="0 0 220 120" aria-hidden="true">
      {/* An option button, with the pair of numbers it carries. */}
      <rect
        x="56"
        y="22"
        width="108"
        height="76"
        rx="4"
        className="ink-line"
      />
      <text x="72" y="44" className="ink-cap">
        OPTION 1
      </text>
      <text x="72" y="72" className="ink-num">
        60.4
      </text>
      <text x="72" y="88" className="ink-cap">
        28.7 nerdball
      </text>
    </svg>
  );
}

interface Step {
  title: string;
  tab?: TourTab;
  art: () => ReactNode;
  body: ReactNode;
}

const STEPS: Step[] = [
  {
    title: "Getting started",
    art: ArtStart,
    body: (
      <>
        <p>
          Welcome to the Fantasy Nerdball optimiser; a tool to let you combine 
          your gut instinct with data and statistics to give you the edge in FPL.
          The process is simple:
        </p>
        <ol>
          <li>
            <strong>Tell it what you had last week.</strong> Import your real
            team, or enter it by hand on a blank pitch.
          </li>
          <li>
            <strong>Set your budget, free transfers and any chip</strong> on the
            Squad page.
          </li>
          <li>
            <strong>Press Run.</strong> A minute later you have a squad and the
            reasoning behind it.
          </li>
        </ol>
      </>
    ),
  },
  {
    title: "Squad",
    tab: "squad",
    art: ArtPitch,
    body: (
      <>
        <p>
          The 'Squad' page is the home of the optimiser. This is where you'll see either
          your squad from last week, or your most recent optimisation for the next gameweek.
          A padlock on a player means you have a good feeling about them, and you've chosen
          to lock them in, no matter what the model says.
        </p>
        <p>
          Above it sit the three things that change weekly — budget, free
          transfers and chips. They save as you change them, so the 'Run' button always uses
          what's on screen.
        </p>
        <p className="tour-tip">
          Tap any player for their stats, and to force them in or rule them
          out of future squads.
        </p>
      </>
    ),
  },
  {
    title: "Squad Transfers",
    tab: "squad",
    art: ArtTransfers,
    body: (
      <>
        <p>
          After you run the optimiser, the model may suggest some transfers.
          Red arrows are out, green in. Underneath is what the move is worth:
          the extra points a week the new side projects over keeping the old
          one.
        </p>
        <p>
          Under the pitch, every player's score is broken down line by line, so
          you can see why someone was picked rather than take it on trust.
        </p>
      </>
    ),
  },
  {
    title: "Nerdball Points vs Projected Points",
    tab: "squad",
    art: ArtScores,
    body: (
      <>
        <p>
          Each option carries two figures.{" "}
          <strong>Projected points</strong>, the large one, is what the
          starting eleven should score in the coming gameweek. The{" "}
          <strong>nerdball score</strong> beneath it is the model's own rating
          of the whole squad, built from form, historic returns, likeliness to play, and fixture
          difficulty across however many gameweeks you set it to look ahead.
        </p>
        <p>
          Squads are picked on the nerdball score rather than the projection,
          because a side has to hold up beyond the weekend. A player with a kind
          run of fixtures is worth having even in a week he is not the highest
          projected, and the projection alone cannot see that.
        </p>
        <p>
          The exception is a Free Hit, where the squad reverts next week. There
          is no beyond-Saturday to plan for, so that week is picked on
          projected points instead.
        </p>
        <p className="tour-tip">
          The recommended option is not simply the highest of either number.
          It is the one where every transfer earned its keep, which is what
          Transfer strategy on the Setup page controls.
        </p>
      </>
    ),
  },
  {
    title: "Planner",
    tab: "planner",
    art: ArtPlan,
    body: (
      <>
        <p>
          Runs the optimiser forward for three to eight gameweeks, feeding each
          week's squad into the next. Free transfers roll over exactly as they
          do in the real game, so it can tell you to hold this week and make a
          double move next.
        </p>
        <p>
          Say which gameweek you mean to play each chip in and it plans around
          them.
        </p>
        <p className="tour-tip">
          It can't see price rises or a player losing form, so trust the first
          week or two and treat the rest as a sketch.
        </p>
      </>
    ),
  },
  {
    title: "Players",
    tab: "players",
    art: ArtTable,
    body: (
      <>
        <p>
          <strong>Best picks</strong> ranks everyone by score, by position.{" "}
          <strong>Differentials</strong> is the same list with the popular
          players removed — useful for catching up in a mini-league.
        </p>
        <p>
          <strong>Lookup</strong> gives one player's full history and next
          fixtures.
        </p>
        <p className="tour-tip">
          Tap a row to force that player into your squad or add them to your
          avoid list.
        </p>
      </>
    ),
  },
  {
    title: "Teams",
    tab: "teams",
    art: ArtTeams,
    body: (
      <>
        <p>
          <strong>xG</strong> is the quality of chances a side creates, and{" "}
          <strong>xGC</strong> the chances they let in. They tell you how good a
          team really is, before luck has its say.
        </p>
        <p>
          Fixtures get two difficulty ratings for attackers and defenders, because
          reducing a team to a single difficulty rating doesn't give you the full story.
          This is how the model sees the teams.
        </p>
      </>
    ),
  },
  {
    title: "Setup",
    tab: "setup",
    art: ArtWeights,
    body: (
      <>
        <p>
          Drag the bar to say what matters most: recent <strong>form</strong>,
          longer-term <strong>history</strong>, or upcoming{" "}
          <strong>fixtures</strong>. There's no right answer — it's your call on
          how much to trust a hot streak.
        </p>
        <p>
          Below that, players you always want and players you never do, plus
          your FPL account link.
        </p>
      </>
    ),
  },
  {
    title: "What the score means",
    art: ArtScore,
    body: (
      <>
        <p>
          Every player gets one number, built from how they're playing now, what
          they've done historically, and who they're about to face. It's not just
          their FPL numbers, but it also considers their xG performance - both past
          and present.
        </p>
        <p>
          It's roughly "points we'd expect per game", so bigger is better and you can compare
          any two players directly.
        </p>
        <p className="tour-tip">
          You don't need to understand the maths to use it. If a number ever
          looks wrong, the breakdown under the pitch shows exactly where it came
          from.
        </p>
      </>
    ),
  },
  {
    title: "That's it",
    art: ArtDone,
    body: (
      <>
        <p>
          Start by getting last week's squad in, then press Run. Everything else
          can wait until you're curious.
        </p>
        <p>
          This tour is always in the footer under <strong>Tutorial</strong> if
          you want it again.
        </p>
      </>
    ),
  },
];

export function Tutorial({
  open,
  onClose,
  onTab,
}: {
  open: boolean;
  onClose: () => void;
  onTab: (tab: TourTab) => void;
}) {
  const { settings, patch } = useSettings();
  const [index, setIndex] = useState(0);

  // Follow along in the app behind the overlay: reading about the Players tab
  // while looking at the Squad one would be a poor introduction.
  useEffect(() => {
    if (!open) return;
    const tab = STEPS[index]?.tab;
    if (tab) onTab(tab);
  }, [open, index, onTab]);

  useEffect(() => {
    if (open) setIndex(0);
  }, [open]);

  const finish = () => {
    onClose();
    // Remembered per account, so it doesn't reappear on another device.
    if (settings && !settings.tutorial_seen) {
      void patch({ tutorial_seen: true }).catch(() => undefined);
    }
  };

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") finish();
      if (event.key === "ArrowRight") setIndex((i) => Math.min(STEPS.length - 1, i + 1));
      if (event.key === "ArrowLeft") setIndex((i) => Math.max(0, i - 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (!open) return null;

  const step = STEPS[index];
  const last = index === STEPS.length - 1;
  const Art = step.art;

  return (
    <div
      className="overlay tour-overlay"
      role="dialog"
      aria-modal="true"
      aria-label="Guided tour"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) finish();
      }}
    >
      <div className="tour-card">
        <div className="tour-art">
          <Art />
        </div>

        <div className="tour-body">
          <div className="tour-head">
            <span className="tour-count">
              {index + 1} of {STEPS.length}
            </span>
            <button className="link-button" type="button" onClick={finish}>
              Skip
            </button>
          </div>

          <h2>{step.title}</h2>
          <div className="tour-copy">{step.body}</div>

          <div className="tour-foot">
            <div className="tour-dots" role="tablist" aria-label="Tour steps">
              {STEPS.map((entry, i) => (
                <button
                  key={entry.title}
                  type="button"
                  role="tab"
                  aria-selected={i === index}
                  aria-label={entry.title}
                  title={entry.title}
                  className={i === index ? "is-on" : ""}
                  onClick={() => setIndex(i)}
                />
              ))}
            </div>
            <div className="tour-actions">
              {index > 0 && (
                <button
                  className="btn quiet small"
                  type="button"
                  onClick={() => setIndex(index - 1)}
                >
                  Back
                </button>
              )}
              <button
                className="btn small"
                type="button"
                onClick={() => (last ? finish() : setIndex(index + 1))}
              >
                {last ? "Get started" : "Next"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/** Opens the tour by itself the first time an account sees the app. */
export function FirstRunTour({ onOpen }: { onOpen: () => void }) {
  const { settings } = useSettings();
  const [fired, setFired] = useState(false);

  useEffect(() => {
    if (fired || !settings || settings.tutorial_seen) return;
    setFired(true);
    onOpen();
  }, [settings, fired, onOpen]);

  return null;
}
