/**
 * A modifier slider per club, running 0 to 2 with 1.00 as the neutral centre.
 *
 * Everything sits at the centre by default, so the eye picks out the handles
 * you've actually moved without needing to read any numbers. Twenty rows are
 * laid out in columns rather than one long list, which is what lets this
 * panel earn the full width of the page.
 */
export function TeamSliders({
  teams,
  modifiers,
  onChange,
}: {
  teams: string[];
  modifiers: Record<string, number>;
  onChange: (modifiers: Record<string, number>) => void;
}) {
  const valueFor = (team: string) => modifiers[team] ?? 1;

  const set = (team: string, value: number) => {
    const next = { ...modifiers };
    // Don't persist neutral values — they're the default, and storing twenty
    // 1.00s makes a diff of the settings unreadable.
    if (Math.abs(value - 1) < 0.001) delete next[team];
    else next[team] = Math.round(value * 100) / 100;
    onChange(next);
  };

  const adjusted = teams.filter((team) => Math.abs(valueFor(team) - 1) >= 0.001);

  return (
    <div className="team-sliders">
      <div className="team-sliders-head">
        <span className="muted">
          {adjusted.length === 0 ? "All clubs neutral" : `${adjusted.length} adjusted`}
        </span>
        {adjusted.length > 0 && (
          <button className="link-button" type="button" onClick={() => onChange({})}>
            Reset all
          </button>
        )}
      </div>

      <div className="team-slider-columns">
        {teams.map((team) => {
          const value = valueFor(team);
          const moved = Math.abs(value - 1) >= 0.001;

          return (
            <div className={`team-slider${moved ? " is-moved" : ""}`} key={team}>
              <label htmlFor={`mod-${team}`}>{team}</label>

              <div className="team-slider-track">
                <input
                  id={`mod-${team}`}
                  type="range"
                  min={0}
                  max={2}
                  step={0.05}
                  value={value}
                  onChange={(event) => set(team, Number(event.target.value))}
                  aria-valuetext={
                    moved
                      ? `${value.toFixed(2)}, ${value > 1 ? "marked up" : "marked down"}`
                      : "1.00, neutral"
                  }
                />
                <span className="team-slider-centre" aria-hidden="true" />
              </div>

              <output
                className={moved ? (value > 1 ? "mod-up" : "mod-down") : "mod-neutral"}
              >
                {value.toFixed(2)}
              </output>
            </div>
          );
        })}
      </div>
    </div>
  );
}
