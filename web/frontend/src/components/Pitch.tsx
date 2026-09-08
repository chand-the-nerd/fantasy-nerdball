import { PlayerShirt } from "./PlayerShirt";
import { normalise } from "../lib/text";
import type { Player, Position } from "../lib/types";

const LINES: Position[] = ["GK", "DEF", "MID", "FWD"];

function PitchMarkings() {
  return (
    <svg
      className="pitch-lines"
      viewBox="0 0 100 130"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      {/* Halfway line across the top, with the centre circle arcing down. */}
      <line x1="0" y1="0.5" x2="100" y2="0.5" />
      <path d="M 36 0.5 A 14 11 0 0 0 64 0.5" />

      {/* Penalty area and six-yard box at the goalkeeper's end. */}
      <rect x="21" y="106" width="58" height="23.5" />
      <rect x="36" y="120" width="28" height="9.5" />
      <path d="M 36 106 A 14 10 0 0 0 64 106" />

      {/* Corner arcs. */}
      <path d="M 0 126 A 4 4 0 0 0 4 129.5" />
      <path d="M 100 126 A 4 4 0 0 1 96 129.5" />
    </svg>
  );
}

interface Props {
  starting: Player[];
  bench: Player[];
  benchBoost?: boolean;
  onSelect?: (player: Player) => void;
  /** Normalised names on the forced-picks list, for the padlock badge. */
  forced?: Set<string>;
}

export function Pitch({
  starting,
  bench,
  benchBoost = false,
  onSelect,
  forced,
}: Props) {
  const rows = LINES.map((position) =>
    starting
      .filter((player) => player.position === position)
      .sort((a, b) => b.projected_points - a.projected_points),
  ).filter((row) => row.length > 0);

  // The reveal runs line by line, so a whole row lands together.
  let index = 0;

  const benchTotal = bench.reduce((sum, p) => sum + p.projected_points, 0);

  return (
    <div className="pitch-frame">
      <div className="pitch">
        <PitchMarkings />
        <div className="lines">
          {rows.map((row, rowIndex) => (
            <div className="line-row" key={rowIndex}>
              {row.map((player) => {
                const delay = rowIndex * 90 + (index++ % 5) * 20;
                return (
                  <PlayerShirt
                    key={player.id ?? `${player.name}-${player.team}`}
                    player={player}
                    delay={delay}
                    onSelect={onSelect}
                    forced={forced?.has(normalise(player.name))}
                  />
                );
              })}
            </div>
          ))}
        </div>
      </div>

      <div className="bench">
        <div className="bench-head">
          <strong>Bench</strong>
          <span>
            {benchBoost
              ? `${benchTotal.toFixed(1)} projected, all counting`
              : `${benchTotal.toFixed(1)} projected if called on`}
          </span>
        </div>
        <div className="line-row">
          {bench.map((player, i) => (
            <PlayerShirt
              key={player.id ?? `${player.name}-bench-${i}`}
              player={player}
              delay={rows.length * 90 + i * 20}
              onSelect={onSelect}
              forced={forced?.has(normalise(player.name))}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
