import { clubAbbr, clubColours } from "../lib/clubs";
import type { Player } from "../lib/types";

function readableOn(hex: string): string {
  const value = hex.replace("#", "");
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  // Rec. 709 luminance: pale kits (Spurs, Fulham, Leeds) need dark lettering.
  const luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  return luminance > 0.6 ? "#14110a" : "#ffffff";
}

/* Small enough to sit in the name band without pushing the name out: a solid
   body reads at 10px where an outlined one turns to mush. */
function LockIcon() {
  return (
    <svg className="lock-mark" viewBox="0 0 12 14" aria-hidden="true">
      <path
        d="M3.4 6.2V4.3a2.6 2.6 0 0 1 5.2 0v1.9"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      <rect x="1.5" y="6.2" width="9" height="6.8" rx="1.4" fill="currentColor" />
    </svg>
  );
}

interface Props {
  player: Player;
  delay?: number;
  onSelect?: (player: Player) => void;
  /** Whether this name is on the forced-picks list. */
  forced?: boolean;
}

export function PlayerShirt({ player, delay = 0, onSelect, forced = false }: Props) {
  const colours = clubColours(player.team);
  const doubtful = player.status !== "a" && player.status !== "";
  const abbr = clubAbbr(player.team, player.team_short);

  const venue =
    player.venue === "Home" ? " (H)" : player.venue === "Away" ? " (A)" : "";
  const fixture = player.next_opponent
    ? `${abbr} · ${player.next_opponent}${venue}`
    : abbr;

  const title = [
    `${player.name} · ${player.team} · ${player.position}`,
    forced ? "Forced pick — the optimiser has to include them" : "",
    `£${player.price.toFixed(1)}m · ${player.projected_points.toFixed(1)} projected`,
    player.form !== null ? `form ${player.form}` : "",
    player.start_rate !== null ? `starts ${player.start_rate}%` : "",
    player.is_double_gameweek ? "Double gameweek" : "",
    player.news ? player.news : "",
  ]
    .filter(Boolean)
    .join("\n");

  const className = [
    "shirt",
    player.is_captain ? "is-captain" : "",
    forced ? "is-forced" : "",
    doubtful ? "is-doubtful" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <button
      type="button"
      className={className}
      style={{ animationDelay: `${delay}ms` }}
      title={title}
      onClick={() => onSelect?.(player)}
    >
      {/* The name sits on the club colour. It's what you scan the pitch for,
          so it gets the full width of the card and the strongest position. */}
      <span
        className="band"
        style={{ background: colours.band, color: readableOn(colours.band) }}
      >
        {forced && <LockIcon />}
        <span className="name">{player.name}</span>
        {player.is_double_gameweek && <span className="dgw-mark">••</span>}
      </span>

      {player.is_captain && <span className="armband">C</span>}
      {player.is_vice && <span className="armband vice">V</span>}
      {player.on_bench && player.bench_order !== null && player.position !== "GK" && (
        <span className="bench-number">{player.bench_order - 1}</span>
      )}

      <span className="points">{player.projected_points.toFixed(1)}</span>
      <span className="meta">{fixture}</span>
      <span className="price">£{player.price.toFixed(1)}m</span>
    </button>
  );
}
