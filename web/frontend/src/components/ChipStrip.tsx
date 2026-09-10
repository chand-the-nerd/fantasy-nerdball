import { useGuest } from "../lib/guest";
import type { Settings } from "../lib/types";

/* Icons are drawn here rather than pulled from a set, so they sit on the same
   1.6px stroke as everything else and inherit the tile's colour. */

function WildcardIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 20 14.5 9.5" />
      <path d="M13 8 16 11" />
      <path d="M18 3v4M16 5h4" />
      <path d="M19.5 13v3M18 14.5h3" />
      <path d="M8 3v2.5M6.75 4.25h2.5" />
    </svg>
  );
}

function BenchBoostIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 19h16" />
      <path d="M7 19v-3M17 19v-3" />
      <path d="M5 16h14" />
      <path d="M12 12V3" />
      <path d="M8.5 6.5 12 3l3.5 3.5" />
    </svg>
  );
}

function TripleCaptainIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="8.5" />
      <path d="M14.5 9.6a3.6 3.6 0 1 0 0 4.8" />
      <path d="M17.5 6.5 20 4M17.5 17.5 20 20" />
    </svg>
  );
}

function FreeHitIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3.5 14.2 8l5 .7-3.6 3.5.9 4.9-4.5-2.4-4.5 2.4.9-4.9L4.8 8.7l5-.7z" />
      <path d="M12 17.5V21" />
    </svg>
  );
}

function FreeHitPrevIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M3.5 12a8.5 8.5 0 1 0 2.6-6.1" />
      <path d="M3.5 4v4.5H8" />
      <path d="M12 8v4.5l3 1.8" />
    </svg>
  );
}

interface Chip {
  id: keyof Settings;
  label: string;
  hint: string;
  icon: () => JSX.Element;
  /** Wildcard, Bench Boost and Triple Captain can't overlap. Free Hit can. */
  exclusive: boolean;
}

const CHIPS: Chip[] = [
  {
    id: "wildcard",
    label: "Wildcard",
    hint: "Removes the transfer limit and rebuilds the squad from scratch.",
    icon: WildcardIcon,
    exclusive: true,
  },
  {
    id: "free_hit",
    label: "Free Hit",
    hint: "Unlimited transfers for this gameweek only. The squad reverts afterwards, so the look-ahead is fixed to one gameweek.",
    icon: FreeHitIcon,
    exclusive: true,
  },
  {
    id: "bench_boost",
    label: "Bench Boost",
    hint: "All fifteen players score, so the bench is optimised properly.",
    icon: BenchBoostIcon,
    exclusive: true,
  },
  {
    id: "triple_captain",
    label: "Triple Captain",
    hint: "Your captain returns three times their points.",
    icon: TripleCaptainIcon,
    exclusive: true,
  },
  {
    id: "free_hit_prev_gw",
    label: "Free Hit last week",
    hint: "Loads the squad from two gameweeks ago, since the Free Hit side has reverted.",
    icon: FreeHitPrevIcon,
    exclusive: false,
  },
];

interface Props {
  settings: Settings;
  onChange: (changes: Partial<Settings>) => void;
  disabled?: boolean;
}

export function ChipStrip({ settings, onChange, disabled = false }: Props) {
  const guest = useGuest();
  // Nothing carries over between guest sessions, so there is no Free Hit
  // side from last week for the optimiser to look past.
  const chips = guest
    ? CHIPS.filter((chip) => chip.id !== "free_hit_prev_gw")
    : CHIPS;

  const toggle = (chip: Chip) => {
    const next = !settings[chip.id];
    if (!chip.exclusive) {
      onChange({ [chip.id]: next } as Partial<Settings>);
      return;
    }
    // Only one chip can be played in a gameweek, and the server rejects two
    // anyway, so the others are cleared in the same save rather than a second.
    onChange({
      wildcard: chip.id === "wildcard" ? next : false,
      free_hit: chip.id === "free_hit" ? next : false,
      bench_boost: chip.id === "bench_boost" ? next : false,
      triple_captain: chip.id === "triple_captain" ? next : false,
    });
  };

  return (
    <div className="chip-strip" role="group" aria-label="Chips">
      {chips.map((chip) => {
        const active = Boolean(settings[chip.id]);
        const Icon = chip.icon;
        return (
          <button
            key={chip.id}
            type="button"
            className={active ? "chip-tile is-on" : "chip-tile"}
            aria-pressed={active}
            title={chip.hint}
            disabled={disabled}
            onClick={() => toggle(chip)}
          >
            <Icon />
            <span>{chip.label}</span>
          </button>
        );
      })}
    </div>
  );
}
