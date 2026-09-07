import { useEffect, useMemo, useRef, useState } from "react";
import { normalise } from "../lib/text";

export interface PoolPlayer {
  id: number;
  name: string;
  full_name: string;
  position: string;
  team: string;
  price: number;
  status: string;
  news: string;
  chance_of_playing: number | null;
  total_points: number;
  selected_by: number;
  ambiguous: boolean;
}

/** FPL availability codes, mapped to what the chip should look like. */
export function availability(status: string): {
  tone: "ok" | "doubt" | "out";
  word: string;
} {
  switch (status) {
    case "a":
      return { tone: "ok", word: "available" };
    case "d":
      return { tone: "doubt", word: "doubtful" };
    case "i":
      return { tone: "out", word: "injured" };
    case "s":
      return { tone: "out", word: "suspended" };
    case "u":
      return { tone: "out", word: "unavailable" };
    case "n":
      return { tone: "out", word: "not in squad" };
    default:
      return { tone: "doubt", word: "unknown" };
  }
}


interface Props {
  pool: PoolPlayer[];
  selected: string[];
  onChange: (names: string[]) => void;
  /** Restrict suggestions to one position. Omit to search everyone. */
  position?: string;
  limit?: number;
  placeholder?: string;
  inputId?: string;
}

export function PlayerPicker({
  pool,
  selected,
  onChange,
  position,
  limit,
  placeholder = "Start typing a name",
  inputId,
}: Props) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [highlighted, setHighlighted] = useState(0);
  const boxRef = useRef<HTMLDivElement>(null);

  const chosen = useMemo(
    () => new Set(selected.map((name) => normalise(name))),
    [selected],
  );

  const full = limit !== undefined && selected.length >= limit;

  const matches = useMemo(() => {
    const q = normalise(query.trim());
    if (!q) return [];
    // Accent-insensitive as well as case-insensitive: nobody types "Guéhi".
    return pool
      .filter((player) => {
        if (position && player.position !== position) return false;
        if (chosen.has(normalise(player.name))) return false;
        return (
          normalise(player.name).includes(q) ||
          normalise(player.full_name).includes(q) ||
          normalise(player.team).includes(q)
        );
      })
      .slice(0, 8);
  }, [query, pool, position, chosen]);

  useEffect(() => setHighlighted(0), [query]);

  useEffect(() => {
    const away = (event: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", away);
    return () => document.removeEventListener("mousedown", away);
  }, []);

  const add = (player: PoolPlayer) => {
    if (full) return;
    onChange([...selected, player.name]);
    setQuery("");
    setOpen(false);
  };

  const remove = (name: string) =>
    onChange(selected.filter((entry) => entry !== name));

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setOpen(true);
      setHighlighted((i) => Math.min(i + 1, matches.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setHighlighted((i) => Math.max(i - 1, 0));
    } else if (event.key === "Enter") {
      event.preventDefault();
      if (matches[highlighted]) add(matches[highlighted]);
    } else if (event.key === "Escape") {
      setOpen(false);
    } else if (event.key === "Backspace" && !query && selected.length) {
      remove(selected[selected.length - 1]);
    }
  };

  const lookup = (name: string) =>
    pool.find((player) => normalise(player.name) === normalise(name));

  return (
    <div className="picker" ref={boxRef}>
      {selected.length > 0 && (
        <div className="chips">
          {selected.map((name) => {
            const player = lookup(name);
            const state = player ? availability(player.status) : null;
            const tone = state ? state.tone : "unknown";
            const detail = player
              ? `${player.team} · £${player.price.toFixed(1)}m · ${state!.word}` +
                (player.news ? `\n${player.news}` : "") +
                (player.ambiguous
                  ? "\nMore than one player goes by this name; the optimiser will take the first."
                  : "")
              : "Not in this season's player pool — the optimiser will skip this name.";

            return (
              <span className={`chip tone-${tone}`} key={name} title={detail}>
                {name}
                {player?.ambiguous && <span className="chip-warn">!</span>}
                <button
                  type="button"
                  aria-label={`Remove ${name}`}
                  onClick={() => remove(name)}
                >
                  ×
                </button>
              </span>
            );
          })}
        </div>
      )}

      <input
        id={inputId}
        type="text"
        role="combobox"
        aria-expanded={open && matches.length > 0}
        aria-autocomplete="list"
        autoComplete="off"
        value={query}
        disabled={full}
        placeholder={full ? `Limit of ${limit} reached` : placeholder}
        onChange={(event) => {
          setQuery(event.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={onKeyDown}
      />

      {open && matches.length > 0 && (
        <ul className="picker-list" role="listbox">
          {matches.map((player, index) => {
            const state = availability(player.status);
            return (
              <li key={player.id} role="option" aria-selected={index === highlighted}>
                <button
                  type="button"
                  className={index === highlighted ? "is-highlighted" : ""}
                  onMouseEnter={() => setHighlighted(index)}
                  onClick={() => add(player)}
                >
                  <span className={`dot tone-${state.tone}`} />
                  <span className="picker-name">{player.name}</span>
                  <span className="picker-meta">
                    {player.position} · {player.team}
                  </span>
                  <span className="picker-price">£{player.price.toFixed(1)}m</span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
