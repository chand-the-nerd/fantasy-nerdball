# Visual direction

**Subject.** A private FPL model dashboard for six people who already know
FPL's own interface. Its two jobs: show this week's optimised side on a pitch,
and say whether the model is beating the crowd.

## Palette

Floodlit night football, not generic dark-mode SaaS. The ground is a deep
green-black — grass under floodlights — rather than a tinted near-black grey.

| Token | Hex | Role |
| --- | --- | --- |
| `--turf-night` | `#0B1A16` | page ground |
| `--panel` | `#10241E` | raised surfaces |
| `--line` | `#1E3B33` | hairline borders, pitch markings |
| `--chalk` | `#EAF2EE` | primary text |
| `--fade` | `#7C978D` | secondary text |
| `--floodlight` | `#F2C14E` | accent: sodium-lamp amber. Model picks, captain, primary actions |
| `--crowd` | `#56C7E8` | reserved for the global-average benchmark, nothing else |
| `--flag` | `#E5484D` | injuries, unavailability, points hits |

Two accents, each with exactly one meaning: amber is you, blue is everyone
else. That is the whole comparison the app exists to make.

## Type

One family, **Archivo**, used across the app. **Archivo Narrow** appears in
one place only — surnames on the shirt cards — because horizontal space there
is genuinely tight, not for decoration. Tabular figures everywhere, since the
app is columns of numbers that need to line up. No monospace.

Scale: 12 / 14 / 16 / 21 / 28 / 42.

## Layout

```
┌─────────┬──────────────────────────────────────────────┐
│         │  GW7 · deadline Sat 11:30      [ Optimise ]  │
│  rail   ├──────────────────────────────────────────────┤
│         │  ┌────────────────────────┐ ┌──────────────┐ │
│  Squad  │  │  scoreline bar         │ │ transfers    │ │
│  Form   │  │  ─────────────────     │ │ in / out     │ │
│  League │  │      THE PITCH         │ │              │ │
│  Setup  │  │                        │ │ model's own  │ │
│         │  │  ── bench strip ──     │ │ XI           │ │
│         │  └────────────────────────┘ └──────────────┘ │
└─────────┴──────────────────────────────────────────────┘
```

Left aligned throughout; numbers right aligned in tables.

## Principles

- **The pitch is the hero.** Real chalk markings in SVG, faint mowing stripes,
  the bench inset below it the way FPL does. Everything else is flat,
  hairline-bordered, no shadows, no gradient washes.
- **No stat-card grid.** The summary is a single scoreline bar above the
  pitch, read like a broadcast lower-third.
- **One piece of motion.** When a run finishes, the side settles onto the
  pitch line by line. Nothing else animates on entry. `prefers-reduced-motion`
  is respected.
- Boldness is spent in one place. The navigation is deliberately quiet.
