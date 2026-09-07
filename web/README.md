# Fantasy Nerdball, on the web

A browser front end for the [fantasy-nerdball](https://github.com/a-m-c-z/fantasy-nerdball)
FPL optimiser. Up to six managers sign in with Google, each keeps their own
settings and squad history, and each week's optimised side is drawn on a pitch
and tracked against the global average.

The model itself is untouched. This repository wraps it.

---

## How it fits together

```
  browser ── React ── FastAPI ─┬─ Postgres      users, settings, squads, results
                              │
                              ├─ worker thread ─ the optimiser (main.py, src/)
                              │                    │
                              └─ volume ───────────┘  data/ and squads/gw*/
```

The optimiser is copied out of this same repository during the Docker build,
so the model and the app serving it are always the same commit. Push a model
change, redeploy, and it's live. Nothing is vendored, cloned or forked.

### Three things the CLI tool assumed, and how each is handled

**Config is a file you edit by hand.** `app/engine/runtime_config.py` builds a
`Config` subclass per run from that manager's database row, so six people can
hold six different sets of preferences against one deployment. Only keys in
`ALLOWED_OVERRIDES` can be set, so a stray value can't reach into the engine.

It also fixes two things that otherwise break on startup: `CURRENT_SEASON` is
removed from `PAST_SEASONS` if present, and `HISTORIC_SEASON_WEIGHTS` is
renormalised to sum to 1.0 afterwards. Worth knowing that's happening rather
than being surprised by it in August.

**Relative paths against an ephemeral disk.** The engine reads and writes
`data/` and `squads/gw{n}/` from the working directory. Each manager gets a
directory on a mounted volume; the shared reference data (team ratings,
standings history, player history) is symlinked in, so it isn't downloaded six
times over. Squads live in Postgres as JSON and are written back out as the
exact `full_squad.csv` the engine expects to find, so week-to-week transfer
logic works.

**`os.chdir` is process-wide.** Runs are serialised behind a lock and executed
by a single worker thread. This is also why the container runs one uvicorn
worker — a second would fight the first over the working directory.

Because a run takes minutes, `POST /api/runs` returns immediately and the
browser polls. The engine's own `print` output is captured and streamed to a
console in the UI, so you can watch it work.

---

## Deploying to Railway

### 1. Create the service

Point a new Railway service at this repository and leave **Root Directory
empty** — the build context has to be the repository root, because the image
needs your optimiser as well as the web app. `railway.json` selects the
Dockerfile builder and sets the health check, so there's nothing else to
configure.

### 2. Add Postgres and a volume

Add the Postgres plugin — it injects `DATABASE_URL` automatically. Then attach
a volume mounted at `/data`. Without the volume the app still runs, but cached
reference data is re-downloaded after every redeploy and each run gets slower.

### 3. Set up Google sign-in

In the Google Cloud console, create an OAuth 2.0 Client ID of type **Web
application**, and add this authorised redirect URI:

```
https://<your-railway-domain>/api/auth/callback
```

Configure the consent screen as **External**, then add each friend's Google
address as a test user. That's the second gate; the app's own allowlist is the
first.

### 4. Set the variables

| Variable | Notes |
| --- | --- |
| `SECRET_KEY` | `openssl rand -hex 32`. Rotating it signs everyone out. |
| `GOOGLE_CLIENT_ID` | From the console. |
| `GOOGLE_CLIENT_SECRET` | From the console. |
| `ALLOWED_EMAILS` | Your address. First one listed becomes the owner. |
| `CURRENT_SEASON` | e.g. `2026-27`. Must not be in the engine's `PAST_SEASONS`. |
| `MAX_USERS` | Defaults to 6. |
| `NERDBALL_DATA_DIR` | `/data`, matching the volume mount. |

`DATABASE_URL` and `RAILWAY_PUBLIC_DOMAIN` are injected by Railway. Set
`PUBLIC_BASE_URL` only if you move to a custom domain.

### 5. Invite the others

Sign in yourself first — the first address in `ALLOWED_EMAILS` is granted owner
rights. Then add the other five from **Setup → Managers**. They sign in with
Google; no passwords anywhere.

---

## Running it locally

```bash
cp .env.example .env      # fill in SECRET_KEY at minimum
docker compose up --build
```

Then open http://localhost:8000 and use the developer sign-in link, which
skips Google entirely. It only appears when `DEV_MODE=true`.

For a proper local loop with hot reload, see `DEVELOPING.md`. The short
version is two terminals:

```bash
# terminal 1
./web/dev.sh

# terminal 2
cd web/frontend && npm run dev
```

Then open http://localhost:5173 — Vite proxies `/api` to port 8000. Run
`python web/backend/seed_demo.py` to put a squad on screen without waiting for
a real optimisation.

---

## Using it

**Setup** holds your budget, free transfers, chips, and the model knobs worth
changing week to week — the transfer threshold, how many fixtures ahead the
difficulty looks, how long a transfer is assumed to be held. Each manager's
settings are their own.

**Squad** runs the optimiser and draws the result. Click any shirt for its
form, historic points per game, fixture difficulty, start rate and expected
goals modifier. The side panel says whether to transfer or hold and why, and
compares your XI against what the model would pick with a free hand.

**Form** charts your points against the global average and the week's highest
score, plus the model's projection so you can see how well it's calibrated.
Link your FPL team id under Setup and this fills in automatically each week;
otherwise enter scores by hand.

**League** is the six of you side by side, with the global average as a line in
the table — so it's clear who's actually beating the game rather than just each
other.

---

## Notes

- The optimiser fetches three seasons of history from the vaastav dataset on
  first run of a season. That run is slow; later ones reuse the cached files on
  the volume.
- Chips are set per gameweek and don't clear themselves. Turn Wildcard or Bench
  Boost back off after the deadline, or the next run will still assume it.
- `scikit-learn` is installed so the ML weight training scripts can be run as a
  Railway one-off command. Drop it from `requirements.txt` if you'd rather have
  a smaller image and train weights locally.
- The container starts as root only long enough for `entrypoint.sh` to take
  ownership of the mounted volume, which arrives owned by root and would
  otherwise be unwritable. It then drops to the unprivileged `nerdball` user
  via `gosu`. The application itself never runs as root, and the `chown` is
  skipped on reboots once the mount is already owned correctly.
- Sessions are signed cookies with a thirty-day life, `Secure` and `SameSite=Lax`.
  `DEV_MODE` relaxes the `Secure` flag for local HTTP and is the only thing that
  enables the developer sign-in route.
