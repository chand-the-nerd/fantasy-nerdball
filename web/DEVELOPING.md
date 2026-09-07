# Working on this locally

Two terminals, hot reload, no pushing to Railway to see a CSS change.

---

## First, why your change didn't show

Before setting anything up, check the files actually landed. The most common
cause is copying them to the repository root instead of into `web/frontend/src`:

```bash
cd /path/to/fantasy-nerdball
grep -c "class=\"band\"\|className=\"band\"" web/frontend/src/components/PlayerShirt.tsx
grep -c "width: 116px" web/frontend/src/styles.css
```

Both should print `1`. If either prints `0`, the file didn't land where it
needed to. And check nothing is sitting loose at the root:

```bash
ls PlayerShirt.tsx styles.css 2>/dev/null && echo "^ these are in the wrong place"
```

If the files are right and Railway rebuilt, it's browser cache. The JS and CSS
filenames are content-hashed so they can't go stale, but `index.html` — which
points at them — can. Hard reload with **Cmd+Shift+R**, or open the site in a
private window to rule it out in one go.

---

## Setting up

You need Python (you have it) and Node. If `node --version` fails:

```bash
brew install node
```

Then install the frontend packages once:

```bash
cd web/frontend && npm install && cd ../..
```

This is also what fixes the `react/jsx-runtime` error your editor was showing —
those types live in `node_modules`, which isn't in the repository.

Install the backend packages into your active venv:

```bash
pip install -r web/backend/requirements.txt
```

---

## Running it

**Terminal 1** — the API:

```bash
./web/dev.sh
```

**Terminal 2** — the interface:

```bash
cd web/frontend && npm run dev
```

Then open **http://localhost:5173**, not 8000. Vite serves the interface and
proxies `/api` through to the Python process. Going to 8000 directly gets you
the last *built* frontend, which is exactly the stale-file confusion you're
trying to escape.

Click **Sign in as the local developer** — no Google needed.

Now edit `styles.css` and save. The browser updates without a reload, keeping
your scroll position. Edit a `.tsx` file and it re-renders in about a second.
Python changes under `web/backend/app` restart uvicorn automatically.

---

## Getting a squad on screen without waiting

A fresh local database has no squads, and a real optimisation takes minutes.
When you're working on the pitch, that loop is far too slow:

```bash
python web/backend/seed_demo.py
```

That writes a full 15-man squad plus four scored gameweeks so the Form page has
a chart. Refresh and it's there.

The demo squad is deliberately awkward — it contains the longest names in the
league (`Alexander-Arnold`, `Calvert-Lewin`, `Wan-Bissaka`), an accented name
(`João Pedro`), pale kits that need dark lettering (Spurs, Leeds), a captain
and vice-captain, a doubtful player, a suspended one, and a double gameweek. If
the card layout survives that lot it'll survive the real thing.

Re-run it any time; it overwrites the demo squad and leaves real ones alone.

---

## Running the real optimiser locally

It works, and `dev.sh` copies your existing `./data` directory into the local
workspace on first start, so it won't re-download three seasons of history you
already have.

Just remember it holds a lock and switches the working directory while it runs,
so the app is single-threaded for those couple of minutes. That's the same
behaviour as production.

---

## Where things live

| Path | |
| --- | --- |
| `web/frontend/src/styles.css` | All styling, including the pitch and cards |
| `web/frontend/src/components/PlayerShirt.tsx` | The card |
| `web/frontend/src/components/Pitch.tsx` | Pitch markings, rows, bench |
| `web/backend/app/routers/` | The API |
| `web/backend/app/engine/pipeline.py` | Where the optimiser is driven |
| `localdata/` | Local database and cached data — gitignored |

To wipe local state and start fresh:

```bash
rm -rf localdata && ./web/dev.sh
```

---

## Before pushing

```bash
cd web/frontend && npm run build
```

If that passes, Railway's build will too — it runs the same command. Catching a
TypeScript error here takes five seconds instead of five minutes.
