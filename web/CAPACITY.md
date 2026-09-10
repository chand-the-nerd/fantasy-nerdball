# Capacity

How many people this can serve at once, how to find out, and how to
raise it without finding out the hard way.

## The ceiling

`os.chdir` is process-wide and the optimiser reads and writes relative
paths, so `workspace.RUN_LOCK` serialises runs **within a process**. With
`--workers 1`, that means one optimisation at a time for the entire site,
however many cores the container has.

Two different questions get called "concurrent users", and they have very
different answers:

**Browsing** — squad, teams, players, setup. Millisecond database reads
against cached FPL data, served from a thread pool. Several hundred
simultaneous users is fine. This is not the constraint.

**Running the optimiser** — one at a time. This is the constraint.

Above about 70% worker utilisation, queue waits grow faster than
throughput does: at 80% the average wait is four times a run's length, at
90% it's nine times. So usable capacity is roughly

```
runs per hour  =  0.7 × 60 / run_minutes
```

and the number of people that supports is that, divided by how many runs
each person makes in the peak. For a two-hour pre-deadline window at two
runs a head:

| Run time | Runs/hour | Users in a 2h peak |
|---|---|---|
| 1 min | 42 | ~40 |
| 2 min | 21 | ~20 |
| 3 min | 14 | ~14 |
| 5 min | 8 | ~8 |

## Measuring rather than guessing

Every completed run now reports what it cost. The **Usage** tab in the
admin pane shows all of it; the same figures are in the log as
`run_finished` events.

| What | Where | Why it matters |
|---|---|---|
| `seconds` | "Typical run", with p95 | The input to the table above |
| `waited_seconds` | "Worst queue wait" | Above a minute means the worker is behind |
| `rss_cost_mb` | "Memory a run costs" | What one parallel run needs |
| `rss_peak_mb` | — | Where that left the process overall |
| container limit | "Container" | Read from the cgroup, not assumed |
| queue depth | heartbeat | Counted from the database, so it's correct with any number of workers |

Memory is sampled every second for the length of a run and reported as a
**peak**, not an end-state: the engine builds its dataframes, solves, and
releases most of it, so a reading taken afterwards would suggest a run is
far cheaper to hold than it is.

Let a normal week accumulate before acting on any of it. One run after a
deploy is measuring the cold cache, not the workload.

## Raising it

### Step 1 — more worker processes

`WEB_CONCURRENCY` on the Railway service sets the uvicorn worker count.
Each worker is a separate process with its own working directory, and
every manager has their own workspace under `/data/managers/{id}`, so
`WEB_CONCURRENCY=4` gives four concurrent runs. The app already refuses
to start two runs for the same manager, so two processes can't collide
on one workspace.

Three things were true before that made this unsafe, and are now handled:

- **Queue depth** was counted from the in-process queue, which with four
  workers would report a quarter of the real backlog. It's now counted
  from the database.
- **The history scheduler and the heartbeat** write things that belong to
  the deployment, not to a process. They're now behind a lease
  (`main.claim`), an exclusive lock on a file on the volume, so exactly
  one worker runs them.
- **`MAX_QUEUED_RUNS`** is still per process, so the effective cap is
  `WEB_CONCURRENCY × MAX_QUEUED_RUNS`. It's a safety valve, not a
  guarantee; lower it if you raise the worker count a long way.

**Sizing it.** The dashboard's "Runs at once, on memory" is
`(limit × 0.8 − baseline) ÷ p95 run cost`, holding a fifth of the
container back — running to the stated limit is how you discover what the
OOM killer does to a queue. Take the lower of that number and your vCPU
count: a run is largely single-threaded Python, so more workers than
cores buys queueing, not throughput.

With 8 vCPU / 8 GB, if a run costs 800 MB you get about 7 on memory and 8
on CPU, so 4–6 workers is a sensible landing spot with room for the web
layer. If a run costs 2 GB, it's 2–3. Measure before choosing.

Raise it one step at a time and watch the failure rate and p95 run time
for a week between steps. Four workers doing 4× the work also means 4×
the pandas allocations against the same ceiling.

### Step 2 — separate the worker

The proper fix, when workers-in-the-web-process stops being enough: run
each optimisation in a **subprocess** that `chdir`s into its own
workspace, and move the queue into Postgres with the worker as its own
Railway service. That removes the process-wide-cwd constraint entirely
without touching the engine, lets web and worker scale independently, and
stops a long run from competing with request handling for the GIL.

At that point the sums change: capacity becomes `workers ×` the table
above, and the constraint moves to whatever the optimiser's own memory
profile allows.

### What not to bother with

Raising `numReplicas` before doing step 2 doesn't help the way it looks
like it should. Each replica gets its own in-memory queue, its own
`/data` volume mount, and the lease file is per-container rather than
per-deployment — so the scheduler would run once per replica. Horizontal
replicas are for after the queue lives in the database.

## Moving from SQLite to Postgres

The app falls back to SQLite when `DATABASE_URL` isn't set on the **app**
service. Note *app* service: a `DATABASE_URL` on the Postgres service is
Postgres's own variable and the app never sees it. Use a Railway
variable reference (`${{Postgres.DATABASE_URL}}`) so the two are linked.

Pointing the app at Postgres does not bring the old data with it —
Postgres starts empty, and it looks exactly as though every account was
deleted. `web/backend/migrate_to_postgres.py` copies it across:

The SQLite file is on the container's volume, so this runs **in the
container**. `railway run` executes locally with the service's variables
injected and never sees `/data`; `railway ssh` is the one that goes
inside:

```
railway ssh
python -m app.migrate_to_postgres --sqlite /data/nerdball.db --dry-run
```

Drop `--dry-run` when the counts look right. It copies every table in
dependency order, resets the id sequences so the next insert doesn't
collide with a row it just copied, and refuses to run against a target
that already has managers in it unless you pass `--force`.

Two things it fixes on the way through, both of which bite when moving:

- **NaN in JSON.** pandas returns NaN for a missing number — an injured
  player's chance of playing, most often. SQLite stored it as text and
  never looked; Postgres parses JSON and rejects the row, failing a run
  after the optimiser has done all the work. The database engine now
  serialises non-finite floats as null everywhere, so this is handled
  for new rows as well as migrated ones.
- **Concurrent table creation.** With `WEB_CONCURRENCY` above 1, every
  worker runs `create_all` at once, two can pass the existence check
  before either finishes, and the loser dies with a duplicate key on
  `pg_type_typname_nsp_index`, taking the deploy with it. Schema
  creation now runs under a Postgres advisory lock.

## Backups without a paid plan

`app/backup.py` writes the whole database to a compressed JSON file on
the volume once a day, keeps the last 14, and lists them in the admin
page for download. By hand, inside the container:

```
railway ssh
python -m app.backup create
python -m app.backup list
python -m app.backup restore nerdball-20260910-204759.json.gz --force
```

Restoring empties every table and refills it from the file, then resets
the id sequences. It refuses to run against a database that has users
unless you pass `--force`, because there is no undo.

**What this covers and what it doesn't.** It covers the likely disaster:
a bug, or a misconfigured `INACTIVE_DAYS`, deleting rows that mattered —
yesterday's file still has them. It does not cover losing the volume,
because the backups are on it. Downloading one occasionally from the
admin page is what closes that gap, and is the thing worth doing before
advertising.

## Tests

```
python web/backend/tests/run_all.py           # everything it can
python web/backend/tests/run_all.py --quick   # skip the Postgres ones
```

Twelve suites, each a standalone script that drives the real app through
a test client against a throwaway database. Three need a real Postgres —
migration, concurrent schema creation, and the backup round trip —
because the bugs they cover only exist there. Point `NERDBALL_TEST_PG` at
a server to run them; without it they're skipped with a note rather than
failing.

No pytest and no fixtures on purpose. Every bug that has actually bitten
this app has been an integration bug — a race between worker processes, a
NaN Postgres won't take, an orphaned row SQLite allowed — and none of
them would have been caught by a unit test with the database mocked out.

## The scored pool cache

`engine/scoring_cache.py` keeps the output of `process_player_data` for
`SCORING_CACHE_MINUTES` and hands it to any run whose scoring settings
match. Scoring is the expensive half of a run — merging past seasons,
per-player fixture difficulty across the horizon, then building scores —
and it depends on none of the things that make one manager's run
different from another's.

Guests all share locked settings, so their runs hit the cache almost
every time. Managers on the defaults hit it too. Anyone who has tuned
their weights misses it, which is correct: their scores really are
different.

The dangerous failure here would be serving one manager numbers computed
for another's settings, so the key is built the safe way round: every
uppercase attribute on the config goes in, minus a named list in
`NOT_SCORING` of fields that only affect squad selection (budget, free
transfers, forced picks, chips, transfer threshold, bench weight). A
scoring setting added later is therefore included automatically, and
forgetting to update the list costs a cache miss rather than a wrong
answer. `tests/test_scoring_cache.py` asserts both directions.

Frames are copied on the way out, because the optimiser writes into
them. The hit rate is shown in the admin Managers tab.

## Prioritising signed-in managers

One worker serves everybody, and a guest run costs exactly what a
member's does. Two things follow from that:

- Runs are queued by priority. Signed-in managers come out ahead of
  guests, first-come within each group.
- Past `GUEST_PAUSE_DEPTH` queued jobs, guest runs are refused outright
  with an explanation — that the site is a free beta on one small server
  and signing in gets priority — rather than being left in a queue they
  won't reach the front of.

The load banner on the squad page appears only while this is true, and
says something different to guests than to members. `/api/status` is
what it reads.

## Before going public

Four things become load-bearing that currently aren't:

0. **A DMARC record.** Resend sets up SPF and DKIM when you verify a
   domain, but not DMARC, and a sending domain without one is itself a
   spam signal. Add a TXT record at `_dmarc.fplnerdball.com` with
   `v=DMARC1; p=none; rua=mailto:you@gmail.com;`. Five minutes in
   GoDaddy, and it matters as soon as you're emailing people who aren't
   you.
1. **Postgres**, not the SQLite fallback. Single-writer under real
   concurrency means lock errors. Check `database` in `/api/health`.
2. **Rate-limit guest creation.** `POST /api/auth/guest` is
   unauthenticated, and every guest can queue CPU-minutes of work.
3. **Delete guest workspaces.** `guest.discard()` clears the database
   rows but leaves `/data/managers/{id}` behind.
4. **Cache the scored player pool.** Done — see below.
