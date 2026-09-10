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

## Before going public

Four things become load-bearing that currently aren't:

1. **Postgres**, not the SQLite fallback. Single-writer under real
   concurrency means lock errors. Check `database` in `/api/health`.
2. **Rate-limit guest creation.** `POST /api/auth/guest` is
   unauthenticated, and every guest can queue CPU-minutes of work.
3. **Delete guest workspaces.** `guest.discard()` clears the database
   rows but leaves `/data/managers/{id}` behind.
4. **Cache the scored player pool.** Scoring every player is the
   expensive half of a run, and every guest runs with identical locked
   settings — so every guest run for a gameweek recomputes byte-identical
   scores. Caching on (season, gameweek, settings hash) would cut guest
   CPU by most of its cost, and help signed-in managers on defaults too.
