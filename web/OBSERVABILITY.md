# Observability

The app writes one JSON line per thing worth counting. Railway parses
JSON on stdout into filterable attributes, so every event below can be
charted from the Observability dashboard without shipping anything off
the platform.

Note that these are **deploy logs**, the ones the Observability log widget
reads — not Railway's HTTP logs, which live under each deployment's own
HTTP Logs tab and aren't queryable from the dashboard.

## Making a widget

Build the filter in the Log Explorer first, check it returns what you
expect, then add it to the dashboard. The widget uses the same query
syntax.

| Question | Filter |
|---|---|
| How many visits today | `@event:session_start` |
| Guests vs signed-in | `@event:session_start @guest:true` |
| New managers | `@event:sign_in @new_user:true` |
| People taking the guest route | `@event:guest_start` |
| Optimisations started | `@event:run_started` |
| Runs that failed | `@event:run_failed` |
| Queue turning people away | `@event:run_rejected` |
| Totals and queue depth | `@event:heartbeat` |
| Anyone refused entry | `@event:sign_in_denied` |
| Errors of any kind | `@level:error` |

Attach a monitor to the `run_failed`, `run_rejected` and `@level:error`
widgets via the three-dot menu — those are the three that mean something
is wrong rather than something is happening.

## The events

### Sessions and accounts

| Event | Fields | When |
|---|---|---|
| `session_start` | `user`, `guest` | First request from a user after 30 minutes' silence. This is the visit counter. |
| `sign_in` | `user`, `method`, `new_user` | A completed Google sign-in. `new_user` separates growth from activity. |
| `sign_in_denied` | `reason`, `email` | Someone was turned away: `not_invited`, `league_full`, `no_verified_email`. |
| `guest_start` | `user` | Someone chose "Continue without signing in". |
| `sign_out` | `user`, `guest` | Session ended. For a guest this is also when the account is deleted. |
| `guests_purged` | `count` | Idle guest accounts swept up. |
| `guest_blocked` | `feature`, `kind` | A guest hit a locked feature or a list cap. **The most useful event here**: it tells you which restriction people actually run into, and so what is worth opening up. |

### Optimiser

| Event | Fields | When |
|---|---|---|
| `run_queued` | `run`, `gameweek`, `queue_depth`, `user`, `guest` | A run was accepted. |
| `run_started` | `run`, `waited_seconds`, `queue_depth`, … | The worker picked it up. `waited_seconds` is queue latency — the number that goes bad first as usage grows. |
| `run_finished` | `run`, `seconds`, `projected_points`, `transfers`, `chip`, `rss_cost_mb`, `rss_peak_mb` | Success. `seconds` and `rss_cost_mb` are the two inputs to sizing: see web/CAPACITY.md. |
| `run_failed` | `run`, `seconds`, `error_type` | The engine threw. |
| `run_rejected` | `reason`, `queue_depth` | Refused: `queue_full` or `already_running`. `queue_full` means the single worker is overrun. |
| `run_cancelled` | `run` | Dropped from the queue by the user. |
| `plan_queued` / `plan_started` / `plan_finished` / `plan_failed` / `plan_rejected` | as above, plus `weeks` | The planner. |
| `worker_error` | `kind`, `id` | The worker loop itself threw, which is the case that used to lose the queue silently. |

### Usage

| Event | Fields | When |
|---|---|---|
| `settings_saved` | `horizon`, `strategy`, `bench`, `ml_weights`, `forced`, `blacklist`, `clubs_adjusted`, `chips` | The shape of what someone saved, not the row. Answers whether anyone tunes the model or whether the defaults do for everybody. |
| `squad_entered` | `method`, `gameweek` | `manual` or `fpl_import`. |
| `option_activated` | `option`, `recommended`, `gameweek` | Which alternative was chosen over the recommendation — the closest thing to feedback on the optimiser. |
| `squad_deleted` | `gameweek` | |
| `fpl_linked` | `user` | A team id was connected. |
| `player_lookup` | `player` | A player id, never a name. |
| `invite_added` / `member_removed` | `by`, `email` / `removed` | Admin actions. |

### Platform

| Event | Fields | When |
|---|---|---|
| `http_request` | `method`, `route`, `status`, `ms`, `user` | One per API call. `route` is the template (`/api/runs/{run_id}`), so cardinality stays low. |
| `http_error` | `path`, `method`, `ms` | An unhandled exception, which never reaches the line above. |
| `page_view` | `path` | The SPA served for an extension-less path. |
| `boot` | `commit`, `database`, `guest_mode` | Startup. Confirms which build is live. |
| `heartbeat` | `members`, `guests_live`, `runs_24h`, `runs_failed_24h`, `queue_depth`, `worker_busy`, `rss_mb`, `memory_limit_mb`, `workers` | Every `HEARTBEAT_MINUTES`. This is what makes "how many users do I have" a line on a chart rather than a query someone has to remember to run. |

## The dashboard in the admin pane

The same events are also written to a `metric_events` table, which is
what the **Usage** tab in the admin pane reads. Windows: last hour, 6
hours, 24 hours, 7 days. It shows unique visitors split by guest and
member, visits, runs started, typical and 95th-percentile run time, the
worst queue wait, the failure rate, a chart of visitors and runs over
time, who has been on the site, and where guests hit a locked feature.

The table is deliberately **not** joined to users by a foreign key.
Guest accounts are deleted the moment their session ends, and a cascade
would take the history with them — leaving a dashboard that could only
ever describe the people still signed in.

Only meaningful events are stored (`metrics.STORED`); `http_request` and
`page_view` stay in the log, because a page load is unauthenticated by
nature and storing it would give every signed-in manager a second,
address-keyed identity and double the unique count. Rows are written from
a background thread behind a bounded queue, so a metric is never
something a request waits for, and are pruned after
`METRICS_RETENTION_DAYS`.

### Identifying visitors

| Who | Counted as | Shown as |
|---|---|---|
| Signed-in manager | their account id | name and email, resolved live |
| Guest | a salted hash of their address | whatever `VISITOR_IP_MODE` allows |
| Never signed in | a salted hash of their address | "Visitor" |

Unique visitors are counted from the hash regardless of the setting, so
counting still works on `none` — the setting governs only what a person
reading the dashboard can see. The hash is salted with `SECRET_KEY`, so
it isn't reversible into an address by anyone reading the table.

**If you make the site public, this is the part to think about.** An IP
address is personal data under UK GDPR, so storing one needs a lawful
basis, a privacy notice saying you do it, and a retention period. The
default (`truncated`, 90 days, admin-only) is the defensible position:
a /24 still separates one visitor from another well enough to count them
without the table holding a list of addresses. `full` is there if you
decide you need it, but don't switch it on for a public site without a
privacy notice.

## What is deliberately not logged

**Names and email addresses.** Identities are numeric ids. A log line is
readable by anyone with dashboard access and is retained for weeks, so
`user: 4` is enough to count people and follow one session without the
log becoming a copy of the users table. The single exception is
`sign_in_denied`, where the address is the only actionable part of the
record — if you'd rather it weren't there, remove the `email=` argument
in `routers/auth.py`.

**Polling.** `GET /api/runs/{run_id}` and `GET /api/plans/{plan_id}` are
polled every 2.5 seconds while a run is in flight, so one three-minute
run would otherwise generate around 70 near-identical lines. They're in
`QUIET_ROUTES` in `main.py`, along with `/api/health`. The run's actual
lifecycle is recorded by the worker in far more useful detail.

**Squad contents.** Player ids appear only for explicit lookups.

## Settings

| Variable | Default | Effect |
|---|---|---|
| `LOG_JSON` | `true` (`false` in dev) | JSON lines vs plain text. JSON is what makes attribute filtering work. |
| `LOG_REQUESTS` | `true` | The per-request `http_request` event. Turn off if log volume becomes the cost rather than the insight. |
| `HEARTBEAT_MINUTES` | `15` | How often totals are written. `0` switches it off. |
| `METRICS` | `true` | The database copy behind the admin dashboard. |
| `METRICS_RETENTION_DAYS` | `90` | How long stored events are kept. `0` keeps them forever. |
| `VISITOR_IP_MODE` | `truncated` | `truncated` stores the /24 a guest came from, `full` the address, `none` neither. |

With JSON logging on, uvicorn's own access log is disabled — the
`http_request` event carries the same request with the user attached.

## Retention

Railway keeps logs for 30 days on Pro. For anything longitudinal, either
forward them (the Locomotive template is a Railway-native log drain to
Axiom, BetterStack, Datadog and others) or rely on the database: `runs`
already stores `started_at` and `finished_at`, so run history is durable
regardless of what the log does.
