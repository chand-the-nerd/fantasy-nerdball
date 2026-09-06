# Getting this deployed

Written assuming you've not done a Railway deploy or a Google OAuth setup
before. There are four parts: getting the files into your repo, creating the
Railway service, setting up Google sign-in, and letting your friends in.

Budget about 45 minutes for the first run through. Most of it is waiting for
builds and clicking through Google's consent screen forms.

---

## Part 1 — Get the files into your repo

### 1.1 Extract the archive

Download `fantasy-nerdball-web.tar.gz` and unpack it somewhere temporary:

```bash
mkdir ~/nerdball-web && tar -xzf fantasy-nerdball-web.tar.gz -C ~/nerdball-web
ls -a ~/nerdball-web
```

You should see `Dockerfile`, `railway.json`, `.dockerignore`,
`docker-compose.yml`, `.env.example`, `gitignore-additions.txt` and a `web/`
directory.

### 1.2 Make a branch

Never do this on `main` directly. From your clone of `fantasy-nerdball`:

```bash
cd /path/to/fantasy-nerdball
git checkout main
git pull
git checkout -b web-app
```

If `git status` shows uncommitted work, commit or stash it first — you want a
clean starting point so it's obvious what this change adds.

### 1.3 Copy the files in

```bash
cp -r ~/nerdball-web/web ./web
cp ~/nerdball-web/Dockerfile ~/nerdball-web/.dockerignore \
   ~/nerdball-web/railway.json ~/nerdball-web/docker-compose.yml \
   ~/nerdball-web/.env.example ./
```

Your repository now looks like this. Nothing existing has been touched:

```
fantasy-nerdball/
├── main.py                 ← yours, unchanged
├── src/  ml/  utility_scripts/   ← yours, unchanged
├── config_example.py       ← yours, unchanged
├── Dockerfile              ← new
├── railway.json            ← new
├── docker-compose.yml      ← new
├── .env.example            ← new
├── .dockerignore           ← new
└── web/                    ← new
    ├── backend/            FastAPI app
    ├── frontend/           React app
    └── README.md
```

The Dockerfile copies your optimiser out of this same repository, so the model
and the app serving it are always the same commit. No submodules, no cloning,
nothing to keep in sync.

### 1.4 Update .gitignore

```bash
cat ~/nerdball-web/gitignore-additions.txt >> .gitignore
```

Then open `.gitignore` and check the lines landed on their own line — if your
existing file didn't end with a newline, the first added line will have merged
onto the last existing one.

### 1.5 Check nothing secret is about to be committed

```bash
git status
git check-ignore -v .env config.py
```

The second command should confirm both are ignored. `.env` doesn't exist yet,
which is fine — the point is that it's ignored before you create it.

### 1.6 Commit and push

```bash
git add .
git commit -m "Add web app for browser-based optimisation"
git push -u origin web-app
```

You can merge to `main` now or after the deploy works. Railway can deploy from
any branch, so leaving it on `web-app` until you're happy is the safer path.

### 1.7 Optional but recommended: run it locally first

This catches problems in two minutes instead of after a five-minute cloud
build. You need Docker Desktop running.

```bash
docker compose up --build
```

The first build takes a few minutes. When it settles, open
http://localhost:8000 and click **Sign in as the local developer** — that link
skips Google entirely and only exists when `DEV_MODE=true`.

If it comes up, the image is sound and everything from here is configuration.
`Ctrl-C` to stop, then `docker compose down -v` to clear the local database.

---

## Part 2 — Create the Railway service

### 2.1 Create the project

Sign in at railway.app with GitHub. Choose **New Project → Deploy from GitHub
repo**, authorise Railway to see your repositories if prompted, and pick
`fantasy-nerdball`.

Railway will immediately try to build and **it will fail**. That's expected —
there's no database or secret key yet. Ignore it.

### 2.2 Point it at the right branch

Open the service, go to **Settings → Source**, and set the branch to `web-app`
(or `main` if you merged). Leave Root Directory empty — the Dockerfile needs
the whole repository as its build context.

You shouldn't need to set a builder; `railway.json` already tells Railway to
use the Dockerfile.

### 2.3 Add Postgres

In the project canvas: **New → Database → Add PostgreSQL**.

This is where users, settings, squads and results live. Railway injects
`DATABASE_URL` into your service automatically — you don't copy anything.

### 2.4 Add the volume

Select your **web service** (not the database), then **Settings → Volumes →
Add Volume**. Set the mount path to exactly:

```
/data
```

This holds the optimiser's cached reference data — three seasons of history,
team ratings, standings. Without it, that gets re-downloaded on every deploy
and every run is slow. It's the step most easily forgotten and the one you'll
most regret skipping.

### 2.5 Get your public URL

**Settings → Networking → Generate Domain.** You'll get something like
`fantasy-nerdball-production.up.railway.app`.

Copy it. You need it for the Google setup next.

---

## Part 3 — Set up Google sign-in

This is the fiddliest part. Take it slowly.

### 3.1 Create a Google Cloud project

Go to console.cloud.google.com. Create a new project — call it
`fantasy-nerdball`. Wait for it to be created and make sure it's selected in
the picker at the top of the page.

### 3.2 Configure the consent screen

Navigate to **APIs & Services → OAuth consent screen**.

- User type: **External**. (Internal is only for Google Workspace organisations.)
- App name: `Fantasy Nerdball`
- User support email: yours
- Developer contact email: yours

Save and continue. On the **Scopes** step, add nothing — the defaults for email
and profile are what the app requests, and they're granted automatically.

On the **Test users** step, add your own Gmail address and your five friends'.
This matters: while the app is unpublished, only listed test users can sign in
at all. This is your second gate, on top of the app's own allowlist.

### 3.3 Create the credentials

Go to **APIs & Services → Credentials → Create Credentials → OAuth client ID**.

- Application type: **Web application**
- Name: `Fantasy Nerdball web`
- Under **Authorised redirect URIs**, click Add URI and enter, exactly:

```
https://your-app.up.railway.app/api/auth/callback
```

Substitute your real domain. This has to match character for character —
`https` not `http`, no trailing slash, and the path is `/api/auth/callback`.
A mismatch here is the single most common reason sign-in fails, and Google's
error message will tell you exactly what it expected, so read it carefully if
you get stuck.

Click Create. You'll get a **Client ID** and a **Client Secret**. Keep the tab
open.

### 3.4 Set the variables in Railway

Back in Railway, select your web service and open the **Variables** tab. Add
these one at a time:

| Variable | Value |
| --- | --- |
| `SECRET_KEY` | Run `openssl rand -hex 32` and paste the result |
| `GOOGLE_CLIENT_ID` | From the Google console |
| `GOOGLE_CLIENT_SECRET` | From the Google console |
| `ALLOWED_EMAILS` | Your Gmail address |
| `CURRENT_SEASON` | `2026-27` |
| `NERDBALL_DATA_DIR` | `/data` |
| `MAX_USERS` | `6` |

Notes:

- `SECRET_KEY` signs the login cookies. If you ever change it, everyone gets
  signed out. That's the only consequence — no data is lost.
- `ALLOWED_EMAILS` is a comma-separated list, and **the first address becomes
  the owner**. Put yours first. You can add everyone else from inside the app
  afterwards, which is easier.
- `CURRENT_SEASON` must not appear in your engine's `PAST_SEASONS` list. The
  app strips it out if it does and renormalises the season weights, but it's
  cleaner to keep them consistent.

Don't set `DATABASE_URL` or `PUBLIC_BASE_URL` — Railway provides the first, and
the second is derived from your Railway domain automatically. Only set
`PUBLIC_BASE_URL` if you later move to a custom domain.

### 3.5 Deploy

Saving variables triggers a redeploy. Watch **Deployments → View Logs**.

The build takes roughly four to six minutes the first time: installing pandas,
numpy, PuLP and matplotlib is the slow part. Later builds reuse cached layers
and are much quicker.

You're looking for `Uvicorn running on http://0.0.0.0:8000` and the health
check going green.

---

## Part 4 — First run

### 4.1 Sign in

Open your Railway URL. Click **Continue with Google**. Google will warn you the
app isn't verified — click **Advanced → Go to Fantasy Nerdball**. That warning
is normal for an unverified app with a handful of test users and doesn't need
fixing for six people.

You should land on the Squad page.

### 4.2 Set yourself up

Go to **Setup** and work down it:

- **This gameweek** — budget and free transfers. Budget is squad value plus
  bank, not a flat 100 once the season is going.
- **Chips** — leave off unless you're actually playing one.
- **Model** — the transfer threshold and fixture horizon.
- **How players are scored** — the form/history/fixtures split per position.
- **Players you always want** and **Players to avoid**.
- **Team adjustments** — leave everything at 1.00 until you've a reason.
- **Your FPL team** — paste your team id, the number from your team's URL on
  the FPL site.

Click **Save settings**.

### 4.3 Run it

Back on **Squad**, the dropdown defaults to whatever gameweek FPL says is next.
Click **Run**.

**The first run is slow — five to ten minutes.** It's downloading three seasons
of history from the vaastav dataset. Watch the console panel; you'll see your
own progress messages streaming through. Subsequent runs reuse the cached data
on the volume and take one to two minutes.

If it fails, the error appears in that same panel, and the full traceback is in
Railway's deployment logs.

### 4.4 Invite your friends

**Setup → Managers → Add manager.** Enter each Gmail address.

Each person also has to be a test user in the Google consent screen (step 3.2),
or Google blocks them before they reach the app. If someone gets "not invited",
they're missing from the app's list. If they get an error from Google itself,
they're missing from the test users list.

---

## When something goes wrong

**"redirect_uri_mismatch" from Google.** The URI in the Google console doesn't
exactly match your Railway domain. Compare them character by character. The
error page tells you what was actually sent.

**"That Google account isn't on the list."** They're not in `ALLOWED_EMAILS`
and haven't been invited. Add them under Setup → Managers.

**Build fails on `COPY main.py ...`.** You're building from a branch that
doesn't have your optimiser, or Root Directory is set. Clear Root Directory in
Settings → Source.

**Runs fail with "No valid solution found".** That's your optimiser, not the
web layer — usually forced selections that can't fit the budget or the
three-per-club limit. Relax them under Setup.

**Everything's slow after a redeploy.** The volume isn't mounted, or isn't
mounted at `/data`. Check Settings → Volumes.

**Sign-in works, then immediately signs you out.** `SECRET_KEY` isn't set, so
the app is using its development default and the cookie isn't stable.

---

## Afterwards

To ship a model change, commit to the branch Railway watches and push. Railway
rebuilds and redeploys on its own — the new model is live in a few minutes.

Nothing about your local workflow changes. `config.py` is still yours, still
gitignored, and running `python main.py` on your laptop works exactly as it
did.
