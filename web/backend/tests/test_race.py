"""Several processes calling init_db at once, as WEB_CONCURRENCY does."""
import os, pathlib, subprocess, sys, tempfile

PG = os.getenv(
    "NERDBALL_TEST_PG",
    "postgresql+psycopg://postgres@localhost:5433/racetest",
)
CHILD = """
import os, pathlib, sys
os.environ.update({
    "DATABASE_URL": %r, "NERDBALL_DATA_DIR": %r, "STATIC_DIR": %r,
    "NERDBALL_ENGINE_DIR": %r, "SECRET_KEY": "t",
    "HISTORY_AUTO_UPDATE": "false", "HEARTBEAT_MINUTES": "0",
})
sys.path.insert(0, %r)
from app.db import init_db
init_db()
print("ok")
"""

tmp = tempfile.mkdtemp()
backend = str(pathlib.Path("web/backend").resolve())
script = CHILD % (PG, tmp, f"{tmp}/s", f"{tmp}/e", backend)

ok = True
def check(label, cond, extra=""):
    global ok
    if not cond: ok = False
    print(f"{'PASS' if cond else 'FAIL'}  {label} {extra}")

# Six at once against an empty database: the original crash needed only
# two, so this is well past the point it used to fail.
procs = [
    subprocess.Popen([sys.executable, "-c", script],
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     text=True)
    for _ in range(6)
]
results = [p.communicate() for p in procs]
codes = [p.returncode for p in procs]

check("every process started cleanly", all(c == 0 for c in codes), codes)
failures = [err[-300:] for (out, err), c in zip(results, codes) if c != 0]
if failures:
    print("      " + failures[0].replace("\n", "\n      "))
check("no duplicate key errors",
      not any("duplicate key" in err for _, err in results),
      [err[-120:] for _, err in results if "duplicate" in err][:1])
check("all six report success",
      sum(1 for out, _ in results if "ok" in out) == 6,
      [out.strip() for out, _ in results])

# And again now the tables exist — the everyday redeploy case.
procs = [
    subprocess.Popen([sys.executable, "-c", script],
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     text=True)
    for _ in range(4)
]
results = [p.communicate() for p in procs]
check("a redeploy against existing tables is fine",
      all(p.returncode == 0 for p in procs), [p.returncode for p in procs])

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
