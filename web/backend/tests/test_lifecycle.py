"""Invite expiry, inactivity removal, queue position, orphan recovery."""
import datetime as dt, os, pathlib, sys, tempfile

tmp = tempfile.mkdtemp()
os.environ.update({
    "NERDBALL_DATA_DIR": tmp, "DATABASE_URL": f"sqlite:///{tmp}/t.db",
    "STATIC_DIR": f"{tmp}/s", "NERDBALL_ENGINE_DIR": f"{tmp}/e",
    "SECRET_KEY": "test", "HISTORY_AUTO_UPDATE": "false",
    "HEARTBEAT_MINUTES": "0", "DEV_MODE": "true",
    "DEV_LOGIN_EMAIL": "owner@example.com",
    "ALLOWED_EMAILS": "owner@example.com", "ADMIN_EMAILS": "owner@example.com",
    "MAX_USERS": "3", "INVITE_TTL_HOURS": "72", "INACTIVE_DAYS": "28",
    "MAIL_TO": "owner@example.com", "RESEND_API_KEY": "re_test",
})
BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from fastapi.testclient import TestClient
from app import lifecycle, mailer, emails
from app.auth import is_allowed
from app.db import SessionLocal, init_db
from app.main import app
from app.models import InboxItem, Invite, Run, User, utcnow

init_db()
posted = []
mailer.send = lambda **k: posted.append(k)
mailer.send_template = lambda tpl, to="", reply_to="": posted.append(
    {"subject": tpl[0], "html": tpl[1], "text": tpl[2], "to": to})
import app.routers.inbox as ib
ib.mailer = mailer
lifecycle.mailer = mailer

ok = True
def check(label, cond, extra=""):
    global ok
    if not cond: ok = False
    print(f"{'PASS' if cond else 'FAIL'}  {label} {extra}")

client = TestClient(app)
owner = TestClient(app, headers={"x-forwarded-for": "203.0.113.1"})
owner.post("/api/auth/dev-login")

# ── Invite expiry ────────────────────────────────────────────────────────
with SessionLocal() as s:
    s.add(Invite(email="fresh@x.com", expires_at=utcnow() + dt.timedelta(hours=5)))
    s.add(Invite(email="stale@x.com", expires_at=utcnow() - dt.timedelta(hours=1)))
    s.add(Invite(email="forever@x.com", expires_at=None))
    s.commit()
    check("a live invite is allowed", is_allowed("fresh@x.com", s))
    check("an expired invite is not", not is_allowed("stale@x.com", s))
    check("an invite with no deadline still is", is_allowed("forever@x.com", s))

posted.clear()
with SessionLocal() as s:
    gone = lifecycle.expire_invites(s)
check("the expired one is swept", gone == 1, gone)
check("and they were told", any("expired" in p["subject"].lower() for p in posted), [p["subject"] for p in posted])
with SessionLocal() as s:
    left = {i.email for i in s.scalars(__import__("sqlalchemy").select(Invite))}
check("only the expired one went", left == {"fresh@x.com", "forever@x.com"}, left)

# ── Inactivity ───────────────────────────────────────────────────────────
with SessionLocal() as s:
    s.add(User(email="active@x.com", name="Active", last_seen_at=utcnow()))
    s.add(User(email="gone@x.com", name="Gone",
               last_seen_at=utcnow() - dt.timedelta(days=40)))
    s.add(User(email="admin2@x.com", name="Admin", is_admin=True,
               last_seen_at=utcnow() - dt.timedelta(days=99)))
    s.commit()

posted.clear()
with SessionLocal() as s:
    freed = lifecycle.make_dormant(s)
check("the inactive manager's place is freed", freed == 1, freed)
with SessionLocal() as s:
    from sqlalchemy import select as _sel
    rows = {u.email: u for u in s.scalars(_sel(User))}
check("nobody is deleted", "gone@x.com" in rows, sorted(rows))
check("they're marked dormant instead",
      rows["gone@x.com"].dormant_at is not None)
check("the active one is untouched", rows["active@x.com"].dormant_at is None)
check("an admin is never made dormant",
      rows["admin2@x.com"].dormant_at is None)
check("the owner is never made dormant",
      rows["owner@example.com"].dormant_at is None)
check("both parties were emailed", len(posted) == 2, [p["subject"] for p in posted])
check("one went to them",
      any(p.get("to") == "gone@x.com" for p in posted), posted)
check("one went to the admin",
      any(p.get("to") == "" for p in posted), [p.get("to") for p in posted])
check("and it says the data is kept",
      any("still here" in p["text"].lower() for p in posted),
      [p["text"][:60] for p in posted])

# ── Queue position ───────────────────────────────────────────────────────
with SessionLocal() as s:
    # Three seats, three taken -> a queue forms.
    s.add(User(email="a@x.com", last_seen_at=utcnow()))
    s.add(User(email="b@x.com", last_seen_at=utcnow()))
    s.commit()
    # Earlier steps left more managers than MAX_USERS, so raise the cap
    # for a moment to test the "there's room" branch honestly.
    from app.config import settings as cfg
    original = cfg.max_users
    cfg.max_users = 99
    check("no queue while a place is free",
          lifecycle.queue_position(s, "someone@x.com") is None)
    cfg.max_users = original
    s.add(User(email="c@x.com", last_seen_at=utcnow()))
    s.add(User(email="d@x.com", last_seen_at=utcnow()))
    s.commit()
    check("a queue forms once every place is taken",
          lifecycle.queue_position(s, "someone@x.com") is not None)
    s.add(InboxItem(kind="access_request", email="first@x.com",
                    created_at=utcnow() - dt.timedelta(hours=3)))
    s.add(InboxItem(kind="access_request", email="second@x.com",
                    created_at=utcnow() - dt.timedelta(hours=2)))
    s.commit()
    check("oldest request is first", lifecycle.queue_position(s, "first@x.com") == 1)
    check("next is second", lifecycle.queue_position(s, "second@x.com") == 2)
    check("a newcomer joins the end",
          lifecycle.queue_position(s, "new@x.com") == 3)

box = owner.get("/api/admin/inbox").json()
requests = [i for i in box["items"] if i["kind"] == "access_request"]
check("inbox lists requests oldest first",
      [r["email"] for r in requests][:2] == ["first@x.com", "second@x.com"],
      [r["email"] for r in requests])
check("and numbers the queue",
      [r["queue_position"] for r in requests][:2] == [1, 2],
      [r["queue_position"] for r in requests])

# ── Orphan recovery ──────────────────────────────────────────────────────
from app.engine import jobs
with SessionLocal() as s:
    s.add(Run(user_id=1, season="2025-26", gameweek=5, status="running"))
    s.add(Run(user_id=1, season="2025-26", gameweek=6, status="queued"))
    s.commit()
found = jobs.recover_orphans()
check("orphans are recovered", found == 2, found)
with SessionLocal() as s:
    from sqlalchemy import select as sel
    statuses = sorted(r.status for r in s.scalars(sel(Run)))
check("the interrupted one is failed, the queued one still queued",
      statuses == ["failed", "queued"], statuses)
with SessionLocal() as s:
    failed = s.scalars(sel(Run).where(Run.status == "failed")).first()
check("with an explanation the user can act on",
      "restarted" in (failed.error or ""), failed.error)

# ── Emails render ────────────────────────────────────────────────────────
subject, html, text = emails.access_approved("x@y.com", 72, 28)
check("approval email mentions the 72 hours", "72 hours" in html, "")
check("and the 28 days", "28 days" in html)
check("and has a plain-text twin", "72 HOURS" in text)
check("HTML is a full document", html.startswith("<!DOCTYPE html>"))
check("uses the site palette", "#f2c14e" in html)
check("carries the not-affiliated line", "Not affiliated" in html)
subject, html, text = emails.access_received("x@y.com", 3, 0)
check("queue email says the position", "3rd in the queue" in html, "")
subject, html, text = emails.access_received("x@y.com", None, 2)
check("or that there's room", "2</strong>" in html)
check("HTML escapes what people type",
      "&lt;script&gt;" in emails.admin_feedback("F", "a guest", "<script>")[1])

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
