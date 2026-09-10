"""Ask for access, send feedback, and check both reach the admin."""

import os
import pathlib
import sys
import tempfile
import time

tmp = tempfile.mkdtemp()
os.environ["NERDBALL_DATA_DIR"] = tmp
os.environ["DATABASE_URL"] = f"sqlite:///{tmp}/test.db"
os.environ["STATIC_DIR"] = f"{tmp}/nostatic"
os.environ["NERDBALL_ENGINE_DIR"] = f"{tmp}/noengine"
os.environ["HISTORY_AUTO_UPDATE"] = "false"
os.environ["HEARTBEAT_MINUTES"] = "0"
os.environ["SECRET_KEY"] = "test-only"
os.environ["DEV_MODE"] = "true"
os.environ["DEV_LOGIN_EMAIL"] = "owner@example.com"
os.environ["ALLOWED_EMAILS"] = "owner@example.com"
os.environ["ADMIN_EMAILS"] = "owner@example.com"
os.environ["MAX_USERS"] = "3"

BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402

from app import mailer  # noqa: E402
from app.db import init_db  # noqa: E402
from app.main import app  # noqa: E402
from app.routers import inbox as inbox_router  # noqa: E402

init_db()

# Capture what would have been emailed instead of sending it.
sent: list[tuple[str, str, str]] = []


def _capture(subject="", body="", reply_to="", to="", html=""):
    sent.append((subject, body, reply_to))


def _capture_template(template, to="", reply_to=""):
    sent.append((template[0], template[2], reply_to))


mailer.send = _capture
mailer.send_template = _capture_template
inbox_router.mailer = mailer

ok = True


def check(label, condition, extra=""):
    global ok
    if not condition:
        ok = False
    print(f"{'PASS' if condition else 'FAIL'}  {label} {extra}")


owner = TestClient(app, headers={"x-forwarded-for": "203.0.113.1"})
owner.post("/api/auth/dev-login")

config = TestClient(app).get("/api/auth/config").json()
check(
    "sign-in page gets seat numbers",
    {"seats_used", "seats_total", "seats_free"} <= set(config),
    config,
)
check("a seat is taken", config["seats_used"] == 1, config["seats_used"])
check("two free of three", config["seats_free"] == 2, config["seats_free"])
check(
    "no addresses in the public config",
    "@" not in str(config),
    config,
)

stranger = TestClient(app, headers={"x-forwarded-for": "198.51.100.7"})

bad = stranger.post("/api/access-request", json={"email": "not-an-email"})
check("a malformed address is refused", bad.status_code == 400)

made = stranger.post(
    "/api/access-request",
    json={"email": "Newcomer@Gmail.com", "note": "Dave sent me."},
)
check("a request is accepted", made.status_code == 201, made.text[:100])
check(
    "the reply says what happens next",
    "Google" in made.json()["message"],
    made.json(),
)
check("it emailed you and them", len(sent) == 2,
      [m[0] for m in sent])
if sent:
    subject, body, reply_to = sent[0]
    check(
        "the subject names the address",
        subject == "Access request — newcomer@gmail.com",
        subject,
    )
    check("the note is in the body", "Dave sent me." in body, body[:80])
    check(
        "replying reaches them, not the app",
        reply_to == "newcomer@gmail.com",
        reply_to,
    )

again = stranger.post(
    "/api/access-request", json={"email": "newcomer@gmail.com"}
)
check("asking twice is accepted", again.status_code == 201)
check(
    "and says the admin already has it",
    again.json().get("status") == "pending",
    again.json(),
)
check("but doesn't email again", len(sent) == 2, len(sent))

# The case that was silently doing nothing: an address already welcome.
member = stranger.post(
    "/api/access-request", json={"email": "owner@example.com"}
)
check(
    "an existing manager is told they're already in",
    member.json().get("status") == "already_approved",
    member.json(),
)
check(
    "and is pointed at the sign-in button",
    "Sign in" in member.json()["message"],
    member.json()["message"],
)
check("no pointless email for that", len(sent) == 2, len(sent))

# Third distinct call from the same address trips the limit.
stranger.post("/api/access-request", json={"email": "third@gmail.com"})
limited = stranger.post(
    "/api/access-request", json={"email": "fourth@gmail.com"}
)
check("the rate limit bites", limited.status_code == 429, limited.status_code)

# Approved-but-never-signed-in has no user row, and used to slip through
# as a fresh request.
owner.post("/api/admin/invites", json={"email": "invited@gmail.com"})
patient = TestClient(app, headers={"x-forwarded-for": "198.51.100.55"})
invited_reply = patient.post(
    "/api/access-request", json={"email": "invited@gmail.com"}
)
check(
    "someone already invited is told so",
    invited_reply.json().get("status") == "already_approved",
    invited_reply.json(),
)

fresh = patient.post("/api/access-request", json={"email": "new@gmail.com"})
check(
    "a genuinely new address still goes through",
    fresh.json().get("status") == "sent",
    fresh.json(),
)

# Feedback, including from a guest.
guest = TestClient(app, headers={"x-forwarded-for": "198.51.100.99"})
guest.post("/api/auth/guest")
posted = guest.post(
    "/api/feedback",
    json={"kind": "broken", "body": "The pitch renders behind the header."},
)
check("a guest can send feedback", posted.status_code == 201, posted.text[:90])
feedback_mail = [m for m in sent if m[0].startswith("Something is broken")]
check("that emailed you too", len(feedback_mail) == 1, [m[0] for m in sent])
if feedback_mail:
    check(
        "the body says it came from a guest",
        "a guest" in feedback_mail[0][1].lower(),
        feedback_mail[0][1][:60],
    )
    check(
        "no reply-to for a guest's throwaway address",
        feedback_mail[0][2] == "",
        feedback_mail[0][2],
    )

wrong = owner.post("/api/feedback", json={"kind": "rant", "body": "hello"})
check("an unknown kind is refused", wrong.status_code == 400)
empty = owner.post("/api/feedback", json={"kind": "general", "body": "   "})
check("an empty message is refused", empty.status_code == 400)

denied = guest.get("/api/admin/inbox")
check("a guest can't read the inbox", denied.status_code == 403)

box = owner.get("/api/admin/inbox").json()
titles = [item["title"] for item in box["items"]]
print(f"      inbox: {titles}")
# Three, not four: the rate-limited one was never stored.
check("everything arrived", len(box["items"]) == 3, len(box["items"]))
check(
    "access requests are titled as you asked",
    "Access request - newcomer@gmail.com" in titles,
    titles,
)
check(
    "feedback is titled by kind",
    "Something is broken" in titles,
    titles,
)
check("unread count is right", box["unread"] == 3, box["unread"])
check("it knows email isn't configured", box["email_configured"] is False)

item = next(
    i for i in box["items"] if i["email"] == "newcomer@gmail.com"
)
approved = owner.post(f"/api/admin/inbox/{item['id']}/approve")
check("approving works", approved.status_code == 200, approved.text[:90])

members = owner.get("/api/admin/members").json()
invited = [entry["email"] for entry in members["invites"]]
check(
    "approving puts them on the allowlist",
    "newcomer@gmail.com" in invited,
    invited,
)

after = owner.get("/api/admin/inbox").json()
check(
    "the handled item leaves the list", after["unread"] == 2, after["unread"]
)
check(
    "and is still there if you look",
    len(owner.get("/api/admin/inbox?include_done=true").json()["items"]) == 3,
)

feedback_item = next(
    i for i in after["items"] if i["title"] == "Something is broken"
)
done = owner.post(f"/api/admin/inbox/{feedback_item['id']}/done")
check("marking feedback done works", done.status_code == 200)
check(
    "only the untouched request remains",
    owner.get("/api/admin/inbox").json()["unread"] == 1,
)

box2 = owner.get("/api/admin/inbox").json()
check("mail status is reported", "email" in box2, list(box2))
check(
    "and says it isn't set up",
    box2["email"]["mode"] == "off" and box2["email"]["configured"] is False,
    box2["email"],
)

from app import mailer as real_mailer  # noqa: E402

check(
    "the test send explains what's missing rather than failing silently",
    "MAIL_TO" in real_mailer.send_now("t", "b"),
    real_mailer.send_now("t", "b"),
)

test = owner.post("/api/admin/test-email")
check("the test button reports back", test.status_code == 200, test.text[:80])
check("and says it can't send", test.json()["ok"] is False, test.json())
check(
    "with a reason a person can act on",
    "MAIL_TO" in test.json()["detail"],
    test.json()["detail"],
)

denied_test = guest.post("/api/admin/test-email")
check("guests can't send test emails", denied_test.status_code == 403)

missing = owner.post("/api/admin/inbox/9999/done")
check("an unknown item is a 404", missing.status_code == 404)

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
