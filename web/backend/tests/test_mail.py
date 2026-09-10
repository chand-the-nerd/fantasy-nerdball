"""Check the Resend request is one Cloudflare will let through."""
import os, pathlib, sys, tempfile, urllib.error, urllib.request, io

tmp = tempfile.mkdtemp()
os.environ.update({
    "NERDBALL_DATA_DIR": tmp, "DATABASE_URL": f"sqlite:///{tmp}/t.db",
    "STATIC_DIR": f"{tmp}/s", "NERDBALL_ENGINE_DIR": f"{tmp}/e",
    "SECRET_KEY": "test", "HISTORY_AUTO_UPDATE": "false",
    "MAIL_TO": "me@example.com", "RESEND_API_KEY": "re_test123",
    "MAIL_FROM": "Fantasy Nerdball <alerts@fplnerdball.com>",
})
BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from app import mailer

ok = True
def check(label, cond, extra=""):
    global ok
    if not cond: ok = False
    print(f"{'PASS' if cond else 'FAIL'}  {label} {extra}")

captured = {}
class FakeResponse:
    def read(self): return b'{"id":"abc"}'
    def __enter__(self): return self
    def __exit__(self, *a): return False

def fake_urlopen(request, timeout=None):
    captured["headers"] = dict(request.headers)
    captured["url"] = request.full_url
    captured["body"] = request.data.decode()
    return FakeResponse()

mailer.urllib.request.urlopen = fake_urlopen

error = mailer.send_now("Subject here", "Body here", "them@example.com")
check("send reports success", error == "", error)

headers = captured.get("headers", {})
lowered = {k.lower(): v for k, v in headers.items()}
print(f"      headers: {sorted(lowered)}")
check("a User-Agent is sent", "user-agent" in lowered, lowered)
check(
    "and it isn't Python's default",
    not lowered.get("user-agent", "").lower().startswith("python-urllib"),
    lowered.get("user-agent"),
)
check("the key is sent as a bearer token",
      lowered.get("authorization") == "Bearer re_test123")
check("content type is json",
      lowered.get("content-type") == "application/json")
check("posting to the emails endpoint",
      captured["url"] == "https://api.resend.com/emails", captured["url"])
check("from uses the verified domain",
      "alerts@fplnerdball.com" in captured["body"])
check("reply-to is carried", "them@example.com" in captured["body"])
check("status records the success", mailer.status()["last_error"] == "")
check("and when it last sent", bool(mailer.status()["last_sent"]))

# A Cloudflare block must be named as one, not blamed on the key.
def cf_block(request, timeout=None):
    raise urllib.error.HTTPError(
        "https://api.resend.com/emails", 403, "Forbidden", {},
        io.BytesIO(b"error code: 1010"))

mailer.urllib.request.urlopen = cf_block
message = mailer.send_now("s", "b")
check("a 1010 is identified as Cloudflare", "Cloudflare" in message, message)
check("and doesn't blame the API key",
      "key" not in message.lower().split("cloudflare")[0], message)
check("it's recorded for the admin page",
      "Cloudflare" in mailer.status()["last_error"])

# A real Resend 403 should still point at the likely cause.
def resend_403(request, timeout=None):
    raise urllib.error.HTTPError(
        "https://api.resend.com/emails", 403, "Forbidden", {},
        io.BytesIO(b'{"message":"domain is not verified"}'))

mailer.urllib.request.urlopen = resend_403
message = mailer.send_now("s", "b")
check("a real 403 points at MAIL_FROM", "MAIL_FROM" in message, message)

def bad_key(request, timeout=None):
    raise urllib.error.HTTPError(
        "https://api.resend.com/emails", 401, "Unauthorized", {},
        io.BytesIO(b'{"message":"invalid api key"}'))

mailer.urllib.request.urlopen = bad_key
check("a 401 names the key", "key" in mailer.send_now("s", "b").lower())

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
sys.exit(0 if ok else 1)
