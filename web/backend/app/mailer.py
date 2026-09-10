"""Getting a message to you when something arrives.

An inbox nobody looks at is a suggestion box in a cupboard, so anything
that lands in it is also emailed. Two ways to send, because hosts differ
in what they allow out: an HTTP API if one is configured, and plain SMTP
otherwise. If neither is set up the app carries on quietly — the inbox in
the admin pane is still the record, and email is the reminder to go and
read it.

Sending happens on a background thread. Nobody submitting feedback should
wait on a mail server, and a mail server being down should never turn
their submission into an error.
"""

from __future__ import annotations

import json
import logging
import smtplib
import threading
import urllib.error
import urllib.request
from email.message import EmailMessage

from .config import settings

log = logging.getLogger("nerdball.mail")

RESEND_ENDPOINT = "https://api.resend.com/emails"


def configured() -> bool:
    """Whether there is anywhere to send to, and a way to send it."""
    if not settings.mail_to:
        return False
    return bool(settings.resend_api_key or settings.smtp_host)


def send(subject: str, body: str, reply_to: str = "") -> None:
    """Queue one email. Returns immediately; never raises."""
    if not configured():
        return

    thread = threading.Thread(
        target=_send,
        args=(subject, body, reply_to),
        name="mail",
        daemon=True,
    )
    thread.start()


def _send(subject: str, body: str, reply_to: str) -> None:
    try:
        if settings.resend_api_key:
            _send_via_resend(subject, body, reply_to)
        else:
            _send_via_smtp(subject, body, reply_to)
    except Exception:
        # The message is already saved; email is the notification, not
        # the record. Logged so a misconfiguration is visible.
        log.warning("Couldn't send the notification email", exc_info=True)


def _send_via_resend(subject: str, body: str, reply_to: str) -> None:
    payload = {
        "from": settings.mail_from,
        "to": [settings.mail_to],
        "subject": subject,
        "text": body,
    }
    if reply_to:
        # So replying to the notification reaches the person who wrote
        # in, rather than the app.
        payload["reply_to"] = reply_to

    request = urllib.request.Request(
        RESEND_ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {settings.resend_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"Resend refused it ({error.code}): {detail}")


def _send_via_smtp(subject: str, body: str, reply_to: str) -> None:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.mail_from
    message["To"] = settings.mail_to
    if reply_to:
        message["Reply-To"] = reply_to
    message.set_content(body)

    port = settings.smtp_port
    if port == 465:
        server = smtplib.SMTP_SSL(settings.smtp_host, port, timeout=20)
    else:
        server = smtplib.SMTP(settings.smtp_host, port, timeout=20)

    with server:
        if port != 465:
            server.starttls()
        if settings.smtp_user:
            server.login(settings.smtp_user, settings.smtp_password)
        server.send_message(message)
