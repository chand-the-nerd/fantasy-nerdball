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

import datetime as dt
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
USER_AGENT = "fantasy-nerdball/1.0"


def configured() -> bool:
    """Whether there is anywhere to send to, and a way to send it."""
    if not settings.mail_to:
        return False
    return bool(settings.resend_api_key or settings.smtp_host)


# The last thing that went wrong, so "no email came through" has an
# answer in the admin pane rather than only in the deploy logs.
_last_error: str = ""
_last_sent: str = ""

# A day's sending, counted. The access-request endpoint is
# unauthenticated and every request now sends two messages, so without a
# ceiling a stranger with a script could exhaust the provider's free
# tier before lunch — and a sending domain that suddenly emits hundreds
# of messages to strangers is one that stops being trusted.
_day: str = ""
_sent_today: int = 0
_suppressed: int = 0
_count_lock = threading.Lock()


def _within_daily_limit() -> bool:
    global _day, _sent_today, _suppressed

    if settings.max_emails_per_day <= 0:
        return True

    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    with _count_lock:
        if today != _day:
            _day = today
            _sent_today = 0
            _suppressed = 0
        if _sent_today >= settings.max_emails_per_day:
            _suppressed += 1
            return False
        _sent_today += 1
    return True


def status() -> dict:
    """What the mailer is set up to do, and how it last got on."""
    if settings.resend_api_key:
        mode = "resend"
    elif settings.smtp_host:
        mode = "smtp"
    else:
        mode = "off"
    return {
        "configured": configured(),
        "mode": mode,
        "to": settings.mail_to,
        "from": settings.mail_from,
        "last_error": _last_error,
        "last_sent": _last_sent,
        "sent_today": _sent_today,
        "daily_limit": settings.max_emails_per_day,
        "suppressed_today": _suppressed,
    }


def send(
    subject: str,
    body: str,
    reply_to: str = "",
    to: str = "",
    html: str = "",
) -> None:
    """Queue one email. Returns immediately; never raises.

    `to` defaults to MAIL_TO, which is you. Pass it to write to somebody
    else — which only works once a domain is verified with the provider.
    `body` is the plain-text version and is always required: some
    clients show it, and a message with only HTML looks like spam to the
    filters that decide whether the rest arrives.
    """
    if not configured():
        return

    if not _within_daily_limit():
        log.warning(
            "Daily email limit reached; not sending %r", subject[:60]
        )
        return

    thread = threading.Thread(
        target=_send,
        args=(subject, body, reply_to, to, html),
        name="mail",
        daemon=True,
    )
    thread.start()


def send_template(
    template: tuple[str, str, str], to: str = "", reply_to: str = ""
) -> None:
    """Queue one of the messages from emails.py."""
    subject, html, text = template
    send(subject=subject, body=text, html=html, to=to, reply_to=reply_to)


def send_now(
    subject: str,
    body: str,
    reply_to: str = "",
    to: str = "",
    html: str = "",
) -> str:
    """Send on this thread and say what happened.

    Used by the admin page's test button. The background path is right
    for real messages — nobody should wait on a mail server — but a
    diagnostic that returns before it knows the answer is no diagnostic
    at all.

    Returns an empty string on success, or the reason it failed.
    """
    global _last_error, _last_sent

    recipient = to or settings.mail_to
    if not recipient:
        return "MAIL_TO isn't set, so there's nowhere to send to."
    if not (settings.resend_api_key or settings.smtp_host):
        return (
            "Neither RESEND_API_KEY nor SMTP_HOST is set, so there's no "
            "way to send."
        )

    try:
        if settings.resend_api_key:
            _send_via_resend(subject, body, reply_to, recipient, html)
        else:
            _send_via_smtp(subject, body, reply_to, recipient, html)
    except Exception as error:
        _last_error = f"{type(error).__name__}: {error}"[:500]
        log.warning("Couldn't send the email: %s", _last_error)
        return _last_error

    _last_error = ""
    _last_sent = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    return ""


def _send(
    subject: str,
    body: str,
    reply_to: str,
    to: str = "",
    html: str = "",
) -> None:
    # The message is already saved; email is the notification, not the
    # record. Failures are recorded rather than raised.
    send_now(subject, body, reply_to, to, html)


def _send_via_resend(
    subject: str,
    body: str,
    reply_to: str,
    to: str = "",
    html: str = "",
) -> None:
    payload = {
        "from": settings.mail_from,
        "to": [to or settings.mail_to],
        "subject": subject,
        "text": body,
    }
    if html:
        payload["html"] = html
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
            "Accept": "application/json",
            # Not decoration. Cloudflare sits in front of the Resend API
            # and bans urllib's default "Python-urllib/3.x" signature at
            # the edge, returning a 403 that never reaches Resend and has
            # nothing to do with the API key. Any ordinary agent string
            # gets through.
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:300]

        # A 403 carrying a Cloudflare error code never reached Resend at
        # all, so it says nothing about the key, the domain or MAIL_TO.
        if "error code: 1010" in detail:
            raise RuntimeError(
                "Blocked by Cloudflare in front of the Resend API, not by "
                "Resend. This means the request went out without a proper "
                "User-Agent header — the deployed code is older than the "
                f"fix for it. ({detail})"
            )

        if error.code == 403 and "resend.dev" in settings.mail_from:
            raise RuntimeError(
                "Resend refused it. The onboarding@resend.dev sender can "
                "only deliver to the address on your Resend account, and "
                f"MAIL_TO is {settings.mail_to}. Either change MAIL_TO to "
                "that address, or verify a domain and set MAIL_FROM to "
                f"use it. ({detail})"
            )
        if error.code == 403:
            raise RuntimeError(
                "Resend refused it. Usually MAIL_FROM isn't on a verified "
                f"domain — it's currently {settings.mail_from}. ({detail})"
            )
        if error.code == 401:
            raise RuntimeError(
                f"Resend rejected the API key. ({detail})"
            )
        raise RuntimeError(f"Resend refused it ({error.code}): {detail}")


def _send_via_smtp(
    subject: str,
    body: str,
    reply_to: str,
    to: str = "",
    html: str = "",
) -> None:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.mail_from
    message["To"] = to or settings.mail_to
    if reply_to:
        message["Reply-To"] = reply_to
    message.set_content(body)
    if html:
        # Text first, HTML as the alternative: the order is what tells a
        # client which to prefer.
        message.add_alternative(html, subtype="html")

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
