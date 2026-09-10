"""The emails, written once and in the site's own colours.

Every message is built here so the wording and the look stay in one
place rather than being scattered through the routers that trigger them.
Each builder returns (subject, html, text): the HTML is what almost
everyone sees, and the plain text is what the rest see, so both have to
say the whole thing rather than one being a stub for the other.

Email is not the web. There is no stylesheet, no class attribute worth
relying on, and no flexbox — so this is tables and inline styles, which
is unlovely and correct. The palette is lifted from styles.css so the
messages look like they came from the same place as the site.
"""

from __future__ import annotations

import html as html_escape

from .config import settings

# From styles.css. Duplicated rather than imported because an email is
# rendered by somebody else's client, months later, from a copy.
NIGHT = "#0b1a16"
PANEL = "#10241e"
LINE = "#1e3b33"
CHALK = "#eaf2ee"
FADE = "#7c978d"
FLOODLIGHT = "#f2c14e"
ON_ACCENT = "#14110a"
FLAG = "#e5484d"

FONT = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "Helvetica, Arial, sans-serif"
)


def _url() -> str:
    return settings.public_base_url or "https://www.fplnerdball.com"


def _escape(value: str) -> str:
    return html_escape.escape(value or "")


def button(label: str, href: str) -> str:
    """A link that looks like a button without needing CSS to do it."""
    return f"""
      <table role="presentation" cellpadding="0" cellspacing="0"
             style="margin:24px 0;">
        <tr>
          <td style="background:{FLOODLIGHT};border-radius:8px;">
            <a href="{href}"
               style="display:inline-block;padding:11px 22px;
                      font-family:{FONT};font-size:15px;font-weight:600;
                      color:{ON_ACCENT};text-decoration:none;">
              {_escape(label)}
            </a>
          </td>
        </tr>
      </table>
    """


def note(text: str, colour: str = FLOODLIGHT) -> str:
    """A boxed aside, for the things people miss and then email you about."""
    return f"""
      <table role="presentation" width="100%" cellpadding="0"
             cellspacing="0" style="margin:18px 0;">
        <tr>
          <td style="padding:12px 14px;border:1px solid {LINE};
                     border-left:3px solid {colour};border-radius:6px;
                     font-family:{FONT};font-size:14px;line-height:1.55;
                     color:{CHALK};">
            {text}
          </td>
        </tr>
      </table>
    """


def layout(title: str, body: str) -> str:
    """The shell every message sits in."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark light">
<title>{_escape(title)}</title>
</head>
<body style="margin:0;padding:0;background:{NIGHT};">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
         style="background:{NIGHT};padding:28px 12px;">
    <tr>
      <td align="center">
        <table role="presentation" width="100%" cellpadding="0"
               cellspacing="0"
               style="max-width:520px;background:{PANEL};
                      border:1px solid {LINE};border-radius:14px;
                      overflow:hidden;">
          <tr>
            <td style="padding:22px 26px 6px;">
              <div style="font-family:{FONT};font-size:19px;
                          font-weight:600;color:{CHALK};
                          letter-spacing:-0.01em;">
                Fantasy<span style="color:{FLOODLIGHT};font-style:italic;
                                    ">Nerdball</span>
              </div>
            </td>
          </tr>
          <tr>
            <td style="padding:6px 26px 26px;font-family:{FONT};
                       font-size:15px;line-height:1.6;color:{CHALK};">
              {body}
            </td>
          </tr>
        </table>

        <table role="presentation" width="100%" cellpadding="0"
               cellspacing="0" style="max-width:520px;">
          <tr>
            <td style="padding:14px 26px;font-family:{FONT};
                       font-size:12px;line-height:1.5;color:{FADE};">
              Sent by Fantasy Nerdball because you asked about access or
              use the site. Not affiliated with, endorsed by or connected
              to the Premier League or Fantasy Premier League.
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""


def heading(text: str) -> str:
    return (
        f'<h1 style="margin:0 0 14px;font-family:{FONT};font-size:20px;'
        f'font-weight:600;color:{CHALK};">{_escape(text)}</h1>'
    )


def para(text: str) -> str:
    return (
        f'<p style="margin:0 0 14px;font-family:{FONT};font-size:15px;'
        f'line-height:1.6;color:{CHALK};">{text}</p>'
    )


def muted(text: str) -> str:
    return (
        f'<p style="margin:0 0 14px;font-family:{FONT};font-size:13px;'
        f'line-height:1.55;color:{FADE};">{text}</p>'
    )


# ── The messages ─────────────────────────────────────────────────────────


def access_received(
    email: str, position: int | None, seats_free: int
) -> tuple[str, str, str]:
    """Confirming a request arrived, and saying where they stand."""
    if position is None:
        standing = para(
            f"There {'is' if seats_free == 1 else 'are'} "
            f"<strong>{seats_free}</strong> "
            f"{'place' if seats_free == 1 else 'places'} free, so you "
            "should hear back shortly."
        )
        standing_text = (
            f"There are {seats_free} places free, so you should hear "
            "back shortly."
        )
    else:
        ordinal = _ordinal(position)
        standing = note(
            f"Every place is currently taken, so you've gone on the "
            f"waiting list. You're <strong>{ordinal} in the queue</strong>."
        ) + muted(
            "Places come free when a manager stops using the site, so "
            "this does move — but I can't say how quickly."
        )
        standing_text = (
            f"Every place is currently taken, so you've gone on the "
            f"waiting list. You're {ordinal} in the queue. Places come "
            "free when a manager stops using the site, so this does "
            "move, but I can't say how quickly."
        )

    body = (
        heading("Your request has arrived")
        + para(
            f"Thanks for asking about Fantasy Nerdball. Your request for "
            f"<strong>{_escape(email)}</strong> is with the site admin."
        )
        + standing
        + para(
            "You'll get another email if you're approved. There's "
            "nothing else for you to do until then."
        )
        + muted(
            "If this wasn't you, ignore this message — nothing has been "
            "created in your name."
        )
    )

    text = (
        "Your request has arrived\n\n"
        f"Thanks for asking about Fantasy Nerdball. Your request for "
        f"{email} is with the site admin.\n\n"
        f"{standing_text}\n\n"
        "You'll get another email if you're approved. There's nothing "
        "else for you to do until then.\n\n"
        "If this wasn't you, ignore this — nothing has been created in "
        "your name."
    )
    return "Fantasy Nerdball — your request has arrived", layout(
        "Your request has arrived", body
    ), text


def access_approved(
    email: str, hours: int, inactive_days: int
) -> tuple[str, str, str]:
    """You're in — with the two deadlines said plainly, twice."""
    url = _url()
    body = (
        heading("You're in")
        + para(
            "You've been given a place on Fantasy Nerdball. It works out "
            "the best Fantasy Premier League squad it can from your "
            "budget, your current side and the fixtures ahead."
        )
        + note(
            f"<strong>Sign in within {hours} hours.</strong> After that "
            "the invitation expires and you'd need to ask again. Places "
            "are limited, so they don't sit unused."
        )
        + button("Sign in now", url)
        + para(
            f"Sign in with the Google account for "
            f"<strong>{_escape(email)}</strong>. It has to be that "
            "address — that's the one that's been approved."
        )
        + heading("Getting started")
        + para(
            "<strong>1. Add your squad.</strong> Either link your FPL "
            "team id and let it import, or enter your fifteen by hand."
            "<br><br>"
            "<strong>2. Check Setup.</strong> How far ahead to look, how "
            "keen it should be to make transfers, and anyone you always "
            "want in or never want near your squad."
            "<br><br>"
            "<strong>3. Run it.</strong> It takes a couple of minutes. "
            "You'll get a squad, a starting eleven, a captain, and the "
            "reasoning behind every pick."
        )
        + note(
            f"Places are freed after <strong>{inactive_days} days</strong> "
            "without signing in, so someone waiting can have a go. You'll "
            "be emailed if that happens, and you can always ask again.",
            colour=FADE,
        )
        + muted(
            "The app is in beta, so expect the odd rough edge. There's a "
            "feedback link in the footer — please use it."
        )
    )

    text = (
        "You're in\n\n"
        "You've been given a place on Fantasy Nerdball.\n\n"
        f"SIGN IN WITHIN {hours} HOURS. After that the invitation "
        "expires and you'd need to ask again.\n\n"
        f"{url}\n\n"
        f"Sign in with the Google account for {email}. It has to be that "
        "address.\n\n"
        "Getting started:\n"
        "1. Add your squad — link your FPL team id, or enter your "
        "fifteen by hand.\n"
        "2. Check Setup — how far ahead to look, how keen it should be "
        "to transfer, and any players to force or avoid.\n"
        "3. Run it. Takes a couple of minutes.\n\n"
        f"Places are freed after {inactive_days} days without signing "
        "in, so someone waiting can have a go. You'll be emailed if that "
        "happens, and you can always ask again.\n\n"
        "The app is in beta. There's a feedback link in the footer."
    )
    return "Fantasy Nerdball — you're in", layout("You're in", body), text


def invite_expired(email: str, hours: int) -> tuple[str, str, str]:
    body = (
        heading("Your invitation has expired")
        + para(
            f"The place held for <strong>{_escape(email)}</strong> wasn't "
            f"used within {hours} hours, so it's gone back to the "
            "waiting list."
        )
        + para(
            "No hard feelings — ask again whenever you like and you'll "
            "rejoin the queue."
        )
        + button("Request access again", _url())
    )
    text = (
        "Your invitation has expired\n\n"
        f"The place held for {email} wasn't used within {hours} hours, "
        "so it's gone back to the waiting list.\n\n"
        f"Ask again whenever you like: {_url()}"
    )
    return (
        "Fantasy Nerdball — your invitation has expired",
        layout("Your invitation has expired", body),
        text,
    )


def removed_for_inactivity(email: str, days: int) -> tuple[str, str, str]:
    body = (
        heading("Your place has been freed up")
        + para(
            f"You haven't signed in to Fantasy Nerdball for {days} days, "
            "so your place has gone to someone on the waiting list. "
            "Places are limited, and this was mentioned when you joined."
        )
        + para(
            "Your squads and settings have been removed along with it. "
            "You're very welcome back — just ask again."
        )
        + button("Request access again", _url())
    )
    text = (
        "Your place has been freed up\n\n"
        f"You haven't signed in to Fantasy Nerdball for {days} days, so "
        "your place has gone to someone on the waiting list. Your squads "
        "and settings have been removed along with it.\n\n"
        f"You're welcome back — ask again: {_url()}"
    )
    return (
        "Fantasy Nerdball — your place has been freed up",
        layout("Your place has been freed up", body),
        text,
    )


# ── To you ───────────────────────────────────────────────────────────────


def admin_access_request(
    email: str, note_text: str, position: int | None, seats: str
) -> tuple[str, str, str]:
    where = (
        "There's room for them."
        if position is None
        else f"All places are taken — they're {_ordinal(position)} in "
        "the queue."
    )
    body = (
        heading("Access request")
        + para(f"<strong>{_escape(email)}</strong> has asked for access.")
        + note(_escape(note_text) if note_text else "No message.")
        + muted(f"{where} Seats: {seats}.")
        + button("Open the admin inbox", _url())
    )
    text = (
        f"{email} has asked for access to Fantasy Nerdball.\n\n"
        f"{note_text or 'No message.'}\n\n"
        f"{where}\nSeats: {seats}.\n\n"
        "Approve or dismiss it in the admin inbox."
    )
    return f"Access request — {email}", layout("Access request", body), text


def admin_feedback(
    kind_label: str, who: str, message: str
) -> tuple[str, str, str]:
    body = (
        heading(kind_label)
        + muted(f"From {_escape(who)}")
        + note(_escape(message).replace("\n", "<br>"))
        + button("Open the admin inbox", _url())
    )
    text = f"{kind_label}\n\nFrom: {who}\n\n{message}"
    return (
        f"{kind_label} — Fantasy Nerdball",
        layout(kind_label, body),
        text,
    )


def admin_removed(email: str, days: int) -> tuple[str, str, str]:
    body = (
        heading("A place has been freed")
        + para(
            f"<strong>{_escape(email)}</strong> hasn't signed in for "
            f"{days} days, so their account has been removed and the "
            "place is free."
        )
        + muted("They've been emailed and can ask for it back.")
    )
    text = (
        f"{email} hasn't signed in for {days} days, so their account has "
        "been removed and the place is free. They've been emailed."
    )
    return (
        f"Place freed — {email}",
        layout("A place has been freed", body),
        text,
    )


def admin_test() -> tuple[str, str, str]:
    body = (
        heading("It works")
        + para(
            "If you're reading this, access requests and feedback will "
            "reach you too."
        )
    )
    return (
        "Fantasy Nerdball — test",
        layout("It works", body),
        "It works. Access requests and feedback will reach you too.",
    )


def _ordinal(number: int) -> str:
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"
