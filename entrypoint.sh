#!/bin/sh
# Fixes ownership of the mounted volume, then drops to the app user.
#
# A volume is attached at runtime and arrives owned by root, which discards
# whatever ownership the image set at build time. So the container starts as
# root, corrects the mount, and hands off to the unprivileged user. Nothing
# after this line runs with root privileges.

set -e

DATA_DIR="${NERDBALL_DATA_DIR:-/data}"
APP_USER="nerdball"
APP_UID="10001"

if [ "$(id -u)" = "0" ]; then
    mkdir -p "$DATA_DIR"

    # Re-chowning a full season of cached data on every boot is slow and
    # pointless, so only do it when the mount isn't already ours.
    current_owner="$(stat -c %u "$DATA_DIR" 2>/dev/null || echo unknown)"
    if [ "$current_owner" != "$APP_UID" ]; then
        echo "Taking ownership of $DATA_DIR (was uid $current_owner)"
        chown -R "$APP_USER:$APP_USER" "$DATA_DIR"
    fi

    if command -v gosu >/dev/null 2>&1; then
        exec gosu "$APP_USER" "$@"
    elif command -v setpriv >/dev/null 2>&1; then
        exec setpriv --reuid="$APP_UID" --regid="$APP_UID" --init-groups "$@"
    else
        # Neither is present. Starting as root beats not starting at all, but
        # this shouldn't happen: the image installs gosu.
        echo "WARNING: no gosu or setpriv found, continuing as root" >&2
        exec "$@"
    fi
fi

# Already unprivileged, which is the case when the image is run with a
# --user flag. Nothing to fix; just start.
exec "$@"
