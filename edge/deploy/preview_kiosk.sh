#!/usr/bin/env bash
# Show the live camera view full screen when a monitor is attached.
#
# Started at desktop login by ~/.config/autostart/gaitedge-preview.desktop
# (install_pi.sh sets that up). The service itself runs as a system account
# that cannot open windows on the desktop, so it only publishes the view; this
# opens it. With no monitor connected it exits and does nothing.
#
# Alt+F4 closes the view. Run this script again to reopen it.
set -u

URL="${GAITEDGE_PREVIEW_URL:-http://127.0.0.1:8080/}"

screen_connected() {
  local s
  for s in /sys/class/drm/card*-*/status; do
    [[ -r "$s" ]] && [[ "$(cat "$s")" == "connected" ]] && return 0
  done
  return 1
}

if ! screen_connected; then
  echo "gaitedge preview: no monitor connected; not opening the live view"
  exit 0
fi

# The service may still be starting (the camera and model take a few
# seconds); a browser opened too early shows an error page and stays there.
for _ in $(seq 1 90); do
  curl -fsS -o /dev/null "${URL%/}/status" 2>/dev/null && break
  sleep 2
done

PROFILE="$HOME/.cache/gaitedge-preview"   # kept apart from the normal browser
for browser in chromium-browser chromium; do
  if command -v "$browser" >/dev/null 2>&1; then
    exec "$browser" --kiosk --app="$URL" --user-data-dir="$PROFILE" \
      --noerrdialogs --disable-infobars --no-first-run \
      --disable-session-crashed-bubble --password-store=basic
  fi
done
if command -v firefox >/dev/null 2>&1; then
  exec firefox --kiosk "$URL"
fi
echo "gaitedge preview: no browser found; open $URL by hand" >&2
exit 1
