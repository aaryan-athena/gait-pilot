#!/usr/bin/env bash
# Install gaitedge on a Raspberry Pi 5 running Raspberry Pi OS (64-bit, Bookworm).
#
#   git clone <this repository> ~/gait && sudo bash ~/gait/edge/deploy/install_pi.sh
#
# Re-running it upgrades in place; settings in /etc/gaitedge are never
# overwritten.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX=/opt/gaitedge
STATE=/var/lib/gaitedge
CONF=/etc/gaitedge

if [[ $EUID -ne 0 ]]; then echo "run with sudo" >&2; exit 1; fi
if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "64-bit Raspberry Pi OS is required (MediaPipe has no 32-bit ARM build)." >&2
  exit 1
fi

echo "== system packages"
apt-get update
# python3-picamera2: the Pi camera library, only packaged through apt.
# ffmpeg: makes the annotated replay playable in a browser.
apt-get install -y python3-venv python3-picamera2 ffmpeg rsync curl

echo "== service account"
id gaitedge >/dev/null 2>&1 || useradd --system --home "$STATE" --shell /usr/sbin/nologin gaitedge
usermod -aG video,render gaitedge
# Let the person installing read recordings and results without sudo. The
# storage folder is closed to everyone else: it holds footage of people.
if [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
  usermod -aG gaitedge "$SUDO_USER"
  ADDED_USER="$SUDO_USER"
fi

echo "== code"
mkdir -p "$PREFIX"
rsync -a --delete --exclude venv --exclude data --exclude 'app/samples' \
  --exclude '.git' --exclude '__pycache__' "$REPO"/ "$PREFIX"/src-tree/
ln -sfn "$PREFIX/src-tree/edge" "$PREFIX/edge"

echo "== python environment"
# --system-site-packages so the apt-installed picamera2 is importable.
[[ -d "$PREFIX/venv" ]] || python3 -m venv --system-site-packages "$PREFIX/venv"
"$PREFIX/venv/bin/pip" install --upgrade pip
"$PREFIX/venv/bin/pip" install -r "$PREFIX/edge/requirements.txt"
# The analysis package, without its desktop OpenCV dependency. Editable (-e)
# so it stays in the source tree: it reads config/default.yaml from the
# repository root, which a normal install would leave behind.
"$PREFIX/venv/bin/pip" install --no-deps -e "$PREFIX/src-tree"
# Make the device code importable from any working directory, so
# `python -m gaitedge ...` works wherever it is run from, not only from
# inside the edge folder.
SITE="$("$PREFIX/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$PREFIX/edge" > "$SITE/gaitedge.pth"

echo "== settings and storage"
mkdir -p "$CONF" "$STATE"
[[ -f "$CONF/edge.yaml" ]] || cp "$PREFIX/edge/config/edge.example.yaml" "$CONF/edge.yaml"
chown -R gaitedge:gaitedge "$STATE"
chmod 750 "$STATE"
chmod 640 "$CONF/edge.yaml"; chgrp gaitedge "$CONF/edge.yaml"

echo "== service"
cp "$PREFIX/edge/deploy/gaitedge.service" /etc/systemd/system/gaitedge.service
systemctl daemon-reload
systemctl enable gaitedge

echo "== live view on an attached screen"
chmod +x "$PREFIX/edge/deploy/preview_kiosk.sh"
if [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
  # At this user's desktop login, open the camera full screen if a monitor
  # is connected. Delete the file to stop it.
  USER_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
  install -d -o "$SUDO_USER" -g "$SUDO_USER" "$USER_HOME/.config/autostart"
  install -m 644 -o "$SUDO_USER" -g "$SUDO_USER" "$PREFIX/edge/deploy/gaitedge-preview.desktop" \
    "$USER_HOME/.config/autostart/gaitedge-preview.desktop"
fi

if [[ -n "${ADDED_USER:-}" ]]; then
  echo "Added $ADDED_USER to the gaitedge group: log out and back in to browse $STATE."
fi

cat <<EOF

Installed. Before starting the service:

  1. Edit $CONF/edge.yaml: device id, coverage_m, server_url and token.
  2. Check the unit end to end:
       sudo -u gaitedge $PREFIX/venv/bin/python -m gaitedge --config $CONF/edge.yaml doctor
  3. Measure what this Pi can do (record a walk first with
     \`... gaitedge record --seconds 40\`, or copy one over):
       sudo -u gaitedge $PREFIX/venv/bin/python -m gaitedge --config $CONF/edge.yaml \\
         bench --clip <walk.mp4> --accuracy
  4. sudo systemctl start gaitedge && journalctl -u gaitedge -f
EOF
