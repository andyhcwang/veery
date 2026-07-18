#!/bin/bash
# Build the Veery dev .app (py2app alias mode) and install it to /Applications
# so Spotlight can launch it: press Cmd+Space, type "Veery", hit Enter.
#
# The alias-mode bundle runs against THIS checkout (live source + .venv), so:
#   - re-run this script only if the checkout moves; code changes are picked
#     up automatically on next launch (no rebuild needed)
#   - don't delete or move this directory while the app is installed
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

echo "==> Installing app-build dependencies (py2app)..."
uv sync --group app

echo "==> Building alias-mode Veery.app..."
./.venv/bin/python setup.py py2app -A >/dev/null

APP_SRC="$REPO_ROOT/dist/Veery.app"
APP_DST="/Applications/Veery.app"

if [[ ! -d "$APP_SRC" ]]; then
    echo "Build failed: $APP_SRC not found" >&2
    exit 1
fi

echo "==> Installing to $APP_DST..."
rm -rf "$APP_DST"
ditto "$APP_SRC" "$APP_DST"

echo ""
echo "Done. Press Cmd+Space and type 'Veery' to launch."
echo "The app runs against: $REPO_ROOT"
echo "On first launch, grant Microphone / Accessibility / Input Monitoring"
echo "permissions to 'Veery' when the permission guide appears."
