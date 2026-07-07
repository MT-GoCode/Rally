#!/bin/bash
# Build, sign (Developer ID → self-signed → ad-hoc, via minh-mac-utils' shared picker),
# and install CIVM.app to /Applications. Idempotent — rerun after any code change.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

cd "$HERE/app"
swift build -c release
cp .build/release/CIVM CIVM.app/Contents/MacOS/CIVM
# SwiftPM resource bundles (Bundle.module): the markdown renderer's KaTeX/marked/highlight assets
# (CIVM_CIVM.bundle) + Highlightr's. Bundle.module resolves them from Contents/Resources.
mkdir -p CIVM.app/Contents/Resources
find CIVM.app/Contents/Resources -maxdepth 1 -name '*.bundle' -exec rm -rf {} +
for b in .build/release/*.bundle; do [ -e "$b" ] && cp -R "$b" CIVM.app/Contents/Resources/; done

SIGN_SH="$HOME/code/minh-mac-utils/sign-identity.sh"
if [ -f "$SIGN_SH" ]; then ID="$(bash "$SIGN_SH")"; else ID="${CODESIGN_IDENTITY:--}"; fi
[ -z "${ID:-}" ] && ID="-"

echo "▸ codesign with: $ID"
SIGN=(--force --options runtime --entitlements civm.entitlements --sign "$ID")
[ "$ID" != "-" ] && SIGN+=(--timestamp)
codesign "${SIGN[@]}" CIVM.app
codesign --verify --strict --verbose=2 CIVM.app

# ~/Applications only (Spotlight-indexed; user's choice — avoids the /Applications duplicate)
DEST="$HOME/Applications"; mkdir -p "$DEST"
rm -rf "$DEST/Rally.app" "$DEST/CIVM.app"
ditto CIVM.app "$DEST/Rally.app"
echo "✓ installed → $DEST/Rally.app"
echo "  launch: Spotlight 'Rally'  ·  open -a Rally  ·  drag to Dock for one-click"
