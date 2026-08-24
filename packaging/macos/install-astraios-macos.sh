#!/usr/bin/env bash
# ============================================================
#  Astraios — macOS smart installer
# ============================================================
#  Tiny download. Installs PyTorch and Astraios into a private environment
#  and creates an Astraios.app you can launch from Launchpad or Spotlight.
#  Same idea as the Windows and Linux installers.
#
#  Apple Silicon Macs get real GPU acceleration: PyTorch's Metal (MPS)
#  backend is used automatically, which Astraios already supports alongside
#  CUDA. Intel Macs run on CPU.
#
#  Uses `uv` (a standalone Python manager) rather than the system Python or
#  Homebrew, so it does not care what else is installed. No root needed;
#  everything lands under ~/Library/Application Support/Astraios.
# ============================================================
set -euo pipefail

REPO="majmichu1/Astraios"
APP_DIR="$HOME/Library/Application Support/Astraios"
VENV="$APP_DIR/venv"
UV="$APP_DIR/uv"
LOG="$APP_DIR/setup.log"
APP_BUNDLE="${ASTRAIOS_APP_BUNDLE:-$HOME/Applications/Astraios.app}"

G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; N='\033[0m'
say()  { echo -e "${G}$*${N}"; }
warn() { echo -e "${Y}$*${N}"; }
die()  { echo -e "${R}x $*${N}"; exit 1; }

echo ""
echo "============================================"
echo "   Astraios - macOS installer"
echo "============================================"
echo ""

[ "$(uname -s)" = "Darwin" ] || die "This installer is for macOS. On Linux use install-astraios.sh."

ARCH="$(uname -m)"
case "$ARCH" in
    arm64)  UV_ASSET="uv-aarch64-apple-darwin.tar.gz";  say "* Apple Silicon detected - Metal (MPS) GPU acceleration will be used" ;;
    x86_64) UV_ASSET="uv-x86_64-apple-darwin.tar.gz";   warn "-> Intel Mac detected - Astraios will run on CPU" ;;
    *)      die "Unsupported architecture $ARCH." ;;
esac

mkdir -p "$APP_DIR" "$(dirname "$APP_BUNDLE")"
: > "$LOG"

# ---- 1. Fetch uv (standalone, no system Python required) ----------------
if [ ! -x "$UV" ]; then
    warn "-> Downloading uv (Python manager)..."
    url="https://github.com/astral-sh/uv/releases/latest/download/$UV_ASSET"
    curl -fsSL "$url" -o "$APP_DIR/uv.tar.gz" || die "Could not download uv (check your connection)."
    # BSD tar has no --wildcards; extract the whole archive and move the binary.
    tar -xzf "$APP_DIR/uv.tar.gz" -C "$APP_DIR"
    found="$(find "$APP_DIR" -maxdepth 3 -type f -name uv -perm -u+x | head -1)"
    [ -n "$found" ] || die "uv binary not found in the downloaded archive."
    [ "$found" = "$UV" ] || mv "$found" "$UV"
    rm -rf "$APP_DIR/uv.tar.gz" "$APP_DIR"/uv-*-apple-darwin
    chmod +x "$UV"
fi
say "* uv ready"

# ---- 2. Create the environment (managed Python 3.11) -------------------
if [ ! -x "$VENV/bin/python" ]; then
    warn "-> Creating environment..."
    "$UV" venv --python 3.11 "$VENV" >>"$LOG" 2>&1 || die "Failed to create the environment (see $LOG)."
fi
say "* Environment ready"

# ---- 3. Install PyTorch ------------------------------------------------
# macOS wheels on PyPI already carry the Metal (MPS) backend on Apple
# Silicon, so there is no separate index to choose the way CUDA needs on
# Linux and Windows.
warn "-> Installing PyTorch - this is the large download, please wait..."
"$UV" pip install --python "$VENV/bin/python" torch torchvision >>"$LOG" 2>&1 \
    || die "PyTorch install failed (see $LOG)."
say "* PyTorch installed"

# ---- 4. Install Astraios from the latest GitHub release wheel ----------
warn "-> Fetching the latest Astraios..."
WHEEL_URL=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
    | grep -o '"browser_download_url": *"[^"]*\.whl"' | head -1 | cut -d'"' -f4)
[ -n "$WHEEL_URL" ] || die "Could not find an Astraios wheel in the latest release."
# Keep the wheel's real PEP 427 filename; uv rejects a versionless name.
WHEEL_FILE="$APP_DIR/$(basename "$WHEEL_URL")"
curl -fsSL "$WHEEL_URL" -o "$WHEEL_FILE" || die "Failed to download Astraios."
"$UV" pip install --python "$VENV/bin/python" "$WHEEL_FILE" >>"$LOG" 2>&1 \
    || die "Astraios install failed (see $LOG)."
rm -f "$WHEEL_FILE"
say "* Astraios installed"

# ---- 5. Smoke test -----------------------------------------------------
"$VENV/bin/python" - <<'PYCHECK' | tee -a "$LOG"
import astraios, torch
mps = getattr(torch.backends, "mps", None)
print(f"Astraios {astraios.__version__} | torch {torch.__version__} | "
      f"MPS available: {bool(mps and mps.is_available())}")
PYCHECK

# ---- 6. Application bundle --------------------------------------------
# A minimal .app so Astraios appears in Launchpad and Spotlight like any
# other Mac application, rather than only being runnable from a terminal.
warn "-> Creating Astraios.app..."
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources"

cat > "$APP_BUNDLE/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Astraios</string>
    <key>CFBundleDisplayName</key><string>Astraios</string>
    <key>CFBundleIdentifier</key><string>app.astraios.desktop</string>
    <key>CFBundleVersion</key><string>1.0</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleExecutable</key><string>astraios</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
</dict>
</plist>
PLIST

cat > "$APP_BUNDLE/Contents/MacOS/astraios" << LAUNCHER
#!/bin/bash
exec "$VENV/bin/python" -m astraios "\$@"
LAUNCHER
chmod +x "$APP_BUNDLE/Contents/MacOS/astraios"
say "* Astraios.app created"

echo ""
echo "============================================"
say  "  Installation complete!"
echo "============================================"
echo ""
echo "Launch Astraios from Launchpad or Spotlight,"
echo "or run:  \"$VENV/bin/python\" -m astraios"
echo "Re-run this script any time to update."
echo ""
