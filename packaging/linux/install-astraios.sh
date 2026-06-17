#!/usr/bin/env bash
# ============================================================
#  Astraios — Linux smart installer
# ============================================================
#  Tiny download. Detects your GPU and installs the matching PyTorch (CUDA or
#  CPU) plus Astraios into a private environment, then adds a menu/desktop
#  shortcut. Same idea as the Windows installer.
#
#  Uses `uv` (a standalone Python manager) rather than the system Python or
#  apt — so it works the same on Fedora/Bazzite, Arch, Debian, etc. No root
#  needed; everything lands under ~/.local/share/Astraios.
# ============================================================
set -euo pipefail

REPO="majmichu1/Astraios"
APP_DIR="$HOME/.local/share/Astraios"
VENV="$APP_DIR/venv"
DESKTOP="$HOME/.local/share/applications/astraios.desktop"
UV="$APP_DIR/uv"
LOG="$APP_DIR/setup.log"

G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; N='\033[0m'
say()  { echo -e "${G}$*${N}"; }
warn() { echo -e "${Y}$*${N}"; }
die()  { echo -e "${R}✗ $*${N}"; exit 1; }

echo ""
echo "============================================"
echo "   Astraios — Linux installer"
echo "============================================"
echo ""

[ "$(uname -m)" = "x86_64" ] || die "Unsupported architecture $(uname -m) — Astraios needs x86_64."

mkdir -p "$APP_DIR" "$HOME/.local/share/applications"
: > "$LOG"

# ---- 1. Fetch uv (standalone, no system Python required) ----------------
if [ ! -x "$UV" ]; then
    warn "→ Downloading uv (Python manager)..."
    url="https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz"
    curl -fsSL "$url" -o "$APP_DIR/uv.tar.gz" || die "Could not download uv (check your connection)."
    tar -xzf "$APP_DIR/uv.tar.gz" -C "$APP_DIR" --strip-components=1 --wildcards '*/uv'
    rm -f "$APP_DIR/uv.tar.gz"
    chmod +x "$UV"
fi
say "✓ uv ready"

# ---- 2. Detect NVIDIA GPU -> choose the PyTorch build -------------------
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    HAS_GPU=1
elif lspci 2>/dev/null | grep -qi 'nvidia'; then
    HAS_GPU=1
else
    HAS_GPU=0
fi

if [ "$HAS_GPU" = "1" ]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"   # CUDA 12.8 — incl. RTX 50xx
    say "✓ NVIDIA GPU detected — installing CUDA PyTorch"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    warn "→ No NVIDIA GPU detected — installing CPU PyTorch"
fi

# ---- 3. Create the environment (managed Python 3.11) -------------------
if [ ! -x "$VENV/bin/python" ]; then
    warn "→ Creating environment..."
    "$UV" venv --python 3.11 "$VENV" >>"$LOG" 2>&1 || die "Failed to create the environment (see $LOG)."
fi
say "✓ Environment ready"

# ---- 4. Install PyTorch from the chosen index (the big download) -------
warn "→ Installing PyTorch — this is the large download, please wait..."
"$UV" pip install --python "$VENV/bin/python" torch torchvision --index-url "$TORCH_INDEX" >>"$LOG" 2>&1 \
    || die "PyTorch install failed (see $LOG)."
say "✓ PyTorch installed"

# ---- 5. Install Astraios from the latest GitHub release wheel ----------
# (PyPI only for the rest — torch is already in place and satisfies the pin.)
warn "→ Fetching the latest Astraios..."
WHEEL_URL=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
    | grep -o '"browser_download_url": *"[^"]*\.whl"' | head -1 | cut -d'"' -f4)
[ -n "$WHEEL_URL" ] || die "Could not find an Astraios wheel in the latest release."
curl -fsSL "$WHEEL_URL" -o "$APP_DIR/astraios.whl" || die "Failed to download Astraios."
"$UV" pip install --python "$VENV/bin/python" "$APP_DIR/astraios.whl" >>"$LOG" 2>&1 \
    || die "Astraios install failed (see $LOG)."
rm -f "$APP_DIR/astraios.whl"
say "✓ Astraios installed"

# ---- 6. Smoke test -----------------------------------------------------
"$VENV/bin/python" -c "import astraios, torch; print('Astraios', astraios.__version__, '| torch', torch.__version__, '| CUDA', torch.cuda.is_available())" \
    | tee -a "$LOG"

# ---- 7. Desktop / menu shortcut ---------------------------------------
ICON=$(find "$VENV" -name 'astraios.svg' 2>/dev/null | head -1)
[ -n "$ICON" ] || ICON="applications-graphics"
cat > "$DESKTOP" << EOF
[Desktop Entry]
Version=1.0
Name=Astraios
Comment=GPU-accelerated astrophotography image processing
Exec=$VENV/bin/python -m astraios
Icon=$ICON
Terminal=false
Type=Application
Categories=Graphics;Science;Astronomy;
StartupNotify=true
EOF
chmod +x "$DESKTOP"
update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
cp "$DESKTOP" "$HOME/Desktop/" 2>/dev/null && \
    (command -v gio &>/dev/null && gio set "$HOME/Desktop/astraios.desktop" metadata::trusted true 2>/dev/null || true)

echo ""
echo "============================================"
say  "  Installation complete!"
echo "============================================"
echo ""
echo "Launch from your app menu (search 'Astraios'),"
echo "or run:  $VENV/bin/python -m astraios"
echo "Re-run this script any time to update."
echo ""
