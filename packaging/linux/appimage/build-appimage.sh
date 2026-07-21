#!/usr/bin/env bash
# Build the Astraios AppImage.
#
# Produces a single portable x86_64 file that runs on any glibc >= 2.28
# distro (Fedora/Bazzite/Silverblue, Ubuntu 20.04+, Arch, ...) with no
# installation and no root. It bundles its own Python, PyQt6, numpy, scipy,
# astropy and Astraios; PyTorch is provisioned per-machine on first launch
# by _appimage_bootstrap.py (see AppRun for why).
#
# Usage:  packaging/linux/appimage/build-appimage.sh [version]
# Output: dist/Astraios-<version>-x86_64.AppImage

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
VERSION="${1:-$(python3 -c "import tomllib;print(tomllib.load(open('${REPO}/pyproject.toml','rb'))['project']['version'])")}"

BUILD="${BUILD_DIR:-${REPO}/build/appimage}"
APPDIR="${BUILD}/Astraios.AppDir"
CACHE="${BUILD}/cache"

# Pinned so a rebuild of an old tag reproduces: python-build-standalone is a
# rolling release and only keeps recent assets.
PBS_TAG="20260623"
PBS_FILE="cpython-3.12.13+${PBS_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz"
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${PBS_FILE}"
APPIMAGETOOL_URL="https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"

echo "==> Building Astraios ${VERSION} AppImage"
rm -rf "${APPDIR}"
mkdir -p "${APPDIR}/usr/lib/astraios" "${CACHE}" "${REPO}/dist"

# ---------------------------------------------------------------- Python ----
if [ ! -f "${CACHE}/${PBS_FILE}" ]; then
    echo "==> Downloading standalone CPython 3.12"
    curl -fL --retry 3 -o "${CACHE}/${PBS_FILE}" "${PBS_URL}"
fi
echo "==> Unpacking interpreter"
tar -xzf "${CACHE}/${PBS_FILE}" -C "${BUILD}"
# the tarball unpacks to ./python
rm -rf "${APPDIR}/usr"
mkdir -p "${APPDIR}/usr"
cp -a "${BUILD}/python/." "${APPDIR}/usr/"
rm -rf "${BUILD}/python"
PY="${APPDIR}/usr/bin/python3"
"${PY}" --version

# ------------------------------------------------------------------ Deps ----
# Everything except torch/torchvision, which AppRun provisions per machine.
echo "==> Installing dependencies (excluding PyTorch)"
"${PY}" -m pip install --no-cache-dir --upgrade pip >/dev/null
"${PY}" -m pip install --no-cache-dir --target "${APPDIR}/usr/lib/astraios" \
    "PyQt6>=6.6,<7" \
    "numpy>=1.26,<3" \
    "scipy>=1.12,<2" \
    "astropy>=6.0" \
    "ccdproc>=2.4" \
    "opencv-python-headless>=4.9,<5" \
    "Pillow>=10.2" \
    "requests>=2.31,<3" \
    "packaging>=24.0" \
    "platformdirs>=4.2,<5" \
    "qimage2ndarray>=1.10,<2" \
    "PyWavelets>=1.5" \
    "scikit-image>=0.22"

# ------------------------------------------------------------- Astraios -----
echo "==> Installing Astraios"
# Must match THIS build's version. dist/ is a shared output dir that can hold
# stale wheels from earlier releases -- picking one of those silently ships
# months-old code in a correctly-named AppImage, so match on version and
# rebuild if the right wheel is not already there.
WHEEL_GLOB="${REPO}/dist/astraios-${VERSION}-*.whl"
WHEEL="$(ls ${WHEEL_GLOB} 2>/dev/null | head -1 || true)"
if [ -z "${WHEEL}" ]; then
    echo "    no wheel for ${VERSION}, building one"
    (cd "${REPO}" && python3 -m pip install --no-cache-dir poetry >/dev/null \
        && poetry build --format wheel >/dev/null)
    WHEEL="$(ls ${WHEEL_GLOB} 2>/dev/null | head -1 || true)"
fi
if [ -z "${WHEEL}" ]; then
    echo "ERROR: could not produce a wheel for version ${VERSION}" >&2
    exit 1
fi
echo "    using ${WHEEL}"
"${PY}" -m pip install --no-cache-dir --no-deps --target "${APPDIR}/usr/lib/astraios" "${WHEEL}"

# The first-run provisioner lives beside the app code.
cp "${HERE}/_appimage_bootstrap.py" "${APPDIR}/usr/lib/astraios/"

# ------------------------------------------------------------------ Trim ----
# Strip test suites, caches and bundled Qt modules we never load. This is the
# difference between a ~700 MB and a ~300 MB download.
echo "==> Trimming"
SP="${APPDIR}/usr/lib/astraios"
find "${SP}" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
# NOTE: do NOT blanket-delete */tests. astropy does `from .tests.runner import
# TestRunner` in its __init__, so removing its test package breaks `import
# astropy` outright. Only drop the test suites of packages that are known not
# to import them at module scope; the verification step below is what actually
# guarantees we did not cut too deep.
for pkg in scipy numpy skimage cv2 PIL; do
    find "${SP}/${pkg}" -type d \( -name "tests" -o -name "test" \) \
        -prune -exec rm -rf {} + 2>/dev/null || true
done
rm -rf "${SP}"/PyQt6/Qt6/qml \
       "${SP}"/PyQt6/Qt6/translations \
       "${SP}"/PyQt6/Qt6/lib/libQt6WebEngine* \
       "${SP}"/PyQt6/Qt6/lib/libQt6Quick* \
       "${SP}"/PyQt6/Qt6/lib/libQt6Qml* \
       "${SP}"/PyQt6/Qt6/lib/libQt6Designer* \
       "${SP}"/PyQt6/Qt6/lib/libQt6Pdf* \
       "${SP}"/PyQt6/Qt6/lib/libQt6Bluetooth* \
       "${SP}"/PyQt6/Qt6/lib/libQt6Nfc* \
       2>/dev/null || true
# The standalone interpreter ships a full test suite and static lib.
rm -rf "${APPDIR}/usr/lib/python3.12/test" \
       "${APPDIR}/usr/lib/python3.12/idlelib" \
       "${APPDIR}/usr/lib/python3.12/tkinter" \
       "${APPDIR}"/usr/lib/libpython3.12.a \
       2>/dev/null || true
find "${APPDIR}/usr/lib/python3.12" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true

# --------------------------------------------------------------- AppDir -----
echo "==> Assembling AppDir"
install -m 755 "${HERE}/AppRun" "${APPDIR}/AppRun"
install -m 644 "${HERE}/astraios.desktop" "${APPDIR}/astraios.desktop"
install -m 644 "${REPO}/packaging/linux/astraios.png" "${APPDIR}/astraios.png"
mkdir -p "${APPDIR}/usr/share/icons/hicolor/256x256/apps"
install -m 644 "${REPO}/packaging/linux/astraios.png" \
    "${APPDIR}/usr/share/icons/hicolor/256x256/apps/astraios.png"
mkdir -p "${APPDIR}/usr/share/applications"
install -m 644 "${HERE}/astraios.desktop" "${APPDIR}/usr/share/applications/astraios.desktop"

du -sh "${APPDIR}"

# ----------------------------------------------------------- Verify --------
# Trimming has broken imports before (astropy's __init__ pulls in its own test
# runner). Import the whole stack with the bundled interpreter so a bad trim
# fails the build here instead of shipping.
echo "==> Verifying the bundled stack imports"
PYTHONPATH="${SP}" EXPECTED_VERSION="${VERSION}" "${PY}" - <<'PYCHECK'
import os
import sys

for mod in ("PyQt6.QtWidgets", "numpy", "scipy", "astropy.io.fits", "cv2",
            "skimage", "PIL", "pywt", "qimage2ndarray", "ccdproc",
            "platformdirs", "requests", "astraios"):
    __import__(mod)
print("  all bundled imports OK")

# Guard against packaging a stale wheel. pyproject carries the PEP 440 build
# form ("0.1.23.dev0") while __version__ uses the human form ("0.1.23-dev");
# Version() normalizes both to the same thing, so compare parsed versions
# rather than strings.
import astraios
from packaging.version import InvalidVersion, Version

expected_raw = os.environ["EXPECTED_VERSION"]
try:
    same = Version(astraios.__version__) == Version(expected_raw)
except InvalidVersion:
    same = astraios.__version__ == expected_raw
if not same:
    sys.exit(f"ERROR: bundled Astraios is {astraios.__version__}, expected "
             f"{expected_raw} (a stale wheel in dist/ was packaged)")
print(f"  bundled Astraios version {astraios.__version__} matches the build")

# torch must NOT be bundled: it is provisioned per machine on first launch.
try:
    import torch  # noqa: F401
except ImportError:
    print("  torch correctly absent (provisioned on first run)")
else:
    sys.exit("ERROR: torch was bundled; the AppImage would exceed the asset limit")
PYCHECK

# ------------------------------------------------------------- AppImage -----
if [ ! -f "${CACHE}/appimagetool" ]; then
    echo "==> Downloading appimagetool"
    curl -fL --retry 3 -o "${CACHE}/appimagetool" "${APPIMAGETOOL_URL}"
    chmod +x "${CACHE}/appimagetool"
fi

OUT="${REPO}/dist/Astraios-${VERSION}-x86_64.AppImage"
echo "==> Packing ${OUT}"
# --appimage-extract-and-run: works on CI and in containers without FUSE.
ARCH=x86_64 "${CACHE}/appimagetool" --appimage-extract-and-run \
    --comp zstd --no-appstream "${APPDIR}" "${OUT}"

chmod +x "${OUT}"
ls -lh "${OUT}"
echo "==> Done: ${OUT}"
