"""StarNet Integration — GPL-isolated star removal via subprocess.

StarNet is GPL licensed, so it MUST be run as a subprocess only.
No imports of StarNet code are allowed. Communication is via temp files.

The reference command-line tool is StarNet++ v2 (``StarNetv2CLI``). It reads and
writes **16-bit TIFF** (not FITS), handles colour natively, and loads its neural
network weights from files next to the executable — so the subprocess must run
with its working directory set to the binary's folder, or it fails to find them.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Executable names across platforms/versions, most-specific first.
_STARNET_NAMES = (
    "StarNetv2CLI",
    "starnet2",
    "starnet++",
    "starnet",
    "StarNetv2CLI.exe",
    "starnet2.exe",
    "starnet++.exe",
    "starnet.exe",
)


@dataclass
class StarNetResult:
    """Result from StarNet processing."""

    starless: np.ndarray  # image with stars removed
    stars_only: np.ndarray | None = None  # extracted stars (original - starless)
    success: bool = True
    message: str = ""


def find_starnet_binary() -> Path | None:
    """Find the StarNet binary on the system.

    A path the user set in Preferences wins; otherwise we search PATH and the
    usual install locations (including one directory level deep, since StarNet
    typically unzips into its own subfolder).

    Returns
    -------
    Path or None
        Path to StarNet binary, or None if not found.
    """
    from astraios.core.user_paths import starnet_binary as _configured_starnet

    configured = _configured_starnet()
    if configured is not None:
        return configured

    for name in _STARNET_NAMES:
        path = shutil.which(name)
        if path:
            return Path(path)

    common_bases = [
        Path.home() / "StarNet",
        Path.home() / "starnet",
        Path.home() / ".local" / "bin",
        Path("/usr/local/bin"),
        Path("/opt/starnet"),
        Path("/opt/StarNet"),
        Path("C:/Program Files/StarNet"),
        Path("C:/StarNet"),
    ]
    for base in common_bases:
        if not base.exists():
            continue
        for name in _STARNET_NAMES:
            candidate = base / name
            if candidate.is_file():
                return candidate
        # One level deep — StarNet often extracts into a versioned subfolder.
        try:
            for sub in base.iterdir():
                if not sub.is_dir():
                    continue
                for name in _STARNET_NAMES:
                    candidate = sub / name
                    if candidate.is_file():
                        return candidate
        except OSError:
            pass

    return None


def _write_starnet_tiff(image: np.ndarray, path: Path) -> None:
    """Write float [0,1] ``(H, W)`` or ``(C, H, W)`` as a true 16-bit TIFF.

    Uses ``tifffile`` so colour images keep full 16-bit depth (Pillow silently
    downgrades 16-bit RGB to 8-bit, which would posterize faint nebulosity).
    Written uncompressed for maximum reader compatibility.
    """
    import tifffile

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:  # (C, H, W) -> (H, W, C)
        arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[2] > 3:
            arr = arr[:, :, :3]

    u16 = (np.clip(arr, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    photometric = "rgb" if u16.ndim == 3 else "minisblack"
    tifffile.imwrite(str(path), u16, photometric=photometric)


def _read_starnet_tiff(path: Path) -> np.ndarray:
    """Read a StarNet output TIFF as float [0,1] in ``(H, W)`` or ``(C, H, W)``."""
    import tifffile

    raw = np.asarray(tifffile.imread(str(path)))
    arr = raw.astype(np.float32)

    if raw.dtype == np.uint16:
        arr /= 65535.0
    elif raw.dtype == np.uint8:
        arr /= 255.0
    elif np.issubdtype(raw.dtype, np.floating):
        peak = float(arr.max()) if arr.size else 1.0
        if peak > 1.5:  # some builds write float in [0, 65535]
            arr /= peak
    else:
        peak = float(arr.max()) if arr.size else 1.0
        arr /= peak or 1.0

    if arr.ndim == 3:  # (H, W, C) -> (C, H, W)
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            arr = np.transpose(arr[:, :, :3], (2, 0, 1))
    return np.clip(arr, 0.0, 1.0)


def run_starnet(
    image: np.ndarray,
    starnet_path: Path | str | None = None,
    extract_stars: bool = True,
) -> StarNetResult:
    """Run StarNet as a subprocess for GPL isolation.

    Parameters
    ----------
    image : ndarray
        Image data, shape (H, W) or (C, H, W), float32 in [0, 1].
    starnet_path : Path or str, optional
        Path to StarNet binary. If None, auto-detected.
    extract_stars : bool
        If True, also compute stars-only image (original - starless).

    Returns
    -------
    StarNetResult
        Result containing starless image and optionally stars-only. On any
        failure ``success`` is False, ``starless`` is the unmodified input, and
        ``message`` explains why so the caller can fall back gracefully.
    """
    if starnet_path is None:
        starnet_path = find_starnet_binary()
    else:
        starnet_path = Path(starnet_path)

    if starnet_path is None or not starnet_path.exists():
        return StarNetResult(
            starless=image.copy(),
            success=False,
            message=(
                "StarNet binary not found. Install StarNet v2 (StarNetv2CLI) and "
                "set its path in Preferences, or add it to PATH."
            ),
        )

    original = np.asarray(image, dtype=np.float32)

    try:
        with tempfile.TemporaryDirectory(prefix="astraios_starnet_") as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.tif"
            output_path = tmpdir / "starless.tif"

            _write_starnet_tiff(original, input_path)

            cmd = [str(starnet_path), str(input_path), str(output_path)]
            log.info("Running StarNet: %s", " ".join(cmd))

            # StarNet loads its model weights from its own directory, so run the
            # subprocess from there rather than wherever Astraios was launched.
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900,  # 15 min — large colour frames are slow
                cwd=str(starnet_path.parent),
            )

            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip()
                return StarNetResult(
                    starless=original.copy(),
                    success=False,
                    message=f"StarNet exited with code {result.returncode}: {detail}",
                )

            if not output_path.exists():
                return StarNetResult(
                    starless=original.copy(),
                    success=False,
                    message=(
                        "StarNet produced no output file. Check that the binary is "
                        "the command-line version (StarNetv2CLI) and its model "
                        "files are present beside it."
                    ),
                )

            starless = _read_starnet_tiff(output_path)

            # Coerce layout back to the input's (StarNet may emit mono as RGB).
            if original.ndim == 2 and starless.ndim == 3:
                starless = starless.mean(axis=0)
            elif original.ndim == 3 and starless.ndim == 2:
                starless = np.broadcast_to(starless, original.shape).copy()

            if starless.shape != original.shape:
                return StarNetResult(
                    starless=original,
                    success=False,
                    message=(
                        f"StarNet output shape {starless.shape} does not match "
                        f"input {original.shape}"
                    ),
                )

            starless = np.clip(starless, 0.0, 1.0).astype(np.float32)
            stars_only = None
            if extract_stars:
                stars_only = np.clip(original - starless, 0.0, 1.0).astype(np.float32)

            return StarNetResult(
                starless=starless,
                stars_only=stars_only,
                success=True,
                message="StarNet processing complete",
            )

    except subprocess.TimeoutExpired:
        return StarNetResult(
            starless=image.copy(),
            success=False,
            message="StarNet timed out after 15 minutes",
        )
    except Exception as e:
        return StarNetResult(
            starless=image.copy(),
            success=False,
            message=f"StarNet error: {e}",
        )
