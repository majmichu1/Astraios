"""StarNet Integration — GPL-isolated star removal via subprocess.

StarNet is GPL licensed, so it MUST be run as a subprocess only.
No imports of StarNet code are allowed. Communication is via temp files.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from astraios.core.image_io import ImageData, load_image, save_fits

log = logging.getLogger(__name__)


@dataclass
class StarNetResult:
    """Result from StarNet processing."""

    starless: np.ndarray  # image with stars removed
    stars_only: np.ndarray | None = None  # extracted stars (original - starless)
    success: bool = True
    message: str = ""


def find_starnet_binary() -> Path | None:
    """Find the StarNet binary on the system.

    Searches common installation locations and PATH.

    Returns
    -------
    Path or None
        Path to StarNet binary, or None if not found.
    """
    # A path the user set in Preferences wins over auto-detection.
    from astraios.core.user_paths import starnet_binary as _configured_starnet

    configured = _configured_starnet()
    if configured is not None:
        return configured

    # Check common names
    for name in ("starnet++", "starnet2", "StarNetv2CLI", "starnet"):
        path = shutil.which(name)
        if path:
            return Path(path)

    # Check common install locations
    common_paths = [
        Path.home() / "StarNet",
        Path.home() / ".local" / "bin",
        Path("/usr/local/bin"),
        Path("/opt/starnet"),
    ]
    for base in common_paths:
        for name in ("starnet++", "starnet2", "StarNetv2CLI"):
            candidate = base / name
            if candidate.exists() and candidate.is_file():
                return candidate

    return None


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
        Result containing starless image and optionally stars-only.
    """
    if starnet_path is None:
        starnet_path = find_starnet_binary()
    else:
        starnet_path = Path(starnet_path)

    if starnet_path is None or not starnet_path.exists():
        return StarNetResult(
            starless=image.copy(),
            success=False,
            message="StarNet binary not found. Install StarNet++ and ensure it's in PATH.",
        )

    original = image.copy()
    process_data = image
    lum: np.ndarray | None = None
    if image.ndim == 3:
        if image.shape[0] >= 3:
            lum = (
                0.2126 * image[0] + 0.7152 * image[1] + 0.0722 * image[2]
            ).astype(np.float32)
        else:
            lum = image[0].astype(np.float32)
        process_data = lum

    try:
        with tempfile.TemporaryDirectory(prefix="astraios_starnet_") as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.fits"
            output_path = tmpdir / "starless.fits"

            # Save input as FITS (mono luminance for color images)
            from astropy.io import fits as _fits
            header = _fits.Header({'NAXIS': 2, 'BITPIX': -32, 'SIMPLE': True})
            img_data = ImageData(data=process_data, header=header)
            save_fits(img_data, input_path)

            # Run StarNet subprocess
            cmd = [str(starnet_path), str(input_path), str(output_path)]
            log.info("Running StarNet: %s", " ".join(cmd))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode != 0:
                return StarNetResult(
                    starless=image.copy(),
                    success=False,
                    message=f"StarNet failed: {result.stderr}",
                )

            if not output_path.exists():
                return StarNetResult(
                    starless=image.copy(),
                    success=False,
                    message="StarNet produced no output file",
                )

            # Load result
            starless_img = load_image(str(output_path))
            starless_mono = starless_img.data
            if starless_mono.ndim == 3:
                starless_mono = starless_mono[0]

            if lum is not None:
                if starless_mono.shape != lum.shape:
                    return StarNetResult(
                        starless=original,
                        success=False,
                        message=(
                            f"StarNet output shape {starless_mono.shape} "
                            f"does not match input {lum.shape}"
                        ),
                    )
                star_residual = lum - starless_mono
                if original.ndim == 3:
                    starless = original - star_residual
                else:
                    starless = starless_mono
            else:
                starless = starless_mono

            stars_only = None
            if extract_stars:
                if starless.shape != original.shape:
                    return StarNetResult(
                        starless=original,
                        success=False,
                        message="Starless image shape does not match input",
                    )
                stars_only = np.clip(original - starless, 0, 1).astype(np.float32)

            return StarNetResult(
                starless=starless.astype(np.float32),
                stars_only=stars_only,
                success=True,
                message="StarNet processing complete",
            )

    except subprocess.TimeoutExpired:
        return StarNetResult(
            starless=image.copy(),
            success=False,
            message="StarNet timed out after 10 minutes",
        )
    except Exception as e:
        return StarNetResult(
            starless=image.copy(),
            success=False,
            message=f"StarNet error: {e}",
        )
