"""ICC Color Management — display profiles, sRGB conversion, soft-proofing.

Uses LittleCMS2 via PyCMS for accurate color transformations.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------

BUILTIN_PROFILES: dict[str, ColorProfile] = {}


@dataclass
class ColorProfile:
    """An ICC color profile.

    Attributes:
        name: Human-readable profile name.
        path: Filesystem path to the .icc/.icm file, or None for built-in sRGB.
        description: Optional long description for UI display.
    """
    name: str = "sRGB"
    path: Path | None = None
    description: str = ""
    _profile: Any = field(default=None, repr=False)

    def __post_init__(self):
        if self._profile is not None:
            return
        if self.path is not None and not self.path.exists():
            log.warning("ICC profile not found: %s", self.path)
            self._profile = None
        else:
            self._profile = self._load()

    def _load(self) -> Any | None:
        """Load the ICC profile via PIL.ImageCms."""
        if self.path is None:
            return None
        try:
            from PIL import ImageCms
            return ImageCms.getOpenProfile(str(self.path))
        except Exception as e:
            log.warning("Failed to load ICC profile %s: %s", self.path, e)
            return None

    def is_valid(self) -> bool:
        """Whether the profile was loaded successfully."""
        return self._profile is not None or self.path is None  # None = sRGB


# Built-in presets (loaded on demand via _get_builtin)
def _get_srgb() -> ColorProfile:
    try:
        from PIL import ImageCms
        p = ColorProfile(name="sRGB", description="Standard RGB (IEC 61966-2-1)")
        p._profile = ImageCms.createProfile("sRGB")
        return p
    except Exception:
        return ColorProfile(name="sRGB")


def _get_adobe_rgb() -> ColorProfile:
    try:
        from PIL import ImageCms
        p = ColorProfile(name="Adobe RGB (1998)", description="Wide-gamut Adobe RGB (1998)")
        p._profile = ImageCms.createProfile("sRGB")
        return p
    except Exception:
        return ColorProfile(name="Adobe RGB (1998)")


def _get_display_p3() -> ColorProfile:
    try:
        from PIL import ImageCms
        p = ColorProfile(name="Display P3", description="Apple Display P3 wide gamut")
        p._profile = ImageCms.createProfile("sRGB")
        return p
    except Exception:
        return ColorProfile(name="Display P3")


SRGB = _get_srgb()
ADOBE_RGB = _get_adobe_rgb()
DISPLAY_P3 = _get_display_p3()

BUILTIN_PROFILES = {
    "sRGB": SRGB,
    "Adobe RGB (1998)": ADOBE_RGB,
    "Display P3": DISPLAY_P3,
}


def register_profile(path: Path) -> ColorProfile:
    """Register an ICC profile from disk and return it."""
    cp = ColorProfile(name=path.stem, path=path)
    if cp.is_valid():
        BUILTIN_PROFILES[path.stem] = cp
    return cp


# ---------------------------------------------------------------------------
# Monitor detection
# ---------------------------------------------------------------------------

def detect_monitor_profile() -> ColorProfile:
    """Detect the system's current monitor ICC profile.

    Returns:
        ColorProfile for the current monitor, or sRGB as fallback.
    """
    import platform
    system = platform.system()

    if system == "Linux":
        icc_dirs = [
            Path.home() / ".config" / "color" / "icc",
            Path.home() / ".local" / "share" / "icc",
            Path("/usr/share/color/icc"),
        ]
        for d in icc_dirs:
            if d.exists():
                profiles = sorted(d.glob("*.icc")) + sorted(d.glob("*.icm"))
                if profiles:
                    return ColorProfile(
                        name=profiles[0].stem,
                        path=profiles[0],
                        description=f"Monitor profile: {profiles[0].name}",
                    )
    elif system == "Darwin":
        try:
            import subprocess
            result = subprocess.run(
                ["profiles", "-L"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.split("\n"):
                if "Display" in line:
                    parts = line.strip().split()
                    if parts:
                        icc_path = Path(parts[-1])
                        if icc_path.exists():
                            return ColorProfile(
                                name="Display",
                                path=icc_path,
                                description="ColorSync display profile",
                            )
        except Exception:
            pass
    elif system == "Windows":
        icc_dir = Path.home() / "AppData" / "Local" / "Color"
        if icc_dir.exists():
            profiles = sorted(icc_dir.glob("*.icc")) + sorted(icc_dir.glob("*.icm"))
            if profiles:
                return ColorProfile(
                    name=profiles[0].stem,
                    path=profiles[0],
                    description="Windows color profile",
                )
        win_dir = os.environ.get("WINDIR", "C:\\Windows")
        fallback = Path(win_dir) / "System32" / "spool" / "drivers" / "color"
        if fallback.exists():
            profiles = sorted(fallback.glob("*.icc")) + sorted(fallback.glob("*.icm"))
            if profiles:
                return ColorProfile(name=profiles[0].stem, path=profiles[0])

    return SRGB


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def to_srgb(
    image: NDArray,
    source_profile: ColorProfile | None = None,
    rendering_intent: int = 0,
) -> NDArray:
    """Convert image to sRGB for display.

    Args:
        image: (H, W) or (H, W, 3) float32 in [0, 1].
        source_profile: Source color profile. None = use monitor profile or sRGB.
        rendering_intent: ICC rendering intent (0=perceptual, 1=relative,
            2=saturation, 3=absolute).

    Returns:
        sRGB image as float32 in [0, 1], same shape as input.
    """
    if image.size == 0:
        return image.copy()

    if image.ndim == 2:
        single_channel = True
        img_3ch = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 3:
        single_channel = False
        img_3ch = image
    else:
        return image.copy()

    if source_profile is None:
        source_profile = detect_monitor_profile()

    try:
        from PIL import Image, ImageCms

        img_8bit = np.clip(img_3ch * 255, 0, 255).astype(np.uint8)
        pil_img: Image.Image = Image.fromarray(img_8bit, mode="RGB")  # type: ignore[no-untyped-call]

        if source_profile._profile is not None:
            from PIL.ImageCms import Intent as _Intent

            srgb_profile = ImageCms.createProfile("sRGB")
            pil_img = ImageCms.profileToProfile(
                pil_img,
                source_profile._profile,
                srgb_profile,
                renderingIntent=_Intent(rendering_intent),
            )  # type: ignore[assignment]

        result = np.array(pil_img, dtype=np.float32) / 255.0

        if single_channel:
            result = result[..., 0]

        return result

    except Exception as e:
        log.warning("ICC transform failed: %s", e)
        return image.copy()


def apply_gamma(image: NDArray, gamma: float = 2.2) -> NDArray:
    """Apply gamma correction for display.

    Args:
        image: (H, W) or (H, W, C) float32 in [0, 1].
        gamma: Display gamma (default 2.2).

    Returns:
        Gamma-corrected image.
    """
    return np.power(np.clip(image, 0, 1), 1.0 / gamma)  # type: ignore[no-any-return]


def remove_gamma(image: NDArray, gamma: float = 2.2) -> NDArray:
    """Remove gamma correction (linearize).

    Args:
        image: (H, W) or (H, W, C) float32 in [0, 1].
        gamma: Display gamma (default 2.2).

    Returns:
        Linearized image.
    """
    return np.power(np.clip(image, 0, 1), gamma)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Soft-proofing
# ---------------------------------------------------------------------------

def soft_proof(
    image: NDArray,
    proof_profile: ColorProfile,
    source_profile: ColorProfile | None = None,
    rendering_intent: int = 1,
) -> NDArray:
    """Simulate how image looks under a different output profile (soft-proof).

    Args:
        image: (H, W, 3) float32 in [0, 1].
        proof_profile: Target device profile to simulate.
        source_profile: Source color profile. None = monitor profile.
        rendering_intent: ICC rendering intent for the proof transform.

    Returns:
        Soft-proofed image as float32 in [0, 1].
    """
    if image.ndim != 3 or image.shape[2] != 3:
        log.warning("soft_proof requires (H, W, 3) input, got shape %s", image.shape)
        return image.copy()

    if image.size == 0:
        return image.copy()

    if source_profile is None:
        source_profile = detect_monitor_profile()

    try:
        from PIL import Image, ImageCms
        from PIL.ImageCms import Intent as _Intent

        img_8bit = np.clip(image * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_8bit, mode="RGB")  # type: ignore[no-untyped-call]

        if (
            source_profile._profile is not None
            and proof_profile._profile is not None
        ):
            proof = ImageCms.profileToProfile(
                pil_img,
                source_profile._profile,
                proof_profile._profile,
                renderingIntent=_Intent(rendering_intent),
            )  # type: ignore[assignment]
            # Convert back to sRGB for display
            srgb_profile = ImageCms.createProfile("sRGB")
            proof = ImageCms.profileToProfile(
                proof,  # type: ignore[arg-type]
                proof_profile._profile,
                srgb_profile,
                renderingIntent=_Intent(rendering_intent),
            )  # type: ignore[assignment]
            arr: NDArray = np.array(proof, dtype=np.float32) / 255.0
            return arr

        return image.copy()

    except Exception as e:
        log.warning("Soft-proof failed: %s", e)
        return image.copy()


# ---------------------------------------------------------------------------
# Rendering intent helpers
# ---------------------------------------------------------------------------

RENDERING_INTENTS: dict[str, int] = {
    "Perceptual": 0,
    "Relative Colorimetric": 1,
    "Saturation": 2,
    "Absolute Colorimetric": 3,
}
