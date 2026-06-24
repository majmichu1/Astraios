"""Mask System — foundation for selective processing in Astraios.

Every processing function can accept an optional Mask. The mask is applied as:
    result = processed * mask + original * (1 - mask)

Masks are float32 arrays in [0, 1] with shape (H, W).
A value of 1.0 means fully processed, 0.0 means fully protected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np

log = logging.getLogger(__name__)


class MaskType(Enum):
    """Types of masks available."""
    LUMINANCE = auto()
    RANGE = auto()
    STAR = auto()
    MANUAL = auto()
    COMBINED = auto()


@dataclass
class Mask:
    """A single-channel mask for selective processing.

    Attributes
    ----------
    data : ndarray
        Float32 array of shape (H, W) with values in [0, 1].
        1.0 = fully selected (will be processed).
        0.0 = fully protected (will not be processed).
    name : str
        User-visible name for the mask.
    mask_type : MaskType
        How the mask was created.
    """

    data: np.ndarray
    name: str = "Untitled Mask"
    mask_type: MaskType = MaskType.MANUAL

    def __post_init__(self):
        self.data = np.clip(self.data.astype(np.float32), 0.0, 1.0)

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def to_display(self) -> np.ndarray:
        """Convert mask to uint8 RGB array (H, W, 3) for display."""
        gray = np.clip(self.data * 255, 0, 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)


def apply_mask(
    original: np.ndarray,
    processed: np.ndarray,
    mask: Mask | None,
) -> np.ndarray:
    """Blend processed result with original using the mask.

    Parameters
    ----------
    original : ndarray
        Original image data. Shape (H, W) or (C, H, W).
    processed : ndarray
        Processed image data. Same shape as original.
    mask : Mask or None
        If None, returns processed unchanged.

    Returns
    -------
    ndarray
        Blended result: processed * mask + original * (1 - mask).
    """
    if mask is None:
        return processed

    m = mask.data
    # Guard against a wrong-resolution mask (e.g. built before a crop/resize):
    # without this, numpy either raises a cryptic broadcast error or — worse —
    # silently mis-blends when the shapes happen to broadcast (e.g. a (H, 1) or
    # (1, W) mask). A clear error beats corrupted output.
    if m.shape != original.shape[-2:]:
        raise ValueError(
            f"Mask spatial shape {m.shape} does not match image "
            f"{original.shape[-2:]}"
        )

    # All-zero mask: every pixel is fully protected — skip the blending entirely
    if not m.any():
        return original

    if original.ndim == 3:
        # Broadcast (H, W) mask to (C, H, W)
        m = m[np.newaxis, :, :]

    return processed * m + original * (1.0 - m)


# ---------- Mask creation functions ----------


def create_luminance_mask(
    image: np.ndarray,
    low: float = 0.0,
    high: float = 1.0,
    name: str = "Luminance Mask",
) -> Mask:
    """Create a mask based on image luminance.

    Pixels with luminance between `low` and `high` get mask=1.
    Smooth falloff at boundaries using a cosine taper.

    Parameters
    ----------
    image : ndarray
        Image data, shape (H, W) or (C, H, W), values in [0, 1].
    low : float
        Lower luminance threshold (0-1).
    high : float
        Upper luminance threshold (0-1).
    name : str
        Name for the resulting mask.

    Returns
    -------
    Mask
        Luminance-based mask.
    """
    luminance = _get_luminance(image)

    # Smooth transitions using linear ramp with small edge width
    edge = max(0.01, (high - low) * 0.1)

    mask_data = np.ones_like(luminance)

    # Low edge: ramp from 0 to 1 (skip if low is at absolute minimum)
    if low > 0.0:
        below = luminance < low
        ramp_low = luminance < (low + edge)
        mask_data[below] = 0.0
        transition = ramp_low & ~below
        if transition.any():
            mask_data[transition] = (luminance[transition] - low) / edge

    # High edge: ramp from 1 to 0 (skip if high is at absolute maximum)
    if high < 1.0:
        above = luminance > high
        ramp_high = luminance > (high - edge)
        mask_data[above] = 0.0
        transition = ramp_high & ~above
        if transition.any():
            mask_data[transition] = (high - luminance[transition]) / edge

    return Mask(data=mask_data, name=name, mask_type=MaskType.LUMINANCE)


def create_range_mask(
    image: np.ndarray,
    channel: int = -1,
    low: float = 0.0,
    high: float = 1.0,
    name: str = "Range Mask",
) -> Mask:
    """Create a mask from a value range on a specific channel.

    Parameters
    ----------
    image : ndarray
        Image data, shape (H, W) or (C, H, W).
    channel : int
        Channel index (0=R, 1=G, 2=B for color). -1 = luminance.
    low : float
        Lower bound (0-1).
    high : float
        Upper bound (0-1).
    name : str
        Name for the resulting mask.

    Returns
    -------
    Mask
        Range-based mask.
    """
    if channel < 0 or image.ndim == 2:
        data = _get_luminance(image)
    else:
        if image.ndim == 3 and channel < image.shape[0]:
            data = image[channel]
        else:
            data = _get_luminance(image)

    mask_data = ((data >= low) & (data <= high)).astype(np.float32)
    return Mask(data=mask_data, name=name, mask_type=MaskType.RANGE)


# ---------- Mask manipulation functions ----------


def invert_mask(mask: Mask, name: str | None = None) -> Mask:
    """Invert a mask (swap selected and protected regions).

    Parameters
    ----------
    mask : Mask
        Input mask.
    name : str, optional
        Name for the inverted mask. Defaults to "Inverted <original>".

    Returns
    -------
    Mask
        Inverted mask where new_value = 1.0 - old_value.
    """
    inv_name = name or f"Inverted {mask.name}"
    return Mask(data=1.0 - mask.data, name=inv_name, mask_type=mask.mask_type)


def combine_masks(
    masks: list[Mask],
    mode: str = "multiply",
    name: str = "Combined Mask",
) -> Mask:
    """Combine multiple masks into one.

    Parameters
    ----------
    masks : list[Mask]
        Masks to combine. Must all have the same shape.
    mode : str
        Combination mode: "multiply", "add", "min", "max", "screen".
    name : str
        Name for the resulting mask.

    Returns
    -------
    Mask
        Combined mask.
    """
    if not masks:
        raise ValueError("No masks to combine")
    if len(masks) == 1:
        return Mask(data=masks[0].data.copy(), name=name, mask_type=MaskType.COMBINED)

    result = masks[0].data.copy()
    for m in masks[1:]:
        if mode == "multiply":
            result = result * m.data
        elif mode == "add":
            result = result + m.data
        elif mode == "min":
            result = np.minimum(result, m.data)
        elif mode == "max":
            result = np.maximum(result, m.data)
        elif mode == "screen":
            result = 1.0 - (1.0 - result) * (1.0 - m.data)
        else:
            raise ValueError(f"Unknown combine mode: {mode}")

    return Mask(data=result, name=name, mask_type=MaskType.COMBINED)


def blur_mask(
    mask: Mask,
    radius: float = 5.0,
    name: str | None = None,
) -> Mask:
    """Apply Gaussian blur to soften mask edges.

    Parameters
    ----------
    mask : Mask
        Input mask.
    radius : float
        Blur radius in pixels (sigma of Gaussian).
    name : str, optional
        Name for the blurred mask.

    Returns
    -------
    Mask
        Blurred mask.
    """
    if radius <= 0:
        return Mask(data=mask.data.copy(), name=name or mask.name, mask_type=mask.mask_type)

    # Kernel size must be odd
    ksize = int(np.ceil(radius * 3)) * 2 + 1
    blurred = cv2.GaussianBlur(mask.data, (ksize, ksize), radius)
    blur_name = name or f"Blurred {mask.name}"
    return Mask(data=blurred, name=blur_name, mask_type=mask.mask_type)


def binarize_mask(
    mask: Mask,
    threshold: float = 0.5,
    name: str | None = None,
) -> Mask:
    """Convert a soft mask to a hard binary mask.

    Parameters
    ----------
    mask : Mask
        Input mask.
    threshold : float
        Values above this become 1.0, below become 0.0.
    name : str, optional
        Name for the binary mask.

    Returns
    -------
    Mask
        Binary mask.
    """
    binary = (mask.data >= threshold).astype(np.float32)
    bin_name = name or f"Binary {mask.name}"
    return Mask(data=binary, name=bin_name, mask_type=mask.mask_type)


def grow_mask(
    mask: Mask,
    pixels: int = 3,
    name: str | None = None,
) -> Mask:
    """Dilate (grow) the mask by a given number of pixels.

    Parameters
    ----------
    mask : Mask
        Input mask.
    pixels : int
        Number of pixels to grow.
    name : str, optional
        Name for the grown mask.

    Returns
    -------
    Mask
        Dilated mask.
    """
    if pixels <= 0:
        return Mask(data=mask.data.copy(), name=name or mask.name, mask_type=mask.mask_type)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
    dilated = cv2.dilate(mask.data, kernel)
    grow_name = name or f"Grown {mask.name}"
    return Mask(data=dilated, name=grow_name, mask_type=mask.mask_type)


def shrink_mask(
    mask: Mask,
    pixels: int = 3,
    name: str | None = None,
) -> Mask:
    """Erode (shrink) the mask by a given number of pixels.

    Parameters
    ----------
    mask : Mask
        Input mask.
    pixels : int
        Number of pixels to shrink.
    name : str, optional
        Name for the shrunk mask.

    Returns
    -------
    Mask
        Eroded mask.
    """
    if pixels <= 0:
        return Mask(data=mask.data.copy(), name=name or mask.name, mask_type=mask.mask_type)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
    eroded = cv2.erode(mask.data, kernel)
    shrink_name = name or f"Shrunk {mask.name}"
    return Mask(data=eroded, name=shrink_name, mask_type=mask.mask_type)


# ---------- Internal helpers ----------


def _get_luminance(image: np.ndarray) -> np.ndarray:
    """Extract luminance from an image.

    For mono (H, W), returns the image itself.
    For color (C, H, W), computes weighted luminance.
    """
    if image.ndim == 2:
        return image
    if image.shape[0] >= 3:
        return 0.2126 * image[0] + 0.7152 * image[1] + 0.0722 * image[2]
    return image[0]
