"""Built-in star removal — zero-setup morphological approach.

Uses median-background subtraction + star detection + inpainting.
No external model download needed, works immediately on any image.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


def remove_stars_builtin(
    image: NDArray,
    threshold: float = 0.5,
) -> NDArray:
    """Remove stars using morphological detection + median inpainting.

    Args:
        image: (H, W) or (C, H, W) float32/float64 in [0, 1].
        threshold: 0-1 slider value (lower = remove more / more aggressive).

    Returns:
        Starless image, same shape and dtype.
    """
    return _remove_stars_morph(image, threshold)


def _remove_stars_morph(
    image: NDArray,
    threshold: float,
) -> NDArray:
    """Morphological star removal via median background + mask."""
    # float32, not float64: morphology + inpainting of a [0,1] image needs no
    # extra precision, and float64 doubles memory (1.75GB for a 73MP colour
    # frame in one allocation — enough to OOM a RAM-tight machine here). asarray
    # is a no-op when the input is already float32 (it is, in the pipeline).
    img = np.asarray(image, dtype=np.float32)
    is_color = img.ndim == 3

    if is_color:
        if img.shape[0] >= 4:
            lum = 0.2126 * img[0] + 0.7152 * (img[1] + img[2]) * 0.5 + 0.0722 * img[3]
        elif img.shape[0] >= 3:
            lum = 0.2126 * img[0] + 0.7152 * img[1] + 0.0722 * img[2]
        else:
            lum = img[0].copy()
    else:
        lum = img.copy()

    orig_h, orig_w = lum.shape

    # ── 1. Median-filtered background ────────────────────────────────
    # Kernel size adapts to image dimensions (stars are ~1-5% of width)
    ksize = max(15, min(orig_h, orig_w) // 30)
    if ksize % 2 == 0:
        ksize += 1
    if ksize > 199:
        ksize = 199  # OpenCV medianBlur limit

    # OpenCV medianBlur only supports 8-bit or float32
    lum_u8 = (np.clip(lum, 0, 1) * 255).astype(np.uint8)
    bg_u8 = cv2.medianBlur(lum_u8, ksize)
    bg = bg_u8.astype(np.float32) / 255.0

    # ── 2. Residuals (star signal) ──────────────────────────────────
    resid = lum - bg
    diff = np.clip(resid, 0, None)

    # Noise estimate from the SYMMETRIC residual, not the clipped one.
    # Clipping negatives to zero collapses the median/MAD to ~0 on a smooth
    # background, which drives the threshold to nil and masks the ENTIRE frame —
    # the inpainter then replaces the whole image (a saturated core's value
    # floods everything, blowing the background to ~1.0). Using the unclipped
    # residual keeps the noise estimate honest, and an absolute floor guards the
    # clean-background case.
    mad = np.median(np.abs(resid - np.median(resid)))
    noise_est = max(mad * 1.4826, 1e-4)

    # threshold maps 0..1 → sigma 12..1 (lower slider = more aggressive)
    sigma = 1.0 + (1.0 - threshold) * 12.0
    star_thresh = sigma * noise_est
    binary = (diff > star_thresh).astype(np.uint8) * 255

    # Safety: stars occupy a small fraction of any real frame. If the mask
    # explodes (degenerate/smooth data), raise the threshold until it is sane
    # rather than inpainting the whole image. If it still won't converge, the
    # detection is unreliable — return the image untouched instead of wrecking it.
    star_frac = float(np.mean(binary > 0))
    bump = 0
    while star_frac > 0.10 and bump < 6:
        star_thresh *= 1.8
        binary = (diff > star_thresh).astype(np.uint8) * 255
        star_frac = float(np.mean(binary > 0))
        bump += 1
    if star_frac > 0.10:
        log.warning(
            "Star mask covers %.0f%% of frame — detection unreliable, "
            "skipping star removal", star_frac * 100,
        )
        return image.astype(image.dtype, copy=True)

    # ── 2b. Keep only blobs that are STAR-SIZED ─────────────────────
    # Too SMALL = noise: background grain throws tens of thousands of 1–2px
    # specks above threshold; left in, dilation merges them into a mask covering
    # most of the frame, and the inpainter floods the dark sky with the bright
    # nebula's value. Too LARGE = nebula core (M42's Trapezium region): inpainting
    # it leaves a dark hole. A real star is a small compact blob in between.
    from scipy import ndimage

    labels, n_blob = ndimage.label(binary > 0)
    if n_blob > 0:
        min_star_area = 4.0  # px — below this is background grain, not a star
        max_star_diam = max(20.0, min(orig_h, orig_w) * 0.04)
        max_area = np.pi * (max_star_diam / 2.0) ** 2
        areas = ndimage.sum(np.ones_like(labels, dtype=np.float64), labels,
                            index=np.arange(1, n_blob + 1))
        drop = np.nonzero((areas < min_star_area) | (areas > max_area))[0] + 1
        if drop.size:
            binary[np.isin(labels, drop)] = 0
            log.debug("Star removal: dropped %d/%d non-star blobs (noise or nebula)",
                      int(drop.size), n_blob)

    # ── 3. Dilate mask to cover halos — radius from the ACTUAL star size ──
    # Keying the halo to the image size (the old min(H,W)//200, x2 iterations)
    # over-dilates a dense field of small stars: a ~40px halo on a 3-5px star
    # merges neighbours, over-removes, and reads as bloated/smeared after
    # inpainting. Scale the halo to the median detected star instead, so small
    # stars get a small halo and large stars still get theirs covered.
    kept_labels, n_kept = ndimage.label(binary > 0)
    if n_kept > 0:
        kept_areas = ndimage.sum(
            np.ones_like(kept_labels, dtype=np.float64), kept_labels,
            index=np.arange(1, n_kept + 1),
        )
        median_diam = 2.0 * np.sqrt(max(float(np.median(kept_areas)), 1.0) / np.pi)
        radius = int(np.clip(round(median_diam * 0.7), 2, max(4, min(orig_h, orig_w) // 100)))
    else:
        radius = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Final runaway backstop AFTER dilation: the min-area filter above already
    # removes the noise specks that used to dilate into a frame-covering mask, so
    # this only fires on a genuine catastrophe (the inpainter would then fill the
    # masked majority with the bright object's value and flood the sky). A dense
    # but legitimate star field can reach ~30-40% after dilation, so keep the
    # threshold high enough not to disable real star removal.
    dilated_frac = float(np.mean(binary > 0))
    if dilated_frac > 0.45:
        log.warning(
            "Star mask covers %.0f%% of frame after dilation — unreliable, "
            "skipping star removal", dilated_frac * 100,
        )
        return image.astype(image.dtype, copy=True)

    mask = binary > 0

    # ── 4. Reconstruct the sky under the stars by diffusion inpainting ──
    # Replacing star pixels with the median background leaves residual halos
    # and dark holes (the median kernel partly contains the star itself).
    # Diffusion inpainting fills each masked region purely from the
    # surrounding nebula/sky, in float precision — a much cleaner starless.
    if is_color:
        result = img.copy()
        for c in range(min(3, img.shape[0])):
            result[c] = _diffuse_inpaint(img[c], mask)
        if img.shape[0] > 3:
            for c in range(3, img.shape[0]):
                result[c] = _diffuse_inpaint(img[c], mask)
    else:
        result = _diffuse_inpaint(lum, mask)

    # ── 5. Feather edges so star boundaries blend smoothly ──────────────
    if np.any(mask):
        feather = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), sigmaX=radius * 0.8) / 255.0
        # Keep the mask INTERIOR fully inpainted (blend=1): the Gaussian feather
        # alone dips below 1 at the centre of a small star, leaving a residual
        # bright core. Use it only to add a soft ring OUTSIDE the mask edge.
        blend = np.clip(np.maximum(mask.astype(np.float32), feather), 0, 1)
        if is_color:
            for c in range(img.shape[0]):
                result[c] = img[c] * (1.0 - blend) + result[c] * blend
        else:
            result = img * (1.0 - blend) + result * blend

    return np.clip(result, 0, 1).astype(image.dtype)


def _diffuse_inpaint(
    channel: NDArray,
    mask: NDArray,
    iterations: int = 40,
    sigma: float = 3.0,
) -> NDArray:
    """Fill ``mask`` (True = remove) by diffusing surrounding values inward.

    Repeatedly blurs the image and re-imposes the known (unmasked) pixels, so
    the masked star regions converge to a smooth interpolation of the
    surrounding sky/nebula. Float precision, no 8-bit quantisation.
    """
    if not np.any(mask):
        return channel.astype(np.float32, copy=True)

    known = ~mask
    # float32 (not float64): halves this buffer and lets cv2.GaussianBlur run on
    # it directly instead of re-casting every iteration.
    result = channel.astype(np.float32, copy=True)
    # Seed holes with the local mean so diffusion starts from a sane value.
    if np.any(known):
        result[mask] = np.float32(np.median(channel[known]))
    for _ in range(iterations):
        blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=sigma)
        result[mask] = blurred[mask]
        result[known] = channel[known]
    return result
