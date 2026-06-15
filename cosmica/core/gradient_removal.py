"""Post-stretch gradient & flat (vignette) correction.

Removes the smooth large-scale background variation that survives into the
stretched image — light-pollution gradients and residual vignetting — by
modelling the SKY (everything that isn't the object or a star) and either
subtracting it (gradient mode) or dividing by it (flat/vignette mode).

Works on mono ``(H, W)`` or colour ``(C, H, W)`` float32 in ``[0, 1]``. Designed
to run *after* the stretch, where the gradient is actually visible, and to be
object-aware: pass the object mask so the subject's real signal is never fitted
as background.

Reuses the proven sample/clip/fit machinery in :mod:`cosmica.core.background`.
"""

from __future__ import annotations

import logging

import numpy as np

from cosmica.core.background import BackgroundParams, extract_background

log = logging.getLogger(__name__)

__all__ = ["GradientRemovalParams", "remove_gradient"]


class GradientRemovalParams:
    """Parameters for :func:`remove_gradient`.

    Attributes
    ----------
    mode : {"subtract", "divide"}
        ``subtract`` removes an additive gradient (light pollution / sky glow).
        ``divide`` corrects a multiplicative vignette (a synthetic flat) — it
        brightens darkened corners but can amplify their noise, so it's gentler.
    order : int
        Polynomial order of the background surface (kept low — gradients are
        smooth; high orders chase real structure).
    grid_size : int
        Sample grid density per axis.
    smoothing : float
        Gaussian smoothing of the model (fraction of image size).
    protect_floor : float
        Keep the corrected background near this level rather than crushing it to
        zero (avoids clipping faint shadow detail).
    """

    def __init__(
        self,
        mode: str = "subtract",
        order: int = 4,
        grid_size: int = 16,
        smoothing: float = 0.5,
        protect_floor: float = 0.06,
    ) -> None:
        self.mode = mode
        self.order = order
        self.grid_size = grid_size
        self.smoothing = smoothing
        self.protect_floor = protect_floor


def _channel_model(
    channel: np.ndarray,
    exclusion: np.ndarray | None,
    params: GradientRemovalParams,
) -> np.ndarray:
    """Fit the smooth sky background of one channel and return the model."""
    bg_params = BackgroundParams(
        grid_size=params.grid_size,
        polynomial_order=params.order,
        sigma_clip=2.5,
        smoothing=params.smoothing,
        object_aware=exclusion is not None,
        exclusion_mask=exclusion,
    )
    # extract_background returns (corrected, model); we want the model so we can
    # apply it ourselves (subtract or divide) with floor protection.
    _, model = extract_background(channel, bg_params)
    return model.astype(np.float32)


def remove_gradient(
    image: np.ndarray,
    object_mask: np.ndarray | None = None,
    params: GradientRemovalParams | None = None,
) -> np.ndarray:
    """Remove the residual background gradient / vignette from *image*.

    Parameters
    ----------
    image : ndarray
        Stretched image, ``(H, W)`` or ``(C, H, W)``, float32 in ``[0, 1]``.
    object_mask : ndarray, optional
        Soft ``[0, 1]`` mask of the subject (1 = object). Object pixels are
        excluded from the background fit so the subject isn't flattened away.
    params : GradientRemovalParams, optional

    Returns
    -------
    ndarray
        Corrected image, same shape and dtype, in ``[0, 1]``.
    """
    if params is None:
        params = GradientRemovalParams()

    is_color = image.ndim == 3
    h, w = (image.shape[-2], image.shape[-1])

    # Background sampling must AVOID the subject (object mask) and bright stars.
    # Stars are handled by the sample sigma-clip inside extract_background; the
    # object mask is converted to the exclusion convention (1 = exclude).
    exclusion = None
    if object_mask is not None and object_mask.shape == (h, w):
        exclusion = (object_mask > 0.4).astype(np.float32)
        if float(np.mean(exclusion)) > 0.9:
            # Object fills the frame — no sky to model a gradient from; bail.
            log.info("Gradient removal: object fills the frame, skipping")
            return image.astype(np.float32, copy=True)

    channels = [image] if not is_color else list(image)
    out = []
    for ch in channels:
        ch = ch.astype(np.float32)
        model = _channel_model(ch, exclusion, params)

        if params.mode == "divide":
            # Multiplicative flat: normalise the illumination to its median and
            # divide, so darkened corners are scaled back up. Clamp the gain so
            # very dark corners don't explode.
            ref = float(np.median(model[model > 0])) if np.any(model > 0) else 1.0
            ref = max(ref, 1e-3)
            gain = np.clip(ref / np.clip(model, 1e-3, None), 0.5, 2.5)
            corrected = ch * gain
        else:
            # Additive gradient: subtract the model, then add back a flat pedestal
            # so the background sits at the protect floor rather than at zero.
            corrected = ch - model + params.protect_floor

        out.append(np.clip(corrected, 0.0, 1.0).astype(np.float32))

    result = out[0] if not is_color else np.stack(out, axis=0)
    return result.astype(np.float32)
