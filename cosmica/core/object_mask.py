"""Soft elliptical object masks for object-aware processing.

Given catalog objects (centre + angular size) and the plate scale, build a
smooth ``[0, 1]`` mask that is 1.0 over the object and fades to 0 over the
surrounding sky. Unlike :func:`cosmica.core.background.create_object_exclusion_mask`
(a hard circle used only to *exclude* samples), this is a soft *elliptical*
mask meant to *steer* processing — enhance the subject, leave the sky alone.

The mask is deliberately conservative and self-contained: every consumer treats
it as optional and falls back to whole-image behaviour when it is ``None``, so
this module can be removed without touching the rest of the pipeline.
"""

from __future__ import annotations

import numpy as np

__all__ = ["build_object_mask"]


def build_object_mask(
    shape: tuple[int, int],
    objects: list[dict],
    plate_scale: float,
    feather: float = 0.45,
    min_radius_px: float = 8.0,
) -> np.ndarray | None:
    """Build a soft elliptical mask covering the catalog *objects*.

    Parameters
    ----------
    shape : (H, W)
        Image dimensions in pixels.
    objects : list of dict
        Each dict needs ``center_x``, ``center_y`` (pixel coords) and
        ``major_axis_arcmin`` / ``minor_axis_arcmin`` (angular extent). A
        ``rotation_deg`` key (sky position angle, optional) tilts the ellipse.
    plate_scale : float
        Arcseconds per pixel.
    feather : float
        Width of the gaussian skirt outside the ellipse, as a fraction of the
        object radius. Larger = softer transition into the sky.
    min_radius_px : float
        Floor on the semi-axes so a tiny/uncertain object still gets a usable
        mask rather than a single pixel.

    Returns
    -------
    ndarray or None
        Float32 ``(H, W)`` mask in ``[0, 1]``, or ``None`` if no usable object
        is given (so callers can cleanly skip object-aware behaviour).
    """
    if not objects or plate_scale <= 0:
        return None

    h, w = shape
    yy, xx = np.ogrid[0:h, 0:w]
    mask = np.zeros((h, w), dtype=np.float32)
    used = False

    for obj in objects:
        try:
            cx = float(obj["center_x"])
            cy = float(obj["center_y"])
            major = float(obj.get("major_axis_arcmin", 0.0))
            minor = float(obj.get("minor_axis_arcmin", major))
        except (KeyError, TypeError, ValueError):
            continue
        if major <= 0:
            continue

        a = max(major * 60.0 / plate_scale / 2.0, min_radius_px)  # semi-major (px)
        b = max(minor * 60.0 / plate_scale / 2.0, min_radius_px)  # semi-minor (px)

        dx = xx - cx
        dy = yy - cy
        rot = float(obj.get("rotation_deg", 0.0))
        if rot:
            th = np.radians(rot)
            ct, st = np.cos(th), np.sin(th)
            dxr = dx * ct + dy * st
            dyr = -dx * st + dy * ct
            dx, dy = dxr, dyr

        # Normalised elliptical radius: 1.0 exactly on the ellipse boundary.
        t = np.sqrt((dx / a) ** 2 + (dy / b) ** 2)
        # 1.0 inside; gaussian fade to 0 over `feather` of the radius outside.
        m = np.where(t <= 1.0, 1.0, np.exp(-(((t - 1.0) / max(feather, 1e-3)) ** 2)))
        mask = np.maximum(mask, m.astype(np.float32))
        used = True

    if not used:
        return None
    return np.clip(mask, 0.0, 1.0).astype(np.float32)
