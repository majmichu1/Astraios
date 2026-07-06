"""Continuum Subtraction — isolate narrowband emission by removing a scaled
continuum/broadband frame.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro

SASpro's ``continuum_subtract.py`` builds a synthetic RGB triple
``(narrowband, continuum, continuum)`` and runs it through a general-purpose
star-based *white balance* routine (designed for real R/G/B starlight color)
purely so its by-product — a per-channel linear gain that equalizes stellar
flux across channels — can be reused to match the narrowband channel's star
brightness to the continuum channel's star brightness before the weighted
subtraction ``result = clip(narrowband - Q * (continuum - median(continuum)),
0, 1)``.

Because the reused routine's "green" and "blue" channels are identical in
this application, its blackbody-locus tilt math degenerates to a fixed
(non-data-dependent) pair of constants and the whole detour collapses to one
thing that actually matters: *find a robust multiplicative gain so that the
median stellar flux of the continuum (anchored at its own background pivot)
matches the median stellar flux of the narrowband channel*. That net effect
is what is ported here directly (see :func:`_estimate_star_gain`), using
Astraios's own star detector instead of SASpro's ``sep``/``cv2`` overlay
pipeline (which exists only to draw a diagnostic UI overlay — out of scope
for this headless core port).

Ported settings/steps:
    - ``_compute_bg_pedestal`` / ``_apply_pedestal`` -> background pedestal
      matching via a random-restart darkest-box search (kept faithful to the
      original iterative hill-climb).
    - ``_normalize_red_to_green`` -> MAD/median gain-and-offset match of the
      narrowband channel onto the continuum channel (kept faithful, exact
      formula).
    - ``apply_star_based_white_balance`` -> replaced by
      :func:`_estimate_star_gain`, which reproduces its net numerical effect
      (matching star-core flux between channels) without the degenerate
      RGB-locus tilt detour.
    - ``_linear_subtract`` -> :func:`subtract_continuum`'s final weighted
      subtraction, byte-for-byte the same formula.

GPU decision: this module is pure light-weight statistics (medians, MAD,
small per-star aperture sampling, a handful of box searches) on at most a
few hundred sample points — nothing here benefits from GPU dispatch, and
using exact NumPy/Astropy statistics keeps the port numerically faithful to
SASpro. No ``device_manager`` usage, consistent with other light-math core
modules (e.g. ``banding.py``, ``narrowband_normalization.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from astraios.core.masks import Mask, apply_mask
from astraios.core.star_detection import detect_stars

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class ContinuumSubtractParams:
    """Parameters for narrowband continuum subtraction.

    Attributes:
        scaling_method: "star_based" runs the full SASpro-equivalent
            pipeline (background pedestal + MAD gain match + star-based
            gain refinement) before subtracting. "manual" skips all gain
            estimation and subtracts using the raw channels and
            `scale_factor` only (background pedestal still optional).
        scale_factor: SASpro's "Q factor" — the final subtraction strength,
            `result = narrowband - scale_factor * (continuum - median)`.
            Typical range 0.1-2.0; SASpro's UI default is 0.80.
        background_pedestal: Match narrowband/continuum backgrounds with a
            random-restart darkest-box search before scaling (SASpro
            `_compute_bg_pedestal`/`_apply_pedestal`).
        pedestal_num_boxes: Number of randomly placed sample boxes searched.
        pedestal_box_size: Side length (px) of each sample box.
        pedestal_iterations: Hill-climb iterations refining each box toward
            the locally darkest neighbor.
        normalize_gain: Robust MAD/median gain-and-offset match of the
            narrowband channel onto the continuum channel (SASpro
            `_normalize_red_to_green`), applied before star-based refinement.
        star_based_gain: Refine the continuum gain using detected stars so
            stellar flux matches between narrowband and continuum (SASpro
            `apply_star_based_white_balance`, net effect only).
        star_threshold: Star detection significance threshold, in
            MAD-sigma above background (SASpro's WB sigma threshold).
        max_stars: Cap on the number of stars used for the gain fit.
        star_sample_radius: Radius (px) of the circular aperture sampled at
            each detected star centroid (SASpro uses radius=3).
        min_stars_for_gain: Minimum stars required to trust the star-based
            gain estimate; falls back to the MAD-normalized gain otherwise.
        gain_clip: (min, max) bounds clamping the estimated star-based gain,
            guarding against div-by-near-zero on pathological inputs.
        clip_output: Clip the final result to [clip_min, clip_max].
        clip_min: Output floor (SASpro always uses 0.0).
        clip_max: Output ceiling (SASpro always uses 1.0).
    """

    scaling_method: str = "star_based"  # "star_based" | "manual"
    scale_factor: float = 0.80
    background_pedestal: bool = True
    pedestal_num_boxes: int = 200
    pedestal_box_size: int = 25
    pedestal_iterations: int = 25
    normalize_gain: bool = True
    star_based_gain: bool = True
    star_threshold: float = 5.0
    max_stars: int = 500
    star_sample_radius: int = 3
    min_stars_for_gain: int = 10
    gain_clip: tuple[float, float] = (0.05, 20.0)
    clip_output: bool = True
    clip_min: float = 0.0
    clip_max: float = 1.0


def _compute_pedestal(
    nb: np.ndarray,
    cont: np.ndarray,
    num_boxes: int,
    box_size: int,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Random-restart darkest-box search, adapted from SASpro's
    ``_compute_bg_pedestal``.

    The original operates on a synthetic 3-channel (R=nb, G=cont, B=cont)
    array; here the same hill-climbing search is done directly on the
    (nb, cont) pair, and the continuum pedestal is synced to the
    narrowband's darkest-patch level exactly like SASpro syncs G/B to R.

    Returns
    -------
    (pedestal_nb, pedestal_cont) : tuple[float, float]
    """
    height, width = nb.shape
    box_size = min(box_size, max(1, height - 1), max(1, width - 1))
    if box_size < 1 or height <= box_size or width <= box_size:
        return 0.0, 0.0

    boxes = [
        (int(rng.integers(0, height - box_size)), int(rng.integers(0, width - box_size)))
        for _ in range(num_boxes)
    ]
    best = np.full(num_boxes, np.inf, dtype=np.float64)

    for _ in range(iterations):
        for i, (y, x) in enumerate(boxes):
            if y + box_size > height or x + box_size > width:
                continue
            patch = np.concatenate(
                [
                    nb[y : y + box_size, x : x + box_size].ravel(),
                    cont[y : y + box_size, x : x + box_size].ravel(),
                ]
            )
            med = float(np.median(patch)) if patch.size else np.inf
            best[i] = min(best[i], med)

            sv: list[float] = []
            neighbor_offsets: list[tuple[int, int]] = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy * box_size, x + dx * box_size
                    if 0 <= yy < height - box_size and 0 <= xx < width - box_size:
                        p2 = np.concatenate(
                            [
                                nb[yy : yy + box_size, xx : xx + box_size].ravel(),
                                cont[yy : yy + box_size, xx : xx + box_size].ravel(),
                            ]
                        )
                        if p2.size:
                            sv.append(float(np.median(p2)))
                            neighbor_offsets.append((dy, dx))
            if sv:
                k = int(np.argmin(sv))
                ndy, ndx = neighbor_offsets[k]
                boxes[i] = (y + ndy * box_size, x + ndx * box_size)

    darkest = np.inf
    ref_box = None
    for y, x in boxes:
        if y + box_size <= height and x + box_size <= width:
            patch = np.concatenate(
                [
                    nb[y : y + box_size, x : x + box_size].ravel(),
                    cont[y : y + box_size, x : x + box_size].ravel(),
                ]
            )
            med = float(np.median(patch)) if patch.size else np.inf
            if med < darkest:
                darkest, ref_box = med, (y, x)

    if ref_box is None:
        return 0.0, 0.0

    y, x = ref_box
    ref_nb = float(np.median(nb[y : y + box_size, x : x + box_size]))
    ref_cont = float(np.median(cont[y : y + box_size, x : x + box_size]))

    med_nb = float(np.median(nb))
    med_cont = float(np.median(cont))

    ped_nb = max(0.0, med_nb - ref_nb)
    ped_cont = max(0.0, med_cont - ref_cont)

    # Sync the continuum's darkest-patch level to the narrowband's, exactly
    # like SASpro syncs G/B pedestal to R's reference level.
    if ref_cont < ref_nb:
        ped_cont += ref_nb - ref_cont

    return ped_nb, ped_cont


def _normalize_gain(nb: np.ndarray, cont: np.ndarray) -> np.ndarray:
    """Robust MAD/median gain-and-offset match of `nb` onto `cont`.

    Exact port of SASpro's ``_normalize_red_to_green``: matches the
    narrowband channel's noise amplitude (mean absolute deviation) and
    median to the continuum channel's, via a single affine transform.
    """
    mad_nb = float(np.mean(np.abs(nb - np.mean(nb))))
    mad_cont = float(np.mean(np.abs(cont - np.mean(cont))))
    med_nb = float(np.median(nb))
    med_cont = float(np.median(cont))

    gain = mad_cont / max(mad_nb, 1e-9)
    offset = -gain * med_nb + med_cont
    return np.clip(nb * gain + offset, 0.0, 1.0).astype(np.float32, copy=False)


def _sample_star_medians(channel: np.ndarray, positions: np.ndarray, radius: int) -> np.ndarray:
    """Median pixel value in a small circular aperture around each star.

    Adapted from SASpro's ``_sample_star_circle_medians``.
    """
    h, w = channel.shape
    rr2 = radius * radius
    samples = []
    for cx_f, cy_f in positions:
        cx, cy = int(round(float(cx_f))), int(round(float(cy_f)))
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        aperture = ((xx - cx) ** 2 + (yy - cy) ** 2) <= rr2
        vals = channel[y0:y1, x0:x1][aperture]
        if vals.size == 0:
            continue
        samples.append(float(np.median(vals)))
    return np.asarray(samples, dtype=np.float64)


def _estimate_star_gain(
    nb: np.ndarray,
    cont: np.ndarray,
    params: ContinuumSubtractParams,
) -> float:
    """Robust star-based gain so continuum star flux matches narrowband star flux.

    This reproduces the net numerical effect of SASpro's
    ``apply_star_based_white_balance`` for the (nb, continuum) pair: after
    scaling, ``(median_star_cont - pivot_cont) * gain ≈ median_star_nb -
    pivot_nb``, anchoring each channel at its own background pivot (median)
    the same way SASpro anchors at a background-rectangle pivot. Falls back
    to gain=1.0 when too few stars are detected.
    """
    detection_image = ((nb.astype(np.float64) + cont.astype(np.float64)) / 2.0).astype(np.float32)
    stars = detect_stars(
        detection_image,
        max_stars=params.max_stars,
        sigma_threshold=params.star_threshold,
    )
    if len(stars) < params.min_stars_for_gain:
        log.debug("Too few stars (%d) for star-based gain; falling back to gain=1.0", len(stars))
        return 1.0

    positions = stars.positions
    nb_star = _sample_star_medians(nb, positions, params.star_sample_radius)
    cont_star = _sample_star_medians(cont, positions, params.star_sample_radius)
    n = min(len(nb_star), len(cont_star))
    if n < params.min_stars_for_gain:
        return 1.0
    nb_star, cont_star = nb_star[:n], cont_star[:n]

    pivot_nb = float(np.median(nb))
    pivot_cont = float(np.median(cont))

    med_nb_star = float(np.median(nb_star))
    med_cont_star = float(np.median(cont_star))

    denom = med_cont_star - pivot_cont
    if abs(denom) < 1e-9:
        return 1.0

    gain = (med_nb_star - pivot_nb) / denom
    lo, hi = params.gain_clip
    return float(np.clip(gain, lo, hi))


def _subtract_channel(
    nb: np.ndarray,
    cont: np.ndarray,
    params: ContinuumSubtractParams,
    progress: ProgressCallback,
    seed: int,
) -> np.ndarray:
    """Run the full pedestal + gain-match + star-based-gain + subtract
    pipeline on a single (H, W) narrowband/continuum pair."""
    nb = np.asarray(nb, dtype=np.float32)
    cont = np.asarray(cont, dtype=np.float32)

    if params.background_pedestal:
        progress(0.1, "Background pedestal...")
        rng = np.random.default_rng(seed)
        ped_nb, ped_cont = _compute_pedestal(
            nb,
            cont,
            params.pedestal_num_boxes,
            params.pedestal_box_size,
            params.pedestal_iterations,
            rng,
        )
        nb = np.clip(nb + ped_nb, 0.0, 1.0).astype(np.float32, copy=False)
        cont = np.clip(cont + ped_cont, 0.0, 1.0).astype(np.float32, copy=False)

    if params.scaling_method == "star_based":
        if params.normalize_gain:
            progress(0.35, "Normalizing narrowband to continuum...")
            nb = _normalize_gain(nb, cont)

        cont_adj = cont
        if params.star_based_gain:
            progress(0.6, "Star-based gain refinement...")
            gain = _estimate_star_gain(nb, cont, params)
            pivot_cont = float(np.median(cont))
            cont_adj = (cont - pivot_cont) * gain + pivot_cont
            cont_adj = np.clip(cont_adj, 0.0, 1.0).astype(np.float32, copy=False)
    else:
        cont_adj = cont

    progress(0.85, "Subtracting continuum...")
    green_median = float(np.median(cont_adj))
    result = nb - params.scale_factor * (cont_adj - green_median)

    if params.clip_output:
        result = np.clip(result, params.clip_min, params.clip_max)

    return result.astype(np.float32, copy=False)


def subtract_continuum(
    narrowband: np.ndarray,
    continuum: np.ndarray,
    params: ContinuumSubtractParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Isolate narrowband emission by subtracting a scaled continuum frame.

    Parameters
    ----------
    narrowband : ndarray
        Narrowband (e.g. Ha) image, shape (H, W) mono or (C, H, W) color,
        float32 in [0, 1].
    continuum : ndarray
        Broadband/continuum reference image. Either the same shape as
        `narrowband`, or (H, W) mono to be broadcast across all channels
        of a color `narrowband`.
    params : ContinuumSubtractParams, optional
        Subtraction parameters. Defaults are used if None.
    mask : Mask, optional
        Selective processing mask.
    progress : callable, optional
        Progress callback `(fraction, message)`.

    Returns
    -------
    ndarray
        Continuum-subtracted image, same shape as `narrowband`.

    Raises
    ------
    ValueError
        If shapes are incompatible.
    """
    if params is None:
        params = ContinuumSubtractParams()
    if params.scaling_method not in ("star_based", "manual"):
        raise ValueError(
            f"scaling_method must be 'star_based' or 'manual', got {params.scaling_method!r}"
        )

    original = np.asarray(narrowband, dtype=np.float32).copy()
    nb = np.asarray(narrowband, dtype=np.float32)
    cont = np.asarray(continuum, dtype=np.float32)

    if nb.ndim not in (2, 3):
        raise ValueError(f"narrowband must be (H, W) or (C, H, W), got shape {nb.shape}")

    if nb.ndim == 2:
        if cont.shape != nb.shape:
            raise ValueError(
                f"continuum shape {cont.shape} does not match mono narrowband shape {nb.shape}"
            )
        result = _subtract_channel(nb, cont, params, progress, seed=0)
    else:
        n_channels = nb.shape[0]
        if cont.ndim == 2:
            if cont.shape != nb.shape[1:]:
                raise ValueError(
                    f"continuum shape {cont.shape} does not match "
                    f"narrowband spatial shape {nb.shape[1:]}"
                )
            cont_channels = [cont] * n_channels
        elif cont.ndim == 3 and cont.shape == nb.shape:
            cont_channels = [cont[c] for c in range(n_channels)]
        else:
            raise ValueError(
                f"continuum shape {cont.shape} is not compatible with "
                f"color narrowband shape {nb.shape}"
            )

        out_channels = []
        for c in range(n_channels):
            progress(c / n_channels, f"Channel {c + 1}/{n_channels}...")
            out_channels.append(
                _subtract_channel(nb[c], cont_channels[c], params, progress, seed=c)
            )
        result = np.stack(out_channels, axis=0)

    progress(1.0, "Continuum subtraction complete")
    return apply_mask(original, result, mask)
