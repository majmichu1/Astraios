"""Transient Hunter — supernova and asteroid/comet candidate detection.

Finds things that appear, vanish, or move between a reference image and one
or more later images of the same field:

- NEW: a source present in a later frame but absent from the reference
  (supernova / nova candidate).
- MOVED: a source present in both frames but at a different position
  (asteroid / comet candidate); reports the motion vector.
- VANISHED: a source present in the reference but absent from the later
  frame.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro

Notes on the port
------------------
SASpro's ``SupernovaAsteroidHunterDialog`` builds a difference image
(``new - reference``, clipped to [0, 1]) and finds blobs above a flat
threshold with ``cv2.connectedComponentsWithStats`` (``detectAnomaliesConnected``),
skipping a 5% border — that residual-detection idea is what
``_detect_residual_sources``/``_match_frame`` below port, using Astraios's
shared ``star_detection.detect_stars`` (MAD/sigma thresholding + contour
stats) in place of the flat-threshold + raw connected-components call.

SASpro's "asteroid" support is actually a *catalog* cross-match: it
plate-solves the reference frame, queries a local minor-body database,
predicts asteroid/comet positions at the exposure's JD with skyfield, and
matches those predicted pixel positions against detected anomalies
(``_get_predicted_minor_bodies_for_field`` / ``_match_anomalies_to_minor_bodies``).
It has no self-contained two-frame motion detector. The MOVED classification
here — nearest-neighbor matching between vanished and newly-appeared
residual sources — is new work built for this port; it reuses the same
greedy nearest-neighbor matching idea as ``star_detection.find_transform``.
Catalog-based (SIMBAD/minor-planet) cross-checking is not ported; see the
module docstring in the final report for deferred items.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from scipy.spatial.distance import cdist

from astraios.core.image_io import FrameType, ImageData
from astraios.core.stacking import (
    NormalizationMethod,
    RegistrationMode,
    StackingParams,
    align_frames,
    normalize_stack,
)
from astraios.core.star_detection import Star, StarField, detect_stars

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


class TransientKind(Enum):
    """Classification of a detected transient candidate."""

    NEW = auto()  # source in the new frame absent from the reference (supernova candidate)
    MOVED = auto()  # source present in both frames but shifted (asteroid/comet candidate)
    VANISHED = auto()  # source present in the reference but absent from the new frame


@dataclass
class TransientHunterParams:
    """All tunable settings for :func:`hunt_transients`.

    Field docstrings double as the source of truth for a future UI panel.
    """

    detection_sigma: float = 5.0
    """Residual-detection threshold, in MAD-noise sigmas above background.

    Adapted from SASpro's flat "Anomaly Detection Threshold" slider (0.10 by
    default) to a noise-relative sigma threshold, matching how
    ``star_detection.detect_stars`` already thresholds everywhere else in
    Astraios (more robust to varying stack depth / SNR than a fixed cut)."""

    min_flux: float = 0.02
    """Minimum residual peak amplitude (post-normalization, [0, 1] scale)
    for a detected blob to be kept as a candidate."""

    min_area: int = 25
    """Minimum blob area in pixels (SASpro rejected area < 25)."""

    max_area: int = 40_000
    """Maximum blob area in pixels (SASpro rejected boxes wider/taller than
    200 px; 200*200 = 40000 is the equivalent area cap)."""

    match_radius: float = 50.0
    """Search radius, in pixels, used to pair a newly-appeared residual
    source with a vanished one when classifying MOVED (asteroid/comet)
    candidates."""

    variable_star_radius: float = 3.0
    """Radius, in pixels, within which a brightened residual source that
    coincides with an existing reference-frame star is treated as a
    brightening variable star rather than a NEW (supernova) candidate.

    This cross-match against the reference frame's own star list is new
    behaviour added for this port — SASpro's dialog has no equivalent
    check and would flag a bright variable as an anomaly."""

    normalize: bool = True
    """Robustly rescale each new frame's background level/contrast to match
    the reference before differencing, so a plain brightness offset between
    frames doesn't create false residuals."""

    normalization_method: NormalizationMethod = NormalizationMethod.ADDITIVE_SCALING
    """Which :class:`astraios.core.stacking.NormalizationMethod` to use when
    ``normalize`` is enabled."""

    register: bool = True
    """Align each new frame onto the reference frame's pixel grid before
    differencing. Ignored when ``already_aligned=True`` is passed to
    :func:`hunt_transients`."""

    registration_mode: RegistrationMode = RegistrationMode.STAR_1_PASS
    """Which :class:`astraios.core.stacking.RegistrationMode` to use when
    ``register`` is enabled."""

    edge_margin_fraction: float = 0.05
    """Fraction of width/height excluded at each border from candidate
    detection (SASpro skips a 5% border in ``detectAnomaliesConnected``)."""

    max_candidates: int = 500
    """Hard cap on the number of candidates returned per frame (highest
    residual flux first), to bound pathological worst-case result sizes."""

    use_gpu: bool = True
    """Prefer GPU registration (via ``stacking.align_frames``); falls back
    to CPU automatically when no GPU is available."""


@dataclass
class TransientCandidate:
    """A single detected transient (supernova/asteroid/comet candidate)."""

    x: float
    y: float
    kind: TransientKind
    flux: float
    frame_index: int = 0
    dx: float | None = None  # motion vector x-component, MOVED only
    dy: float | None = None  # motion vector y-component, MOVED only


@dataclass
class TransientResult:
    """Result of :func:`hunt_transients`."""

    candidates: list[TransientCandidate] = field(default_factory=list)
    diff_images: list[np.ndarray] = field(default_factory=list)
    """Positive residual (new - reference, luminance, clipped >= 0) per new
    frame, in the same order as the input ``new_images``, for display."""


def _to_luminance(image: np.ndarray) -> np.ndarray:
    """Return a (H, W) float32 luminance view of a mono or color image."""
    if image.ndim == 3:
        return image.mean(axis=0).astype(np.float32)
    return image.astype(np.float32)


def _normalize_pair(
    ref: np.ndarray, new: np.ndarray, method: NormalizationMethod
) -> tuple[np.ndarray, np.ndarray]:
    """Robustly match ``new``'s background level/contrast to ``ref``.

    Reuses ``stacking.normalize_stack`` on a 2-frame stack; the reference is
    always stack index 0, so it is returned unchanged (scale=1, shift=0).
    """
    stack = np.stack([ref, new], axis=0)
    normed = normalize_stack(stack, method=method)
    return normed[0], normed[1]


def _within_margin(x: float, y: float, w: int, h: int, margin_x: int, margin_y: int) -> bool:
    return margin_x <= x <= (w - 1 - margin_x) and margin_y <= y <= (h - 1 - margin_y)


def _detect_residual_sources(
    diff: np.ndarray, params: TransientHunterParams
) -> list[Star]:
    """Detect blobs in a (clipped, non-negative) residual image."""
    max_stars = max(4 * params.max_candidates, 200)
    residual_field: StarField = detect_stars(
        diff,
        max_stars=max_stars,
        sigma_threshold=params.detection_sigma,
        min_area=params.min_area,
        max_area=params.max_area,
    )
    return [s for s in residual_field.stars if s.flux >= params.min_flux]


def _match_frame(
    ref_lum: np.ndarray,
    new_lum: np.ndarray,
    ref_stars: StarField,
    frame_index: int,
    params: TransientHunterParams,
) -> tuple[list[TransientCandidate], np.ndarray]:
    """Diff one (already registered/normalized) frame pair and classify residuals."""
    h, w = ref_lum.shape

    diff_pos = np.clip(new_lum - ref_lum, 0.0, None).astype(np.float32)
    diff_neg = np.clip(ref_lum - new_lum, 0.0, None).astype(np.float32)

    pos_sources = _detect_residual_sources(diff_pos, params)
    neg_sources = _detect_residual_sources(diff_neg, params)

    margin_x = int(round(params.edge_margin_fraction * w))
    margin_y = int(round(params.edge_margin_fraction * h))
    pos_sources = [s for s in pos_sources if _within_margin(s.x, s.y, w, h, margin_x, margin_y)]
    neg_sources = [s for s in neg_sources if _within_margin(s.x, s.y, w, h, margin_x, margin_y)]

    ref_positions = ref_stars.positions  # (N, 2)
    dist_to_ref: np.ndarray | None = None
    if len(ref_positions) and pos_sources:
        pos_positions = np.array([(s.x, s.y) for s in pos_sources], dtype=np.float32)
        dist_to_ref = cdist(pos_positions, ref_positions)

    candidates: list[TransientCandidate] = []
    used_neg: set[int] = set()

    for i, s in enumerate(pos_sources):
        nearest_ref_dist = dist_to_ref[i].min() if dist_to_ref is not None else float("inf")
        if nearest_ref_dist <= params.variable_star_radius:
            # Coincides with a known reference star -- a brightening variable,
            # not a genuinely new source.
            continue

        best_j = -1
        best_d = float("inf")
        for j, ns in enumerate(neg_sources):
            if j in used_neg:
                continue
            d = float(np.hypot(s.x - ns.x, s.y - ns.y))
            if d < best_d:
                best_d = d
                best_j = j

        if best_j >= 0 and best_d <= params.match_radius:
            ns = neg_sources[best_j]
            used_neg.add(best_j)
            candidates.append(
                TransientCandidate(
                    x=float(s.x),
                    y=float(s.y),
                    kind=TransientKind.MOVED,
                    flux=float(s.flux),
                    frame_index=frame_index,
                    dx=float(s.x - ns.x),
                    dy=float(s.y - ns.y),
                )
            )
        else:
            candidates.append(
                TransientCandidate(
                    x=float(s.x),
                    y=float(s.y),
                    kind=TransientKind.NEW,
                    flux=float(s.flux),
                    frame_index=frame_index,
                )
            )

    for j, ns in enumerate(neg_sources):
        if j in used_neg:
            continue
        candidates.append(
            TransientCandidate(
                x=float(ns.x),
                y=float(ns.y),
                kind=TransientKind.VANISHED,
                flux=float(ns.flux),
                frame_index=frame_index,
            )
        )

    candidates.sort(key=lambda c: -c.flux)
    return candidates[: params.max_candidates], diff_pos


def hunt_transients(
    reference: np.ndarray,
    new_images: np.ndarray | list[np.ndarray],
    params: TransientHunterParams | None = None,
    already_aligned: bool = False,
    progress: ProgressCallback = _noop_progress,
) -> TransientResult:
    """Find new, moved, and vanished sources between a reference and later image(s).

    Parameters
    ----------
    reference : ndarray
        Reference frame, float32 [0, 1], mono (H, W) or color (C, H, W).
    new_images : ndarray or list of ndarray
        One later frame, or a list of them. Each must be the same mono/color
        layout as ``reference``; if not pre-aligned it will be registered
        onto ``reference``'s pixel grid.
    params : TransientHunterParams, optional
    already_aligned : bool
        If True, skip registration entirely and assume ``new_images`` are
        already pixel-aligned to ``reference`` (shapes must match exactly).
    progress : callable(fraction, message), optional

    Returns
    -------
    TransientResult
    """
    if params is None:
        params = TransientHunterParams()

    ref = np.asarray(reference, dtype=np.float32)

    if isinstance(new_images, np.ndarray) and new_images.ndim == ref.ndim:
        images_in: list[np.ndarray] = [new_images]
    else:
        images_in = list(new_images)
    images_in = [np.asarray(img, dtype=np.float32) for img in images_in]

    n = len(images_in)
    if n == 0:
        raise ValueError("hunt_transients: no new_images provided")

    progress(0.0, "Preparing frames...")

    if already_aligned:
        aligned_new = images_in
        for i, img in enumerate(aligned_new):
            if img.shape != ref.shape:
                raise ValueError(
                    f"hunt_transients: already_aligned=True but new_images[{i}] shape "
                    f"{img.shape} != reference shape {ref.shape}"
                )
    elif params.register:
        progress(0.05, "Registering frames to reference...")
        stack_params = StackingParams(
            registration_mode=params.registration_mode,
            reference_frame_index=0,
            use_gpu=params.use_gpu,
        )
        wrapped = [ImageData(data=ref, frame_type=FrameType.LIGHT)] + [
            ImageData(data=img, frame_type=FrameType.LIGHT) for img in images_in
        ]
        aligned_wrapped = align_frames(
            wrapped,
            stack_params,
            progress=lambda f, m: progress(0.05 + 0.35 * f, m),
        )
        aligned_new = [w.data for w in aligned_wrapped[1:]]
    else:
        aligned_new = images_in
        for i, img in enumerate(aligned_new):
            if img.shape != ref.shape:
                raise ValueError(
                    f"hunt_transients: register=False and new_images[{i}] shape "
                    f"{img.shape} != reference shape {ref.shape} -- enable "
                    "params.register or pass already_aligned frames."
                )

    ref_lum_raw = _to_luminance(ref)
    ref_stars = detect_stars(ref_lum_raw)
    log.info("hunt_transients: %d reference stars detected", len(ref_stars))

    candidates_all: list[TransientCandidate] = []
    diff_images: list[np.ndarray] = []

    for i, new_img in enumerate(aligned_new):
        progress(0.4 + 0.55 * (i / n), f"Analyzing frame {i + 1}/{n}...")

        if new_img.shape != ref.shape:
            log.warning(
                "hunt_transients: frame %d shape mismatch after registration, skipping", i
            )
            diff_images.append(np.zeros_like(ref_lum_raw))
            continue

        if params.normalize:
            ref_n, new_n = _normalize_pair(ref, new_img, params.normalization_method)
        else:
            ref_n, new_n = ref, new_img

        ref_lum = _to_luminance(ref_n)
        new_lum = _to_luminance(new_n)

        frame_candidates, diff_pos = _match_frame(ref_lum, new_lum, ref_stars, i, params)
        candidates_all.extend(frame_candidates)
        diff_images.append(diff_pos)

    progress(1.0, f"Transient hunt complete: {len(candidates_all)} candidate(s)")
    return TransientResult(candidates=candidates_all, diff_images=diff_images)
