"""Offline plate solving against a local Gaia DR3 catalog.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

Ports SASpro's ``_solve_with_GAIA`` / ``_hough_match_catalog_to_image`` /
``_gaia_fit_and_validate`` (``plate_solver.py``) — a *hint-based* solver: it
requires an approximate RA/Dec/pixel-scale seed (from a FITS header, mount
position, or user input) and refines it against :class:`~astraios.core.gaia_catalog.GaiaCatalog`.
It does **not** blind-solve (see "What remains" below).

Algorithm (Groth 1986; Hough 1962 / Ballard 1981; Valdes et al. 1995;
Tabur 2007; Fischler & Bolles 1981 for the RANSAC refinement stage):

1. Detect stars in the image (reuses :func:`astraios.core.star_detection.detect_stars`).
2. Build a seed TAN WCS from the RA/Dec/scale hint.
3. Search a small spiral of nearby field centers x pixel-scale variants
   (the hint is rarely exact). For each candidate:
   a. Query the local Gaia catalog for the field footprint.
   b. Project catalog stars to pixel space via the candidate seed WCS.
   c. Match image stars to projected catalog stars with a Generalized
      Hough Transform on (dx, dy) translation, refined by a RANSAC
      similarity-transform fit, then a final affine inlier pass.
4. On a match, fit a TAN WCS from the matched pairs, iteratively refine by
   re-querying the catalog against the improving WCS, run a quality gate on
   RMS residual and pair count, and (if enough pairs) fit a SIP polynomial
   for field distortion, keeping the SIP fit only if it improves RMS.
5. On a low-quality match, exclude the offending catalog stars and retry.

This module is pure CPU/numpy + astropy/scipy — pattern matching and WCS
fitting are geometry/linear-algebra problems, not GPU workloads, so it
intentionally does not touch ``device_manager``.

What works: hint-based ("near-field") solving as described above.
What remains (not ported): SASpro's blind solve fallback lives entirely in
its ASTAP/astrometry.net code paths, not in the Gaia-only path — SASpro's
in-house Gaia solver itself is hint-only. A true blind solve (no RA/Dec/scale
seed at all) would need a separate all-sky index (e.g. healpix-tiled
quad/triangle hashing) that SASpro does not build from these catalog files;
that piece is not implemented here either.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from astraios.core.gaia_catalog import GaiaCatalog, GaiaCatalogNotFoundError
from astraios.core.star_detection import detect_stars

if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS

log = logging.getLogger(__name__)


@dataclass
class GaiaSolveParams:
    """Settings for the offline Gaia plate solver.

    Hints are required — this solver refines a seed, it does not blind-solve.
    """

    ra_hint: float | None = None  # seed field-center RA, degrees (J2000)
    dec_hint: float | None = None  # seed field-center Dec, degrees (J2000)
    scale_hint: float | None = None  # seed pixel scale, arcsec/pixel
    rotation_hint: float = 0.0  # seed camera rotation, degrees

    search_radius: float = 1.0  # deg; kept for API/UI parity, informational
    n_search_rings: int = 2  # spiral rings of alternate centers around the hint
    scale_tolerance: float = 0.10  # +/- fractional pixel-scale search range

    mag_limit: float = 16.0  # faintest catalog G magnitude to consider
    max_stars: int = 1000  # max detected image stars to use
    max_catalog_stars: int = 1000  # max catalog stars per match attempt
    downsample: int = 1  # bin image by this factor before star detection

    match_tolerance_px: float = 10.0  # Hough bin size / nearest-neighbour tolerance
    min_matches: int = 6  # minimum matched pairs to accept a candidate transform

    max_iterations: int = 5  # WCS refinement iterations after the initial fit
    rms_converge_arcsec: float = 0.05  # stop refining once RMS improves less than this
    refine_tolerance_px: float = 8.0  # nearest-neighbour tolerance during refinement

    quality_rms_arcsec: float = 3.0  # max acceptable residual RMS for a final solution
    quality_min_pairs: int = 20  # min matched pairs for a final accepted solution
    sip_degree_max: int = 4  # cap on SIP polynomial distortion degree

    max_retry_rounds: int = 2  # initial attempt + N-1 retries excluding bad stars

    catalog_dir: Path | None = None  # None -> GaiaCatalog default (~/.astraios/gaia)


@dataclass
class GaiaSolveResult:
    """Result of an offline Gaia plate solve."""

    success: bool
    ra_center: float = 0.0  # degrees
    dec_center: float = 0.0  # degrees
    pixel_scale: float = 0.0  # arcsec/pixel
    rotation: float = 0.0  # degrees
    n_stars_matched: int = 0
    rms_arcsec: float = 0.0
    wcs_header: dict | None = None  # FITS WCS header, e.g. from wcs.to_header()
    message: str = ""


def _to_gray2d_unit(image: np.ndarray) -> np.ndarray:
    """Collapse to 2-D float32 in [0, 1]. Mono passes through, color->luminance."""
    a = np.asarray(image, dtype=np.float32)
    if a.ndim == 3:
        # Astraios color layout is (C, H, W).
        a = 0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2] if a.shape[0] >= 3 else a[0]
    mn, mx = float(a.min()), float(a.max())
    if mx > mn:
        a = (a - mn) / (mx - mn)
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def _detect_image_stars(gray: np.ndarray, params: GaiaSolveParams) -> np.ndarray:
    """Detect stars, adapting the threshold to land near a usable star count.

    Adapted from SASpro's multi-sigma-level adaptive loop, using
    :func:`astraios.core.star_detection.detect_stars` in place of SEP.
    """
    sigma_levels = [50.0, 25.0, 15.0, 10.0, 5.0, 3.0]
    target_min = min(30, params.max_stars)
    best_positions: np.ndarray | None = None

    for sigma in sigma_levels:
        sf = detect_stars(gray, max_stars=params.max_stars, sigma_threshold=sigma)
        n = len(sf)
        log.debug("Gaia solver: star detection sigma=%.0f -> %d stars", sigma, n)
        if n >= target_min:
            return sf.positions
        if n > 0:
            best_positions = sf.positions

    if best_positions is None:
        return np.empty((0, 2), dtype=np.float32)
    return best_positions


def _spiral_centers(
    ra0: float, dec0: float, fov_deg: float, n_rings: int
) -> list[tuple[float, float]]:
    """Concentric ring offsets around (ra0, dec0), spaced by 0.4x the FOV."""
    centers = [(ra0, dec0)]
    step_deg = fov_deg * 0.4
    for ring in range(1, n_rings + 1):
        radius = ring * step_deg
        n_pts = max(6, int(2 * math.pi * ring))
        for i in range(n_pts):
            angle = 2 * math.pi * i / n_pts
            d_ra = radius * math.cos(angle) / max(math.cos(math.radians(dec0)), 0.01)
            d_dec = radius * math.sin(angle)
            new_ra = (ra0 + d_ra) % 360.0
            new_dec = max(-90.0, min(90.0, dec0 + d_dec))
            centers.append((new_ra, new_dec))
    return centers


def _scale_variants(scale: float, tolerance: float) -> list[float]:
    """Pixel-scale candidates: exact hint plus +/- fractional offsets."""
    fracs = [0.0, -0.5, 0.5, -1.0, 1.0]
    return [scale * (1.0 + tolerance * f) for f in fracs]


def _make_seed_wcs(
    ra_deg: float, dec_deg: float, scale_arcsec: float, img_w: int, img_h: int, rot_deg: float
) -> WCS:
    """Build a TAN seed WCS with the given center, scale, and rotation."""
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.crpix = [img_w / 2.0, img_h / 2.0]
    w.wcs.crval = [float(ra_deg), float(dec_deg)]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    scale_deg = float(scale_arcsec) / 3600.0
    rot_rad = math.radians(rot_deg)
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    w.wcs.cd = np.array(
        [
            [-scale_deg * cos_r, scale_deg * sin_r],
            [-scale_deg * sin_r, -scale_deg * cos_r],
        ]
    )
    w.wcs.set()
    return w


def _grid_sample(pts: np.ndarray, n: int, w: float, h: float) -> np.ndarray:
    """Spatially-balanced subsample of at most *n* points over a w x h field."""
    if len(pts) <= n:
        return pts.copy()
    cols = max(1, int(np.sqrt(n * w / max(h, 1))) + 1)
    rows = max(1, int(np.sqrt(n * h / max(w, 1))) + 1)
    per = max(1, n // (cols * rows) + 1)
    cw, rh = w / cols, h / rows
    out = []
    for r in range(rows):
        for c in range(cols):
            mask = (
                (pts[:, 0] >= c * cw)
                & (pts[:, 0] < (c + 1) * cw)
                & (pts[:, 1] >= r * rh)
                & (pts[:, 1] < (r + 1) * rh)
            )
            cell = pts[mask]
            if len(cell):
                out.append(cell[:per])
    result = np.vstack(out) if out else pts[:n]
    return result[:n]


def _hough_match_catalog_to_image(
    img_stars: np.ndarray,
    cat_xy: np.ndarray,
    img_w: int,
    img_h: int,
    max_stars: int = 1000,
    min_matches: int = 6,
    match_tol_px: float = 10.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Match image stars to projected catalog stars via translation voting.

    A Generalized Hough Transform restricted to pure translation: the seed
    WCS already encodes scale and rotation, so the residual transform is
    dominated by a small translation. Refined by a RANSAC similarity-
    transform fit over the peak-bin neighbourhood (handles residual
    rotation/scale error in the seed), then a final affine inlier pass.

    References: Hough (1962) US Patent 3,069,654; Ballard (1981) Pattern
    Recognition 13(2):111-122; Groth (1986) AJ 91, 1244; Valdes et al.
    (1995) PASP 107, 1119; Tabur (2007) PASA 24, 189 (arXiv:0710.3618);
    Fischler & Bolles (1981) Comm. ACM 24(6):381-395 (RANSAC).

    Returns
    -------
    (matched_image_xy, matched_catalog_xy) or (None, None) on failure.
    """
    from scipy.spatial import KDTree

    src = _grid_sample(img_stars, max_stars, img_w, img_h)
    ref = _grid_sample(cat_xy, max_stars, img_w, img_h)

    n_src, n_ref = len(src), len(ref)
    log.debug("Hough match: %d image stars, %d catalog stars", n_src, n_ref)
    if n_src < 3 or n_ref < 3:
        return None, None

    bin_px = match_tol_px * 2.0
    dx_range, dy_range = img_w, img_h

    votes: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i, s in enumerate(src):
        for j, r in enumerate(ref):
            dx = r[0] - s[0]
            dy = r[1] - s[1]
            if abs(dx) > dx_range * 0.5 or abs(dy) > dy_range * 0.5:
                continue
            key = (int(np.round(dx / bin_px)), int(np.round(dy / bin_px)))
            votes.setdefault(key, []).append((i, j))

    if not votes:
        return None, None

    best_key = max(votes, key=lambda k: len(votes[k]))
    bx0, by0 = best_key
    merged = list(votes[best_key])
    for dbx in range(-1, 2):
        for dby in range(-1, 2):
            if dbx == 0 and dby == 0:
                continue
            merged.extend(votes.get((bx0 + dbx, by0 + dby), []))

    if len(merged) < 4:
        return None, None

    cand_src = np.array([src[i] for i, _ in merged], dtype=np.float64)
    cand_ref = np.array([ref[j] for _, j in merged], dtype=np.float64)

    def _fit_similarity(pts_src: np.ndarray, pts_ref: np.ndarray) -> np.ndarray:
        n = len(pts_src)
        mat = np.zeros((2 * n, 4), dtype=np.float64)
        b = np.zeros(2 * n, dtype=np.float64)
        x, y = pts_src[:, 0], pts_src[:, 1]
        mat[0::2, 0] = x
        mat[0::2, 1] = -y
        mat[0::2, 2] = 1.0
        mat[1::2, 0] = y
        mat[1::2, 1] = x
        mat[1::2, 3] = 1.0
        b[0::2] = pts_ref[:, 0]
        b[1::2] = pts_ref[:, 1]
        sol, *_ = np.linalg.lstsq(mat, b, rcond=None)
        return sol

    def _apply_sim(pts: np.ndarray, sol: np.ndarray) -> np.ndarray:
        a_, b_, tx, ty = sol
        xs, ys = pts[:, 0], pts[:, 1]
        return np.column_stack([a_ * xs - b_ * ys + tx, b_ * xs + a_ * ys + ty])

    src_transformed: np.ndarray
    try:
        n_cand = len(cand_src)
        if n_cand < 4:
            raise ValueError("not enough candidates for similarity fit")

        rng = np.random.default_rng(0)
        ransac_tol = match_tol_px * 2.0
        idx_pairs = [(i, j) for i in range(n_cand) for j in range(i + 1, n_cand)]
        rng.shuffle(idx_pairs)
        n_trials = min(200, len(idx_pairs))

        best_inliers = None
        best_sol = None
        for i, j in idx_pairs[:n_trials]:
            try:
                sol = _fit_similarity(cand_src[[i, j]], cand_ref[[i, j]])
            except np.linalg.LinAlgError:
                continue
            pred = _apply_sim(cand_src, sol)
            err = np.hypot(pred[:, 0] - cand_ref[:, 0], pred[:, 1] - cand_ref[:, 1])
            inliers = err < ransac_tol
            if best_inliers is None or inliers.sum() > best_inliers.sum():
                best_inliers = inliers
                best_sol = sol

        if best_sol is None or best_inliers is None or best_inliers.sum() < 4:
            raise ValueError("RANSAC found no consensus")

        sol = _fit_similarity(cand_src[best_inliers], cand_ref[best_inliers])
        src_transformed = _apply_sim(src, sol)
    except (ValueError, np.linalg.LinAlgError) as e:
        log.debug("Hough match: similarity fit failed (%s), using translation only", e)
        best_dx = best_key[0] * bin_px
        best_dy = best_key[1] * bin_px
        src_transformed = src + np.array([best_dx, best_dy])

    tree = KDTree(ref)
    dists, idxs = tree.query(src_transformed, k=1, workers=-1)

    inlier_mask = dists < match_tol_px * 1.5
    img_m = src[inlier_mask]
    cat_m = ref[idxs[inlier_mask]]
    if len(img_m) < min_matches:
        return None, None

    try:
        from scipy.linalg import lstsq

        ones = np.ones((len(img_m), 1))
        design = np.hstack([img_m, ones])
        coef_x, _, _, _ = lstsq(design, cat_m[:, 0])
        coef_y, _, _, _ = lstsq(design, cat_m[:, 1])
        pred_x = design @ coef_x
        pred_y = design @ coef_y
        res = np.sqrt((cat_m[:, 0] - pred_x) ** 2 + (cat_m[:, 1] - pred_y) ** 2)
        inliers = res < match_tol_px
        if inliers.sum() < min_matches:
            inliers = res < match_tol_px * 3
        if inliers.sum() < min_matches:
            return None, None
        img_m = img_m[inliers]
        cat_m = cat_m[inliers]
    except np.linalg.LinAlgError as e:
        log.debug("Hough match: affine polish failed: %s", e)

    return img_m, cat_m


def _project_catalog(
    catalog: GaiaCatalog,
    seed_wcs: WCS,
    img_w: int,
    img_h: int,
    mag_limit: float,
    margin: int,
    excluded_sids: set[int],
) -> tuple[np.ndarray, SkyCoord, np.ndarray]:
    """Query the catalog around *seed_wcs*'s footprint and project to pixels.

    Returns (cat_xy, cat_sky, cat_sid) restricted to the image + margin.
    """
    from astropy.coordinates import SkyCoord

    corners_pix = [(0, 0), (img_w - 1, 0), (0, img_h - 1), (img_w - 1, img_h - 1)]
    corner_sky = [seed_wcs.pixel_to_world(x, y) for x, y in corners_pix]
    center = seed_wcs.pixel_to_world(img_w / 2.0, img_h / 2.0)
    ra_span = max(abs(c.ra.deg - center.ra.deg) for c in corner_sky)
    dec_span = max(abs(c.dec.deg - center.dec.deg) for c in corner_sky)
    radius_deg = math.hypot(ra_span, dec_span) + 0.1

    sources = catalog.cone_query(
        float(center.ra.deg), float(center.dec.deg), radius_deg, mag_limit=mag_limit
    )
    sources = [s for s in sources if s.source_id not in excluded_sids]
    if not sources:
        return (
            np.empty((0, 2), dtype=np.float32),
            SkyCoord(ra=[], dec=[], unit="deg"),
            np.empty(0, dtype=np.int64),
        )

    cat_sky = SkyCoord(ra=[s.ra for s in sources], dec=[s.dec for s in sources], unit="deg")
    cat_sid = np.array([s.source_id for s in sources], dtype=np.int64)
    px, py = seed_wcs.world_to_pixel(cat_sky)
    cat_xy = np.column_stack([px, py]).astype(np.float32)

    in_bounds = (
        (cat_xy[:, 0] >= -margin)
        & (cat_xy[:, 0] < img_w + margin)
        & (cat_xy[:, 1] >= -margin)
        & (cat_xy[:, 1] < img_h + margin)
    )
    return cat_xy[in_bounds], cat_sky[in_bounds], cat_sid[in_bounds]


def _fit_and_validate(
    img_matched: np.ndarray,
    cat_matched_xy: np.ndarray,
    seed_wcs: WCS,
    img_w: int,
    img_h: int,
    catalog: GaiaCatalog,
    params: GaiaSolveParams,
) -> tuple[bool, WCS | str, float]:
    """Fit + iteratively refine a WCS from matched pairs, then quality-gate it.

    Returns (ok, wcs_or_error, rms_arcsec).
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import fit_wcs_from_points
    from scipy.spatial import KDTree

    n_matched = len(img_matched)

    try:
        matched_sky = seed_wcs.pixel_to_world(cat_matched_xy[:, 0], cat_matched_xy[:, 1])
    except Exception as e:  # noqa: BLE001 - astropy raises assorted exceptions here
        return False, f"sky coordinate recovery failed: {e}", 0.0

    def _proj_point(sky: SkyCoord) -> SkyCoord:
        return SkyCoord(
            ra=float(np.mean(sky.ra.deg)) * u.deg,
            dec=float(np.mean(sky.dec.deg)) * u.deg,
        )

    try:
        wcs_solution = fit_wcs_from_points(
            (img_matched[:, 0], img_matched[:, 1]),
            matched_sky,
            projection="TAN",
            proj_point=_proj_point(matched_sky),
        )
        wcs_solution.array_shape = (img_h, img_w)
    except Exception as e:  # noqa: BLE001
        return False, f"WCS fit failed: {e}", 0.0

    prev_rms = None
    rms = 0.0
    for _iteration in range(params.max_iterations):
        sky_fit = wcs_solution.pixel_to_world(img_matched[:, 0], img_matched[:, 1])
        sep_arcsec = matched_sky.separation(sky_fit).arcsec
        rms = float(np.sqrt(np.mean(sep_arcsec**2)))
        if prev_rms is not None and (prev_rms - rms) < params.rms_converge_arcsec:
            break
        prev_rms = rms

        try:
            t_cat_xy, t_cat_sky, _sid = _project_catalog(
                catalog, wcs_solution, img_w, img_h, params.mag_limit, margin=5, excluded_sids=set()
            )
            if len(t_cat_xy) < params.quality_min_pairs // 2:
                break

            tree = KDTree(t_cat_xy)
            dists, idxs = tree.query(img_matched, k=1, workers=-1)
            inlier_mask = dists < params.refine_tolerance_px
            t_img_m = img_matched[inlier_mask]
            t_cat_sky_m = t_cat_sky[idxs[inlier_mask]]
            if len(t_img_m) < params.quality_min_pairs // 2:
                break

            t_wcs = fit_wcs_from_points(
                (t_img_m[:, 0], t_img_m[:, 1]),
                t_cat_sky_m,
                projection="TAN",
                proj_point=_proj_point(t_cat_sky_m),
            )
            t_wcs.array_shape = (img_h, img_w)
            t_sky_fit = t_wcs.pixel_to_world(t_img_m[:, 0], t_img_m[:, 1])
            t_rms = float(np.sqrt(np.mean(t_cat_sky_m.separation(t_sky_fit).arcsec**2)))

            if len(t_img_m) > n_matched or (len(t_img_m) >= n_matched and t_rms <= rms):
                img_matched, matched_sky = t_img_m, t_cat_sky_m
                n_matched, rms, wcs_solution = len(t_img_m), t_rms, t_wcs
            else:
                break
        except (GaiaCatalogNotFoundError, ValueError) as e:
            log.debug("Gaia solver: refinement iteration failed: %s", e)
            break

    sky_fit = wcs_solution.pixel_to_world(img_matched[:, 0], img_matched[:, 1])
    sep_arcsec = matched_sky.separation(sky_fit).arcsec
    rms = float(np.sqrt(np.mean(sep_arcsec**2)))

    if rms > params.quality_rms_arcsec or n_matched < params.quality_min_pairs:
        return False, (
            f"match quality too low (RMS={rms:.2f}\", n={n_matched}; "
            f"need RMS<={params.quality_rms_arcsec}\" and n>={params.quality_min_pairs})"
        ), rms

    if n_matched >= 20:
        try:
            sip_degree = 4 if n_matched >= 100 else 3 if n_matched >= 50 else 2
            sip_degree = min(sip_degree, params.sip_degree_max)
            wcs_sip = fit_wcs_from_points(
                (img_matched[:, 0], img_matched[:, 1]),
                matched_sky,
                projection="TAN",
                proj_point=_proj_point(matched_sky),
                sip_degree=sip_degree,
            )
            wcs_sip.array_shape = (img_h, img_w)
            res_sip = matched_sky.separation(wcs_sip.pixel_to_world(*img_matched.T)).arcsec
            rms_sip = float(np.sqrt(np.mean(res_sip**2)))
            if rms_sip < rms:
                wcs_solution, rms = wcs_sip, rms_sip
        except Exception as e:  # noqa: BLE001 - SIP fit failure should never abort the solve
            log.debug("Gaia solver: SIP fit failed, keeping TAN solution: %s", e)

    return True, wcs_solution, rms


def solve_with_gaia_catalog(
    image: np.ndarray,
    params: GaiaSolveParams,
) -> GaiaSolveResult:
    """Solve *image* against the local Gaia catalog using a seed RA/Dec/scale.

    Parameters
    ----------
    image : ndarray
        float32 image in [0, 1], mono (H, W) or color (C, H, W).
    params : GaiaSolveParams
        Must set ``ra_hint``, ``dec_hint``, and ``scale_hint``.

    Returns
    -------
    GaiaSolveResult
        ``success=False`` with a ``message`` describing why on failure —
        never raises for expected failure modes (missing catalog, no stars,
        no match). Programming errors (bad params) still raise.
    """
    if params.ra_hint is None or params.dec_hint is None:
        return GaiaSolveResult(success=False, message="Gaia solver: no RA/Dec seed provided")
    if not params.scale_hint or params.scale_hint <= 0:
        return GaiaSolveResult(success=False, message="Gaia solver: no pixel scale seed provided")

    img_h = int(image.shape[-2])
    img_w = int(image.shape[-1])
    if params.downsample > 1:
        img_h = max(1, img_h // params.downsample)
        img_w = max(1, img_w // params.downsample)

    fov_deg = (max(img_h, img_w) * params.scale_hint) / 3600.0
    scale_variants = _scale_variants(params.scale_hint, params.scale_tolerance)
    search_centers = _spiral_centers(
        params.ra_hint, params.dec_hint, fov_deg, params.n_search_rings
    )

    try:
        catalog = GaiaCatalog(params.catalog_dir)
    except OSError as e:
        return GaiaSolveResult(success=False, message=f"Gaia solver: could not open catalog: {e}")

    if not catalog.installed_bands:
        catalog.close()
        return GaiaSolveResult(
            success=False,
            message=(
                f"Gaia solver: no catalog files installed in {catalog.catalog_dir}. "
                "Download a magnitude band first — see "
                "astraios.core.gaia_catalog.download_band()."
            ),
        )

    try:
        gray = _to_gray2d_unit(image)
        if params.downsample > 1:
            gray = gray[:: params.downsample, :: params.downsample]
        img_stars = _detect_image_stars(gray, params)

        if len(img_stars) < params.min_matches:
            return GaiaSolveResult(
                success=False,
                message=f"Gaia solver: only {len(img_stars)} stars detected — too few",
            )

        excluded_sids: set[int] = set()
        last_error = "Gaia solver: no match found across all search attempts"

        for retry_round in range(params.max_retry_rounds):
            best_result = None

            for c_ra, c_dec in search_centers:
                if best_result is not None:
                    break
                for c_scale in scale_variants:
                    if best_result is not None:
                        break
                    try:
                        seed_wcs = _make_seed_wcs(
                            c_ra, c_dec, c_scale, img_w, img_h, params.rotation_hint
                        )
                        cat_xy_all, _cat_sky_all, cat_sid_all = _project_catalog(
                            catalog, seed_wcs, img_w, img_h, params.mag_limit, 50, excluded_sids
                        )
                        if len(cat_xy_all) < params.min_matches:
                            continue

                        cat_xy = _grid_sample(cat_xy_all, params.max_catalog_stars, img_w, img_h)

                        img_matched, cat_matched_xy = _hough_match_catalog_to_image(
                            img_stars,
                            cat_xy,
                            img_w,
                            img_h,
                            max_stars=params.max_catalog_stars,
                            min_matches=params.min_matches,
                            match_tol_px=params.match_tolerance_px,
                        )
                        if img_matched is not None and len(img_matched) >= params.min_matches:
                            best_result = (
                                img_matched,
                                cat_matched_xy,
                                seed_wcs,
                                cat_xy_all,
                                cat_sid_all,
                            )
                    except GaiaCatalogNotFoundError:
                        raise
                    except (ValueError, np.linalg.LinAlgError) as e:
                        log.debug("Gaia solver: attempt (%.4f, %.4f) failed: %s", c_ra, c_dec, e)
                        continue

            if best_result is None:
                break

            img_matched, cat_matched_xy, seed_wcs, cat_xy_full, cat_sid_full = best_result

            matched_sids: list[int] = []
            lookup = {
                (round(float(xy[0]), 2), round(float(xy[1]), 2)): int(sid)
                for xy, sid in zip(cat_xy_full, cat_sid_full, strict=True)
            }
            for mxy in cat_matched_xy:
                key = (round(float(mxy[0]), 2), round(float(mxy[1]), 2))
                if key in lookup:
                    matched_sids.append(lookup[key])

            ok, result_or_err, rms = _fit_and_validate(
                img_matched, cat_matched_xy, seed_wcs, img_w, img_h, catalog, params
            )

            if ok:
                wcs_solution = result_or_err
                sky_center = wcs_solution.pixel_to_world(img_w / 2.0, img_h / 2.0)
                cd = np.asarray(wcs_solution.wcs.cd) if wcs_solution.wcs.has_cd() else None
                if cd is None:
                    scale_deg = float(
                        np.sqrt(np.abs(np.linalg.det(wcs_solution.pixel_scale_matrix)))
                    )
                    rotation = 0.0
                else:
                    scale_deg = float(np.sqrt(np.abs(np.linalg.det(cd))))
                    rotation = float(np.degrees(np.arctan2(cd[0, 1], cd[0, 0])))
                header = dict(wcs_solution.to_header(relax=True))
                return GaiaSolveResult(
                    success=True,
                    ra_center=float(sky_center.ra.deg),
                    dec_center=float(sky_center.dec.deg),
                    pixel_scale=scale_deg * 3600.0,
                    rotation=rotation,
                    n_stars_matched=len(img_matched),
                    rms_arcsec=rms,
                    wcs_header=header,
                    message="ok",
                )

            last_error = str(result_or_err)
            if matched_sids and retry_round + 1 < params.max_retry_rounds:
                excluded_sids.update(matched_sids)
                continue
            break

        return GaiaSolveResult(success=False, message=f"Gaia solver: {last_error}")
    finally:
        catalog.close()


def plate_solve_gaia(
    image: np.ndarray,
    ra_hint: float,
    dec_hint: float,
    scale_hint: float,
    params: GaiaSolveParams | None = None,
) -> dict | None:
    """Adapter matching :mod:`astraios.core.star_catalog`'s solver-dict convention.

    Returns ``{"ra", "dec", "scale", "rotation", "wcs_header"}`` on success (so
    :func:`astraios.core.plate_solve._result_from_solver_dict` can consume it
    exactly like the ASTAP/astrometry.net adapters), or ``None`` on failure.
    Not wired into ``plate_solve.py`` here — see that module for the dispatch.
    """
    p = params or GaiaSolveParams()
    p.ra_hint, p.dec_hint, p.scale_hint = ra_hint, dec_hint, scale_hint
    result = solve_with_gaia_catalog(image, p)
    if not result.success:
        log.warning("Gaia plate solve failed: %s", result.message)
        return None
    return {
        "ra": result.ra_center,
        "dec": result.dec_center,
        "scale": result.pixel_scale,
        "rotation": result.rotation,
        "wcs_header": result.wcs_header,
    }
