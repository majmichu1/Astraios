"""Plate Solving — determine WCS coordinates from star positions.

Uses astropy WCS (BSD) for coordinate transforms and local solving
via triangle matching against a reference catalog.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from astraios.core.star_detection import detect_stars

log = logging.getLogger(__name__)


@dataclass
class PlateSolveResult:
    """Result from plate solving."""

    success: bool
    ra_center: float = 0.0  # Right Ascension in degrees
    dec_center: float = 0.0  # Declination in degrees
    pixel_scale: float = 0.0  # arcsec/pixel
    rotation: float = 0.0  # field rotation in degrees
    n_stars_matched: int = 0
    wcs_header: dict | None = None


@dataclass
class PlateSolveParams:
    """Parameters for plate solving."""

    # Approximate field center (if known)
    ra_hint: float | None = None
    dec_hint: float | None = None
    # Approximate pixel scale (arcsec/pixel)
    scale_hint: float | None = None
    scale_tolerance: float = 0.2  # +/- fraction of scale_hint
    max_stars: int = 100
    # Search radius in degrees (if hint provided)
    search_radius: float = 5.0


def plate_solve(
    image: np.ndarray,
    params: PlateSolveParams | None = None,
) -> PlateSolveResult:
    """Attempt to plate-solve an image using detected star positions.

    This performs local solving using triangle matching. For more robust
    solving, use the astrometry.net API integration.

    Parameters
    ----------
    image : ndarray
        Image data, shape (H, W) or (C, H, W), float32 in [0, 1].
    params : PlateSolveParams, optional
        Solving parameters.

    Returns
    -------
    PlateSolveResult
        Solving result with WCS information if successful.
    """
    if params is None:
        params = PlateSolveParams()

    sf = detect_stars(image, max_stars=params.max_stars)
    if len(sf) < 4:
        log.warning("Too few stars detected for plate solving (%d)", len(sf))
        return PlateSolveResult(success=False)

    positions = sf.positions  # Nx2 (x, y)

    # Build triangle index from detected stars
    triangles = _build_triangle_index(positions[:min(30, len(positions))])

    if not triangles:
        return PlateSolveResult(success=False)

    # If no hint is provided, we can only compute relative geometry
    if params.scale_hint is None:
        log.info("No scale hint provided, computing relative geometry only")
        result = _estimate_field_geometry(positions, sf.image_width, sf.image_height)
        return result

    raise NotImplementedError(
        "Local plate_solve with scale_hint requires a reference catalog. "
        "Use plate_solve_astap / plate_solve_astrometry_net instead."
    )


def _write_temp_mono_fits(image: np.ndarray, params: PlateSolveParams) -> "object | None":
    """Write a mono uint16 FITS for an external solver. Returns a Path or None.

    Centralises the array→FITS conversion shared by the ASTAP and
    astrometry.net adapters so the two paths can't drift apart.
    """
    import tempfile
    from pathlib import Path

    try:
        from astropy.io import fits as _fits
    except ImportError:
        log.warning("astropy required for plate solving")
        return None

    mono = image.mean(axis=0) if image.ndim == 3 else image
    mono_u16 = (np.clip(mono, 0, 1) * 65535).astype(np.uint16)
    hdu = _fits.PrimaryHDU(mono_u16)
    if params.ra_hint is not None:
        hdu.header["OBJCTRA"] = params.ra_hint
    if params.dec_hint is not None:
        hdu.header["OBJCTDEC"] = params.dec_hint

    fd = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    fd.close()
    path = Path(fd.name)
    hdu.writeto(str(path), overwrite=True)
    return path


def _result_from_solver_dict(d: dict | None, image: np.ndarray) -> PlateSolveResult:
    """Convert a :mod:`astraios.core.star_catalog` solver dict to a result.

    star_catalog returns ``{ra, dec, scale, rotation, wcs_header}`` where
    ``wcs_header`` is the full solved FITS header. We feed that header through
    :func:`_parse_wcs_header` so the resulting ``wcs_header`` is in the
    canonical ``ra_center`` format that plate-solve consumers expect (see
    :mod:`astraios.core.wcs`). Falls back to the flat fields if no FITS header
    is present.
    """
    if not d:
        return PlateSolveResult(success=False)

    fits_header = d.get("wcs_header") or {}
    if fits_header and ("CRVAL1" in fits_header or "CD1_1" in fits_header):
        return _parse_wcs_header(dict(fits_header), image)

    ra = float(d.get("ra") or 0.0)
    dec = float(d.get("dec") or 0.0)
    scale = float(d.get("scale") or 0.0)
    rotation = float(d.get("rotation") or 0.0)
    h = image.shape[-2] if image.ndim >= 2 else 1
    w = image.shape[-1] if image.ndim >= 2 else 1
    scale_deg = scale / 3600.0
    wcs_dict = {
        "ra_center": ra, "dec_center": dec, "scale": scale, "rotation": rotation,
        "width": w, "height": h,
        "cd11": scale_deg, "cd12": 0.0, "cd21": 0.0, "cd22": scale_deg,
        "crpix1": w / 2, "crpix2": h / 2,
    }
    return PlateSolveResult(
        success=(scale > 0 or ra != 0.0 or dec != 0.0),
        ra_center=ra, dec_center=dec, pixel_scale=scale,
        rotation=rotation, wcs_header=wcs_dict,
    )


def plate_solve_astap(
    image: np.ndarray,
    params: PlateSolveParams | None = None,
    progress=None,
) -> PlateSolveResult:
    """Plate-solve an array using ASTAP CLI (offline, fast).

    Thin adapter over :func:`astraios.core.star_catalog.plate_solve_astap` —
    the single source of truth for the ASTAP subprocess. Writes the array to a
    temporary FITS, delegates, then converts the WCS dict to a
    :class:`PlateSolveResult`.

    ASTAP: https://www.hnsky.org/astap.htm  (free, GPL)
    """
    from astraios.core import star_catalog

    if params is None:
        params = PlateSolveParams()
    if progress:
        progress(0.1, "Running ASTAP plate solver…")

    path = _write_temp_mono_fits(image, params)
    if path is None:
        return PlateSolveResult(success=False)
    try:
        d = star_catalog.plate_solve_astap(
            path, params.ra_hint, params.dec_hint, params.scale_hint
        )
    finally:
        path.unlink(missing_ok=True)
        # ASTAP writes sidecar files next to the input
        for suffix in (".wcs", ".ini"):
            path.with_suffix(suffix).unlink(missing_ok=True)
    if progress:
        progress(1.0, "ASTAP solve complete" if d else "ASTAP solve failed")
    return _result_from_solver_dict(d, image)


#: Human-readable labels for the solver backends this module dispatches to,
#: for UI solver-choice combos that want a name -> label mapping.
SOLVER_LABELS: dict[str, str] = {
    "auto": "Auto",
    "gaia": "GAIA (offline)",
    "astap": "ASTAP",
    "astrometry_net": "Astrometry.net",
}


def _gaia_catalog_ready(catalog_dir: Path | str | None = None) -> bool:
    """Cheap check for whether any local Gaia catalog band is installed.

    Used to gate the offline Gaia attempt in :func:`plate_solve_auto` without
    the cost of opening every SQLite file — a missing/empty catalog directory
    must leave ``plate_solve_auto``'s behavior identical to before this
    backend existed.
    """
    try:
        from astraios.core.gaia_catalog import installed_files

        return bool(installed_files(catalog_dir))
    except Exception:  # noqa: BLE001 - never let a presence check crash the solve
        return False


def plate_solve_gaia_offline(
    image: np.ndarray,
    params: PlateSolveParams | None = None,
    catalog_dir: Path | str | None = None,
    progress=None,
) -> PlateSolveResult:
    """Offline plate-solve against a local Gaia DR3 catalog (no internet).

    Thin adapter over :func:`astraios.core.gaia_solver.plate_solve_gaia` —
    requires at least one Gaia catalog band downloaded locally (see
    :mod:`astraios.core.gaia_catalog`) *and* an approximate RA/Dec/pixel-scale
    hint (``params.ra_hint`` / ``dec_hint`` / ``scale_hint``); this solver
    refines a seed, it does not blind-solve. Never raises for expected
    failure modes (missing catalog, no hint, no match) — returns
    ``PlateSolveResult(success=False)`` instead, so callers can fall through
    to another backend.
    """
    if params is None:
        params = PlateSolveParams()
    if params.ra_hint is None or params.dec_hint is None or not params.scale_hint:
        return PlateSolveResult(success=False)

    if progress:
        progress(0.1, "Solving against local Gaia catalog…")

    try:
        from astraios.core.gaia_solver import GaiaSolveParams
        from astraios.core.gaia_solver import plate_solve_gaia as _gaia_solve

        gp = GaiaSolveParams(catalog_dir=Path(catalog_dir) if catalog_dir else None)
        d = _gaia_solve(image, params.ra_hint, params.dec_hint, params.scale_hint, gp)
    except Exception as exc:  # noqa: BLE001 - fall through to other backends on any failure
        log.warning("Gaia offline solve raised, falling through: %s", exc)
        d = None

    if progress:
        progress(1.0, "Gaia offline solve complete" if d else "Gaia offline solve failed")
    return _result_from_solver_dict(d, image)


def _parse_wcs_header(header: dict, image: np.ndarray) -> PlateSolveResult:
    """Extract RA/Dec/scale/rotation from a solved WCS FITS header."""
    ra = float(header.get("CRVAL1", 0.0))
    dec = float(header.get("CRVAL2", 0.0))

    # Pixel scale from CD matrix or CDELT
    cd11 = float(header.get("CD1_1", header.get("CDELT1", 0.0)))
    cd12 = float(header.get("CD1_2", 0.0))
    cd21 = float(header.get("CD2_1", 0.0))
    cd22 = float(header.get("CD2_2", header.get("CDELT2", 0.0)))

    scale_x = np.sqrt(cd11**2 + cd21**2) * 3600.0  # arcsec/pixel
    scale_y = np.sqrt(cd12**2 + cd22**2) * 3600.0
    scale = (abs(scale_x) + abs(scale_y)) / 2.0
    rotation = float(np.degrees(np.arctan2(cd12, cd11)))

    h = image.shape[-2] if image.ndim >= 2 else 1
    w = image.shape[-1] if image.ndim >= 2 else 1

    wcs_dict = {
        "ra_center": ra, "dec_center": dec,
        "scale": scale, "rotation": rotation,
        "width": w, "height": h,
        "cd11": cd11, "cd12": cd12, "cd21": cd21, "cd22": cd22,
        "crpix1": float(header.get("CRPIX1", w / 2)),
        "crpix2": float(header.get("CRPIX2", h / 2)),
    }

    return PlateSolveResult(
        success=True,
        ra_center=ra,
        dec_center=dec,
        pixel_scale=scale,
        rotation=rotation,
        wcs_header=wcs_dict,
    )


def plate_solve_astrometry_net(
    image: np.ndarray,
    api_key: str | None = None,
    params: PlateSolveParams | None = None,
    progress=None,
) -> PlateSolveResult:
    """Plate-solve an array via nova.astrometry.net.

    Thin adapter over :func:`astraios.core.star_catalog.plate_solve_astrometry_net`
    — the single source of truth for the astrometry.net upload/poll logic.
    Requires an internet connection and an API key from nova.astrometry.net.
    """
    from astraios.core import star_catalog

    if params is None:
        params = PlateSolveParams()
    if not api_key:
        log.warning("No astrometry.net API key provided")
        return PlateSolveResult(success=False)
    if progress:
        progress(0.1, "Uploading image to astrometry.net…")

    path = _write_temp_mono_fits(image, params)
    if path is None:
        return PlateSolveResult(success=False)
    try:
        d = star_catalog.plate_solve_astrometry_net(
            path, api_key, params.ra_hint, params.dec_hint, params.scale_hint
        )
    finally:
        path.unlink(missing_ok=True)
    if progress:
        progress(1.0, "astrometry.net solve complete" if d else "astrometry.net solve failed")
    return _result_from_solver_dict(d, image)


def plate_solve_auto(
    image: np.ndarray,
    params: PlateSolveParams | None = None,
    api_key: str | None = None,
    progress=None,
    catalog_dir: Path | str | None = None,
) -> PlateSolveResult:
    """Try local Gaia (offline) first, then ASTAP, then astrometry.net.

    The Gaia attempt only fires when both a local catalog is installed
    (cheap presence check, see :func:`_gaia_catalog_ready`) and an RA/Dec/
    scale hint is available in ``params`` — with no catalog installed (the
    default, out-of-the-box state) this is a no-op and behavior is identical
    to before this backend existed. On :class:`~astraios.core.gaia_catalog.GaiaCatalogNotFoundError`
    or any other solve failure, falls through to ASTAP/astrometry.net exactly
    as before.
    """
    if (
        params is not None
        and params.ra_hint is not None
        and params.dec_hint is not None
        and params.scale_hint
        and _gaia_catalog_ready(catalog_dir)
    ):
        log.info("Using local Gaia catalog for plate solving")
        result = plate_solve_gaia_offline(image, params, catalog_dir=catalog_dir, progress=progress)
        if result.success:
            return result
        log.info("Gaia offline solve failed or found no match, trying ASTAP…")

    import shutil
    astap = shutil.which("astap_cli") or shutil.which("astap")
    if astap:
        log.info("Using ASTAP for plate solving")
        result = plate_solve_astap(image, params, progress=progress)
        if result.success:
            return result
        log.info("ASTAP failed, trying astrometry.net…")
    if api_key:
        return plate_solve_astrometry_net(image, api_key, params, progress=progress)
    log.warning("No solver available — install ASTAP or provide an astrometry.net API key")
    return PlateSolveResult(success=False)


def _build_triangle_index(
    points: np.ndarray,
) -> list[tuple[tuple[int, int, int], tuple[float, float]]]:
    """Build triangle descriptors from star positions for matching.

    Each triangle is described by its two invariant ratios.
    """
    n = len(points)
    if n < 3:
        return []

    triangles = []
    for i in range(min(n, 15)):
        for j in range(i + 1, min(n, 15)):
            for k in range(j + 1, min(n, 15)):
                p = points[[i, j, k]]
                # Compute pairwise distances
                d01 = np.linalg.norm(p[0] - p[1])
                d02 = np.linalg.norm(p[0] - p[2])
                d12 = np.linalg.norm(p[1] - p[2])

                sides = sorted([d01, d02, d12])
                if sides[2] < 1e-6:
                    continue

                # Two invariant ratios
                r1 = sides[0] / sides[2]
                r2 = sides[1] / sides[2]
                triangles.append(((i, j, k), (r1, r2)))

    return triangles


def _estimate_field_geometry(
    positions: np.ndarray,
    width: int,
    height: int,
) -> PlateSolveResult:
    """Estimate basic field geometry from star positions."""
    if len(positions) < 3:
        return PlateSolveResult(success=False)

    # Compute mean position
    center_x = np.mean(positions[:, 0])
    center_y = np.mean(positions[:, 1])

    # Estimate field rotation from star distribution
    centered = positions - np.array([center_x, center_y])
    if len(centered) > 3:
        # PCA for orientation
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        rotation = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    else:
        rotation = 0.0

    return PlateSolveResult(
        success=True,
        n_stars_matched=len(positions),
        rotation=rotation,
    )
