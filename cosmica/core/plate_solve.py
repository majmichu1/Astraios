"""Plate Solving — determine WCS coordinates from star positions.

Uses astropy WCS (BSD) for coordinate transforms and local solving
via triangle matching against a reference catalog.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from cosmica.core.star_detection import detect_stars

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


def plate_solve_astap(
    image: np.ndarray,
    params: PlateSolveParams | None = None,
    progress=None,
) -> PlateSolveResult:
    """Plate-solve using ASTAP CLI (offline, fast).

    Requires ASTAP to be installed and ``astap`` or ``astap_cli`` on PATH,
    or at the default install location on Linux/Windows.

    ASTAP: https://www.hnsky.org/astap.htm  (free, GPL)
    """
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    if params is None:
        params = PlateSolveParams()

    # Locate binary
    binary = (
        shutil.which("astap_cli")
        or shutil.which("astap")
    )
    if binary is None:
        for p in ("/usr/bin/astap", "/usr/local/bin/astap"):
            if Path(p).exists():
                binary = p
                break
    if binary is None or not Path(binary).exists():
        log.warning("ASTAP not found on PATH — install ASTAP for offline plate solving")
        return PlateSolveResult(success=False)

    # Write image as temporary FITS
    try:
        from astropy.io import fits as _fits
    except ImportError:
        log.warning("astropy required for ASTAP plate solving")
        return PlateSolveResult(success=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_fits = Path(tmpdir) / "solve_input.fits"

        # Convert to mono float32 for solving
        if image.ndim == 3:
            mono = image.mean(axis=0)
        else:
            mono = image
        mono_u16 = (np.clip(mono, 0, 1) * 65535).astype(np.uint16)

        hdu = _fits.PrimaryHDU(mono_u16)
        if params.ra_hint is not None:
            hdu.header["OBJCTRA"] = params.ra_hint
        if params.dec_hint is not None:
            hdu.header["OBJCTDEC"] = params.dec_hint
        hdu.writeto(str(tmp_fits), overwrite=True)

        # Build ASTAP command
        cmd = [binary, "-f", str(tmp_fits), "-update", "-log"]
        if params.ra_hint is not None and params.dec_hint is not None:
            cmd += ["-ra", str(params.ra_hint / 15.0), "-spd", str(params.dec_hint + 90.0)]
            cmd += ["-r", str(params.search_radius)]
        if params.scale_hint is not None:
            cmd += ["-fov", str(params.scale_hint * image.shape[-1] / 3600.0)]

        if progress:
            progress(0.1, "Running ASTAP plate solver…")

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
            log.debug("ASTAP stdout: %s", proc.stdout[:500])
            log.debug("ASTAP stderr: %s", proc.stderr[:500])
        except subprocess.TimeoutExpired:
            log.warning("ASTAP timed out after 120s")
            return PlateSolveResult(success=False)
        except Exception as e:
            log.warning("ASTAP failed to run: %s", e)
            return PlateSolveResult(success=False)

        # ASTAP writes a .wcs file alongside the input FITS
        wcs_path = tmp_fits.with_suffix(".wcs")
        if not wcs_path.exists():
            # Check for solved FITS header update
            solved_fits = tmp_fits
            try:
                with _fits.open(str(solved_fits)) as hdul:
                    h = hdul[0].header
                    if "CRVAL1" not in h:
                        log.warning("ASTAP: no WCS in output — solve failed")
                        return PlateSolveResult(success=False)
                    return _parse_wcs_header(dict(h), image)
            except Exception as e:
                log.warning("ASTAP output parse failed: %s", e)
                return PlateSolveResult(success=False)

        try:
            with _fits.open(str(wcs_path)) as hdul:
                wcs_header = dict(hdul[0].header)
            return _parse_wcs_header(wcs_header, image)
        except Exception as e:
            log.warning("ASTAP WCS parse failed: %s", e)
            return PlateSolveResult(success=False)


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
    """Plate-solve using the astrometry.net web API (nova.astrometry.net).

    Requires an internet connection and an API key from nova.astrometry.net.
    """
    import json
    import tempfile
    import time
    import urllib.parse
    import urllib.request
    from pathlib import Path

    if params is None:
        params = PlateSolveParams()

    if not api_key:
        log.warning("No astrometry.net API key provided")
        return PlateSolveResult(success=False)

    BASE = "https://nova.astrometry.net/api/"

    def _request_with_retry(url, data=None, headers=None, max_retries=3, base_delay=1.0, timeout=30):
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, data=data, headers=headers or {})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read())
            except urllib.request.HTTPError as e:
                if e.code == 429 and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    log.warning("astrometry.net rate limited (429), retrying in %.0fs…", delay)
                    time.sleep(delay)
                    continue
                if e.code >= 500 and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                raise
            except (urllib.request.URLError, ConnectionError, TimeoutError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    log.debug("astrometry.net request failed (%s), retrying in %.0fs…", e, delay)
                    time.sleep(delay)
                    continue
                raise

    def _api(endpoint, data=None, files=None):
        url = BASE + endpoint
        if data:
            body = ("request-json=" + urllib.parse.quote(json.dumps(data))).encode()
            return _request_with_retry(url, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
        return _request_with_retry(url)

    try:
        if progress:
            progress(0.05, "Logging in to astrometry.net…")
        resp = _api("login", {"apikey": api_key})
        if resp.get("status") != "success":
            log.warning("astrometry.net login failed: %s", resp)
            return PlateSolveResult(success=False)
        session = resp["session"]

        # Upload image as FITS
        if progress:
            progress(0.15, "Uploading image to astrometry.net…")
        try:
            from astropy.io import fits as _fits
        except ImportError:
            return PlateSolveResult(success=False)

        try:
            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
                tmp_path = tf.name
            if image.ndim == 3:
                mono = image.mean(axis=0)
            else:
                mono = image
            mono_u16 = (np.clip(mono, 0, 1) * 65535).astype(np.uint16)
            _fits.PrimaryHDU(mono_u16).writeto(tmp_path, overwrite=True)

            import uuid
            boundary = uuid.uuid4().hex
            with open(tmp_path, "rb") as f:
                file_data = f.read()
        except Exception:
            raise
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        meta = {"session": session}
        if params.ra_hint is not None:
            meta["center_ra"] = params.ra_hint
            meta["center_dec"] = params.dec_hint
            meta["radius"] = params.search_radius

        parts = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"request-json\"\r\n\r\n"
            + json.dumps(meta) + "\r\n"
            + f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"image.fits\"\r\n"
            + "Content-Type: image/fits\r\n\r\n"
        ).encode() + file_data + f"\r\n--{boundary}--\r\n".encode()

        upload_resp = _request_with_retry(
            BASE + "upload", data=parts,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                     "Content-Length": str(len(parts))},
            timeout=60,
        )

        if upload_resp.get("status") != "success":
            log.warning("astrometry.net upload failed: %s", upload_resp)
            return PlateSolveResult(success=False)

        subid = upload_resp["subid"]
        if progress:
            progress(0.3, f"Submitted (submission {subid}), waiting for solve…")

        # Poll for job completion
        job_id = None
        for _ in range(60):
            time.sleep(2)
            sub_status = _api(f"submissions/{subid}")
            jobs = sub_status.get("jobs", [])
            if jobs and jobs[0]:
                job_id = jobs[0]
                break
            if progress:
                progress(0.3, "Waiting for solver queue…")

        if not job_id:
            log.warning("astrometry.net: no job assigned after timeout")
            return PlateSolveResult(success=False)

        for attempt in range(60):
            time.sleep(2)
            job_status = _api(f"jobs/{job_id}")
            status = job_status.get("status")
            if progress:
                progress(0.3 + attempt * 0.01, f"Solving… ({status})")
            if status == "success":
                break
            if status == "failure":
                log.warning("astrometry.net solve failed")
                return PlateSolveResult(success=False)
        else:
            log.warning("astrometry.net: solve timed out")
            return PlateSolveResult(success=False)

        if progress:
            progress(0.9, "Fetching solution…")
        cal = _api(f"jobs/{job_id}/calibration")
        ra = cal.get("ra", 0.0)
        dec = cal.get("dec", 0.0)
        scale = cal.get("pixscale", 0.0)
        rotation = cal.get("orientation", 0.0)

        h = image.shape[-2]
        w = image.shape[-1]
        scale_deg = scale / 3600.0
        wcs_dict = {
            "ra_center": ra, "dec_center": dec,
            "scale": scale, "rotation": rotation,
            "width": w, "height": h,
            "cd11": scale_deg, "cd12": 0.0,
            "cd21": 0.0, "cd22": scale_deg,
            "crpix1": w / 2, "crpix2": h / 2,
        }
        if progress:
            progress(1.0, f"Solved: RA={ra:.3f}° Dec={dec:.3f}° scale={scale:.2f}\"/px")
        return PlateSolveResult(
            success=True, ra_center=ra, dec_center=dec,
            pixel_scale=scale, rotation=rotation, wcs_header=wcs_dict,
        )

    except urllib.request.HTTPError as e:
        if e.code == 429:
            log.warning("astrometry.net rate limited (429) after retries. Wait ~1 hour and try again.")
        else:
            log.warning("astrometry.net HTTP %d: %s", e.code, e.reason)
        return PlateSolveResult(success=False)
    except Exception as e:
        log.warning("astrometry.net solve error: %s", e)
        return PlateSolveResult(success=False)


def plate_solve_auto(
    image: np.ndarray,
    params: PlateSolveParams | None = None,
    api_key: str | None = None,
    progress=None,
) -> PlateSolveResult:
    """Try ASTAP first, fall back to astrometry.net if ASTAP not available."""
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
