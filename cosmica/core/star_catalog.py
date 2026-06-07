"""Star Catalog — query Vizier for reference stars (Gaia DR3).

Provides access to star catalogs for photometry and plate solving.
Uses CDS TAP/Vizier for catalog queries, ASTAP CLI or astrometry.net for solving.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

CATALOG_CACHE_DIR = Path.home() / ".cosmica" / "catalogs"
CATALOG_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class StarCatalogEntry:
    """Single star from a catalog query."""

    ra_deg: float
    dec_deg: float
    g_mag: float | None
    bp_mag: float | None
    rp_mag: float | None
    source_id: str | None


def query_gaia_dr3(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float = 0.5,
    max_stars: int = 500,
) -> list[StarCatalogEntry]:
    """Query Gaia DR3 stars around a given position via CDS TAP.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Field center in degrees (J2000).
    radius_deg : float
        Search radius in degrees (max 2.0).
    max_stars : int
        Maximum stars to return.
    """
    radius_deg = min(radius_deg, 2.0)

    # Check cache
    cache_key = f"gaia_{ra_deg:.4f}_{dec_deg:.4f}_{radius_deg:.3f}_{max_stars}.json"
    cache_path = CATALOG_CACHE_DIR / cache_key
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                raw = json.load(f)
            stars = [StarCatalogEntry(**s) for s in raw]
            log.info("Gaia DR3 cache hit: %d stars", len(stars))
            return stars
        except Exception:
            cache_path.unlink(missing_ok=True)

    log.info("Querying Gaia DR3: RA=%.4f Dec=%.4f r=%.3f°", ra_deg, dec_deg, radius_deg)

    stars = _query_vizier_tap(ra_deg, dec_deg, radius_deg, max_stars)

    if stars:
        try:
            with open(cache_path, "w") as f:
                json.dump([s.__dict__ for s in stars], f)
        except Exception:
            log.warning("Could not write Gaia cache to %s", cache_path)

    return stars


def _query_vizier_tap(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    max_stars: int,
) -> list[StarCatalogEntry]:
    """Query Gaia DR3 via CDS TAPVizieR ADQL endpoint."""
    import xml.etree.ElementTree as ET

    adql = (
        f'SELECT TOP {max_stars} RAJ2000,DEJ2000,Gmag,BPmag,RPmag,Source '
        f'FROM "I/355/gaiadr3" '
        f"WHERE CONTAINS(POINT('ICRS',RAJ2000,DEJ2000),"
        f"CIRCLE('ICRS',{ra_deg},{dec_deg},{radius_deg}))=1 "
        f"AND Gmag IS NOT NULL "
        f"ORDER BY Gmag ASC"
    )

    params = urllib.parse.urlencode({
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "votable",
        "QUERY": adql,
    })

    url = f"https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync?{params}"

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Cosmica/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read().decode("utf-8")
            break
        except Exception as e:
            if attempt < 2:
                delay = 1.0 * (2 ** attempt)
                log.debug("Vizier TAP query failed (%s), retrying in %.0fs…", e, delay)
                time.sleep(delay)
                continue
            log.error("Vizier TAP query failed after 3 attempts: %s", e)
            return []

    stars: list[StarCatalogEntry] = []
    try:
        root = ET.fromstring(data)
        ns = {"v": "http://www.ivoa.net/xml/VOTable/v1.3"}
        # Try with namespace first, then without
        rows = root.findall(".//v:TR", ns) or root.findall(".//TR")
        for row in rows:
            cells = row.findall("v:TD", ns) or row.findall("TD")
            if len(cells) < 5:
                continue
            try:
                ra = float(cells[0].text or 0)
                dec = float(cells[1].text or 0)
                g_mag = float(cells[2].text) if cells[2].text else None
                bp_mag = float(cells[3].text) if cells[3].text else None
                rp_mag = float(cells[4].text) if cells[4].text else None
                source_id = cells[5].text if len(cells) > 5 else None
                stars.append(StarCatalogEntry(
                    ra_deg=ra, dec_deg=dec,
                    g_mag=g_mag, bp_mag=bp_mag, rp_mag=rp_mag,
                    source_id=source_id,
                ))
            except (ValueError, TypeError):
                continue
    except ET.ParseError as e:
        log.error("Failed to parse Vizier VOTable: %s", e)

    log.info("Gaia DR3: retrieved %d stars", len(stars))
    return stars


def plate_solve_astap(
    image_path: Path,
    ra_hint: float | None = None,
    dec_hint: float | None = None,
    scale_hint: float | None = None,
) -> dict | None:
    """Plate-solve using ASTAP CLI.

    ASTAP writes a `<filename>.wcs` file with the plate solution.
    Requires astap_cli in PATH.

    Returns
    -------
    dict with keys: ra, dec, scale, rotation, wcs_header
    or None if failed.
    """
    astap_path = shutil.which("astap_cli") or shutil.which("astap")
    if astap_path is None:
        log.warning("ASTAP not found in PATH (install astap or astap_cli)")
        return None

    cmd = [astap_path, "-f", str(image_path), "-update"]

    if ra_hint is not None and dec_hint is not None:
        cmd += ["-ra", f"{ra_hint / 15:.6f}", "-dec", f"{dec_hint:.6f}", "-r", "30"]

    if scale_hint is not None:
        cmd += ["-scale", f"{scale_hint:.4f}"]

    log.info("Running ASTAP: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired:
        log.error("ASTAP timed out after 120s")
        return None
    except Exception as e:
        log.error("ASTAP launch error: %s", e)
        return None

    # ASTAP writes <image>.wcs alongside the input file
    wcs_path = image_path.with_suffix(".wcs")
    ini_path = image_path.with_suffix(".ini")

    if wcs_path.exists():
        result = _parse_wcs_fits(wcs_path)
        if result:
            return result

    # Fallback: parse .ini text file ASTAP also writes
    if ini_path.exists():
        result = _parse_astap_ini(ini_path)
        if result:
            return result

    log.error("ASTAP solve failed (no .wcs output). stderr: %s", proc.stderr[:500])
    return None


def _parse_wcs_fits(wcs_path: Path) -> dict | None:
    """Parse a FITS WCS file written by ASTAP using astropy."""
    try:
        from astropy.io import fits as afits
        from astropy.wcs import WCS

        with afits.open(wcs_path) as hdul:
            header = hdul[0].header
            wcs = WCS(header)

        # Get image center coordinates
        ny = header.get("NAXIS2", 1000)
        nx = header.get("NAXIS1", 1000)
        sky = wcs.pixel_to_world(nx / 2, ny / 2)
        ra_center = float(sky.ra.deg)
        dec_center = float(sky.dec.deg)

        # Pixel scale from CD matrix or CDELT
        try:
            import numpy as np
            cd = wcs.pixel_scale_matrix
            scale_deg = float(np.sqrt(abs(np.linalg.det(cd))))
            scale_arcsec = scale_deg * 3600
        except Exception:
            scale_arcsec = abs(float(header.get("CDELT1", 0.001))) * 3600

        rotation = float(header.get("CROTA2", 0.0))

        log.info("ASTAP WCS: RA=%.4f Dec=%.4f scale=%.3f\"/px", ra_center, dec_center, scale_arcsec)
        return {
            "ra": ra_center,
            "dec": dec_center,
            "scale": scale_arcsec,
            "rotation": rotation,
            "wcs_header": dict(header),
        }
    except Exception as e:
        log.error("WCS FITS parse error: %s", e)
        return None


def _parse_astap_ini(ini_path: Path) -> dict | None:
    """Parse ASTAP .ini text output as fallback."""
    result = {"ra": None, "dec": None, "scale": None, "rotation": None, "wcs_header": {}}
    try:
        text = ini_path.read_text()
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("PLTSOLVD=T"):
                pass
            elif "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                try:
                    if key in ("CRVAL1", "RA"):
                        result["ra"] = float(val)
                    elif key in ("CRVAL2", "DEC"):
                        result["dec"] = float(val)
                    elif key in ("CDELT1",):
                        result["scale"] = abs(float(val)) * 3600
                    elif key == "CROTA2":
                        result["rotation"] = float(val)
                except ValueError:
                    pass
    except Exception as e:
        log.error("ASTAP ini parse error: %s", e)
        return None

    if result["ra"] is not None and result["dec"] is not None:
        return result
    return None


def plate_solve_astrometry_net(
    image_path: Path,
    api_key: str,
    ra_hint: float | None = None,
    dec_hint: float | None = None,
    scale_hint: float | None = None,
    timeout: int = 120,
) -> dict | None:
    """Plate-solve via nova.astrometry.net HTTP API.

    Parameters
    ----------
    image_path : Path
        FITS file to upload.
    api_key : str
        Your nova.astrometry.net API key.
    timeout : int
        Max seconds to wait for solve (default 120).
    """
    import io
    BASE = "http://nova.astrometry.net/api"

    def _request_with_retry(url, data=None, headers=None, max_retries=3, base_delay=1.0, timeout=30):
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, data=data, headers=headers or {})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return json.loads(r.read())
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

    def _post_json(url: str, data: dict) -> dict:
        payload = ("request-json=" + urllib.parse.quote(json.dumps(data))).encode()
        return _request_with_retry(url, data=payload, headers={"Content-Type": "application/x-www-form-urlencoded"})

    def _get_json(url: str) -> dict:
        return _request_with_retry(url, headers={"User-Agent": "Cosmica/1.0"})

    # 1. Login
    log.info("Logging in to astrometry.net...")
    try:
        resp = _post_json(f"{BASE}/login", {"apikey": api_key})
        session = resp.get("session")
        if not session:
            log.error("astrometry.net login failed: %s", resp)
            return None
    except Exception as e:
        log.error("astrometry.net login error: %s", e)
        return None

    # 2. Upload file with multipart/form-data
    log.info("Uploading %s to astrometry.net...", image_path.name)
    try:
        boundary = "CosmicaBoundary"
        body = io.BytesIO()

        # JSON options part
        options: dict = {"session": session, "publicly_visible": "n",
                         "allow_modifications": "n", "allow_commercial_use": "n"}
        if ra_hint is not None and dec_hint is not None:
            options["center_ra"] = ra_hint
            options["center_dec"] = dec_hint
            options["radius"] = 5.0
        if scale_hint is not None:
            options["scale_units"] = "arcsecperpix"
            options["scale_est"] = scale_hint
            options["scale_err"] = 20.0

        def _write(b: bytes):
            body.write(b)

        _write(f"--{boundary}\r\n".encode())
        _write(b'Content-Disposition: form-data; name="request-json"\r\n\r\n')
        _write(json.dumps(options).encode())
        _write(f"\r\n--{boundary}\r\n".encode())
        _write(f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'.encode())
        _write(b"Content-Type: application/octet-stream\r\n\r\n")
        _write(image_path.read_bytes())
        _write(f"\r\n--{boundary}--\r\n".encode())

        upload_resp = _request_with_retry(
            f"{BASE}/upload",
            data=body.getvalue(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            timeout=60,
        )

        subid = upload_resp.get("subid")
        if not subid:
            log.error("astrometry.net upload failed: %s", upload_resp)
            return None
        log.info("Submission ID: %s", subid)
    except Exception as e:
        log.error("astrometry.net upload error: %s", e)
        return None

    # 3. Poll for job completion
    deadline = time.time() + timeout
    job_id = None

    while time.time() < deadline:
        time.sleep(5)
        try:
            sub_info = _get_json(f"{BASE}/submissions/{subid}")
            jobs = sub_info.get("jobs", [])
            if jobs and jobs[0]:
                job_id = jobs[0]
                log.info("Job ID: %s", job_id)
                break
        except Exception:
            log.debug("astrometry.net job poll error, continuing")

    if job_id is None:
        log.error("astrometry.net: no job started within timeout")
        return None

    while time.time() < deadline:
        time.sleep(5)
        try:
            job_info = _get_json(f"{BASE}/jobs/{job_id}/info")
            status = job_info.get("status")
            if status == "success":
                break
            elif status == "failure":
                log.error("astrometry.net solve failed")
                return None
        except Exception:
            log.debug("astrometry.net status poll error, continuing")
    else:
        log.error("astrometry.net timed out waiting for solve")
        return None

    # 4. Download and parse WCS FITS
    log.info("Downloading WCS from astrometry.net job %s...", job_id)
    try:
        wcs_url = f"http://nova.astrometry.net/wcs_file/{job_id}"
        for attempt in range(3):
            try:
                req = urllib.request.Request(wcs_url, headers={"User-Agent": "Cosmica/1.0"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    wcs_data = r.read()
                break
            except Exception:
                if attempt < 2:
                    time.sleep(1.0 * (2 ** attempt))
                    continue
                raise

        # Write to temp file and parse
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wcs", delete=False) as tmp:
            tmp.write(wcs_data)
            tmp_path = Path(tmp.name)

        result = _parse_wcs_fits(tmp_path)
        tmp_path.unlink(missing_ok=True)
        return result
    except Exception as e:
        log.error("astrometry.net WCS download error: %s", e)
        return None


def plate_solve_auto(
    image_path: Path,
    api_key: str | None = None,
    ra_hint: float | None = None,
    dec_hint: float | None = None,
    scale_hint: float | None = None,
) -> dict | None:
    """Try ASTAP first, fall back to astrometry.net.

    Parameters
    ----------
    api_key : str, optional
        astrometry.net API key (needed for fallback).
    """
    # Try ASTAP first (offline, fast)
    if shutil.which("astap_cli") or shutil.which("astap"):
        result = plate_solve_astap(image_path, ra_hint, dec_hint, scale_hint)
        if result:
            return result
        log.warning("ASTAP solve failed, trying astrometry.net...")

    # Fall back to astrometry.net
    if api_key:
        return plate_solve_astrometry_net(
            image_path, api_key, ra_hint, dec_hint, scale_hint
        )

    log.error("No plate solver available (ASTAP not found, no astrometry.net API key)")
    return None
