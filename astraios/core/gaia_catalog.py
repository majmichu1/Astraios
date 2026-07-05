"""Gaia DR3 offline catalog — local SQLite catalog reader + downloader.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

SASpro ships bulk-downloaded Gaia DR3 SQLite files ("Gaia XP Spectral
Library") split into magnitude bands. Each file has a ``sources`` table
(source_id, ra, dec, phot_g_mean_mag, bp_rp, parallax, pmra, pmdec,
has_xp_spectrum) and a ``spectra`` table with compressed XP spectra. This
module ports only the ``sources`` reading/cone-query path, which is all the
offline GAIA plate solver in :mod:`astraios.core.gaia_solver` needs — the
spectra themselves are unrelated to astrometry and are not ported here.

Files are downloaded from the same Backblaze bucket SASpro uses
(``LIBRARY_DOWNLOAD_BASE`` there), so users who already have SASpro's
``gaia_xp_*.sqlite`` files installed can point Astraios at the same
directory and reuse them directly — no re-download needed.

This module is pure CPU/numpy + sqlite3 — catalog lookups and spatial
cone queries are not GPU workloads, so it intentionally does not touch
``device_manager``.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

# ── Download source ──────────────────────────────────────────────────────
# Same public Backblaze bucket SASpro's gaia_database.py uses, so the two
# apps can share downloaded DB files interchangeably.
GAIA_DOWNLOAD_BASE_URL = "https://f005.backblazeb2.com/file/setiastro-gaia/"

#: Env var to override the default catalog directory without passing a path
#: everywhere (mirrors SASpro's QSettings override, without a Qt dependency).
GAIA_DIR_ENV_VAR = "ASTRAIOS_GAIA_DIR"


@dataclass
class GaiaBand:
    """A downloadable Gaia magnitude band (one or more SQLite files)."""

    key: str
    label: str
    mag_lo: float | None
    mag_hi: float
    filenames: list[str]
    est_size_mb: float  # approximate total download size, for UI display
    est_stars: str  # human-readable star count estimate
    description: str


# Ported verbatim from SASpro's GROUP_DEFS (gaia_database.py) — same
# filenames/URLs so DB files are interchangeable between the two apps.
GAIA_BANDS: list[GaiaBand] = [
    GaiaBand(
        key="ultra_bright",
        label="Ultra-Bright (G < 8)",
        mag_lo=None,
        mag_hi=8.0,
        filenames=["gaia_xp_lt8.sqlite"],
        est_size_mb=220.0,
        est_stars="~55k stars",
        description="Bright stars G<8. Gaia saturates below G~2.2, so the very "
        "brightest naked-eye stars are not included.",
    ),
    GaiaBand(
        key="bright",
        label="Bright (G 8-10)",
        mag_lo=8.0,
        mag_hi=10.0,
        filenames=["gaia_xp_8_10.sqlite"],
        est_size_mb=1500.0,
        est_stars="~385k stars",
        description="Covers most calibration stars reachable from backyard setups.",
    ),
    GaiaBand(
        key="medium",
        label="Medium (G 10-12)",
        mag_lo=10.0,
        mag_hi=12.0,
        filenames=[
            "gaia_xp_100_105.sqlite",
            "gaia_xp_105_110.sqlite",
            "gaia_xp_110_115.sqlite",
            "gaia_xp_115_120.sqlite",
        ],
        est_size_mb=9500.0,
        est_stars="~2.4M stars",
        description="Dense coverage - recommended for wide-field and narrowband imaging.",
    ),
    GaiaBand(
        key="faint",
        label="Faint (G 12-14)",
        mag_lo=12.0,
        mag_hi=14.0,
        filenames=[
            "gaia_xp_120_122.sqlite",
            "gaia_xp_122_124.sqlite",
            "gaia_xp_124_126.sqlite",
            "gaia_xp_126_128.sqlite",
            "gaia_xp_128_130.sqlite",
            "gaia_xp_130_132.sqlite",
            "gaia_xp_132_134.sqlite",
            "gaia_xp_134_136.sqlite",
            "gaia_xp_136_138.sqlite",
            "gaia_xp_138_140.sqlite",
        ],
        est_size_mb=50000.0,
        est_stars="~12.9M stars",
        description="Deep coverage for long-exposure narrowband work (~50 GB total).",
    ),
    GaiaBand(
        key="very_faint",
        label="Very Faint (G 14-15)",
        mag_lo=14.0,
        mag_hi=15.0,
        filenames=[
            "gaia_xp_140_141.sqlite",
            "gaia_xp_141_142.sqlite",
            "gaia_xp_142_143.sqlite",
            "gaia_xp_143_144.sqlite",
            "gaia_xp_144_145.sqlite",
            "gaia_xp_145_146.sqlite",
            "gaia_xp_146_147.sqlite",
            "gaia_xp_147_148.sqlite",
            "gaia_xp_148_149.sqlite",
            "gaia_xp_149_150.sqlite",
        ],
        est_size_mb=73000.0,
        est_stars="~18.7M stars",
        description="Maximum depth for extreme deep-field work (~73 GB total).",
    ),
]

_BAND_BY_KEY: dict[str, GaiaBand] = {b.key: b for b in GAIA_BANDS}


def band_by_key(key: str) -> GaiaBand:
    """Look up a :class:`GaiaBand` by its key. Raises ``KeyError`` if unknown."""
    try:
        return _BAND_BY_KEY[key]
    except KeyError:
        valid = ", ".join(sorted(_BAND_BY_KEY))
        raise KeyError(f"Unknown Gaia band {key!r}. Valid keys: {valid}") from None


def gaia_download_url(filename: str) -> str:
    """Build the download URL for a single Gaia catalog file."""
    return GAIA_DOWNLOAD_BASE_URL + filename


def default_gaia_dir() -> Path:
    """Default catalog directory: ``$ASTRAIOS_GAIA_DIR`` or ``~/.astraios/gaia``."""
    override = os.environ.get(GAIA_DIR_ENV_VAR)
    d = Path(override) if override else Path.home() / ".astraios" / "gaia"
    d.mkdir(parents=True, exist_ok=True)
    return d


def installed_files(catalog_dir: Path | str | None = None) -> list[str]:
    """List installed ``gaia_xp_*.sqlite`` filenames in *catalog_dir*."""
    d = Path(catalog_dir) if catalog_dir else default_gaia_dir()
    if not d.exists():
        return []
    return sorted(p.name for p in d.glob("gaia_xp_*.sqlite"))


def band_status(
    band: GaiaBand, catalog_dir: Path | str | None = None
) -> tuple[list[str], list[str]]:
    """Return ``(installed, missing)`` filenames for *band* in *catalog_dir*."""
    have = set(installed_files(catalog_dir))
    installed = [f for f in band.filenames if f in have]
    missing = [f for f in band.filenames if f not in have]
    return installed, missing


# ── Downloader ───────────────────────────────────────────────────────────

ProgressCB = Callable[[int, int, str], None]  # (bytes_done, bytes_total, message)


def download_file(
    filename: str,
    catalog_dir: Path | str | None = None,
    *,
    chunk_size: int = 1024 * 1024,
    timeout: float = 7200.0,
    progress: ProgressCB | None = None,
    overwrite: bool = False,
) -> Path:
    """Download one Gaia catalog file to *catalog_dir*.

    Streams to a ``.tmp`` sibling and renames on success (same pattern
    SASpro's ``_FileDownloadWorker`` uses), so an interrupted download never
    leaves a corrupt file with the final name.

    Raises
    ------
    URLError
        On network failure.
    """
    d = Path(catalog_dir) if catalog_dir else default_gaia_dir()
    d.mkdir(parents=True, exist_ok=True)
    dest = d / filename

    if dest.exists() and not overwrite:
        log.info("Gaia catalog file already present: %s", dest)
        return dest

    url = gaia_download_url(filename)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    log.info("Downloading Gaia catalog file %s from %s", filename, url)

    req = Request(url, headers={"User-Agent": "Astraios-GaiaCatalog/1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            with open(tmp, "wb") as f:
                while True:
                    buf = resp.read(chunk_size)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    if progress is not None:
                        progress(done, total, filename)
        tmp.rename(dest)
    except URLError:
        tmp.unlink(missing_ok=True)
        raise
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def download_band(
    band_key: str,
    catalog_dir: Path | str | None = None,
    *,
    progress: ProgressCB | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download all missing files for the named band. Returns paths written."""
    band = band_by_key(band_key)
    _installed, missing = band_status(band, catalog_dir) if not overwrite else ([], band.filenames)
    return [
        download_file(f, catalog_dir, progress=progress, overwrite=overwrite) for f in missing
    ]


# ── Catalog reader ───────────────────────────────────────────────────────


class GaiaCatalogError(RuntimeError):
    """Base error for GAIA catalog problems."""


class GaiaCatalogNotFoundError(GaiaCatalogError):
    """Raised when no local catalog files are installed."""


@dataclass
class GaiaSource:
    """A single Gaia DR3 source as read from the local catalog."""

    source_id: int
    ra: float  # degrees, ICRS
    dec: float  # degrees, ICRS
    mag: float | None = None  # phot_g_mean_mag


@dataclass
class GaiaCatalogParams:
    """Settings controlling how :class:`GaiaCatalog` queries are performed."""

    catalog_dir: Path | None = None  # None -> default_gaia_dir()
    mag_limit: float = 16.0  # faintest G magnitude to include
    max_stars: int | None = None  # cap results per query (brightest first)


def _wrap_ra(ra_deg: float) -> float:
    r = float(ra_deg) % 360.0
    return r + 360.0 if r < 0 else r


class GaiaCatalog:
    """Read-only access to locally installed Gaia DR3 SQLite catalog files.

    Opens every ``gaia_xp_*.sqlite`` file in *catalog_dir* (default
    ``~/.astraios/gaia``) read-only and answers cone queries across all of
    them, matching SASpro's ``GaiaBulkLibrary.query_region`` logic: a
    small-angle RA/Dec bounding-box prefilter (with RA wrap handling) in SQL,
    followed by an exact great-circle-ish planar distance check in Python
    (fine for the sub-degree radii plate solving uses).
    """

    def __init__(self, catalog_dir: Path | str | None = None):
        self.catalog_dir = Path(catalog_dir) if catalog_dir else default_gaia_dir()
        self._connections: dict[str, sqlite3.Connection] = {}
        self._open_installed()

    def _open_installed(self) -> None:
        if not self.catalog_dir.exists():
            return
        for path in sorted(self.catalog_dir.glob("gaia_xp_*.sqlite")):
            fname = path.name
            if fname in self._connections:
                continue
            try:
                conn = sqlite3.connect(str(path), check_same_thread=False)
                conn.execute("PRAGMA query_only = ON;")
                self._connections[fname] = conn
            except sqlite3.Error as e:
                log.warning("Could not open Gaia catalog file %s: %s", fname, e)

    @property
    def installed_bands(self) -> list[str]:
        """Filenames of currently open catalog files."""
        return sorted(self._connections)

    def close(self) -> None:
        for conn in self._connections.values():
            with contextlib.suppress(sqlite3.Error):
                conn.close()
        self._connections.clear()

    def __enter__(self) -> GaiaCatalog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def cone_query(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
        mag_limit: float | None = None,
        max_stars: int | None = None,
    ) -> list[GaiaSource]:
        """Return sources within *radius_deg* of (ra_deg, dec_deg).

        Sorted brightest-first (ascending G magnitude; sources with no
        magnitude sort last). Deduplicated by ``source_id`` across catalog
        files (a star near a band boundary can appear in more than one file).

        Raises
        ------
        GaiaCatalogNotFoundError
            If no catalog files are installed in ``self.catalog_dir``.
        """
        if not self._connections:
            raise GaiaCatalogNotFoundError(
                f"No Gaia catalog files found in {self.catalog_dir}. "
                "Download at least one magnitude band first, e.g.:\n"
                "    from astraios.core.gaia_catalog import download_band\n"
                "    download_band('ultra_bright')  # or 'bright' for wider coverage\n"
                "See astraios.core.gaia_catalog.GAIA_BANDS for all available bands."
            )

        ra = _wrap_ra(ra_deg)
        dec = float(dec_deg)
        radius_deg = float(radius_deg)

        cosd = max(1e-6, abs(math.cos(math.radians(dec))))
        dra = radius_deg / cosd
        ra_min = _wrap_ra(ra - dra)
        ra_max = _wrap_ra(ra + dra)
        dec_min = dec - radius_deg
        dec_max = dec + radius_deg

        query = "SELECT source_id, ra, dec, phot_g_mean_mag FROM sources WHERE dec BETWEEN ? AND ?"
        params: list[object] = [dec_min, dec_max]
        if ra_min <= ra_max:
            query += " AND ra BETWEEN ? AND ?"
            params.extend([ra_min, ra_max])
        else:
            # Field straddles the 0/360 RA seam.
            query += " AND (ra >= ? OR ra <= ?)"
            params.extend([ra_min, ra_max])

        eff_mag_limit = mag_limit
        if eff_mag_limit is not None:
            query += " AND phot_g_mean_mag <= ?"
            params.append(float(eff_mag_limit))

        seen: dict[int, GaiaSource] = {}
        for fname, conn in self._connections.items():
            try:
                cur = conn.execute(query, params)
            except sqlite3.Error as e:
                log.warning("Gaia cone query failed on %s: %s", fname, e)
                continue
            for sid, sra, sdec, gmag in cur.fetchall():
                sid = int(sid)
                if sid in seen:
                    continue
                d_ra = (float(sra) - ra + 540.0) % 360.0 - 180.0
                d_dec = float(sdec) - dec
                d = math.hypot(d_ra * cosd, d_dec)
                if d <= radius_deg:
                    seen[sid] = GaiaSource(
                        source_id=sid,
                        ra=float(sra),
                        dec=float(sdec),
                        mag=float(gmag) if gmag is not None else None,
                    )

        results = list(seen.values())
        results.sort(key=lambda s: s.mag if s.mag is not None else float("inf"))
        if max_stars is not None:
            results = results[:max_stars]
        return results
