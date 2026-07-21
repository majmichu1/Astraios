"""Minor body (asteroid + comet) catalog helper.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

Data source (ported architecture, verified against ``minorbodycatalog.py``):
this is **not** a live per-request query against astroquery's
``jplhorizons``/MPC services. SASpro instead ships/downloads a small SQLite
database of osculating orbital elements (semimajor axis, eccentricity,
inclination, node, argument of perihelion, mean anomaly + mean daily motion,
epoch) for a bulk set of asteroids and comets, fetched from a JSON manifest
hosted in the maintainer's public ``saspro-minorbody-data`` GitHub repo
(mirroring the pattern this codebase already uses in
:mod:`astraios.core.gaia_catalog` for Gaia DR3 — same "point at the same
public asset SASpro uses" approach, same manifest/version-check/atomic-
download shape). Positions are then computed **locally** by propagating the
elements with a standalone Kepler solver (Newton-Raphson on ``M = E -
e*sin(E)``) and rotating into ICRS via astropy — this is the exact
computation ported from ``MinorBodyCatalog.compute_positions_astropy``, the
one live piece of position math the source file contains. Skyfield/MPC
imports in the source are legacy/unused (guarded by a bare ``try/except``)
and are not ported.

Offline behavior (per this port's requirement): network access is used only
to fetch the manifest + database the *first* time (or on ``force_refresh``);
:func:`query_minor_bodies` never raises for network or missing-file
conditions — it logs and returns an empty list. Pass
``MinorBodyQueryParams(auto_download=False)`` (the default) to guarantee no
network calls are attempted at all: the catalog is used only if a database
already exists on disk (e.g. dropped in by the user, or shared with an
installed copy of SASpro, exactly as :mod:`astraios.core.gaia_catalog`
already lets users reuse SASpro's downloaded Gaia files).

Comets vs. asteroids: the source file's ``compute_positions_astropy`` takes
generic "asteroid_rows" and is not restricted to the ``asteroids`` table by
any check inside it — this port applies the identical Kepler propagator to
whichever table's rows it is given (asteroids or comets), since the file
defines no separate comet-specific propagator. Comet rows whose eccentricity
is near-parabolic/hyperbolic (``e >= 0.98``, where ``a = q/(1-e)`` blows up
or goes negative) are skipped for position purposes and reported by
designation/magnitude only, since the ported formula does not support that
regime (matches the *absence* of any such handling in the source).

This module is pure CPU/astropy/sqlite3 — no GPU, no benchmark, consistent
with the astraios convention that ``device_manager`` is reserved for
image-tensor workloads.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from astropy.coordinates import angular_separation, position_angle
from astropy.time import Time

log = logging.getLogger(__name__)

# ── Data source (see module docstring) ──────────────────────────────────────

MANIFEST_URL = (
    "https://raw.githubusercontent.com/setiastro/"
    "saspro-minorbody-data/main/saspro_minor_bodies_manifest.json"
)
DEFAULT_DB_BASENAME = "saspro_minor_bodies.sqlite"
DEFAULT_MANIFEST_BASENAME = "saspro_minor_bodies_manifest.json"

#: Env var to override the default catalog directory (mirrors GAIA_DIR_ENV_VAR
#: in astraios.core.gaia_catalog).
MINORBODY_DIR_ENV_VAR = "ASTRAIOS_MINORBODY_DIR"

#: Eccentricity above which the elliptical Kepler propagator (a = q/(1-e))
#: is no longer valid/stable — near-parabolic and hyperbolic comets.
_MAX_ELLIPTICAL_ECCENTRICITY = 0.98

_REQUIRED_ELEMENT_COLUMNS = (
    "mean_anomaly_degrees",
    "argument_of_perihelion_degrees",
    "longitude_of_ascending_node_degrees",
    "inclination_degrees",
    "eccentricity",
    "mean_daily_motion_degrees",
    "semimajor_axis_au",
)

_COMET_MAG_CANDIDATES = (
    "magnitude_g", "magnitude_k", "absolute_magnitude", "magnitude_H", "H",
)


def default_minor_body_dir() -> Path:
    """Default catalog directory: ``$ASTRAIOS_MINORBODY_DIR`` or ``~/.astraios/minor_bodies``."""
    override = os.environ.get(MINORBODY_DIR_ENV_VAR)
    d = Path(override) if override else Path.home() / ".astraios" / "minor_bodies"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Kepler propagation helpers (ported verbatim) ────────────────────────────


def _solve_kepler(M: float, e: float, tol: float = 1e-10, max_iter: int = 50) -> float:
    """Solve Kepler's equation ``M = E - e*sin(E)`` for E via Newton-Raphson.

    ``M`` may be any real angle. It is reduced to one revolution before
    iterating and the whole revolutions are added back, so the returned ``E``
    still satisfies ``M = E - e*sin(E)`` exactly.

    Naive Newton from a fixed ``E = pi`` start diverges violently for
    ``e >= 0.8`` once ``M`` falls outside a single revolution (it returned
    ``E = -245012`` for ``e=0.95, M=9.11``), because ``1 - e*cos(E)`` gets
    close to zero and the step explodes. The current caller happens to wrap
    ``M`` into ``[0, 2*pi)`` first, so that was never reachable in practice,
    but comets run at ``e`` near 1 and the guard costs nothing: use Danby's
    starting value and clamp each step to one radian.
    """
    if e < 0.0:
        raise ValueError(f"eccentricity must be non-negative, got {e}")

    # Reduce to [-pi, pi], remembering how many full revolutions were removed.
    revolutions = math.floor((M + math.pi) / (2.0 * math.pi))
    m_reduced = M - 2.0 * math.pi * revolutions

    if e < 0.8:
        E = m_reduced
    else:
        # Danby's starter: biases the guess toward perihelion, where the
        # curvature that breaks a fixed pi start actually lives.
        E = m_reduced + 0.85 * e * math.copysign(1.0, math.sin(m_reduced) or 1.0)

    for _ in range(max_iter):
        denom = 1.0 - e * math.cos(E)
        if abs(denom) < 1e-12:  # near-parabolic: keep the step finite
            denom = math.copysign(1e-12, denom or 1.0)
        dE = (m_reduced - E + e * math.sin(E)) / denom
        # A full-radian cap still converges quadratically near the root but
        # stops a single bad step from throwing the iteration to infinity.
        dE = max(-1.0, min(1.0, dE))
        E += dE
        if abs(dE) < tol:
            break

    return E + 2.0 * math.pi * revolutions


def _decode_packed_epoch(packed: str) -> float:
    """Decode an MPC packed epoch string (e.g. ``'K245N'``) to a Julian date."""
    if not packed or len(packed) < 5:
        return 2451545.0  # fallback to J2000

    def n(c: str) -> int:
        return ord(c) - (48 if c.isdigit() else 55)

    try:
        century = 100 * n(packed[0]) + int(packed[1:3])
        month = n(packed[3])
        day = n(packed[4])

        y, m, d = century, month, day
        if m <= 2:
            y -= 1
            m += 12
        A = int(y / 100)
        B = 2 - A + int(A / 4)
        return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    except Exception:
        return 2451545.0


# ── Network helpers (urllib only, minimal deps) ─────────────────────────────


def _http_get_json(url: str, timeout: float = 15.0) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "Astraios-MinorBodyCatalog/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} retrieving {url}")
        data = resp.read().decode("utf-8")
    return json.loads(data)


def _http_download_binary(
    url: str, dest: Path, chunk_size: int = 65536, timeout: float = 30.0
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Astraios-MinorBodyCatalog/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} retrieving {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f_out:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
        tmp.replace(dest)


# ── Manifest + DB management ─────────────────────────────────────────────────


@dataclass
class MinorBodyManifest:
    schema_version: int
    version: str
    generated_utc: str
    download_url: str
    download_filename: str
    counts_asteroids: int
    counts_comets: int
    raw: dict[str, Any]


def _manifest_from_json(data: dict[str, Any]) -> MinorBodyManifest:
    dl = data.get("download", {})
    counts = data.get("counts", {})
    return MinorBodyManifest(
        schema_version=int(data.get("schema_version", 1)),
        version=str(data.get("version", "unknown")),
        generated_utc=str(data.get("generated_utc", "")),
        download_url=str(dl.get("url", "")),
        download_filename=str(dl.get("filename", DEFAULT_DB_BASENAME)),
        counts_asteroids=int(counts.get("asteroids", 0)),
        counts_comets=int(counts.get("comets", 0)),
        raw=data,
    )


def fetch_remote_manifest(url: str = MANIFEST_URL) -> MinorBodyManifest:
    """Fetch the remote manifest from GitHub and parse it. Raises on network failure."""
    return _manifest_from_json(_http_get_json(url))


def load_local_manifest(path: Path) -> MinorBodyManifest | None:
    """Load a previously saved local manifest (if it exists)."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return _manifest_from_json(json.load(f))
    except Exception:
        return None


def save_local_manifest(path: Path, manifest: MinorBodyManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest.raw, f, indent=2)


def ensure_minor_body_db(
    data_dir: Path, manifest_url: str = MANIFEST_URL, force_refresh: bool = False
) -> tuple[Path, MinorBodyManifest]:
    """Ensure the minor-body SQLite DB exists locally (and is up to date).

    Network-only; any UI/progress should wrap this and catch exceptions —
    :func:`query_minor_bodies` does exactly that so it never raises.
    """
    data_dir = Path(data_dir).resolve()
    local_manifest_path = data_dir / DEFAULT_MANIFEST_BASENAME

    remote = fetch_remote_manifest(manifest_url)
    db_path = data_dir / remote.download_filename
    local = load_local_manifest(local_manifest_path)

    needs_download = force_refresh or local is None or local.version != remote.version
    needs_download = needs_download or not db_path.is_file()

    if needs_download:
        if not remote.download_url:
            raise RuntimeError("Manifest does not contain a download URL for the DB.")
        _http_download_binary(remote.download_url, db_path)
        save_local_manifest(local_manifest_path, remote)
        manifest = remote
    else:
        manifest = local

    return db_path, manifest


# ── Catalog reader (sqlite3 only — no pandas dependency in astraios) ────────


class MinorBodyCatalog:
    """Thin wrapper around the minor-body SQLite DB (read-only queries)."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path).resolve()
        if not self.db_path.is_file():
            raise FileNotFoundError(f"Minor body DB not found: {self.db_path}")
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            uri = f"file:{self.db_path.as_posix()}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def counts(self) -> dict[str, int]:
        conn = self._get_conn()
        cur = conn.cursor()
        result = {}
        for table in ("asteroids", "comets"):
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608 - fixed table names
                result[table] = int(cur.fetchone()[0])
            except sqlite3.Error:
                result[table] = 0
        return result

    def table_columns(self, table: str) -> set[str]:
        """Column names present in *table* (introspection, e.g. for magnitude
        column auto-detection). Empty set if the table doesn't exist."""
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            cur.execute(f"PRAGMA table_info({table})")  # noqa: S608 - fixed table names
        except sqlite3.Error:
            return set()
        return {row[1] for row in cur.fetchall()}

    def get_bright_asteroids(
        self, H_max: float = 20.0, limit: int | None = 100000
    ) -> list[dict[str, Any]]:
        """Bright asteroids (``magnitude_H`` <= H_max), brightest first."""
        conn = self._get_conn()
        sql = (
            "SELECT * FROM asteroids WHERE CAST(magnitude_H AS REAL) <= ? "
            "ORDER BY CAST(magnitude_H AS REAL) ASC"
        )
        params: list[Any] = [H_max]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        cur = conn.cursor()
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def get_bright_comets(
        self, H_max: float = 15.0, limit: int | None = 5000
    ) -> list[dict[str, Any]]:
        """Bright comets. Auto-detects the magnitude column, same as the source."""
        conn = self._get_conn()
        cols = self.table_columns("comets")
        mag_col = next((c for c in _COMET_MAG_CANDIDATES if c in cols), None)

        cur = conn.cursor()
        if mag_col is None:
            sql = "SELECT * FROM comets"
            params: list[Any] = []
            if limit is not None:
                sql += " LIMIT ?"
                params.append(limit)
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

        mag_expr = f"CAST({mag_col} AS REAL)"
        sql = f"SELECT * FROM comets WHERE {mag_expr} <= ? ORDER BY {mag_expr} ASC"  # noqa: S608
        params = [H_max]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def get_asteroids_by_designation(self, designations: list[str]) -> list[dict[str, Any]]:
        if not designations:
            return []
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in designations)
        sql = f"SELECT * FROM asteroids WHERE designation IN ({placeholders})"  # noqa: S608
        cur = conn.cursor()
        cur.execute(sql, designations)
        return [dict(row) for row in cur.fetchall()]


# ── Kepler propagation (ported from MinorBodyCatalog.compute_positions_astropy) ──


def _propagate_kepler_positions(
    rows: list[dict[str, Any]], jd: float, debug: bool = False
) -> list[dict[str, Any]]:
    """Propagate a set of osculating elements to ``jd`` (TT) and return ICRS ra/dec.

    Ported from ``compute_positions_astropy`` (dict-of-rows in, no pandas):
    Kepler's equation solved per body, rotated orbital-plane -> ecliptic J2000
    -> ICRS -> barycentric -> topocentric (Earth-relative) vector -> ra/dec.
    Rows missing a required numeric element, or with eccentricity too high
    for the elliptical solver, are skipped (not raised).
    """
    from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris

    clean_rows = []
    for row in rows:
        try:
            if any(row.get(c) in (None, "") for c in _REQUIRED_ELEMENT_COLUMNS):
                continue
            elements = {c: float(row[c]) for c in _REQUIRED_ELEMENT_COLUMNS}
        except (TypeError, ValueError):
            continue
        if elements["eccentricity"] >= _MAX_ELLIPTICAL_ECCENTRICITY:
            continue
        clean_rows.append((row, elements))

    if not clean_rows:
        return []

    t_obs = Time(jd, format="jd", scale="tt")
    with solar_system_ephemeris.set("builtin"):
        earth_bary, _ = get_body_barycentric_posvel("earth", t_obs)
        sun_bary, _ = get_body_barycentric_posvel("sun", t_obs)
    earth_xyz = (
        earth_bary.x.to_value("au"), earth_bary.y.to_value("au"), earth_bary.z.to_value("au"),
    )
    sun_xyz = (
        sun_bary.x.to_value("au"), sun_bary.y.to_value("au"), sun_bary.z.to_value("au"),
    )

    results: list[dict[str, Any]] = []
    ok = failed = 0

    for row, el in clean_rows:
        try:
            epoch_packed = str(row.get("epoch_packed", "")).strip()
            epoch_jd = _decode_packed_epoch(epoch_packed)
            t_epoch = Time(epoch_jd, format="jd", scale="tt")

            a = el["semimajor_axis_au"]
            e = el["eccentricity"]
            inc = math.radians(el["inclination_degrees"])
            Om = math.radians(el["longitude_of_ascending_node_degrees"])
            om = math.radians(el["argument_of_perihelion_degrees"])
            M0 = math.radians(el["mean_anomaly_degrees"])
            n = math.radians(el["mean_daily_motion_degrees"])  # rad/day

            dt_days = float((t_obs - t_epoch).jd)
            M = (M0 + n * dt_days) % (2 * math.pi)

            E = _solve_kepler(M, e)
            cos_E, sin_E = math.cos(E), math.sin(E)
            nu = math.atan2(math.sqrt(1 - e * e) * sin_E, cos_E - e)
            r = a * (1 - e * cos_E)

            x_orb = r * math.cos(nu)
            y_orb = r * math.sin(nu)

            cos_Om, sin_Om = math.cos(Om), math.sin(Om)
            cos_om, sin_om = math.cos(om), math.sin(om)
            cos_i, sin_i = math.cos(inc), math.sin(inc)

            Xx = cos_Om * cos_om - sin_Om * sin_om * cos_i
            Xy = -cos_Om * sin_om - sin_Om * cos_om * cos_i
            Yx = sin_Om * cos_om + cos_Om * sin_om * cos_i
            Yy = -sin_Om * sin_om + cos_Om * cos_om * cos_i
            Zx = sin_om * sin_i
            Zy = cos_om * sin_i

            x_ecl = Xx * x_orb + Xy * y_orb
            y_ecl = Yx * x_orb + Yy * y_orb
            z_ecl = Zx * x_orb + Zy * y_orb

            eps = math.radians(23.439291111)  # obliquity J2000.0
            cos_e, sin_e = math.cos(eps), math.sin(eps)
            x_icrs = x_ecl
            y_icrs = cos_e * y_ecl - sin_e * z_ecl
            z_icrs = sin_e * y_ecl + cos_e * z_ecl

            x_bary = x_icrs + sun_xyz[0]
            y_bary = y_icrs + sun_xyz[1]
            z_bary = z_icrs + sun_xyz[2]

            dx = x_bary - earth_xyz[0]
            dy = y_bary - earth_xyz[1]
            dz = z_bary - earth_xyz[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            ra_rad = math.atan2(dy, dx) % (2 * math.pi)
            dec_rad = math.asin(dz / dist) if dist > 0 else 0.0

            results.append({
                "designation": row.get("designation", ""),
                "ra_deg": math.degrees(ra_rad),
                "dec_deg": math.degrees(dec_rad),
                "distance_au": dist,
            })
            ok += 1
        except Exception as exc:
            failed += 1
            if debug and failed <= 10:
                log.debug("Minor body propagation failed for %r: %r", row.get("designation"), exc)

    if debug:
        log.debug("Minor body propagation: ok=%d failed=%d", ok, failed)
    return results


# ── Public query API ─────────────────────────────────────────────────────────


@dataclass
class MinorBody:
    """A single minor body's computed field position at query time."""

    designation: str
    kind: str  # "asteroid" | "comet"
    ra_deg: float
    dec_deg: float
    magnitude: float | None = None
    distance_au: float | None = None
    motion_arcsec_per_hour: float | None = None
    motion_position_angle_deg: float | None = None  # East of North


@dataclass
class MinorBodyQueryParams:
    """Options controlling a :func:`query_minor_bodies` call."""

    data_dir: Path | str = field(default_factory=default_minor_body_dir)
    manifest_url: str = MANIFEST_URL
    include_asteroids: bool = True
    include_comets: bool = True
    h_max_asteroid: float = 18.0
    h_max_comet: float = 16.0
    #: Cap on how many bright-magnitude candidates get propagated per table
    #: (Kepler propagation is cheap per body, but this bounds worst case).
    max_candidates: int = 20000
    max_results: int = 200
    compute_motion: bool = True
    motion_dt_hours: float = 1.0
    #: If False (default), never touches the network — only a database
    #: already present in ``data_dir`` is used. If True and no local DB is
    #: found, :func:`ensure_minor_body_db` is attempted (and any failure is
    #: caught, logged, and treated as "offline").
    auto_download: bool = False
    force_refresh: bool = False


def _resolve_db_path(params: MinorBodyQueryParams) -> Path | None:
    data_dir = Path(params.data_dir)
    candidate = data_dir / DEFAULT_DB_BASENAME
    if candidate.is_file() and not params.force_refresh:
        return candidate

    if not params.auto_download and not params.force_refresh:
        if candidate.is_file():
            return candidate
        log.info(
            "Minor body catalog not found at %s and auto_download is off — "
            "returning no results (offline mode).",
            candidate,
        )
        return None

    try:
        db_path, _manifest = ensure_minor_body_db(
            data_dir, manifest_url=params.manifest_url, force_refresh=params.force_refresh
        )
        return db_path
    except Exception as exc:
        log.warning("Minor body catalog download failed (offline?): %s", exc)
        return None


def query_minor_bodies(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    time: Time | str | datetime,
    params: MinorBodyQueryParams | None = None,
) -> list[MinorBody]:
    """Return minor bodies whose computed position at ``time`` falls within
    ``radius_deg`` of (``ra_deg``, ``dec_deg``).

    Never raises for network/missing-catalog conditions — returns ``[]`` and
    logs instead (see the module docstring's "Offline behavior" section).
    """
    params = params or MinorBodyQueryParams()

    db_path = _resolve_db_path(params)
    if db_path is None:
        return []

    try:
        catalog = MinorBodyCatalog(db_path)
    except FileNotFoundError as exc:
        log.warning("Minor body catalog unavailable: %s", exc)
        return []

    time_obj = time if isinstance(time, Time) else Time(time)
    jd = time_obj.tt.jd

    out: list[MinorBody] = []
    try:
        table_specs = []
        if params.include_asteroids:
            asteroid_rows = catalog.get_bright_asteroids(
                H_max=params.h_max_asteroid, limit=params.max_candidates
            )
            table_specs.append(("asteroid", asteroid_rows, "magnitude_H"))
        if params.include_comets:
            comet_cols = catalog.table_columns("comets")
            comet_mag_col = next((c for c in _COMET_MAG_CANDIDATES if c in comet_cols), None)
            table_specs.append((
                "comet",
                catalog.get_bright_comets(H_max=params.h_max_comet, limit=params.max_candidates),
                comet_mag_col,
            ))

        for kind, rows, mag_col in table_specs:
            if not rows:
                continue
            positions = _propagate_kepler_positions(rows, jd)
            positions_dt = None
            if params.compute_motion:
                jd_dt = jd + params.motion_dt_hours / 24.0
                by_desig = {r.get("designation", ""): r for r in rows}
                rows_at_t0 = [
                    by_desig[p["designation"]]
                    for p in positions
                    if p["designation"] in by_desig
                ]
                positions_dt = {
                    p["designation"]: p
                    for p in _propagate_kepler_positions(rows_at_t0, jd_dt)
                }

            rows_by_desig = {r.get("designation", ""): r for r in rows}
            for pos in positions:
                sep = angular_separation(
                    math.radians(ra_deg), math.radians(dec_deg),
                    math.radians(pos["ra_deg"]), math.radians(pos["dec_deg"]),
                )
                sep_deg = math.degrees(float(sep))
                if sep_deg > radius_deg:
                    continue

                row = rows_by_desig.get(pos["designation"], {})
                mag = None
                if mag_col is not None:
                    try:
                        mag = float(row.get(mag_col))
                    except (TypeError, ValueError):
                        mag = None

                motion_asec_hr = None
                motion_pa = None
                if positions_dt is not None and pos["designation"] in positions_dt:
                    p2 = positions_dt[pos["designation"]]
                    sep2 = angular_separation(
                        math.radians(pos["ra_deg"]), math.radians(pos["dec_deg"]),
                        math.radians(p2["ra_deg"]), math.radians(p2["dec_deg"]),
                    )
                    motion_asec_hr = math.degrees(float(sep2)) * 3600.0 / params.motion_dt_hours
                    pa = position_angle(
                        math.radians(pos["ra_deg"]), math.radians(pos["dec_deg"]),
                        math.radians(p2["ra_deg"]), math.radians(p2["dec_deg"]),
                    )
                    motion_pa = math.degrees(pa.rad) % 360.0

                out.append(MinorBody(
                    designation=str(pos["designation"]),
                    kind=kind,
                    ra_deg=pos["ra_deg"],
                    dec_deg=pos["dec_deg"],
                    magnitude=mag,
                    distance_au=pos["distance_au"],
                    motion_arcsec_per_hour=motion_asec_hr,
                    motion_position_angle_deg=motion_pa,
                ))
    finally:
        catalog.close()

    out.sort(key=lambda b: b.magnitude if b.magnitude is not None else 99.0)
    return out[: params.max_results]
