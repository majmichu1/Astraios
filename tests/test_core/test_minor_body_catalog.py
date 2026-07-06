"""Tests for the minor body (asteroid/comet) catalog (ported from SASpro).

All tests build a small synthetic SQLite DB with the exact schema
``query_minor_bodies`` expects (see ``_REQUIRED_ELEMENT_COLUMNS`` in the
module) and point ``MinorBodyQueryParams.data_dir`` straight at it, so no
test ever needs the real (unavailable) ``saspro-minorbody-data`` release.
Network-hitting code paths (``ensure_minor_body_db`` / the manifest+download
step) are exercised only via monkeypatch, never for real.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from astraios.core.minor_body_catalog import (
    DEFAULT_DB_BASENAME,
    MinorBodyQueryParams,
    query_minor_bodies,
)

_TIME = "2024-02-15T00:00:00"

_ASTEROID_COLUMNS = (
    "designation", "epoch_packed", "mean_anomaly_degrees",
    "argument_of_perihelion_degrees", "longitude_of_ascending_node_degrees",
    "inclination_degrees", "eccentricity", "mean_daily_motion_degrees",
    "semimajor_axis_au", "magnitude_H",
)
_COMET_COLUMNS = (*_ASTEROID_COLUMNS[:-1], "magnitude_g")


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(f"CREATE TABLE asteroids ({', '.join(_ASTEROID_COLUMNS)})")
    cur.execute(f"CREATE TABLE comets ({', '.join(_COMET_COLUMNS)})")

    # A well-behaved elliptical asteroid.
    cur.execute(
        f"INSERT INTO asteroids VALUES ({','.join('?' * len(_ASTEROID_COLUMNS))})",
        ("1 TestAst", "K242A", "10.0", "50.0", "80.0", "5.0", "0.10", "0.25", "2.50", "12.0"),
    )
    # A second asteroid on a very different orbital plane (ends up elsewhere on sky).
    cur.execute(
        f"INSERT INTO asteroids VALUES ({','.join('?' * len(_ASTEROID_COLUMNS))})",
        ("2 FarAst", "K242A", "200.0", "310.0", "200.0", "25.0", "0.20", "0.35", "1.80", "15.0"),
    )
    # A row with a NULL required element — must be skipped gracefully, not crash.
    cur.execute(
        f"INSERT INTO asteroids VALUES ({','.join('?' * len(_ASTEROID_COLUMNS))})",
        ("3 BadAst", "K242A", "10.0", "50.0", "80.0", "5.0", None, "0.25", "2.50", "16.0"),
    )
    # An elliptical comet.
    cur.execute(
        f"INSERT INTO comets VALUES ({','.join('?' * len(_COMET_COLUMNS))})",
        ("1P/TestComet", "K242A", "20.0", "100.0", "20.0", "15.0", "0.55", "0.15", "4.0", "8.0"),
    )
    # A near-parabolic comet (e >= 0.98) — must be excluded (no propagator for this regime).
    cur.execute(
        f"INSERT INTO comets VALUES ({','.join('?' * len(_COMET_COLUMNS))})",
        ("C/HyperComet", "K242A", "0.0", "0.0", "0.0", "90.0", "0.99", "0.01", "50.0", "5.0"),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def db_dir(tmp_path):
    _make_db(tmp_path / DEFAULT_DB_BASENAME)
    return tmp_path


# ── Offline / no-network behavior ───────────────────────────────────────────


def test_missing_catalog_returns_empty_without_network(tmp_path, monkeypatch):
    """No DB on disk + auto_download off (default) -> [] gracefully, no exception,
    and the network path is never even attempted."""

    def _no_network(*a, **kw):
        raise AssertionError("network should not be touched in offline mode")

    monkeypatch.setattr("urllib.request.urlopen", _no_network)

    params = MinorBodyQueryParams(data_dir=tmp_path)  # auto_download=False by default
    result = query_minor_bodies(180.0, 0.0, 180.0, _TIME, params)
    assert result == []


def test_auto_download_failure_returns_empty_gracefully(tmp_path, monkeypatch):
    """auto_download=True but the (mocked) network fetch fails -> [] gracefully."""
    import astraios.core.minor_body_catalog as mbc

    def _fail(*a, **kw):
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(mbc, "ensure_minor_body_db", _fail)

    params = MinorBodyQueryParams(data_dir=tmp_path, auto_download=True)
    result = query_minor_bodies(180.0, 0.0, 180.0, _TIME, params)
    assert result == []


def test_auto_download_success_path_is_mocked(tmp_path, monkeypatch):
    """auto_download=True with the download step mocked to materialize our
    synthetic DB — exercises the download-then-query path with zero real HTTP."""
    import astraios.core.minor_body_catalog as mbc

    real_db = tmp_path / "real_data"
    real_db.mkdir()
    _make_db(real_db / DEFAULT_DB_BASENAME)

    download_dir = tmp_path / "cache"

    def _fake_ensure(data_dir, manifest_url=mbc.MANIFEST_URL, force_refresh=False):
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        dest = data_dir / DEFAULT_DB_BASENAME
        dest.write_bytes((real_db / DEFAULT_DB_BASENAME).read_bytes())
        return dest, None

    monkeypatch.setattr(mbc, "ensure_minor_body_db", _fake_ensure)

    params = MinorBodyQueryParams(data_dir=download_dir, auto_download=True)
    result = query_minor_bodies(180.0, 0.0, 180.0, _TIME, params)
    assert len(result) > 0


# ── Parsing / columns / filtering-by-field ──────────────────────────────────


def test_whole_sky_query_parses_expected_bodies(db_dir):
    params = MinorBodyQueryParams(data_dir=db_dir, compute_motion=False)
    result = query_minor_bodies(0.0, 0.0, 180.0, _TIME, params)

    designations = {b.designation for b in result}
    assert "1 TestAst" in designations
    assert "2 FarAst" in designations
    assert "1P/TestComet" in designations
    # Skipped: NULL required element, and near-parabolic eccentricity.
    assert "3 BadAst" not in designations
    assert "C/HyperComet" not in designations

    by_desig = {b.designation: b for b in result}
    assert by_desig["1 TestAst"].kind == "asteroid"
    assert by_desig["1 TestAst"].magnitude == pytest.approx(12.0)
    assert by_desig["1P/TestComet"].kind == "comet"
    assert by_desig["1P/TestComet"].magnitude == pytest.approx(8.0)
    for b in result:
        assert -90.0 <= b.dec_deg <= 90.0
        assert 0.0 <= b.ra_deg < 360.0
        assert b.distance_au is not None and b.distance_au > 0.0


def test_field_radius_filter_includes_and_excludes(db_dir):
    params = MinorBodyQueryParams(data_dir=db_dir, compute_motion=False)
    everything = query_minor_bodies(0.0, 0.0, 180.0, _TIME, params)
    assert len(everything) >= 3

    target = everything[0]

    # Tiny field centered exactly on one body's own position -> that body
    # (at least) must come back.
    narrow = query_minor_bodies(target.ra_deg, target.dec_deg, 0.01, _TIME, params)
    assert any(b.designation == target.designation for b in narrow)

    # A field far from every computed body -> nothing comes back.
    occupied_positions = [(b.ra_deg, b.dec_deg) for b in everything]

    def _min_sep_deg(ra, dec):
        import math
        best = 999.0
        for r2, d2 in occupied_positions:
            # crude but adequate separation proxy for a synthetic-test check
            dra = min(abs(ra - r2), 360.0 - abs(ra - r2))
            best = min(best, math.hypot(dra * math.cos(math.radians(dec)), dec - d2))
        return best

    # Search a small grid for a spot far from all bodies.
    far_ra, far_dec = 0.0, 0.0
    for cand_ra in range(0, 360, 15):
        for cand_dec in (-80, -40, 0, 40, 80):
            if _min_sep_deg(cand_ra, cand_dec) > 5.0:
                far_ra, far_dec = cand_ra, cand_dec
                break
        else:
            continue
        break

    empty = query_minor_bodies(far_ra, far_dec, 0.05, _TIME, params)
    assert empty == []


def test_include_flags_filter_by_kind(db_dir):
    only_asteroids = query_minor_bodies(
        0.0, 0.0, 180.0, _TIME,
        MinorBodyQueryParams(data_dir=db_dir, include_comets=False, compute_motion=False),
    )
    assert only_asteroids
    assert all(b.kind == "asteroid" for b in only_asteroids)

    only_comets = query_minor_bodies(
        0.0, 0.0, 180.0, _TIME,
        MinorBodyQueryParams(data_dir=db_dir, include_asteroids=False, compute_motion=False),
    )
    assert only_comets
    assert all(b.kind == "comet" for b in only_comets)


def test_max_results_truncates(db_dir):
    params = MinorBodyQueryParams(data_dir=db_dir, compute_motion=False, max_results=1)
    result = query_minor_bodies(0.0, 0.0, 180.0, _TIME, params)
    assert len(result) == 1


def test_motion_computed_when_requested(db_dir):
    with_motion = query_minor_bodies(
        0.0, 0.0, 180.0, _TIME,
        MinorBodyQueryParams(data_dir=db_dir, compute_motion=True, motion_dt_hours=1.0),
    )
    assert with_motion
    for b in with_motion:
        assert b.motion_arcsec_per_hour is not None
        assert b.motion_arcsec_per_hour >= 0.0
        assert b.motion_position_angle_deg is not None
        assert 0.0 <= b.motion_position_angle_deg < 360.0

    without_motion = query_minor_bodies(
        0.0, 0.0, 180.0, _TIME,
        MinorBodyQueryParams(data_dir=db_dir, compute_motion=False),
    )
    assert all(b.motion_arcsec_per_hour is None for b in without_motion)


def test_deterministic(db_dir):
    params = MinorBodyQueryParams(data_dir=db_dir, compute_motion=False)
    r1 = query_minor_bodies(10.0, 5.0, 90.0, _TIME, params)
    r2 = query_minor_bodies(10.0, 5.0, 90.0, _TIME, params)
    assert [b.designation for b in r1] == [b.designation for b in r2]
    for b1, b2 in zip(r1, r2, strict=True):
        assert b1.ra_deg == b2.ra_deg
        assert b1.dec_deg == b2.dec_deg
