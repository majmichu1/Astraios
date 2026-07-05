"""Tests for the offline Gaia catalog reader/downloader and plate solver.

Builds a synthetic Gaia-format SQLite catalog (same schema SASpro's bulk
library files use) plus a synthetic star image whose stars are a WCS
projection of a subset of that catalog, then verifies the solver recovers
the known field center/scale. No test in this module touches the network.
"""

from __future__ import annotations

import math
import sqlite3

import numpy as np
import pytest
from astropy.wcs import WCS

from astraios.core.gaia_catalog import (
    GAIA_BANDS,
    GaiaCatalog,
    GaiaCatalogNotFoundError,
    band_by_key,
    band_status,
    default_gaia_dir,
    download_band,
    download_file,
    gaia_download_url,
    installed_files,
)
from astraios.core.gaia_solver import (
    GaiaSolveParams,
    plate_solve_gaia,
    solve_with_gaia_catalog,
)

# ── Shared synthetic-field fixture ──────────────────────────────────────

IMG_W = IMG_H = 400
SCALE_TRUE = 2.0  # arcsec/pixel
ROT_TRUE = 12.0  # degrees
RA0, DEC0 = 180.0, 20.0


def _make_wcs(ra: float, dec: float, scale_arcsec: float, rot_deg: float, w: int, h: int) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [w / 2.0, h / 2.0]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    scale_deg = scale_arcsec / 3600.0
    rot = math.radians(rot_deg)
    cr, sr = math.cos(rot), math.sin(rot)
    wcs.wcs.cd = np.array(
        [[-scale_deg * cr, scale_deg * sr], [-scale_deg * sr, -scale_deg * cr]]
    )
    wcs.wcs.set()
    return wcs


def _write_catalog_db(db_path, rows: list[tuple]) -> None:
    """Write a SASpro-format Gaia sources table.

    rows: (source_id, ra, dec, phot_g_mean_mag)
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE sources (
            source_id INTEGER PRIMARY KEY, ra REAL, dec REAL, phot_g_mean_mag REAL,
            bp_rp REAL, parallax REAL, pmra REAL, pmdec REAL,
            has_xp_spectrum INTEGER DEFAULT 0
        )"""
    )
    conn.executemany(
        "INSERT INTO sources (source_id, ra, dec, phot_g_mean_mag) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def _build_synthetic_field(tmp_path, seed: int = 42):
    """Build a synthetic Gaia catalog file + a matching star image.

    Returns (catalog_dir, image, true_wcs).
    """
    rng = np.random.default_rng(seed)
    true_wcs = _make_wcs(RA0, DEC0, SCALE_TRUE, ROT_TRUE, IMG_W, IMG_H)

    n_cat = 400
    px = rng.uniform(-150, IMG_W + 150, n_cat)
    py = rng.uniform(-150, IMG_H + 150, n_cat)
    mags = rng.uniform(8.0, 15.0, n_cat)
    sky = true_wcs.pixel_to_world(px, py)

    catalog_dir = tmp_path / "gaia_catalog"
    catalog_dir.mkdir()
    rows = [
        (i, float(sky.ra.deg[i]), float(sky.dec.deg[i]), float(mags[i])) for i in range(n_cat)
    ]
    _write_catalog_db(catalog_dir / "gaia_xp_test.sqlite", rows)

    # Stars actually inside the frame (with a small margin) become image stars.
    in_bounds = (px >= 5) & (px < IMG_W - 5) & (py >= 5) & (py < IMG_H - 5)
    img_px, img_py, img_mags = px[in_bounds], py[in_bounds], mags[in_bounds]

    image = np.full((IMG_H, IMG_W), 0.03, dtype=np.float32)
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    for x, y, m in zip(img_px, img_py, img_mags, strict=True):
        amp = float(np.clip(1.5 * 10 ** (-0.4 * (m - 8.0)), 0.05, 0.9))
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        image += (amp * np.exp(-d2 / (2 * 2.2**2))).astype(np.float32)
    image = np.clip(image, 0.0, 1.0).astype(np.float32)

    return catalog_dir, image, true_wcs


# ── GaiaCatalog cone-query tests ────────────────────────────────────────


class TestGaiaCatalogConeQuery:
    def test_cone_query_returns_only_stars_within_radius(self, tmp_path):
        center_ra, center_dec, radius = 100.0, 30.0, 0.5
        # 3 stars inside the radius, 1 clearly outside.
        rows = [
            (1, center_ra, center_dec, 10.0),  # dead center
            (2, center_ra + 0.1, center_dec + 0.1, 12.0),
            (3, center_ra - 0.2, center_dec, 9.0),
            (4, center_ra + 5.0, center_dec, 11.0),  # outside
        ]
        _write_catalog_db(tmp_path / "gaia_xp_a.sqlite", rows)

        cat = GaiaCatalog(tmp_path)
        try:
            assert cat.installed_bands == ["gaia_xp_a.sqlite"]
            results = cat.cone_query(center_ra, center_dec, radius)
            ids = {s.source_id for s in results}
            assert ids == {1, 2, 3}
        finally:
            cat.close()

    def test_cone_query_sorted_brightest_first(self, tmp_path):
        rows = [
            (1, 50.0, 10.0, 14.0),
            (2, 50.0, 10.0, 9.0),
            (3, 50.0, 10.0, 11.0),
        ]
        _write_catalog_db(tmp_path / "gaia_xp_a.sqlite", rows)
        cat = GaiaCatalog(tmp_path)
        try:
            results = cat.cone_query(50.0, 10.0, 1.0)
            mags = [s.mag for s in results]
            assert mags == sorted(mags)
            assert [s.source_id for s in results] == [2, 3, 1]
        finally:
            cat.close()

    def test_cone_query_mag_limit_filters(self, tmp_path):
        rows = [(1, 50.0, 10.0, 10.0), (2, 50.0, 10.0, 16.0)]
        _write_catalog_db(tmp_path / "gaia_xp_a.sqlite", rows)
        cat = GaiaCatalog(tmp_path)
        try:
            results = cat.cone_query(50.0, 10.0, 1.0, mag_limit=12.0)
            assert [s.source_id for s in results] == [1]
        finally:
            cat.close()

    def test_cone_query_dedups_across_files(self, tmp_path):
        # Same source_id present in two band files (boundary overlap) —
        # must only be counted once.
        _write_catalog_db(tmp_path / "gaia_xp_a.sqlite", [(1, 50.0, 10.0, 10.0)])
        _write_catalog_db(tmp_path / "gaia_xp_b.sqlite", [(1, 50.0, 10.0, 10.0)])
        cat = GaiaCatalog(tmp_path)
        try:
            assert len(cat.installed_bands) == 2
            results = cat.cone_query(50.0, 10.0, 1.0)
            assert len(results) == 1
        finally:
            cat.close()

    def test_cone_query_handles_ra_wrap(self, tmp_path):
        # Field straddles the 0/360 RA seam.
        rows = [(1, 0.05, 5.0, 10.0), (2, 359.95, 5.0, 10.0), (3, 180.0, 5.0, 10.0)]
        _write_catalog_db(tmp_path / "gaia_xp_a.sqlite", rows)
        cat = GaiaCatalog(tmp_path)
        try:
            results = cat.cone_query(0.0, 5.0, 0.5)
            ids = {s.source_id for s in results}
            assert ids == {1, 2}
        finally:
            cat.close()

    def test_no_catalog_files_raises_clear_error(self, tmp_path):
        cat = GaiaCatalog(tmp_path)
        try:
            with pytest.raises(GaiaCatalogNotFoundError) as exc_info:
                cat.cone_query(10.0, 10.0, 1.0)
            msg = str(exc_info.value)
            assert "download_band" in msg
            assert str(tmp_path) in msg
        finally:
            cat.close()

    def test_missing_directory_is_graceful(self, tmp_path):
        missing_dir = tmp_path / "does_not_exist"
        cat = GaiaCatalog(missing_dir)
        try:
            assert cat.installed_bands == []
            with pytest.raises(GaiaCatalogNotFoundError):
                cat.cone_query(10.0, 10.0, 1.0)
        finally:
            cat.close()

    def test_context_manager_closes(self, tmp_path):
        _write_catalog_db(tmp_path / "gaia_xp_a.sqlite", [(1, 50.0, 10.0, 10.0)])
        with GaiaCatalog(tmp_path) as cat:
            assert cat.installed_bands == ["gaia_xp_a.sqlite"]
        assert cat._connections == {}


# ── Downloader tests (no network) ───────────────────────────────────────


class TestGaiaDownloaderUrlsAndStatus:
    def test_download_url_construction(self):
        url = gaia_download_url("gaia_xp_lt8.sqlite")
        assert url == "https://f005.backblazeb2.com/file/setiastro-gaia/gaia_xp_lt8.sqlite"

    def test_all_bands_have_valid_filenames(self):
        assert len(GAIA_BANDS) >= 5
        for band in GAIA_BANDS:
            assert band.filenames
            for fname in band.filenames:
                assert fname.startswith("gaia_xp_")
                assert fname.endswith(".sqlite")

    def test_band_by_key_roundtrip(self):
        band = band_by_key("ultra_bright")
        assert band.key == "ultra_bright"
        with pytest.raises(KeyError):
            band_by_key("not_a_real_band")

    def test_default_gaia_dir_honors_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ASTRAIOS_GAIA_DIR", str(tmp_path / "custom_gaia"))
        d = default_gaia_dir()
        assert d == tmp_path / "custom_gaia"
        assert d.exists()

    def test_installed_files_and_band_status(self, tmp_path):
        band = band_by_key("ultra_bright")  # single-file band
        assert installed_files(tmp_path) == []
        installed, missing = band_status(band, tmp_path)
        assert installed == []
        assert missing == band.filenames

        (tmp_path / band.filenames[0]).write_bytes(b"fake sqlite bytes")
        installed, missing = band_status(band, tmp_path)
        assert installed == band.filenames
        assert missing == []

    def test_download_file_does_not_touch_network(self, tmp_path, monkeypatch):
        """download_file() must go through urlopen — mock it and verify the
        request URL + streamed bytes, with no real socket ever opened."""
        payload = b"0123456789" * 100
        requested_urls = []

        class _FakeResponse:
            def __init__(self, data: bytes):
                self._data = data
                self._pos = 0
                self.headers = {"Content-Length": str(len(data))}

            def read(self, n):
                chunk = self._data[self._pos : self._pos + n]
                self._pos += len(chunk)
                return chunk

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def _fake_urlopen(request, timeout=0):
            requested_urls.append(request.full_url)
            return _FakeResponse(payload)

        monkeypatch.setattr("astraios.core.gaia_catalog.urlopen", _fake_urlopen)

        dest = download_file("gaia_xp_lt8.sqlite", tmp_path)

        assert requested_urls == [gaia_download_url("gaia_xp_lt8.sqlite")]
        assert dest == tmp_path / "gaia_xp_lt8.sqlite"
        assert dest.read_bytes() == payload
        assert not dest.with_suffix(dest.suffix + ".tmp").exists()

    def test_download_file_skips_existing(self, tmp_path, monkeypatch):
        dest = tmp_path / "gaia_xp_lt8.sqlite"
        dest.write_bytes(b"already here")

        def _fail_if_called(*a, **k):
            raise AssertionError("should not attempt a network call")

        monkeypatch.setattr("astraios.core.gaia_catalog.urlopen", _fail_if_called)

        result = download_file("gaia_xp_lt8.sqlite", tmp_path)
        assert result == dest
        assert dest.read_bytes() == b"already here"

    def test_download_band_only_fetches_missing_files(self, tmp_path, monkeypatch):
        band = band_by_key("medium")  # multi-file band
        # Pre-install the first file.
        (tmp_path / band.filenames[0]).write_bytes(b"already installed")

        fetched = []

        def _fake_download_file(filename, catalog_dir=None, **kwargs):
            fetched.append(filename)
            path = tmp_path / filename
            path.write_bytes(b"downloaded")
            return path

        monkeypatch.setattr("astraios.core.gaia_catalog.download_file", _fake_download_file)

        download_band("medium", tmp_path)

        assert band.filenames[0] not in fetched
        assert set(fetched) == set(band.filenames[1:])


# ── Solver tests ─────────────────────────────────────────────────────────


class TestGaiaSolver:
    def test_recovers_known_wcs_from_hint(self, tmp_path):
        catalog_dir, image, _true_wcs = _build_synthetic_field(tmp_path)

        params = GaiaSolveParams(
            ra_hint=RA0 + 0.01,
            dec_hint=DEC0 - 0.008,
            scale_hint=SCALE_TRUE * 1.03,
            rotation_hint=ROT_TRUE,
            mag_limit=16.0,
            max_stars=200,
            max_catalog_stars=400,
            match_tolerance_px=8.0,
            min_matches=6,
            quality_min_pairs=10,
            catalog_dir=catalog_dir,
        )

        result = solve_with_gaia_catalog(image, params)

        assert result.success, result.message
        assert result.ra_center == pytest.approx(RA0, abs=0.02)
        assert result.dec_center == pytest.approx(DEC0, abs=0.02)
        assert result.pixel_scale == pytest.approx(SCALE_TRUE, rel=0.02)
        assert result.n_stars_matched >= params.quality_min_pairs
        assert result.wcs_header is not None
        assert "CRVAL1" in result.wcs_header

    def test_plate_solve_gaia_adapter_returns_solver_dict(self, tmp_path):
        catalog_dir, image, _true_wcs = _build_synthetic_field(tmp_path)
        params = GaiaSolveParams(
            rotation_hint=ROT_TRUE,
            mag_limit=16.0,
            max_catalog_stars=400,
            match_tolerance_px=8.0,
            quality_min_pairs=10,
            catalog_dir=catalog_dir,
        )

        d = plate_solve_gaia(
            image,
            ra_hint=RA0 + 0.01,
            dec_hint=DEC0 - 0.008,
            scale_hint=SCALE_TRUE * 1.03,
            params=params,
        )

        assert d is not None
        assert d["ra"] == pytest.approx(RA0, abs=0.02)
        assert d["dec"] == pytest.approx(DEC0, abs=0.02)
        assert d["scale"] == pytest.approx(SCALE_TRUE, rel=0.02)
        assert "CRVAL1" in d["wcs_header"]

    def test_no_ra_dec_hint_fails_cleanly(self, tmp_path):
        image = np.zeros((100, 100), dtype=np.float32)
        result = solve_with_gaia_catalog(image, GaiaSolveParams(catalog_dir=tmp_path))
        assert not result.success
        assert "RA/Dec" in result.message

    def test_no_scale_hint_fails_cleanly(self, tmp_path):
        image = np.zeros((100, 100), dtype=np.float32)
        params = GaiaSolveParams(ra_hint=10.0, dec_hint=10.0, catalog_dir=tmp_path)
        result = solve_with_gaia_catalog(image, params)
        assert not result.success
        assert "pixel scale" in result.message

    def test_missing_catalog_fails_with_clear_message(self, tmp_path):
        image = np.zeros((200, 200), dtype=np.float32)
        params = GaiaSolveParams(
            ra_hint=10.0, dec_hint=10.0, scale_hint=1.0, catalog_dir=tmp_path / "empty"
        )
        result = solve_with_gaia_catalog(image, params)
        assert not result.success
        assert "download_band" in result.message

    def test_blank_image_no_stars_fails_cleanly(self, tmp_path):
        catalog_dir, _image, _true_wcs = _build_synthetic_field(tmp_path)
        blank = np.full((200, 200), 0.5, dtype=np.float32)
        params = GaiaSolveParams(
            ra_hint=RA0, dec_hint=DEC0, scale_hint=SCALE_TRUE, catalog_dir=catalog_dir
        )
        result = solve_with_gaia_catalog(blank, params)
        assert not result.success
        assert "too few" in result.message or "stars" in result.message
