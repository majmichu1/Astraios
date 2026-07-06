"""Tests for the finder chart renderer (ported from Seti Astro Suite Pro)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from astraios.core.finder_chart import (
    FinderChartParams,
    _fit_canvas,
    _make_background_canvas,
    _pixel_scale_arcsec,
    _prepare_markers,
    _to_wcs,
    _world_to_pixel,
    render_finder_chart,
)

SIZE = 300
CENTER_RA = 83.822
CENTER_DEC = -5.391
SCALE_DEG = 0.0008  # deg/px, tangent-plane


def _make_header(size=SIZE, ra=CENTER_RA, dec=CENTER_DEC, scale=SCALE_DEG):
    hdr = fits.Header()
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    # FITS CRPIX is 1-indexed; +1 here makes the *0-indexed* array/WCS pixel
    # (as returned by world_to_pixel) land exactly on size/2.
    hdr["CRPIX1"] = size / 2.0 + 1.0
    hdr["CRPIX2"] = size / 2.0 + 1.0
    hdr["CRVAL1"] = ra
    hdr["CRVAL2"] = dec
    hdr["CDELT1"] = -scale
    hdr["CDELT2"] = scale
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    return hdr


def _make_image(size=SIZE):
    rng = np.random.default_rng(42)
    img = rng.normal(0.1, 0.01, size=(3, size, size)).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def _offset_radec(ra, dec, dx_px, dy_px, scale=SCALE_DEG):
    """A catalog entry offset from the field center by (dx_px, dy_px) pixels,
    computed independently of the module under test (simple tangent-plane math,
    consistent with CDELT1<0 = east-is-negative-x, CDELT2>0 = north-is-positive-y).
    """
    import math

    cosd = math.cos(math.radians(dec))
    d_ra = -dx_px * scale / cosd
    d_dec = dy_px * scale
    return ra + d_ra, dec + d_dec


class TestToWcs:
    def test_accepts_header_dict_and_wcs(self):
        hdr = _make_header()
        w1 = _to_wcs(hdr)
        w2 = _to_wcs(dict(hdr))
        w3 = _to_wcs(w1)
        assert isinstance(w1, WCS)
        assert isinstance(w2, WCS)
        assert w3 is w1

    def test_rejects_unsupported_type(self):
        with pytest.raises(TypeError):
            _to_wcs(12345)


class TestPixelScale:
    def test_matches_known_cdelt(self):
        wcs = _to_wcs(_make_header())
        arcsec_per_px = _pixel_scale_arcsec(wcs)
        assert arcsec_per_px == pytest.approx(SCALE_DEG * 3600.0, rel=1e-3)


class TestWorldToPixel:
    def test_center_projects_to_image_center(self):
        wcs = _to_wcs(_make_header())
        xs, ys = _world_to_pixel(wcs, np.array([CENTER_RA]), np.array([CENTER_DEC]))
        assert xs[0] == pytest.approx(SIZE / 2.0, abs=0.5)
        assert ys[0] == pytest.approx(SIZE / 2.0, abs=0.5)

    def test_offset_star_projects_to_expected_pixel(self):
        wcs = _to_wcs(_make_header())
        ra, dec = _offset_radec(CENTER_RA, CENTER_DEC, dx_px=40, dy_px=-25)
        xs, ys = _world_to_pixel(wcs, np.array([ra]), np.array([dec]))
        assert xs[0] == pytest.approx(SIZE / 2.0 + 40, abs=0.5)
        assert ys[0] == pytest.approx(SIZE / 2.0 - 25, abs=0.5)


class TestPrepareMarkers:
    def test_filters_by_magnitude_and_sorts_brightest_first(self):
        wcs = _to_wcs(_make_header())
        entries = [
            {"name": "faint", "ra_deg": CENTER_RA, "dec_deg": CENTER_DEC, "mag": 15.0},
            {"name": "bright", "ra_deg": CENTER_RA, "dec_deg": CENTER_DEC, "mag": 2.0},
        ]
        kept = _prepare_markers(entries, wcs, SIZE, SIZE, mag_limit=10.0, max_labels=10, cell_px=1)
        names = [k["name"] for k in kept]
        assert "faint" not in names
        assert names == ["bright"]

    def test_declutter_keeps_one_per_cell(self):
        wcs = _to_wcs(_make_header())
        ra, dec = _offset_radec(CENTER_RA, CENTER_DEC, 0, 0)
        entries = [
            {"name": "a", "ra_deg": ra, "dec_deg": dec, "mag": 3.0},
            {"name": "b", "ra_deg": ra, "dec_deg": dec, "mag": 4.0},
        ]
        kept = _prepare_markers(entries, wcs, SIZE, SIZE, mag_limit=None, max_labels=10, cell_px=50)
        assert len(kept) == 1
        assert kept[0]["name"] == "a"  # brighter one wins the cell

    def test_out_of_frame_entries_are_dropped(self):
        wcs = _to_wcs(_make_header())
        entries = [{"name": "far", "ra_deg": CENTER_RA + 30.0, "dec_deg": CENTER_DEC, "mag": 1.0}]
        kept = _prepare_markers(entries, wcs, SIZE, SIZE, mag_limit=None, max_labels=10, cell_px=1)
        assert kept == []


class TestFitCanvas:
    def test_noop_when_size_matches(self):
        wcs = _to_wcs(_make_header())
        canvas = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        out, wcs_out = _fit_canvas(canvas, wcs, SIZE)
        assert out.shape == (SIZE, SIZE, 3)
        assert wcs_out is wcs

    def test_crop_shifts_crpix(self):
        wcs = _to_wcs(_make_header())
        canvas = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        out, wcs_out = _fit_canvas(canvas, wcs, 100)
        assert out.shape == (100, 100, 3)
        # center should still map to the (now-cropped) canvas center
        xs, ys = wcs_out.world_to_pixel(SkyCoord(CENTER_RA, CENTER_DEC, unit="deg"))
        assert float(xs) == pytest.approx(50.0, abs=0.5)
        assert float(ys) == pytest.approx(50.0, abs=0.5)

    def test_pad_shifts_crpix(self):
        wcs = _to_wcs(_make_header())
        canvas = np.full((SIZE, SIZE, 3), 7, dtype=np.uint8)
        out, wcs_out = _fit_canvas(canvas, wcs, 400)
        assert out.shape == (400, 400, 3)
        # original content should be centered inside the padded canvas
        pad = (400 - SIZE) // 2
        assert np.all(out[pad, pad] == 7)


class TestMakeBackgroundCanvas:
    def test_black_background(self):
        params = FinderChartParams(background="black")
        canvas = _make_background_canvas(_make_image(), params)
        assert canvas.shape == (SIZE, SIZE, 3)
        assert canvas.max() == 0

    def test_white_background(self):
        params = FinderChartParams(background="white")
        canvas = _make_background_canvas(_make_image(), params)
        assert np.all(canvas == 255)

    def test_image_background_produces_rgb(self):
        params = FinderChartParams(background="image")
        canvas = _make_background_canvas(_make_image(), params)
        assert canvas.shape == (SIZE, SIZE, 3)
        assert canvas.dtype == np.uint8

    def test_mono_image_supported(self):
        rng = np.random.default_rng(0)
        mono = np.clip(rng.normal(0.1, 0.01, size=(SIZE, SIZE)), 0, 1).astype(np.float32)
        params = FinderChartParams(background="image")
        canvas = _make_background_canvas(mono, params)
        assert canvas.shape == (SIZE, SIZE, 3)


class TestRenderFinderChart:
    def _stars(self):
        ra1, dec1 = _offset_radec(CENTER_RA, CENTER_DEC, 80, 50)
        ra2, dec2 = _offset_radec(CENTER_RA, CENTER_DEC, -60, -70)
        return [
            {"name": "Star A", "ra_deg": ra1, "dec_deg": dec1, "mag": 3.0},
            {"name": "Star B", "ra_deg": ra2, "dec_deg": dec2, "mag": 5.0},
        ]

    def _dsos(self):
        ra, dec = _offset_radec(CENTER_RA, CENTER_DEC, 20, -20)
        return [{"name": "M42", "ra_deg": ra, "dec_deg": dec, "mag": 4.0, "size_arcmin": 6.0}]

    def test_output_shape_and_dtype(self):
        img = _make_image()
        out = render_finder_chart(img, _make_header(), catalog_stars=self._stars())
        assert out.shape == (3, SIZE, SIZE)
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_star_markers_drawn_at_projected_positions(self):
        img = _make_image()
        params = FinderChartParams(
            background="black", show_dso=False, show_compass=False,
            show_scale_bar=False, show_field_marker=False, star_color=(255, 0, 0),
        )
        out = render_finder_chart(img, _make_header(), params=params, catalog_stars=self._stars())
        rgb = np.transpose(out, (1, 2, 0))

        wcs = _to_wcs(_make_header())
        for star in self._stars():
            xs, ys = _world_to_pixel(wcs, np.array([star["ra_deg"]]), np.array([star["dec_deg"]]))
            x, y = int(round(xs[0])), int(round(ys[0]))
            patch = rgb[max(0, y - 6):y + 6, max(0, x - 6):x + 6]
            # red marker channel should dominate near the projected star position
            assert patch[..., 0].max() > patch[..., 1].max()

    def test_dso_size_circle_present(self):
        img = _make_image()
        params = FinderChartParams(
            background="black", show_stars=False, show_compass=False,
            show_scale_bar=False, show_field_marker=False,
            dso_color=(0, 255, 0), dso_circle_color=(0, 0, 255),
        )
        out = render_finder_chart(img, _make_header(), params=params, dso_list=self._dsos())
        rgb = np.transpose(out, (1, 2, 0))
        assert np.any((rgb[..., 2] > 0) & (rgb[..., 0] == 0) & (rgb[..., 1] == 0))

    def test_compass_and_scale_bar_present_as_non_background_pixels(self):
        img = _make_image()
        params = FinderChartParams(
            background="black", show_stars=False, show_dso=False, show_field_marker=False,
        )
        out = render_finder_chart(img, _make_header(), params=params)
        rgb = np.transpose(out, (1, 2, 0))
        assert np.any(rgb > 0)

    def test_toggles_remove_annotations(self):
        img = _make_image()
        params_off = FinderChartParams(
            background="black", show_stars=False, show_dso=False, show_compass=False,
            show_scale_bar=False, show_field_marker=False, show_grid=False,
        )
        out_off = render_finder_chart(img, _make_header(), params=params_off)
        assert out_off.max() == 0.0

        params_on = FinderChartParams(
            background="black", show_stars=False, show_dso=False, show_compass=True,
            show_scale_bar=True, show_field_marker=True, show_grid=True,
        )
        out_on = render_finder_chart(img, _make_header(), params=params_on)
        assert out_on.max() > 0.0

    def test_show_stars_false_hides_star_markers(self):
        img = _make_image()
        params_on = FinderChartParams(
            background="black", show_dso=False, show_compass=False,
            show_scale_bar=False, show_field_marker=False,
        )
        params_off = FinderChartParams(
            background="black", show_stars=False, show_dso=False, show_compass=False,
            show_scale_bar=False, show_field_marker=False,
        )
        stars = self._stars()
        out_on = render_finder_chart(img, _make_header(), params=params_on, catalog_stars=stars)
        out_off = render_finder_chart(img, _make_header(), params=params_off, catalog_stars=stars)
        assert out_on.max() > 0.0
        assert out_off.max() == 0.0

    def test_fov_box_toggle(self):
        img = _make_image()
        common = dict(
            background="black", show_stars=False, show_dso=False, show_compass=False,
            show_scale_bar=False, show_field_marker=False,
        )
        # Sensor sized so the projected FOV box edges actually cross the 300x300 canvas.
        out_off = render_finder_chart(img, _make_header(), params=FinderChartParams(**common))
        out_on = render_finder_chart(
            img, _make_header(),
            params=FinderChartParams(
                show_fov_box=True, sensor_w_px=600, sensor_h_px=450, **common
            ),
        )
        assert out_off.max() == 0.0
        assert out_on.max() > 0.0

    def test_out_size_changes_output_shape(self):
        img = _make_image()
        params = FinderChartParams(out_size=150)
        out = render_finder_chart(img, _make_header(), params=params, catalog_stars=self._stars())
        assert out.shape == (3, 150, 150)

    def test_no_wcs_dso_or_stars_still_renders(self):
        img = _make_image()
        out = render_finder_chart(img, _make_header())
        assert out.shape == (3, SIZE, SIZE)
