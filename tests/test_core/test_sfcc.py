"""Tests for Spectral Flux Color Calibration (SFCC)."""

from __future__ import annotations

import numpy as np
import pytest

from astraios.core.sfcc import (
    FILTER_CURVES,
    SENSOR_QE_CURVES,
    SFCCParams,
    apply_sfcc,
    bp_rp_to_teff,
    build_system_response,
    expected_channel_ratios,
    sfcc_calibrate,
    wavelength_grid,
)

# ── Synthetic star field with a known, physically-consistent color cast ─────

_BP_RP = 0.7  # roughly G2V-like
_CAST = (1.4, 1.0, 0.6)  # (R, G, B) systematic instrumental gain to recover
_GRID_POSITIONS = [
    (x, y) for y in (40, 110, 180) for x in (40, 110, 180)
]  # 9 well-separated stars, >= SFCCParams.min_stars (8)


def _expected_ratio_for_default_params(bp_rp: float) -> tuple[float, float, float]:
    """Same physics apply_sfcc will use internally, for building the synthetic star field."""
    params = SFCCParams()
    grid = wavelength_grid()
    sensor_curve = SENSOR_QE_CURVES[params.sensor]
    t_r = build_system_response(grid, FILTER_CURVES[params.filter_r], sensor_curve)
    t_g = build_system_response(grid, FILTER_CURVES[params.filter_g], sensor_curve)
    t_b = build_system_response(grid, FILTER_CURVES[params.filter_b], sensor_curve)
    teff = bp_rp_to_teff(bp_rp)
    s_r, s_g, s_b = expected_channel_ratios(teff, grid, t_r, t_g, t_b)
    return s_r / s_g, 1.0, s_b / s_g


def _cast_star_field(
    size: int = 220, sigma: float = 2.5, amplitude: float = 0.5,
) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    """RGB image with stars whose R/G/B ratio deliberately differs from the
    physically expected ratio by a fixed multiplicative cast — the thing SFCC
    is supposed to detect and undo."""
    r_exp, g_exp, b_exp = _expected_ratio_for_default_params(_BP_RP)
    cast_r, cast_g, cast_b = _CAST

    image = np.zeros((3, size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    catalog_stars: list[tuple[float, float, float]] = []

    rng = np.random.default_rng(0)
    for sx, sy in _GRID_POSITIONS:
        amp = amplitude * (1.0 + 0.1 * rng.standard_normal())  # varying brightness, same cast
        dist_sq = (xx - sx) ** 2 + (yy - sy) ** 2
        profile = np.exp(-dist_sq / (2 * sigma**2))
        image[0] += amp * cast_r * r_exp * profile
        image[1] += amp * cast_g * g_exp * profile
        image[2] += amp * cast_b * b_exp * profile
        catalog_stars.append((float(sx), float(sy), _BP_RP))

    return np.clip(image, 0, 1).astype(np.float32), catalog_stars


class TestFluxIntegration:
    """The ported physical core: filter x QE x flux integration."""

    def test_ideal_system_response_is_unity(self):
        grid = wavelength_grid()
        t_sys = build_system_response(
            grid, FILTER_CURVES["Clear / No Filter"], SENSOR_QE_CURVES["Ideal (100% QE)"]
        )
        np.testing.assert_allclose(t_sys, 1.0)

    def test_hot_star_is_bluer_than_cool_star(self):
        """A known blackbody/filter combo should give sane (monotonic) ratios:
        hotter stars have relatively more blue flux than cooler stars."""
        grid = wavelength_grid()
        t_r = build_system_response(
            grid, FILTER_CURVES["Broadband-R (generic LRGB interference)"],
            SENSOR_QE_CURVES["Ideal (100% QE)"],
        )
        t_g = build_system_response(
            grid, FILTER_CURVES["Broadband-G (generic LRGB interference)"],
            SENSOR_QE_CURVES["Ideal (100% QE)"],
        )
        t_b = build_system_response(
            grid, FILTER_CURVES["Broadband-B (generic LRGB interference)"],
            SENSOR_QE_CURVES["Ideal (100% QE)"],
        )

        s_r_hot, _, s_b_hot = expected_channel_ratios(30000.0, grid, t_r, t_g, t_b)
        s_r_cool, _, s_b_cool = expected_channel_ratios(3500.0, grid, t_r, t_g, t_b)

        assert (s_b_hot / s_r_hot) > (s_b_cool / s_r_cool)

    def test_ratios_are_finite_and_positive(self):
        grid = wavelength_grid()
        t_r = build_system_response(
            grid, FILTER_CURVES["Bayer-R (generic OSC)"],
            SENSOR_QE_CURVES["Generic CMOS back-illuminated (Sony IMX-class)"],
        )
        t_g = build_system_response(
            grid, FILTER_CURVES["Bayer-G (generic OSC)"],
            SENSOR_QE_CURVES["Generic CMOS back-illuminated (Sony IMX-class)"],
        )
        t_b = build_system_response(
            grid, FILTER_CURVES["Bayer-B (generic OSC)"],
            SENSOR_QE_CURVES["Generic CMOS back-illuminated (Sony IMX-class)"],
        )
        for teff in (3500.0, 5778.0, 9600.0, 30000.0):
            s_r, s_g, s_b = expected_channel_ratios(teff, grid, t_r, t_g, t_b)
            for s in (s_r, s_g, s_b):
                assert np.isfinite(s)
                assert s > 0


class TestSFCCCalibrate:
    def test_reduces_known_color_cast(self):
        image, catalog_stars = _cast_star_field()
        _result, sfcc_result = sfcc_calibrate(image, catalog_stars)

        # The fitted per-channel scale should recover the injected cast
        # (within a modest tolerance for aperture photometry noise).
        np.testing.assert_allclose(sfcc_result.scales, _CAST, rtol=0.1)
        assert sfcc_result.n_used >= 8
        assert sfcc_result.rms_residual < 0.1

    def test_correction_actually_changes_the_image_ratio(self):
        image, catalog_stars = _cast_star_field()
        corrected, _ = sfcc_calibrate(image, catalog_stars)

        r_exp, _g_exp, b_exp = _expected_ratio_for_default_params(_BP_RP)
        sx, sy = _GRID_POSITIONS[4]  # center star

        def ratios(img):
            patch = img[:, sy - 3 : sy + 4, sx - 3 : sx + 4]
            peak = patch.reshape(3, -1).max(axis=1)
            return peak[0] / peak[1], peak[2] / peak[1]

        before_rg, before_bg = ratios(image)
        after_rg, after_bg = ratios(corrected)

        # Corrected ratios should land closer to the physically-expected
        # ratio than the original, cast-corrupted ones.
        assert abs(after_rg - r_exp) < abs(before_rg - r_exp)
        assert abs(after_bg - b_exp) < abs(before_bg - b_exp)

    def test_deterministic(self):
        image, catalog_stars = _cast_star_field()
        result1, sfcc1 = sfcc_calibrate(image, catalog_stars)
        result2, sfcc2 = sfcc_calibrate(image, catalog_stars)
        np.testing.assert_array_equal(result1, result2)
        assert sfcc1.scales == sfcc2.scales

    def test_mono_raises(self):
        mono = np.ones((100, 100), dtype=np.float32) * 0.5
        with pytest.raises(ValueError, match="3-channel"):
            sfcc_calibrate(mono, [(50.0, 50.0, 0.7)])

    def test_too_few_stars_raises(self):
        image, catalog_stars = _cast_star_field()
        params = SFCCParams(min_stars=50)
        with pytest.raises(ValueError, match="usable stars"):
            sfcc_calibrate(image, catalog_stars, params=params)

    def test_unknown_filter_raises(self):
        image, catalog_stars = _cast_star_field()
        params = SFCCParams(filter_r="Nonexistent Filter Name")
        with pytest.raises(ValueError, match="Unknown filter"):
            sfcc_calibrate(image, catalog_stars, params=params)

    def test_unknown_sensor_raises(self):
        image, catalog_stars = _cast_star_field()
        params = SFCCParams(sensor="Nonexistent Sensor Name")
        with pytest.raises(ValueError, match="Unknown sensor"):
            sfcc_calibrate(image, catalog_stars, params=params)

    def test_custom_filter_curve_override(self):
        image, catalog_stars = _cast_star_field()
        custom = {"My Filter": FILTER_CURVES["Broadband-R (generic LRGB interference)"]}
        params = SFCCParams(filter_r="My Filter", custom_filter_curves=custom)
        _result, sfcc_result = sfcc_calibrate(image, catalog_stars, params=params)
        assert sfcc_result.n_used >= 8

    def test_reference_ratios_are_diagnostic_only(self):
        image, catalog_stars = _cast_star_field()
        _result, sfcc_result = sfcc_calibrate(image, catalog_stars)
        assert len(sfcc_result.reference_ratios) == 3
        assert sfcc_result.reference_ratios[1] == 1.0


class TestApplySFCC:
    def test_apply_sfcc_with_catalog_stars_kwarg(self):
        image, catalog_stars = _cast_star_field()
        corrected = apply_sfcc(image, catalog_stars=catalog_stars)
        assert isinstance(corrected, np.ndarray)
        assert corrected.shape == image.shape
        assert corrected.dtype == np.float32

    def test_apply_sfcc_mono_raises(self):
        mono = np.ones((1, 100, 100), dtype=np.float32)[0]
        with pytest.raises(ValueError, match="3-channel"):
            apply_sfcc(mono, catalog_stars=[(50.0, 50.0, 0.7)])

    def test_apply_sfcc_no_catalog_no_wcs_raises(self):
        image, _catalog_stars = _cast_star_field()
        with pytest.raises(ValueError, match="wcs_header"):
            apply_sfcc(image)

    def test_apply_sfcc_offline_catalog_raises_clear_error(self):
        image, catalog_stars = _cast_star_field()
        params = SFCCParams(catalog="offline_gaia")
        with pytest.raises(ValueError, match="BP/RP"):
            apply_sfcc(image, params=params, wcs_header={"ra": 10.0, "dec": 20.0})


class TestGainInvariance:
    """The defining property of a colour calibrator, independent of whatever
    stellar model it uses internally: applying a per-channel gain to the input
    must not change the calibrated COLOUR.

    Overall brightness is deliberately not restored -- SFCC balances channels
    relative to the reference channel, it is not an exposure correction -- so
    the two results may differ by one uniform scale factor across all
    channels, but never by a per-channel one.
    """

    @staticmethod
    def _field():
        rng = np.random.default_rng(3)
        h = w = 300
        yy, xx = np.mgrid[0:h, 0:w]
        n = 45
        xs = rng.uniform(30, w - 30, n)
        ys = rng.uniform(30, h - 30, n)
        bp_rp = rng.uniform(0.2, 2.0, n)
        amps = rng.uniform(0.10, 0.35, n)
        img = np.full((3, h, w), 0.02, np.float32)
        for x, y, c, a in zip(xs, ys, bp_rp, amps):
            g = np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0**2))).astype(np.float32)
            rgb = np.array([0.5 + 0.35 * c, 1.0, 1.6 - 0.45 * c], np.float32)
            rgb /= rgb.mean()
            for ch in range(3):
                img[ch] += a * rgb[ch] * g
        cat = [(float(x), float(y), float(c)) for x, y, c in zip(xs, ys, bp_rp)]
        return np.clip(img, 0, 1), cat

    def test_colour_is_invariant_under_per_channel_gain(self):
        img, cat = self._field()
        params = SFCCParams(min_stars=10, detection_sigma=3.0,
                            neutralize_background=False)
        gain = np.array([0.9, 0.8, 0.7], np.float32)
        gained = img * gain[:, None, None]
        assert gained.max() <= 1.0, "test setup must not clip"

        out_a = apply_sfcc(img, params=params, catalog_stars=cat)
        out_b = apply_sfcc(gained, params=params, catalog_stars=cat)

        # per-channel ratio between the two calibrated results
        ratios = []
        for ch in range(3):
            sel = out_a[ch] > 0.05
            ratios.append(float(np.median(out_b[ch][sel] / out_a[ch][sel])))

        # all three channels must be scaled by the SAME factor: colour matched.
        assert max(ratios) - min(ratios) < 0.02, (
            f"per-channel ratios {ratios} differ: a colour cast survived calibration"
        )

    def test_identity_gain_is_a_no_op(self):
        img, cat = self._field()
        params = SFCCParams(min_stars=10, detection_sigma=3.0,
                            neutralize_background=False)
        a = apply_sfcc(img, params=params, catalog_stars=cat)
        b = apply_sfcc(img.copy(), params=params, catalog_stars=cat)
        assert np.allclose(a, b), "SFCC is not deterministic"
