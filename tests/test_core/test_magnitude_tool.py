"""Tests for the magnitude tool (ported from Seti Astro Suite Pro)."""

from __future__ import annotations

import numpy as np
import pytest

from astraios.core.magnitude_tool import (
    MagnitudeParams,
    MagnitudeResult,
    ReferenceStar,
    _estimate_limiting_magnitude,
    _fit_zero_point,
    measure_magnitudes,
)

SIZE = 200


def _add_gaussian_star(img, x, y, amp, sigma=2.2):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float64)
    img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))


def _synthetic_field(star_specs, size=SIZE, noise=0.002, seed=0):
    """star_specs: list of (x, y, amplitude)."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0.01, noise, size=(size, size)).astype(np.float64)
    img = np.clip(img, 0.0, None)
    for x, y, amp in star_specs:
        _add_gaussian_star(img, x, y, amp)
    return img.astype(np.float32)


def _match_by_position(result: MagnitudeResult, x, y, tol=3.0):
    d2 = (result.x - x) ** 2 + (result.y - y) ** 2
    i = int(np.argmin(d2))
    assert d2[i] <= tol**2, f"No detected star near ({x},{y})"
    return i


class TestMeasureMagnitudesBasic:
    def test_detects_all_synthetic_stars(self):
        specs = [(50, 50, 1.0), (100, 60, 0.4), (150, 150, 0.1)]
        img = _synthetic_field(specs)
        params = MagnitudeParams(aperture_radius=6, annulus_inner=9, annulus_outer=13)
        result = measure_magnitudes(img, params=params)
        assert result.n_stars >= 3
        assert result.calibrated_mag is None  # no zero point / reference stars supplied
        assert result.zero_point is None

    def test_no_sources_returns_empty_result(self):
        img = np.zeros((SIZE, SIZE), dtype=np.float32)
        result = measure_magnitudes(img)
        assert result.n_stars == 0
        assert result.x.size == 0
        assert result.calibrated_mag is None

    def test_progress_callback_invoked(self):
        specs = [(50, 50, 1.0), (100, 60, 0.4)]
        img = _synthetic_field(specs)
        calls = []
        measure_magnitudes(img, progress=lambda f, m: calls.append((f, m)))
        assert calls[0][0] == 0.0
        assert calls[-1][0] == 1.0
        assert len(calls) >= 2


class TestInstrumentalMagnitudeRatios:
    def test_brighter_star_has_smaller_magnitude(self):
        specs = [(50, 50, 1.0), (150, 150, 0.2)]
        img = _synthetic_field(specs)
        params = MagnitudeParams(aperture_radius=6, annulus_inner=9, annulus_outer=13)
        result = measure_magnitudes(img, params=params)

        i_bright = _match_by_position(result, 50, 50)
        i_faint = _match_by_position(result, 150, 150)
        assert result.instrumental_mag[i_bright] < result.instrumental_mag[i_faint]

    def test_magnitude_difference_matches_flux_ratio(self):
        specs = [(60, 60, 1.0), (140, 60, 0.25), (60, 140, 0.05)]
        img = _synthetic_field(specs, noise=0.0005)
        params = MagnitudeParams(aperture_radius=7, annulus_inner=10, annulus_outer=15)
        result = measure_magnitudes(img, params=params)

        idx = [_match_by_position(result, x, y) for x, y, _ in specs]
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            ia, ib = idx[a], idx[b]
            expected = -2.5 * np.log10(result.flux[ia] / result.flux[ib])
            actual = result.instrumental_mag[ia] - result.instrumental_mag[ib]
            assert actual == pytest.approx(expected, abs=1e-6)


class TestZeroPointCalibration:
    def test_fixed_zero_point_calibrates_directly(self):
        specs = [(50, 50, 1.0), (150, 150, 0.2)]
        img = _synthetic_field(specs)
        params = MagnitudeParams(
            aperture_radius=6, annulus_inner=9, annulus_outer=13, zero_point=25.0
        )
        result = measure_magnitudes(img, params=params)
        assert result.zero_point == 25.0
        np.testing.assert_allclose(result.calibrated_mag, result.instrumental_mag + 25.0)

    def test_reference_stars_recover_calibrated_magnitudes(self):
        specs = [(50, 50, 1.0), (100, 100, 0.3), (150, 60, 0.05)]
        img = _synthetic_field(specs, noise=0.0003)
        params = MagnitudeParams(
            aperture_radius=6, annulus_inner=9, annulus_outer=13, match_radius_px=3.0
        )

        # Assign catalog magnitudes for the two brightest stars consistent with a known
        # zero point (derived from their *measured* instrumental mags, not invented
        # numbers), then verify the third (non-reference) star recovers its true
        # calibrated magnitude under that same zero point.
        known_zp = 20.0
        probe = measure_magnitudes(img, params=params)
        ref_stars = []
        for x, y in [(50, 50), (100, 100)]:
            i = _match_by_position(probe, x, y)
            ref_stars.append(
                ReferenceStar(
                    x=float(probe.x[i]), y=float(probe.y[i]),
                    catalog_mag=float(probe.instrumental_mag[i]) + known_zp,
                )
            )

        result = measure_magnitudes(img, params=params, reference_stars=ref_stars)
        assert result.zero_point == pytest.approx(known_zp, abs=1e-6)
        assert result.zero_point_n == 2

        i_test = _match_by_position(result, 150, 60)
        expected = float(result.instrumental_mag[i_test]) + known_zp
        assert result.calibrated_mag[i_test] == pytest.approx(expected, abs=1e-6)

    def test_reference_stars_accept_dict_entries(self):
        specs = [(50, 50, 1.0), (150, 150, 0.2)]
        img = _synthetic_field(specs)
        params = MagnitudeParams(aperture_radius=6, annulus_inner=9, annulus_outer=13)
        probe = measure_magnitudes(img, params=params)
        i = _match_by_position(probe, 50, 50)
        ref = [{"x": float(probe.x[i]), "y": float(probe.y[i]), "catalog_mag": 8.0}]
        result = measure_magnitudes(img, params=params, reference_stars=ref)
        assert result.zero_point is not None

    def test_no_reference_or_zero_point_leaves_uncalibrated(self):
        specs = [(50, 50, 1.0)]
        img = _synthetic_field(specs)
        result = measure_magnitudes(img)
        assert result.zero_point is None
        assert result.calibrated_mag is None

    def test_fit_zero_point_rejects_out_of_radius_matches(self):
        xs = np.array([10.0, 200.0])
        ys = np.array([10.0, 200.0])
        inst_mag = np.array([-5.0, -3.0])
        refs = [ReferenceStar(x=10.0, y=10.0, catalog_mag=12.0)]
        params = MagnitudeParams(match_radius_px=2.0)
        zp, zp_std, n = _fit_zero_point(xs, ys, inst_mag, refs, params)
        assert n == 1
        assert zp == pytest.approx(12.0 - (-5.0))


class TestLimitingMagnitude:
    def test_limiting_mag_is_finite_and_sane(self):
        specs = [(50, 50, 1.0)]
        img = _synthetic_field(specs)
        params = MagnitudeParams(aperture_radius=6, annulus_inner=9, annulus_outer=13)
        result = measure_magnitudes(img, params=params)
        assert result.limiting_mag is not None
        assert np.isfinite(result.limiting_mag)

    def test_limiting_mag_shifts_with_zero_point(self):
        img = _synthetic_field([(50, 50, 1.0)])
        instrumental = _estimate_limiting_magnitude(img, MagnitudeParams(aperture_radius=6), None)
        calibrated = _estimate_limiting_magnitude(img, MagnitudeParams(aperture_radius=6), 20.0)
        assert calibrated == pytest.approx(instrumental + 20.0)

    def test_noisier_sky_gives_brighter_limiting_magnitude(self):
        quiet = _synthetic_field([(50, 50, 1.0)], noise=0.001, seed=1)
        noisy = _synthetic_field([(50, 50, 1.0)], noise=0.02, seed=1)
        params = MagnitudeParams(aperture_radius=6, annulus_inner=9, annulus_outer=13)
        lim_quiet = _estimate_limiting_magnitude(quiet, params, None)
        lim_noisy = _estimate_limiting_magnitude(noisy, params, None)
        # More sky noise -> brighter (numerically smaller) detection limit.
        assert lim_noisy < lim_quiet


class TestColorImageSupport:
    def test_accepts_chw_color_image(self):
        mono = _synthetic_field([(50, 50, 1.0), (120, 120, 0.3)])
        color = np.stack([mono, mono, mono], axis=0)
        result = measure_magnitudes(color, params=MagnitudeParams(aperture_radius=6))
        assert result.n_stars >= 2
