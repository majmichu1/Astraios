"""Tests for isophote/contour (galaxy) analysis."""

import numpy as np

from astraios.core.isophote_analysis import IsophoteParams, IsophoteResult, fit_isophotes


def _synthetic_galaxy(size, cx, cy, eps, pa_deg, scale=15.0, amp=1.0):
    """A smooth elliptical-exponential-disk test image with known
    ellipticity/position-angle isophotes (independent of the module's
    internal geometry helpers -- same standard ellipse-rotation math).
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    dx, dy = xx - cx, yy - cy
    pa = np.deg2rad(pa_deg)
    c, s = np.cos(pa), np.sin(pa)
    xr = dx * c + dy * s
    yr = -dx * s + dy * c
    b_over_a = 1.0 - eps
    rad = np.sqrt(xr**2 + (yr / b_over_a) ** 2)
    return (amp * np.exp(-rad / scale)).astype(np.float32)


def _wrapped_pa_diff(a_deg, b_deg):
    """Smallest angular difference between two position angles, accounting
    for their 180-degree (ellipse) symmetry.
    """
    d = (a_deg - b_deg + 90.0) % 180.0 - 90.0
    return abs(d)


class TestFitIsophotes:
    def test_recovers_injected_ellipticity_and_position_angle(self):
        size = 121
        cx, cy = 60.0, 60.0
        eps_true, pa_true = 0.4, 30.0

        img = _synthetic_galaxy(size, cx, cy, eps_true, pa_true, scale=15.0)
        rng = np.random.default_rng(0)
        img = np.clip(img + rng.normal(0.0, 0.002, size=img.shape).astype(np.float32), 0.0, 1.0)

        params = IsophoteParams(
            cx=cx, cy=cy, sma0=15.0, minsma=5.0, maxsma=40.0, step=0.3,
            eps0=0.1, pa0_deg=0.0, fix_center=True, sclip=3.0, nclip=2,
        )
        result = fit_isophotes(img, params)

        assert isinstance(result, IsophoteResult)
        assert result.backend == "numpy-fallback"
        assert result.n_rings >= 3

        # Use rings in a well-resolved radius band (avoid the very center
        # and the noisy/faint outskirts).
        band = (result.sma >= 8) & (result.sma <= 35)
        assert band.sum() >= 3

        eps_med = float(np.median(result.eps[band]))
        pa_med = float(np.median(result.pa_deg[band]))

        assert abs(eps_med - eps_true) < 0.12
        assert _wrapped_pa_diff(pa_med, pa_true) < 15.0

    def test_model_and_residual_shapes(self):
        size = 80
        img = _synthetic_galaxy(size, 40, 40, 0.3, -20.0, scale=12.0)
        params = IsophoteParams(
            cx=40, cy=40, sma0=10.0, minsma=3.0, maxsma=25.0, step=0.4, fix_center=True
        )
        result = fit_isophotes(img, params)

        assert result.model is not None
        assert result.residual is not None
        assert result.model.shape == img.shape
        assert result.residual.shape == img.shape
        assert np.isfinite(result.model).all()
        assert np.isfinite(result.residual).all()
        np.testing.assert_allclose(result.residual, img - result.model, atol=1e-6)

    def test_build_model_false_skips_rendering(self):
        img = _synthetic_galaxy(60, 30, 30, 0.2, 0.0, scale=8.0)
        params = IsophoteParams(cx=30, cy=30, sma0=8.0, maxsma=20.0, build_model=False)
        result = fit_isophotes(img, params)
        assert result.model is None
        assert result.residual is None
        assert result.n_rings >= 1

    def test_intensity_profile_decreases_outward(self):
        """An exponential-disk profile's fitted intensity should broadly
        decline with increasing semi-major axis."""
        img = _synthetic_galaxy(100, 50, 50, 0.25, 45.0, scale=14.0)
        params = IsophoteParams(
            cx=50, cy=50, sma0=10.0, minsma=4.0, maxsma=35.0, step=0.35, fix_center=True
        )
        result = fit_isophotes(img, params)
        assert result.n_rings >= 4
        # Not strictly monotonic due to sigma-clipped noise, but the first
        # ring should be much brighter than the last.
        assert result.intens[0] > result.intens[-1]

    def test_color_input_reduces_to_luminance(self):
        mono = _synthetic_galaxy(60, 30, 30, 0.3, 10.0, scale=10.0)
        color = np.stack([mono, mono, mono], axis=0)
        params = IsophoteParams(cx=30, cy=30, sma0=8.0, maxsma=20.0, build_model=False)
        result = fit_isophotes(color, params)
        assert result.n_rings >= 1
        assert np.isfinite(result.eps).all()

    def test_downsample_runs_and_upscales_geometry(self):
        img = _synthetic_galaxy(128, 64, 64, 0.35, 60.0, scale=18.0)
        params = IsophoteParams(
            cx=64, cy=64, sma0=16.0, minsma=6.0, maxsma=40.0, step=0.4,
            fix_center=True, downsample=2,
        )
        result = fit_isophotes(img, params)
        assert result.n_rings >= 3
        assert result.model.shape == img.shape
        # sma values should be reported back in full-resolution pixel units.
        assert result.sma.max() > 20.0

    def test_high_harmonics_and_wedge_do_not_crash(self):
        img = _synthetic_galaxy(90, 45, 45, 0.3, 15.0, scale=12.0)
        params = IsophoteParams(
            cx=45, cy=45, sma0=10.0, minsma=4.0, maxsma=25.0, step=0.4,
            fix_center=True, high_harmonics=True,
            use_wedge=True, wedge_pa_deg=0.0, wedge_width_deg=25.0,
        )
        result = fit_isophotes(img, params)
        assert result.n_rings >= 2
        assert np.isfinite(result.a3).all()
        assert np.isfinite(result.b4).all()

    def test_fixed_geometry_holds_eps_and_pa_constant(self):
        img = _synthetic_galaxy(70, 35, 35, 0.4, 0.0, scale=10.0)
        params = IsophoteParams(
            cx=35, cy=35, sma0=8.0, maxsma=20.0, eps0=0.4, pa0_deg=0.0,
            fix_center=True, fix_pa=True, fix_eps=True,
        )
        result = fit_isophotes(img, params)
        assert np.allclose(result.eps, 0.4)
        assert np.allclose(result.pa_deg, 0.0)
