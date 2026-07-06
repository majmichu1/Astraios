"""Tests for the exoplanet transit detector (ported from SASpro)."""

from __future__ import annotations

import numpy as np
import pytest

from astraios.core.exoplanet_transit import (
    ComparisonCombine,
    DetrendMethod,
    ExoplanetTransitParams,
    TimeSource,
    TransitStar,
    analyze_transit,
)

IMG_SHAPE = (150, 150)
TARGET_XY = (30.0, 30.0)
COMPARISON_XYS = [
    (120.0, 30.0),
    (30.0, 120.0),
    (120.0, 120.0),
    (75.0, 75.0),
    (75.0, 30.0),
]
STAR_FWHM = 3.0
BACKGROUND = 0.05
BASE_AMPLITUDE = 0.5

DEFAULT_PARAMS = ExoplanetTransitParams(
    aperture_radius=6.0,
    annulus_inner=9.0,
    annulus_outer=14.0,
    # No systematic drift in these synthetic sequences, so skip detrending for the
    # quantitative depth/mid-time checks — the quadratic detrend fits some of the
    # box dip itself (expected, ported behavior; exercised separately below).
    detrend_method=DetrendMethod.NONE,
)


def _render_frame(
    star_specs: list[tuple[float, float, float]],
    rng: np.random.Generator,
    noise_sigma: float = 0.0015,
) -> np.ndarray:
    """Render a synthetic mono frame with Gaussian stars + background + noise."""
    h, w = IMG_SHAPE
    sigma = STAR_FWHM / 2.3548
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.full((h, w), BACKGROUND, dtype=np.float64)
    for x, y, amplitude in star_specs:
        img += amplitude * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)))
    img += rng.normal(0.0, noise_sigma, size=img.shape)
    np.clip(img, 0.0, 1.0, out=img)
    return img.astype(np.float32)


def _trapezoid_depths(n_frames: int, center: float, half_flat: float, ramp: float) -> np.ndarray:
    """Fractional dip depth per frame: 1.0 at full transit depth, 0.0 outside."""
    idx = np.arange(n_frames, dtype=np.float64)
    d = np.abs(idx - center)
    depths = np.zeros(n_frames)
    flat = d <= half_flat
    depths[flat] = 1.0
    slope = (d > half_flat) & (d <= half_flat + ramp)
    depths[slope] = 1.0 - (d[slope] - half_flat) / ramp
    return depths


def _make_sequence(
    n_frames: int,
    depth: float,
    center: float,
    seed: int = 42,
    half_flat: float = 2.0,
    ramp: float = 3.0,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Build a frame sequence: target dips per a trapezoid profile, comparisons constant."""
    rng = np.random.default_rng(seed)
    depths = _trapezoid_depths(n_frames, center, half_flat, ramp) * depth

    frames = []
    for i in range(n_frames):
        target_amp = BASE_AMPLITUDE * (1.0 - depths[i])
        specs = [(TARGET_XY[0], TARGET_XY[1], target_amp)]
        specs += [(x, y, BASE_AMPLITUDE) for x, y in COMPARISON_XYS]
        frames.append(_render_frame(specs, rng))
    return frames, depths


class TestInjectedTransit:
    """A box/trapezoid dip on the target, constant comparisons, should be recovered."""

    def test_recovers_dip_and_flags_detection(self):
        n_frames = 30
        depth = 0.05  # 5% = 50 ppt, well above the 20 ppt default threshold
        center = 15.0
        frames, depths = _make_sequence(n_frames, depth, center)

        result = analyze_transit(
            frames, TARGET_XY, comparison_xys=COMPARISON_XYS, params=DEFAULT_PARAMS
        )

        assert result.relative_flux.shape == (n_frames,)
        assert result.time_is_jd is False
        np.testing.assert_allclose(result.times, np.arange(n_frames))

        # light curve dips in the middle relative to the out-of-transit edges
        out_of_transit = np.r_[result.relative_flux[:5], result.relative_flux[-5:]]
        in_transit = result.relative_flux[int(center) - 1 : int(center) + 2]
        assert np.nanmean(in_transit) < np.nanmean(out_of_transit) - 0.02

        assert result.transit_detected is True
        assert result.transit_depth == pytest.approx(depth, abs=0.02)
        assert result.mid_transit_time is not None
        assert abs(result.mid_transit_time - center) <= 3

        assert result.target_star.role == "target"
        assert all(isinstance(s, TransitStar) for s in result.comparison_stars)
        assert len(result.comparison_stars) == len(COMPARISON_XYS)

    def test_progress_callback_invoked(self):
        n_frames = 12
        frames, _ = _make_sequence(n_frames, depth=0.05, center=6.0)
        calls = []
        analyze_transit(
            frames,
            TARGET_XY,
            comparison_xys=COMPARISON_XYS,
            params=DEFAULT_PARAMS,
            progress=lambda frac, msg: calls.append((frac, msg)),
        )
        # one initial "loading reference frame" call, then one per measured frame
        assert len(calls) == n_frames + 1
        assert calls[0][0] == pytest.approx(0.0)
        assert calls[-1][0] == pytest.approx(1.0)


class TestNoTransit:
    def test_constant_sequence_reports_no_detection(self):
        n_frames = 30
        frames, _ = _make_sequence(n_frames, depth=0.0, center=15.0, seed=7)

        result = analyze_transit(
            frames, TARGET_XY, comparison_xys=COMPARISON_XYS, params=DEFAULT_PARAMS
        )

        assert result.transit_detected is False
        # a few ppt of noise at most, nowhere near the 50 ppt injected-depth case
        assert result.transit_depth < 0.02


class TestComparisonAutoSelection:
    def test_auto_selected_comparisons_avoid_target_and_recover_transit(self):
        n_frames = 30
        depth = 0.06
        center = 15.0
        frames, _ = _make_sequence(n_frames, depth, center, seed=11)

        params = ExoplanetTransitParams(
            aperture_radius=6.0,
            annulus_inner=9.0,
            annulus_outer=14.0,
            n_comparison_stars=3,
            star_detection_sigma=4.0,
        )
        result = analyze_transit(frames, TARGET_XY, comparison_xys=None, params=params)

        assert 1 <= len(result.comparison_stars) <= 3
        for star in result.comparison_stars:
            dist = np.hypot(star.x - TARGET_XY[0], star.y - TARGET_XY[1])
            assert dist > params.annulus_outer  # excluded the target itself

        assert result.transit_detected is True
        assert result.transit_depth == pytest.approx(depth, abs=0.025)

    def test_raises_without_candidates_or_explicit_comparisons(self):
        n_frames = 5
        rng = np.random.default_rng(1)
        # No comparison stars rendered at all, and nothing near target either.
        frames = [_render_frame([(30.0, 30.0, 0.5)], rng) for _ in range(n_frames)]
        params = ExoplanetTransitParams(
            aperture_radius=6.0,
            annulus_inner=9.0,
            annulus_outer=14.0,
            star_detection_sigma=50.0,  # impossibly high threshold -> no detections
        )
        with pytest.raises(ValueError):
            analyze_transit(frames, TARGET_XY, comparison_xys=None, params=params)


class TestFramesFromPaths:
    def test_fits_paths_yield_increasing_jd_time_axis(self, tmp_path):
        from astropy.io import fits

        n_frames = 8
        depth = 0.05
        center = 4.0
        frames, _ = _make_sequence(n_frames, depth, center, seed=3)

        paths = []
        base = "2024-03-01T00:00:00.000"
        from astropy.time import Time

        t0 = Time(base, scale="utc")
        for i, frame in enumerate(frames):
            hdu = fits.PrimaryHDU(frame.astype(np.float32))
            obs_time = t0 + i * 5.0 * (1.0 / 1440.0)  # +5 minutes per frame, in days
            hdu.header["DATE-OBS"] = obs_time.isot
            path = tmp_path / f"frame_{i:02d}.fits"
            hdu.writeto(str(path), overwrite=True)
            paths.append(path)

        result = analyze_transit(
            paths, TARGET_XY, comparison_xys=COMPARISON_XYS, params=DEFAULT_PARAMS
        )

        assert result.time_is_jd is True
        assert result.times.shape == (n_frames,)
        assert np.all(np.diff(result.times) > 0)
        # ~5 minutes apart => ~5/1440 days
        np.testing.assert_allclose(np.diff(result.times), 5.0 / 1440.0, atol=1e-6)


class TestDetrend:
    """Detrend methods should run cleanly and still surface the injected dip,
    even though quadratic detrending against out-of-transit baseline necessarily
    fits away some of the dip depth (ported, expected SASpro behavior)."""

    @pytest.mark.parametrize("method", [DetrendMethod.LINEAR, DetrendMethod.QUADRATIC])
    def test_detrend_methods_still_detect(self, method):
        n_frames = 40
        depth = 0.06
        center = 20.0
        frames, _ = _make_sequence(n_frames, depth, center, seed=5, half_flat=2.0, ramp=3.0)

        params = ExoplanetTransitParams(
            aperture_radius=6.0,
            annulus_inner=9.0,
            annulus_outer=14.0,
            detrend_method=method,
        )
        result = analyze_transit(
            frames, TARGET_XY, comparison_xys=COMPARISON_XYS, params=params
        )

        assert result.transit_detected is True
        assert 0.0 < result.transit_depth <= depth + 0.02
        assert abs(result.mid_transit_time - center) <= 4


class TestParamsAndDataclasses:
    def test_default_params_roundtrip(self):
        p = ExoplanetTransitParams()
        assert p.detrend_method == DetrendMethod.QUADRATIC
        assert p.comparison_combine == ComparisonCombine.MEAN
        assert p.time_source == TimeSource.AUTO

    def test_analyze_transit_requires_minimum_frames(self):
        rng = np.random.default_rng(0)
        frames = [_render_frame([(30.0, 30.0, 0.5)], rng) for _ in range(2)]
        with pytest.raises(ValueError):
            analyze_transit(frames, TARGET_XY, comparison_xys=COMPARISON_XYS)
