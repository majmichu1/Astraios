"""Tests for auto-stretch."""

import numpy as np
import pytest

from cosmica.core.stretch import (
    StatisticalStretchParams,
    StretchParams,
    _solve_midtone,
    auto_stretch,
    compute_channel_stats,
    compute_histogram,
    midtone_transfer_function,
    statistical_stretch,
)


class TestStatisticalStretch:
    @staticmethod
    def _linear_image(median=0.02, shape=(64, 64)):
        rng = np.random.default_rng(0)
        # Skewed, mostly-dark image like linear astro data.
        img = np.abs(rng.normal(0, median, shape)).astype(np.float32)
        return np.clip(img, 0, 1)

    def test_solve_midtone_maps_median_to_target(self):
        for x in (0.01, 0.05, 0.2):
            m = _solve_midtone(x, 0.25)
            got = float(midtone_transfer_function(np.array([x], np.float32), m)[0])
            assert abs(got - 0.25) < 1e-3

    def test_result_median_hits_target_mono(self):
        img = self._linear_image()
        out = statistical_stretch(img, StatisticalStretchParams(target_median=0.25))
        assert out.shape == img.shape
        assert out.min() >= 0.0 and out.max() <= 1.0
        med = float(np.median(out[out > 0]))
        assert abs(med - 0.25) < 0.03

    def test_different_targets(self):
        img = self._linear_image()
        lo = statistical_stretch(img, StatisticalStretchParams(target_median=0.15))
        hi = statistical_stretch(img, StatisticalStretchParams(target_median=0.4))
        assert float(np.median(hi[hi > 0])) > float(np.median(lo[lo > 0]))

    def test_color_linked_preserves_relative_channels(self):
        rng = np.random.default_rng(1)
        base = np.abs(rng.normal(0, 0.02, (64, 64))).astype(np.float32)
        img = np.clip(np.stack([base * 1.4, base, base * 0.7]), 0, 1).astype(np.float32)
        out = statistical_stretch(img, StatisticalStretchParams(target_median=0.25, linked=True))
        assert out.shape == img.shape
        # R was brightest, B dimmest — ordering should survive a linked stretch.
        assert out[0].mean() > out[1].mean() > out[2].mean()


class TestMTF:
    def test_mtf_zero_stays_zero(self):
        x = np.array([0.0], dtype=np.float32)
        result = midtone_transfer_function(x, 0.25)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_mtf_one_stays_one(self):
        x = np.array([1.0], dtype=np.float32)
        result = midtone_transfer_function(x, 0.25)
        assert result[0] == pytest.approx(1.0, abs=1e-6)

    def test_mtf_monotonic(self):
        x = np.linspace(0, 1, 100, dtype=np.float32)
        result = midtone_transfer_function(x, 0.25)
        assert np.all(np.diff(result) >= -1e-6)  # non-decreasing

    def test_mtf_midpoint(self):
        # When x = midtone, result should be 0.5
        m = 0.25
        x = np.array([m], dtype=np.float32)
        result = midtone_transfer_function(x, m)
        assert result[0] == pytest.approx(0.5, abs=0.01)


class TestAutoStretch:
    def test_mono_stretch(self):
        data = np.random.random((100, 120)).astype(np.float32) * 0.01
        result = auto_stretch(data)
        assert result.shape == data.shape
        assert result.min() >= 0
        assert result.max() <= 1

    def test_color_stretch_linked(self):
        data = np.random.random((3, 100, 120)).astype(np.float32) * 0.01
        params = StretchParams(linked=True)
        result = auto_stretch(data, params)
        assert result.shape == data.shape

    def test_color_stretch_unlinked(self):
        data = np.random.random((3, 100, 120)).astype(np.float32) * 0.01
        params = StretchParams(linked=False)
        result = auto_stretch(data, params)
        assert result.shape == data.shape

    def test_stretch_brightens_dark_image(self):
        data = np.random.random((100, 120)).astype(np.float32) * 0.001
        result = auto_stretch(data)
        assert np.median(result) > np.median(data)


class TestChannelStats:
    def test_stats_keys(self):
        data = np.random.random((100,)).astype(np.float32)
        stats = compute_channel_stats(data)
        assert "median" in stats
        assert "mad" in stats
        assert "mean" in stats

    def test_stats_values_reasonable(self):
        data = np.random.normal(0.5, 0.1, 1000).astype(np.float32)
        data = np.clip(data, 0, 1)
        stats = compute_channel_stats(data)
        assert abs(stats["median"] - 0.5) < 0.05
        assert abs(stats["mean"] - 0.5) < 0.05


class TestGHS:
    def test_identity_when_D_zero(self):
        from cosmica.core.stretch import GHSParams, generalized_hyperbolic_stretch
        data = np.random.rand(100, 100).astype(np.float32)
        result = generalized_hyperbolic_stretch(data, GHSParams(D=0))
        np.testing.assert_array_almost_equal(result, data)

    def test_output_in_range(self):
        from cosmica.core.stretch import GHSParams, generalized_hyperbolic_stretch
        data = np.random.rand(100, 100).astype(np.float32)
        result = generalized_hyperbolic_stretch(data, GHSParams(D=5.0, b=1.0))
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_color_image(self):
        from cosmica.core.stretch import GHSParams, generalized_hyperbolic_stretch
        data = np.random.rand(3, 50, 50).astype(np.float32)
        result = generalized_hyperbolic_stretch(data, GHSParams(D=3.0))
        assert result.shape == (3, 50, 50)

    def test_stretches_image(self):
        from cosmica.core.stretch import GHSParams, generalized_hyperbolic_stretch
        data = np.ones((100, 100), dtype=np.float32) * 0.1  # dark image
        result = generalized_hyperbolic_stretch(data, GHSParams(D=5.0))
        # Stretched image should use more of the range
        assert result.max() - result.min() >= data.max() - data.min()


class TestHistogram:
    def test_mono_histogram(self):
        data = np.random.random((100, 120)).astype(np.float32)
        hist = compute_histogram(data, bins=64)
        assert "gray" in hist
        assert len(hist["gray"]) == 64

    def test_color_histogram(self):
        data = np.random.random((3, 100, 120)).astype(np.float32)
        hist = compute_histogram(data, bins=128)
        assert "red" in hist
        assert "green" in hist
        assert "blue" in hist
        assert "luminance" in hist
        assert len(hist["red"]) == 128
