"""Tests for image statistics computation."""

import numpy as np
import pytest

from astraios.core.statistics import compute_image_statistics, ImageStatistics


class TestComputeImageStatisticsMono:
    """Tests for statistics on mono (H, W) images."""

    def test_basic_mono_statistics(self):
        """Should compute correct basic statistics for a uniform image."""
        image = np.full((64, 64), 0.5, dtype=np.float32)
        stats = compute_image_statistics(image)

        assert isinstance(stats, ImageStatistics)
        assert stats.n_channels == 1
        assert stats.width == 64
        assert stats.height == 64
        assert stats.total_pixels == 64 * 64
        assert len(stats.channels) == 1

        ch = stats.channels[0]
        assert ch.mean == pytest.approx(0.5, abs=1e-5)
        assert ch.median == pytest.approx(0.5, abs=1e-5)
        assert ch.std == pytest.approx(0.0, abs=1e-5)
        assert ch.min_val == pytest.approx(0.5, abs=1e-5)
        assert ch.max_val == pytest.approx(0.5, abs=1e-5)
        assert ch.pixel_count == 64 * 64

    def test_mono_channel_name(self):
        """Mono image channel should be named 'Mono'."""
        image = np.full((16, 16), 0.3, dtype=np.float32)
        stats = compute_image_statistics(image)
        assert stats.channels[0].name == "Mono"

    def test_mono_mad(self):
        """MAD should be zero for a constant image."""
        image = np.full((32, 32), 0.4, dtype=np.float32)
        stats = compute_image_statistics(image)
        assert stats.channels[0].mad == pytest.approx(0.0, abs=1e-6)

    def test_mono_mad_nonzero(self):
        """MAD should be nonzero for an image with spread values."""
        rng = np.random.RandomState(42)
        image = rng.rand(64, 64).astype(np.float32)
        stats = compute_image_statistics(image)
        assert stats.channels[0].mad > 0.0

    def test_mono_min_max(self):
        """Min and max should reflect actual pixel extremes."""
        image = np.full((32, 32), 0.5, dtype=np.float32)
        image[0, 0] = 0.1
        image[31, 31] = 0.9
        stats = compute_image_statistics(image)
        assert stats.channels[0].min_val == pytest.approx(0.1, abs=1e-5)
        assert stats.channels[0].max_val == pytest.approx(0.9, abs=1e-5)


class TestComputeImageStatisticsColor:
    """Tests for statistics on color (C, H, W) images."""

    def test_rgb_channel_names(self):
        """3-channel image should have R, G, B channel names."""
        image = np.full((3, 32, 32), 0.5, dtype=np.float32)
        stats = compute_image_statistics(image)
        assert stats.n_channels == 3
        assert len(stats.channels) == 3
        names = [ch.name for ch in stats.channels]
        assert names == ["R", "G", "B"]

    def test_lrgb_channel_names(self):
        """4-channel image should have L, R, G, B channel names."""
        image = np.full((4, 16, 16), 0.5, dtype=np.float32)
        stats = compute_image_statistics(image)
        assert stats.n_channels == 4
        names = [ch.name for ch in stats.channels]
        assert names == ["L", "R", "G", "B"]

    def test_five_channels_fallback_names(self):
        """5-channel image should get fallback Ch0..Ch4 names."""
        image = np.full((5, 8, 8), 0.5, dtype=np.float32)
        stats = compute_image_statistics(image)
        assert stats.n_channels == 5
        names = [ch.name for ch in stats.channels]
        assert names == ["Ch0", "Ch1", "Ch2", "Ch3", "Ch4"]

    def test_color_per_channel_independence(self):
        """Each channel should have independently computed statistics."""
        image = np.zeros((3, 32, 32), dtype=np.float32)
        image[0] = 0.2  # R
        image[1] = 0.5  # G
        image[2] = 0.8  # B
        stats = compute_image_statistics(image)

        assert stats.channels[0].mean == pytest.approx(0.2, abs=1e-5)
        assert stats.channels[1].mean == pytest.approx(0.5, abs=1e-5)
        assert stats.channels[2].mean == pytest.approx(0.8, abs=1e-5)

    def test_color_total_pixels(self):
        """Total pixels should be H * W * C."""
        image = np.full((3, 20, 30), 0.5, dtype=np.float32)
        stats = compute_image_statistics(image)
        assert stats.total_pixels == 3 * 20 * 30

    def test_color_dimensions(self):
        """Width and height should reflect the spatial dimensions."""
        image = np.full((3, 40, 80), 0.5, dtype=np.float32)
        stats = compute_image_statistics(image)
        assert stats.width == 80
        assert stats.height == 40


class TestSNREstimation:
    """Tests for signal-to-noise ratio estimation."""

    def test_snr_zero_for_constant_image(self):
        """Constant image has MAD=0 => noise_est ~0 => SNR should be 0.0."""
        image = np.full((32, 32), 0.5, dtype=np.float32)
        stats = compute_image_statistics(image)
        assert stats.channels[0].snr_estimate == pytest.approx(0.0, abs=1e-6)

    def test_snr_positive_for_noisy_image(self):
        """An image with a signal above the noise floor should have positive SNR."""
        rng = np.random.RandomState(100)
        # Signal level 0.5 with small noise
        image = (0.5 + rng.randn(64, 64).astype(np.float32) * 0.01).clip(0, 1)
        stats = compute_image_statistics(image)
        assert stats.channels[0].snr_estimate > 0.0

    def test_snr_increases_with_signal(self):
        """Higher signal relative to noise should yield higher SNR."""
        rng = np.random.RandomState(200)
        noise = rng.randn(64, 64).astype(np.float32) * 0.02

        image_low = (0.1 + noise).clip(0, 1).astype(np.float32)
        image_high = (0.6 + noise).clip(0, 1).astype(np.float32)

        snr_low = compute_image_statistics(image_low).channels[0].snr_estimate
        snr_high = compute_image_statistics(image_high).channels[0].snr_estimate

        assert snr_high > snr_low

    def test_snr_formula_manual(self):
        """SNR should match median / (MAD * 1.4826) for a known distribution."""
        rng = np.random.RandomState(300)
        image = rng.rand(128, 128).astype(np.float32)
        stats = compute_image_statistics(image)
        ch = stats.channels[0]

        # Manually compute expected SNR
        flat = image.ravel()
        median_val = float(np.median(flat))
        mad_val = float(np.median(np.abs(flat - median_val)))
        noise_est = mad_val * 1.4826
        expected_snr = median_val / noise_est if noise_est > 1e-10 else 0.0

        assert ch.snr_estimate == pytest.approx(expected_snr, rel=1e-4)


class TestLinearityDetection:
    """Tests for the is_linear heuristic."""

    def test_linear_image_detected(self):
        """Image with median < 0.1 should be flagged as linear."""
        # Typical linear astro data: mostly dark background
        image = np.full((64, 64), 0.03, dtype=np.float32)
        stats = compute_image_statistics(image)
        assert stats.is_linear is True

    def test_stretched_image_not_linear(self):
        """Image with median >= 0.1 should not be flagged as linear."""
        image = np.full((64, 64), 0.5, dtype=np.float32)
        stats = compute_image_statistics(image)
        assert stats.is_linear is False

    def test_linearity_uses_first_channel(self):
        """Linearity should be determined from the first channel only."""
        image = np.zeros((3, 32, 32), dtype=np.float32)
        image[0] = 0.05  # R channel: linear
        image[1] = 0.5   # G channel: stretched
        image[2] = 0.5   # B channel: stretched
        stats = compute_image_statistics(image)
        # First channel median = 0.05 < 0.1 => linear
        assert stats.is_linear is True

    def test_linearity_boundary(self):
        """Median exactly at 0.1 should not be linear (< 0.1 required)."""
        image = np.full((32, 32), 0.1, dtype=np.float32)
        stats = compute_image_statistics(image)
        assert stats.is_linear is False


class TestPercentiles:
    """Tests for percentile computation."""

    def test_percentiles_constant_image(self):
        """Constant image should have p1 == p99 == that constant value."""
        image = np.full((32, 32), 0.4, dtype=np.float32)
        stats = compute_image_statistics(image)
        ch = stats.channels[0]
        assert ch.percentile_01 == pytest.approx(0.4, abs=1e-5)
        assert ch.percentile_99 == pytest.approx(0.4, abs=1e-5)

    def test_percentiles_ordering(self):
        """p1 should be less than or equal to p99."""
        rng = np.random.RandomState(500)
        image = rng.rand(64, 64).astype(np.float32)
        stats = compute_image_statistics(image)
        ch = stats.channels[0]
        assert ch.percentile_01 <= ch.percentile_99

    def test_percentiles_within_minmax(self):
        """Percentiles should be between min and max values."""
        rng = np.random.RandomState(600)
        image = rng.rand(64, 64).astype(np.float32)
        stats = compute_image_statistics(image)
        ch = stats.channels[0]
        assert ch.percentile_01 >= ch.min_val
        assert ch.percentile_99 <= ch.max_val


class TestClippingDetection:
    """Tests for shadow/highlight clipping detection."""

    def test_no_clipping_on_midtone_image(self):
        """An image with no 0.0 or 1.0 pixels should report 0% clipping."""
        image = np.full((32, 32), 0.5, dtype=np.float32)
        stats = compute_image_statistics(image)
        ch = stats.channels[0]
        assert ch.clipped_low_pct == pytest.approx(0.0, abs=1e-6)
        assert ch.clipped_high_pct == pytest.approx(0.0, abs=1e-6)

    def test_shadow_clipping_detected(self):
        """Pixels at exactly 0.0 should be counted as shadow-clipped."""
        image = np.full((10, 10), 0.5, dtype=np.float32)
        image[0, :] = 0.0  # 10 out of 100 pixels
        stats = compute_image_statistics(image)
        ch = stats.channels[0]
        assert ch.clipped_low_pct == pytest.approx(10.0, abs=0.1)

    def test_highlight_clipping_detected(self):
        """Pixels at exactly 1.0 should be counted as highlight-clipped."""
        image = np.full((10, 10), 0.5, dtype=np.float32)
        image[0:2, :] = 1.0  # 20 out of 100 pixels
        stats = compute_image_statistics(image)
        ch = stats.channels[0]
        assert ch.clipped_high_pct == pytest.approx(20.0, abs=0.1)

    def test_both_clipping(self):
        """Should correctly report both shadow and highlight clipping."""
        image = np.full((10, 10), 0.5, dtype=np.float32)
        image[0, :] = 0.0   # 10 shadow-clipped
        image[9, :] = 1.0   # 10 highlight-clipped
        stats = compute_image_statistics(image)
        ch = stats.channels[0]
        assert ch.clipped_low_pct == pytest.approx(10.0, abs=0.1)
        assert ch.clipped_high_pct == pytest.approx(10.0, abs=0.1)


class TestEdgeCases:
    """Edge case tests for compute_image_statistics."""

    def test_invalid_ndim_raises(self):
        """Should raise ValueError for 4D input."""
        image = np.zeros((2, 3, 4, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="Unexpected image shape"):
            compute_image_statistics(image)

    def test_small_image(self):
        """Should work on a very small 2x2 image."""
        image = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        stats = compute_image_statistics(image)
        assert stats.width == 2
        assert stats.height == 2
        assert stats.channels[0].pixel_count == 4
        assert stats.channels[0].mean == pytest.approx(0.25, abs=1e-5)

    def test_all_zeros(self):
        """An all-zero image should have mean=0, clipped_low=100%."""
        image = np.zeros((16, 16), dtype=np.float32)
        stats = compute_image_statistics(image)
        ch = stats.channels[0]
        assert ch.mean == pytest.approx(0.0, abs=1e-6)
        assert ch.min_val == pytest.approx(0.0, abs=1e-6)
        assert ch.max_val == pytest.approx(0.0, abs=1e-6)
        assert ch.clipped_low_pct == pytest.approx(100.0, abs=0.1)
        assert stats.is_linear is True

    def test_all_ones(self):
        """An all-one image should have mean=1, clipped_high=100%."""
        image = np.ones((16, 16), dtype=np.float32)
        stats = compute_image_statistics(image)
        ch = stats.channels[0]
        assert ch.mean == pytest.approx(1.0, abs=1e-6)
        assert ch.clipped_high_pct == pytest.approx(100.0, abs=0.1)
        assert stats.is_linear is False
