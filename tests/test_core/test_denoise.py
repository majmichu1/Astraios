"""Tests for noise reduction."""

import numpy as np

from astraios.core.denoise import (
    DenoiseMethod,
    DenoiseParams,
    denoise,
    measure_noise,
    recommend_strength,
)
from astraios.core.masks import Mask


def _noisy_image(h=100, w=100, noise_level=0.1):
    """Create an image with additive Gaussian noise."""
    clean = np.ones((h, w), dtype=np.float32) * 0.3
    rng = np.random.default_rng(42)
    noise = rng.normal(0, noise_level, (h, w)).astype(np.float32)
    return np.clip(clean + noise, 0, 1)


class TestDenoise:
    def test_nlm_reduces_noise(self):
        noisy = _noisy_image(noise_level=0.1)
        params = DenoiseParams(method=DenoiseMethod.NLM, strength=0.5)
        result = denoise(noisy, params)
        assert result.shape == noisy.shape
        # Standard deviation should decrease
        assert result.std() < noisy.std()

    def test_wavelet_reduces_noise(self):
        noisy = _noisy_image(noise_level=0.1)
        params = DenoiseParams(method=DenoiseMethod.WAVELET, strength=0.5)
        result = denoise(noisy, params)
        assert result.shape == noisy.shape
        assert result.std() < noisy.std()
        # Brightness must be preserved: a regression that dropped the wavelet
        # residual collapsed the image toward black (mean 0.40 -> 0.045) yet
        # still passed the std check, so assert the mean explicitly.
        assert abs(float(result.mean()) - float(noisy.mean())) < 0.05

    def test_output_in_range(self):
        noisy = _noisy_image()
        result = denoise(noisy)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_color_image(self):
        rng = np.random.default_rng(42)
        noisy = np.clip(0.3 + rng.normal(0, 0.05, (3, 64, 64)), 0, 1).astype(np.float32)
        result = denoise(noisy, DenoiseParams(method=DenoiseMethod.WAVELET))
        assert result.shape == (3, 64, 64)

    def test_chrominance_only(self):
        rng = np.random.default_rng(42)
        noisy = np.clip(0.3 + rng.normal(0, 0.05, (3, 64, 64)), 0, 1).astype(np.float32)
        params = DenoiseParams(chrominance_only=True)
        result = denoise(noisy, params)
        assert result.shape == (3, 64, 64)

    def test_mask_support(self):
        noisy = _noisy_image()
        mask_data = np.zeros((100, 100), dtype=np.float32)
        mask_data[50:, :] = 1.0
        mask = Mask(data=mask_data)
        result = denoise(noisy, mask=mask)
        # Top half should be unchanged
        np.testing.assert_array_almost_equal(result[:50, :], noisy[:50, :])


class TestNoiseMeasurement:
    def test_noisier_image_reports_higher_sigma(self):
        rng = np.random.default_rng(0)
        clean = np.full((128, 128), 0.3, np.float32) + rng.normal(0, 0.002, (128, 128)).astype(np.float32)
        noisy = np.full((128, 128), 0.3, np.float32) + rng.normal(0, 0.03, (128, 128)).astype(np.float32)
        assert measure_noise(noisy)[0] > measure_noise(clean)[0]

    def test_handles_channel_first_color(self):
        # Astraios stores colour channel-first (C, H, W); must not crash or misread axes.
        rng = np.random.default_rng(1)
        img = np.clip(0.3 + rng.normal(0, 0.02, (3, 96, 96)), 0, 1).astype(np.float32)
        sigma, snr = measure_noise(img)
        assert 0.0 < sigma < 0.1
        assert snr > 0

    def test_recommend_strength_in_range_and_monotonic(self):
        rng = np.random.default_rng(2)
        clean = np.full((128, 128), 0.3, np.float32) + rng.normal(0, 0.002, (128, 128)).astype(np.float32)
        noisy = np.full((128, 128), 0.3, np.float32) + rng.normal(0, 0.04, (128, 128)).astype(np.float32)
        s_clean = recommend_strength(clean)[0]
        s_noisy = recommend_strength(noisy)[0]
        assert 0.15 <= s_clean <= 0.9
        assert 0.15 <= s_noisy <= 0.9
        assert s_noisy > s_clean
