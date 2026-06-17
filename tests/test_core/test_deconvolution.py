"""Tests for deconvolution."""

import numpy as np

from astraios.core.deconvolution import (
    DeconvolutionParams,
    _create_gaussian_psf,
    richardson_lucy,
)
from astraios.core.masks import Mask


class TestGaussianPSF:
    """Tests for PSF creation."""

    def test_psf_normalized(self):
        psf = _create_gaussian_psf(3.0)
        np.testing.assert_allclose(psf.sum(), 1.0, atol=1e-6)

    def test_psf_symmetric(self):
        psf = _create_gaussian_psf(5.0)
        np.testing.assert_array_almost_equal(psf, psf.T)

    def test_psf_peak_at_center(self):
        psf = _create_gaussian_psf(3.0)
        center = psf.shape[0] // 2
        assert psf[center, center] == psf.max()

    def test_psf_size_scales_with_fwhm(self):
        small = _create_gaussian_psf(2.0)
        large = _create_gaussian_psf(8.0)
        assert large.shape[0] > small.shape[0]


class TestRichardsonLucy:
    """Tests for RL deconvolution."""

    def test_basic_deconvolution(self):
        """Deconvolution should produce a result with same shape."""
        image = np.random.rand(64, 64).astype(np.float32) * 0.5 + 0.2
        params = DeconvolutionParams(psf_fwhm=2.0, iterations=5)
        result = richardson_lucy(image, params)
        assert result.shape == image.shape
        assert result.dtype == np.float32

    def test_output_in_range(self):
        """Output values should be clipped to [0, 1]."""
        image = np.random.rand(64, 64).astype(np.float32)
        params = DeconvolutionParams(iterations=10)
        result = richardson_lucy(image, params)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_color_image(self):
        """Should handle multi-channel images."""
        image = np.random.rand(3, 48, 48).astype(np.float32) * 0.5 + 0.1
        params = DeconvolutionParams(iterations=3)
        result = richardson_lucy(image, params)
        assert result.shape == (3, 48, 48)

    def test_few_iterations_gentle(self):
        """Few iterations should not drastically change the image."""
        image = np.random.rand(64, 64).astype(np.float32) * 0.3 + 0.3
        params = DeconvolutionParams(iterations=2, regularization=0.01)
        result = richardson_lucy(image, params)
        # Result should be correlated with input
        diff = np.abs(result - image).mean()
        assert diff < 0.3

    def test_mask_support(self):
        """Mask should protect unmasked regions."""
        image = np.ones((64, 64), dtype=np.float32) * 0.5
        params = DeconvolutionParams(iterations=5)

        mask_data = np.zeros((64, 64), dtype=np.float32)
        mask_data[32:, :] = 1.0
        mask = Mask(data=mask_data)

        result = richardson_lucy(image, params, mask=mask)
        # Top half should be unchanged
        np.testing.assert_allclose(result[:32, :].mean(), 0.5, atol=0.01)
