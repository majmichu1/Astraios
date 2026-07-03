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


class TestSpatialDeconvolution:
    """richardson_lucy_spatial: zone grid, PSF dedup, blending."""

    def _starfield(self, h=240, w=320, seed=5):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:h, 0:w]
        img = np.full((h, w), 0.05, dtype=np.float32)
        for _ in range(60):
            cy, cx = rng.uniform(10, h - 10), rng.uniform(10, w - 10)
            s = rng.uniform(1.2, 2.0)
            amp = rng.uniform(0.2, 0.8)
            img += (amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s)))).astype(
                np.float32
            )
        img += rng.normal(0, 0.005, (h, w)).astype(np.float32)
        return np.clip(img, 0, 1)

    def test_basic_mono(self):
        from astraios.core.deconvolution import SpatialDeconvParams, richardson_lucy_spatial

        img = self._starfield()
        params = SpatialDeconvParams(grid_zones=2, iterations=5)
        result = richardson_lucy_spatial(img, params=params)
        assert result.shape == img.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))
        assert 0.0 <= result.min() and result.max() <= 1.0

    def test_color_shape(self):
        from astraios.core.deconvolution import SpatialDeconvParams, richardson_lucy_spatial

        mono = self._starfield(h=96, w=128)
        img = np.stack([mono, mono * 0.9, mono * 0.8])
        params = SpatialDeconvParams(grid_zones=2, iterations=3)
        result = richardson_lucy_spatial(img, params=params)
        assert result.shape == img.shape

    def test_psf_dedup_skips_duplicate_zone_runs(self, monkeypatch):
        """Zones sharing a measured FWHM must reuse one RL run (bit-identical)."""
        import astraios.core.deconvolution as dc

        img = self._starfield()
        calls = {"n": 0}
        orig = dc._rl_channel

        def counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        monkeypatch.setattr(dc, "_rl_channel", counting)
        params = dc.SpatialDeconvParams(grid_zones=3, iterations=3)
        dc.richardson_lucy_spatial(img, params=params)
        # 9 zones; a synthetic uniform starfield yields far fewer unique PSFs
        assert calls["n"] < 9, f"expected deduped RL runs, got {calls['n']}/9"

    def test_deterministic(self):
        from astraios.core.deconvolution import SpatialDeconvParams, richardson_lucy_spatial

        img = self._starfield(h=96, w=128)
        params = SpatialDeconvParams(grid_zones=2, iterations=4)
        a = richardson_lucy_spatial(img, params=params)
        b = richardson_lucy_spatial(img, params=params)
        assert np.array_equal(a, b)
