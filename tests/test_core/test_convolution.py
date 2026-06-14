"""Tests for the convolution (blur) tool."""

import numpy as np

from cosmica.core.filters import ConvolutionKernel, ConvolutionParams, convolve
from cosmica.core.masks import Mask


def _img(color=False):
    rng = np.random.default_rng(0)
    if color:
        return rng.random((3, 48, 48)).astype(np.float32)
    return rng.random((48, 48)).astype(np.float32)


class TestConvolve:
    def test_gaussian_reduces_variance(self):
        img = _img()
        out = convolve(img, ConvolutionParams(kernel=ConvolutionKernel.GAUSSIAN, radius=3))
        assert out.shape == img.shape
        assert float(np.var(out)) < float(np.var(img))  # blur smooths

    def test_box_blur(self):
        img = _img()
        out = convolve(img, ConvolutionParams(kernel=ConvolutionKernel.BOX, radius=2))
        assert out.shape == img.shape
        assert float(np.var(out)) < float(np.var(img))

    def test_color_layout(self):
        img = _img(color=True)
        out = convolve(img, ConvolutionParams(radius=2))
        assert out.shape == img.shape
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_amount_blends(self):
        img = _img()
        full = convolve(img, ConvolutionParams(radius=4, amount=1.0))
        half = convolve(img, ConvolutionParams(radius=4, amount=0.5))
        # 0.5 amount sits between the original and the full blur.
        assert float(np.var(half)) > float(np.var(full))
        assert float(np.var(half)) < float(np.var(img))

    def test_zero_radius_is_noop(self):
        img = _img()
        out = convolve(img, ConvolutionParams(radius=0))
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_mask_protects_region(self):
        img = _img()
        mask_data = np.zeros((48, 48), np.float32)
        mask_data[24:, :] = 1.0
        out = convolve(img, ConvolutionParams(radius=4), mask=Mask(data=mask_data))
        np.testing.assert_allclose(out[:24, :], img[:24, :], atol=1e-6)
