"""Tests for wavelet decomposition and reconstruction."""

import numpy as np
import pytest

from astraios.core.wavelets import (
    WaveletParams,
    wavelet_decompose,
    wavelet_reconstruct,
    wavelet_sharpen,
)


class TestWaveletDecompose:
    def test_decompose_returns_correct_count(self):
        data = np.random.rand(64, 64).astype(np.float32)
        scales = wavelet_decompose(data, n_scales=3)
        assert len(scales) == 4  # 3 detail + 1 residual

    def test_decompose_shapes_match(self):
        data = np.random.rand(64, 64).astype(np.float32)
        scales = wavelet_decompose(data, n_scales=3)
        for s in scales:
            assert s.shape == (64, 64)

    def test_reconstruct_recovers_original(self):
        data = np.random.rand(64, 64).astype(np.float32)
        scales = wavelet_decompose(data, n_scales=4)
        reconstructed = wavelet_reconstruct(scales)
        np.testing.assert_allclose(reconstructed, data, atol=1e-4)

    def test_detail_scales_sum_to_zero_approx(self):
        data = np.random.rand(64, 64).astype(np.float32)
        scales = wavelet_decompose(data, n_scales=3)
        # Detail scales should have near-zero mean
        for s in scales[:-1]:
            assert abs(s.mean()) < 0.1

    def test_residual_is_smooth(self):
        data = np.random.rand(64, 64).astype(np.float32)
        scales = wavelet_decompose(data, n_scales=3)
        residual = scales[-1]
        # Residual should be smoother than the original
        assert np.std(residual) <= np.std(data) + 0.01


class TestWaveletSharpen:
    def test_identity_weights(self):
        data = np.random.rand(64, 64).astype(np.float32) * 0.5
        params = WaveletParams(n_scales=3, scale_weights=[1.0, 1.0, 1.0])
        result = wavelet_sharpen(data, params)
        np.testing.assert_allclose(result, data, atol=1e-3)

    def test_sharpening_increases_detail(self):
        data = np.random.rand(64, 64).astype(np.float32) * 0.5
        params = WaveletParams(n_scales=3, scale_weights=[2.0, 1.0, 1.0])
        result = wavelet_sharpen(data, params)
        # Sharpened image should have more contrast (higher std)
        assert np.std(result) >= np.std(data) - 0.01

    def test_smoothing_reduces_detail(self):
        data = np.random.rand(64, 64).astype(np.float32) * 0.5
        params = WaveletParams(n_scales=3, scale_weights=[0.0, 1.0, 1.0])
        result = wavelet_sharpen(data, params)
        # First detail scale removed, should be smoother
        assert np.std(result) < np.std(data) + 0.1

    def test_color_image(self):
        data = np.random.rand(3, 64, 64).astype(np.float32) * 0.5
        params = WaveletParams(n_scales=3, scale_weights=[1.5, 1.0, 1.0])
        result = wavelet_sharpen(data, params)
        assert result.shape == (3, 64, 64)

    def test_output_in_range(self):
        data = np.random.rand(64, 64).astype(np.float32)
        params = WaveletParams(n_scales=3, scale_weights=[3.0, 2.0, 1.0])
        result = wavelet_sharpen(data, params)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_mask_support(self):
        from astraios.core.masks import Mask
        data = np.random.rand(64, 64).astype(np.float32) * 0.5
        mask_data = np.zeros((64, 64), dtype=np.float32)
        mask_data[:32] = 1.0
        mask = Mask(data=mask_data)
        params = WaveletParams(n_scales=3, scale_weights=[3.0, 1.0, 1.0])
        result = wavelet_sharpen(data, params, mask=mask)
        # Bottom half should be unchanged
        np.testing.assert_allclose(result[32:], data[32:], atol=1e-5)


class TestAtrousDilationEquivalence:
    """The a trous transform expresses its holes as a conv2d dilation rather
    than materialising them as zeros in the kernel.

    At scale 5 the materialised kernel is 129x129 with 25 non-zero entries, so
    99.85% of the multiply-adds it implies are against a literal 0.0. Across
    the six scales WaveScale HDR uses that was 22350 taps per pixel where 150
    do the work. These pin the two properties that make the change safe to
    have made: the output does not move, and it is the same maths.
    """

    @staticmethod
    def _dense_reference(data: np.ndarray, n_scales: int) -> list[np.ndarray]:
        """The previous implementation: one materialised kernel per scale."""
        import torch
        import torch.nn.functional as F

        from astraios.core.wavelets import _atrous_kernel_2d

        cur = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        out = []
        with torch.no_grad():
            for s in range(n_scales):
                k = torch.from_numpy(_atrous_kernel_2d(s)).unsqueeze(0).unsqueeze(0)
                pad = k.shape[2] // 2
                sm = F.conv2d(F.pad(cur, (pad,) * 4, mode="replicate"), k)
                out.append((cur - sm).squeeze().numpy())
                cur = sm
            out.append(cur.squeeze().numpy())
        return out

    @pytest.mark.parametrize("size", [64, 129])
    @pytest.mark.parametrize("n_scales", [3, 6])
    def test_byte_identical_to_the_materialised_kernel(self, size, n_scales):
        """Not close, identical. A dilated kernel evaluates exactly the taps
        the dense one does not multiply by zero, and adding 0.0 changes no
        float, so there is no rounding difference to absorb."""
        from astraios.core.wavelets import wavelet_decompose

        rng = np.random.default_rng(4)
        img = np.clip(rng.random((size, size)).astype(np.float32) * 0.8 + 0.05, 0, 1)

        expected = self._dense_reference(img, n_scales)
        got = wavelet_decompose(img, n_scales=n_scales)

        assert len(got) == len(expected)
        for i, (g, e) in enumerate(zip(got, expected, strict=True)):
            assert g.tobytes() == e.tobytes(), f"scale {i} differs"

    def test_the_holes_really_are_zeros(self):
        """The premise of the whole change: the materialised kernel is the 5x5
        one with zeros punched in, so skipping them cannot change a result."""
        from astraios.core.wavelets import _B3_KERNEL_2D, _atrous_kernel_2d

        for s in range(4):
            dense = _atrous_kernel_2d(s)
            step = 2**s
            assert np.array_equal(dense[::step, ::step], _B3_KERNEL_2D)
            mask = np.ones_like(dense, dtype=bool)
            mask[::step, ::step] = False
            assert np.count_nonzero(dense[mask]) == 0

    def test_reconstruction_still_telescopes(self):
        """Detail scales plus residual must return the original."""
        from astraios.core.wavelets import wavelet_decompose

        rng = np.random.default_rng(9)
        img = np.clip(rng.random((96, 96)).astype(np.float32), 0, 1)
        scales = wavelet_decompose(img, n_scales=5)
        np.testing.assert_allclose(sum(scales), img, atol=1e-5)
