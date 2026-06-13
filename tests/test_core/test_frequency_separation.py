"""Tests for frequency separation."""

import numpy as np

from cosmica.core.frequency_separation import (
    FrequencySeparationParams,
    SeparationMethod,
    frequency_separation,
    recombine,
    separate,
)
from cosmica.core.masks import Mask


def _image(h=96, w=128, color=False):
    rng = np.random.default_rng(0)
    if color:
        return np.clip(rng.random((3, h, w)) * 0.5 + 0.25, 0, 1).astype(np.float32)
    return np.clip(rng.random((h, w)) * 0.5 + 0.25, 0, 1).astype(np.float32)


class TestSeparateRecombine:
    def test_subtract_roundtrip_is_identity(self):
        img = _image()
        low, high = separate(img, sigma=4.0, method=SeparationMethod.SUBTRACT)
        out = recombine(low, high, SeparationMethod.SUBTRACT)
        # LF + HF == original (within clip range); image is well inside [0,1]
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_divide_roundtrip_is_identity(self):
        img = _image()
        low, high = separate(img, sigma=4.0, method=SeparationMethod.DIVIDE)
        out = recombine(low, high, SeparationMethod.DIVIDE)
        np.testing.assert_allclose(out, img, atol=1e-4)

    def test_color_layout_roundtrip(self):
        img = _image(color=True)
        low, high = separate(img, sigma=3.0)
        out = recombine(low, high)
        assert low.shape == high.shape == img.shape
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_hf_carries_detail_lf_is_smooth(self):
        img = _image()
        low, high = separate(img, sigma=4.0, method=SeparationMethod.SUBTRACT)
        # The LF layer must be smoother (lower local variance) than the original;
        # the HF layer holds the removed detail (non-trivial energy).
        assert float(np.var(low)) < float(np.var(img))
        assert float(np.std(high)) > 1e-3


class TestFrequencySeparationProcess:
    def test_default_params_is_noop_roundtrip(self):
        img = _image()
        out = frequency_separation(img, FrequencySeparationParams())
        np.testing.assert_allclose(out, img, atol=1e-5)

    def test_hf_boost_increases_detail_energy(self):
        img = _image()
        base = frequency_separation(img, FrequencySeparationParams(hf_boost=1.0))
        boosted = frequency_separation(img, FrequencySeparationParams(hf_boost=2.0))
        # Boosting HF should raise high-frequency energy (gradient magnitude).
        def hf_energy(a):
            gx = np.diff(a, axis=-1)
            gy = np.diff(a, axis=-2)
            return float(np.mean(gx**2)) + float(np.mean(gy**2))
        assert hf_energy(boosted) > hf_energy(base)

    def test_divide_boost_runs_and_stays_in_range(self):
        img = _image(color=True)
        out = frequency_separation(
            img, FrequencySeparationParams(method=SeparationMethod.DIVIDE, hf_boost=1.5)
        )
        assert out.shape == img.shape
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_mask_protects_region(self):
        img = _image()
        mask_data = np.zeros_like(img)
        mask_data[48:, :] = 1.0  # only bottom half processed
        out = frequency_separation(
            img, FrequencySeparationParams(hf_boost=3.0), mask=Mask(data=mask_data)
        )
        np.testing.assert_allclose(out[:48, :], img[:48, :], atol=1e-6)
