"""Tests for chroma (colour) noise reduction."""

import numpy as np

from cosmica.core.chroma_denoise import chroma_denoise


def _noisy_color(h=120, w=160, seed=0):
    rng = np.random.default_rng(seed)
    # Smooth colour regions + heavy per-pixel CHROMA noise, light luminance noise.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 0.2 + 0.1 * (xx / w)
    img = np.stack([base, base, base]).astype(np.float32)
    img[0] += 0.05  # gentle real colour
    img[2] -= 0.05
    # chroma speckle: opposite-sign per channel so luminance stays ~constant
    n = rng.normal(0, 0.06, (h, w)).astype(np.float32)
    img[0] += n
    img[2] -= n
    return np.clip(img, 0, 1).astype(np.float32)


def _chroma_noise_level(img):
    lum = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    return float(np.std(img[0] - lum))


def test_reduces_chroma_noise():
    img = _noisy_color()
    before = _chroma_noise_level(img)
    out = chroma_denoise(img, strength=1.0)
    after = _chroma_noise_level(out)
    assert out.shape == img.shape
    assert after < before * 0.6  # colour noise substantially reduced


def test_preserves_luminance():
    img = _noisy_color()
    out = chroma_denoise(img, strength=1.5)
    lum_in = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    lum_out = 0.299 * out[0] + 0.587 * out[1] + 0.114 * out[2]
    # Luminance is essentially unchanged (chroma-only operation).
    assert float(np.mean(np.abs(lum_in - lum_out))) < 0.01


def test_mono_unchanged():
    mono = np.clip(np.random.default_rng(1).random((50, 50)).astype(np.float32), 0, 1)
    np.testing.assert_array_equal(chroma_denoise(mono), mono)


def test_zero_strength_noop():
    img = _noisy_color()
    np.testing.assert_array_equal(chroma_denoise(img, strength=0.0), img)


def test_preserves_real_color_regions():
    # A genuinely red vs blue region should stay distinguishable.
    img = _noisy_color()
    out = chroma_denoise(img, strength=1.0)
    # Channel 0 (red-boosted) still > channel 2 (blue-reduced) on average.
    assert float(np.mean(out[0])) > float(np.mean(out[2]))
