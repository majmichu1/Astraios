"""Tests for background-masked luminance-grain reduction."""

import numpy as np

from astraios.core.luma_denoise import (
    LumaDenoiseParams,
    _auto_bg_threshold,
    denoise_background_luma,
)


def _scene(color=False, seed=0):
    """Dark, grainy background with a bright, smooth subject blob in the centre."""
    rng = np.random.default_rng(seed)
    h, w = 96, 120
    bg = 0.15
    img = np.full((h, w), bg, dtype=np.float32)
    img += rng.normal(0, 0.03, (h, w)).astype(np.float32)  # luminance grain
    # bright smooth subject in the centre (well above the background)
    yy, xx = np.mgrid[0:h, 0:w]
    blob = 0.45 * np.exp(-(((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (2 * 12.0**2)))
    img = np.clip(img + blob.astype(np.float32), 0, 1)
    if not color:
        return img
    # colour: give channels distinct offsets so chroma is non-trivial
    r = np.clip(img * 1.0, 0, 1)
    g = np.clip(img * 0.8 + 0.02, 0, 1)
    b = np.clip(img * 1.2, 0, 1)
    return np.stack([r, g, b]).astype(np.float32)


def _bg_region(shape):
    """Boolean mask of a corner patch that is pure background (no subject)."""
    h, w = shape[-2:]
    m = np.zeros((h, w), dtype=bool)
    m[:20, :20] = True
    return m


def test_reduces_background_grain_mono():
    img = _scene(color=False)
    out = denoise_background_luma(img, LumaDenoiseParams(strength=0.8))
    reg = _bg_region(img.shape)
    assert out.shape == img.shape
    assert np.std(out[reg]) < np.std(img[reg])  # grain reduced in the background


def test_subject_core_preserved():
    img = _scene(color=False)
    out = denoise_background_luma(img, LumaDenoiseParams(strength=0.8))
    # The bright subject core is far above the background threshold -> mask ~0.
    h, w = img.shape
    cy, cx = h // 2, w // 2
    core = img[cy - 3:cy + 3, cx - 3:cx + 3]
    core_out = out[cy - 3:cy + 3, cx - 3:cx + 3]
    assert np.max(np.abs(core_out - core)) < 0.02


def test_zero_strength_is_noop():
    img = _scene(color=False)
    out = denoise_background_luma(img, LumaDenoiseParams(strength=0.0))
    np.testing.assert_array_equal(out, img)


def test_background_mean_preserved_no_darkening():
    img = _scene(color=False)
    out = denoise_background_luma(img, LumaDenoiseParams(strength=0.8))
    reg = _bg_region(img.shape)
    # The bilateral filter is mean-preserving, so the background level must not
    # drift (no darkening, the failure mode of an aggressive wavelet denoise).
    assert abs(float(out[reg].mean()) - float(img[reg].mean())) < 0.005


def test_color_preserves_chroma_in_background():
    img = _scene(color=True)
    out = denoise_background_luma(img, LumaDenoiseParams(strength=0.8))
    reg = _bg_region(img.shape)
    # Bilateral smoothing preserves each channel's mean, so the background's
    # mean colour ratio (R/G) is preserved — no colour shift.
    rg_before = float(img[0][reg].mean() / img[1][reg].mean())
    rg_after = float(out[0][reg].mean() / out[1][reg].mean())
    assert abs(rg_after - rg_before) < 0.01
    # and luminance grain still drops
    assert np.std(out[0][reg]) < np.std(img[0][reg])


def test_object_mask_protects_subject():
    img = _scene(color=False)
    h, w = img.shape
    # Protect the whole frame -> background mask becomes empty -> no change.
    full_protect = np.ones((h, w), dtype=np.float32)
    out = denoise_background_luma(
        img, LumaDenoiseParams(strength=0.8), object_mask=full_protect
    )
    np.testing.assert_array_equal(out, img)


def test_auto_threshold_in_range():
    img = _scene(color=False)
    thr = _auto_bg_threshold(img, 3.0)
    assert 0.02 <= thr <= 0.7
    # threshold should sit above the sky level but below the subject peak
    assert thr > 0.15
