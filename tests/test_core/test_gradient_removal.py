"""Tests for post-stretch gradient / flat (vignette) correction."""

import numpy as np

from astraios.core.gradient_removal import GradientRemovalParams, remove_gradient


def _gradient_field(h=300, w=400, axis="x"):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = (xx / w) if axis == "x" else (yy / h)
    return (0.08 + 0.25 * g).astype(np.float32)  # 0.08 → 0.33 across the frame


def _corner_spread(img):
    a = img if img.ndim == 2 else img.mean(0)
    c = [a[:40, :40].mean(), a[:40, -40:].mean(), a[-40:, :40].mean(), a[-40:, -40:].mean()]
    return float(max(c) - min(c))


def test_subtract_flattens_gradient():
    rng = np.random.default_rng(0)
    img = _gradient_field() + rng.normal(0, 0.005, (300, 400)).astype(np.float32)
    img = np.clip(img, 0, 1).astype(np.float32)
    before = _corner_spread(img)
    out = remove_gradient(img)
    after = _corner_spread(out)
    assert out.shape == img.shape
    assert after < before * 0.5   # gradient substantially reduced


def test_divide_brightens_dark_corners():
    # Multiplicative vignette: corners darker.
    h = w = 300
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = (yy - h / 2) ** 2 + (xx - w / 2) ** 2
    vig = 1.0 - 0.5 * (r2 / r2.max())
    img = (0.3 * vig).astype(np.float32)
    out = remove_gradient(img, params=GradientRemovalParams(mode="divide"))
    assert _corner_spread(out) < _corner_spread(img)


def test_object_is_protected():
    # A bright central object on a gradient must NOT be flattened away.
    img = _gradient_field(300, 400)
    yy, xx = np.mgrid[0:300, 0:400].astype(np.float32)
    r2 = (yy - 150) ** 2 + (xx - 200) ** 2
    obj = np.exp(-r2 / (2 * 30 ** 2)).astype(np.float32) * 0.4
    img = np.clip(img + obj, 0, 1).astype(np.float32)
    mask = (obj > 0.05).astype(np.float32)
    out = remove_gradient(img, object_mask=mask)
    # The object centre stays bright relative to its surroundings.
    assert out[150, 200] > out[150, 60]


def test_object_fills_frame_bails():
    img = np.full((100, 100), 0.3, np.float32)
    mask = np.ones((100, 100), np.float32)
    out = remove_gradient(img, object_mask=mask)
    np.testing.assert_array_equal(out, img)


def test_color_image():
    rng = np.random.default_rng(1)
    base = _gradient_field(200, 260)
    img = np.stack([base, base * 0.9, base * 1.1]).astype(np.float32)
    img = np.clip(img + rng.normal(0, 0.004, img.shape).astype(np.float32), 0, 1)
    out = remove_gradient(img)
    assert out.shape == img.shape
    assert _corner_spread(out) < _corner_spread(img)
