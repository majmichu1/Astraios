"""Registration resampling: one kernel set, same result on CPU and GPU, clamped.

Background in ``astraios/core/interpolation.py``: our GPU path used unclamped
bicubic and our CPU path unclamped Lanczos-4, so stacks differed by hardware,
and neither clamped ringing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from astraios.core.device_manager import get_device_manager
from astraios.core.interpolation import (
    INTERPOLATIONS,
    invert_affine,
    warp_affine_cpu,
    warp_affine_gpu,
)

H, W = 240, 320
INNER = (slice(12, H - 12), slice(12, W - 12))


def _field(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W]
    img = 0.2 + 0.05 * np.sin(xx / 13.0) * np.cos(yy / 17.0) + rng.normal(0, 0.01, (H, W))
    img = img.astype(np.float32)
    for cx, cy in ((100, 80), (220, 150), (60, 200)):
        img += 0.6 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.4**2)).astype(np.float32)
    return img


def _transform() -> np.ndarray:
    th = math.radians(0.8)
    s = 1.002
    c, sn = s * math.cos(th), s * math.sin(th)
    return np.array([[c, -sn, 2.37], [sn, c, -1.61]], np.float32)


@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
@pytest.mark.parametrize("clamp", [False, True])
def test_gpu_matches_cpu(interpolation, clamp):
    img = _field()
    m = _transform()
    dev = get_device_manager().device
    cpu = warp_affine_cpu(img, m, (H, W), interpolation, clamp)
    t = torch.from_numpy(img).to(dev)
    gpu = warp_affine_gpu(t, invert_affine(m), None, interpolation, clamp).cpu().numpy()
    diff = np.abs(cpu - gpu)[INNER]
    # OpenCV evaluates its kernels with 15-bit fixed-point weights; torch in float32.
    assert diff.max() < 8e-3 and diff.mean() < 5e-4, (interpolation, clamp, diff.max(), diff.mean())


@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
def test_color_and_mono_agree(interpolation):
    img = _field()
    m = _transform()
    color = np.stack([img, img * 0.5, img * 0.25])
    mono = warp_affine_cpu(img, m, (H, W), interpolation, True)
    rgb = warp_affine_cpu(color, m, (H, W), interpolation, True)
    assert rgb.shape == (3, H, W)
    np.testing.assert_allclose(rgb[0], mono, atol=1e-6)
    np.testing.assert_allclose(rgb[1], mono * 0.5, atol=1e-5)
    dev = get_device_manager().device
    t = torch.from_numpy(color).to(dev)
    g = warp_affine_gpu(t, invert_affine(m), None, interpolation, True).cpu().numpy()
    assert g.shape == (3, H, W)
    np.testing.assert_allclose(g[2], g[0] * 0.25, atol=1e-5)


@pytest.mark.parametrize("interpolation", ["bicubic", "lanczos4"])
def test_clamp_removes_ringing_on_both_devices(interpolation):
    """A hot pixel on flat sky: the unclamped kernels ring below the sky level
    and overshoot; clamped output stays inside the source range."""
    img = np.full((H, W), 0.2, np.float32)
    img[100, 150] = 1.0
    m = np.array([[1, 0, 0.5], [0, 1, 0.5]], np.float32)
    dev = get_device_manager().device
    t = torch.from_numpy(img).to(dev)
    for warped in (
        warp_affine_cpu(img, m, (H, W), interpolation, clamp=False),
        warp_affine_gpu(t, invert_affine(m), None, interpolation, clamp=False).cpu().numpy(),
    ):
        assert warped[INNER].min() < 0.2 - 1e-4, "unclamped kernel should ring below the sky"
    for warped in (
        warp_affine_cpu(img, m, (H, W), interpolation, clamp=True),
        warp_affine_gpu(t, invert_affine(m), None, interpolation, clamp=True).cpu().numpy(),
    ):
        assert warped[INNER].min() >= 0.2 - 1e-6
        assert warped.max() <= 1.0 + 1e-6


def test_clamp_lowers_noise_and_keeps_bilinear_softest():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, (H, W)).astype(np.float32)
    factors = {}
    for key in INTERPOLATIONS:
        vals = []
        for clamp in (False, True):
            r = []
            for dx, dy in ((0.5, 0.5), (0.25, 0.75), (0.1, 0.9)):
                m = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
                r.append(float(warp_affine_cpu(noise, m, (H, W), key, clamp)[INNER].std()))
            vals.append(float(np.mean(r)))
        factors[key] = vals
    assert factors["bilinear"][0] < factors["bicubic"][0] < factors["lanczos4"][0] < 1.0
    assert factors["lanczos4"][1] < factors["lanczos4"][0]
    assert factors["bicubic"][1] <= factors["bicubic"][0]


def test_zero_border_outside_footprint():
    img = np.full((H, W), 0.3, np.float32)
    m = np.array([[1, 0, 20.0], [0, 1, 10.0]], np.float32)  # left/top become padding
    for key in INTERPOLATIONS:
        covered = (slice(16, H - 4), slice(26, W - 4))  # inside the shifted footprint
        out = warp_affine_cpu(img, m, (H, W), key, True)
        assert out[:, :15].max() == 0.0 and out[:5].max() == 0.0
        assert abs(out[covered].mean() - 0.3) < 1e-3
        dev = get_device_manager().device
        t = torch.from_numpy(img).to(dev)
        g = warp_affine_gpu(t, invert_affine(m), None, key, True).cpu().numpy()
        assert g[:, :15].max() == 0.0 and abs(g[covered].mean() - 0.3) < 1e-3


def test_bad_kernel_name_is_rejected():
    with pytest.raises(ValueError):
        warp_affine_cpu(_field(), _transform(), (H, W), "nearest", True)
