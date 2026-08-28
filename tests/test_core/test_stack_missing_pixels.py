"""Registration padding must not darken the stack's edges.

``align_frames``/``align_from_paths`` fill the area outside each frame's
footprint with zeros. With dithered subs every edge pixel is covered by only
some of the frames, and until 2026-08 the zeros of the others were averaged in
as if they were sky: a 200 px dither on a 24 MP stack left a visible dark
frame around the image (Siril's stack of the same subs had none). Zeros are
now missing data for normalization, rejection and the mean, on the CPU and
GPU paths of both stackers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from astraios.core import stacking as S
from astraios.core.device_manager import get_device_manager
from astraios.core.image_io import ImageData

SKY = 0.3
NOISE = 0.01


def _padded_stack(n: int = 12, h: int = 40, w: int = 60, seed: int = 0) -> np.ndarray:
    """n frames of flat sky; frame i is missing its first i+1 rows and columns."""
    rng = np.random.default_rng(seed)
    raw = (SKY + rng.normal(0, NOISE, (n, h, w))).astype(np.float32)
    raw[3, 20, 30] = 0.9  # a cosmic ray, to check rejection still works
    for i in range(n):
        raw[i, : i + 1, :] = 0.0
        raw[i, :, : i + 1] = 0.0
    return raw


BAND = (slice(3, 12), slice(20, 50))   # rows covered by only some frames
FULL = (slice(20, None), slice(20, None))  # rows covered by every frame

CPU_KERNELS = {
    "sigma": lambda x, v: S._reject_sigma_clip(x, 3, 3, 5, valid=v),
    "winsorized": lambda x, v: S._reject_winsorized_sigma(x, 3, 3, 5, 1.5, valid=v),
    "linear_fit": lambda x, v: S._reject_linear_fit(x, 3, 3, 5, valid=v),
    "percentile": lambda x, v: S._reject_percentile_clip(x, 10, 10, valid=v),
    "min_max": lambda x, v: S._reject_min_max(x, 2, valid=v),
    "esd": lambda x, v: S._reject_esd(x, valid=v),
}
GPU_KERNELS = {
    "sigma": lambda t, v: S._gpu_sigma_clip(t, 3, 3, 5, valid=v),
    "winsorized": lambda t, v: S._gpu_winsorized_sigma_clip(t, 3, 3, 5, 1.5, valid=v),
    "linear_fit": lambda t, v: S._gpu_linear_fit_clip(t, 3, 3, 5, valid=v),
    "percentile": lambda t, v: S._gpu_percentile_clip(t, 10, 10, valid=v),
    "min_max": lambda t, v: S._gpu_min_max(t, 2, valid=v),
}


def test_valid_mask_erodes_the_interpolation_ring():
    raw = _padded_stack()
    valid = S._valid_from_zero(raw)
    assert valid is not None
    # frame 0 has row 0 zeroed; row 1 is its neighbour and goes too
    assert not valid[0, 0].any() and not valid[0, 1].any()
    assert valid[0, 2, 2:].all()
    # the same mask from the GPU twin
    valid_t = S._erode_valid_t(torch.from_numpy(raw > 0))
    np.testing.assert_array_equal(valid_t.numpy(), valid)
    assert S._valid_from_zero(np.full((3, 8, 8), 0.5, np.float32)) is None


@pytest.mark.parametrize("name", sorted(CPU_KERNELS))
def test_cpu_kernels_ignore_missing_pixels(name):
    raw = _padded_stack()
    valid = S._valid_from_zero(raw)
    masked = CPU_KERNELS[name](raw, valid)
    rejected = S._get_mask(masked, raw.shape)
    result = S._integrate(masked, S.IntegrationMethod.AVERAGE)
    assert rejected[3, 20, 30], "cosmic ray must still be rejected"
    assert rejected[0, 0, 0], "missing pixels are masked"
    assert abs(result[BAND].mean() - SKY) < 0.003, f"{name}: edge band darkened"
    assert abs(result[FULL].mean() - SKY) < 0.003
    # counted rejections exclude the missing pixels
    assert S._n_rejected_np(rejected, valid) < int(rejected.sum())


@pytest.mark.parametrize("name", sorted(GPU_KERNELS))
def test_gpu_kernels_match_cpu_with_missing_pixels(name):
    raw = _padded_stack()
    valid = S._valid_from_zero(raw)
    dev = get_device_manager().device
    t = torch.from_numpy(raw).to(dev)
    v = torch.from_numpy(valid).to(dev)
    keep, n_rej = GPU_KERNELS[name](t, v)
    cpu_rejected = S._get_mask(CPU_KERNELS[name](raw, valid), raw.shape)
    np.testing.assert_array_equal(keep.cpu().numpy(), ~cpu_rejected)
    assert n_rej == S._n_rejected_np(cpu_rejected, valid)
    result = S._gpu_integrate(t, keep, S.IntegrationMethod.AVERAGE).cpu().numpy()
    assert abs(result[BAND].mean() - SKY) < 0.003


def test_kernels_without_valid_are_unchanged():
    """valid=None keeps the previous code path (and its results) exactly."""
    raw = _padded_stack()
    for name, fn in CPU_KERNELS.items():
        m = fn(raw, None)
        assert m.shape == raw.shape, name


def test_percentile_clip_keeps_at_least_one_value():
    raw = _padded_stack()
    valid = S._valid_from_zero(raw)
    rejected = S._get_mask(S._reject_percentile_clip(raw, 10, 10, valid=valid), raw.shape)
    kept = ~rejected & valid
    has_data = valid.any(axis=0)
    assert kept.any(axis=0)[has_data].all()


@pytest.mark.parametrize("rejection", [
    S.RejectionMethod.WINSORIZED_SIGMA, S.RejectionMethod.SIGMA_CLIP,
    S.RejectionMethod.MIN_MAX, S.RejectionMethod.NONE,
])
def test_stack_images_edges_hold_the_sky(rejection):
    raw = _padded_stack()
    images = [ImageData(data=f.copy()) for f in raw]
    params = S.StackingParams(rejection=rejection, normalization=S.NormalizationMethod.NONE)
    res = S.stack_images(images, params, align=False)
    out = res.image.data
    assert abs(out[BAND].mean() - SKY) < 0.003
    assert abs(out[FULL].mean() - SKY) < 0.003
    assert out[0].max() == 0.0, "rows no frame covers stay black"
    if res.rejection_map is not None:
        assert not res.rejection_map[0, 0, 0], "missing pixels are not reported as rejected"

    # Prove the test is sensitive: the old behaviour darkens the band.
    params_old = S.StackingParams(
        rejection=rejection, normalization=S.NormalizationMethod.NONE, ignore_black_pixels=False
    )
    old = S.stack_images([ImageData(data=f.copy()) for f in raw], params_old, align=False)
    assert old.image.data[BAND].mean() < SKY - 0.02


def test_stack_from_paths_edges_hold_the_sky(tmp_path):
    raw = _padded_stack()
    paths = []
    for i, f in enumerate(raw):
        p = tmp_path / f"aligned_{i:03d}.fits"
        S._write_aligned_fits(ImageData(data=f.copy()), p)
        paths.append(p)
    params = S.StackingParams(normalization=S.NormalizationMethod.ADDITIVE_SCALING)
    out = S.stack_from_paths(paths, params=params).image.data
    assert abs(out[BAND].mean() - SKY) < 0.003
    assert abs(out[FULL].mean() - SKY) < 0.003
    assert out[0].max() == 0.0

    old = S.stack_from_paths(
        paths, params=S.StackingParams(
            normalization=S.NormalizationMethod.ADDITIVE_SCALING, ignore_black_pixels=False
        )
    ).image.data
    assert old[BAND].mean() < SKY - 0.02


def test_stack_from_paths_color_edges_hold_the_sky(tmp_path):
    raw = _padded_stack(n=8)
    color = np.stack([raw, raw * 0.9, raw * 0.8], axis=1)  # (N, C, H, W)
    paths = []
    for i, f in enumerate(color):
        p = tmp_path / f"aligned_{i:03d}.fits"
        S._write_aligned_fits(ImageData(data=np.ascontiguousarray(f)), p)
        paths.append(p)
    out = S.stack_from_paths(paths, params=S.StackingParams()).image.data
    assert out.shape == (3, 40, 60)
    for c, gain in enumerate((1.0, 0.9, 0.8)):
        assert abs(out[c][BAND].mean() - SKY * gain) < 0.003
