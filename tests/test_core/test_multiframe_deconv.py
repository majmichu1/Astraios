"""Tests for multi-frame deconvolution."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask
from astraios.core.multiframe_deconv import (
    MultiFrameDeconvParams,
    _auto_ksize_from_fwhm,
    _EarlyStopper,
    _normalize_psf,
    estimate_frame_psf,
    multiframe_deconvolve,
)


def _gaussian_psf(fwhm: float, k: int) -> np.ndarray:
    sigma = fwhm / 2.3548
    r = (k - 1) / 2
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return (g / g.sum()).astype(np.float32)


def _star_scene(h: int = 64, w: int = 64) -> np.ndarray:
    """A small synthetic scene: a few sharp point sources on a dim background."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    scene = np.full((h, w), 0.05, dtype=np.float32)
    for cy, cx, amp in [(20, 20, 1.0), (42, 46, 0.8), (30, 12, 0.6), (50, 30, 0.5)]:
        scene += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 0.8**2))
    return np.clip(scene, 0.0, 1.0).astype(np.float32)


def _blurred_frames(
    scene: np.ndarray, fwhms: list[float], noise_sigma: float, ksize: int = 15, seed: int = 0
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    frames, psfs = [], []
    for fwhm in fwhms:
        psf = _gaussian_psf(fwhm, ksize)
        blurred = fftconvolve(scene, psf, mode="same").astype(np.float32)
        noisy = blurred + rng.normal(0.0, noise_sigma, size=blurred.shape).astype(np.float32)
        frames.append(np.clip(noisy, 0.0, 1.0).astype(np.float32))
        psfs.append(psf)
    return frames, psfs


def _gradient_energy(img: np.ndarray) -> float:
    gy, gx = np.gradient(img)
    return float(np.mean(gx**2 + gy**2))


class TestPSFHelpers:
    def test_normalize_psf_sums_to_one(self):
        psf = np.random.default_rng(0).random((11, 11)).astype(np.float32)
        out = _normalize_psf(psf)
        np.testing.assert_allclose(out.sum(), 1.0, atol=1e-5)

    def test_normalize_psf_degenerate_falls_back_to_delta(self):
        psf = np.zeros((9, 9), dtype=np.float32)
        out = _normalize_psf(psf)
        np.testing.assert_allclose(out.sum(), 1.0, atol=1e-5)
        assert out[4, 4] == out.max()

    def test_auto_ksize_grows_with_fwhm(self):
        assert _auto_ksize_from_fwhm(2.0) < _auto_ksize_from_fwhm(8.0)

    def test_auto_ksize_always_odd(self):
        for fwhm in (1.0, 2.3, 5.7, 10.0):
            assert _auto_ksize_from_fwhm(fwhm) % 2 == 1

    def test_estimate_frame_psf_on_star_field(self):
        scene = _star_scene()
        params = MultiFrameDeconvParams(psf_ksize=15)
        psf = estimate_frame_psf(scene, params)
        assert psf.shape == (15, 15)
        np.testing.assert_allclose(psf.sum(), 1.0, atol=1e-4)
        # Peak should be roughly centered for a well-behaved empirical PSF.
        cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
        assert abs(cy - 7) <= 2 and abs(cx - 7) <= 2

    def test_estimate_frame_psf_falls_back_without_stars(self):
        blank = np.full((48, 48), 0.1, dtype=np.float32)
        params = MultiFrameDeconvParams(psf_ksize=11)
        psf = estimate_frame_psf(blank, params)
        assert psf.shape == (11, 11)
        np.testing.assert_allclose(psf.sum(), 1.0, atol=1e-4)


class TestEarlyStopper:
    def test_triggers_when_updates_plateau(self):
        stopper = _EarlyStopper(
            tol_upd_floor=1e-3, tol_rel_floor=1e-3, early_frac=0.9, patience=2, min_iters=2
        )
        stopped = False
        for it in range(1, 10):
            stopped = stopper.step(it, um=1e-6, rc=1e-6)
            if stopped:
                break
        assert stopped
        assert it < 9

    def test_does_not_trigger_before_min_iters(self):
        stopper = _EarlyStopper(
            tol_upd_floor=1.0, tol_rel_floor=1.0, early_frac=0.9, patience=1, min_iters=5
        )
        for it in range(1, 5):
            assert stopper.step(it, um=1e-9, rc=1e-9) is False


class TestMultiFrameDeconvolve:
    def test_recovers_sharper_than_average(self):
        """The joint solve should out-sharpen a plain average of the blurred frames."""
        scene = _star_scene()
        frames, psfs = _blurred_frames(scene, [2.0, 2.3, 2.6], noise_sigma=0.01, ksize=15)

        params = MultiFrameDeconvParams(
            iterations=10, min_iterations=2, psf_ksize=15, early_stop=False
        )
        result = multiframe_deconvolve(frames, params, psfs=psfs)

        average = np.mean(np.stack(frames, axis=0), axis=0)
        assert _gradient_energy(result) > _gradient_energy(average)

    def test_early_stop_terminates_before_max_iterations(self):
        """Identical, noise-free frames converge almost immediately."""
        rng = np.random.default_rng(5)
        base = (rng.random((48, 48)).astype(np.float32) * 0.4) + 0.3
        frames = [base.copy() for _ in range(4)]

        calls: list[str] = []
        params = MultiFrameDeconvParams(
            iterations=40,
            min_iterations=2,
            psf_ksize=9,
            early_stop=True,
            early_stop_frac=0.9,
            early_stop_patience=2,
        )
        multiframe_deconvolve(frames, params, progress=lambda f, m: calls.append(m))
        used = sum(1 for m in calls if "iteration" in m)
        assert used < params.iterations

    def test_early_stop_disabled_runs_all_iterations(self):
        rng = np.random.default_rng(5)
        base = (rng.random((48, 48)).astype(np.float32) * 0.4) + 0.3
        frames = [base.copy() for _ in range(4)]

        calls: list[str] = []
        params = MultiFrameDeconvParams(
            iterations=12, min_iterations=2, psf_ksize=9, early_stop=False
        )
        multiframe_deconvolve(frames, params, progress=lambda f, m: calls.append(m))
        used = sum(1 for m in calls if "iteration" in m)
        assert used == params.iterations

    def test_single_frame_degrades_gracefully(self):
        """A single-frame call must not crash and should stay well-behaved."""
        rng = np.random.default_rng(6)
        frame = (rng.random((48, 48)).astype(np.float32) * 0.5) + 0.2
        params = MultiFrameDeconvParams(iterations=5, psf_ksize=9)
        result = multiframe_deconvolve([frame], params)

        assert result.shape == frame.shape
        assert result.dtype == np.float32
        assert np.isfinite(result).all()
        assert result.min() >= 0.0 and result.max() <= 1.0

    def test_mono_frames(self):
        rng = np.random.default_rng(7)
        frames = [(rng.random((40, 40)).astype(np.float32) * 0.4) + 0.3 for _ in range(3)]
        params = MultiFrameDeconvParams(iterations=3, psf_ksize=9)
        result = multiframe_deconvolve(frames, params)
        assert result.shape == (40, 40)

    def test_color_perchannel(self):
        rng = np.random.default_rng(8)
        frames = [(rng.random((3, 40, 40)).astype(np.float32) * 0.4) + 0.3 for _ in range(3)]
        params = MultiFrameDeconvParams(iterations=3, psf_ksize=9, color_mode="perchannel")
        result = multiframe_deconvolve(frames, params)
        assert result.shape == (3, 40, 40)

    def test_color_luma_collapses_to_mono(self):
        rng = np.random.default_rng(8)
        frames = [(rng.random((3, 40, 40)).astype(np.float32) * 0.4) + 0.3 for _ in range(3)]
        params = MultiFrameDeconvParams(iterations=3, psf_ksize=9, color_mode="luma")
        result = multiframe_deconvolve(frames, params)
        assert result.shape == (40, 40)

    def test_stacked_ndarray_input(self):
        rng = np.random.default_rng(9)
        stack = np.stack([rng.random((36, 36)).astype(np.float32) for _ in range(4)], axis=0)
        params = MultiFrameDeconvParams(iterations=2, psf_ksize=9)
        result = multiframe_deconvolve(stack, params)
        assert result.shape == (36, 36)

    def test_mismatched_frame_shapes_raise(self):
        a = np.zeros((32, 32), dtype=np.float32)
        b = np.zeros((30, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="registered"):
            multiframe_deconvolve([a, b], MultiFrameDeconvParams())

    def test_invalid_color_mode_raises(self):
        frames = [np.zeros((16, 16), dtype=np.float32)]
        with pytest.raises(ValueError):
            multiframe_deconvolve(frames, MultiFrameDeconvParams(color_mode="bogus"))

    def test_frame_masks_change_output_where_rejected(self):
        """Rejecting a corrupted patch in one frame should visibly change that
        region relative to ignoring the corruption."""
        rng = np.random.default_rng(10)
        good = [(rng.random((40, 40)).astype(np.float32) * 0.3) + 0.3 for _ in range(3)]
        corrupted = good[0].copy()
        corrupted[10:20, 10:20] = 1.0
        frames = [corrupted, *good[1:]]

        mask0 = np.ones((40, 40), dtype=np.float32)
        mask0[10:20, 10:20] = 0.0
        ones = np.ones((40, 40), dtype=np.float32)
        frame_masks = [mask0, ones, ones]

        params_reject = MultiFrameDeconvParams(
            iterations=6, psf_ksize=9, frame_masks=frame_masks, rejection_strength=1.0
        )
        params_ignore = MultiFrameDeconvParams(
            iterations=6, psf_ksize=9, frame_masks=frame_masks, rejection_strength=0.0
        )
        out_reject = multiframe_deconvolve(frames, params_reject)
        out_ignore = multiframe_deconvolve(frames, params_ignore)

        patch_diff = np.abs(
            out_reject[10:20, 10:20] - out_ignore[10:20, 10:20]
        ).mean()
        assert patch_diff > 0.01

    def test_rejection_strength_blend_runs(self):
        rng = np.random.default_rng(11)
        frames = [(rng.random((32, 32)).astype(np.float32) * 0.4) + 0.3 for _ in range(3)]
        frame_masks = [np.ones((32, 32), dtype=np.float32) for _ in range(3)]
        params = MultiFrameDeconvParams(
            iterations=3, psf_ksize=9, frame_masks=frame_masks, rejection_strength=0.5
        )
        result = multiframe_deconvolve(frames, params)
        assert np.isfinite(result).all()

    def test_seed_modes_all_produce_valid_output(self):
        rng = np.random.default_rng(12)
        frames = [(rng.random((32, 32)).astype(np.float32) * 0.4) + 0.3 for _ in range(3)]
        for mode in ("median", "mean", "robust"):
            params = MultiFrameDeconvParams(iterations=2, psf_ksize=9, seed_mode=mode)
            result = multiframe_deconvolve(frames, params)
            assert np.isfinite(result).all()

    def test_integrated_seed_requires_seed_image(self):
        frames = [np.zeros((16, 16), dtype=np.float32)]
        params = MultiFrameDeconvParams(seed_mode="integrated")
        with pytest.raises(ValueError):
            multiframe_deconvolve(frames, params)

    def test_super_resolution_upsamples_output(self):
        rng = np.random.default_rng(13)
        frames = [(rng.random((32, 32)).astype(np.float32) * 0.4) + 0.3 for _ in range(3)]
        params = MultiFrameDeconvParams(iterations=2, psf_ksize=7, super_resolution=2)
        result = multiframe_deconvolve(frames, params)
        assert result.shape == (64, 64)
        assert np.isfinite(result).all()

    def test_variance_maps_do_not_crash(self):
        rng = np.random.default_rng(14)
        frames = [(rng.random((32, 32)).astype(np.float32) * 0.4) + 0.3 for _ in range(3)]
        params = MultiFrameDeconvParams(iterations=2, psf_ksize=9, use_variance_maps=True)
        result = multiframe_deconvolve(frames, params)
        assert np.isfinite(result).all()

    def test_l2_rho_and_positive_huber_delta(self):
        rng = np.random.default_rng(15)
        frames = [(rng.random((32, 32)).astype(np.float32) * 0.4) + 0.3 for _ in range(3)]
        out_l2 = multiframe_deconvolve(
            frames, MultiFrameDeconvParams(iterations=2, psf_ksize=9, rho="l2")
        )
        out_hd = multiframe_deconvolve(
            frames, MultiFrameDeconvParams(iterations=2, psf_ksize=9, huber_delta=0.05)
        )
        assert np.isfinite(out_l2).all()
        assert np.isfinite(out_hd).all()

    def test_final_mask_protects_region(self):
        """Final protect mask should leave the protected half close to the seed."""
        rng = np.random.default_rng(16)
        frames = [(rng.random((32, 32)).astype(np.float32) * 0.4) + 0.3 for _ in range(3)]
        mask_data = np.zeros((32, 32), dtype=np.float32)
        mask_data[16:, :] = 1.0  # only bottom half is processed
        mask = Mask(data=mask_data)

        seed = np.mean(np.stack(frames, axis=0), axis=0)
        params = MultiFrameDeconvParams(iterations=6, psf_ksize=9)
        result = multiframe_deconvolve(frames, params, mask=mask)

        np.testing.assert_allclose(result[:16, :], seed[:16, :], atol=1e-5)

    def test_psf_count_mismatch_raises(self):
        frames = [np.zeros((16, 16), dtype=np.float32) for _ in range(3)]
        psfs = [_gaussian_psf(2.0, 7)]
        with pytest.raises(ValueError):
            multiframe_deconvolve(frames, MultiFrameDeconvParams(), psfs=psfs)


class TestGpuCpuAgreement:
    def test_gpu_matches_cpu(self):
        dm = get_device_manager()
        if not dm.is_gpu:
            pytest.skip("no GPU available")

        rng = np.random.default_rng(17)
        frames = [(rng.random((40, 40)).astype(np.float32) * 0.5) + 0.2 for _ in range(3)]

        out_gpu = multiframe_deconvolve(
            frames, MultiFrameDeconvParams(iterations=6, psf_ksize=9, force_cpu=False)
        )
        out_cpu = multiframe_deconvolve(
            frames, MultiFrameDeconvParams(iterations=6, psf_ksize=9, force_cpu=True)
        )
        assert np.abs(out_gpu - out_cpu).mean() < 1e-3
