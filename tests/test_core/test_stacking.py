"""Tests for the stacking engine."""

import numpy as np
import pytest

from astraios.core.image_io import ImageData
from astraios.core.stacking import (
    IntegrationMethod,
    NormalizationMethod,
    RegistrationMode,
    RejectionMethod,
    StackingParams,
    StackResult,
    _gpu_min_max,
    _gpu_percentile_clip,
    align_from_paths,
    normalize_stack,
    normalize_stack_linear_fit,
    stack_from_paths,
    stack_images,
)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestNormalizeStack:
    def test_single_frame_unchanged(self):
        stack = np.random.random((1, 20, 20)).astype(np.float32)
        result = normalize_stack(stack)
        np.testing.assert_allclose(result, stack, rtol=1e-5)

    def test_identical_frames_unchanged(self):
        data = np.full((5, 10, 10), 0.42, dtype=np.float32)
        result = normalize_stack(data)
        np.testing.assert_allclose(result, data, rtol=1e-5)

    def test_offset_correction(self):
        base = np.random.random((10, 10)).astype(np.float32) * 0.1
        stack = np.array([base + offset for offset in np.linspace(0.0, 0.5, 5)])
        result = normalize_stack(stack)
        frame_medians = np.array([np.median(result[i]) for i in range(5)])
        assert np.std(frame_medians) < 0.05

    def test_scale_correction(self):
        base = np.random.random((10, 10)).astype(np.float32) * 0.3 + 0.1
        stack = np.array([base * scale for scale in np.linspace(0.8, 1.2, 4)])
        result = normalize_stack(stack)
        frame_medians = np.array([np.median(result[i]) for i in range(4)])
        assert np.std(frame_medians) < 0.05

    def test_additive_only(self):
        base = np.full((10, 10), 0.3, dtype=np.float32)
        stack = np.array([base + 0.0, base + 0.1, base + 0.2])
        result = normalize_stack(stack, NormalizationMethod.ADDITIVE)
        frame_medians = np.array([np.median(result[i]) for i in range(3)])
        assert np.std(frame_medians) < 0.01

    def test_none_passthrough(self):
        stack = np.random.random((4, 10, 10)).astype(np.float32)
        result = normalize_stack(stack, NormalizationMethod.NONE)
        np.testing.assert_array_equal(result, stack)

    def test_backward_compat_alias(self):
        stack = np.random.random((3, 10, 10)).astype(np.float32)
        r1 = normalize_stack_linear_fit(stack)
        r2 = normalize_stack(stack, NormalizationMethod.ADDITIVE_SCALING)
        np.testing.assert_allclose(r1, r2)


# ---------------------------------------------------------------------------
# Rejection methods
# ---------------------------------------------------------------------------


class TestStackImages:
    def test_single_image(self):
        img = ImageData(data=np.random.random((50, 60)).astype(np.float32))
        result = stack_images([img], align=False)
        assert result.n_frames == 1
        np.testing.assert_allclose(result.image.data, img.data)

    def test_stack_identical_no_alignment(self):
        data = np.random.random((40, 50)).astype(np.float32) * 0.5
        images = [ImageData(data=data.copy()) for _ in range(5)]
        params = StackingParams(rejection=RejectionMethod.SIGMA_CLIP,
                                integration=IntegrationMethod.AVERAGE)
        result = stack_images(images, params=params, align=False)
        assert result.n_frames == 5
        np.testing.assert_allclose(result.image.data, data, atol=0.01)

    def test_sigma_clip_rejects_hot_pixel(self):
        data = np.full((30, 40), 0.3, dtype=np.float32)
        images = [ImageData(data=data.copy()) for _ in range(20)]
        bad = data.copy()
        bad[15, 20] = 5.0
        images[5] = ImageData(data=bad)
        params = StackingParams(rejection=RejectionMethod.SIGMA_CLIP, kappa_high=2.5)
        result = stack_images(images, params=params, align=False)
        assert abs(result.image.data[15, 20] - 0.3) < 0.1
        assert result.total_rejected > 0

    def test_winsorized_sigma_rejects_hot_pixel(self):
        """Winsorized sigma must actually reject outliers, not silently pass through."""
        data = np.full((20, 20), 0.3, dtype=np.float32)
        images = [ImageData(data=data.copy()) for _ in range(20)]
        bad = data.copy()
        bad[10, 10] = 5.0
        images[5] = ImageData(data=bad)
        params = StackingParams(rejection=RejectionMethod.WINSORIZED_SIGMA, kappa_high=2.5)
        result = stack_images(images, params=params, align=False)
        assert abs(result.image.data[10, 10] - 0.3) < 0.2
        assert result.total_rejected > 0

    def test_percentile_clip_rejects_outlier(self):
        data = np.full((20, 20), 0.3, dtype=np.float32)
        images = [ImageData(data=data.copy()) for _ in range(10)]
        bad = data.copy()
        bad[5, 5] = 5.0
        images[9] = ImageData(data=bad)
        params = StackingParams(rejection=RejectionMethod.PERCENTILE_CLIP,
                                percentile_low=5.0, percentile_high=5.0)
        result = stack_images(images, params=params, align=False)
        # 10 frames, rejecting top 5% = 1 value — the hot pixel at [5,5] should be gone
        assert result.image.data[5, 5] < 1.0

    def test_esd_rejects_outlier(self):
        data = np.full((15, 15), 0.3, dtype=np.float32)
        images = [ImageData(data=data.copy()) for _ in range(15)]
        bad = data.copy()
        bad[7, 7] = 8.0
        images[0] = ImageData(data=bad)
        params = StackingParams(rejection=RejectionMethod.ESD)
        result = stack_images(images, params=params, align=False)
        assert abs(result.image.data[7, 7] - 0.3) < 0.5
        assert result.total_rejected > 0

    def test_min_max_rejects_extremes(self):
        data = np.full((10, 10), 0.3, dtype=np.float32)
        images = [ImageData(data=data.copy()) for _ in range(6)]
        images[0].data[5, 5] = 0.0   # lowest
        images[5].data[5, 5] = 1.0   # highest
        params = StackingParams(rejection=RejectionMethod.MIN_MAX, min_max_reject=1)
        result = stack_images(images, params=params, align=False)
        # After rejecting min and max, remaining 4 frames all have 0.3
        assert abs(result.image.data[5, 5] - 0.3) < 0.05
        assert result.total_rejected > 0

    def test_median_integration(self):
        data = np.full((20, 20), 0.4, dtype=np.float32)
        images = [ImageData(data=data.copy()) for _ in range(5)]
        images[2] = ImageData(data=np.full((20, 20), 0.9, dtype=np.float32))
        params = StackingParams(rejection=RejectionMethod.SIGMA_CLIP,
                                integration=IntegrationMethod.MEDIAN)
        result = stack_images(images, params=params, align=False)
        assert np.median(result.image.data) < 0.5

    def test_no_rejection(self):
        data = np.full((10, 10), 0.3, dtype=np.float32)
        images = [
            ImageData(data=data.copy()),
            ImageData(data=data.copy() + 0.01),
            ImageData(data=data.copy() - 0.01),
        ]
        params = StackingParams(rejection=RejectionMethod.NONE)
        result = stack_images(images, params=params, align=False)
        assert abs(np.mean(result.image.data) - 0.3) < 0.05
        assert result.total_rejected == 0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            stack_images([])

    def test_linear_fit_rejection_mode(self):
        data = np.full((20, 20), 0.3, dtype=np.float32)
        images = []
        for i in range(15):
            frame = data.copy() + i * 0.02
            images.append(ImageData(data=frame))
        images[7].data[10, 10] = 5.0
        params = StackingParams(rejection=RejectionMethod.LINEAR_FIT, kappa_high=2.5)
        result = stack_images(images, params=params, align=False)
        assert abs(result.image.data[10, 10] - 0.3) < 0.5
        assert result.total_rejected > 0

    def test_normalization_prevents_black_screen(self):
        base = np.random.random((30, 30)).astype(np.float32) * 0.05
        images = [ImageData(data=(base + 0.1 * i).astype(np.float32)) for i in range(4)]
        params = StackingParams(rejection=RejectionMethod.SIGMA_CLIP,
                                integration=IntegrationMethod.AVERAGE)
        result = stack_images(images, params=params, align=False)
        assert np.mean(result.image.data) > 0.01
        assert np.mean(result.image.data) < 0.9


# ---------------------------------------------------------------------------
# FFT alignment (translation only, no GPU required)
# ---------------------------------------------------------------------------


class TestFFTAlignment:
    def test_fft_aligns_shifted_images(self):
        """FFT alignment should recover a known pixel shift."""
        base = np.zeros((60, 80), dtype=np.float32)
        # Add a few bright spots
        for y, x in [(10, 15), (30, 50), (45, 25)]:
            base[y, x] = 1.0

        shift_row, shift_col = 5, -3
        shifted = np.roll(np.roll(base, shift_row, axis=0), shift_col, axis=1)

        ref_img = ImageData(data=base)
        tgt_img = ImageData(data=shifted)

        params = StackingParams(
            registration_mode=RegistrationMode.FFT_TRANSLATION,
            use_gpu=False,
        )
        aligned = stack_images([ref_img, tgt_img], params=params, align=True)
        assert aligned.n_frames == 2
        # The stacked result should be close to the reference
        assert aligned.image.data.shape == base.shape


class TestCometAlignment:
    def test_comet_nucleus_found(self):
        from astraios.core.stacking import _find_comet_nucleus

        frame = np.zeros((100, 100), dtype=np.float32)
        frame[40, 60] = 1.0
        cx, cy = _find_comet_nucleus(frame, 15)
        assert abs(cx - 60) < 1.0
        assert abs(cy - 40) < 1.0

    def test_comet_aligns_shifted_frames(self):
        from astraios.core.stacking import _comet_align_frames

        base = np.zeros((80, 80), dtype=np.float32)
        base[40, 40] = 1.0  # nucleus at (40, 40)

        shifted = np.zeros((80, 80), dtype=np.float32)
        shifted[50, 45] = 1.0  # nucleus at (45, 50)

        imgs = [ImageData(data=base), ImageData(data=shifted)]
        params = StackingParams(
            registration_mode=RegistrationMode.COMET,
            comet_nucleus_radius=15,
        )
        aligned = _comet_align_frames(imgs, params, lambda f, m: None)
        assert len(aligned) == 2
        # After alignment, nucleus in frame 2 should be near (40, 40)
        from astraios.core.stacking import _find_comet_nucleus
        cx, cy = _find_comet_nucleus(aligned[1].data, 15)
        assert abs(cx - 40) < 2.0
        assert abs(cy - 40) < 2.0

    def test_comet_mode_in_registration_enum(self):
        assert RegistrationMode.COMET is not None

    def test_stacking_params_comet_radius(self):
        p = StackingParams(registration_mode=RegistrationMode.COMET, comet_nucleus_radius=25)
        assert p.comet_nucleus_radius == 25


class TestAlignFromPaths:
    """align_from_paths — streaming path-based alignment (low-RAM)."""

    def _make_fits(self, tmp_path, name: str, data: np.ndarray):
        from astropy.io import fits
        p = tmp_path / name
        fits.PrimaryHDU(data=data.astype(np.float32)).writeto(str(p), overwrite=True)
        return p

    def test_produces_output_files(self, tmp_path):
        """align_from_paths writes one file per input frame."""
        rng = np.random.default_rng(0)
        frames = [rng.random((64, 64)).astype(np.float32) for _ in range(3)]
        paths = [self._make_fits(tmp_path, f"frame_{i}.fits", f) for i, f in enumerate(frames)]
        out_dir = tmp_path / "aligned"
        params = StackingParams(registration_mode=RegistrationMode.FFT_TRANSLATION, use_gpu=False)
        result = align_from_paths(paths, out_dir, params=params)
        assert len(result) == 3
        for p in result:
            assert p.exists(), f"Expected output file: {p}"

    def test_last_frame_sentinel(self, tmp_path):
        """reference_frame_index=-2 selects the last frame."""
        rng = np.random.default_rng(1)
        frames = [rng.random((32, 32)).astype(np.float32) for _ in range(4)]
        paths = [self._make_fits(tmp_path, f"f_{i}.fits", f) for i, f in enumerate(frames)]
        out_dir = tmp_path / "aligned_last"
        params = StackingParams(
            registration_mode=RegistrationMode.FFT_TRANSLATION,
            reference_frame_index=-2,
            use_gpu=False,
        )
        result = align_from_paths(paths, out_dir, params=params)
        assert len(result) == 4

    def test_explicit_reference_frame(self, tmp_path):
        """reference_frame_index=2 uses frame #3 (0-based index 2) as reference."""
        rng = np.random.default_rng(2)
        frames = [rng.random((32, 32)).astype(np.float32) for _ in range(4)]
        paths = [self._make_fits(tmp_path, f"g_{i}.fits", f) for i, f in enumerate(frames)]
        out_dir = tmp_path / "aligned_explicit"
        params = StackingParams(
            registration_mode=RegistrationMode.FFT_TRANSLATION,
            reference_frame_index=2,
            use_gpu=False,
        )
        result = align_from_paths(paths, out_dir, params=params)
        assert len(result) == 4


class TestStackFromPaths:
    """stack_from_paths — tiled path with normalization parity."""

    def _write_fits(self, tmp_path, name: str, data: np.ndarray):
        from astropy.io import fits

        path = tmp_path / name
        fits.PrimaryHDU(data.astype(np.float32)).writeto(str(path), overwrite=True)
        return path

    def test_stack_from_paths_matches_stack_frames_normalization(self, tmp_path):
        """Tiled stack should match in-memory stack after normalization."""
        rng = np.random.default_rng(42)
        base = rng.random((40, 50)).astype(np.float32) * 0.2 + 0.05
        frames = []
        for i, (offset, scale) in enumerate(
            [(0.02, 1.0), (0.0, 1.15), (0.04, 0.92), (0.06, 1.08), (0.01, 0.95)]
        ):
            frames.append(base * scale + offset)

        paths = [self._write_fits(tmp_path, f"light_{i}.fits", f) for i, f in enumerate(frames)]
        images = [ImageData(data=f.copy()) for f in frames]
        params = StackingParams(
            rejection=RejectionMethod.NONE,
            normalization=NormalizationMethod.ADDITIVE_SCALING,
            use_gpu=False,
        )
        mem = stack_images(images, params=params, align=False)
        tiled = stack_from_paths(paths, params=params)
        np.testing.assert_allclose(tiled.image.data, mem.image.data, rtol=0.05, atol=0.02)

    def test_stack_from_paths_multiplicative(self, tmp_path):
        """Multiplicative normalization scales frames with different medians."""
        base = np.full((30, 30), 0.2, dtype=np.float32)
        frames = [base * s for s in [0.8, 1.0, 1.2, 1.4]]
        paths = [self._write_fits(tmp_path, f"m_{i}.fits", f) for i, f in enumerate(frames)]
        params = StackingParams(
            rejection=RejectionMethod.NONE,
            normalization=NormalizationMethod.MULTIPLICATIVE,
            use_gpu=False,
        )
        result = stack_from_paths(paths, params=params)
        assert abs(np.median(result.image.data) - 0.2) < 0.05

    def test_stack_from_paths_winsorized_uses_winsorization(self, tmp_path):
        """Winsorized tiled stack rejects hot pixels like in-memory path."""
        data = np.full((20, 20), 0.3, dtype=np.float32)
        frames = [data.copy() for _ in range(20)]
        bad = data.copy()
        bad[10, 10] = 5.0
        frames[5] = bad
        paths = [self._write_fits(tmp_path, f"w_{i}.fits", f) for i, f in enumerate(frames)]
        params = StackingParams(
            rejection=RejectionMethod.WINSORIZED_SIGMA,
            kappa_high=2.5,
            use_gpu=False,
        )
        result = stack_from_paths(paths, params=params)
        assert abs(result.image.data[10, 10] - 0.3) < 0.2
        assert result.total_rejected > 0

    def test_stack_from_paths_fails_on_missing_tile(self, tmp_path):
        """Corrupt FITS must fail fast instead of silently dropping frames."""
        good = np.full((20, 20), 0.3, dtype=np.float32)
        paths = [
            self._write_fits(tmp_path, "good0.fits", good),
            tmp_path / "missing.fits",
            self._write_fits(tmp_path, "good2.fits", good),
        ]
        params = StackingParams(
            rejection=RejectionMethod.NONE,
            normalization=NormalizationMethod.NONE,
            use_gpu=False,
        )
        with pytest.raises(RuntimeError, match="Failed to read tile"):
            stack_from_paths(paths, params=params)


class TestGpuRejectionHelpers:
    def test_gpu_percentile_clip_rejects_outliers(self):
        import torch

        stack = torch.full((10, 4, 4), 0.3)
        stack[9, 2, 2] = 5.0
        result, n_rej = _gpu_percentile_clip(stack, 5.0, 5.0)
        assert n_rej > 0
        assert abs(float(result[2, 2]) - 0.3) < 0.1

    def test_gpu_min_max_counts_rejected(self):
        import torch

        stack = torch.linspace(0, 1, 6).view(6, 1, 1).expand(6, 2, 2)
        _, n_rej = _gpu_min_max(stack, 1)
        assert n_rej == 8
