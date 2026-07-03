"""Tests for local normalization (in-memory and streamed disk paths)."""

import numpy as np
from astropy.io import fits

from astraios.core.local_normalization import (
    LocalNormParams,
    local_normalize,
    local_normalize_to_disk,
)


def _frames_with_gradients(n=6, h=100, w=140, seed=3):
    """Same sky signal per frame, but each frame gets its own LP gradient."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    signal = 0.08 + 0.5 * np.exp(-(((yy - 50) ** 2 + (xx - 70) ** 2) / (2 * 3.0 ** 2)))
    frames = []
    for _ in range(n):
        gx, gy = rng.uniform(-0.04, 0.04, 2)
        gradient = gx * (xx / w - 0.5) + gy * (yy / h - 0.5)
        f = (signal + gradient + rng.normal(0, 0.003, (h, w))).astype(np.float32)
        frames.append(f)
    return frames


def _bg_spread(frames):
    """Max pairwise difference of smoothed backgrounds (gradient mismatch)."""
    from scipy.ndimage import gaussian_filter
    bgs = [gaussian_filter(f, 20) for f in frames]
    return max(
        float(np.max(np.abs(a - b))) for a in bgs for b in bgs if a is not b
    )


class TestLocalNormalizeInMemory:
    def test_removes_gradient_differences(self):
        frames = _frames_with_gradients()
        before = _bg_spread(frames)
        out = local_normalize(np.stack(frames), params=LocalNormParams(sigma=15.0))
        after = _bg_spread(list(out))
        assert after < before * 0.25, f"gradient mismatch {before:.4f} -> {after:.4f}"

    def test_preserves_stars(self):
        frames = _frames_with_gradients()
        out = local_normalize(np.stack(frames), params=LocalNormParams(sigma=15.0))
        # The bright star peak must survive in every corrected frame
        for f in out:
            assert f.max() > 0.4, "star signal lost"

    def test_color_shape(self):
        mono = _frames_with_gradients(n=3, h=40, w=50)
        stack = np.stack([np.stack([m, m * 0.9, m * 0.8]) for m in mono])
        out = local_normalize(stack, params=LocalNormParams(sigma=8.0))
        assert out.shape == stack.shape

    def test_single_frame_passthrough(self):
        one = np.stack(_frames_with_gradients(n=1))
        out = local_normalize(one)
        assert out.shape == one.shape


class TestLocalNormalizeToDisk:
    def _write(self, tmp_path, frames):
        paths = []
        for i, f in enumerate(frames):
            p = tmp_path / f"f{i:02d}.fits"
            hdu = fits.PrimaryHDU(f)
            hdu.header["IMAGETYP"] = "Light"
            hdu.header["CREATOR"] = "Astraios"
            hdu.writeto(str(p))
            paths.append(p)
        return paths

    def test_streamed_removes_gradient_differences(self, tmp_path):
        from astraios.core.image_io import load_image

        frames = _frames_with_gradients()
        paths = self._write(tmp_path, frames)
        out_paths = local_normalize_to_disk(
            paths, tmp_path / "ln", params=LocalNormParams(sigma=15.0)
        )
        assert len(out_paths) == len(paths)
        corrected = [load_image(p).data for p in out_paths]
        before = _bg_spread(frames)
        after = _bg_spread(corrected)
        assert after < before * 0.25, f"gradient mismatch {before:.4f} -> {after:.4f}"

    def test_skips_bad_frame(self, tmp_path):
        frames = _frames_with_gradients(n=3)
        paths = self._write(tmp_path, frames)
        bad = tmp_path / "bad.fits"
        bad.write_bytes(b"garbage")
        out = local_normalize_to_disk(
            [paths[0], bad, paths[1], paths[2]], tmp_path / "ln"
        )
        assert len(out) == 3

    def test_too_few_frames_returns_empty(self, tmp_path):
        frames = _frames_with_gradients(n=1)
        paths = self._write(tmp_path, frames)
        assert local_normalize_to_disk(paths, tmp_path / "ln") == []


class TestStackFromPathsLocal:
    def test_local_normalization_applied_in_disk_stack(self, tmp_path):
        """Choosing LOCAL in the disk path must correct gradients, not skip."""
        from astraios.core.stacking import (
            NormalizationMethod,
            RejectionMethod,
            StackingParams,
            stack_from_paths,
        )
        frames = _frames_with_gradients()
        paths = []
        for i, f in enumerate(frames):
            p = tmp_path / f"a{i:02d}.fits"
            hdu = fits.PrimaryHDU(f)
            hdu.header["CREATOR"] = "Astraios"
            hdu.writeto(str(p))
            paths.append(p)

        params_local = StackingParams(
            normalization=NormalizationMethod.LOCAL,
            rejection=RejectionMethod.NONE,
        )
        params_none = StackingParams(
            normalization=NormalizationMethod.NONE,
            rejection=RejectionMethod.NONE,
        )
        r_local = stack_from_paths(paths, params=params_local)
        r_none = stack_from_paths(paths, params=params_none)
        assert r_local.n_frames == len(paths)

        # The LOCAL path must actually run the correction (it used to be
        # silently skipped, making LOCAL identical to NONE)...
        diff = float(np.max(np.abs(r_local.image.data - r_none.image.data)))
        assert diff > 1e-4, "LOCAL was skipped: result identical to NONE"

        # ...and must match stacking the explicitly stream-corrected frames.
        corrected_paths = local_normalize_to_disk(paths, tmp_path / "manual")
        r_manual = stack_from_paths(corrected_paths, params=params_none)
        assert np.allclose(r_local.image.data, r_manual.image.data, atol=1e-6), \
            "LOCAL stack must equal stack of stream-corrected frames"

