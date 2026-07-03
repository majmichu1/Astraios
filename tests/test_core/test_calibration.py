"""Tests for calibration pipeline."""

import numpy as np
import pytest
from astropy.io import fits
from pathlib import Path

from astraios.core.calibration import (
    calibrate_light,
    create_master_bias,
    create_master_dark,
    create_master_flat,
)
from astraios.core.image_io import FrameType, ImageData


def _make_fits(tmp_path, name, data, image_type="Light"):
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    hdu.header["IMAGETYP"] = image_type
    path = tmp_path / name
    hdu.writeto(str(path))
    return path


@pytest.fixture
def bias_files(tmp_path):
    """Create synthetic bias frames with small fixed offset + noise."""
    paths = []
    for i in range(5):
        data = np.full((50, 60), 0.01, dtype=np.float32) + np.random.normal(0, 0.002, (50, 60)).astype(np.float32)
        data = np.clip(data, 0, 1)
        paths.append(_make_fits(tmp_path, f"bias_{i}.fits", data, "Bias"))
    return paths


@pytest.fixture
def dark_files(tmp_path):
    """Create synthetic dark frames."""
    paths = []
    for i in range(5):
        data = np.full((50, 60), 0.05, dtype=np.float32) + np.random.normal(0, 0.005, (50, 60)).astype(np.float32)
        data = np.clip(data, 0, 1)
        paths.append(_make_fits(tmp_path, f"dark_{i}.fits", data, "Dark"))
    return paths


@pytest.fixture
def flat_files(tmp_path):
    """Create synthetic flat frames with vignetting pattern."""
    paths = []
    y, x = np.mgrid[0:50, 0:60]
    vignette = 1.0 - 0.3 * ((x - 30) ** 2 + (y - 25) ** 2) / (30 ** 2 + 25 ** 2)
    for i in range(5):
        data = vignette + np.random.normal(0, 0.01, (50, 60))
        data = np.clip(data, 0, 1).astype(np.float32)
        paths.append(_make_fits(tmp_path, f"flat_{i}.fits", data, "Flat"))
    return paths


class TestMasterCreation:
    def test_master_bias(self, bias_files):
        result = create_master_bias(bias_files)
        assert result.master.data.shape == (50, 60)
        assert result.n_frames == 5
        assert result.method == "median"
        # Master bias should be close to 0.01
        assert abs(np.median(result.master.data) - 0.01) < 0.01

    def test_master_dark(self, dark_files):
        result = create_master_dark(dark_files)
        assert result.master.data.shape == (50, 60)
        assert abs(np.median(result.master.data) - 0.05) < 0.02

    def test_master_flat(self, flat_files):
        result = create_master_flat(flat_files)
        # Flat should be normalized to mean ≈ 1.0
        assert abs(np.mean(result.master.data) - 1.0) < 0.1

    def test_master_dark_with_bias(self, dark_files, bias_files):
        bias_result = create_master_bias(bias_files)
        dark_result = create_master_dark(dark_files, master_bias=bias_result.master)
        assert dark_result.master.data.shape == (50, 60)
        # Dark - bias should be lower than dark alone
        dark_only = create_master_dark(dark_files)
        assert np.median(dark_result.master.data) < np.median(dark_only.master.data) + 0.02

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="No bias"):
            create_master_bias([])


class TestCalibration:
    def test_calibrate_light(self):
        light = ImageData(data=np.full((50, 60), 0.5, dtype=np.float32))
        bias = ImageData(data=np.full((50, 60), 0.01, dtype=np.float32))
        dark = ImageData(data=np.full((50, 60), 0.05, dtype=np.float32))
        flat = ImageData(data=np.ones((50, 60), dtype=np.float32))

        result = calibrate_light(light, bias, dark, flat)
        expected = 0.5 - 0.01 - 0.05
        assert abs(np.median(result.data) - expected) < 0.01

    def test_calibrate_no_masters(self):
        light = ImageData(data=np.full((50, 60), 0.5, dtype=np.float32))
        result = calibrate_light(light)
        np.testing.assert_allclose(result.data, light.data, atol=1e-6)

    def test_calibrate_light_scales_dark_by_exposure(self):
        light = ImageData(
            data=np.full((50, 60), 0.5, dtype=np.float32),
            header={"EXPTIME": 120.0},
        )
        dark = ImageData(
            data=np.full((50, 60), 0.06, dtype=np.float32),
            header={"EXPTIME": 60.0},
        )
        result = calibrate_light(light, master_dark=dark)
        expected = 0.5 - 0.06 * (120.0 / 60.0)
        assert abs(np.median(result.data) - expected) < 0.01

    def test_calibrate_preserves_linear_negative(self):
        light = ImageData(data=np.full((50, 60), 0.02, dtype=np.float32))
        bias = ImageData(data=np.full((50, 60), 0.05, dtype=np.float32))
        result = calibrate_light(light, master_bias=bias)
        assert np.median(result.data) < 0


class TestTiledMaster:
    """The bounded-memory tiled-median path must match the standard np.median."""

    def _frames(self, tmp_path, n, shape, prefix):
        rng = np.random.default_rng(1)
        base = rng.random(shape).astype(np.float32) * 0.1 + 0.2
        paths = []
        for i in range(n):
            data = (base + rng.normal(0, 0.01, shape)).astype(np.float32)
            idx = tuple(rng.integers(0, s) for s in shape)
            data[idx] = 0.95  # cosmic-ray spike the median must reject
            paths.append(_make_fits(tmp_path, f"{prefix}_{i}.fits",
                                    np.clip(data, 0, 1), "Bias"))
        return paths

    def _ground_truth(self, paths):
        from astraios.core.image_io import load_image
        stack = np.stack([load_image(p).data for p in paths])
        return np.median(stack, axis=0).astype(np.float32)

    @pytest.mark.parametrize("n,shape", [(7, (40, 50)), (6, (40, 50)), (6, (3, 24, 28))])
    def test_tiled_matches_npmedian(self, tmp_path, monkeypatch, n, shape):
        import astraios.core.calibration as cal
        paths = self._frames(tmp_path, n, shape, "bias")
        # Force the tiled path with a tiny budget.
        monkeypatch.setattr(cal, "_MASTER_MEM_BUDGET", 1)
        got = cal.create_master_bias(paths).master.data
        expected = self._ground_truth(paths)
        assert got.shape == shape
        assert np.max(np.abs(got - expected)) < 1e-5
        assert got.max() < 0.6  # spike rejected

    def test_tiled_subtraction_matches(self, tmp_path, monkeypatch):
        import astraios.core.calibration as cal
        from astraios.core.image_io import load_image
        paths = self._frames(tmp_path, 6, (40, 50), "dark")
        bias = ImageData(data=self._ground_truth(paths) * 0.5)
        monkeypatch.setattr(cal, "_MASTER_MEM_BUDGET", 1)
        got = cal.create_master_dark(paths, master_bias=bias).master.data
        expected = np.median(
            np.stack([load_image(p).data - bias.data for p in paths]), axis=0
        ).astype(np.float32)
        assert np.max(np.abs(got - expected)) < 1e-5


class TestBatchAndDiskCalibration:
    """calibrate_lights_batch / calibrate_lights_to_disk parity and path tracking."""

    def _lights_and_masters(self, tmp_path):
        rng = np.random.default_rng(42)
        light_paths = []
        for i in range(4):
            data = np.clip(
                0.2 + rng.normal(0, 0.05, (40, 50)).astype(np.float32), 0, 1
            )
            light_paths.append(_make_fits(tmp_path, f"light_{i}.fits", data, "Light"))
        bias = ImageData(data=np.full((40, 50), 0.01, dtype=np.float32))
        flat = ImageData(data=np.clip(
            1.0 - 0.2 * rng.random((40, 50)).astype(np.float32), 0.5, 1.0
        ))
        return light_paths, bias, flat

    def test_batch_output_dir_retargets_file_path(self, tmp_path):
        from astraios.core.calibration import calibrate_lights_batch
        light_paths, bias, flat = self._lights_and_masters(tmp_path)
        out_dir = tmp_path / "cal"
        out_dir.mkdir()
        results = calibrate_lights_batch(
            light_paths, master_bias=bias, master_flat=flat, output_dir=out_dir
        )
        assert len(results) == 4
        for src, img in zip(light_paths, results):
            expected = out_dir / f"cal_{src.stem}.fits"
            assert img.file_path == expected
            assert expected.exists()

    def test_to_disk_bit_identical_to_batch(self, tmp_path):
        from astraios.core.calibration import (
            calibrate_lights_batch,
            calibrate_lights_to_disk,
        )
        from astraios.core.image_io import load_image
        light_paths, bias, flat = self._lights_and_masters(tmp_path)

        in_memory = calibrate_lights_batch(light_paths, master_bias=bias, master_flat=flat)

        out_dir = tmp_path / "cal_stream"
        out_paths = calibrate_lights_to_disk(
            light_paths, out_dir, master_bias=bias, master_flat=flat
        )
        assert len(out_paths) == len(in_memory)
        for mem, p in zip(in_memory, out_paths):
            assert p.exists()
            disk = load_image(p)
            assert np.array_equal(mem.data, disk.data), "disk round-trip must be bit-identical"

    def test_to_disk_skips_unreadable_frame(self, tmp_path):
        from astraios.core.calibration import calibrate_lights_to_disk
        light_paths, bias, _ = self._lights_and_masters(tmp_path)
        bad = tmp_path / "broken.fits"
        bad.write_bytes(b"not a fits file")
        out_paths = calibrate_lights_to_disk(
            [light_paths[0], bad, light_paths[1]], tmp_path / "out", master_bias=bias
        )
        assert len(out_paths) == 2
