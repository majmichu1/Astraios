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
