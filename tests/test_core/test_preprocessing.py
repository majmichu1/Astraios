"""Tests for the batch preprocessing pipeline (calibrate -> register -> stack)."""

import numpy as np
import pytest
from astropy.io import fits
from pathlib import Path

from astraios.core.preprocessing import run_preprocessing
from astraios.core.stacking import RejectionMethod, StackingParams


def _write_light(path: Path, shift_y: int, shift_x: int, seed: int) -> None:
    """A starfield shifted by (shift_y, shift_x), simulating dither drift."""
    rng = np.random.default_rng(seed)
    h, w = 120, 160
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w), 0.06, dtype=np.float32)
    # Fixed star pattern (same sky), shifted per frame
    star_pos = [(30, 40), (60, 100), (90, 60), (25, 120), (85, 130)]
    for cy, cx in star_pos:
        img += (0.6 * np.exp(
            -(((yy - cy - shift_y) ** 2 + (xx - cx - shift_x) ** 2) / (2 * 1.8 ** 2))
        )).astype(np.float32)
    img += rng.normal(0, 0.004, (h, w)).astype(np.float32)
    img = np.clip(img, 0, 1)
    hdu = fits.PrimaryHDU(img)
    hdu.header["IMAGETYP"] = "Light"
    hdu.writeto(str(path))


@pytest.fixture
def dithered_lights(tmp_path):
    paths = []
    shifts = [(0, 0), (3, -2), (-2, 4), (5, 1), (-4, -3), (2, 3)]
    for i, (sy, sx) in enumerate(shifts):
        p = tmp_path / f"light_{i:02d}.fits"
        _write_light(p, sy, sx, seed=i)
        paths.append(p)
    return paths


class TestRunPreprocessing:
    def test_register_actually_aligns(self, dithered_lights, tmp_path):
        """The register stage must run: stacked stars stay sharp, not smeared."""
        out = tmp_path / "out"
        result = run_preprocessing(
            dithered_lights,
            output_dir=out,
            calibrate=False,
            register=True,
            stack=True,
            cosmetic=False,
            stacking_params=StackingParams(rejection=RejectionMethod.NONE),
        )
        assert result.stacked_image is not None
        assert (out / "aligned").exists() and any((out / "aligned").iterdir())

        stacked = result.stacked_image.data
        # Aligned stacking keeps the star peak close to a single frame's peak.
        # Unaligned averaging of +/-5 px dithers smears it far below.
        peak = float(stacked.max())
        assert peak > 0.45, f"stars smeared (peak={peak:.3f}) — registration did not run"

    def test_stack_without_calibration(self, dithered_lights, tmp_path):
        """calibrate=False + stack=True must still stack (was silently skipped)."""
        result = run_preprocessing(
            dithered_lights,
            output_dir=tmp_path / "out2",
            calibrate=False,
            register=False,
            stack=True,
            cosmetic=False,
            stacking_params=StackingParams(rejection=RejectionMethod.NONE),
        )
        assert result.stacked_image is not None

    def test_calibrate_without_output_dir_keeps_calibration(self, dithered_lights, tmp_path):
        """Without output_dir, calibrated pixels must still reach the stack."""
        bias = np.full((120, 160), 0.02, dtype=np.float32)
        bias_paths = []
        for i in range(3):
            p = tmp_path / f"bias_{i}.fits"
            hdu = fits.PrimaryHDU(bias)
            hdu.header["IMAGETYP"] = "Bias"
            hdu.writeto(str(p))
            bias_paths.append(p)

        result = run_preprocessing(
            dithered_lights,
            bias_paths=bias_paths,
            output_dir=None,
            calibrate=True,
            register=False,
            stack=True,
            cosmetic=False,
            stacking_params=StackingParams(rejection=RejectionMethod.NONE),
        )
        assert result.stacked_image is not None
        assert result.n_calibrated == len(dithered_lights)
        # Calibrated paths must be cal_*.fits files, not the raw inputs
        assert all(p.name.startswith("cal_") for p in result.calibrated_paths)
        # Bias subtraction visible: background below the raw 0.06 floor
        bg = float(np.median(result.stacked_image.data))
        assert bg < 0.055, f"bias subtraction lost (bg={bg:.4f})"
