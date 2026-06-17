"""Tests for image export to TIFF, PNG, JPEG."""

import numpy as np
import pytest
from pathlib import Path

from astraios.core.image_io import ImageData, save_image


def _make_mono_image(w=64, h=64) -> ImageData:
    data = np.linspace(0, 1, w * h, dtype=np.float32).reshape(h, w)
    return ImageData(data=data)


def _make_rgb_image(w=64, h=64) -> ImageData:
    r = np.linspace(0, 1, w * h, dtype=np.float32).reshape(h, w)
    g = np.linspace(1, 0, w * h, dtype=np.float32).reshape(h, w)
    b = np.full((h, w), 0.5, dtype=np.float32)
    data = np.stack([r, g, b], axis=0)
    return ImageData(data=data)


class TestSaveTIFF:
    def test_save_mono_tiff_16bit(self, tmp_path: Path):
        img = _make_mono_image()
        path = tmp_path / "mono.tif"
        save_image(img, path, bit_depth=16)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_save_mono_tiff_8bit(self, tmp_path: Path):
        img = _make_mono_image()
        path = tmp_path / "mono_8.tif"
        save_image(img, path, bit_depth=8)
        assert path.exists()

    def test_save_rgb_tiff_16bit(self, tmp_path: Path):
        img = _make_rgb_image()
        path = tmp_path / "rgb.tif"
        save_image(img, path, bit_depth=16)
        assert path.exists()
        # Pillow falls back to 8-bit for RGB 16-bit, but file should still be valid

    def test_save_rgb_tiff_8bit(self, tmp_path: Path):
        img = _make_rgb_image()
        path = tmp_path / "rgb_8.tif"
        save_image(img, path, bit_depth=8)
        assert path.exists()


class TestSavePNG:
    def test_save_mono_png_16bit(self, tmp_path: Path):
        img = _make_mono_image()
        path = tmp_path / "mono.png"
        save_image(img, path, bit_depth=16)
        assert path.exists()

    def test_save_mono_png_8bit(self, tmp_path: Path):
        img = _make_mono_image()
        path = tmp_path / "mono_8.png"
        save_image(img, path, bit_depth=8)
        assert path.exists()

    def test_save_rgb_png_16bit(self, tmp_path: Path):
        img = _make_rgb_image()
        path = tmp_path / "rgb.png"
        save_image(img, path, bit_depth=16)
        assert path.exists()
        # Pillow falls back to 8-bit for RGB 16-bit, but file should still be valid

    def test_save_rgb_png_8bit(self, tmp_path: Path):
        img = _make_rgb_image()
        path = tmp_path / "rgb_8.png"
        save_image(img, path, bit_depth=8)
        assert path.exists()


class TestSaveJPEG:
    def test_save_mono_jpeg(self, tmp_path: Path):
        img = _make_mono_image()
        path = tmp_path / "mono.jpg"
        save_image(img, path, jpeg_quality=90)
        assert path.exists()

    def test_save_rgb_jpeg(self, tmp_path: Path):
        img = _make_rgb_image()
        path = tmp_path / "rgb.jpg"
        save_image(img, path, jpeg_quality=95)
        assert path.exists()

    def test_jpeg_quality(self, tmp_path: Path):
        img = _make_rgb_image()
        path_hi = tmp_path / "rgb_hi.jpg"
        path_lo = tmp_path / "rgb_lo.jpg"
        save_image(img, path_hi, jpeg_quality=99)
        save_image(img, path_lo, jpeg_quality=10)
        # Higher quality should produce larger file
        assert path_hi.stat().st_size >= path_lo.stat().st_size


class TestSaveFITS:
    def test_save_fits_via_save_image(self, tmp_path: Path):
        img = _make_mono_image()
        path = tmp_path / "test.fits"
        save_image(img, path)
        assert path.exists()

    def test_save_fits_with_header(self, tmp_path: Path):
        img = _make_mono_image()
        img.header["EXPTIME"] = 120.0
        img.header["OBJECT"] = "M31"
        path = tmp_path / "m31.fits"
        save_image(img, path)
        assert path.exists()


class TestSaveXISF:
    def test_save_xisf_via_save_image(self, tmp_path: Path):
        img = _make_mono_image()
        path = tmp_path / "test.xisf"
        save_image(img, path)
        assert path.exists()


class TestUnsupportedFormat:
    def test_unsupported_extension_raises(self, tmp_path: Path):
        img = _make_mono_image()
        path = tmp_path / "test.bmp"
        with pytest.raises(ValueError, match="Unsupported export format"):
            save_image(img, path)
