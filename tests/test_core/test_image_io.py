"""Tests for image I/O."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from cosmica.core.image_io import (
    FrameType,
    ImageData,
    auto_stretch_for_display,
    load_fits,
    load_image,
    save_fits,
)


@pytest.fixture
def mono_fits_file(tmp_path):
    """Create a temporary mono FITS file."""
    data = np.random.random((100, 120)).astype(np.float32)
    hdu = fits.PrimaryHDU(data)
    hdu.header["IMAGETYP"] = "Light"
    hdu.header["EXPTIME"] = 300.0
    path = tmp_path / "test_mono.fits"
    hdu.writeto(str(path))
    return path


@pytest.fixture
def color_fits_file(tmp_path):
    """Create a temporary color FITS file."""
    data = np.random.random((3, 100, 120)).astype(np.float32)
    hdu = fits.PrimaryHDU(data)
    hdu.header["IMAGETYP"] = "Light"
    path = tmp_path / "test_color.fits"
    hdu.writeto(str(path))
    return path


@pytest.fixture
def uint16_fits_file(tmp_path):
    """Create a 16-bit unsigned FITS file."""
    data = np.random.randint(0, 65535, (100, 120), dtype=np.uint16)
    hdu = fits.PrimaryHDU(data)
    hdu.header["IMAGETYP"] = "Dark"
    path = tmp_path / "test_uint16.fits"
    hdu.writeto(str(path))
    return path


class TestImageData:
    def test_mono_properties(self):
        data = np.random.random((100, 120)).astype(np.float32)
        img = ImageData(data=data)
        assert not img.is_color
        assert img.channels == 1
        assert img.height == 100
        assert img.width == 120
        assert "120x100" in img.shape_str

    def test_color_properties(self):
        data = np.random.random((3, 100, 120)).astype(np.float32)
        img = ImageData(data=data)
        assert img.is_color
        assert img.channels == 3
        assert img.height == 100
        assert img.width == 120

    def test_to_display_mono(self):
        data = np.random.random((50, 60)).astype(np.float32)
        img = ImageData(data=data)
        rgb = img.to_display(stretch=False)
        assert rgb.shape == (50, 60, 3)
        assert rgb.dtype == np.uint8

    def test_to_display_color(self):
        data = np.random.random((3, 50, 60)).astype(np.float32)
        img = ImageData(data=data)
        rgb = img.to_display(stretch=False)
        assert rgb.shape == (50, 60, 3)
        assert rgb.dtype == np.uint8


class TestFitsIO:
    def test_load_mono(self, mono_fits_file):
        img = load_fits(mono_fits_file)
        assert img.data.ndim == 2
        assert img.data.dtype == np.float32
        assert img.frame_type == FrameType.LIGHT
        assert img.exposure == 300.0

    def test_load_color(self, color_fits_file):
        img = load_fits(color_fits_file)
        assert img.data.ndim == 3
        assert img.data.shape[0] == 3

    def test_load_uint16(self, uint16_fits_file):
        img = load_fits(uint16_fits_file)
        assert img.data.dtype == np.float32
        assert img.data.min() >= 0.0
        assert img.data.max() <= 1.0

    def test_save_and_reload(self, mono_fits_file, tmp_path):
        img = load_fits(mono_fits_file)
        out_path = tmp_path / "saved.fits"
        save_fits(img, out_path)
        assert out_path.exists()
        reloaded = load_fits(out_path)
        assert reloaded.data.shape == img.data.shape
        np.testing.assert_allclose(reloaded.data, img.data, atol=1e-6)

    def test_load_image_auto_detect(self, mono_fits_file):
        img = load_image(mono_fits_file)
        assert img.data.ndim == 2

    def test_frame_type_from_filename(self, tmp_path):
        data = np.random.random((50, 60)).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        path = tmp_path / "dark_001.fits"
        hdu.writeto(str(path))
        img = load_fits(path)
        assert img.frame_type == FrameType.DARK


class TestNormalizeFitsTile:
    def test_fits_tile_normalization_matches_full_load(self, tmp_path):
        from cosmica.core.stacking import _load_fits_tile

        raw = np.random.randint(100, 60000, (40, 50), dtype=np.uint16)
        hdu = fits.PrimaryHDU(raw)
        hdu.header["BZERO"] = 32768
        path = tmp_path / "uint16_bzero.fits"
        hdu.writeto(str(path), overwrite=True)

        full = load_fits(path)
        tile = _load_fits_tile(path, 0, 40)
        np.testing.assert_allclose(tile, full.data, rtol=1e-5, atol=1e-5)


class TestAutoStretch:
    def test_stretch_range(self):
        img = np.random.random((50, 60, 3)).astype(np.float32) * 0.01
        stretched = auto_stretch_for_display(img)
        assert stretched.min() >= 0
        assert stretched.max() <= 1

    def test_stretch_preserves_shape(self):
        img = np.random.random((50, 60, 3)).astype(np.float32)
        stretched = auto_stretch_for_display(img)
        assert stretched.shape == img.shape
