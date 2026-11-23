from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from astraios.core.image_io import ImageData, load_image, save_image


def _image(channels: int) -> np.ndarray:
    """Asymmetric test image: gradient, top-left marker, different per channel."""
    yy, xx = np.mgrid[0:60, 0:80].astype(np.float32)
    img = np.stack([(0.1 + 0.6 * xx / 80) * f for f in (1.0, 0.7, 0.4)[:channels]])
    img[:, 2:8, 3:12] = np.array([0.95, 0.76, 0.57][:channels])[:, None, None]
    return img if channels == 3 else img[0]


@pytest.mark.parametrize("ext", [".tif", ".png"])
def test_16_bit_colour_export_keeps_16_bits(tmp_path: Path, ext: str) -> None:
    """16-bit RGB TIFF/PNG round-trips at 16-bit precision, not 8-bit."""
    src = _image(3)
    save_image(ImageData(src.copy()), tmp_path / f"c{ext}", bit_depth=16)
    out = load_image(tmp_path / f"c{ext}").data
    assert out.shape == src.shape
    # Regression: Pillow fell back to 8-bit, an error of ~1/255 = 0.0039.
    assert np.abs(out - src).max() < 2.0 / 65535


def test_32_bit_tiff_export_is_exact(tmp_path: Path) -> None:
    """32-bit TIFF stores float32 values exactly."""
    src = _image(3)
    save_image(ImageData(src.copy()), tmp_path / "f.tif", bit_depth=32)
    out = load_image(tmp_path / "f.tif").data
    assert np.array_equal(out, src)


@pytest.mark.parametrize(("ext", "bit_depth"), [(".tif", 8), (".png", 8), (".jpg", 8),
                                                (".tif", 16), (".png", 16)])
def test_mono_stays_mono(tmp_path: Path, ext: str, bit_depth: int) -> None:
    """A mono image saves and loads as mono, not as a 3-channel colour image."""
    src = _image(1)
    save_image(ImageData(src.copy()), tmp_path / f"m{ext}", bit_depth=bit_depth)
    out = load_image(tmp_path / f"m{ext}").data
    assert out.shape == src.shape


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("shape", [(40, 50), (40, 50, 3)])
def test_foreign_integer_tiff_keeps_its_scale(tmp_path: Path, dtype, shape) -> None:
    """An uncompressed TIFF from another program is scaled by its type, not stretched."""
    truth = np.linspace(0.1, 0.46, int(np.prod(shape))).reshape(shape)
    tifffile.imwrite(tmp_path / "foreign.tif", (truth * np.iinfo(dtype).max).astype(dtype))
    out = load_image(tmp_path / "foreign.tif").data
    # Regression: the per-file min/max stretch loaded the 0.46 peak as 1.00.
    assert abs(float(out.max()) - 0.46) < 0.01
