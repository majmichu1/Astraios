"""Tests for astraios.core.batch_convert (ported from Seti Astro Suite Pro)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astraios.core.batch_convert import (
    ALLOWED_BIT_DEPTHS,
    BatchConvertParams,
    _resolve_bit_depth,
    batch_convert,
)
from astraios.core.image_io import ImageData, load_image, save_fits


def _make_fits(path: Path, mono: bool = True, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    if mono:
        data = rng.random((32, 24), dtype=np.float32)
    else:
        data = rng.random((3, 32, 24)).astype(np.float32)
    save_fits(ImageData(data=data, header={"OBJECT": "M42"}), path)
    return path


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    d = tmp_path / "src"
    d.mkdir()
    _make_fits(d / "mono.fits", mono=True, seed=1)
    _make_fits(d / "color.fits", mono=False, seed=2)
    return d


class TestBatchConvert:
    def test_convert_fits_to_tiff(self, source_dir: Path, tmp_path: Path):
        out_dir = tmp_path / "out"
        paths = sorted(source_dir.glob("*.fits"))
        params = BatchConvertParams(output_format=".tiff", bit_depth=16)

        outputs = batch_convert(paths, out_dir, params)

        assert len(outputs) == 2
        for out_path in outputs:
            assert out_path.exists()
            assert out_path.suffix == ".tiff"
            reloaded = load_image(out_path)
            assert reloaded.data.dtype == np.float32

        mono_src = load_image(source_dir / "mono.fits")
        mono_out = load_image(out_dir / "mono.tiff")
        assert mono_out.data.shape == mono_src.data.shape

        color_src = load_image(source_dir / "color.fits")
        color_out = load_image(out_dir / "color.tiff")
        assert color_out.data.ndim == color_src.data.ndim

    def test_convert_fits_to_png_8bit(self, source_dir: Path, tmp_path: Path):
        out_dir = tmp_path / "out"
        paths = [source_dir / "mono.fits"]
        params = BatchConvertParams(output_format=".png", bit_depth=8)

        outputs = batch_convert(paths, out_dir, params)

        assert len(outputs) == 1
        assert outputs[0].exists()
        reloaded = load_image(outputs[0])
        src = load_image(source_dir / "mono.fits")
        # 8-bit PNG round-trips mono through Pillow's RGB convert (existing
        # image_io behavior); compare pixel dimensions, not channel count.
        assert reloaded.data.shape[-2:] == src.data.shape[-2:]

    def test_progress_callback_invoked(self, source_dir: Path, tmp_path: Path):
        events: list[tuple[float, str]] = []
        params = BatchConvertParams(output_format=".tiff")
        batch_convert(
            sorted(source_dir.glob("*.fits")), tmp_path / "out", params,
            progress=lambda f, m: events.append((f, m)),
        )
        assert events
        assert events[-1][0] == 1.0

    def test_skips_unreadable_file(self, source_dir: Path, tmp_path: Path):
        bogus = source_dir / "bogus.fits"
        bogus.write_bytes(b"not a real fits file at all")

        params = BatchConvertParams(output_format=".tiff")
        outputs = batch_convert(
            [source_dir / "mono.fits", bogus], tmp_path / "out", params
        )

        assert len(outputs) == 1
        assert outputs[0].name == "mono.tiff"

    def test_skip_existing(self, source_dir: Path, tmp_path: Path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "mono.tiff").write_bytes(b"placeholder")

        params = BatchConvertParams(output_format=".tiff", skip_existing=True)
        outputs = batch_convert([source_dir / "mono.fits"], out_dir, params)

        assert outputs == []
        # Existing placeholder must not have been overwritten.
        assert (out_dir / "mono.tiff").read_bytes() == b"placeholder"

    def test_invalid_output_format_raises(self, source_dir: Path, tmp_path: Path):
        params = BatchConvertParams(output_format=".bmp")
        with pytest.raises(ValueError):
            batch_convert([source_dir / "mono.fits"], tmp_path / "out", params)

    def test_creates_output_dir(self, source_dir: Path, tmp_path: Path):
        out_dir = tmp_path / "nested" / "out"
        params = BatchConvertParams(output_format=".fits")
        batch_convert([source_dir / "mono.fits"], out_dir, params)
        assert out_dir.is_dir()


class TestBitDepthResolution:
    def test_all_formats_have_allowed_depths(self):
        from astraios.core.batch_convert import OUTPUT_FORMATS
        for fmt in OUTPUT_FORMATS:
            assert fmt in ALLOWED_BIT_DEPTHS

    @pytest.mark.parametrize(
        "fmt, expected",
        [(".png", 8), (".jpg", 8), (".jpeg", 8), (".tif", 16), (".tiff", 16),
         (".fits", 32), (".fit", 32), (".xisf", 32)],
    )
    def test_auto_picks_sane_default(self, fmt, expected):
        assert _resolve_bit_depth("auto", fmt) == expected

    def test_explicit_valid_depth_kept(self):
        assert _resolve_bit_depth(8, ".png") == 8
        assert _resolve_bit_depth(16, ".tiff") == 16

    def test_invalid_explicit_depth_falls_back(self):
        # 32-bit isn't offered for PNG; must fall back to an allowed value.
        assert _resolve_bit_depth(32, ".png") in ALLOWED_BIT_DEPTHS[".png"]
