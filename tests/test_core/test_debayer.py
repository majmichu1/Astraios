"""Tests for Bayer mosaic debayering (astraios/core/debayer.py)."""

import numpy as np
import pytest

from astraios.core.debayer import debayer, detect_bayer_pattern

# (row, col) offsets of each colour channel within the 2x2 CFA quad, keyed by pattern.
_PATTERN_OFFSETS = {
    "RGGB": {"R": (0, 0), "G1": (0, 1), "G2": (1, 0), "B": (1, 1)},
    "BGGR": {"B": (0, 0), "G1": (0, 1), "G2": (1, 0), "R": (1, 1)},
    "GRBG": {"G1": (0, 0), "R": (0, 1), "B": (1, 0), "G2": (1, 1)},
    "GBRG": {"G1": (0, 0), "B": (0, 1), "R": (1, 0), "G2": (1, 1)},
}


def _make_solid_mosaic(pattern: str, h: int, w: int, r: float, g: float, b: float) -> np.ndarray:
    """Build a raw Bayer mosaic that represents a perfectly solid RGB color."""
    offs = _PATTERN_OFFSETS[pattern]
    mosaic = np.zeros((h, w), dtype=np.float32)
    mosaic[offs["R"][0]::2, offs["R"][1]::2] = r
    mosaic[offs["G1"][0]::2, offs["G1"][1]::2] = g
    mosaic[offs["G2"][0]::2, offs["G2"][1]::2] = g
    mosaic[offs["B"][0]::2, offs["B"][1]::2] = b
    return mosaic


class TestDetectBayerPattern:
    @pytest.mark.parametrize("key", ["BAYERPAT", "COLORTYP", "CFA-PAT", "BAYER", "CFATYPE"])
    def test_recognizes_each_header_key(self, key):
        assert detect_bayer_pattern({key: "RGGB"}) == "RGGB"

    def test_case_insensitive(self):
        assert detect_bayer_pattern({"BAYERPAT": "bggr"}) == "BGGR"

    def test_returns_none_when_absent(self):
        assert detect_bayer_pattern({}) is None

    def test_returns_none_for_unrecognized_value(self):
        assert detect_bayer_pattern({"BAYERPAT": "MONO"}) is None


class TestDebayerSuperpixel:
    """Superpixel debayer directly samples sensor pixels, so a solid-color mosaic
    must reconstruct that exact color with no interpolation error."""

    @pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
    def test_recovers_solid_color_exactly(self, pattern):
        mosaic = _make_solid_mosaic(pattern, 64, 64, 0.2, 0.5, 0.8)
        out = debayer(mosaic, pattern=pattern, method="superpixel")
        assert out.shape == (3, 32, 32)
        assert np.allclose(out[0], 0.2, atol=1e-5)
        assert np.allclose(out[1], 0.5, atol=1e-5)
        assert np.allclose(out[2], 0.8, atol=1e-5)


class TestDebayerInterpolated:
    def test_bilinear_recovers_solid_color_in_interior(self):
        mosaic = _make_solid_mosaic("RGGB", 64, 64, 0.2, 0.5, 0.8)
        out = debayer(mosaic, pattern="RGGB", method="bilinear")
        assert out.shape == (3, 64, 64)
        interior = out[:, 8:-8, 8:-8]
        expected = np.array([0.2, 0.5, 0.8]).reshape(3, 1, 1)
        assert np.allclose(interior, expected, atol=2e-3)

    def test_vng_output_shape_dtype_and_range(self):
        mosaic = _make_solid_mosaic("RGGB", 64, 64, 0.2, 0.5, 0.8)
        out = debayer(mosaic, pattern="RGGB", method="vng")
        assert out.shape == (3, 64, 64)
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0
        # uint8 quantization inside VNG limits precision to ~1/255
        interior = out[:, 8:-8, 8:-8]
        expected = np.array([0.2, 0.5, 0.8]).reshape(3, 1, 1)
        assert np.allclose(interior, expected, atol=5e-3)

    def test_ea_method_runs_and_matches_solid_color(self):
        mosaic = _make_solid_mosaic("RGGB", 64, 64, 0.2, 0.5, 0.8)
        out = debayer(mosaic, pattern="RGGB", method="ea")
        assert out.shape == (3, 64, 64)
        interior = out[:, 8:-8, 8:-8]
        expected = np.array([0.2, 0.5, 0.8]).reshape(3, 1, 1)
        assert np.allclose(interior, expected, atol=5e-3)


class TestDebayerEdgeCases:
    def test_already_color_input_passes_through_unchanged(self):
        color = np.random.RandomState(0).rand(3, 10, 10).astype(np.float32)
        out = debayer(color)
        assert out.shape == color.shape
        assert np.array_equal(out, color)

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError):
            debayer(np.zeros((5,), dtype=np.float32))

    def test_unknown_pattern_defaults_to_rggb(self):
        mosaic = _make_solid_mosaic("RGGB", 32, 32, 0.1, 0.4, 0.9)
        out_unknown = debayer(mosaic, pattern="ZZZZ", method="superpixel")
        out_rggb = debayer(mosaic, pattern="RGGB", method="superpixel")
        assert np.array_equal(out_unknown, out_rggb)

    def test_unknown_method_defaults_to_vng(self):
        mosaic = _make_solid_mosaic("RGGB", 32, 32, 0.1, 0.4, 0.9)
        out_unknown = debayer(mosaic, pattern="RGGB", method="not_a_method")
        out_vng = debayer(mosaic, pattern="RGGB", method="vng")
        assert np.array_equal(out_unknown, out_vng)
