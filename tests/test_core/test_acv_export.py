"""Tests for the Photoshop .acv curves exporter.

See astraios/core/acv_export.py's module docstring: this is NOT ported from
SASpro (its acv_exporter.py is an unrelated image-export-by-catalog feature
with no curves/struct-packing code) — it's an independent implementation of
Adobe's publicly documented Curves file format. These tests verify the
binary layout matches that spec byte-for-byte.
"""

from __future__ import annotations

import struct

import pytest

from astraios.core.acv_export import export_acv


def _read_acv(path):
    """Minimal reference reader mirroring the documented .acv layout,
    independent of export_acv's own implementation."""
    data = path.read_bytes()
    offset = 0
    version, count = struct.unpack_from(">HH", data, offset)
    offset += 4
    curves = []
    for _ in range(count):
        (n_points,) = struct.unpack_from(">H", data, offset)
        offset += 2
        points = []
        for _ in range(n_points):
            out_val, in_val = struct.unpack_from(">HH", data, offset)
            offset += 4
            points.append((in_val, out_val))  # (x, y) in 0-255 space
        curves.append(points)
    assert offset == len(data)  # no trailing/missing bytes
    return version, curves


class TestBinaryStructure:
    def test_version_and_count(self, tmp_path):
        curves = [("Master", [(0.0, 0.0), (1.0, 1.0)])]
        out = export_acv(curves, tmp_path / "linear.acv")
        version, parsed = _read_acv(out)
        assert version == 1
        assert len(parsed) == 1

    def test_point_coords_scaled_to_0_255(self, tmp_path):
        curves = [("Master", [(0.0, 0.0), (0.5, 0.25), (1.0, 1.0)])]
        out = export_acv(curves, tmp_path / "curve.acv")
        _, parsed = _read_acv(out)
        pts = parsed[0]
        assert pts == [(0, 0), (128, 64), (255, 255)]

    def test_multi_channel_order_preserved(self, tmp_path):
        curves = [
            ("Composite", [(0.0, 0.0), (1.0, 1.0)]),
            ("Red", [(0.0, 0.0), (1.0, 0.8)]),
            ("Green", [(0.0, 0.1), (1.0, 1.0)]),
            ("Blue", [(0.0, 0.0), (0.5, 0.5), (1.0, 0.9)]),
        ]
        out = export_acv(curves, tmp_path / "rgb.acv")
        version, parsed = _read_acv(out)
        assert version == 1
        assert len(parsed) == 4
        assert parsed[0] == [(0, 0), (255, 255)]
        assert parsed[1] == [(0, 0), (255, 204)]
        assert parsed[2] == [(0, 26), (255, 255)]
        assert parsed[3] == [(0, 0), (128, 128), (255, 230)]

    def test_points_sorted_by_x_before_writing(self, tmp_path):
        # Deliberately out of order input.
        curves = [("Master", [(1.0, 1.0), (0.0, 0.0), (0.5, 0.9)])]
        out = export_acv(curves, tmp_path / "unsorted.acv")
        _, parsed = _read_acv(out)
        xs = [p[0] for p in parsed[0]]
        assert xs == sorted(xs)


class TestRoundTrip:
    def test_linear_curve_round_trips_identity(self, tmp_path):
        curves = [("Master", [(0.0, 0.0), (1.0, 1.0)])]
        out = export_acv(curves, tmp_path / "identity.acv")
        _, parsed = _read_acv(out)
        assert parsed == [[(0, 0), (255, 255)]]

    def test_file_bytes_exact_length(self, tmp_path):
        # 4 (header) + [2 (count) + n*4 (points)] per curve
        curves = [
            ("Master", [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]),  # 3 points
            ("Red", [(0.0, 0.0), (1.0, 1.0)]),  # 2 points
        ]
        out = export_acv(curves, tmp_path / "sized.acv")
        expected_len = 4 + (2 + 3 * 4) + (2 + 2 * 4)
        assert len(out.read_bytes()) == expected_len


class TestValidation:
    def test_empty_curve_list_raises(self, tmp_path):
        with pytest.raises(ValueError):
            export_acv([], tmp_path / "empty.acv")

    def test_single_point_curve_raises(self, tmp_path):
        with pytest.raises(ValueError):
            export_acv([("Master", [(0.5, 0.5)])], tmp_path / "bad.acv")

    def test_too_many_points_raises(self, tmp_path):
        pts = [(i / 19.0, i / 19.0) for i in range(20)]  # 20 > max of 19
        with pytest.raises(ValueError):
            export_acv([("Master", pts)], tmp_path / "bad.acv")

    def test_max_points_allowed(self, tmp_path):
        pts = [(i / 18.0, i / 18.0) for i in range(19)]  # exactly 19, allowed
        out = export_acv([("Master", pts)], tmp_path / "max.acv")
        _, parsed = _read_acv(out)
        assert len(parsed[0]) == 19

    def test_creates_parent_directories(self, tmp_path):
        curves = [("Master", [(0.0, 0.0), (1.0, 1.0)])]
        out = export_acv(curves, tmp_path / "nested" / "dir" / "out.acv")
        assert out.exists()

    def test_values_clamped_to_0_1(self, tmp_path):
        curves = [("Master", [(-0.5, -0.5), (1.5, 1.5)])]
        out = export_acv(curves, tmp_path / "clamped.acv")
        _, parsed = _read_acv(out)
        assert parsed[0] == [(0, 0), (255, 255)]
