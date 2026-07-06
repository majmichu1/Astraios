"""Tests for dither quality analysis (ported from SASpro dither_analysis.py)."""

import numpy as np
import pytest

from astraios.core.dither_analysis import (
    DitherAnalysisParams,
    DitherOffsetMethod,
    analyze_dither,
)

_CDELT = 0.0002


def _header(crval1: float, crval2: float) -> dict:
    return {
        "NAXIS": 2, "NAXIS1": 1024, "NAXIS2": 1024,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRPIX1": 512.0, "CRPIX2": 512.0,
        "CRVAL1": crval1, "CRVAL2": crval2,
        "CDELT1": -_CDELT, "CDELT2": _CDELT,
        "CUNIT1": "deg", "CUNIT2": "deg",
    }


def _rolled_frames(offsets, size=128, seed=0):
    """Reference random frame + copies circularly rolled by (dy, dx) offsets."""
    rng = np.random.default_rng(seed)
    ref = rng.random((size, size)).astype(np.float32)
    frames = [ref]
    for dy, dx in offsets:
        frames.append(np.roll(ref, shift=(dy, dx), axis=(0, 1)))
    return frames


class TestOffsetRecovery:
    def test_recovers_injected_integer_offsets(self):
        injected = [(5, 3), (-4, 7), (0, -6), (8, 8)]
        frames = _rolled_frames(injected)
        result = analyze_dither(frames, DitherAnalysisParams(upsample_factor=20))

        assert result.n_frames == 5
        assert result.dx[0] == pytest.approx(0.0, abs=0.1)
        assert result.dy[0] == pytest.approx(0.0, abs=0.1)
        for i, (dy, dx) in enumerate(injected, start=1):
            assert result.dy[i] == pytest.approx(dy, abs=0.2)
            assert result.dx[i] == pytest.approx(dx, abs=0.2)

    def test_recovers_offsets_from_headers(self):
        headers = [
            _header(180.0, 0.0),
            _header(180.0 - 5 * _CDELT, 0.0),  # dx=+5, dy=0
            _header(180.0, 5 * _CDELT),  # dx=0, dy=+5
            _header(180.0 + 3 * _CDELT, 2 * _CDELT),  # dx=-3, dy=+2
        ]
        params = DitherAnalysisParams(offset_method=DitherOffsetMethod.HEADER_WCS)
        result = analyze_dither(headers, params, headers=headers)

        assert result.dx[0] == pytest.approx(0.0, abs=1e-6)
        assert result.dy[0] == pytest.approx(0.0, abs=1e-6)
        assert result.dx[1] == pytest.approx(5.0, abs=1e-6)
        assert result.dy[1] == pytest.approx(0.0, abs=1e-6)
        assert result.dx[2] == pytest.approx(0.0, abs=1e-6)
        assert result.dy[2] == pytest.approx(5.0, abs=1e-6)
        assert result.dx[3] == pytest.approx(-3.0, abs=1e-6)
        assert result.dy[3] == pytest.approx(2.0, abs=1e-6)

    def test_header_wcs_without_headers_or_paths_raises(self):
        headers = [_header(180.0, 0.0), _header(180.0, 0.001)]
        params = DitherAnalysisParams(offset_method=DitherOffsetMethod.HEADER_WCS)
        with pytest.raises(ValueError):
            analyze_dither(headers, params)  # not file paths, no headers= given


class TestSpreadStatistics:
    def test_zero_dither_reports_near_zero_spread(self):
        rng = np.random.default_rng(1)
        ref = rng.random((64, 64)).astype(np.float32)
        frames = [ref.copy() for _ in range(6)]
        result = analyze_dither(frames)

        assert result.std_dx == pytest.approx(0.0, abs=0.05)
        assert result.std_dy == pytest.approx(0.0, abs=0.05)
        assert result.mean_radius == pytest.approx(0.0, abs=0.05)
        assert result.max_radius == pytest.approx(0.0, abs=0.05)
        assert result.coverage_px == pytest.approx(0.0, abs=1e-6)

    def test_good_dither_reports_larger_spread_than_zero_dither(self):
        rng = np.random.default_rng(2)
        ref = rng.random((128, 128)).astype(np.float32)
        offsets = [(6, -8), (-10, 5), (12, 12), (-6, -14), (9, -3)]
        good_frames = [ref] + [np.roll(ref, shift=o, axis=(0, 1)) for o in offsets]
        zero_frames = [ref.copy() for _ in range(len(good_frames))]

        good = analyze_dither(good_frames, DitherAnalysisParams(upsample_factor=20))
        zero = analyze_dither(zero_frames)

        assert good.std_dx > zero.std_dx
        assert good.std_dy > zero.std_dy
        assert good.mean_radius > zero.mean_radius
        assert good.coverage_px > zero.coverage_px

    def test_nearest_neighbor_stat_is_sane(self):
        rng = np.random.default_rng(3)
        ref = rng.random((128, 128)).astype(np.float32)
        offsets = [(6, -8), (-10, 5), (12, 12), (-6, -14), (9, -3)]
        frames = [ref] + [np.roll(ref, shift=o, axis=(0, 1)) for o in offsets]
        result = analyze_dither(frames, DitherAnalysisParams(upsample_factor=20))

        assert result.nearest_neighbor_min_px >= 0.0
        assert result.nearest_neighbor_mean_px >= result.nearest_neighbor_min_px
        assert result.nearest_neighbor_min_px <= result.max_radius + 1.0


class TestClusterAndWalkingFlags:
    def test_all_identical_frames_flagged_clustered(self):
        rng = np.random.default_rng(4)
        ref = rng.random((64, 64)).astype(np.float32)
        frames = [ref.copy() for _ in range(6)]
        result = analyze_dither(frames)
        assert result.is_clustered is True
        assert result.n_clusters == 1

    def test_linear_drift_flagged_as_walking(self):
        rng = np.random.default_rng(5)
        ref = rng.random((160, 160)).astype(np.float32)
        # Strictly increasing offset along one direction -> linear drift.
        frames = [ref] + [
            np.roll(ref, shift=(2 * i, 3 * i), axis=(0, 1)) for i in range(1, 8)
        ]
        result = analyze_dither(frames, DitherAnalysisParams(upsample_factor=20))
        assert result.is_walking is True
        assert result.linearity_ratio > DitherAnalysisParams().linearity_threshold


class TestValidation:
    def test_needs_at_least_two_frames(self):
        with pytest.raises(ValueError):
            analyze_dither([np.zeros((16, 16), dtype=np.float32)])

    def test_reference_index_out_of_range_raises(self):
        frames = _rolled_frames([(1, 1)])
        with pytest.raises(ValueError):
            analyze_dither(frames, DitherAnalysisParams(reference_index=5))
