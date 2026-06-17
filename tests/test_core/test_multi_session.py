"""Tests for multi-session stacking."""

from __future__ import annotations

import numpy as np
import pytest

from astraios.core.image_io import ImageData
from astraios.core.multi_session import (
    MultiSessionParams,
    SessionGroup,
    auto_group_sessions,
    stack_multi_session,
)
from astraios.core.stacking import StackingParams, RejectionMethod


def _make_session(n_frames=3, shape=(3, 32, 32), offset=0.0, name="S") -> SessionGroup:
    rng = np.random.default_rng(seed=42)
    frames = [
        ImageData(
            data=np.clip(rng.normal(0.3 + offset, 0.05, shape).astype(np.float32), 0, 1),
            header={"EXPTIME": 60.0},
        )
        for _ in range(n_frames)
    ]
    return SessionGroup(frames=frames, name=name, integration_time=n_frames * 60.0)


class TestMultiSession:
    def test_two_sessions_same_size(self):
        s1 = _make_session(name="Night1")
        s2 = _make_session(offset=0.05, name="Night2")
        params = MultiSessionParams(
            per_session_params=StackingParams(rejection=RejectionMethod.NONE),
            align_sub_stacks=False,
        )
        result = stack_multi_session([s1, s2], params)
        assert result.n_sessions == 2
        assert result.image.data.shape == (3, 32, 32)
        assert 0.0 <= result.image.data.min()
        assert result.image.data.max() <= 1.0

    def test_weights_sum_to_one(self):
        s1 = _make_session(name="A")
        s2 = _make_session(name="B")
        params = MultiSessionParams(
            per_session_params=StackingParams(rejection=RejectionMethod.NONE),
            align_sub_stacks=False,
        )
        result = stack_multi_session([s1, s2], params)
        assert abs(sum(result.weights) - 1.0) < 1e-5

    def test_single_session_degenerate(self):
        s1 = _make_session(n_frames=3, name="Only")
        params = MultiSessionParams(
            per_session_params=StackingParams(rejection=RejectionMethod.NONE),
        )
        result = stack_multi_session([s1], params)
        assert result.n_sessions == 1
        assert result.image.data.shape == (3, 32, 32)

    def test_equal_weight_mode(self):
        s1 = _make_session(name="A")
        s2 = _make_session(name="B")
        params = MultiSessionParams(
            per_session_params=StackingParams(rejection=RejectionMethod.NONE),
            weight_mode="equal",
            align_sub_stacks=False,
        )
        result = stack_multi_session([s1, s2], params)
        assert abs(result.weights[0] - 0.5) < 1e-5
        assert abs(result.weights[1] - 0.5) < 1e-5

    def test_time_weight_mode(self):
        s1 = _make_session(n_frames=2, name="Short")  # 120s
        s2 = _make_session(n_frames=6, name="Long")   # 360s
        params = MultiSessionParams(
            per_session_params=StackingParams(rejection=RejectionMethod.NONE),
            weight_mode="time",
            align_sub_stacks=False,
        )
        result = stack_multi_session([s1, s2], params)
        # Long session should have higher weight
        assert result.weights[1] > result.weights[0]

    def test_mismatched_sizes_padded(self):
        """Smaller sub-stacks should be padded to the larger size."""
        rng = np.random.default_rng(1)
        big = SessionGroup(
            frames=[ImageData(data=rng.random((3, 64, 64)).astype(np.float32), header={})],
            name="Big",
        )
        small = SessionGroup(
            frames=[ImageData(data=rng.random((3, 32, 32)).astype(np.float32), header={})],
            name="Small",
        )
        params = MultiSessionParams(
            per_session_params=StackingParams(rejection=RejectionMethod.NONE),
            align_sub_stacks=False,
        )
        result = stack_multi_session([big, small], params)
        assert result.image.data.shape == (3, 64, 64)

    def test_progress_callback_called(self):
        s1 = _make_session(name="A")
        s2 = _make_session(name="B")
        calls = []
        params = MultiSessionParams(
            per_session_params=StackingParams(rejection=RejectionMethod.NONE),
            align_sub_stacks=False,
        )
        stack_multi_session([s1, s2], params, progress=lambda f, m: calls.append((f, m)))
        assert len(calls) > 0
        fractions = [c[0] for c in calls]
        assert fractions[-1] == pytest.approx(1.0)

    def test_empty_sessions_raises(self):
        with pytest.raises((ValueError, Exception)):
            stack_multi_session([], MultiSessionParams())


class TestAutoGroup:
    def test_groups_by_instrument_and_filter(self):
        h_cam1 = {"INSTRUME": "ASI2600MC", "FILTER": "L", "DATE-OBS": "2024-01-01"}
        h_cam2 = {"INSTRUME": "ASI294MC",  "FILTER": "Ha", "DATE-OBS": "2024-01-02"}
        frames = [
            ImageData(data=np.zeros((3, 8, 8), dtype=np.float32), header=h_cam1),
            ImageData(data=np.zeros((3, 8, 8), dtype=np.float32), header=h_cam1),
            ImageData(data=np.zeros((3, 8, 8), dtype=np.float32), header=h_cam2),
        ]
        groups = auto_group_sessions(frames)
        assert len(groups) == 2
        sizes = sorted(len(g.frames) for g in groups)
        assert sizes == [1, 2]

    def test_same_instrument_different_dates(self):
        frames = [
            ImageData(data=np.zeros((3, 8, 8), dtype=np.float32),
                      header={"INSTRUME": "Cam", "DATE-OBS": "2024-01-01", "EXPTIME": 300.0}),
            ImageData(data=np.zeros((3, 8, 8), dtype=np.float32),
                      header={"INSTRUME": "Cam", "DATE-OBS": "2024-01-02", "EXPTIME": 300.0}),
        ]
        groups = auto_group_sessions(frames)
        assert len(groups) == 2

    def test_integration_time_summed(self):
        frames = [
            ImageData(data=np.zeros((3, 8, 8), dtype=np.float32),
                      header={"INSTRUME": "Cam", "DATE-OBS": "2024-01-01", "EXPTIME": 120.0}),
            ImageData(data=np.zeros((3, 8, 8), dtype=np.float32),
                      header={"INSTRUME": "Cam", "DATE-OBS": "2024-01-01", "EXPTIME": 180.0}),
        ]
        groups = auto_group_sessions(frames)
        assert len(groups) == 1
        assert groups[0].integration_time == pytest.approx(300.0)
