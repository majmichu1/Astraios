"""Tests for live stacking (astraios/core/live_stack.py)."""

import threading

import numpy as np

from astraios.core.live_stack import LiveStacker


class TestAddFrameDeadlockRegression:
    """The first add_frame() on a fresh stacker used to self-deadlock:
    add_frame held the non-reentrant lock and called the public
    set_reference(), which re-acquired it. Since the Live Stack dialog never
    seeds the reference itself, live stacking hung forever on frame one.
    Fixed by a lock-free internal _set_reference_locked(); the first frame's
    signal is also accumulated now instead of being silently dropped.
    """

    def test_first_add_frame_completes_and_counts(self):
        ls = LiveStacker()
        frame = np.full((10, 10), 0.25, dtype=np.float32)
        t = threading.Thread(target=ls.add_frame, args=(frame,), daemon=True)
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive(), "first add_frame() deadlocked"
        assert ls.n_frames == 1
        assert np.allclose(ls.get_result(), 0.25)

    def test_first_frame_signal_included(self):
        ls = LiveStacker()
        a = np.full((8, 8), 0.2, dtype=np.float32)
        b = np.full((8, 8), 0.4, dtype=np.float32)
        ls.add_frame(a)
        ls.add_frame(b)
        assert ls.n_frames == 2
        assert np.allclose(ls.get_result(), 0.3), "mean of both frames expected"


class TestSetReference:
    def test_initializes_empty_stack(self):
        ls = LiveStacker()
        frame = np.random.RandomState(0).rand(40, 50).astype(np.float32)
        ls.set_reference(frame)
        assert ls.n_frames == 0
        assert ls.stack_sum.shape == frame.shape
        assert np.array_equal(ls.stack_sum, np.zeros_like(frame))
        assert ls.stack_count.shape == frame.shape[-2:]


class TestIncrementalMean:
    """Once a reference is set (working around the deadlock above),
    subsequent add_frame() calls take the non-recursive locking path safely.
    """

    def test_mean_of_k_identical_frames_equals_the_frame_mono(self):
        ls = LiveStacker(alignment_mode="none")
        frame = np.random.RandomState(1).rand(50, 60).astype(np.float32) * 0.5
        ls.set_reference(frame)
        for _ in range(6):
            ls.add_frame(frame.copy())
        assert ls.n_frames == 6
        result = ls.get_result()
        assert np.allclose(result, frame, atol=1e-6)

    def test_mean_of_k_identical_frames_with_fft_alignment(self):
        # Identical frames measure zero shift, so fft alignment is a no-op here.
        ls = LiveStacker(alignment_mode="fft")
        frame = np.random.RandomState(2).rand(48, 48).astype(np.float32) * 0.5
        ls.set_reference(frame)
        for _ in range(3):
            ls.add_frame(frame.copy())
        result = ls.get_result()
        assert np.allclose(result, frame, atol=1e-5)

    def test_color_frames_stack_correctly(self):
        ls = LiveStacker(alignment_mode="none")
        color = np.random.RandomState(3).rand(3, 30, 30).astype(np.float32) * 0.4
        ls.set_reference(color)
        for _ in range(4):
            ls.add_frame(color.copy())
        result = ls.get_result()
        assert result.shape == color.shape
        assert np.allclose(result, color, atol=1e-6)

    def test_running_mean_of_varying_frames(self):
        ls = LiveStacker(alignment_mode="none")
        values = [0.1, 0.2, 0.3]
        frames = [np.full((20, 20), v, dtype=np.float32) for v in values]
        ls.set_reference(frames[0])
        for f in frames:
            ls.add_frame(f)
        result = ls.get_result()
        assert np.allclose(result, np.mean(values), atol=1e-6)


class TestPreviewAndReset:
    def test_preview_before_any_reference_is_zero_placeholder(self):
        ls = LiveStacker()
        preview = ls.get_live_preview()
        assert preview.shape == (100, 100)
        assert np.array_equal(preview, np.zeros((100, 100), dtype=np.float32))
        assert ls.get_result() is None

    def test_preview_is_stretched_into_unit_range(self):
        ls = LiveStacker(alignment_mode="none")
        frame = np.random.RandomState(4).rand(25, 25).astype(np.float32) * 0.5
        ls.set_reference(frame)
        ls.add_frame(frame.copy())
        preview = ls.get_live_preview()
        assert preview.shape == frame.shape
        assert preview.min() >= 0.0
        assert preview.max() <= 1.0

    def test_reset_clears_stack_state(self):
        ls = LiveStacker(alignment_mode="none")
        frame = np.random.RandomState(5).rand(20, 20).astype(np.float32)
        ls.set_reference(frame)
        ls.add_frame(frame.copy())
        ls.reset()
        assert ls.reference is None
        assert ls.stack_sum is None
        assert ls.n_frames == 0
        assert ls.get_result() is None
        assert ls.get_live_preview().shape == (100, 100)
