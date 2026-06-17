"""Regression tests for channel-layout bugs in colour-only tool handlers.

Astraios stores colour images channel-first ``(C, H, W)`` but
``channel_match.align_channels`` and ``lens_distortion.correct_distortion``
expect channel-last ``(H, W, C)``. The main-window handlers used to pass the
internal array straight through, which made "Align RGB Channels" crash with a
ValueError on every colour image and made "Lens Distortion" silently a no-op
with a transposed camera matrix. These tests pin the fixed behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from astraios.core.image_io import ImageData


def _color_chw(h=96, w=128):
    rng = np.random.default_rng(0)
    base = rng.random((h, w)).astype(np.float32)
    # Deliberate per-channel shifts so channel alignment has work to do.
    return np.stack([base, np.roll(base, 1, axis=0), np.roll(base, 2, axis=1)]).astype(np.float32)


@pytest.fixture
def window(qtbot):
    # qtbot ensures a QApplication exists; MainWindow is a controller, not a QWidget.
    from astraios.ui.main_window import MainWindow

    return MainWindow()


class TestChannelLayoutHandlers:
    def test_channel_match_runs_on_color_image(self, window):
        window._current_image = ImageData(data=_color_chw())
        window._show_channel_match_dialog()
        out = window._current_image.data
        assert out.shape == (3, 96, 128)
        assert out.dtype == np.float32

    def test_lens_distortion_preserves_color_shape(self, window):
        window._current_image = ImageData(data=_color_chw())
        window._show_lens_distortion_dialog()
        out = window._current_image.data
        assert out.shape == (3, 96, 128)
        assert out.dtype == np.float32

    def test_channel_match_skips_mono_without_crash(self, window):
        mono = np.random.default_rng(1).random((96, 128)).astype(np.float32)
        window._current_image = ImageData(data=mono)
        # Should warn and leave the image untouched rather than raise.
        window._show_channel_match_dialog()
        assert window._current_image.data.shape == (96, 128)
