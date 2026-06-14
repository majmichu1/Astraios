"""Regression test for the StarNet -> Image Blend (starless+stars) workflow.

StarNet computes a stars-only layer (original - starless). The main window used
to discard it; it must now keep it so the user can screen the stars back onto
the stretched starless image via the Blend dialog.
"""

from __future__ import annotations

import numpy as np
import pytest

from cosmica.ai.inference.starnet import StarNetResult
from cosmica.core.image_io import ImageData


@pytest.fixture
def window(qtbot):
    from cosmica.ui.main_window import MainWindow

    return MainWindow()


def test_starnet_keeps_star_layer_and_blend_uses_it(window):
    starless = np.full((3, 32, 32), 0.2, np.float32)
    stars = np.zeros((3, 32, 32), np.float32)
    stars[:, 16, 16] = 0.8
    window._current_image = ImageData(data=starless.copy())

    # StarNet completion must retain the stars-only layer.
    window._on_starnet_done(
        StarNetResult(success=True, starless=starless, stars_only=stars, message="")
    )
    assert window._extracted_stars is not None

    # Blend dialog should accept the extracted stars and screen them back.
    from cosmica.ui.dialogs.blend_dialog import BlendDialog

    dlg = BlendDialog(base_image=window._current_image.data, parent=window)
    dlg.set_extracted_stars(window._extracted_stars)
    dlg._use_extracted_stars()
    assert dlg._layer is not None

    captured = {}
    dlg.result_ready.connect(lambda r: captured.__setitem__("r", r))
    dlg._mode_combo.setCurrentIndex(0)  # Screen
    dlg._apply()

    result = captured["r"]
    # screen(0.2, 0.8) = 1 - (1-0.2)(1-0.8) = 0.84 at the star pixel
    assert abs(float(result[0, 16, 16]) - 0.84) < 1e-3
    # background unchanged by screening with black
    assert abs(float(result[0, 0, 0]) - 0.2) < 1e-3
