from __future__ import annotations

import numpy as np
import pytest

from astraios.core.ez_scripts import list_presets, run_preset


@pytest.fixture
def test_image():
    return np.random.rand(64, 64).astype(np.float32)


@pytest.mark.parametrize("preset_name", list_presets())
def test_each_preset_runs(preset_name: str, test_image):
    result = run_preset(test_image, preset_name)
    assert result is not None
    assert np.all(np.isfinite(result))
    assert result.dtype == np.float32
    # Regression: the "background" step once subtracted the whole
    # (corrected, model) return tuple, silently broadcasting mono (H,W)
    # input into a garbage 2-channel array.
    assert result.shape == test_image.shape


def test_background_step_preserves_color_shape():
    img = np.random.rand(3, 48, 48).astype(np.float32) * 0.5 + 0.1
    result = run_preset(img, "OSC Quick Processing")
    assert result.shape == img.shape


def test_unknown_preset_raises(test_image):
    with pytest.raises(ValueError, match="Unknown preset"):
        run_preset(test_image, "nonexistent")
