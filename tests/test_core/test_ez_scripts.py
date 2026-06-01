from __future__ import annotations

import numpy as np
import pytest

from cosmica.core.ez_scripts import list_presets, run_preset


@pytest.fixture
def test_image():
    return np.random.rand(64, 64).astype(np.float32)


@pytest.mark.parametrize("preset_name", list_presets())
def test_each_preset_runs(preset_name: str, test_image):
    result = run_preset(test_image, preset_name)
    assert result is not None
    assert np.all(np.isfinite(result))
    assert result.dtype == np.float32


def test_unknown_preset_raises(test_image):
    with pytest.raises(ValueError, match="Unknown preset"):
        run_preset(test_image, "nonexistent")
