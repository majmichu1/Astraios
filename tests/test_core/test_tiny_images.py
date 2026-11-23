from __future__ import annotations

import numpy as np
import pytest

from astraios.core.chroma_denoise import chroma_denoise
from astraios.core.deconvolution import richardson_lucy, wiener_deconvolve
from astraios.core.halo_reduction import reduce_halos
from astraios.core.tgv_denoise import tgv_denoise

SHAPES = [(8, 8), (3, 8, 8), (1, 64), (3, 1, 64)]


def _img(shape: tuple[int, ...]) -> np.ndarray:
    return np.random.default_rng(0).uniform(0.0, 1.0, shape).astype(np.float32)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("tool", [richardson_lucy, wiener_deconvolve, reduce_halos, tgv_denoise])
def test_tiny_image_does_not_crash(tool, shape) -> None:
    """Images smaller than a tool's kernel come back unchanged in shape, not a crash."""
    out = tool(_img(shape))
    out = getattr(out, "data", out)
    assert out.shape == shape
    assert np.isfinite(out).all()


@pytest.mark.parametrize("shape", [(3, 8, 8), (3, 1, 64)])
def test_tiny_colour_image_chroma_denoise(shape) -> None:
    """Chroma denoise on a tiny colour image returns it instead of crashing."""
    out = chroma_denoise(_img(shape), strength=1.0)
    assert out.shape == shape
    assert np.isfinite(out).all()
