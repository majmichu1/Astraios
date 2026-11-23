from __future__ import annotations

import numpy as np
import torch

from astraios.core.denoise import (
    DenoiseParams,
    _denoise_wavelet_gpu,
    _denoise_wavelet_gpu_cpu,
    _starlet_noise_factors,
    denoise,
)


def _field() -> tuple[np.ndarray, np.ndarray]:
    """(clean, noisy) 256x256: sky, a soft nebula, 30 small stars, noise sigma 0.02."""
    rng = np.random.default_rng(3)
    h = w = 256
    yy, xx = np.mgrid[0:h, 0:w]
    clean = 0.1 + 0.3 * np.exp(-((xx - 128) ** 2 + (yy - 128) ** 2) / (2 * 30.0**2))
    clean = clean.astype(np.float32)
    for x, y in rng.uniform(20, 236, (30, 2)):
        clean += (0.5 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.3**2))).astype(np.float32)
    clean = np.clip(clean, 0, 1)
    noisy = np.clip(clean + rng.normal(0, 0.02, clean.shape).astype(np.float32), 0, 1)
    return clean, noisy


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_starlet_noise_factors_match_the_b3_spline() -> None:
    """Per-scale noise factors are the known starlet values."""
    f = _starlet_noise_factors(4)
    assert np.allclose(f, (0.889, 0.200, 0.086, 0.041), atol=0.01)


def test_starlet_path_removes_noise_without_eating_stars() -> None:
    """The starlet (GPU) path must get much closer to the clean image than the input."""
    clean, noisy = _field()
    out = _denoise_wavelet_gpu(noisy, DenoiseParams(strength=0.5), torch.device("cpu"))
    # Regression: one noise level for every scale left this at ~0.019 (no better than input).
    assert _rms(out, clean) < 0.010
    assert _rms(noisy, clean) > 0.019


def test_pywt_path_thresholds_the_finest_level_hardest() -> None:
    """The CPU pywt path beats its old reversed level weighting."""
    clean, noisy = _field()
    out = _denoise_wavelet_gpu_cpu(noisy, DenoiseParams(strength=0.5))
    # Regression: reversed level order gave ~0.0127.
    assert _rms(out, clean) < 0.0105


def test_chrominance_only_is_honoured() -> None:
    """chrominance_only must change the result on every device."""
    _, noisy = _field()
    rgb = np.stack([noisy, noisy * 0.9, noisy * 0.8])
    on = denoise(rgb, DenoiseParams(chrominance_only=True))
    off = denoise(rgb, DenoiseParams(chrominance_only=False))
    assert not np.allclose(on, off)
