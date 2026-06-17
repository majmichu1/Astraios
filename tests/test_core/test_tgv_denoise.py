"""Tests for TGV² denoising.

Regression: TGV was completely broken — it crashed on 2-D input (replicate
padding) and its divergence operators were not the adjoints of the gradients,
so the primal-dual iteration never converged. The default UI denoise method
routed here, so 'Apply Denoise' was broken by default.
"""

import numpy as np
import torch

from astraios.core.tgv_denoise import (
    TGVParams,
    _div,
    _div_sym,
    _grad,
    _sym_grad,
    tgv_denoise,
)


class TestAdjointOperators:
    def test_grad_div_are_adjoint(self):
        torch.manual_seed(0)
        u = torch.rand(40, 40, dtype=torch.float64)
        p = torch.rand(40, 40, 2, dtype=torch.float64)
        lhs = float((_grad(u) * p).sum())
        rhs = float((u * (-_div(p))).sum())
        assert abs(lhs - rhs) < 1e-9

    def test_symgrad_divsym_are_adjoint(self):
        torch.manual_seed(1)
        p = torch.rand(40, 40, 2, dtype=torch.float64)
        e = torch.rand(40, 40, 3, dtype=torch.float64)
        lhs = float((_sym_grad(p) * e).sum())
        rhs = float((p * (-_div_sym(e))).sum())
        assert abs(lhs - rhs) < 1e-9


class TestTGVDenoise:
    def test_runs_on_2d_and_3d(self):
        for shape in [(48, 48), (3, 48, 48)]:
            img = np.clip(np.random.default_rng(0).random(shape) * 0.4 + 0.05, 0, 1).astype(np.float32)
            out = tgv_denoise(img, TGVParams(n_iter=20))
            assert out.shape == shape
            assert np.all(np.isfinite(out))
            assert out.min() >= 0.0 and out.max() <= 1.0

    def test_reduces_noise_on_flat_region(self):
        clean = np.full((96, 96), 0.3, np.float32)
        noisy = np.clip(clean + np.random.default_rng(0).normal(0, 0.06, (96, 96)), 0, 1).astype(np.float32)
        out = tgv_denoise(noisy, TGVParams(strength=0.4, n_iter=80))
        assert np.std(out) < 0.5 * np.std(noisy)

    def test_preserves_linear_ramp(self):
        # TGV's defining property: a smooth gradient is kept (no staircasing).
        ramp = np.tile(np.linspace(0.1, 0.6, 96, dtype=np.float32), (96, 1))
        noisy = np.clip(ramp + np.random.default_rng(1).normal(0, 0.05, (96, 96)), 0, 1).astype(np.float32)
        out = tgv_denoise(noisy, TGVParams(strength=0.5, n_iter=80))
        assert np.std(out - ramp) < np.std(noisy - ramp)
