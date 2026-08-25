"""Cosmic Clarity must report progress, because progress is also the cancel point.

``apply()`` accepted a ``progress`` callback and never called it. That reads
like a cosmetic problem and is not: ``ProcessingWorker`` injects its
``_emit_progress`` as this callback, and that function raises
``_ProcessingCancelled`` when the user cancels. A tiled inference that never
calls progress therefore cannot be interrupted at all -- Cancel does nothing
until the entire image finishes.

These tests use a trivial stand-in network, so they need no Cosmic Clarity
model, no download and no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import astraios.ai.inference.cosmic_clarity as cc


class _Halve(torch.nn.Module):
    """Stand-in for a real Cosmic Clarity network."""

    def forward(self, x):  # noqa: D102
        return x * 0.5


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(cc, "_load_model", lambda name: _Halve())
    return _Halve()


def _params(**kw):
    return cc.CosmicClarityParams(model="denoise", **kw)


class TestProgressReporting:
    def test_progress_is_called_at_all(self, fake_model):
        calls = []
        img = np.random.rand(3, 40, 40).astype(np.float32)
        cc.apply(img, _params(tile_size=16), progress=lambda f, m: calls.append((f, m)))
        assert calls, "apply() must report progress"

    def test_tiled_run_reports_more_than_once_per_channel(self, fake_model):
        """Per tile, not per channel: a coarse tick leaves Cancel feeling dead
        on a large image."""
        calls = []
        img = np.random.rand(80, 80).astype(np.float32)
        cc.apply(img, _params(tile_size=16), progress=lambda f, m: calls.append((f, m)))
        assert len(calls) > 4

    def test_fractions_are_monotonic_and_bounded(self, fake_model):
        fracs = []
        img = np.random.rand(3, 40, 40).astype(np.float32)
        cc.apply(img, _params(tile_size=16), progress=lambda f, m: fracs.append(f))
        assert min(fracs) >= 0.0 and max(fracs) <= 1.0
        assert all(b >= a for a, b in zip(fracs, fracs[1:], strict=False))

    def test_untiled_path_still_reports(self, fake_model):
        """tile_size=0 runs in one shot; without a tick there it would be the
        uncancellable case all over again."""
        calls = []
        img = np.random.rand(24, 24).astype(np.float32)
        cc.apply(img, _params(tile_size=0), progress=lambda f, m: calls.append((f, m)))
        assert calls

    def test_progress_is_optional(self, fake_model):
        img = np.random.rand(24, 24).astype(np.float32)
        out = cc.apply(img, _params(tile_size=0))
        assert out.shape == img.shape
        assert np.isfinite(out).all()


class TestCancellation:
    def test_a_raising_callback_aborts_the_run(self, fake_model):
        """Exactly how ProcessingWorker cancels: _emit_progress raises."""

        class _Cancelled(Exception):
            pass

        def cancel_after_start(frac, _msg):
            if frac > 0.1:
                raise _Cancelled()

        img = np.random.rand(3, 80, 80).astype(np.float32)
        with pytest.raises(_Cancelled):
            cc.apply(img, _params(tile_size=16), progress=cancel_after_start)
