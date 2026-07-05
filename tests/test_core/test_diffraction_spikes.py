"""Tests for synthetic diffraction spike rendering."""

from __future__ import annotations

import numpy as np
import pytest

from astraios.core.device_manager import get_device_manager
from astraios.core.diffraction_spikes import DiffractionSpikeParams, render_spikes
from astraios.core.masks import Mask


def _starfield_mono(h: int = 200, w: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = (rng.rand(h, w).astype(np.float32) * 0.02)
    # A few bright, well-separated stars.
    for cy, cx in ((60, 60), (140, 140), (60, 140)):
        img[cy - 2 : cy + 3, cx - 2 : cx + 3] = 0.9
        img[cy, cx] = 1.0
    return np.clip(img, 0.0, 1.0)


def _starfield_color(h: int = 200, w: int = 200, seed: int = 0) -> np.ndarray:
    mono = _starfield_mono(h, w, seed)
    return np.stack([mono, mono * 0.85, mono * 0.7], axis=0).astype(np.float32)


DEFAULT_PARAMS = DiffractionSpikeParams(detect_sigma_threshold=4.0)


class TestParamsDefaults:
    def test_defaults_construct(self):
        p = DiffractionSpikeParams()
        assert p.quantity == 4
        assert p.enable_halo is False
        assert p.enable_rainbow is False


class TestRunsOnBothShapes:
    def test_mono_finite_and_in_range(self):
        img = _starfield_mono()
        out = render_spikes(img, DEFAULT_PARAMS)
        assert out.shape == img.shape
        assert out.dtype == np.float32
        assert np.isfinite(out).all()
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_color_finite_and_in_range(self):
        img = _starfield_color()
        out = render_spikes(img, DEFAULT_PARAMS)
        assert out.shape == img.shape
        assert out.dtype == np.float32
        assert np.isfinite(out).all()
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestNoOp:
    def test_no_stars_detected_returns_input(self):
        img = np.full((64, 64), 0.1, dtype=np.float32)
        params = DiffractionSpikeParams(detect_sigma_threshold=50.0)
        out = render_spikes(img, params)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_zero_intensity_and_secondary_and_flare_still_finite(self):
        img = _starfield_color()
        params = DiffractionSpikeParams(
            detect_sigma_threshold=4.0,
            intensity=0.0,
            secondary_intensity=0.0,
            soft_flare_intensity=0.0,
        )
        out = render_spikes(img, params)
        assert np.isfinite(out).all()


class TestSpikesAddFlux:
    def test_mono_flux_increases_near_stars(self):
        img = _starfield_mono()
        out = render_spikes(img, DEFAULT_PARAMS)
        assert out.sum() > img.sum()

        # Flux should increase specifically along a spike direction from a star,
        # not just from noise.
        cy, cx = 60, 60
        ring_before = img[cy - 20 : cy - 10, cx - 2 : cx + 2].sum()
        ring_after = out[cy - 20 : cy - 10, cx - 2 : cx + 2].sum()
        assert ring_after >= ring_before

    def test_color_flux_increases_near_stars(self):
        img = _starfield_color()
        out = render_spikes(img, DEFAULT_PARAMS)
        assert out.sum() > img.sum()

    def test_higher_intensity_adds_more_flux(self):
        img = _starfield_color()
        low = render_spikes(img, DiffractionSpikeParams(detect_sigma_threshold=4.0, intensity=0.2))
        high = render_spikes(img, DiffractionSpikeParams(detect_sigma_threshold=4.0, intensity=1.0))
        assert high.sum() >= low.sum()


class TestRotationChangesOutput:
    def test_angle_changes_output(self):
        img = _starfield_color()
        a = render_spikes(img, DiffractionSpikeParams(detect_sigma_threshold=4.0, angle=0.0))
        b = render_spikes(img, DiffractionSpikeParams(detect_sigma_threshold=4.0, angle=30.0))
        assert not np.allclose(a, b)


class TestSecondaryHaloRainbow:
    def test_secondary_spikes_change_output(self):
        img = _starfield_color()
        off = render_spikes(
            img, DiffractionSpikeParams(detect_sigma_threshold=4.0, secondary_intensity=0.0)
        )
        on = render_spikes(
            img, DiffractionSpikeParams(detect_sigma_threshold=4.0, secondary_intensity=0.9)
        )
        assert not np.allclose(off, on)

    def test_halo_changes_output(self):
        img = _starfield_color()
        off = render_spikes(
            img, DiffractionSpikeParams(detect_sigma_threshold=4.0, enable_halo=False)
        )
        on = render_spikes(
            img,
            DiffractionSpikeParams(
                detect_sigma_threshold=4.0, enable_halo=True, halo_intensity=1.0
            ),
        )
        assert not np.allclose(off, on)

    def test_rainbow_changes_output(self):
        img = _starfield_color()
        off = render_spikes(
            img, DiffractionSpikeParams(detect_sigma_threshold=4.0, enable_rainbow=False)
        )
        on = render_spikes(
            img,
            DiffractionSpikeParams(
                detect_sigma_threshold=4.0, enable_rainbow=True, rainbow_spike_intensity=1.0
            ),
        )
        assert not np.allclose(off, on)


class TestStarSelectionFilters:
    def test_star_amount_zero_yields_no_change(self):
        img = _starfield_color()
        params = DiffractionSpikeParams(detect_sigma_threshold=4.0, star_amount=0.0)
        out = render_spikes(img, params)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_max_star_size_filters_large_stars(self):
        img = _starfield_color()
        # Excluding the largest stars (max_star_size near 0) should reduce
        # the amount of added flux relative to including everything.
        restricted = render_spikes(
            img, DiffractionSpikeParams(detect_sigma_threshold=4.0, max_star_size=1.0)
        )
        full = render_spikes(
            img, DiffractionSpikeParams(detect_sigma_threshold=4.0, max_star_size=100.0)
        )
        assert full.sum() >= restricted.sum()


class TestMaskSupport:
    def test_mask_zero_region_unchanged(self):
        img = _starfield_color()
        mask_data = np.ones((200, 200), dtype=np.float32)
        mask_data[:100, :] = 0.0  # protect top half (contains a star at (60,60))
        mask = Mask(data=mask_data)
        out = render_spikes(img, DEFAULT_PARAMS, mask=mask)
        np.testing.assert_allclose(out[:, :100, :], img[:, :100, :], atol=1e-5)

    def test_mask_none_equals_full_mask(self):
        img = _starfield_mono()
        out_none = render_spikes(img, DEFAULT_PARAMS, mask=None)
        full_mask = Mask(data=np.ones_like(img))
        out_full = render_spikes(img, DEFAULT_PARAMS, mask=full_mask)
        np.testing.assert_allclose(out_none, out_full, atol=1e-5)


class TestProgressCallback:
    def test_progress_called(self):
        calls = []

        def progress(frac, msg):
            calls.append((frac, msg))

        img = _starfield_mono()
        render_spikes(img, DEFAULT_PARAMS, progress=progress)
        assert len(calls) >= 2
        assert calls[-1][0] == pytest.approx(1.0)


@pytest.mark.skipif(not get_device_manager().is_gpu, reason="No GPU available")
class TestGPUCPUAgreement:
    def test_gpu_and_cpu_compositing_agree(self, monkeypatch):
        from astraios.core import diffraction_spikes as mod

        img = _starfield_color(h=256, w=256)

        monkeypatch.setattr(mod, "GPU_PIXEL_THRESHOLD", 1)
        out_gpu = render_spikes(img, DEFAULT_PARAMS)

        monkeypatch.setattr(mod, "GPU_PIXEL_THRESHOLD", 10**12)
        out_cpu = render_spikes(img, DEFAULT_PARAMS)

        np.testing.assert_allclose(out_gpu, out_cpu, atol=2e-2)
