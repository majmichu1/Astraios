"""Tests for the SNR (signal-to-noise measurement) tool.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro
"""

from __future__ import annotations

import numpy as np
import pytest

from astraios.core.snr import SNRParams, measure_snr

SIZE = 200


def _noisy_field(size=SIZE, mean=0.10, std=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, size=(size, size)).astype(np.float32)


def _add_flat_signal(img, bbox, level):
    x, y, w, h = bbox
    img[y:y + h, x:x + w] += level


class TestMeasureSNRBasic:
    def test_snr_matches_signal_over_noise_std(self):
        """Known signal + known-std gaussian noise -> measured SNR ~= signal/noise_std."""
        noise_std = 0.01
        signal_level = 0.05
        img = _noisy_field(std=noise_std, seed=1)
        signal_bbox = (80, 80, 40, 40)
        background_bbox = (0, 0, SIZE, 30)  # away from the signal box
        _add_flat_signal(img, signal_bbox, signal_level)

        params = SNRParams(background_bbox=background_bbox, signal_bbox=signal_bbox)
        result = measure_snr(img, params)

        expected_snr = signal_level / noise_std
        assert result.overall.snr == pytest.approx(expected_snr, rel=0.25)
        assert result.background_auto is False
        assert result.signal_auto is False

    def test_background_bbox_path_uses_plain_stats(self):
        """Explicit background bbox -> plain mean/median/std (ddof=1), not sigma-clipped."""
        img = _noisy_field(std=0.02, seed=2)
        bg_bbox = (10, 10, 60, 60)
        params = SNRParams(background_bbox=bg_bbox)
        result = measure_snr(img, params)

        x, y, w, h = bg_bbox
        sample = img[y:y + h, x:x + w].ravel()
        expected_std = float(np.std(sample, ddof=1))
        expected_median = float(np.median(sample))

        assert result.overall.background_std == pytest.approx(expected_std, rel=1e-6)
        assert result.overall.background_median == pytest.approx(expected_median, rel=1e-6)
        assert result.background_auto is False

    def test_no_bbox_uses_sigma_clipped_auto_estimate(self):
        """No background bbox -> robust sigma-clipped MAD-based noise estimate."""
        img = _noisy_field(std=0.01, seed=3)
        # Contaminate with a few bright outlier pixels a plain std would be sensitive to.
        img[5, 5] = 5.0
        img[6, 6] = 6.0

        result = measure_snr(img)  # default params: no bbox at all

        assert result.background_auto is True
        assert result.signal_auto is True
        assert np.isfinite(result.overall.snr)
        assert np.isfinite(result.overall.snr_db)
        # Robust estimate should stay close to the true noise floor despite outliers.
        assert result.overall.background_std < 0.1

    def test_per_channel_color_image(self):
        rng = np.random.default_rng(4)
        img = rng.normal(0.1, 0.01, size=(3, SIZE, SIZE)).astype(np.float32)
        signal_bbox = (80, 80, 40, 40)
        background_bbox = (0, 0, SIZE, 30)
        levels = [0.02, 0.05, 0.08]
        for c, level in enumerate(levels):
            _add_flat_signal(img[c], signal_bbox, level)

        params = SNRParams(
            background_bbox=background_bbox, signal_bbox=signal_bbox, per_channel=True,
        )
        result = measure_snr(img, params)

        assert len(result.channels) == 3
        assert [ch.name for ch in result.channels] == ["R", "G", "B"]
        # Brighter channel signal -> higher net_signal / snr, monotonic with levels.
        snrs = [ch.snr for ch in result.channels]
        assert snrs[0] < snrs[1] < snrs[2]

    def test_per_channel_disabled_still_reports_overall(self):
        img = _noisy_field(seed=5)
        params = SNRParams(per_channel=False)
        result = measure_snr(img, params)
        assert result.channels == []
        assert result.overall is not None

    def test_mono_image_channel_name(self):
        img = _noisy_field(seed=6)
        result = measure_snr(img, SNRParams(per_channel=True))
        assert len(result.channels) == 1
        assert result.channels[0].name == "Mono"
        assert result.overall.name == "Overall"

    def test_progress_callback_invoked(self):
        img = _noisy_field(seed=7)
        calls = []
        measure_snr(img, progress=lambda frac, msg: calls.append((frac, msg)))
        assert len(calls) >= 2
        assert calls[-1][0] == 1.0

    def test_deterministic(self):
        img = _noisy_field(seed=8)
        params = SNRParams(background_bbox=(0, 0, 50, 50), signal_bbox=(100, 100, 30, 30))
        r1 = measure_snr(img, params)
        r2 = measure_snr(img, params)
        assert r1.overall.snr == r2.overall.snr
        assert r1.overall.snr_db == r2.overall.snr_db
        assert r1.overall.background_std == r2.overall.background_std

    def test_empty_region_raises(self):
        img = _noisy_field(seed=9)
        params = SNRParams(background_bbox=(SIZE + 10, SIZE + 10, 5, 5))
        with pytest.raises(ValueError):
            measure_snr(img, params)

    def test_net_signal_never_negative(self):
        """When the signal region is dimmer than background, net_signal clamps to 0."""
        img = _noisy_field(mean=0.2, std=0.01, seed=10)
        signal_bbox = (80, 80, 20, 20)
        img[80:100, 80:100] -= 0.15  # make the "signal" region darker than background
        background_bbox = (0, 0, SIZE, 30)
        params = SNRParams(background_bbox=background_bbox, signal_bbox=signal_bbox)
        result = measure_snr(img, params)
        assert result.overall.net_signal == 0.0
        assert result.overall.snr == 0.0

    def test_unexpected_shape_raises(self):
        with pytest.raises(ValueError):
            measure_snr(np.zeros((2, 3, 4, 5), dtype=np.float32))
