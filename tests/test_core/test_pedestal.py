"""Tests for pedestal (uniform offset) management."""

from __future__ import annotations

import numpy as np

from astraios.core.masks import Mask
from astraios.core.pedestal import PedestalParams, apply_pedestal


class TestIdentity:
    def test_default_params_mono_identity(self):
        data = np.random.default_rng(0).random((64, 64)).astype(np.float32)
        result = apply_pedestal(data)
        np.testing.assert_allclose(result, data, atol=1e-7)

    def test_default_params_color_identity(self):
        data = np.random.default_rng(1).random((3, 64, 64)).astype(np.float32)
        result = apply_pedestal(data)
        np.testing.assert_allclose(result, data, atol=1e-7)

    def test_add_zero_amount_identity(self):
        data = np.random.default_rng(2).random((3, 64, 64)).astype(np.float32) * 0.5 + 0.1
        result = apply_pedestal(data, PedestalParams(mode="add", amount=0.0))
        np.testing.assert_allclose(result, data, atol=1e-7)


class TestAdd:
    def test_add_uniform_amount_mono(self):
        data = np.full((32, 32), 0.2, dtype=np.float32)
        result = apply_pedestal(data, PedestalParams(mode="add", amount=0.1))
        np.testing.assert_allclose(result, 0.3, atol=1e-6)

    def test_add_uniform_amount_color_per_channel_default(self):
        data = np.full((3, 32, 32), 0.2, dtype=np.float32)
        result = apply_pedestal(data, PedestalParams(mode="add", amount=0.1))
        np.testing.assert_allclose(result, 0.3, atol=1e-6)

    def test_add_explicit_channel_amounts(self):
        data = np.zeros((3, 8, 8), dtype=np.float32)
        params = PedestalParams(mode="add", per_channel=True, channel_amounts=[0.1, 0.2, 0.3])
        result = apply_pedestal(data, params)
        np.testing.assert_allclose(result[0], 0.1, atol=1e-6)
        np.testing.assert_allclose(result[1], 0.2, atol=1e-6)
        np.testing.assert_allclose(result[2], 0.3, atol=1e-6)

    def test_add_clips_to_range(self):
        data = np.full((8, 8), 0.95, dtype=np.float32)
        result = apply_pedestal(data, PedestalParams(mode="add", amount=0.5))
        assert result.max() <= 1.0

    def test_add_no_clip_option(self):
        data = np.full((8, 8), 0.95, dtype=np.float32)
        result = apply_pedestal(data, PedestalParams(mode="add", amount=0.5, clip=False))
        assert result.max() > 1.0


class TestRemove:
    def test_remove_auto_per_channel_matches_saspro_math(self):
        rng = np.random.default_rng(3)
        data = rng.random((3, 16, 16)).astype(np.float32)
        data[0] += 0.1  # per-channel floor differs, so per-channel vs global should differ
        data[1] += 0.3
        data = np.clip(data, 0.0, 1.0).astype(np.float32)

        result = apply_pedestal(data, PedestalParams(mode="remove", per_channel=True))
        for c in range(3):
            expected = data[c] - data[c].min()
            np.testing.assert_allclose(result[c], np.clip(expected, 0.0, 1.0), atol=1e-6)
            assert abs(float(result[c].min())) < 1e-6

    def test_remove_auto_mono(self):
        data = np.random.default_rng(4).random((16, 16)).astype(np.float32) * 0.5 + 0.2
        result = apply_pedestal(data, PedestalParams(mode="remove"))
        assert abs(float(result.min())) < 1e-6
        np.testing.assert_allclose(result, data - data.min(), atol=1e-6)

    def test_remove_global_uses_single_minimum(self):
        data = np.zeros((3, 4, 4), dtype=np.float32)
        data[0] = 0.5
        data[1] = 0.3
        data[2] = 0.7
        result = apply_pedestal(data, PedestalParams(mode="remove", per_channel=False))
        # Global minimum across all channels is 0.3 (channel 1); subtract that everywhere.
        np.testing.assert_allclose(result[0], 0.2, atol=1e-6)
        np.testing.assert_allclose(result[1], 0.0, atol=1e-6)
        np.testing.assert_allclose(result[2], 0.4, atol=1e-6)

    def test_remove_per_channel_differs_from_global(self):
        data = np.zeros((3, 4, 4), dtype=np.float32)
        data[0] = 0.5
        data[1] = 0.3
        data[2] = 0.7
        per_channel = apply_pedestal(data, PedestalParams(mode="remove", per_channel=True))
        global_ = apply_pedestal(data, PedestalParams(mode="remove", per_channel=False))
        assert not np.allclose(per_channel, global_)


class TestRoundTrip:
    def test_add_then_remove_exact_roundtrip_mono(self):
        data = np.random.default_rng(5).random((32, 32)).astype(np.float32) * 0.5 + 0.1
        added = apply_pedestal(data, PedestalParams(mode="add", amount=0.15))
        restored = apply_pedestal(added, PedestalParams(mode="remove", remove_amount=0.15))
        np.testing.assert_allclose(restored, data, atol=1e-6)

    def test_add_then_remove_exact_roundtrip_color_uniform(self):
        data = np.random.default_rng(6).random((3, 32, 32)).astype(np.float32) * 0.5 + 0.1
        added = apply_pedestal(data, PedestalParams(mode="add", amount=0.1))
        restored = apply_pedestal(
            added, PedestalParams(mode="remove", remove_amount=0.1, per_channel=False)
        )
        np.testing.assert_allclose(restored, data, atol=1e-6)

    def test_add_then_remove_exact_roundtrip_per_channel(self):
        data = np.random.default_rng(7).random((3, 32, 32)).astype(np.float32) * 0.4 + 0.1
        amounts = [0.05, 0.1, 0.15]
        added = apply_pedestal(
            data, PedestalParams(mode="add", per_channel=True, channel_amounts=amounts)
        )
        restored = apply_pedestal(
            added, PedestalParams(mode="remove", per_channel=True, remove_amount=amounts)
        )
        np.testing.assert_allclose(restored, data, atol=1e-6)


class TestMaskSupport:
    def test_mask_protects_region(self):
        data = np.full((32, 32), 0.3, dtype=np.float32)
        mask_data = np.zeros((32, 32), dtype=np.float32)
        mask_data[16:] = 1.0
        mask = Mask(data=mask_data)
        result = apply_pedestal(data, PedestalParams(mode="add", amount=0.2), mask=mask)
        np.testing.assert_allclose(result[:16], data[:16], atol=1e-6)
        np.testing.assert_allclose(result[16:], 0.5, atol=1e-6)


class TestShapes:
    def test_invalid_mode_raises(self):
        data = np.zeros((8, 8), dtype=np.float32)
        try:
            apply_pedestal(data, PedestalParams(mode="bogus"))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for unknown mode")

    def test_mono_shape_preserved(self):
        data = np.random.default_rng(8).random((16, 16)).astype(np.float32)
        result = apply_pedestal(data, PedestalParams(mode="remove"))
        assert result.shape == data.shape
        assert result.dtype == np.float32

    def test_color_shape_preserved(self):
        data = np.random.default_rng(9).random((3, 16, 16)).astype(np.float32)
        result = apply_pedestal(data, PedestalParams(mode="remove"))
        assert result.shape == data.shape
