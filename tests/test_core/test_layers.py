"""Tests for the layer compositing system (astraios/core/layers.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from astraios.core.layers import (
    BLEND_MODES,
    Layer,
    LayerStack,
    _apply_blend_mode,
)
from astraios.core.masks import Mask, MaskType


def _mono(value: float, h: int = 16, w: int = 16) -> np.ndarray:
    return np.full((h, w), value, dtype=np.float32)


def _color(value: float, h: int = 16, w: int = 16) -> np.ndarray:
    return np.full((3, h, w), value, dtype=np.float32)


def _gradient_mono(h: int = 16, w: int = 16) -> np.ndarray:
    return np.linspace(0.0, 1.0, h * w, dtype=np.float32).reshape(h, w)


def _gradient_color(h: int = 16, w: int = 16) -> np.ndarray:
    g = _gradient_mono(h, w)
    return np.stack([g, g[::-1, ::-1].copy(), np.full((h, w), 0.5, dtype=np.float32)], axis=0)


# ---------------------------------------------------------------------------
# Layer dataclass basics
# ---------------------------------------------------------------------------


class TestLayer:
    def test_clips_data_to_unit_range(self):
        layer = Layer(name="L", data=np.array([[1.5, -0.5], [0.2, 2.0]], dtype=np.float32))
        assert layer.data.min() >= 0.0
        assert layer.data.max() <= 1.0

    def test_unknown_blend_mode_falls_back_to_normal(self):
        layer = Layer(name="L", data=_mono(0.5), blend_mode="Nonsense Mode")
        assert layer.blend_mode == "Normal"

    def test_opacity_clamped(self):
        layer = Layer(name="L", data=_mono(0.5), opacity=5.0)
        assert layer.opacity == 1.0
        layer2 = Layer(name="L", data=_mono(0.5), opacity=-5.0)
        assert layer2.opacity == 0.0

    def test_is_color_and_dimensions(self):
        mono = Layer(name="mono", data=_mono(0.5, 10, 20))
        color = Layer(name="color", data=_color(0.5, 10, 20))
        assert not mono.is_color
        assert color.is_color
        assert mono.height == 10 and mono.width == 20
        assert color.height == 10 and color.width == 20

    def test_copy_is_independent(self):
        layer = Layer(name="orig", data=_mono(0.5), mask=Mask(data=_mono(1.0), name="m"))
        dup = layer.copy()
        assert dup is not layer
        assert dup.data is not layer.data
        assert dup.mask is not layer.mask
        dup.data[:] = 0.0
        assert layer.data.max() == pytest.approx(0.5)
        assert dup.name == "orig copy"


# ---------------------------------------------------------------------------
# Blend mode math — sanity across all modes, mono + color
# ---------------------------------------------------------------------------


class TestAllBlendModesSanity:
    @pytest.mark.parametrize("mode", BLEND_MODES)
    def test_mono_finite_and_bounded(self, mode):
        base = torch.from_numpy(_gradient_mono())
        src = torch.from_numpy(_gradient_mono()[::-1, ::-1].copy())
        layer = Layer(name="L", data=_mono(0.5), blend_mode=mode)
        out = _apply_blend_mode(mode, base, src, layer)
        assert torch.isfinite(out).all()
        assert out.min() >= -1e-4
        assert out.max() <= 1.0 + 1e-4

    @pytest.mark.parametrize("mode", BLEND_MODES)
    def test_color_finite_and_bounded(self, mode):
        base = torch.from_numpy(_gradient_color())
        src = torch.from_numpy(_gradient_color()[:, ::-1, ::-1].copy())
        layer = Layer(name="L", data=_color(0.5), blend_mode=mode)
        out = _apply_blend_mode(mode, base, src, layer)
        assert torch.isfinite(out).all()
        assert out.min() >= -1e-4
        assert out.max() <= 1.0 + 1e-4
        assert out.shape == base.shape


# ---------------------------------------------------------------------------
# Specific blend-mode semantics
# ---------------------------------------------------------------------------


class TestBlendSemantics:
    def test_normal_full_opacity_returns_top_over_base(self):
        base = Layer(name="base", data=_mono(0.2))
        top = Layer(name="top", data=_mono(0.9), blend_mode="Normal", opacity=1.0)
        stack = LayerStack(layers=[top, base])
        result = stack.composite()
        assert np.allclose(result, 0.9, atol=1e-5)

    def test_normal_color_full_opacity_returns_top(self):
        base = Layer(name="base", data=_color(0.1))
        top = Layer(name="top", data=_color(0.8), opacity=1.0)
        stack = LayerStack(layers=[top, base])
        result = stack.composite()
        assert result.shape == (3, 16, 16)
        assert np.allclose(result, 0.8, atol=1e-5)

    def test_normal_opacity_half_is_midpoint(self):
        base = Layer(name="base", data=_mono(0.0))
        top = Layer(name="top", data=_mono(1.0), opacity=0.5)
        stack = LayerStack(layers=[top, base])
        result = stack.composite()
        assert np.allclose(result, 0.5, atol=1e-5)

    def test_screen_brightens_relative_to_normal(self):
        base = Layer(name="base", data=_mono(0.4))
        top_normal = Layer(name="top", data=_mono(0.4), blend_mode="Normal")
        top_screen = Layer(name="top", data=_mono(0.4), blend_mode="Screen")
        normal_result = LayerStack(layers=[top_normal, base]).composite()
        screen_result = LayerStack(layers=[top_screen, base]).composite()
        assert screen_result.mean() >= normal_result.mean()
        # Screen(0.4, 0.4) = 1 - 0.6*0.6 = 0.64 > 0.4
        assert np.allclose(screen_result, 0.64, atol=1e-5)

    def test_multiply_darkens(self):
        base = Layer(name="base", data=_mono(0.6))
        top = Layer(name="top", data=_mono(0.5), blend_mode="Multiply")
        result = LayerStack(layers=[top, base]).composite()
        # Multiply(0.6, 0.5) = 0.3, strictly darker than either input
        assert np.allclose(result, 0.3, atol=1e-5)
        assert result.max() < 0.6

    def test_lighten_picks_max(self):
        base = Layer(name="base", data=_mono(0.3))
        top = Layer(name="top", data=_mono(0.7), blend_mode="Lighten")
        result = LayerStack(layers=[top, base]).composite()
        assert np.allclose(result, 0.7, atol=1e-5)

    def test_darken_picks_min(self):
        base = Layer(name="base", data=_mono(0.3))
        top = Layer(name="top", data=_mono(0.7), blend_mode="Darken")
        result = LayerStack(layers=[top, base]).composite()
        assert np.allclose(result, 0.3, atol=1e-5)

    def test_difference(self):
        base = Layer(name="base", data=_mono(0.7))
        top = Layer(name="top", data=_mono(0.2), blend_mode="Difference")
        result = LayerStack(layers=[top, base]).composite()
        assert np.allclose(result, 0.5, atol=1e-5)

    def test_add_clips_at_one(self):
        base = Layer(name="base", data=_mono(0.8))
        top = Layer(name="top", data=_mono(0.5), blend_mode="Add")
        result = LayerStack(layers=[top, base]).composite()
        assert np.allclose(result, 1.0, atol=1e-5)

    def test_subtract_clips_at_zero(self):
        base = Layer(name="base", data=_mono(0.2))
        top = Layer(name="top", data=_mono(0.5), blend_mode="Subtract")
        result = LayerStack(layers=[top, base]).composite()
        assert np.allclose(result, 0.0, atol=1e-5)

    def test_luminosity_mono_degenerates_to_src(self):
        base = Layer(name="base", data=_mono(0.9))
        top = Layer(name="top", data=_mono(0.3), blend_mode="Luminosity")
        result = LayerStack(layers=[top, base]).composite()
        assert np.allclose(result, 0.3, atol=1e-5)

    def test_invisible_layer_ignored(self):
        base = Layer(name="base", data=_mono(0.2))
        top = Layer(name="top", data=_mono(0.9), visible=False)
        result = LayerStack(layers=[top, base]).composite()
        assert np.allclose(result, 0.2, atol=1e-5)

    def test_empty_or_all_hidden_stack_returns_none(self):
        assert LayerStack(layers=[]).composite() is None
        base = Layer(name="base", data=_mono(0.5), visible=False)
        assert LayerStack(layers=[base]).composite() is None


# ---------------------------------------------------------------------------
# Mixed mono/color promotion
# ---------------------------------------------------------------------------


class TestMixedMonoColor:
    def test_mono_base_color_top_promotes_to_color(self):
        base = Layer(name="base", data=_mono(0.5))
        top = Layer(name="top", data=_color(0.8), opacity=1.0)
        result = LayerStack(layers=[top, base]).composite()
        assert result.shape == (3, 16, 16)
        assert np.allclose(result, 0.8, atol=1e-5)

    def test_color_base_mono_top_promotes_to_color(self):
        base = Layer(name="base", data=_color(0.2))
        top = Layer(name="top", data=_mono(0.9), opacity=1.0)
        result = LayerStack(layers=[top, base]).composite()
        assert result.shape == (3, 16, 16)
        assert np.allclose(result, 0.9, atol=1e-5)

    def test_all_mono_stays_mono(self):
        base = Layer(name="base", data=_mono(0.2))
        top = Layer(name="top", data=_mono(0.9))
        result = LayerStack(layers=[top, base]).composite()
        assert result.ndim == 2


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


class TestMasks:
    def test_mask_confines_blend_region(self):
        base = Layer(name="base", data=_mono(0.0))
        mask_data = np.zeros((16, 16), dtype=np.float32)
        mask_data[:, :8] = 1.0  # left half fully blended, right half protected
        top = Layer(
            name="top",
            data=_mono(1.0),
            opacity=1.0,
            mask=Mask(data=mask_data, name="half", mask_type=MaskType.MANUAL),
        )
        result = LayerStack(layers=[top, base]).composite()
        assert np.allclose(result[:, :8], 1.0, atol=1e-5)
        assert np.allclose(result[:, 8:], 0.0, atol=1e-5)

    def test_mask_combined_with_opacity(self):
        base = Layer(name="base", data=_mono(0.0))
        mask_data = np.full((16, 16), 0.5, dtype=np.float32)
        top = Layer(
            name="top",
            data=_mono(1.0),
            opacity=0.5,
            mask=Mask(data=mask_data, name="half-strength"),
        )
        result = LayerStack(layers=[top, base]).composite()
        # alpha = opacity(0.5) * mask(0.5) = 0.25 -> result = 1*0.25 + 0*0.75
        assert np.allclose(result, 0.25, atol=1e-5)


# ---------------------------------------------------------------------------
# Stack operations: add/remove/move/duplicate/merge
# ---------------------------------------------------------------------------


class TestStackOperations:
    def test_add_default_inserts_at_top(self):
        stack = LayerStack()
        a = Layer(name="a", data=_mono(0.1))
        b = Layer(name="b", data=_mono(0.2))
        stack.add(a)
        stack.add(b)
        assert [layer.name for layer in stack.layers] == ["b", "a"]

    def test_add_at_explicit_index(self):
        stack = LayerStack()
        a = Layer(name="a", data=_mono(0.1))
        b = Layer(name="b", data=_mono(0.2))
        c = Layer(name="c", data=_mono(0.3))
        stack.add(a)
        stack.add(b)
        stack.add(c, index=1)
        assert [layer.name for layer in stack.layers] == ["b", "c", "a"]

    def test_remove(self):
        a = Layer(name="a", data=_mono(0.1))
        b = Layer(name="b", data=_mono(0.2))
        stack = LayerStack(layers=[a, b])
        removed = stack.remove(0)
        assert removed.name == "a"
        assert len(stack) == 1
        assert stack.layers[0].name == "b"

    def test_duplicate_inserts_copy_above(self):
        stack = LayerStack(layers=[Layer(name="a", data=_mono(0.1))])
        dup = stack.duplicate(0)
        assert len(stack) == 2
        assert stack.layers[0] is dup
        assert stack.layers[1].name == "a"
        assert dup.data is not stack.layers[1].data

    def test_move_up_and_down(self):
        a = Layer(name="a", data=_mono(0.1))
        b = Layer(name="b", data=_mono(0.2))
        c = Layer(name="c", data=_mono(0.3))
        stack = LayerStack(layers=[a, b, c])
        new_idx = stack.move(2, -1)  # move c up one slot
        assert new_idx == 1
        assert [layer.name for layer in stack.layers] == ["a", "c", "b"]

    def test_move_out_of_bounds_is_noop(self):
        a = Layer(name="a", data=_mono(0.1))
        stack = LayerStack(layers=[a])
        idx = stack.move(0, -1)
        assert idx == 0
        assert stack.layers == [a]

    def test_merge_down_replaces_two_layers_with_one(self):
        base = Layer(name="base", data=_mono(0.2))
        top = Layer(name="top", data=_mono(0.9), opacity=1.0, blend_mode="Normal")
        stack = LayerStack(layers=[top, base])
        merged = stack.merge_down(0)
        assert len(stack) == 1
        assert stack.layers[0] is merged
        assert merged.blend_mode == "Normal"
        assert merged.opacity == 1.0
        assert np.allclose(merged.data, 0.9, atol=1e-5)

    def test_merge_down_no_layer_below_raises(self):
        stack = LayerStack(layers=[Layer(name="only", data=_mono(0.5))])
        with pytest.raises(IndexError):
            stack.merge_down(0)

    def test_merge_down_preserves_overall_composite(self):
        base = Layer(name="base", data=_mono(0.2))
        mid = Layer(name="mid", data=_mono(0.5), blend_mode="Screen", opacity=0.7)
        top = Layer(name="top", data=_mono(0.9), blend_mode="Multiply", opacity=0.8)
        stack = LayerStack(layers=[top, mid, base])
        before = stack.composite()
        stack.merge_down(0)  # merge top into mid
        after = stack.composite()
        assert np.allclose(before, after, atol=1e-5)


# ---------------------------------------------------------------------------
# GPU vs CPU agreement
# ---------------------------------------------------------------------------


class TestDeviceAgreement:
    @pytest.mark.parametrize("mode", BLEND_MODES)
    def test_cpu_matches_cuda_when_available(self, mode):
        if not torch.cuda.is_available():
            pytest.skip("No CUDA device available")
        base_np = _gradient_color()
        src_np = _gradient_color()[:, ::-1, ::-1].copy()
        layer = Layer(name="L", data=_color(0.5), blend_mode=mode)

        base_cpu = torch.from_numpy(base_np)
        src_cpu = torch.from_numpy(src_np)
        out_cpu = _apply_blend_mode(mode, base_cpu, src_cpu, layer)

        base_gpu = base_cpu.to("cuda")
        src_gpu = src_cpu.to("cuda")
        out_gpu = _apply_blend_mode(mode, base_gpu, src_gpu, layer).to("cpu")

        assert torch.allclose(out_cpu, out_gpu, atol=1e-4)
