"""Tests for AI denoise model architecture."""

import pytest
import torch

from astraios.ai.models.denoise_model import (
    DenoiseUNet,
    NoiseConditioningMLP,
    create_denoise_model,
)


def _random_batch(batch=1, channels=1, h=64, w=64):
    """Create a random float32 tensor in [0, 1]."""
    return torch.rand(batch, channels, h, w, dtype=torch.float32)


class TestNoiseConditioningMLP:
    def test_output_shape(self):
        mlp = NoiseConditioningMLP(out_features=16)
        noise_level = torch.rand(2, 1)
        out = mlp(noise_level)
        assert out.shape == (2, 16)

    def test_single_sample(self):
        mlp = NoiseConditioningMLP(out_features=8)
        noise_level = torch.tensor([[0.5]])
        out = mlp(noise_level)
        assert out.shape == (1, 8)

    def test_different_noise_levels_give_different_outputs(self):
        mlp = NoiseConditioningMLP(out_features=8)
        low = torch.tensor([[0.01]])
        high = torch.tensor([[0.5]])
        out_low = mlp(low)
        out_high = mlp(high)
        assert not torch.allclose(out_low, out_high)


class TestDenoiseUNet:
    def test_output_shape_mono(self):
        model = DenoiseUNet(in_channels=1, base_features=8, depth=2)
        x = _random_batch(channels=1, h=64, w=64)
        out = model(x)
        assert out.shape == x.shape

    def test_output_shape_color(self):
        model = DenoiseUNet(in_channels=3, base_features=8, depth=2)
        x = _random_batch(channels=3, h=64, w=64)
        out = model(x)
        assert out.shape == x.shape

    def test_output_clamped_to_unit_range(self):
        """Output values must be in [0, 1] regardless of input."""
        model = DenoiseUNet(in_channels=1, base_features=8, depth=2)
        x = _random_batch(channels=1, h=64, w=64)
        out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_noise_conditioning_changes_output(self):
        """Providing noise_level should change the output compared to not providing it."""
        model = DenoiseUNet(
            in_channels=1, base_features=8, depth=2, use_noise_conditioning=True
        )
        x = _random_batch(channels=1, h=64, w=64)
        noise_level = torch.tensor([[0.3]])

        out_without = model(x)
        out_with = model(x, noise_level=noise_level)

        assert not torch.allclose(out_without, out_with, atol=1e-6)

    def test_different_noise_levels_give_different_outputs(self):
        model = DenoiseUNet(
            in_channels=1, base_features=8, depth=2, use_noise_conditioning=True
        )
        x = _random_batch(channels=1, h=64, w=64)

        out_low = model(x, noise_level=torch.tensor([[0.01]]))
        out_high = model(x, noise_level=torch.tensor([[0.5]]))

        assert not torch.allclose(out_low, out_high, atol=1e-6)

    def test_without_noise_conditioning(self):
        """Model with use_noise_conditioning=False should ignore noise_level."""
        model = DenoiseUNet(
            in_channels=1, base_features=8, depth=2, use_noise_conditioning=False
        )
        x = _random_batch(channels=1, h=64, w=64)

        out_no_level = model(x)
        out_with_level = model(x, noise_level=torch.tensor([[0.3]]))

        torch.testing.assert_close(out_no_level, out_with_level)

    def test_without_noise_conditioning_no_mlp_attribute(self):
        model = DenoiseUNet(
            in_channels=1, base_features=8, depth=2, use_noise_conditioning=False
        )
        assert not hasattr(model, "noise_mlp")

    def test_gradient_flows(self):
        """Verify that gradients flow through the entire network."""
        model = DenoiseUNet(
            in_channels=1, base_features=8, depth=2, use_noise_conditioning=True
        )
        x = _random_batch(channels=1, h=64, w=64)
        x.requires_grad_(True)
        noise_level = torch.tensor([[0.2]], requires_grad=True)

        out = model(x, noise_level=noise_level)
        loss = out.sum()
        loss.backward()

        # Input gradient should be non-zero
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

        # Noise level gradient should be non-zero
        assert noise_level.grad is not None
        assert noise_level.grad.abs().sum() > 0

        # All model parameters should have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_gradient_flows_without_conditioning(self):
        """Gradients should flow even without noise conditioning."""
        model = DenoiseUNet(
            in_channels=1, base_features=8, depth=2, use_noise_conditioning=False
        )
        x = _random_batch(channels=1, h=64, w=64)
        x.requires_grad_(True)

        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_batch_size_greater_than_one(self):
        model = DenoiseUNet(in_channels=1, base_features=8, depth=2)
        x = _random_batch(batch=4, channels=1, h=64, w=64)
        noise_level = torch.rand(4, 1)
        out = model(x, noise_level=noise_level)
        assert out.shape == (4, 1, 64, 64)

    def test_eval_mode(self):
        """Model should work in eval mode."""
        model = DenoiseUNet(in_channels=1, base_features=8, depth=2)
        model.train()
        x = _random_batch(channels=1, h=64, w=64)
        _ = model(x)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape


class TestCreateDenoiseModel:
    def test_factory_returns_correct_type(self):
        model = create_denoise_model(
            in_channels=1, base_features=8, depth=2
        )
        assert isinstance(model, DenoiseUNet)

    def test_factory_default_has_noise_conditioning(self):
        model = create_denoise_model(base_features=8, depth=2)
        assert model.use_noise_conditioning is True
        assert hasattr(model, "noise_mlp")

    def test_factory_without_noise_conditioning(self):
        model = create_denoise_model(
            base_features=8, depth=2, use_noise_conditioning=False
        )
        assert model.use_noise_conditioning is False

    def test_factory_color_model(self):
        model = create_denoise_model(in_channels=3, base_features=8, depth=2)
        x = _random_batch(channels=3, h=64, w=64)
        out = model(x)
        assert out.shape == x.shape
