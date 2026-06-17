"""Tests for U-Net model architecture."""

import numpy as np
import pytest
import torch

from astraios.ai.models.unet import DoubleConv, UNet


def _random_batch(batch=1, channels=1, h=64, w=64):
    """Create a random float32 tensor in [0, 1]."""
    return torch.rand(batch, channels, h, w, dtype=torch.float32)


class TestDoubleConv:
    def test_output_shape(self):
        block = DoubleConv(in_ch=1, out_ch=16)
        x = _random_batch(channels=1)
        out = block(x)
        assert out.shape == (1, 16, 64, 64)

    def test_different_channels(self):
        block = DoubleConv(in_ch=3, out_ch=32)
        x = _random_batch(channels=3)
        out = block(x)
        assert out.shape == (1, 32, 64, 64)


class TestUNet:
    def test_output_shape_mono(self):
        model = UNet(in_channels=1, out_channels=1, base_features=8, depth=2)
        x = _random_batch(channels=1, h=64, w=64)
        out = model(x)
        assert out.shape == x.shape

    def test_output_shape_color(self):
        model = UNet(in_channels=3, out_channels=3, base_features=8, depth=2)
        x = _random_batch(channels=3, h=64, w=64)
        out = model(x)
        assert out.shape == x.shape

    def test_depth_1(self):
        model = UNet(in_channels=1, out_channels=1, base_features=8, depth=1)
        x = _random_batch(channels=1, h=64, w=64)
        out = model(x)
        assert out.shape == x.shape

    def test_depth_3(self):
        model = UNet(in_channels=1, out_channels=1, base_features=8, depth=3)
        x = _random_batch(channels=1, h=64, w=64)
        out = model(x)
        assert out.shape == x.shape

    def test_different_in_out_channels(self):
        model = UNet(in_channels=1, out_channels=3, base_features=8, depth=2)
        x = _random_batch(channels=1, h=64, w=64)
        out = model(x)
        assert out.shape == (1, 3, 64, 64)

    def test_forward_pass_does_not_crash(self):
        """Smoke test: forward pass completes without errors."""
        model = UNet(in_channels=1, out_channels=1, base_features=8, depth=2)
        x = _random_batch(channels=1, h=64, w=64)
        _ = model(x)

    def test_batch_size_greater_than_one(self):
        model = UNet(in_channels=1, out_channels=1, base_features=8, depth=2)
        x = _random_batch(batch=4, channels=1, h=64, w=64)
        out = model(x)
        assert out.shape == (4, 1, 64, 64)

    def test_gradient_flows(self):
        """Verify that gradients flow through the entire network."""
        model = UNet(in_channels=1, out_channels=1, base_features=8, depth=2)
        x = _random_batch(channels=1, h=64, w=64)
        x.requires_grad_(True)

        out = model(x)
        loss = out.sum()
        loss.backward()

        # Input gradient should be non-zero
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

        # All model parameters should have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_non_power_of_two_size(self):
        """Model should handle sizes that are not powers of two."""
        model = UNet(in_channels=1, out_channels=1, base_features=8, depth=2)
        x = _random_batch(channels=1, h=100, w=100)
        out = model(x)
        assert out.shape == x.shape

    def test_eval_mode(self):
        """Model should work in eval mode (BatchNorm uses running stats)."""
        model = UNet(in_channels=1, out_channels=1, base_features=8, depth=2)
        # Run a forward pass in train mode first to populate running stats
        model.train()
        x = _random_batch(channels=1, h=64, w=64)
        _ = model(x)
        # Now switch to eval mode
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape
