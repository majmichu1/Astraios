"""Tests for AI sharpen model architecture."""

import pytest
import torch

from astraios.ai.models.sharpen_model import (
    PSFConditioningMLP,
    SharpenUNet,
    create_sharpen_model,
)


def _random_batch(batch=1, channels=1, h=64, w=64):
    """Create a random float32 tensor in [0, 1]."""
    return torch.rand(batch, channels, h, w, dtype=torch.float32)


class TestPSFConditioningMLP:
    def test_output_shape(self):
        mlp = PSFConditioningMLP(out_features=16)
        psf_fwhm = torch.rand(2, 1)
        out = mlp(psf_fwhm)
        assert out.shape == (2, 16)

    def test_single_sample(self):
        mlp = PSFConditioningMLP(out_features=8)
        psf_fwhm = torch.tensor([[2.5]])
        out = mlp(psf_fwhm)
        assert out.shape == (1, 8)

    def test_different_fwhm_give_different_outputs(self):
        mlp = PSFConditioningMLP(out_features=8)
        narrow = torch.tensor([[1.0]])
        wide = torch.tensor([[5.0]])
        out_narrow = mlp(narrow)
        out_wide = mlp(wide)
        assert not torch.allclose(out_narrow, out_wide)


class TestSharpenUNet:
    def test_output_shape_mono(self):
        model = SharpenUNet(in_channels=1, base_features=8, depth=2)
        x = _random_batch(channels=1, h=64, w=64)
        out = model(x)
        assert out.shape == x.shape

    def test_output_shape_color(self):
        model = SharpenUNet(in_channels=3, base_features=8, depth=2)
        x = _random_batch(channels=3, h=64, w=64)
        out = model(x)
        assert out.shape == x.shape

    def test_output_clamped_to_unit_range(self):
        """Output values must be in [0, 1] regardless of input."""
        model = SharpenUNet(in_channels=1, base_features=8, depth=2)
        x = _random_batch(channels=1, h=64, w=64)
        out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_psf_conditioning_changes_output(self):
        """Providing psf_fwhm should change the output compared to not providing it."""
        model = SharpenUNet(
            in_channels=1, base_features=8, depth=2, use_psf_conditioning=True
        )
        x = _random_batch(channels=1, h=64, w=64)
        psf_fwhm = torch.tensor([[3.0]])

        out_without = model(x)
        out_with = model(x, psf_fwhm=psf_fwhm)

        assert not torch.allclose(out_without, out_with, atol=1e-6)

    def test_different_psf_fwhm_give_different_outputs(self):
        model = SharpenUNet(
            in_channels=1, base_features=8, depth=2, use_psf_conditioning=True
        )
        x = _random_batch(channels=1, h=64, w=64)

        out_narrow = model(x, psf_fwhm=torch.tensor([[1.0]]))
        out_wide = model(x, psf_fwhm=torch.tensor([[5.0]]))

        assert not torch.allclose(out_narrow, out_wide, atol=1e-6)

    def test_without_psf_conditioning(self):
        """Model with use_psf_conditioning=False should ignore psf_fwhm."""
        model = SharpenUNet(
            in_channels=1, base_features=8, depth=2, use_psf_conditioning=False
        )
        x = _random_batch(channels=1, h=64, w=64)

        out_no_psf = model(x)
        out_with_psf = model(x, psf_fwhm=torch.tensor([[3.0]]))

        torch.testing.assert_close(out_no_psf, out_with_psf)

    def test_without_psf_conditioning_no_mlp_attribute(self):
        model = SharpenUNet(
            in_channels=1, base_features=8, depth=2, use_psf_conditioning=False
        )
        assert not hasattr(model, "psf_mlp")

    def test_gradient_flows(self):
        """Verify that gradients flow through the entire network."""
        model = SharpenUNet(
            in_channels=1, base_features=8, depth=2, use_psf_conditioning=True
        )
        x = _random_batch(channels=1, h=64, w=64)
        x.requires_grad_(True)
        psf_fwhm = torch.tensor([[2.5]], requires_grad=True)

        out = model(x, psf_fwhm=psf_fwhm)
        loss = out.sum()
        loss.backward()

        # Input gradient should be non-zero
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

        # PSF FWHM gradient should be non-zero
        assert psf_fwhm.grad is not None
        assert psf_fwhm.grad.abs().sum() > 0

        # All model parameters should have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_gradient_flows_without_conditioning(self):
        """Gradients should flow even without PSF conditioning."""
        model = SharpenUNet(
            in_channels=1, base_features=8, depth=2, use_psf_conditioning=False
        )
        x = _random_batch(channels=1, h=64, w=64)
        x.requires_grad_(True)

        out = model(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_batch_size_greater_than_one(self):
        model = SharpenUNet(in_channels=1, base_features=8, depth=2)
        x = _random_batch(batch=4, channels=1, h=64, w=64)
        psf_fwhm = torch.rand(4, 1) * 5.0
        out = model(x, psf_fwhm=psf_fwhm)
        assert out.shape == (4, 1, 64, 64)

    def test_eval_mode(self):
        """Model should work in eval mode."""
        model = SharpenUNet(in_channels=1, base_features=8, depth=2)
        model.train()
        x = _random_batch(channels=1, h=64, w=64)
        _ = model(x)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape


class TestCreateSharpenModel:
    def test_factory_returns_correct_type(self):
        model = create_sharpen_model(
            in_channels=1, base_features=8, depth=2
        )
        assert isinstance(model, SharpenUNet)

    def test_factory_default_has_psf_conditioning(self):
        model = create_sharpen_model(base_features=8, depth=2)
        assert model.use_psf_conditioning is True
        assert hasattr(model, "psf_mlp")

    def test_factory_without_psf_conditioning(self):
        model = create_sharpen_model(
            base_features=8, depth=2, use_psf_conditioning=False
        )
        assert model.use_psf_conditioning is False

    def test_factory_color_model(self):
        model = create_sharpen_model(in_channels=3, base_features=8, depth=2)
        x = _random_batch(channels=3, h=64, w=64)
        out = model(x)
        assert out.shape == x.shape
