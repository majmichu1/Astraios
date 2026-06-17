"""AI Denoise Model — U-Net variant specialised for astronomical noise reduction.

Extends the base U-Net with:
- Residual learning (predicts noise, subtracts from input)
- Noise-level conditioning via a small MLP that modulates features
- Slightly deeper architecture (depth=4, base_features=32) for better quality
"""

from __future__ import annotations

import torch
import torch.nn as nn

from astraios.ai.models.unet import UNet


class NoiseConditioningMLP(nn.Module):
    """Small MLP that encodes a scalar noise level into feature-space bias."""

    def __init__(self, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_features),
        )

    def forward(self, noise_level: torch.Tensor) -> torch.Tensor:
        """Map noise level (B, 1) → (B, C) feature bias."""
        return self.net(noise_level)


class DenoiseUNet(nn.Module):
    """U-Net for denoising with residual learning and noise-level conditioning.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for mono processing).
    base_features : int
        Feature count at the first encoder level.
    depth : int
        Number of encoder/decoder levels.
    use_noise_conditioning : bool
        If True, accepts an optional noise_level input for adaptive denoising.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_features: int = 32,
        depth: int = 4,
        use_noise_conditioning: bool = True,
    ):
        super().__init__()
        self.use_noise_conditioning = use_noise_conditioning
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=in_channels,
            base_features=base_features,
            depth=depth,
        )
        if use_noise_conditioning:
            self.noise_mlp = NoiseConditioningMLP(in_channels)

    def forward(
        self,
        x: torch.Tensor,
        noise_level: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass — predicts noise and subtracts from input.

        Parameters
        ----------
        x : Tensor
            Input image (B, C, H, W).
        noise_level : Tensor, optional
            Per-sample noise level (B, 1). Used when noise conditioning is enabled.

        Returns
        -------
        Tensor
            Denoised image (B, C, H, W).
        """
        predicted_noise = self.unet(x)

        # Modulate predicted noise by conditioning
        if self.use_noise_conditioning and noise_level is not None:
            bias = self.noise_mlp(noise_level)  # (B, C)
            predicted_noise = predicted_noise + bias[:, :, None, None]

        # Residual learning: output = input - predicted_noise
        denoised = x - predicted_noise
        return denoised.clamp(0.0, 1.0)


def create_denoise_model(
    in_channels: int = 1,
    base_features: int = 32,
    depth: int = 4,
    use_noise_conditioning: bool = True,
) -> DenoiseUNet:
    """Factory function to create a denoise model.

    Parameters
    ----------
    in_channels : int
        1 for mono, 3 for color (tiled inference uses mono).
    base_features : int
        Width of the first encoder level.
    depth : int
        Number of encoder/decoder levels.
    use_noise_conditioning : bool
        Whether to enable noise-level conditioning.

    Returns
    -------
    DenoiseUNet
        Instantiated model (untrained).
    """
    return DenoiseUNet(
        in_channels=in_channels,
        base_features=base_features,
        depth=depth,
        use_noise_conditioning=use_noise_conditioning,
    )
