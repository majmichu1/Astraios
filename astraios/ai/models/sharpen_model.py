"""AI Sharpen Model — U-Net variant specialised for astronomical deblurring.

Uses a multi-scale architecture with:
- Residual learning (predicts a detail/sharpening map added to input)
- Perceptual-loss-friendly design (deeper features for structure recovery)
- Optional PSF-width conditioning for adaptive deconvolution
"""

from __future__ import annotations

import torch
import torch.nn as nn

from astraios.ai.models.unet import UNet


class PSFConditioningMLP(nn.Module):
    """MLP that encodes a PSF FWHM into feature-space modulation."""

    def __init__(self, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_features),
        )

    def forward(self, psf_fwhm: torch.Tensor) -> torch.Tensor:
        """Map PSF FWHM (B, 1) → (B, C) feature modulation."""
        return self.net(psf_fwhm)


class SharpenUNet(nn.Module):
    """U-Net for sharpening / deblurring with residual detail prediction.

    The model predicts a detail map (high-frequency content) that is added
    to the input to produce a sharper result.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for mono processing).
    base_features : int
        Feature count at the first encoder level.
    depth : int
        Number of encoder/decoder levels.
    use_psf_conditioning : bool
        If True, accepts optional PSF FWHM for adaptive sharpening.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_features: int = 32,
        depth: int = 4,
        use_psf_conditioning: bool = True,
    ):
        super().__init__()
        self.use_psf_conditioning = use_psf_conditioning
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=in_channels,
            base_features=base_features,
            depth=depth,
        )
        if use_psf_conditioning:
            self.psf_mlp = PSFConditioningMLP(in_channels)

    def forward(
        self,
        x: torch.Tensor,
        psf_fwhm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass — predicts detail map and adds to input.

        Parameters
        ----------
        x : Tensor
            Input image (B, C, H, W).
        psf_fwhm : Tensor, optional
            Per-sample PSF FWHM in pixels (B, 1). Used for conditioning.

        Returns
        -------
        Tensor
            Sharpened image (B, C, H, W).
        """
        detail = self.unet(x)

        # Modulate detail by PSF conditioning
        if self.use_psf_conditioning and psf_fwhm is not None:
            scale = self.psf_mlp(psf_fwhm)  # (B, C)
            detail = detail * (1.0 + scale[:, :, None, None])

        # Residual learning: output = input + detail
        sharpened = x + detail
        return sharpened.clamp(0.0, 1.0)


def create_sharpen_model(
    in_channels: int = 1,
    base_features: int = 32,
    depth: int = 4,
    use_psf_conditioning: bool = True,
) -> SharpenUNet:
    """Factory function to create a sharpen model.

    Parameters
    ----------
    in_channels : int
        1 for mono, 3 for color (tiled inference uses mono).
    base_features : int
        Width of the first encoder level.
    depth : int
        Number of encoder/decoder levels.
    use_psf_conditioning : bool
        Whether to enable PSF FWHM conditioning.

    Returns
    -------
    SharpenUNet
        Instantiated model (untrained).
    """
    return SharpenUNet(
        in_channels=in_channels,
        base_features=base_features,
        depth=depth,
        use_psf_conditioning=use_psf_conditioning,
    )
