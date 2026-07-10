"""Supervised deblur training for the AI Sharpen model.

Trains :class:`astraios.ai.models.sharpen_model.SharpenUNet` to invert
realistic seeing blur: sharp patches from finished stacks are degraded on
the fly with randomly drawn PSFs (Moffat or Gaussian, FWHM 1.0-5.0 px,
mild elongation at a random angle) plus a little Gaussian noise, and the
network learns blurred -> sharp.

Design decisions that must not drift from the inference contract
(:mod:`astraios.ai.inference.sharpen` + ``model_manager.load_model``):

- The model is called BLIND -- ``model(x)`` with no ``psf_fwhm`` -- because
  ``tiled_inference`` never passes one. The PSF-conditioning MLP therefore
  receives no gradients and stays at its initialization; it is inert at
  inference for the same reason. Training with conditioning would optimize
  a code path deployment never exercises.
- The exported ``cosmica_sharpen_v1.pt`` is a BARE ``state_dict`` (loaded
  with ``weights_only=True`` into ``create_sharpen_model()`` defaults:
  mono, base 32, depth 4).
- Mono patches in [0, 1], 256 px -- the same memmap datasets produced by
  ``prepare_data_v2`` for the denoise model.

Low-FWHM samples (1.0-1.5 px, near-identity blur) are deliberately
included so the model learns to leave already-sharp data alone instead of
oversharpening it.

Usage:
    # Fresh start (expects training_data/train.dat etc. from prepare_data_v2):
    python -m astraios.ai.training.train_sharpen

    # Resume:
    python -m astraios.ai.training.train_sharpen --resume astraios/ai/models/sharpen_ckpt_epoch_4.pt
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from astraios.ai.models.sharpen_model import create_sharpen_model

log = logging.getLogger(__name__)

# Hyperparameters
PATCH_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 2e-4
KERNEL_SIZE = 33          # PSF support; comfortably holds FWHM 5 Moffat wings
FWHM_RANGE = (1.0, 5.0)   # px; lower end teaches "do no harm"
ELONGATION_MAX = 1.30     # axis ratio cap (tracking-error simulation)
NOISE_SIGMA_MAX = 0.015   # post-blur Gaussian noise, uniform [0, max]
GRAD_LOSS_WEIGHT = 0.1    # edge-fidelity term on top of L1


class SharpPatchDataset(Dataset):
    """Sharp target patches from the prepare_data_v2 memmap, with flips/rots."""

    def __init__(self, memmap_path: Path, meta_path: Path, augmentation: bool = True):
        with open(meta_path) as f:
            total = int(f.read().strip())
        self.data = np.memmap(
            memmap_path, dtype=np.float32, mode="r",
            shape=(total, PATCH_SIZE, PATCH_SIZE),
        )
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        patch = self.data[idx]
        if self.augmentation:
            if np.random.random() > 0.5:
                patch = np.flip(patch, axis=0).copy()
            if np.random.random() > 0.5:
                patch = np.flip(patch, axis=1).copy()
            patch = np.rot90(patch, np.random.randint(0, 4)).copy()
            jitter = np.random.uniform(0.8, 1.2)
            patch = np.clip(patch * jitter, 0, 1).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0)


def make_psf_bank(batch: int, device: torch.device) -> torch.Tensor:
    """Draw `batch` random PSF kernels, shape (batch, 1, K, K), each sum=1.

    Half Moffat (beta 2.5-4.5, realistic seeing wings), half Gaussian.
    Elongated up to ELONGATION_MAX at a random position angle.
    """
    k = KERNEL_SIZE
    half = k // 2
    ax = torch.arange(k, dtype=torch.float32, device=device) - half
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")  # (K, K)

    fwhm = torch.empty(batch, device=device).uniform_(*FWHM_RANGE)
    ratio = torch.empty(batch, device=device).uniform_(1.0, ELONGATION_MAX)
    theta = torch.empty(batch, device=device).uniform_(0.0, math.pi)
    is_moffat = torch.rand(batch, device=device) < 0.5
    beta = torch.empty(batch, device=device).uniform_(2.5, 4.5)

    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    # rotate coordinates per kernel: (B, K, K)
    xr = cos_t[:, None, None] * xx + sin_t[:, None, None] * yy
    yr = -sin_t[:, None, None] * xx + cos_t[:, None, None] * yy
    # squash the minor axis by the elongation ratio
    r2 = xr**2 + (yr * ratio[:, None, None]) ** 2

    # Gaussian: sigma = FWHM / 2.3548
    sigma = (fwhm / 2.3548)[:, None, None]
    gauss = torch.exp(-r2 / (2.0 * sigma**2))

    # Moffat: alpha = FWHM / (2 sqrt(2^(1/beta) - 1))
    alpha = (fwhm[:, None, None] /
             (2.0 * torch.sqrt(2.0 ** (1.0 / beta[:, None, None]) - 1.0)))
    moffat = (1.0 + r2 / alpha**2) ** (-beta[:, None, None])

    psf = torch.where(is_moffat[:, None, None], moffat, gauss)
    psf = psf / psf.sum(dim=(-2, -1), keepdim=True)
    return psf.unsqueeze(1)  # (B, 1, K, K)


def degrade(sharp: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Blur each sample with its own random PSF, then add mild noise.

    Grouped-conv trick: reshape (B, 1, H, W) -> (1, B, H, W) and convolve
    with a (B, 1, K, K) bank at groups=B so every sample gets its own
    kernel in one call.
    """
    b = sharp.shape[0]
    psf = make_psf_bank(b, device)
    x = sharp.reshape(1, b, *sharp.shape[-2:])
    pad = KERNEL_SIZE // 2
    x = functional.pad(x, (pad, pad, pad, pad), mode="reflect")
    blurred = functional.conv2d(x, psf, groups=b).reshape_as(sharp)

    noise_sigma = torch.empty(b, 1, 1, 1, device=device).uniform_(0.0, NOISE_SIGMA_MAX)
    blurred = blurred + torch.randn_like(blurred) * noise_sigma
    return blurred.clamp_(0.0, 1.0)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 on horizontal+vertical finite differences (edge fidelity)."""
    dpx = pred[..., :, 1:] - pred[..., :, :-1]
    dtx = target[..., :, 1:] - target[..., :, :-1]
    dpy = pred[..., 1:, :] - pred[..., :-1, :]
    dty = target[..., 1:, :] - target[..., :-1, :]
    return (dpx - dtx).abs().mean() + (dpy - dty).abs().mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=str, default=None,
                        help="checkpoint to resume from")
    parser.add_argument("--data-dir", type=str, default="training_data",
                        help="dir containing train.dat/train_meta.txt/"
                             "val.dat/val_meta.txt from prepare_data_v2")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    output_dir = Path("astraios/ai/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on %s", device)

    train_ds = SharpPatchDataset(data_dir / "train.dat", data_dir / "train_meta.txt")
    val_ds = SharpPatchDataset(data_dir / "val.dat", data_dir / "val_meta.txt",
                               augmentation=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    log.info("Train patches: %d, val patches: %d", len(train_ds), len(val_ds))

    model = create_sharpen_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("best_val", best_val)
        log.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    # Reproducible per-epoch degradation stream without fixing every batch.
    torch.manual_seed(1234 + start_epoch)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for i, sharp in enumerate(train_loader):
            sharp = sharp.to(device, non_blocking=True)
            with torch.no_grad():
                blurred = degrade(sharp, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(blurred)  # blind: matches tiled_inference
                loss = (functional.l1_loss(pred, sharp)
                        + GRAD_LOSS_WEIGHT * gradient_loss(pred, sharp))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            if i % 50 == 0:
                log.info("epoch %d  batch %d/%d  loss %.5f",
                         epoch, i, len(train_loader), loss.item())
        scheduler.step()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            torch.manual_seed(999)  # same degradations every epoch -> comparable val
            for sharp in val_loader:
                sharp = sharp.to(device, non_blocking=True)
                blurred = degrade(sharp, device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    pred = model(blurred)
                    val_loss += (functional.l1_loss(pred, sharp)
                                 + GRAD_LOSS_WEIGHT * gradient_loss(pred, sharp)).item()
        val_loss /= max(1, len(val_loader))
        log.info("epoch %d done in %.1fs  train %.5f  val %.5f",
                 epoch, time.time() - t0, train_loss, val_loss)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val": best_val,
            "val_loss": val_loss,
        }, output_dir / f"sharpen_ckpt_epoch_{epoch}.pt")

        if val_loss < best_val:
            best_val = val_loss
            # Bare state_dict: exactly what model_manager.load_model expects.
            torch.save(model.state_dict(), output_dir / "cosmica_sharpen_v1.pt")
            log.info("New best (val %.5f) -> cosmica_sharpen_v1.pt", val_loss)

    log.info("Training complete. Best val loss: %.5f", best_val)


if __name__ == "__main__":
    main()
