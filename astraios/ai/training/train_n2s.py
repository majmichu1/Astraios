"""Noise2Self training for U-Net denoising model — memmap compatible.

Self-supervised denoising: mask random pixels, train the network to predict them
from surrounding context. Since noise is random, the network learns the signal.

No clean reference images needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from astraios.ai.models.unet import UNet

log = logging.getLogger(__name__)

# Training config
PATCH_SIZE = 256
MASK_RATIO = 0.15  # Fraction of pixels to mask
BATCH_SIZE = 16
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0  # Must be 0 for memmap compatibility with multiprocessing


class MemmapDataset(Dataset):
    """Dataset wrapper for numpy memmap arrays. Zero-copy reads from disk."""

    def __init__(self, memmap_array: np.memmap, augmentation: bool = True):
        self.data = memmap_array
        self.augmentation = augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch = self.data[idx]  # (H, W) float32

        # Data augmentation
        if self.augmentation:
            # Random flip
            if np.random.random() > 0.5:
                patch = np.flip(patch, axis=0).copy()
            if np.random.random() > 0.5:
                patch = np.flip(patch, axis=1).copy()
            # Random rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            patch = np.rot90(patch, k).copy()
            # Brightness jitter
            jitter = np.random.uniform(0.8, 1.2)
            patch = np.clip(patch * jitter, 0, 1).astype(np.float32)

        return torch.from_numpy(patch).unsqueeze(0)  # (1, H, W)


def create_masked_input(patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create masked input for Noise2Self.

    Parameters
    ----------
    patches : torch.Tensor
        Shape (N, 1, H, W), values in [0, 1].

    Returns
    -------
    (masked, target, mask) : tuple of torch.Tensor
        masked: input with masked pixels zeroed
        target: original values at masked locations
        mask: binary mask of masked pixels
    """
    n, c, h, w = patches.shape
    mask = torch.rand(n, c, h, w, device=patches.device) < MASK_RATIO
    target = patches.clone()
    masked = patches.clone()
    masked[mask] = 0.0
    return masked, target, mask


def train_model(
    train_memmap: np.memmap,
    val_memmap: np.memmap,
    output_dir: Path,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    device: str = "auto",
) -> Path:
    """Train Noise2Self model.

    Parameters
    ----------
    train_memmap : np.memmap
        Shape (N, H, W), values in [0, 1].
    val_memmap : np.memmap
        Shape (M, H, W), values in [0, 1].
    output_dir : Path
        Directory to save checkpoints and final model.
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Batch size.
    lr : float
        Learning rate.
    device : str
        "auto", "cuda", "mps", or "cpu".

    Returns
    -------
    Path to the best model checkpoint.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device selection
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)

    log.info("Training on device: %s", device)
    log.info("Dataset: %d train patches, %d val patches (memmap)", len(train_memmap), len(val_memmap))

    # Model — balanced architecture (31M params for high quality)
    model = UNet(in_channels=1, out_channels=1, base_features=32, depth=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: U-Net (base=32, depth=4), parameters: %d", n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    early_stop_patience = 5
    early_stop_counter = 0

    # Datasets
    train_dataset = MemmapDataset(train_memmap, augmentation=True)
    val_dataset = MemmapDataset(val_memmap, augmentation=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS)

    best_val_loss = float("inf")
    best_model_path = output_dir / "best_n2s_model.pt"

    log.info("Starting training: epochs=%d, batch=%d, lr=%.0e, augmentation=True (flip/rotate/brightness)",
             num_epochs, batch_size, lr)

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            masked, target, mask = create_masked_input(batch)

            pred = model(masked)
            loss = ((pred - target) ** 2)[mask].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                masked, target, mask = create_masked_input(batch)
                pred = model(masked)
                loss = ((pred - target) ** 2)[mask].mean()
                val_loss += loss.item()
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)

        # Logging
        if epoch % 5 == 0 or epoch == 1:
            log.info(
                "Epoch %3d/%d | Train loss: %.6f | Val loss: %.6f | LR: %.0e",
                epoch, num_epochs, avg_train_loss, avg_val_loss,
                optimizer.param_groups[0]["lr"],
            )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": {
                        "in_channels": 1,
                        "out_channels": 1,
                        "base_features": 32,
                        "depth": 4,
                        "patch_size": PATCH_SIZE,
                    },
                },
                best_model_path,
            )
            log.info("  → New best model saved (val_loss=%.6f)", best_val_loss)
        else:
            early_stop_counter += 1

        # Early stopping
        if early_stop_counter >= early_stop_patience:
            log.info("Early stopping after %d epochs without improvement", early_stop_counter)
            break

    log.info("Training complete. Best val loss: %.6f", best_val_loss)
    log.info("Best model saved to: %s", best_model_path)

    return best_model_path


def export_for_inference(
    checkpoint_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Export trained checkpoint to inference-ready .pt file."""
    if output_path is None:
        output_path = checkpoint_path.parent / "cosmica_denoise_n2s_v1.pt"

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    model = UNet(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        base_features=config["base_features"],
        depth=config["depth"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    torch.save(model.state_dict(), output_path)
    log.info("Exported inference model to: %s", output_path)

    # Print model info
    import hashlib
    model_bytes = output_path.read_bytes()
    sha256 = hashlib.sha256(model_bytes).hexdigest()
    size_mb = output_path.stat().st_size / (1024 * 1024)

    log.info("Model size: %.1f MB, SHA-256: %s", size_mb, sha256)
    return output_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train Noise2Self denoising model")
    parser.add_argument("--input", type=Path, default=Path("./astro_data"),
                        help="Directory with FITS files")
    parser.add_argument("--output", type=Path, default=Path("./astraios/ai/models"),
                        help="Output directory for model")
    parser.add_argument("--data", type=Path, default=Path("./training_data"),
                        help="Directory with patches memmap")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Load memmap
    from astraios.ai.training.prepare_data import load_dataset
    memmap_path = args.data / "patches.dat"
    if not memmap_path.exists():
        log.error("No memmap found at %s. Run prepare_data.py first.", memmap_path)
        exit(1)

    log.info("Loading patches from memmap...")
    train_mm, val_mm = load_dataset(memmap_path)
    log.info("Loaded: %d train, %d val patches", len(train_mm), len(val_mm))

    # Train
    best_model = train_model(
        train_mm, val_mm,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    # Export
    export_for_inference(best_model)
