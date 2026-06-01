#!/usr/bin/env python
"""Quick script to start Noise2Self training.

Usage:
    python scripts/train_denoise_model.py [--input DIR] [--epochs N]

If you have FITS files in a directory, point --input to it.
Otherwise it will try to download sample data from MAST.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cosmica.ai.training.train_n2s import train_model, export_for_inference
from cosmica.ai.training.prepare_data import prepare_dataset, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

log = logging.getLogger(__name__)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train N2S denoise model")
    parser.add_argument("--input", type=Path, default=Path("./astro_data"),
                        help="Directory with FITS files")
    parser.add_argument("--output", type=Path, default=Path("./cosmica/ai/models"),
                        help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-patches", type=int, default=500)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  Cosmica Noise2Self Denoise Model Training")
    log.info("=" * 60)

    # Prepare data
    log.info("Preparing training data from %s...", args.input)
    memmap_path = prepare_dataset(args.input, args.output.parent / "training_data")
    train_patches, val_patches = load_dataset(memmap_path, split=0.8)

    if len(train_patches) == 0:
        log.error("""
No training data available!

You need FITS files for training. Options:

1. Use your own FITS files:
   python scripts/train_denoise_model.py --input /path/to/your/fits/

2. Download from MAST (NASA archive):
   - Go to https://mast.stsci.edu
   - Search for any object (e.g. "M31")
   - Download FITS files
   - Point --input to the download directory

3. Use sample data:
   mkdir -p astro_data
   # Place your FITS files there

Place at least 10-50 FITS files in the input directory.
""")
        sys.exit(1)

    log.info("Training data: %d train, %d val patches", len(train_patches), len(val_patches))

    # Train
    best_model = train_model(
        train_patches, val_patches,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    # Export
    export_for_inference(best_model, args.output / "cosmica_denoise_n2s_v1.pt")

    log.info("")
    log.info("Training complete! Model ready for use in Cosmica.")
    log.info("Copy the model file to: cosmica/ai/models/cosmica_denoise_n2s_v1.pt")
