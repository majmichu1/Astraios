"""Prepare training data from FITS files — streaming + memmap for training.

Saves patches as a single memmap file. Training reads directly from memmap.
"""

from __future__ import annotations

import gc
import logging
import random
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

PATCH_SIZE = 256
MAX_PATCHES_PER_IMAGE = 50


def extract_patches_from_file(path: Path, patch_size: int = PATCH_SIZE,
                               max_patches: int = MAX_PATCHES_PER_IMAGE) -> np.ndarray | None:
    """Extract patches from a single FITS/ARW file."""
    try:
        # Handle FITS files
        if path.suffix.lower() in ('.fit', '.fits', '.fts'):
            from astropy.io import fits
            with fits.open(str(path), memmap=False) as hdul:
                data = None
                for hdu in hdul:
                    if hdu.data is not None and hdu.data.ndim >= 2:
                        data = hdu.data.astype(np.float32)
                        break
                if data is None:
                    return None
        # Handle ARW/RAW files (use PIL if available)
        elif path.suffix.lower() == '.arw':
            try:
                from PIL import Image
                img = Image.open(str(path))
                data = np.array(img, dtype=np.float32)
            except Exception:
                return None
        else:
            return None
    except Exception:
        return None

    # For 3D data: FITS cubes use first plane, RGB images use luminance
    if data.ndim == 3:
        if data.shape[2] in (3, 4):  # RGB/RGBA from PIL (H, W, C)
            data = data.mean(axis=2)
        else:  # FITS cube (C, H, W) or single plane
            data = data[0]

    h, w = data.shape
    if h < patch_size or w < patch_size:
        return None

    # Normalize to [0, 1]
    dmin, dmax = float(data.min()), float(data.max())
    if dmax - dmin > 1e-10:
        data = (data - dmin) / (dmax - dmin)
    else:
        return None

    # Extract random patches
    n_patches = min(max_patches, max(1, (h // patch_size) * (w // patch_size)))
    patches = []
    for _ in range(n_patches):
        y = random.randint(0, h - patch_size)
        x = random.randint(0, w - patch_size)
        patches.append(data[y:y + patch_size, x:x + patch_size])

    return np.array(patches, dtype=np.float32)


def prepare_dataset(input_dir: Path, output_dir: Path) -> Path:
    """Prepare training data as a memory-mapped file.

    Returns path to the memmap file. Training should read from this directly.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    memmap_path = output_dir / "patches.dat"
    meta_path = output_dir / "meta.txt"

    # Check if already done
    if memmap_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            total = int(f.read().strip())
        log.info("Using existing memmap: %d patches at %s", total, memmap_path)
        return memmap_path

    fits_files = list(input_dir.glob("*.fits")) + list(input_dir.glob("*.fts")) + \
                 list(input_dir.glob("*.ARW")) + list(input_dir.glob("*.arw"))
    fits_files = [f for f in fits_files if f.exists()]
    random.shuffle(fits_files)

    log.info("Found %d files", len(fits_files))

    # Phase 1: Count total patches
    total_patches = 0
    for f in fits_files:
        result = extract_patches_from_file(f)
        if result is not None:
            total_patches += len(result)

    log.info("Total patches to extract: %d", total_patches)

    # Phase 2: Create memmap and fill
    fp = np.memmap(memmap_path, dtype=np.float32, mode='w+',
                   shape=(total_patches, PATCH_SIZE, PATCH_SIZE))

    idx = 0
    files_processed = 0

    for f in fits_files:
        result = extract_patches_from_file(f)
        if result is None:
            continue

        n = len(result)
        fp[idx:idx+n] = result
        idx += n
        files_processed += 1

        if files_processed % 500 == 0:
            fp.flush()
            log.info("Processed %d/%d files → %d/%d patches",
                     files_processed, len(fits_files), idx, total_patches)
            gc.collect()

    fp.flush()
    actual_size = idx

    # Save metadata
    with open(meta_path, 'w') as f:
        f.write(str(actual_size))

    log.info("Done: %d patches from %d files saved to %s",
             actual_size, files_processed, memmap_path)

    # Clean up memmap reference so file can be used by training
    del fp

    return memmap_path


def load_dataset(memmap_path: Path, split: float = 0.8):
    """Load training/validation splits from memmap.

    Returns (train_memmap, val_memmap) as numpy memmaps.
    """
    with open(memmap_path.parent / "meta.txt") as f:
        total = int(f.read().strip())

    split_idx = int(split * total)

    train = np.memmap(memmap_path, dtype=np.float32, mode='r',
                      shape=(total, PATCH_SIZE, PATCH_SIZE))[:split_idx]
    val = np.memmap(memmap_path, dtype=np.float32, mode='r',
                    shape=(total, PATCH_SIZE, PATCH_SIZE))[split_idx:]

    log.info("Train: %d patches, Val: %d patches", len(train), len(val))
    return train, val


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Prepare N2S training data (memmap)")
    parser.add_argument("--input", type=Path, default=Path("./astro_data"))
    parser.add_argument("--output", type=Path, default=Path("./training_data"))
    args = parser.parse_args()

    memmap_path = prepare_dataset(args.input, args.output)
    train, val = load_dataset(memmap_path)
    print(f"Training patches: {train.shape}")
    print(f"Validation patches: {val.shape}")
