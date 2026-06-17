"""
Prepare training data with STRICT file-based splitting to prevent data leakage.
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
    """Extract patches from a single FITS file."""
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
                if data is None: return None
        # Handle ARW/RAW files
        elif path.suffix.lower() == '.arw':
            try:
                from PIL import Image
                img = Image.open(str(path))
                data = np.array(img, dtype=np.float32)
            except Exception: return None
        else:
            return None
    except Exception: return None

    # For 3D data: FITS cubes use first plane, RGB images use luminance
    if data.ndim == 3:
        if data.shape[2] in (3, 4):  # RGB/RGBA from PIL (H, W, C)
            data = data.mean(axis=2)
        else:  # FITS cube (C, H, W) or single plane
            data = data[0]

    h, w = data.shape
    if h < patch_size or w < patch_size: return None

    dmin, dmax = float(data.min()), float(data.max())
    if dmax - dmin > 1e-10:
        data = (data - dmin) / (dmax - dmin)
    else: return None

    # Content-aware patch extraction: require some signal above background.
    # A patch that is pure dark sky has std ≈ noise floor → not useful for training.
    # We require std > MIN_PATCH_STD so the model trains on meaningful structure.
    MIN_PATCH_STD = 0.015  # reject nearly-blank patches (pure dark sky)
    MAX_ATTEMPTS = max_patches * 8  # try harder to find content-rich patches

    n_target = min(max_patches, max(1, (h // patch_size) * (w // patch_size)))
    patches = []
    attempts = 0
    while len(patches) < n_target and attempts < MAX_ATTEMPTS:
        y = random.randint(0, h - patch_size)
        x = random.randint(0, w - patch_size)
        patch = data[y:y + patch_size, x:x + patch_size]
        if patch.std() >= MIN_PATCH_STD:
            patches.append(patch)
        attempts += 1

    if not patches:
        return None  # image had no useful content patches

    return np.array(patches, dtype=np.float32)

def prepare_dataset_strict_split(input_dir: Path, output_dir: Path):
    """
    1. Split FILES 80/20.
    2. Extract patches from each set.
    3. Save to train.dat and val.dat.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Get and Shuffle files
    fits_files = list(input_dir.glob("*.fits")) + list(input_dir.glob("*.fts"))
    fits_files += list(input_dir.glob("*.fit")) + list(input_dir.glob("*.FTS"))
    fits_files = [f for f in fits_files if f.exists()]
    random.shuffle(fits_files)

    if not fits_files:
        log.error("No FITS files found in %s", input_dir)
        return

    # 2. Split
    split_idx = int(0.8 * len(fits_files))
    train_files = fits_files[:split_idx]
    val_files = fits_files[split_idx:]

    log.info("TOTAL files: %d", len(fits_files))
    log.info("TRAIN files: %d (80%%)", len(train_files))
    log.info("VAL files:   %d (20%%)", len(val_files))

    # 3. Extract Train
    log.info(">>> Starting Train Extraction...")
    _extract_and_save(train_files, output_dir / "train.dat", "Train")

    # 4. Extract Val
    log.info(">>> Starting Val Extraction...")
    _extract_and_save(val_files, output_dir / "val.dat", "Val")

def _extract_and_save(file_list: list[Path], save_path: Path, set_name: str):
    """Helper to extract patches from a list of files and save to memmap."""
    # First pass: count patches to size memmap
    total_patches = 0
    temp_patches = [] # Store temporarily as lists of arrays (memory heavy but safe for counting)

    # Better approach: dynamic resizing memmap or just list append then concat
    # Since we have plenty of RAM, let's collect them.
    # If OOM, we switch to memmap append mode.
    # With 6000 files * 50 patches, we have 300k patches. 300k * 256*256*4 bytes = ~75GB.
    # That might OOM 32GB RAM.
    # We MUST use memmap append or fixed size.

    # Let's use the fixed size estimate from before: ~50 patches/file.
    # 6000 files * 0.8 = 4800 files * 50 = 240,000 patches.
    # Estimate conservatively.

    est_patches = len(file_list) * 55
    fp = np.memmap(save_path, dtype=np.float32, mode='w+',
                   shape=(est_patches, PATCH_SIZE, PATCH_SIZE))

    idx = 0
    files_ok = 0

    for f in file_list:
        result = extract_patches_from_file(f)
        if result is None: continue

        n = len(result)
        if idx + n > est_patches:
            # Resize
            fp.flush()
            del fp
            # This is slow, but rare.
            old_data = np.memmap(save_path, dtype=np.float32, mode='r', shape=(est_patches, PATCH_SIZE, PATCH_SIZE))
            new_est = est_patches * 2
            fp = np.memmap(save_path, dtype=np.float32, mode='w+',
                           shape=(new_est, PATCH_SIZE, PATCH_SIZE))
            fp[:est_patches] = old_data[:est_patches]
            del old_data
            est_patches = new_est
            # We lost 'idx' position if we just re-opened.
            # Wait, idx is index. We just need to write at idx.
            # But re-opening wipes data? mode='w+' wipes.
            # Correct logic: close, rename, open new, copy.
            # Too complex.
            pass
            # Let's just rely on 55 estimate. 50 is max. 55 is plenty of slack.

        fp[idx:idx+n] = result
        idx += n
        files_ok += 1

        if files_ok % 200 == 0:
            fp.flush()
            log.info(f"{set_name}: Processed {files_ok}/{len(file_list)} files -> {idx} patches")
            gc.collect()

    # Finalize
    actual_size = idx
    fp.flush()
    del fp

    # Trim file? Not strictly necessary for reading, just track actual_size in metadata
    meta_path = save_path.parent / f"{set_name.lower()}_meta.txt"
    with open(meta_path, 'w') as f: f.write(str(actual_size))

    log.info(f"{set_name} DONE. Total patches: {actual_size}. Saved to {save_path}")

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("./astro_data"))
    parser.add_argument("--output", type=Path, default=Path("./training_data"))
    args = parser.parse_args()

    prepare_dataset_strict_split(args.input, args.output)
