"""Calibration Pipeline — master frame creation and light calibration.

GPU-accelerated via the device manager.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from astraios.core.device_manager import get_device_manager
from astraios.core.image_io import FrameType, ImageData, load_image, save_fits

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

# Above this resident size for the whole frame stack, master creation switches
# to a bounded-memory tiled median (on-disk memmap) instead of loading every
# frame into RAM at once. 50 x 65MP frames would otherwise need ~39GB.
_MASTER_MEM_BUDGET = 1_500_000_000  # ~1.5 GB
_MASTER_TILE_BYTES = 512_000_000    # ~512 MB working set per median row-band


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class CalibrationResult:
    master: ImageData
    n_frames: int
    method: str
    rejected: int = 0


def create_master_bias(
    bias_paths: list[Path],
    progress: ProgressCallback = _noop_progress,
) -> CalibrationResult:
    """Create a master bias frame by median-combining bias frames."""
    return _create_master(
        paths=bias_paths,
        frame_type=FrameType.MASTER_BIAS,
        method="median",
        label="bias",
        progress=progress,
    )


def create_master_dark(
    dark_paths: list[Path],
    master_bias: ImageData | None = None,
    progress: ProgressCallback = _noop_progress,
) -> CalibrationResult:
    """Create a master dark frame. Optionally subtract master bias first."""
    return _create_master(
        paths=dark_paths,
        frame_type=FrameType.MASTER_DARK,
        method="median",
        label="dark",
        subtract=master_bias,
        progress=progress,
    )


def create_master_flat(
    flat_paths: list[Path],
    master_bias: ImageData | None = None,
    master_dark: ImageData | None = None,
    progress: ProgressCallback = _noop_progress,
) -> CalibrationResult:
    """Create a master flat frame. Optionally subtract bias and/or dark."""
    result = _create_master(
        paths=flat_paths,
        frame_type=FrameType.MASTER_FLAT,
        method="median",
        label="flat",
        subtract=master_bias,
        subtract2=master_dark,
        progress=progress,
    )
    # Normalize flat to mean = 1.0
    mean_val = float(np.mean(result.master.data))
    if mean_val > 0:
        result.master.data /= mean_val
    return result


def _create_master(
    paths: list[Path],
    frame_type: FrameType,
    method: str,
    label: str,
    subtract: ImageData | None = None,
    subtract2: ImageData | None = None,
    progress: ProgressCallback = _noop_progress,
) -> CalibrationResult:
    """Generic master frame creation."""
    dm = get_device_manager()
    n = len(paths)
    if n == 0:
        raise ValueError(f"No {label} frames provided")

    progress(0.0, f"Loading {label} frames...")
    log.info("Creating master %s from %d frames", label, n)

    # Load first frame to get shape
    first = load_image(paths[0])
    shape = first.data.shape

    # Median needs every frame's pixels at once. Holding the whole stack in RAM
    # (n * frame_bytes) is fastest, but OOMs for many large frames (50 x 65MP
    # ~= 39GB). Above a budget, compute the median in row-bands from an on-disk
    # memmap so RAM stays bounded to one frame (load) + one band (median).
    frame_bytes = int(np.prod(shape)) * 4  # float32
    use_tiled = method == "median" and n > 2 and n * frame_bytes > _MASTER_MEM_BUDGET

    if use_tiled:
        log.info(
            "Master %s: %d frames x %.0f MB exceeds RAM budget; using tiled "
            "low-memory median", label, n, frame_bytes / 1e6,
        )
        master_data, n = _master_data_tiled(
            paths, shape, first, subtract, subtract2, label, progress
        )
        master_data = master_data.astype(np.float32)
    else:
        # In-memory path — the whole stack fits the budget (fast).
        stack = np.zeros((n, *shape), dtype=np.float32)
        stack[0] = first.data

        for i in range(1, n):
            progress(0.1 + 0.5 * (i / n), f"Loading {label} {i + 1}/{n}")
            img = load_image(paths[i])
            if img.data.shape != shape:
                log.warning(
                    "Frame %s shape mismatch: %s vs %s, skipping",
                    paths[i], img.data.shape, shape,
                )
                continue
            stack[i] = img.data

        # Subtract calibration frames if provided
        if subtract is not None:
            progress(0.65, f"Subtracting calibration from {label} frames...")
            sub_data = subtract.data
            if sub_data.shape != shape:
                log.warning("Subtraction frame shape mismatch, skipping subtraction")
            else:
                stack -= sub_data[np.newaxis, ...]

        if subtract2 is not None:
            sub_data = subtract2.data
            if sub_data.shape != shape:
                log.warning("Subtraction frame 2 shape mismatch, skipping subtraction")
            else:
                stack -= sub_data[np.newaxis, ...]

        progress(0.7, f"Computing {method} of {label} stack...")

        if dm.is_gpu and method == "median":
            # GPU median over the stack axis. Use quantile(0.5), not
            # torch.median: torch.median returns the lower of the two middle
            # values for an even frame count, while numpy (the CPU and tiled
            # paths) averages them. quantile(0.5) matches numpy, so the master
            # is identical regardless of which path computed it.
            try:
                t_stack = torch.from_numpy(stack).to(dm.device)
                master_data = torch.quantile(t_stack, 0.5, dim=0)
                master_data = dm.to_cpu(master_data).numpy()
                del t_stack
            except RuntimeError:
                # Fall back to CPU if GPU OOM
                log.warning("GPU OOM during %s stacking, falling back to CPU", label)
                master_data = np.median(stack, axis=0)
        else:
            master_data = np.median(stack, axis=0)

        master_data = master_data.astype(np.float32)
        del stack

    progress(1.0, f"Master {label} complete")
    log.info("Master %s created: %s", label, master_data.shape)

    master = ImageData(
        data=master_data,
        header=first.header.copy(),
        frame_type=frame_type,
    )
    master.header["IMAGETYP"] = f"master_{label}"
    master.header["NCOMBINE"] = n

    return CalibrationResult(master=master, n_frames=n, method=method)


def _master_data_tiled(
    paths: list[Path],
    shape: tuple[int, ...],
    first: ImageData,
    subtract: ImageData | None,
    subtract2: ImageData | None,
    label: str,
    progress: ProgressCallback,
) -> tuple[np.ndarray, int]:
    """Median-combine frames with bounded memory via an on-disk memmap.

    Each frame is loaded once and written to a temporary memmapped array (RAM
    holds only one frame during the load), then the median is taken in row-bands
    along the height axis (RAM holds only ``n_valid * band`` pixels at a time).
    Subtracting a constant master before the median is equivalent to after, so
    we subtract per-frame on load. Returns ``(master_data, n_valid)``.
    """
    n = len(paths)
    sub = subtract.data if (subtract is not None and subtract.data.shape == shape) else None
    if subtract is not None and sub is None:
        log.warning("Subtraction frame shape mismatch, skipping subtraction")
    sub2 = subtract2.data if (subtract2 is not None and subtract2.data.shape == shape) else None

    # Co-locate the temp file with the source frames: /tmp is often tmpfs
    # (RAM-backed), which would defeat the point of spilling to disk.
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix="astraios_master_", suffix=".dat", dir=str(Path(paths[0]).parent)
        )
    except OSError:
        fd, tmp_path = tempfile.mkstemp(prefix="astraios_master_", suffix=".dat")
    os.close(fd)

    mm = None
    try:
        mm = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(n, *shape))

        def _prep(arr: np.ndarray) -> np.ndarray:
            d = arr.astype(np.float32, copy=False)
            if sub is not None:
                d = d - sub
            if sub2 is not None:
                d = d - sub2
            return d

        mm[0] = _prep(first.data)
        valid = 1
        for i in range(1, n):
            progress(0.1 + 0.5 * (i / n), f"Loading {label} {i + 1}/{n}")
            img = load_image(paths[i])
            if img.data.shape != shape:
                log.warning(
                    "Frame %s shape mismatch: %s vs %s, skipping",
                    paths[i], img.data.shape, shape,
                )
                continue
            mm[valid] = _prep(img.data)
            valid += 1
            del img
        mm.flush()

        progress(0.7, f"Computing median of {label} stack (tiled)...")
        out = np.empty(shape, dtype=np.float32)
        h = shape[-2]
        elems_per_row = int(np.prod(shape)) // h  # per frame, per height row
        band = max(1, min(h, _MASTER_TILE_BYTES // max(valid * elems_per_row * 4, 1)))
        for r0 in range(0, h, band):
            r1 = min(h, r0 + band)
            # Ellipsis indexes any leading channel axis: (n,H,W) or (n,C,H,W).
            out[..., r0:r1, :] = np.median(mm[:valid, ..., r0:r1, :], axis=0)
        return out, valid

    finally:
        if mm is not None:
            del mm  # close the memmap before unlinking
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _read_exptime(header: dict) -> float | None:
    """Read exposure time from FITS header (seconds)."""
    for key in ("EXPTIME", "EXPOSURE"):
        val = header.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def _scaled_master_dark(
    master_dark: ImageData,
    light_header: dict,
) -> np.ndarray:
    """Scale master dark to match light exposure when EXPTIME is available."""
    dark_data = master_dark.data
    light_exp = _read_exptime(light_header)
    dark_exp = _read_exptime(master_dark.header)
    if light_exp is not None and dark_exp is not None and dark_exp > 0:
        return dark_data * (light_exp / dark_exp)
    if light_exp is not None and dark_exp is None:
        log.warning("Master dark missing EXPTIME; subtracting unscaled dark")
    light_temp = light_header.get("CCD-TEMP")
    dark_temp = master_dark.header.get("CCD-TEMP")
    if light_temp is not None and dark_temp is not None:
        try:
            if abs(float(light_temp) - float(dark_temp)) > 2.0:
                log.warning(
                    "CCD-TEMP differs by >2°C (light=%s, dark=%s); dark scaling is exposure-only",
                    light_temp,
                    dark_temp,
                )
        except (TypeError, ValueError):
            pass
    return dark_data


def calibrate_light(
    light: ImageData,
    master_bias: ImageData | None = None,
    master_dark: ImageData | None = None,
    master_flat: ImageData | None = None,
) -> ImageData:
    """Apply calibration to a single light frame (GPU-accelerated).

    Order: subtract bias, subtract dark, divide by flat.
    """
    dm = get_device_manager()

    if dm.is_gpu:
        return _calibrate_light_gpu(light, master_bias, master_dark, master_flat, dm)
    else:
        return _calibrate_light_cpu(light, master_bias, master_dark, master_flat)


@torch.no_grad()
def _calibrate_light_gpu(
    light: ImageData,
    master_bias: ImageData | None,
    master_dark: ImageData | None,
    master_flat: ImageData | None,
    dm,
) -> ImageData:
    """GPU-accelerated calibration of a single light frame."""
    t_data = dm.from_numpy(light.data)

    if master_bias is not None and master_bias.data.shape == light.data.shape:
        t_bias = dm.from_numpy(master_bias.data)
        t_data = t_data - t_bias

    if master_dark is not None and master_dark.data.shape == light.data.shape:
        dark_scaled = _scaled_master_dark(master_dark, light.header)
        t_dark = dm.from_numpy(dark_scaled)
        t_data = t_data - t_dark

    if master_flat is not None and master_flat.data.shape == light.data.shape:
        t_flat = dm.from_numpy(master_flat.data)
        t_flat_safe = torch.where(t_flat > 0.001, t_flat, torch.tensor(1.0, device=t_flat.device))
        t_data = t_data / t_flat_safe

    data = t_data.cpu().numpy().astype(np.float32)
    return ImageData(
        data=data,
        header=light.header.copy(),
        file_path=light.file_path,
        frame_type=FrameType.LIGHT,
    )


def _calibrate_light_cpu(
    light: ImageData,
    master_bias: ImageData | None,
    master_dark: ImageData | None,
    master_flat: ImageData | None,
) -> ImageData:
    """CPU calibration of a single light frame (fallback)."""
    data = light.data.copy()

    if master_bias is not None and master_bias.data.shape == data.shape:
        data -= master_bias.data

    if master_dark is not None and master_dark.data.shape == data.shape:
        data -= _scaled_master_dark(master_dark, light.header)

    if master_flat is not None and master_flat.data.shape == data.shape:
        flat = master_flat.data
        flat_safe = np.where(flat > 0.001, flat, 1.0)
        data /= flat_safe

    data = data.astype(np.float32)

    return ImageData(
        data=data,
        header=light.header.copy(),
        file_path=light.file_path,
        frame_type=FrameType.LIGHT,
    )


def calibrate_lights_batch(
    light_paths: list[Path],
    master_bias: ImageData | None = None,
    master_dark: ImageData | None = None,
    master_flat: ImageData | None = None,
    output_dir: Path | None = None,
    progress: ProgressCallback = _noop_progress,
) -> list[ImageData]:
    """Calibrate a batch of light frames.

    When output_dir is given, each calibrated frame is also saved to disk and
    the returned ImageData.file_path points at the saved calibrated file (not
    the raw source), so path-based consumers read calibrated pixels.
    """
    results = []
    n = len(light_paths)

    for i, path in enumerate(light_paths):
        progress(i / n, f"Calibrating light {i + 1}/{n}")
        light = load_image(path)
        calibrated = calibrate_light(light, master_bias, master_dark, master_flat)

        if output_dir is not None:
            out_path = Path(output_dir) / f"cal_{Path(path).stem}.fits"
            save_fits(calibrated, out_path)
            calibrated.file_path = out_path

        results.append(calibrated)

    progress(1.0, "Calibration complete")
    return results


def calibrate_lights_to_disk(
    light_paths: list[Path],
    output_dir: Path,
    master_bias: ImageData | None = None,
    master_dark: ImageData | None = None,
    master_flat: ImageData | None = None,
    progress: ProgressCallback = _noop_progress,
) -> list[Path]:
    """Calibrate light frames streaming to disk, one frame in RAM at a time.

    Memory-safe for arbitrarily large datasets: each light is loaded,
    calibrated, saved to output_dir as ``cal_<stem>.fits`` and freed before the
    next one is touched. The written files are bit-identical to what
    calibrate_lights_batch would produce (float32 FITS round-trips exactly).

    Returns the list of calibrated file paths, in input order. Frames that
    fail to load or save are skipped with a warning.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    n = len(light_paths)

    for i, path in enumerate(light_paths):
        progress(i / n, f"Calibrating light {i + 1}/{n}")
        try:
            light = load_image(path)
            calibrated = calibrate_light(light, master_bias, master_dark, master_flat)
            out_path = output_dir / f"cal_{Path(path).stem}.fits"
            save_fits(calibrated, out_path)
            out_paths.append(out_path)
        except Exception as exc:
            log.warning("Failed to calibrate %s: %s", path, exc)
        finally:
            light = None
            calibrated = None

    progress(1.0, "Calibration complete")
    return out_paths
