"""Calibration Pipeline — master frame creation and light calibration.

GPU-accelerated via the device manager.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from cosmica.core.device_manager import get_device_manager
from cosmica.core.image_io import FrameType, ImageData, load_image, save_fits

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


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

    # Process in batches to fit GPU memory
    frame_bytes = np.prod(shape) * 4  # float32
    batch_size = dm.optimal_batch_size(frame_bytes * n) if dm.is_gpu else n
    batch_size = max(1, min(batch_size, n))

    # For median, we need all frames in memory — use CPU if necessary
    # Stack all frames
    stack = np.zeros((n, *shape), dtype=np.float32)
    stack[0] = first.data

    for i in range(1, n):
        progress(0.1 + 0.5 * (i / n), f"Loading {label} {i + 1}/{n}")
        img = load_image(paths[i])
        if img.data.shape != shape:
            log.warning("Frame %s shape mismatch: %s vs %s, skipping", paths[i], img.data.shape, shape)
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
        if sub_data.shape == shape:
            stack -= sub_data[np.newaxis, ...]

    progress(0.7, f"Computing {method} of {label} stack...")

    if dm.is_gpu and method == "median":
        # GPU median: process in chunks along the stack axis
        try:
            t_stack = torch.from_numpy(stack).to(dm.device)
            master_data = torch.median(t_stack, dim=0).values
            master_data = dm.to_cpu(master_data).numpy()
        except RuntimeError:
            # Fall back to CPU if GPU OOM
            log.warning("GPU OOM during %s stacking, falling back to CPU", label)
            master_data = np.median(stack, axis=0)
    else:
        master_data = np.median(stack, axis=0)

    master_data = master_data.astype(np.float32)

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
    """Calibrate a batch of light frames."""
    results = []
    n = len(light_paths)

    for i, path in enumerate(light_paths):
        progress(i / n, f"Calibrating light {i + 1}/{n}")
        light = load_image(path)
        calibrated = calibrate_light(light, master_bias, master_dark, master_flat)

        if output_dir is not None:
            out_path = output_dir / f"cal_{path.stem}.fits"
            save_fits(calibrated, out_path)

        results.append(calibrated)

    progress(1.0, "Calibration complete")
    return results
