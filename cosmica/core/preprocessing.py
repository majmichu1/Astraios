"""Batch Preprocessing — auto-calibration, registration, and stacking.

Mirrors PixInsight's BatchPreprocessing with GPU acceleration.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cosmica.core.calibration import (
    calibrate_light,
    create_master_bias,
    create_master_dark,
    create_master_flat,
)
from cosmica.core.cosmetic import cosmetic_correction
from cosmica.core.image_io import ImageData, load_image, save_fits
from cosmica.core.stacking import StackingParams, stack_from_paths

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class CalibrationGroup:
    biases: list[Path] = field(default_factory=list)
    darks: list[Path] = field(default_factory=list)
    flats: list[Path] = field(default_factory=list)
    lights: list[Path] = field(default_factory=list)

    def has_any(self) -> bool:
        return bool(self.biases or self.darks or self.flats or self.lights)


def _read_header_key(path: Path, key: str) -> str | None:
    """Read a single FITS header key without loading pixel data."""
    try:
        from astropy.io import fits as _fits
        with _fits.open(path, memmap=False) as hdul:
            return str(hdul[0].header.get(key, ""))
    except Exception:
        return None


def _detect_frame_type(path: Path) -> str | None:
    """Detect frame type from FITS header or folder name."""
    imtype = _read_header_key(path, "IMAGETYP")
    if imtype:
        imtype = imtype.strip().upper()
        known = {"BIAS": "bias", "DARK": "dark", "FLAT": "flat", "LIGHT": "light"}
        if imtype in known:
            return known[imtype]
    folder = path.parent.name.lower()
    for tag in ("bias", "dark", "flat", "light"):
        if tag in folder:
            return tag
    return None


def _read_exptime(path: Path) -> float | None:
    """Read exposure time from FITS header (seconds)."""
    for key in ("EXPTIME", "EXPOSURE"):
        val = _read_header_key(path, key)
        if val:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def _read_filter(path: Path) -> str:
    """Read filter name from FITS header."""
    val = _read_header_key(path, "FILTER")
    if val:
        val = val.strip().upper()
        if val and val not in ("NONE", "0", ""):
            return val
    return ""


def _read_binning(path: Path) -> str:
    """Read binning from FITS header."""
    for key in ("XBINNING", "BINNING", "CCD_XBIN"):
        val = _read_header_key(path, key)
        if val:
            val = val.strip()
            if val not in ("", "0"):
                return val
    return "1"


def _read_temperature(path: Path) -> float | None:
    """Read CCD temperature from FITS header (Celsius)."""
    val = _read_header_key(path, "CCD-TEMP")
    if val:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    return None


@dataclass
class CalibrationMatch:
    """Matched calibration frames for a light frame group."""

    light_paths: list[Path]
    master_bias: ImageData | None = None
    master_dark: ImageData | None = None
    master_flat: ImageData | None = None
    filter_name: str = ""
    exptime: float = 0.0


def auto_match_calibration(
    light_paths: list[Path],
    bias_paths: list[Path] | None = None,
    dark_paths: list[Path] | None = None,
    flat_paths: list[Path] | None = None,
    progress: ProgressCallback = _noop_progress,
) -> list[CalibrationMatch]:
    """Match calibration frames to light groups automatically.

    Groups lights by filter, then matches darks by exposure time
    and flats by filter.

    Parameters
    ----------
    light_paths : list[Path]
        Light frame file paths.
    bias_paths : list[Path], optional
        Bias frame file paths.
    dark_paths : list[Path], optional
        Dark frame file paths.
    flat_paths : list[Path], optional
        Flat frame file paths.
    progress : callable
        Progress callback.

    Returns
    -------
    list[CalibrationMatch]
        One entry per unique (filter, exptime) group.
    """
    progress(0.0, "Reading light frame metadata...")

    # Group lights by filter + exptime
    light_groups: dict[tuple[str, float], list[Path]] = defaultdict(list)
    for i, p in enumerate(light_paths):
        progress(0.01 * (i / max(len(light_paths), 1)), f"Reading {p.name}...")
        filt = _read_filter(p)
        exp = _read_exptime(p) or 0.0
        light_groups[(filt, exp)].append(p)

    if not light_groups:
        log.warning("No light frames found")
        return []

    progress(0.1, "Creating master bias...")
    master_bias: ImageData | None = None
    if bias_paths and len(bias_paths) >= 3:
        try:
            result = create_master_bias(list(bias_paths))
            master_bias = result.master
            log.info("Master bias: %d frames", result.n_frames)
        except Exception as e:
            log.warning("Failed to create master bias: %s", e)

    progress(0.2, "Creating master darks...")
    dark_by_exptime: dict[float, list[Path]] = defaultdict(list)
    if dark_paths:
        for p in dark_paths:
            exp = _read_exptime(p) or 0.0
            dark_by_exptime[exp].append(p)

    master_darks: dict[float, ImageData] = {}
    for exp, paths in dark_by_exptime.items():
        if len(paths) >= 3:
            try:
                result = create_master_dark(list(paths), master_bias=master_bias)
                master_darks[exp] = result.master
                log.info("Master dark (%.1fs): %d frames", exp, result.n_frames)
            except Exception as e:
                log.warning("Failed to create master dark (%.1fs): %s", exp, e)

    progress(0.3, "Creating master flats...")
    flat_by_filter: dict[str, list[Path]] = defaultdict(list)
    if flat_paths:
        for p in flat_paths:
            filt = _read_filter(p) or "NONE"
            flat_by_filter[filt].append(p)

    master_flats: dict[str, ImageData] = {}
    for filt, paths in flat_by_filter.items():
        if len(paths) >= 5:
            try:
                result = create_master_flat(list(paths), master_bias=master_bias)
                master_flats[filt] = result.master
                log.info("Master flat (%s): %d frames", filt, result.n_frames)
            except Exception as e:
                log.warning("Failed to create master flat (%s): %s", filt, e)

    # Match each light group to its calibration frames
    matches: list[CalibrationMatch] = []
    total = len(light_groups)
    for i, ((filt, exp), lpaths) in enumerate(light_groups.items()):
        progress(0.3 + 0.1 * (i / total), f"Matching group {i + 1}/{total}...")

        # Best dark: exact EXPTIME match, then scaled
        dark = master_darks.get(exp) or master_darks.get(0.0)
        if dark is None and dark_by_exptime:
            nearest = min(dark_by_exptime, key=lambda k: abs(k - exp))
            dark = master_darks.get(nearest)

        # Flat: exact filter match
        flat = master_flats.get(filt)
        if flat is None and filt:
            flat = master_flats.get("NONE")

        matches.append(CalibrationMatch(
            light_paths=lpaths,
            master_bias=master_bias,
            master_dark=dark,
            master_flat=flat,
            filter_name=filt,
            exptime=exp,
        ))

    progress(0.4, "Calibration matching complete")
    return matches


@dataclass
class PreprocessingResult:
    calibrated_paths: list[Path]
    stacked_image: ImageData | None = None
    n_calibrated: int = 0
    n_failed: int = 0
    errors: list[str] = field(default_factory=list)


def run_preprocessing(
    light_paths: list[Path],
    bias_paths: list[Path] | None = None,
    dark_paths: list[Path] | None = None,
    flat_paths: list[Path] | None = None,
    output_dir: Path | None = None,
    calibrate: bool = True,
    register: bool = True,
    stack: bool = True,
    cosmetic: bool = True,
    stacking_params: StackingParams | None = None,
    progress: ProgressCallback = _noop_progress,
) -> PreprocessingResult:
    """Run full preprocessing pipeline.

    1. Auto-match calibration frames
    2. Create master calibration frames
    3. Calibrate lights
    4. Optionally apply cosmetic correction
    5. Register (align) frames
    6. Stack frames with rejection

    Parameters
    ----------
    light_paths : list[Path]
        Light frame file paths.
    bias_paths, dark_paths, flat_paths : list[Path], optional
        Calibration frame paths.
    output_dir : Path, optional
        Output directory for calibrated files and result.
    calibrate, register, stack : bool
        Enable/disable pipeline stages.
    cosmetic : bool
        Apply hot/cold pixel correction after calibration.
    stacking_params : StackingParams, optional
        Stacking configuration.
    progress : callable
        Progress callback.

    Returns
    -------
    PreprocessingResult
        Pipeline results.
    """
    if stacking_params is None:
        stacking_params = StackingParams()

    errors: list[str] = []

    # Step 1: Auto-match calibration
    progress(0.0, "Matching calibration frames...")
    matches = auto_match_calibration(
        light_paths, bias_paths, dark_paths, flat_paths,
        progress=progress,
    )
    if not matches:
        return PreprocessingResult(
            calibrated_paths=[], stacked_image=None,
            n_failed=len(light_paths),
            errors=["No light frames found after matching"],
        )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    calibrated_paths: list[Path] = []

    if calibrate:
        progress(0.4, "Calibrating light frames...")
        for match_idx, match in enumerate(matches):
            for light_idx, lpath in enumerate(match.light_paths):
                try:
                    frac = (
                        0.4 + 0.2 * (match_idx + light_idx / max(len(match.light_paths), 1))
                        / max(len(matches), 1)
                    )
                    progress(frac, f"Calibrating {lpath.name}...")
                    light = load_image(lpath)

                    # Calibrate
                    cal = calibrate_light(
                        light,
                        master_bias=match.master_bias,
                        master_dark=match.master_dark,
                        master_flat=match.master_flat,
                    )

                    # Cosmetic correction
                    if cosmetic:
                        cal_result = cosmetic_correction(cal.data)
                        cal = ImageData(
                            data=cal_result.data,
                            header=cal.header,
                            file_path=cal.file_path,
                            frame_type=cal.frame_type,
                        )

                    # Save calibrated frame
                    if output_dir:
                        cal_dir = output_dir / "calibrated"
                        cal_dir.mkdir(parents=True, exist_ok=True)
                        out_path = cal_dir / f"cal_{lpath.stem}.fits"
                        save_fits(cal, out_path)
                        calibrated_paths.append(out_path)
                    else:
                        calibrated_paths.append(lpath)

                except Exception as e:
                    log.warning("Failed to calibrate %s: %s", lpath.name, e)
                    errors.append(f"{lpath.name}: {e}")

    # Step 2-3: Register and stack
    stacked: ImageData | None = None
    if stack and calibrated_paths:
        progress(0.6, "Registering and stacking frames...")
        try:
            result = stack_from_paths(
                calibrated_paths if calibrate else light_paths,
                params=stacking_params,
                progress=lambda f, m: progress(0.6 + 0.35 * f, m),
            )
            stacked = result.image

            if output_dir and stacked is not None:
                out_path = output_dir / "stacked_result.fits"
                save_fits(stacked, out_path)
                log.info("Saved stacked result: %s", out_path)
        except Exception as e:
            log.warning("Stacking failed: %s", e)
            errors.append(f"Stacking failed: {e}")

    progress(1.0, "Preprocessing complete")

    return PreprocessingResult(
        calibrated_paths=calibrated_paths,
        stacked_image=stacked,
        n_calibrated=len(calibrated_paths),
        n_failed=len(errors),
        errors=errors,
    )


def scan_folder_for_frames(
    folder: Path,
    recursive: bool = True,
) -> CalibrationGroup:
    """Scan a folder structure and auto-detect calibration frame types.

    Uses standard astrophotography folder naming:
      /project/lights/    → light frames
      /project/darks/     → dark frames
      /project/flats/     → flat frames
      /project/biases/    → bias frames

    If subfolders don't exist, all frames go into ``lights``.
    """
    group = CalibrationGroup()

    if not folder.exists():
        return group

    search_fn = folder.rglob if recursive else folder.glob

    exts = {".fit", ".fits", ".fts", ".xisf"}
    all_fits = [p for p in search_fn("*") if p.suffix.lower() in exts]

    for path in all_fits:
        ftype = _detect_frame_type(path)
        if ftype == "bias":
            group.biases.append(path)
        elif ftype == "dark":
            group.darks.append(path)
        elif ftype == "flat":
            group.flats.append(path)
        elif ftype == "light":
            group.lights.append(path)

    if not group.has_any():
        group.lights = all_fits

    return group
