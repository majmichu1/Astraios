"""Smart Processor — AI-driven adaptive image processing engine.

Analyzes the image, identifies targets via plate solving + catalog lookup,
reads equipment profiles, measures actual PSF, and builds an adaptive
per-channel processing pipeline that checks quality after each step.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

import numpy as np

from cosmica.core.background import (
    BackgroundParams,
    create_object_exclusion_mask,
    extract_background,
)
from cosmica.core.catalog import CatalogDB, TargetInfo
from cosmica.core.color_tools import (
    ColorAdjustParams,
    SCNRMethod,
    SCNRParams,
    color_adjust,
    scnr,
)
from cosmica.core.deconvolution import (
    DeconvolutionParams,
    SpatialDeconvParams,
    richardson_lucy,
    richardson_lucy_spatial,
)
from cosmica.core.denoise import DenoiseMethod, DenoiseParams, denoise
from cosmica.core.equipment import EquipmentProfile, detect_from_fits_header
from cosmica.core.local_contrast import LocalContrastParams, local_contrast_enhance
from cosmica.core.plate_solve import PlateSolveParams, PlateSolveResult, plate_solve
from cosmica.core.psf import PSFMeasurement, measure_psf
from cosmica.core.stretch import StretchParams, auto_stretch, compute_channel_stats

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


# ---------------------------------------------------------------------------
#  Enums & dataclasses
# ---------------------------------------------------------------------------


class InputType(Enum):
    """Detected input image type."""

    OSC_RGB = auto()  # One-Shot Color (Bayer demosaiced)
    MONO_LUMINANCE = auto()  # Single monochrome channel
    MONO_LRGB = auto()  # Monochrome LRGB composite
    NARROWBAND_SHO = auto()  # SII-Ha-OIII mapped to RGB
    NARROWBAND_HOO = auto()  # Ha-OIII-OIII mapped to RGB
    NARROWBAND_CUSTOM = auto()  # Custom narrowband palette
    DUAL_NARROWBAND = auto()  # Dual-band filter (e.g. Optolong L-eXtreme)
    TRIPLE_NARROWBAND = auto()  # Triple-band filter
    UNKNOWN = auto()


class ProcessingStage(Enum):
    """Named stages in the processing pipeline."""

    BACKGROUND_EXTRACTION = "background_extraction"
    NOISE_REDUCTION = "noise_reduction"
    DECONVOLUTION = "deconvolution"
    STRETCH = "stretch"
    COLOR_CORRECTION = "color_correction"
    SCNR = "scnr"
    LOCAL_CONTRAST = "local_contrast"
    COLOR_SATURATION = "color_saturation"
    FINAL_ADJUSTMENTS = "final_adjustments"


@dataclass
class ImageAnalysis:
    """Results of analyzing the input image."""

    input_type: InputType
    n_channels: int
    height: int
    width: int
    # Per-channel statistics
    channel_stats: list[dict[str, float]]
    # Overall quality metrics
    median_snr: float  # estimated signal-to-noise ratio (from median)
    object_snr: float  # SNR from bright pixels (P95) — better for nebula images
    background_level: float  # median background as fraction of range
    dynamic_range_stops: float  # log2(max_signal / noise_floor)
    has_blown_highlights: bool  # >0.5% pixels at max value
    highlight_fraction: float  # fraction of pixels near saturation
    star_density: str  # "sparse", "moderate", "dense"
    # PSF measurement
    psf: PSFMeasurement | None
    # Plate solve
    plate_solve_result: PlateSolveResult | None
    # Identified targets
    targets: list[TargetInfo]
    primary_target: TargetInfo | None
    # Light pollution assessment
    lp_severity: str = "none"  # "none", "mild", "moderate", "severe"
    has_color_cast: bool = False
    # Equipment data (if available)
    plate_scale_arcsec: float | None = None


@dataclass
class ChannelPlan:
    """Processing plan for a single channel."""

    channel_idx: int
    channel_name: str  # "L", "R", "G", "B", "Ha", "OIII", "SII", "mono"
    snr: float
    # Per-stage parameters (None = skip that stage)
    background_params: BackgroundParams | None = None
    denoise_params: DenoiseParams | None = None
    deconvolution_params: DeconvolutionParams | None = None
    use_spatial_deconv: bool = False  # use spatially-varying PSF
    stretch_params: StretchParams | None = None
    local_contrast_params: LocalContrastParams | None = None


@dataclass
class ProcessingPlan:
    """Complete processing plan built by the Smart Processor."""

    channel_plans: list[ChannelPlan]
    # Post-channel-merge steps
    do_scnr: bool = False
    scnr_params: SCNRParams | None = None
    do_color_adjust: bool = False
    color_adjust_params: ColorAdjustParams | None = None
    # Object-aware background
    use_object_aware_bg: bool = False
    exclusion_mask: np.ndarray | None = field(default=None, repr=False)
    # HDR merge hint for extreme-dynamic-range objects like M42
    needs_hdr_merge: bool = False
    hdr_operator: str = "core_blend"  # "reinhard" | "drago" | "core_blend"
    hdr_params: dict | None = None
    # Star-aware processing: separate stars, enhance the starless nebula, then
    # screen the stars back so an aggressive stretch/contrast doesn't bloat them.
    star_aware: bool = False
    # Summary
    notes: list[str] = field(default_factory=list)


@dataclass
class QualityCheck:
    """Result of a quality check after a processing step."""

    stage: ProcessingStage
    passed: bool
    metric_name: str
    metric_value: float
    threshold: float
    adjustment: str | None = None  # description of adjustment made


@dataclass
class SmartProcessorResult:
    """Final result from the Smart Processor."""

    image: np.ndarray
    analysis: ImageAnalysis
    plan: ProcessingPlan
    quality_checks: list[QualityCheck]
    processing_log: list[str]


# ---------------------------------------------------------------------------
#  Subprocess helpers (must be at module level for pickling)
# ---------------------------------------------------------------------------


def _plate_solve_worker(conn, img, solve_params):
    """Run plate_solve in a child process.  Segfaults here won't crash the app."""
    try:
        result = plate_solve(img, solve_params)
        conn.send(result)
    except Exception:
        conn.send(PlateSolveResult(success=False))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
#  Smart Processor
# ---------------------------------------------------------------------------


class SmartProcessor:
    """AI-driven adaptive image processing engine.

    Usage
    -----
    >>> sp = SmartProcessor(equipment=my_profile)
    >>> result = sp.process(image_data, fits_header=header)
    """

    def __init__(
        self,
        equipment: EquipmentProfile | None = None,
        catalog: CatalogDB | None = None,
    ) -> None:
        self.equipment = equipment
        self.catalog = catalog or CatalogDB()
        self._log: list[str] = []
        self._quality_checks: list[QualityCheck] = []

    # ------------------------------------------------------------------ #
    #  Main entry point                                                   #
    # ------------------------------------------------------------------ #

    def process(
        self,
        data: np.ndarray,
        fits_header: dict[str, Any] | None = None,
        input_type_hint: InputType | None = None,
        target_name: str | None = None,
        ra_hint: float | None = None,
        dec_hint: float | None = None,
        wcs_dict: dict[str, Any] | None = None,
        enabled_stages: set[str] | None = None,
        hdr_operator: str = "core_blend",
        hdr_params: dict | None = None,
        star_reduction: float = 0.3,
        progress: ProgressCallback | None = None,
    ) -> SmartProcessorResult:
        """Run the full Smart Processing pipeline.

        Parameters
        ----------
        data : ndarray
            Image data, (H, W) mono or (C, H, W) color, float32 [0, 1].
        fits_header : dict, optional
            FITS header for equipment/object detection.
        input_type_hint : InputType, optional
            Override automatic input type detection.
        target_name : str, optional
            User-provided target name (e.g. "M42") for catalog lookup fallback.
        ra_hint : float, optional
            User-provided RA hint (degrees) for plate solving.
        dec_hint : float, optional
            User-provided Dec hint (degrees) for plate solving.
        wcs_dict : dict, optional
            Existing WCS header dict from a prior plate solve.  If provided
            with valid ``ra_center`` / ``dec_center`` / ``scale`` keys, the
            plate-solve step is skipped and catalog lookup proceeds directly.
        enabled_stages : set of str, optional
            Which pipeline stages to run.  Valid keys: ``"background"``,
            ``"denoise"``, ``"deconv"``, ``"stretch"``, ``"local_contrast"``,
            ``"hdr_merge"``.  ``None`` = run all.
        hdr_operator : str, optional
            HDR tonemap operator: ``"core_blend"``, ``"reinhard"``, or ``"drago"``.
        hdr_params : dict, optional
            Operator-specific parameters dict.
        progress : callable, optional
            Progress callback ``(fraction, message)``.

        Returns
        -------
        SmartProcessorResult
            Processed image with full analysis and log.
        """
        if progress is None:
            progress = _noop_progress
        self._log = []
        self._quality_checks = []
        self._existing_wcs = wcs_dict
        self._enabled_stages = enabled_stages or {
            "background", "denoise", "deconv", "stretch",
            "local_contrast", "hdr_merge", "star_aware",
        }
        self._user_target_name = target_name
        self._user_ra_hint = ra_hint
        self._user_dec_hint = dec_hint
        self._hdr_operator = hdr_operator
        self._hdr_params = hdr_params
        self._star_reduction = float(max(0.0, min(1.0, star_reduction)))

        # Phase 1: Analyze
        progress(0.0, "Analyzing image...")
        self._log_msg("Starting Smart Processor analysis")
        analysis = self._analyze(data, fits_header, input_type_hint)

        # Phase 2: Plan
        progress(0.10, "Building processing plan...")
        plan = self._build_plan(data, analysis)

        # Phase 3: Execute adaptively
        progress(0.15, "Executing processing pipeline...")
        result_image = self._execute(data, plan, analysis, progress)

        progress(1.0, "Smart Processing complete")
        self._log_msg(
            f"Done: {len(self._quality_checks)} quality checks, "
            f"{sum(1 for q in self._quality_checks if q.passed)} passed"
        )

        return SmartProcessorResult(
            image=result_image,
            analysis=analysis,
            plan=plan,
            quality_checks=self._quality_checks,
            processing_log=list(self._log),
        )

    # ------------------------------------------------------------------ #
    #  Phase 1: Analysis                                                  #
    # ------------------------------------------------------------------ #

    def _analyze(
        self,
        data: np.ndarray,
        fits_header: dict[str, Any] | None,
        input_type_hint: InputType | None,
    ) -> ImageAnalysis:
        """Analyze the image to determine what we're working with."""
        if data.ndim == 2:
            h, w = data.shape
            n_ch = 1
        else:
            n_ch, h, w = data.shape

        # --- Channel statistics ---
        channel_stats = []
        if n_ch == 1:
            channel_stats.append(compute_channel_stats(data if data.ndim == 2 else data[0]))
        else:
            for ch in range(n_ch):
                channel_stats.append(compute_channel_stats(data[ch]))

        # --- Detect input type ---
        input_type = input_type_hint or self._detect_input_type(data, fits_header, channel_stats)
        self._log_msg(f"Detected input type: {input_type.name}")

        # --- Image quality metrics ---
        all_medians = [s["median"] for s in channel_stats]
        all_mads = [s["mad"] for s in channel_stats]
        bg_level = float(np.median(all_medians))
        noise_est = float(np.median(all_mads)) * 1.4826  # MAD -> sigma

        # SNR estimate
        signal = bg_level  # in linear data, signal ≈ median for unstretched
        snr = signal / max(noise_est, 1e-10)

        # Object SNR: measure object brightness above background.
        # Compute simple object mask (pixels > bg + 3σ) and use median
        # of those pixels.  The contrast (object - bg) / noise captures
        # nebula visibility for objects like M42 where the bright core
        # is a tiny fraction of the total area.
        if data.ndim == 2:
            flat = data.ravel()
        else:
            flat = np.max(data, axis=0).ravel()
        obj_mask = flat > (bg_level + 3.0 * noise_est)
        if obj_mask.any():
            object_signal = float(np.median(flat[obj_mask]))
            object_snr = max(0.0, (object_signal - bg_level) / max(noise_est, 1e-10))
        else:
            object_signal = bg_level
            object_snr = 0.0
        if object_snr > snr * 2:
            self._log_msg(
                f"Object SNR={object_snr:.1f} (obj median - bg) vs global SNR={snr:.1f} — "
                f"using object SNR for nebula-aware decisions"
            )

        # Dynamic range
        max_val = max(s["max"] for s in channel_stats)
        dr_stops = math.log2(max(max_val, 1e-10) / max(noise_est, 1e-10))

        # Blown highlights
        if data.ndim == 2:
            sat_frac = float(np.mean(data > 0.98))
        else:
            sat_frac = float(np.mean(np.max(data, axis=0) > 0.98))
        blown = sat_frac > 0.005

        # --- Plate scale ---
        plate_scale = None
        if self.equipment:
            plate_scale = self.equipment.plate_scale()
            self._log_msg(f"Plate scale from equipment: {plate_scale:.2f} arcsec/px")

        # --- Header sniffing ---
        header_info: dict[str, Any] = {}
        if fits_header:
            header_info = detect_from_fits_header(fits_header)
            if not plate_scale and "pixel_size_um" in header_info and "focal_length_mm" in header_info:
                plate_scale = 206.265 * header_info["pixel_size_um"] / header_info["focal_length_mm"]
                self._log_msg(f"Plate scale from FITS header: {plate_scale:.2f} arcsec/px")

        # --- PSF measurement ---
        psf = None
        try:
            gray = data if data.ndim == 2 else np.mean(data, axis=0).astype(np.float32)
            psf = measure_psf(gray)
            if psf.n_stars_used > 0:
                self._log_msg(
                    f"PSF measured: FWHM={psf.fwhm:.2f}px, "
                    f"ellipticity={psf.ellipticity:.3f}, "
                    f"{psf.n_stars_used} stars"
                )
                if plate_scale:
                    fwhm_arcsec = psf.fwhm * plate_scale
                    self._log_msg(f"Seeing estimate: {fwhm_arcsec:.1f} arcsec")
        except Exception as exc:
            self._log_msg(f"PSF measurement failed: {exc}")

        # Star density from PSF measurement
        n_stars = psf.n_stars_used if psf else 0
        if n_stars > 30:
            star_density = "dense"
        elif n_stars > 10:
            star_density = "moderate"
        else:
            star_density = "sparse"

        # --- Plate solving + catalog lookup ---
        solve_result = None
        targets: list[TargetInfo] = []
        primary_target: TargetInfo | None = None

        # Reuse existing WCS from a prior plate solve (avoids re-solving)
        if self._existing_wcs:
            ra = self._existing_wcs.get("ra_center", 0.0)
            dec = self._existing_wcs.get("dec_center", 0.0)
            scale = self._existing_wcs.get("scale", 0.0)
            if ra > 0 and scale > 0:
                rotation = self._existing_wcs.get("rotation", 0.0)
                solve_result = PlateSolveResult(
                    success=True,
                    ra_center=ra,
                    dec_center=dec,
                    pixel_scale=scale,
                    rotation=rotation,
                    wcs_header=self._existing_wcs,
                )
                if not plate_scale or plate_scale <= 0:
                    plate_scale = scale
                self._log_msg(
                    f"Using existing WCS: RA={ra:.4f}°, Dec={dec:.4f}°, "
                    f"scale={scale:.2f}\"/px"
                )
            else:
                self._log_msg(
                    "Existing WCS lacks coordinate data — will attempt plate solve"
                )

        # Try plate solving (only if we don't already have a valid WCS)
        if solve_result is None:
            try:
                solve_params = PlateSolveParams()
                if plate_scale:
                    solve_params.scale_hint = plate_scale
                # Use user-provided RA/Dec hints if available
                if self._user_ra_hint is not None and self._user_dec_hint is not None:
                    solve_params.ra_hint = self._user_ra_hint
                    solve_params.dec_hint = self._user_dec_hint
                    self._log_msg(f"Using user-provided coordinates: RA={self._user_ra_hint:.4f}, Dec={self._user_dec_hint:.4f}")
                elif "ra" in header_info and "dec" in header_info:
                    try:
                        solve_params.ra_hint = float(header_info["ra"])
                        solve_params.dec_hint = float(header_info["dec"])
                    except (ValueError, TypeError):
                        pass

                self._log_msg("Attempting plate solve...")
                solve_result = self._safe_plate_solve(data, solve_params)

                if solve_result and solve_result.success:
                    self._log_msg(
                        f"Plate solve success: RA={solve_result.ra_center:.4f}, "
                        f"Dec={solve_result.dec_center:.4f}, "
                        f"scale={solve_result.pixel_scale:.2f} arcsec/px"
                    )
                    if not plate_scale and solve_result.pixel_scale > 0:
                        plate_scale = solve_result.pixel_scale
            except Exception as exc:
                self._log_msg(f"Plate solving failed: {exc}")

        # Catalog lookup
        if solve_result and solve_result.success and solve_result.ra_center > 0:
            fov_arcmin = 60.0  # default search radius
            if plate_scale:
                fov_arcmin = max(plate_scale * max(w, h) / 60.0, 30.0)
            targets = self.catalog.query_region(
                solve_result.ra_center,
                solve_result.dec_center,
                fov_arcmin,
            )
            if targets:
                primary_target = targets[0]
                self._log_msg(
                    f"Primary target: {primary_target.id} "
                    f"({', '.join(primary_target.names[:2]) if primary_target.names else 'unnamed'}) "
                    f"— {primary_target.object_type}, "
                    f"brightness={primary_target.brightness_class}"
                )
                for t in targets[1:3]:
                    self._log_msg(f"Also in field: {t.id}")

        # Fallback: user-provided target name
        if not primary_target and self._user_target_name:
            target = self.catalog.lookup(self._user_target_name)
            if target:
                primary_target = target
                targets = [target]
                self._log_msg(f"Target from user input: {target.id}")
            else:
                self._log_msg(f"User target '{self._user_target_name}' not found in catalog")

        # Fallback: FITS header object name
        if not primary_target and "object_name" in header_info:
            target = self.catalog.lookup(header_info["object_name"])
            if target:
                primary_target = target
                targets = [target]
                self._log_msg(f"Target from FITS header: {target.id}")

        # --- LP severity and color cast detection ---
        lp_severity = self._detect_lp_severity(data, channel_stats)
        self._log_msg(f"Light pollution severity: {lp_severity}")

        has_color_cast = False
        if n_ch >= 3:
            medians = [s["median"] for s in channel_stats[:3]]
            if max(medians) - min(medians) > 0.03:
                has_color_cast = True
                self._log_msg(f"Color cast detected: channel medians {[f'{m:.4f}' for m in medians]}")

        return ImageAnalysis(
            input_type=input_type,
            n_channels=n_ch,
            height=h,
            width=w,
            channel_stats=channel_stats,
            median_snr=snr,
            object_snr=object_snr,
            background_level=bg_level,
            dynamic_range_stops=dr_stops,
            has_blown_highlights=blown,
            highlight_fraction=sat_frac,
            star_density=star_density,
            psf=psf,
            plate_solve_result=solve_result,
            targets=targets,
            primary_target=primary_target,
            lp_severity=lp_severity,
            has_color_cast=has_color_cast,
            plate_scale_arcsec=plate_scale,
        )

    def _detect_input_type(
        self,
        data: np.ndarray,
        fits_header: dict[str, Any] | None,
        stats: list[dict[str, float]],
    ) -> InputType:
        """Detect the type of input image."""
        if data.ndim == 2:
            return InputType.MONO_LUMINANCE

        n_ch = data.shape[0]
        if n_ch == 1:
            return InputType.MONO_LUMINANCE

        # Check FITS header for filter info
        if fits_header:
            info = detect_from_fits_header(fits_header)
            filter_name = info.get("filter", "").lower()

            if any(nb in filter_name for nb in ["sho", "sii", "ha", "oiii"]):
                if "sho" in filter_name:
                    return InputType.NARROWBAND_SHO
                if "hoo" in filter_name:
                    return InputType.NARROWBAND_HOO

            # Check for dual narrowband filters
            if self.equipment and self.equipment.filters:
                for fname, filt in self.equipment.filters.items():
                    if filt.filter_type == "dual_narrowband":
                        return InputType.DUAL_NARROWBAND
                    if filt.filter_type == "triple_narrowband":
                        return InputType.TRIPLE_NARROWBAND

        if n_ch >= 3:
            # Heuristic: compare channel medians
            # Narrowband images often have very different per-channel statistics
            medians = [s["median"] for s in stats[:3]]
            if len(medians) == 3:
                median_range = max(medians) - min(medians)
                avg_median = np.mean(medians)
                if avg_median > 0 and median_range / avg_median > 0.8:
                    # Very different channels — likely narrowband
                    return InputType.NARROWBAND_CUSTOM

            if n_ch == 4:
                return InputType.MONO_LRGB

            return InputType.OSC_RGB

        return InputType.UNKNOWN

    def _detect_lp_severity(
        self,
        data: np.ndarray,
        channel_stats: list[dict[str, float]],
    ) -> str:
        """Detect light pollution severity from background characteristics."""
        bg_level = float(np.median([s["median"] for s in channel_stats]))

        bg_spread = 0.0
        if len(channel_stats) >= 3:
            bg_medians = [s["median"] for s in channel_stats[:3]]
            bg_spread = max(bg_medians) - min(bg_medians)

        # Measure gradient strength via std of heavily-smoothed image
        from scipy.ndimage import gaussian_filter
        if data.ndim == 2:
            smoothed = gaussian_filter(data, sigma=max(data.shape) // 8)
        else:
            smoothed = gaussian_filter(np.mean(data, axis=0), sigma=max(data.shape[1:]) // 8)
        gradient_std = float(np.std(smoothed))

        if bg_level > 0.15 or bg_spread > 0.05 or gradient_std > 0.08:
            return "severe"
        elif bg_level > 0.08 or bg_spread > 0.02 or gradient_std > 0.04:
            return "moderate"
        elif bg_level > 0.03 or gradient_std > 0.02:
            return "mild"
        return "none"

    # ------------------------------------------------------------------ #
    #  Phase 2: Planning                                                  #
    # ------------------------------------------------------------------ #

    def _build_plan(
        self,
        data: np.ndarray,
        analysis: ImageAnalysis,
    ) -> ProcessingPlan:
        """Build the processing plan based on image analysis."""
        hints = {}
        is_extreme_dr = False
        if analysis.primary_target:
            hints = analysis.primary_target.processing_hints.copy()
            if analysis.primary_target.dynamic_range == "extreme":
                hints["dynamic_range"] = "extreme"
                is_extreme_dr = True

        # Determine channel names
        ch_names = self._get_channel_names(analysis)

        # Build per-channel plans
        channel_plans = []
        for i, name in enumerate(ch_names):
            stats = analysis.channel_stats[min(i, len(analysis.channel_stats) - 1)]
            ch_snr = stats["median"] / max(stats["mad"] * 1.4826, 1e-10)
            plan = self._plan_channel(i, name, ch_snr, analysis, hints)
            channel_plans.append(plan)

        # Post-merge color steps
        do_scnr = False
        scnr_params = None
        do_color = False
        color_params = None

        if analysis.input_type in (
            InputType.NARROWBAND_SHO,
            InputType.NARROWBAND_HOO,
            InputType.NARROWBAND_CUSTOM,
            InputType.DUAL_NARROWBAND,
        ):
            # Narrowband images typically need SCNR
            do_scnr = True
            scnr_params = SCNRParams(
                method=SCNRMethod.AVERAGE_NEUTRAL,
                amount=0.8,
                preserve_luminance=True,
            )
            self._log_msg("Plan: SCNR enabled for narrowband data")

        if analysis.n_channels >= 3:
            target_sat = self._get_saturation_target(analysis, hints)
            if abs(target_sat - 1.0) > 0.05:
                do_color = True
                color_params = ColorAdjustParams(saturation=target_sat)
                self._log_msg(f"Plan: Color saturation target: {target_sat:.2f}")

        # Object-aware background
        use_obj_bg = False
        exclusion_mask = None
        if (
            analysis.primary_target
            and analysis.plate_scale_arcsec
            and analysis.plate_solve_result
            and analysis.plate_solve_result.success
        ):
            # Build exclusion mask from catalog objects
            objects = []
            for t in analysis.targets:
                if t.major_axis_arcmin > 1.0:
                    # Convert RA/Dec offset to pixel position
                    cx, cy = self._target_to_pixel(
                        t, analysis.plate_solve_result, analysis.plate_scale_arcsec,
                        analysis.width, analysis.height,
                    )
                    if 0 <= cx < analysis.width and 0 <= cy < analysis.height:
                        objects.append({
                            "center_x": cx,
                            "center_y": cy,
                            "radius_arcmin": t.major_axis_arcmin / 2.0,
                        })

            if objects:
                exclusion_mask = create_object_exclusion_mask(
                    (analysis.height, analysis.width),
                    objects,
                    analysis.plate_scale_arcsec,
                )
                use_obj_bg = True
                self._log_msg(f"Plan: Object-aware background with {len(objects)} object(s)")

        needs_hdr = hints.get("hdr_merge_recommended", False) or is_extreme_dr
        if needs_hdr:
            self._log_msg("Plan: HDR-merge hint active — extreme dynamic range detected")

        star_aware = "star_aware" in self._enabled_stages
        if star_aware:
            # Local contrast is deferred to the starless layer (see _execute),
            # so it enhances nebula detail without amplifying star halos.
            for cp in channel_plans:
                cp.local_contrast_params = None
            self._log_msg(
                "Plan: star-aware processing enabled "
                "(stars separated, nebula enhanced, stars recombined)"
            )

        plan = ProcessingPlan(
            channel_plans=channel_plans,
            do_scnr=do_scnr,
            scnr_params=scnr_params,
            do_color_adjust=do_color,
            color_adjust_params=color_params,
            use_object_aware_bg=use_obj_bg,
            exclusion_mask=exclusion_mask,
            needs_hdr_merge=needs_hdr,
            hdr_operator=self._hdr_operator,
            hdr_params=self._hdr_params,
            star_aware=star_aware,
        )

        # Summary notes
        plan.notes.append(f"Input type: {analysis.input_type.name}")
        plan.notes.append(f"Channels: {', '.join(ch_names)}")
        plan.notes.append(f"SNR estimate: {analysis.median_snr:.1f}")
        if analysis.primary_target:
            plan.notes.append(f"Target: {analysis.primary_target.id}")
        plan.notes.append(f"Pipeline stages per channel: {self._count_stages(channel_plans[0])}")

        return plan

    def _plan_channel(
        self,
        idx: int,
        name: str,
        snr: float,
        analysis: ImageAnalysis,
        hints: dict[str, Any],
    ) -> ChannelPlan:
        """Build a processing plan for a single channel."""
        plan = ChannelPlan(channel_idx=idx, channel_name=name, snr=snr)

        # --- Background extraction ---
        bg_complexity = hints.get("background_complexity", "gradient")
        lp = analysis.lp_severity
        if lp == "severe":
            grid_size = 16
            poly_order = 5
        elif lp == "moderate":
            grid_size = 14
            poly_order = 4
        elif bg_complexity == "complex":
            grid_size = 12
            poly_order = 4
        elif bg_complexity == "flat":
            grid_size = 8
            poly_order = 2
        else:
            grid_size = 10
            poly_order = 3

        plan.background_params = BackgroundParams(
            grid_size=grid_size,
            polynomial_order=poly_order,
            sigma_clip=2.5,
        )
        if "background" not in self._enabled_stages:
            plan.background_params = None

        # --- Noise reduction ---
        nr_level = hints.get("noise_reduction", self._auto_nr_level(snr))
        strength = {
            "minimal": 0.15,
            "light": 0.3,
            "moderate": 0.5,
            "heavy": 0.8,
        }.get(nr_level, 0.5)

        # For extreme-DR objects (M42 etc.) halve NR to preserve fine nebulosity
        if hints.get("dynamic_range") == "extreme":
            strength *= 0.5

        # Luminance / detail channels get less NR; color channels can have more
        if name in ("L", "mono"):
            strength *= 0.7
            detail_preservation = 0.7
        elif name in ("Ha", "OIII", "SII"):
            # Narrowband: stronger NR for faint signal
            if snr < 5:
                strength = min(strength * 1.3, 0.95)
            detail_preservation = 0.5
        else:
            detail_preservation = 0.4

        plan.denoise_params = DenoiseParams(
            method=DenoiseMethod.WAVELET,
            strength=strength,
            detail_preservation=detail_preservation,
        )
        if "denoise" not in self._enabled_stages:
            plan.denoise_params = None

        # --- Deconvolution ---
        # Auto-decide based on measured PSF and SNR.  For nebula-dominated
        # images the *object* SNR (from P95) is often much higher than the
        # global median SNR — use whichever is larger.
        effective_snr = max(snr, analysis.object_snr) if hasattr(analysis, 'object_snr') else snr
        do_deconv = hints.get("deconvolution", False) or hints.get("deconv_aggressive", False)
        if not do_deconv:
            if analysis.psf and analysis.psf.fwhm > 2.5 and effective_snr > 10:
                do_deconv = True

        # For known high-DR targets (M42, etc.) always try deconv if we have
        # a valid PSF — the stars are sharp even if the nebula dominates.
        if not do_deconv and hints.get("dynamic_range") == "extreme":
            if analysis.psf and analysis.psf.n_stars_used >= 3:
                do_deconv = True
                self._log_msg(
                    f"Plan [{name}]: Forcing deconvolution for extreme-DR target "
                    f"(PSF valid, {analysis.psf.n_stars_used} stars)"
                )

        # Guard: skip deconvolution if PSF measurement is unreliable
        if do_deconv and analysis.psf:
            if analysis.psf.n_stars_used < 3:
                self._log_msg(
                    f"Plan [{name}]: Skipping deconvolution — PSF from only "
                    f"{analysis.psf.n_stars_used} star(s), unreliable"
                )
                do_deconv = False
            elif analysis.psf.ellipticity > 0.3:
                self._log_msg(
                    f"Plan [{name}]: Skipping deconvolution — PSF ellipticity "
                    f"{analysis.psf.ellipticity:.3f} too high (tracking/guiding issue)"
                )
                do_deconv = False

        # Catalog hint: gentle stretch means conservative deconvolution
        # Extreme-DR objects (e.g. M42) get even gentler treatment to avoid
        # ringing around bright nebula cores.
        if hints.get("dynamic_range") == "extreme":
            gentle_deconv = True
            extreme_dr_deconv = True
        elif hints.get("stretch") == "gentle":
            gentle_deconv = True
            extreme_dr_deconv = False
        else:
            gentle_deconv = False
            extreme_dr_deconv = False

        if do_deconv and analysis.psf:
            fwhm = analysis.psf.fwhm
            # Conservative parameters — execution phase will adapt
            max_iters = 5 if extreme_dr_deconv else (8 if gentle_deconv else 10)
            reg = 0.03 if extreme_dr_deconv else (0.02 if gentle_deconv else 0.01)
            dering_amount = 1.0 if extreme_dr_deconv else 0.8
            plan.deconvolution_params = DeconvolutionParams(
                psf_fwhm=fwhm,
                iterations=max_iters,
                regularization=reg,
                deringing=True,
                deringing_amount=dering_amount,
            )
            if "deconv" not in self._enabled_stages:
                plan.deconvolution_params = None
                plan.use_spatial_deconv = False

            # Use spatially-varying PSF when the image is large enough
            # and we have sufficient stars across the field. This handles
            # field curvature/coma at edges much better than global deconv.
            min_dim = min(analysis.width, analysis.height)
            if min_dim >= 800 and analysis.psf.n_stars_used >= 8:
                plan.use_spatial_deconv = True
                self._log_msg(
                    f"Plan [{name}]: Spatial deconvolution (zone PSF) "
                    f"FWHM={fwhm:.1f}px "
                    f"({'gentle' if gentle_deconv else 'adaptive'})"
                )
            else:
                self._log_msg(
                    f"Plan [{name}]: Global deconvolution FWHM={fwhm:.1f}px "
                    f"({'gentle' if gentle_deconv else 'adaptive'})"
                )

        # --- Stretch ---
        # Initial guess — adaptive execution will measure and adjust
        plan.stretch_params = StretchParams(
            midtone=0.15,  # starting point, will be tuned to target brightness
            shadow_clip=-2.0,
            linked=False,  # per-channel control
        )
        if "stretch" not in self._enabled_stages:
            plan.stretch_params = None

        # --- Local contrast ---
        contrast_level = hints.get("contrast_enhancement", "moderate")
        # For extreme-DR objects (M42 etc.) use at most subtle LC
        if hints.get("dynamic_range") == "extreme":
            contrast_level = "subtle"
        if contrast_level != "none":
            clip_limit = {
                "subtle": 1.5,
                "moderate": 2.5,
                "strong": 4.0,
            }.get(contrast_level, 2.5)
            plan.local_contrast_params = LocalContrastParams(
                clip_limit=clip_limit,
                tile_size=8,
                amount=0.6,
            )
        if "local_contrast" not in self._enabled_stages:
            plan.local_contrast_params = None

        return plan

    def _get_channel_names(self, analysis: ImageAnalysis) -> list[str]:
        """Determine channel names based on input type."""
        n_ch = analysis.n_channels
        it = analysis.input_type

        if it == InputType.MONO_LUMINANCE:
            return ["mono"]
        if it == InputType.MONO_LRGB and n_ch == 4:
            return ["L", "R", "G", "B"]
        if it == InputType.NARROWBAND_SHO and n_ch >= 3:
            return ["SII", "Ha", "OIII"][:n_ch]
        if it == InputType.NARROWBAND_HOO and n_ch >= 3:
            return ["Ha", "OIII", "OIII"][:n_ch]
        if it in (InputType.OSC_RGB, InputType.DUAL_NARROWBAND, InputType.TRIPLE_NARROWBAND):
            if n_ch >= 3:
                return ["R", "G", "B"][:n_ch]
        if it == InputType.NARROWBAND_CUSTOM and n_ch >= 3:
            return ["CH1", "CH2", "CH3"][:n_ch]
        return [f"CH{i}" for i in range(n_ch)]

    def _auto_nr_level(self, snr: float) -> str:
        """Determine noise reduction level from SNR."""
        if snr < 5:
            return "heavy"
        if snr < 15:
            return "moderate"
        if snr < 40:
            return "light"
        return "minimal"

    def _auto_stretch_level(self, analysis: ImageAnalysis) -> str:
        """Determine stretch aggressiveness based on the image."""
        if analysis.primary_target:
            bc = analysis.primary_target.brightness_class
            if bc in ("very_faint", "faint"):
                return "aggressive"
            if bc == "very_bright":
                return "gentle"
        if analysis.median_snr < 8:
            return "aggressive"
        if analysis.median_snr > 50:
            return "gentle"
        return "moderate"

    def _get_saturation_target(self, analysis: ImageAnalysis, hints: dict) -> float:
        """Determine target saturation multiplier."""
        level = hints.get("color_saturation", "moderate")
        return {"boost": 1.4, "moderate": 1.15, "preserve": 1.0}.get(level, 1.15)

    def _count_stages(self, plan: ChannelPlan) -> int:
        """Count the number of active stages in a channel plan."""
        count = 0
        if plan.background_params:
            count += 1
        if plan.denoise_params:
            count += 1
        if plan.deconvolution_params:
            count += 1
        if plan.stretch_params:
            count += 1
        if plan.local_contrast_params:
            count += 1
        return count

    def _target_to_pixel(
        self,
        target: TargetInfo,
        solve: PlateSolveResult,
        plate_scale: float,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        """Convert a target's RA/Dec to approximate pixel coordinates."""
        # Flat-sky tangent projection
        cos_dec = math.cos(math.radians(solve.dec_center))
        dx_deg = (target.ra_deg - solve.ra_center) * cos_dec
        dy_deg = target.dec_deg - solve.dec_center

        dx_px = dx_deg * 3600.0 / plate_scale
        dy_px = -dy_deg * 3600.0 / plate_scale  # y inverted

        cx = width / 2.0 + dx_px
        cy = height / 2.0 + dy_px
        return (cx, cy)

    # ------------------------------------------------------------------ #
    #  Phase 3: Adaptive Execution                                        #
    # ------------------------------------------------------------------ #

    def _execute(
        self,
        data: np.ndarray,
        plan: ProcessingPlan,
        analysis: ImageAnalysis,
        progress: ProgressCallback,
    ) -> np.ndarray:
        """Execute the processing plan with quality checks."""
        working = data.copy()
        n_stages = 5  # bg, nr, deconv, stretch, post

        # ---- Per-channel processing ----
        if working.ndim == 2:
            # Mono: single channel
            cp = plan.channel_plans[0]
            working = self._execute_channel(
                working, cp, plan, analysis, progress, 0.15, 0.85,
            )
        else:
            n_ch = working.shape[0]

            # --- Pass 1: Background extraction for all channels ---
            progress(0.15, "Background extraction...")
            bg_fraction = 0.12 / n_ch  # fraction of progress for BG per channel
            for ch in range(n_ch):
                if ch < len(plan.channel_plans):
                    cp = plan.channel_plans[ch]
                else:
                    cp = plan.channel_plans[-1]
                frac = 0.15 + ch * bg_fraction
                progress(frac, f"Background extraction [{cp.channel_name}]...")
                working[ch] = self._extract_bg_only(
                    working[ch], cp, analysis,
                )

            # --- Linked signal rescaling (preserves color ratios) ---
            # After BG removal, emission nebulae can have compressed signals
            # (e.g. p99.9 ~ 0.002). Rescale using the SAME factor for all
            # channels so color balance is preserved.
            # Use p99.9 (not p99.5) so the very brightest pixels are not
            # clipped too aggressively, preserving dynamic range in cores.
            ch_peaks = []
            for ch in range(n_ch):
                ch_peaks.append(float(np.percentile(working[ch], 99.9)))
            max_peak = max(ch_peaks)

            if 1e-7 < max_peak < 0.15:
                # Target: bring brightest channel p99.9 to ~0.5
                # (leaves headroom for stretch to work without clipping)
                scale_factor = min(0.5 / max_peak, 100.0)  # cap at 100x
                working = np.clip(working * scale_factor, 0, 1).astype(np.float32)
                self._log_msg(
                    f"Linked signal rescale {scale_factor:.1f}x "
                    f"(channel p99.9: {[f'{p:.4f}' for p in ch_peaks]})"
                )

            # --- Pass 2: Remaining per-channel processing (skip BG) ---
            for ch in range(n_ch):
                if ch < len(plan.channel_plans):
                    cp = plan.channel_plans[ch]
                else:
                    cp = plan.channel_plans[-1]

                ch_start = 0.27 + (ch / n_ch) * 0.53
                ch_end = 0.27 + ((ch + 1) / n_ch) * 0.53

                working[ch] = self._execute_channel(
                    working[ch], cp, plan, analysis, progress, ch_start, ch_end,
                    skip_background=True,
                )

        # ---- Post-merge steps ----
        progress(0.82, "Post-processing...")

        # SCNR
        if plan.do_scnr and working.ndim == 3 and working.shape[0] >= 3:
            self._log_msg("Applying SCNR")
            working = scnr(working, plan.scnr_params)
            self._check_color_balance(working, ProcessingStage.SCNR)

        # Adaptive color balance: equalize channel backgrounds if imbalanced
        if working.ndim == 3 and working.shape[0] >= 3:
            bg_medians = []
            for ch in range(min(3, working.shape[0])):
                # Measure background from the darkest 20% of pixels
                ch_data = working[ch]
                p20 = float(np.percentile(ch_data, 20))
                bg_pixels = ch_data[ch_data <= p20]
                bg_medians.append(float(np.median(bg_pixels)) if len(bg_pixels) > 0 else 0.0)

            bg_spread = max(bg_medians) - min(bg_medians)
            if bg_spread > 0.02:
                # Channels have different background levels — neutralize
                target_bg = min(bg_medians)  # bring all down to darkest
                for ch in range(min(3, working.shape[0])):
                    offset = bg_medians[ch] - target_bg
                    if offset > 0.005:
                        working[ch] = np.clip(working[ch] - offset, 0, 1)
                self._log_msg(
                    f"Color balance: neutralized backgrounds "
                    f"(spread {bg_spread:.3f}, offsets: "
                    f"{[f'{bg_medians[i]-target_bg:.3f}' for i in range(min(3, working.shape[0]))]}"
                    f")"
                )

        # Per-channel gain correction for color balance.
        # Skipped for narrowband: in SHO/HOO palettes the channel intensity
        # ratios (Ha is usually far stronger than SII/OIII) are intentional and
        # define the palette — equalizing the signal levels would mute it.
        _narrowband_types = (
            InputType.NARROWBAND_SHO,
            InputType.NARROWBAND_HOO,
            InputType.NARROWBAND_CUSTOM,
            InputType.DUAL_NARROWBAND,
            InputType.TRIPLE_NARROWBAND,
        )
        is_narrowband = analysis.input_type in _narrowband_types
        if is_narrowband:
            self._log_msg("Skipping channel gain equalization (narrowband palette preserved)")
        if working.ndim == 3 and working.shape[0] >= 3 and not is_narrowband:
            signal_levels = []
            for ch in range(min(3, working.shape[0])):
                ch_data = working[ch]
                # Use p70-p95 range — this captures nebula/object signal
                # rather than sky (low percentiles) or stars (very high)
                p70 = float(np.percentile(ch_data, 70))
                p95 = float(np.percentile(ch_data, 95))
                sig_pixels = ch_data[(ch_data > p70) & (ch_data < p95)]
                signal_levels.append(
                    float(np.median(sig_pixels)) if len(sig_pixels) > 100 else p70
                )

            if min(signal_levels) > 1e-6:
                target_signal = float(np.median(signal_levels))
                applied_gains = []
                for ch in range(min(3, working.shape[0])):
                    gain = target_signal / max(signal_levels[ch], 1e-6)
                    gain = max(0.3, min(3.0, gain))
                    if abs(gain - 1.0) > 0.02:
                        working[ch] = np.clip(working[ch] * gain, 0, 1)
                        applied_gains.append(f"ch{ch}*={gain:.3f}")
                if applied_gains:
                    self._log_msg(f"Color gain correction: {', '.join(applied_gains)}")

        # Color adjustment
        if plan.do_color_adjust and working.ndim == 3 and working.shape[0] >= 3:
            self._log_msg("Applying color adjustment")
            working = color_adjust(working, plan.color_adjust_params)

        # HDR local contrast: single very gentle CLAHE pass for extreme-DR
        # objects.  Reveals faint outer structure without amplifying noise.
        if plan.needs_hdr_merge and "hdr_merge" in self._enabled_stages:
            progress(0.96, "HDR local contrast enhancement...")
            self._log_msg("HDR: gentle single-pass local contrast")
            hdr_lce = LocalContrastParams(
                clip_limit=1.8,
                tile_size=8,
                amount=0.25,
            )
            working = local_contrast_enhance(working, hdr_lce)

        # Star-aware enhancement: separate stars, enhance the starless nebula,
        # then screen the stars back. Done last (on the stretched image) because
        # the star-removal models expect non-linear data.
        if plan.star_aware:
            working = self._apply_star_aware(working, analysis, plan, progress)

        # Final clamp
        working = np.clip(working, 0, 1).astype(np.float32)

        return working

    def _separate_stars(self, image: np.ndarray) -> np.ndarray | None:
        """Return a starless version of ``image`` (StarNet binary if available,
        else the built-in morphological remover). None on failure."""
        try:
            from cosmica.ai.inference.starnet import find_starnet_binary, run_starnet

            if find_starnet_binary() is not None:
                res = run_starnet(image, extract_stars=False)
                if res.success:
                    self._log_msg("Star separation: StarNet")
                    return res.starless.astype(np.float32)
        except Exception as exc:
            self._log_msg(f"StarNet unavailable ({exc}); using built-in remover")

        try:
            from cosmica.ai.inference.star_removal import remove_stars_builtin

            starless = remove_stars_builtin(image, threshold=0.5).astype(np.float32)
            self._log_msg("Star separation: built-in morphological")
            return starless
        except Exception as exc:
            self._log_msg(f"Star removal failed ({exc}); skipping star-aware step")
            return None

    def _apply_star_aware(
        self,
        working: np.ndarray,
        analysis: ImageAnalysis,
        plan: ProcessingPlan,
        progress: ProgressCallback,
    ) -> np.ndarray:
        """Separate stars, enhance the starless nebula, optionally shrink the
        isolated stars, then screen them back."""
        progress(0.95, "Star-aware: separating stars...")
        starless = self._separate_stars(working)
        if starless is None:
            return working  # graceful fallback — keep the non-star-aware result

        # Star layer = what the remover took out. Screening it back over the
        # (enhanced) starless reconstructs the image without clipping.
        stars = np.clip(working - starless, 0.0, 1.0).astype(np.float32)
        is_color = working.ndim == 3 and working.shape[0] >= 3
        has_stars = float(np.mean(stars > 0.05)) > 1e-4

        # Independent star reduction — only possible because the stars are
        # isolated here; far cleaner than eroding them in the full image where
        # nebula detail would be damaged too.
        reduced = False
        if self._star_reduction > 0.0 and has_stars:
            try:
                from cosmica.core.star_reduction import StarReductionParams, reduce_stars

                progress(0.96, "Star-aware: reducing star bloat...")
                sr_iters = 1 + int(round(self._star_reduction * 3))
                stars = reduce_stars(
                    stars,
                    params=StarReductionParams(
                        amount=self._star_reduction,
                        iterations=sr_iters,
                        protect_core=True,
                    ),
                ).astype(np.float32)
                reduced = True
                self._log_msg(
                    f"Star-aware: reduced star sizes (amount={self._star_reduction:.2f})"
                )
            except Exception as exc:
                self._log_msg(f"Star reduction skipped ({exc})")

        # Gentle star colour boost so star colours pop (colour, non-narrowband).
        if is_color and not self._is_narrowband(analysis):
            try:
                stars = color_adjust(stars, ColorAdjustParams(saturation=1.2)).astype(np.float32)
            except Exception:
                pass

        # Enhance the starless nebula — local contrast + (for colour) saturation.
        # Safe here because there are no stars to bloat or ring.
        progress(0.98, "Star-aware: enhancing nebula...")
        enhanced = local_contrast_enhance(
            starless, LocalContrastParams(clip_limit=2.2, tile_size=8, amount=0.5)
        )
        if is_color and not self._is_narrowband(analysis):
            enhanced = color_adjust(enhanced, ColorAdjustParams(saturation=1.15))

        # Screen the stars back on top.
        progress(0.99, "Star-aware: recombining stars...")
        from cosmica.core.image_blend import BlendMode, BlendParams, blend_images

        result = blend_images(
            enhanced, stars, BlendParams(mode=BlendMode.SCREEN, opacity=1.0)
        )
        star_frac = float(np.mean(stars > 0.05))
        self._log_msg(
            f"Star-aware complete: nebula enhanced, stars "
            f"{'reduced & ' if reduced else ''}recombined "
            f"({star_frac * 100:.1f}% star pixels)"
        )
        return result

    @staticmethod
    def _is_narrowband(analysis: ImageAnalysis) -> bool:
        return analysis.input_type in (
            InputType.NARROWBAND_SHO,
            InputType.NARROWBAND_HOO,
            InputType.NARROWBAND_CUSTOM,
            InputType.DUAL_NARROWBAND,
            InputType.TRIPLE_NARROWBAND,
        )

    def _execute_channel(
        self,
        channel: np.ndarray,
        plan: ChannelPlan,
        full_plan: ProcessingPlan,
        analysis: ImageAnalysis,
        progress: ProgressCallback,
        frac_start: float,
        frac_end: float,
        skip_background: bool = False,
    ) -> np.ndarray:
        """Execute the processing plan for a single channel with quality checks."""
        working = channel.copy()
        frac_range = frac_end - frac_start
        name = plan.channel_name

        # Stage 1: Iterative background extraction
        if plan.background_params is not None and not skip_background:
            frac = frac_start + frac_range * 0.0
            progress(frac, f"Background extraction [{name}]...")
            self._log_msg(f"[{name}] Background extraction")

            bg_params = plan.background_params
            # Apply object-aware mode
            if full_plan.use_object_aware_bg and full_plan.exclusion_mask is not None:
                bg_params = BackgroundParams(
                    grid_size=bg_params.grid_size,
                    box_size=bg_params.box_size,
                    polynomial_order=bg_params.polynomial_order,
                    sigma_clip=bg_params.sigma_clip,
                    smoothing=bg_params.smoothing,
                    object_aware=True,
                    exclusion_mask=full_plan.exclusion_mask,
                )

            working, bg_model = extract_background(working, bg_params)

            # Adaptive: measure residual gradient, iterate if not flat enough
            bg_residual = np.std(bg_model)
            max_bg_passes = 3
            bg_pass = 1
            while bg_residual >= 0.05 and bg_pass < max_bg_passes:
                bg_pass += 1
                # Increase grid density and polynomial order for finer correction
                refined_params = BackgroundParams(
                    grid_size=min(bg_params.grid_size + 4, 24),
                    box_size=bg_params.box_size,
                    polynomial_order=min(bg_params.polynomial_order + 1, 6),
                    sigma_clip=bg_params.sigma_clip,
                    smoothing=bg_params.smoothing,
                    object_aware=bg_params.object_aware,
                    exclusion_mask=bg_params.exclusion_mask,
                )
                working, bg_model_2 = extract_background(working, refined_params)
                new_residual = np.std(bg_model_2)
                self._log_msg(
                    f"[{name}] BG pass {bg_pass}: residual {bg_residual:.4f} -> {new_residual:.4f}"
                )
                bg_residual = new_residual

            # Morphological background fallback for severe LP
            if bg_residual >= 0.05:
                self._log_msg(
                    f"[{name}] Polynomial BG insufficient (residual={bg_residual:.4f}), "
                    f"trying morphological fallback"
                )
                morph_result = self._morphological_background_subtract(working)
                if morph_result is not None:
                    morph_working, morph_bg = morph_result
                    morph_residual = float(np.std(morph_bg))
                    if morph_residual < bg_residual:
                        self._log_msg(
                            f"[{name}] Morphological BG better: {morph_residual:.4f} vs {bg_residual:.4f}"
                        )
                        working = morph_working
                        bg_residual = morph_residual

            self._quality_check(
                ProcessingStage.BACKGROUND_EXTRACTION,
                "bg_gradient_std",
                bg_residual,
                threshold=0.05,
                passed=bg_residual < 0.05,
                adjustment=f"Ran {bg_pass} pass(es)" if bg_pass > 1 else None,
            )

            # Validate background extraction output
            if not self._validate_stage_output(working, "background", channel):
                self._log_msg(f"[{name}] Background extraction failed validation, reverting")
                working = channel.copy()

        # Stage 2: Noise reduction
        if plan.denoise_params is not None:
            frac = frac_start + frac_range * 0.2
            progress(frac, f"Noise reduction [{name}]...")
            self._log_msg(
                f"[{name}] Noise reduction: "
                f"strength={plan.denoise_params.strength:.2f}"
            )

            before_stats = compute_channel_stats(working)
            working = denoise(working, plan.denoise_params)
            after_stats = compute_channel_stats(working)

            # Quality check: NR shouldn't destroy detail (std shouldn't drop too much)
            detail_ratio = after_stats["std"] / max(before_stats["std"], 1e-10)
            if detail_ratio < 0.3:
                # Too aggressive — re-run with less strength
                self._log_msg(
                    f"[{name}] NR too aggressive (detail_ratio={detail_ratio:.2f}), "
                    f"reducing strength"
                )
                reduced_params = DenoiseParams(
                    method=plan.denoise_params.method,
                    strength=plan.denoise_params.strength * 0.5,
                    detail_preservation=min(plan.denoise_params.detail_preservation + 0.2, 0.9),
                )
                working = denoise(channel.copy(), reduced_params)
                after_stats = compute_channel_stats(working)
                detail_ratio = after_stats["std"] / max(before_stats["std"], 1e-10)
                self._quality_check(
                    ProcessingStage.NOISE_REDUCTION,
                    "detail_preservation",
                    detail_ratio,
                    threshold=0.4,
                    passed=detail_ratio >= 0.4,
                    adjustment="Reduced NR strength by 50%",
                )
            else:
                self._quality_check(
                    ProcessingStage.NOISE_REDUCTION,
                    "detail_preservation",
                    detail_ratio,
                    threshold=0.4,
                    passed=True,
                )

        # Stage 3: Adaptive deconvolution
        if plan.deconvolution_params is not None:
            frac = frac_start + frac_range * 0.4
            progress(frac, f"Deconvolution [{name}]...")

            dp = plan.deconvolution_params
            pre_deconv = working.copy()

            if plan.use_spatial_deconv:
                # Spatially-varying PSF deconvolution — measures local PSF in
                # each zone and deconvolves with zone-specific PSF. Handles
                # field curvature, coma, astigmatism naturally.
                self._log_msg(
                    f"[{name}] Using spatially-varying PSF deconvolution "
                    f"(3x3 zones, fallback FWHM={dp.psf_fwhm:.1f}px)"
                )
                spatial_params = SpatialDeconvParams(
                    grid_zones=3,
                    iterations=dp.iterations,
                    regularization=dp.regularization,
                    deringing=dp.deringing,
                    deringing_amount=dp.deringing_amount,
                    fallback_fwhm=dp.psf_fwhm,
                    min_stars_per_zone=3,
                    blend_radius_fraction=0.25,
                )
                working = richardson_lucy_spatial(
                    working, spatial_params,
                    progress=lambda f, m: progress(
                        frac + frac_range * 0.25 * f, m
                    ),
                )

                self._quality_check(
                    ProcessingStage.DECONVOLUTION,
                    "spatial_zones",
                    9.0,  # 3x3 zones
                    threshold=50.0,
                    passed=True,
                    adjustment=f"Spatial deconv: 3x3 zones, {dp.iterations} iters each",
                )
            else:
                # Global deconvolution with adaptive iteration count and
                # edge taper fallback for small images / few stars.
                from scipy.ndimage import laplace
                baseline_laplacian_var = float(np.var(laplace(pre_deconv)))

                total_iters = 0
                max_iters = 20
                batch_size = 5
                best_result = working.copy()
                best_iters = 0
                regularization = dp.regularization

                while total_iters < max_iters:
                    batch_iters = min(batch_size, max_iters - total_iters)
                    batch_params = DeconvolutionParams(
                        psf_fwhm=dp.psf_fwhm,
                        iterations=batch_iters,
                        regularization=regularization,
                        deringing=dp.deringing,
                        deringing_amount=dp.deringing_amount,
                    )
                    working = richardson_lucy(working, batch_params)
                    total_iters += batch_iters

                    current_laplacian_var = float(np.var(laplace(working)))
                    ringing_ratio = current_laplacian_var / max(baseline_laplacian_var, 1e-10)
                    neg_frac = float(np.mean(working < 0))

                    self._log_msg(
                        f"[{name}] Deconv iter {total_iters}: "
                        f"ringing_ratio={ringing_ratio:.2f}, neg={neg_frac:.4f}"
                    )

                    if ringing_ratio > 2.0 or neg_frac > 0.003:
                        self._log_msg(
                            f"[{name}] Deconv stopped at {total_iters} iters "
                            f"(ringing detected), rolling back to {best_iters}"
                        )
                        working = best_result
                        break

                    best_result = working.copy()
                    best_iters = total_iters

                    if plan.snr < 15 and total_iters >= 15:
                        break
                    if plan.snr < 30 and total_iters >= 30:
                        break

                self._quality_check(
                    ProcessingStage.DECONVOLUTION,
                    "adaptive_iterations",
                    float(best_iters),
                    threshold=50.0,
                    passed=True,
                    adjustment=f"Adaptively chose {best_iters} iterations",
                )

                # Edge taper fallback for global deconv
                if best_iters > 0:
                    h, w = working.shape
                    cy, cx = h / 2.0, w / 2.0
                    yy, xx = np.mgrid[0:h, 0:w]
                    dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
                    taper = np.clip(1.0 - (dist - 0.6) / 0.5, 0.0, 1.0).astype(np.float32)
                    working = working * taper + pre_deconv * (1.0 - taper)
                    self._log_msg(
                        f"[{name}] Deconv edge taper applied "
                        f"(full center, fades 60%-100% radius)"
                    )

            working = np.clip(working, 0, 1)

        # Stage 4: Adaptive stretch
        if plan.stretch_params is not None:
            frac = frac_start + frac_range * 0.7
            progress(frac, f"Stretching [{name}]...")

            pre_stretch = working.copy()

            # Adaptive target brightness based on LP severity, SNR, and catalog hints
            base_target = 0.25
            if analysis.lp_severity == "severe":
                base_target = 0.18
            elif analysis.lp_severity == "moderate":
                base_target = 0.22

            # Catalog stretch hint
            hints = {}
            if analysis.primary_target:
                hints = analysis.primary_target.processing_hints
            stretch_hint = hints.get("stretch", "moderate")
            if stretch_hint == "gentle":
                base_target *= 0.75  # less aggressive for high-DR targets
            elif stretch_hint == "aggressive":
                base_target *= 1.2

            # Extreme-DR objects (M42 etc.) need the gentlest stretch to
            # preserve both core and outer nebulosity without blowing out.
            if full_plan.needs_hdr_merge and "hdr_merge" in self._enabled_stages:
                base_target *= 0.55  # very conservative

            if plan.snr < 8:
                base_target *= 0.8
            elif plan.snr > 50:
                base_target *= 1.15

            if analysis.has_blown_highlights:
                base_target *= 0.85

            target_median = max(0.10, min(0.30, base_target))
            self._log_msg(
                f"[{name}] Adaptive stretch target: {target_median:.3f} "
                f"(LP={analysis.lp_severity}, SNR={plan.snr:.1f})"
            )
            midtone = plan.stretch_params.midtone
            shadow_clip = plan.stretch_params.shadow_clip

            # Seed the midtone analytically so the loop converges immediately
            # instead of stepping geometrically from a fixed guess. auto_stretch
            # applies the black point then the MTF, so solve for the midtone that
            # maps the *scaled* median to the target (same math as
            # statistical_stretch). The loop below still refines/guards against
            # blown highlights and sky-dominated framing.
            try:
                from cosmica.core.stretch import _solve_midtone

                _st = compute_channel_stats(pre_stretch)
                _shadow0 = max(0.0, _st["median"] + shadow_clip * max(_st["mad"], 1e-8))
                _scale0 = 1.0 / max(1e-10, 1.0 - _shadow0)
                _scaled_med = max(1e-6, (_st["median"] - _shadow0) * _scale0)
                midtone = _solve_midtone(_scaled_med, target_median)
            except Exception as exc:
                self._log_msg(f"[{name}] Analytical midtone seed failed ({exc}), using default")

            best_result = None
            best_midtone = midtone
            best_error = float("inf")

            for attempt in range(12):
                params = StretchParams(
                    midtone=midtone, shadow_clip=shadow_clip, linked=False,
                )
                stretched = auto_stretch(pre_stretch.copy(), params)

                result_median = float(np.median(stretched))
                sat_frac = float(np.mean(stretched > 0.99))

                # For sky-dominated images (e.g. nebulae that don't fill
                # the frame), the overall median stays near 0 even when
                # the target region is properly stretched.  Use mean of
                # signal pixels as the quality metric in this case.
                signal_mask = stretched > 0.02
                signal_frac = float(np.mean(signal_mask))
                if signal_frac > 0.05 and result_median < 0.02:
                    # Sky-dominated image — judge by signal brightness
                    signal_brightness = float(np.mean(stretched[signal_mask]))
                    # Scale target for the signal region
                    effective_target = min(0.5, target_median / max(signal_frac, 0.1))
                    metric = signal_brightness
                    sky_dominated = True
                else:
                    metric = result_median
                    effective_target = target_median
                    sky_dominated = False

                # For HDR objects cap the effective target so the stretch
                # doesn't over-brighten the nebula and blow the core.
                if full_plan.needs_hdr_merge and "hdr_merge" in self._enabled_stages:
                    effective_target = min(effective_target, 0.30)

                error = abs(metric - effective_target)

                self._log_msg(
                    f"[{name}] Stretch attempt {attempt+1}: midtone={midtone:.4f}, "
                    f"median={result_median:.3f}, sat={sat_frac:.4f}"
                    + (f", signal={metric:.3f}/{effective_target:.3f}" if sky_dominated else "")
                )

                # Track best result (closest to target without blown highlights)
                if sat_frac < 0.03 and error < best_error:
                    best_error = error
                    best_result = stretched
                    best_midtone = midtone

                # Close enough — stop searching
                if error < 0.05 and sat_frac < 0.03:
                    break

                # Adjust midtone: lower = brighter result
                if metric < effective_target * 0.8:
                    midtone *= 0.6  # too dark, make more aggressive
                elif metric > effective_target * 1.3:
                    midtone *= 1.4  # too bright, pull back
                elif sat_frac > 0.03:
                    midtone *= 1.3  # blown highlights, ease off
                else:
                    break  # good enough

                # For HDR cap midtone so the core doesn't blow out completely.
                min_midtone = 0.020 if full_plan.needs_hdr_merge else 0.001
                midtone = max(min_midtone, min(0.5, midtone))

            if best_result is not None:
                working = best_result
            else:
                working = auto_stretch(pre_stretch, plan.stretch_params)

            # Background lift: raise the floor slightly so the background
            # isn't completely black.  Common astrophotography technique.
            if full_plan.needs_hdr_merge:
                bg_pedestal = 0.005
                working = np.clip(working + bg_pedestal, 0, 1)
                self._log_msg(f"[{name}] Background lift: +{bg_pedestal:.3f}")

            final_median = float(np.median(working))
            final_signal = float(np.mean(working[working > 0.02])) if np.any(working > 0.02) else 0.0
            self._log_msg(
                f"[{name}] Stretch final: midtone={best_midtone:.3f}, "
                f"median={final_median:.3f}, signal_mean={final_signal:.3f}"
            )
            self._quality_check(
                ProcessingStage.STRETCH,
                "target_brightness",
                max(final_median, final_signal * 0.5),
                threshold=target_median,
                passed=(final_signal > 0.15 or abs(final_median - target_median) < 0.1
                        or (sky_dominated and metric > effective_target * 0.5)),
                adjustment=f"Adaptively chose midtone={best_midtone:.3f}",
            )

        # Core protection for HDR objects: apply the selected tonemap operator
        # to the linear pre-stretch data and blend.  Preserves Trapezium /
        # bright core detail while bringing out faint outer nebulosity.
        if full_plan.needs_hdr_merge and "hdr_merge" in self._enabled_stages:
            from cosmica.core.hdr_operators import apply_hdr
            operator = full_plan.hdr_operator
            params = full_plan.hdr_params
            hdr_result = apply_hdr(pre_stretch, operator, params)
            # Blend with existing working via the core protection mask
            if operator == "core_blend":
                working = hdr_result
            else:
                p99 = float(np.percentile(pre_stretch, 99))
                core_linear = (pre_stretch > p99 * 0.3).astype(np.float32)
                if np.any(core_linear):
                    from scipy.ndimage import gaussian_filter
                    core_mask = gaussian_filter(core_linear, sigma=max(8, min(working.shape) * 0.015))
                    core_mask = np.clip(core_mask, 0, 1)
                    working = working * (1.0 - core_mask) + hdr_result * core_mask
                    core_blend_frac = float(np.mean(core_mask))
                    self._log_msg(
                        f"[{name}] HDR ({operator}) applied over "
                        f"{core_blend_frac:.4f} of the frame"
                    )
                else:
                    working = hdr_result

        # Stage 5: Local contrast
        if plan.local_contrast_params is not None:
            frac = frac_start + frac_range * 0.9
            progress(frac, f"Local contrast [{name}]...")
            self._log_msg(f"[{name}] Local contrast enhancement")
            working = local_contrast_enhance(working, plan.local_contrast_params)

        return np.clip(working, 0, 1).astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Helper methods                                                     #
    # ------------------------------------------------------------------ #

    def _safe_plate_solve(
        self, data: np.ndarray, params: PlateSolveParams
    ) -> PlateSolveResult | None:
        """Run plate solving in a subprocess to protect against segfaults.

        OpenCV's native code can segfault on certain image data.
        By running in a child process we catch the crash gracefully.
        """
        import multiprocessing as mp

        parent_conn, child_conn = mp.Pipe(duplex=False)
        proc = mp.Process(
            target=_plate_solve_worker,
            args=(child_conn, data, params),
            daemon=True,
        )
        proc.start()
        proc.join(timeout=30)  # 30 second timeout

        if proc.is_alive():
            self._log_msg("Plate solve timed out — killing subprocess")
            proc.kill()
            proc.join(timeout=5)
            parent_conn.close()
            return PlateSolveResult(success=False)

        if proc.exitcode != 0:
            self._log_msg(f"Plate solve subprocess crashed (exit code {proc.exitcode})")
            parent_conn.close()
            return PlateSolveResult(success=False)

        if parent_conn.poll():
            result = parent_conn.recv()
            parent_conn.close()
            return result

        parent_conn.close()
        return PlateSolveResult(success=False)

    def _extract_bg_only(
        self,
        channel: np.ndarray,
        plan: ChannelPlan,
        analysis: ImageAnalysis,
    ) -> np.ndarray:
        """Run only the background extraction stage for a channel.

        Used by the two-pass approach in _execute so that linked signal
        rescaling can be applied after BG extraction of ALL channels.
        """
        if plan.background_params is None:
            return channel.copy()

        working = channel.copy()
        name = plan.channel_name
        bg_params = plan.background_params

        self._log_msg(f"[{name}] Background extraction")
        working, bg_model = extract_background(working, bg_params)

        # Iterative refinement
        bg_residual = np.std(bg_model)
        max_bg_passes = 3
        bg_pass = 1
        while bg_residual >= 0.05 and bg_pass < max_bg_passes:
            bg_pass += 1
            refined_params = BackgroundParams(
                grid_size=min(bg_params.grid_size + 4, 24),
                box_size=bg_params.box_size,
                polynomial_order=min(bg_params.polynomial_order + 1, 6),
                sigma_clip=bg_params.sigma_clip,
                smoothing=bg_params.smoothing,
                object_aware=bg_params.object_aware,
                exclusion_mask=bg_params.exclusion_mask,
            )
            working, bg_model_2 = extract_background(working, refined_params)
            new_residual = np.std(bg_model_2)
            self._log_msg(
                f"[{name}] BG pass {bg_pass}: residual {bg_residual:.4f} -> {new_residual:.4f}"
            )
            bg_residual = new_residual

        # Morphological fallback
        if bg_residual >= 0.05:
            morph_result = self._morphological_background_subtract(working)
            if morph_result is not None:
                morph_working, morph_bg = morph_result
                morph_residual = float(np.std(morph_bg))
                if morph_residual < bg_residual:
                    self._log_msg(
                        f"[{name}] Morphological BG better: {morph_residual:.4f} vs {bg_residual:.4f}"
                    )
                    working = morph_working
                    bg_residual = morph_residual

        self._quality_check(
            ProcessingStage.BACKGROUND_EXTRACTION,
            "bg_gradient_std",
            bg_residual,
            threshold=0.05,
            passed=bg_residual < 0.05,
            adjustment=f"Ran {bg_pass} pass(es)" if bg_pass > 1 else None,
        )

        # Validate
        if not self._validate_stage_output(working, "background", channel):
            self._log_msg(f"[{name}] Background extraction failed validation, reverting")
            return channel.copy()

        return working

    def _morphological_background_subtract(
        self, channel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Estimate background via morphological approach (median filter).

        Uses a large kernel to create a smooth background model that
        captures gradients from LP without being affected by stars.
        """
        try:
            from scipy.ndimage import gaussian_filter, median_filter
            h, w = channel.shape
            kernel = max(64, min(h, w) // 8)
            # Make kernel odd
            if kernel % 2 == 0:
                kernel += 1
            bg = median_filter(channel, size=kernel)
            bg = gaussian_filter(bg, sigma=kernel / 4)
            corrected = channel - bg
            c_min = float(np.percentile(corrected, 0.5))
            corrected = corrected - c_min
            corrected = np.clip(corrected, 0, 1).astype(np.float32)
            return corrected, bg
        except Exception as exc:
            self._log_msg(f"Morphological BG failed: {exc}")
            return None

    def _validate_stage_output(
        self, result: np.ndarray, stage_name: str, original: np.ndarray
    ) -> bool:
        """Validate that a processing stage did not corrupt the data."""
        if np.any(~np.isfinite(result)):
            self._log_msg(f"Validation FAIL [{stage_name}]: NaN/Inf detected")
            return False
        orig_range = float(np.percentile(original, 95) - np.percentile(original, 5))
        result_range = float(np.percentile(result, 95) - np.percentile(result, 5))
        if orig_range > 0 and result_range / orig_range < 0.1:
            self._log_msg(f"Validation FAIL [{stage_name}]: Lost >90% of signal range")
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Quality checks                                                     #
    # ------------------------------------------------------------------ #

    def _quality_check(
        self,
        stage: ProcessingStage,
        metric_name: str,
        value: float,
        threshold: float,
        passed: bool,
        adjustment: str | None = None,
    ) -> None:
        """Record a quality check result."""
        qc = QualityCheck(
            stage=stage,
            passed=passed,
            metric_name=metric_name,
            metric_value=value,
            threshold=threshold,
            adjustment=adjustment,
        )
        self._quality_checks.append(qc)
        status = "PASS" if passed else "ADJUST"
        self._log_msg(
            f"  QC [{stage.value}] {metric_name}={value:.4f} "
            f"(threshold={threshold:.4f}) → {status}"
        )

    def _check_color_balance(self, image: np.ndarray, stage: ProcessingStage) -> None:
        """Check if color channels are reasonably balanced after processing."""
        if image.ndim != 3 or image.shape[0] < 3:
            return
        medians = [float(np.median(image[ch])) for ch in range(3)]
        spread = max(medians) - min(medians)
        self._quality_check(
            stage,
            "color_balance_spread",
            spread,
            threshold=0.3,
            passed=spread < 0.3,
        )

    # ------------------------------------------------------------------ #
    #  Logging                                                            #
    # ------------------------------------------------------------------ #

    def _log_msg(self, msg: str) -> None:
        """Log a message both to Python logger and internal log."""
        log.info(msg)
        self._log.append(msg)
