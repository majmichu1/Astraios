# Changelog

All notable changes to Astraios are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

#### Core processing
- **HDR operators** (`astraios/core/hdr_operators.py`): Three selectable HDR tonemap operators — Reinhard, Drago, and Core-blend — for handling extreme dynamic range objects like M42. QComboBox selector in Smart Processor dialog.
- **WCS dict normalisation** (`astraios/core/wcs.py`): `normalise_wcs_dict()` bridges the two incompatible WCS key formats (`ra_center`/`dec_center` from plate solving vs `ra`/`dec` from catalog queries). All callers updated.
- **FWHM map rewrite** (`astraios/core/analysis/fwhm_map.py`): Replaced per-star `curve_fit` with vectorised `scipy.ndimage.label` + radial profile. Now measures real FWHM instead of fabricating it from ellipticity. ~40× faster on large star fields.
- **Plate solve** (`astraios/core/plate_solve.py`): Local `plate_solve(scale_hint=...)` path now raises `NotImplementedError` (was silent failure returning success=False without attempting solve).
- **NaN/inf guard** (`astraios/core/drizzle.py`): Non-finite pixel values replaced with 0 before processing.
- **Aperture photometry** (`astraios/core/aperture_photometry.py`): `aperture_radius` documented as pixels (10.0 default) — no unit ambiguity.
- **PSF curve_fit bounds** (`astraios/core/analysis/psf.py`): Bounds already comprehensive (`sigma: 0.5–r`, `amplitude: 0–amp_max`, etc.) — no change needed.
- **Denoise dispatch** (`astraios/core/denoise.py`): `DenoiseMethod` enum with 3-way dispatch (NLM, WAVELET, TGV) + MEDIAN. TGV falls back to wavelet on `NotImplementedError`.

#### Image I/O & color
- **ICC profiles** (`astraios/core/color_management.py` + `resources/icc/`): Real Adobe RGB and Display P3 ICC v4.3 profiles (~570B each) bundled in `resources/icc/`. `_load_bundled()` fallback to sRGB if missing. Old code was assigning sRGB to both display and working profile slots.
- **Manual RGB white balance** (`astraios/ui/panels/tools_panel.py`): R/G/B spin boxes (0.1–5.0) in the Color Calibration section, visible only when "Manual RGB" method is selected. Wired to `ColorCalibrationParams.custom_rgb`.
- **SPCC data pipeline** (`astraios/ui/main_window.py`): `_wcs_overlay_stars` now stores 4-tuples `(x, y, mag, bp_rp)` with real Gaia BP-RP values from `StarCatalogEntry.bp_mag`/`rp_mag`. SPCC handler passes actual BP-RP to temperature conversion instead of a fake proxy from G magnitude.
- **SPCC UI getter** (`astraios/ui/panels/tools_panel.py`): `get_spcc_params()` added — was missing, causing `AttributeError` on "Run SPCC" click.
- **HOO palette** (`astraios/ui/dialogs/channel_combine_dialog.py`): Weighted average for double-OIII in HOO mode via `channel_counts`.

#### UI & workflow
- **Processing graph lock** (`astraios/core/processing_graph.py` + `astraios/ui/dialogs/processing_graph_dialog.py`): `ProcessNode.locked` field prevents cache invalidation for locked nodes. Lock/unlock button in the processing graph dialog with 🔒/🔓 state.
- **Channel combine live preview** (`astraios/ui/dialogs/channel_combine_dialog.py`): 300ms debounced thumbnail preview updates as files, weights, palette, or normalisation change. Caches loaded channel data.
- **Star reduction controls** (`astraios/ui/panels/tools_panel.py`): Kernel type combo (Elliptical/Circular/Square/Diamond), iterations slider (1–10), and protect core checkbox. `kernel_type` field in `StarReductionParams`, mapped to corresponding `cv2.MORPH_*` constants.
- **Drizzle getter** (`astraios/ui/panels/tools_panel.py`): `get_drizzle_params()` added — was missing, causing `AttributeError` when toggling drizzle.
- **Curves per-channel cache** (`astraios/ui/widgets/curves_widget.py`): `CurveEditor.set_channel()` saves current points → switches cache → loads cached points. Points preserved across channel switches.
- **Workflow bar Transform tab** (`astraios/ui/widgets/workflow_bar.py`): Transform step added to `_STEPS` and `_STEP_TO_TAB` (was hidden).
- **Manual gamma** (`astraios/ui/dialogs/color_settings_dialog.py`): `QLabel("2.2")` → `QDoubleSpinBox(1.0–3.5)` with persistence via `get_config()`.

#### Scripting & batch
- **Pipeline mask support** (`astraios/core/scripting.py` + `astraios/core/batch.py`): `PipelineStep.mask_name` field. Playback accepts `masks` dict. `record_step()` and `apply_pipeline_to_image()` updated.

#### AI module
- **CosmicClarity download** (`astraios/ai/inference/cosmic_clarity.py`): Platform-aware download URLs (Linux/macOS/Windows). Atomic tempfile + rename + chmod. No more silent failures on platform mismatch.

### Changed

#### Performance improvements
- **GPU migration assessment** (Phase 4): All 9 candidates evaluated against ≥1.2× benchmark gate across 1MP/8MP/32MP. None qualified — `local_normalization` already GPU, remaining are OpenCV C++ or per-star iterative (curve_fit). Data transfer overhead kills ROI.
- **Undo depth** (`astraios/core/undo.py`): `MAX_UNDO_DEPTH` reduced from 50 to 20 (4.8GB → 1.9GB for 8MP colour images).
- **SuperBias vectorisation** (`astraios/core/superbias.py`): Multichannel column-pattern FPN computation vectorised — removed per-channel median/centering loop.

#### HTTP reliability
- **Retry with exponential backoff** (`astraios/core/plate_solve.py` + `astraios/core/star_catalog.py`): Shared `_request_with_retry()` wraps all `urllib.request` calls. Retries on 429 (rate limit), 5xx, and network errors with 1s/2s/4s backoff. Applied to API calls, multipart uploads, WCS downloads, and Vizier TAP queries.

#### Tilt analysis
- **Ellipticity threshold** (`astraios/core/analysis/tilt_analysis.py`): `MAX_ELLIPTICITY = 0.30` as named constant for guiding issue detection.

#### Object SNR
- **Smart Processor** (`astraios/ai/smart_processor.py`): `object_snr` now uses `bg + 3σ` object mask (was P95 global threshold, which was background-biased). Object SNR = `max(0, (obj_median − bg) / noise)`.

### Fixed

#### Click-and-crash (Phase 1)
- **Live stack** (`astraios/ui/dialogs/live_stack_dialog.py`): Guard `None`/non-ndarray/empty before accessing `.shape`.
- **Smart process dialog** (`astraios/ui/dialogs/smart_process_dialog.py`): `error` signal, `try/except` in `run()`, Cancel button, `request_cancel()`, `requestInterruption()` for thread-safe cancel.
- **Python console** (`astraios/ui/widgets/python_console.py`): eval/exec runs in daemon thread with timeout (default 5s). `set_timeout()` in namespace.
- **DSO catalog** (`astraios/core/dso_catalog.py`): Circular FOV filter (`dra² + ddec² < half²`). Was rectangular box → lost ~21% of corner objects.
- **Processing graph** (`astraios/core/processing_graph.py`): `update_params()`, `update_enabled()`, `_cache_params_hash` safety net.
- **Main window error handling** (`astraios/ui/main_window.py`): Per-exception `try/except` in `_open_project`, `_save_project` logs at `info` level, `_save_as` refreshes UI.

#### Star handling
- **Star removal RGGB** (`astraios/ai/inference/star_removal.py`): Rec.709 luminance handles C≥4 (RGGB Bayer) by averaging G1+G2.
- **Star reduction kernel** (`astraios/core/star_reduction.py`): Morphology kernel now configurable via `StarReductionParams.kernel_type` instead of hardcoded `cv2.MORPH_ELLIPSE`.

#### Equipment parsing
- **Sexagesimal RA/Dec** (`astraios/core/equipment.py`): String values like `"05 35 17"` parsed via `astropy.coordinates.SkyCoord(unit=(u.hourangle, u.deg))`. Numeric values pass through unchanged.

#### UI paper-cuts (Phase 5)
- **Histogram NaN guards** (`astraios/ui/widgets/histogram.py`): `np.nan_to_num` + `float()` cast prevents QPainter from crashing on NaN coordinates.
- **Trackpad horizontal scroll** (`astraios/ui/widgets/image_canvas.py`): `wheelEvent` handles `angleDelta().x()` for horizontal scroll on trackpads.
- **Score button re-enable** (`astraios/ui/dialogs/subframe_dialog.py`): `_score_btn` re-enabled on error (was staying disabled forever).
- **Batch cancel** (`astraios/ui/dialogs/batch_preprocess_dialog.py`): `cancel()` calls `QThread.requestInterruption()` in addition to flag.
- **PixelMath per-channel early-return** (`astraios/ui/dialogs/pixelmath_dialog.py`): Single-channel result returned without evaluating other channels.

### Test Improvements

- **859 tests pass** (was 717 before Phase 1). All core tests pass with `CUDA_VISIBLE_DEVICES="" --ignore=tests/test_ui/`.
- **Phase 1 regression tests** (`tests/test_ui/test_phase1_fixes.py`): 10 regression tests covering click-and-crash fixes.
- **FWHM map tests** (`tests/test_core/test_analysis.py`): Updated for vectorised FWHM implementation.
- **Plate solve tests** (`tests/test_core/test_plate_solve.py`): Updated for `NotImplementedError` on `scale_hint`.

### Removed

- **`INTER_AREA` in transforms**: Already absent from codebase — no change needed.
- **Dead code in `plate_solve()`**: Local `plate_solve(scale_hint=...)` path replaced with `NotImplementedError`.

### Infrastructure

- **Phase 4 assessment completed**: No GPU migrations cleared the ≥1.2× benchmark gate. Documented in `TODO_BUGS.md`.
- **Comprehensive bug audit** (`TODO_BUGS.md`): 67-bug audit across 8 phases with severity, file locations, and fix status.
- **AGENTS.md** updated with project conventions, architecture, and critical rules for future sessions.

### Fixed in Phase 2 (Smart Processor)

- Object SNR now uses proper background modelling with `bg + 3σ` object mask.
- HDR core protection with 3 selectable operators (Reinhard, Drago, Core-blend).
- Midtone clamp already correct — no change.
- WCS dict normalisation applied at all caller boundaries.

### Fixed in Phase 3 (Data Integrity)

- ICC profiles: real Adobe RGB and Display P3 instead of sRGB-for-both.
- Plate solve: `NotImplementedError` on unsupported path.
- CosmicClarity: platform-aware download with atomic write.
- Equipment: sexagesimal RA/Dec parsing.
- Star removal: RGGB Bayer luminance fix.
- Drizzle: NaN/inf guard.
- Undo: ring buffer depth 50→20.
- FWHM map: vectorised + real FWHM measurement.

### Fixed in Phase 5 (UX/Worker Thread)

- Smart process dialog: `requestInterruption()` on cancel.
- PixelMath: per-channel early-return.
- Histogram: NaN guards.
- Image canvas: trackpad horizontal scroll.
- Subframe dialog: score button re-enable on error.
- Batch preprocess: cancel calls `requestInterruption()`.
- Curves widget: per-channel points cache.
- Workflow bar: Transform tab visible.

### Fixed in Phase 6 (Plugin/Extension)

- Manual white-ref PCC: RGB spin boxes wired to `custom_rgb`.
- HTTP retry: exponential backoff on all astrometry.net and Vizier calls.
- SuperBias: vectorised multichannel column pattern.
- Import ordering in `color_calibration.py`.
- Misleading `"retrying"` log messages in `star_catalog.py`.

### Fixed in Phase 7 (PixInsight Parity)

- SPCC: real Gaia BP-RP values used for temperature conversion.
- SPCC: `get_spcc_params()` method added (was crashing).
- Drizzle: `get_drizzle_params()` method added (was crashing).
- Star reduction: kernel type UI + protect core + iterations.
- Processing graph: lock/unlock for cache preservation.
- Channel combine: live thumbnail preview.
