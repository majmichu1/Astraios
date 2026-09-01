# Changelog

All notable changes to Astraios are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

#### Documentation, website and trust
- **Project facts** (`docs/project-facts.md`): one dated source of truth for every public claim (version, platforms, CI-tested versus user-verified compatibility, GPU and CPU behaviour, network use and uploads, formats, AI models, limitations). README, the website, the FAQ, the structured data and `llms.txt` all repeat facts from it.
- **README** rewritten as a landing document: definition, alpha status, current screenshot (synthetic field, labelled), download, install matrix, verified compatibility, workflow, features summary, measurements without fabricated speed-ups, a dated comparison, limitations and FAQ. The full feature inventory moved to `docs/features.md`; install details to `docs/installation.md`.
- **Stale text removed**: "source code only" installation, "AI coming soon", the "AI Pro Tab (Pro License Required)" section, an unclosed code fence in `docs/getting-started.md`, "everything runs on the GPU", and a `SECURITY.md` that described a licence system and denied every upload while the astrometry.net backend uploads a mono FITS of the image being solved.
- **Website** at https://majmichu1.github.io/Astraios/ (`site/`, deployed by `.github/workflows/pages.yml`): home, features, install, benchmarks, Astraios vs PixInsight, Astraios vs Siril and FAQ, with canonical URLs, Open Graph and Twitter metadata, JSON-LD (`SoftwareApplication`, `FAQPage`), a sitemap, a scoped `llms.txt`, skip link, visible focus, reduced-motion support and scrollable tables. No analytics, cookies or external resources. `scripts/check_site.py` validates it before every deployment.
- **Comparisons** with PixInsight 1.9.4 and Siril 1.4.4 are dated, sourced from official pages, and separate core-engine GPU acceleration from paid third-party plugins. `docs/BENCHMARKS.md` publishes the like-for-like quality measurement against Siril (same subs, settings and reference) and states that no speed comparison is published yet.
- **License file** restored to the canonical GNU GPL v3 text. The previous file carried an inserted "additional permission" that restated section 4 and a changed preamble line, which stopped GitHub from recognising the licence. The declared licence (`GPL-3.0-or-later` in `pyproject.toml`) is unchanged.
- **Community and packaging metadata**: `CODE_OF_CONDUCT.md`, `SUPPORT.md`, `CITATION.cff`, an issue-template chooser that points questions at Discussions and vulnerabilities at private reporting, a compatibility-report form that separates CI runs from real hardware, and complete `project.urls`, keywords and classifiers in `pyproject.toml`.

#### Release integrity
- **Checksum sidecars**: tagged releases built by `.github/workflows/build.yml` now publish one `<asset>.sha256` per asset, computed from the final artifacts after every build job. The in-app updater already verified downloads against exactly that sidecar name; until now no release carried one. `scripts/check_release_checksums.py` validates the sidecars before upload and `tests/test_updater/test_release_checksums.py` covers the format, the updater's parser and the installers' wheel lookup, which must skip sidecars.
- Alpha releases stay normal (non-prerelease) GitHub releases on purpose: the updater and the Linux and macOS installers resolve `/releases/latest`, which ignores prereleases.

## [0.1.25] - 2026-08-25

### Added

#### Getting started
- **Guided Processing** (`astraios/core/guided.py`, `astraios/ui/dialogs/guided_dialog.py`): A step-by-step wizard (Ctrl+G, or the button at the top of the Tools panel) that walks the correct order one stage at a time: trim, gradient, colour, sharpen, denoise, stretch, saturation. Each step explains what it does in plain language, suggests settings measured from your own image, previews before committing, and can be skipped or stepped back. The order is the professional one, so the common mistakes (colour balancing after a stretch, saturating linear data) are not offered.
- **Smart telescope recognition** (`astraios/core/smart_telescope.py`): Seestar S50/S30, Dwarf II/3, Vespera, Stellina and Unistellar frames are identified from their headers, and already-stacked files are pointed straight at the guided workflow. Identity is matched across nine header keys, because vendors disagree about where they write it.
- **Teaching tooltips on every control**: All 220 controls in the Tools panel now carry a plain-language explanation of what the setting does and what raising or lowering it changes.

#### Platforms
- **macOS support** (`packaging/macos/install-astraios-macos.sh`): An installer that creates a real `Astraios.app` for Launchpad and Spotlight. Apple Silicon runs on the GPU through PyTorch's Metal (MPS) backend; Intel Macs run on CPU. Both architectures are validated on real Apple hardware in CI, including a check that every GPU tool produces the same pixels on Metal as on the CPU.

#### Core processing
- **Mosaic gradient matching** (`astraios/core/mosaic.py`): `match_gradient` was a documented setting that nothing read, so panels differing by a sloped sky kept that slope into the output. Photometric normalization cannot cover for it, deriving one scale factor per panel and so correcting only uniform brightness. Each panel is now matched to the composite of those already placed by fitting a plane over the overlap. On a two-panel test with a 0.08 tilt injected, 99% of it is removed.
- **Reinhard local and chromatic adaptation** (`astraios/core/hdr_operators.py`): `light_adapt` and `color_adapt` were declared and ignored. `light_adapt` is the fix for a blown core: a core clipped across its whole area at 0 comes back entirely below clipping by 0.5, with the background essentially unmoved.
- **Gaussian drizzle kernel** (`astraios/core/drizzle.py`): `pixel_weight="gaussian"` now applies a real falloff from the drop centre instead of silently using the square kernel.
- **Multi-session integration methods** (`astraios/core/multi_session.py`): MEDIAN and AVERAGE were both ignored; every stack came out a weighted average. MEDIAN survives one bad session in a way no average does.
- **Lens field of view** (`astraios/core/lens_distortion.py`): `fov` is honoured, so knowing your field is enough without knowing your sensor size.
- **Median denoise kernel** (`astraios/core/denoise.py`): `median_kernel` can now set the window explicitly instead of only deriving it from strength.

#### AI
- **AI Super-Resolution is genuinely neural** (`astraios/ai/models/rrdbnet.py`): It imported RRDBNet from `basicsr`, which is not a declared dependency and does not import against modern torchvision, so every user silently received bicubic interpolation from a menu entry advertising a neural upscale. The architecture is now vendored (BSD-3). Verified against the real released checkpoint: 702 of 702 tensors load with no missing or unexpected keys.

### Fixed

- **Colour calibration left a magenta cast** (`astraios/core/color_calibration.py`): White balance is multiplicative and background neutralization is additive, so the two do not commute. They ran in the wrong order. Measured background green excess after correction moved from -22.8% to -1.2% on the worst case.
- **Morphology gave different results on GPU and CPU** (`astraios/core/morphology.py`): The GPU path used a square window for every structuring element while the CPU path used the shape you asked for. On Apple Silicon this meant visibly wrong pixels from the default settings.
- **Eleven tools could not be cancelled** (`astraios/ui/main_window.py`, `astraios/ai/inference/cosmic_clarity.py`): The progress callback is also the cancellation checkpoint, so tools that never called it ignored Cancel entirely until they finished. Among them were the slowest ones in the application: denoise, star reduction, wavelet sharpen, local contrast.
- **First upscale gave no sign of the 67 MB weight download** it now performs.
- **Guided workflow aliased the caller's image** rather than copying it.

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

### Changed

#### Audit round 2 — hardening and debt
- **Licensing removed entirely**: Astraios is free open-source software — no license checks, no Pro tier, no subscriptions.
- **Updater hardened**: fail-closed SHA-256 verification (unverified installers are rejected), checksum sidecars are never picked as the installer, notify-only update check 5 s after startup on a background thread.
- **Super-resolution weights**: downloaded with a 30 s timeout, 3 retries and pinned SHA-256 of the official Real-ESRGAN checkpoints.
- **Thread hygiene**: processing-history replay, HDR compose, mosaic panel loading, auto-import header scan all moved off the GUI thread; mask preview debounced.
- **Full-depth processing**: median denoise at uint16, color CLAHE via float Lab + uint16 L (no more 8-bit banding), morphology CIRCLE/DIAMOND use exact CPU kernels (GPU max-pool is square-only).
- **Lint**: 646 → ~270 (style-only remainder); every `zip()` is now `strict=True`; pep8-naming deselected and UP042 ignored by documented config decision.
- **Tests**: the 10 skipped dialog construction tests are now real tests (UI suite runs with zero skips); updater verification tests added.

### Fixed

#### Full-codebase audit fixes
- **Silent image corruption in EZ presets** (`astraios/core/ez_scripts.py`): The "background" step subtracted the whole `(corrected, model)` return tuple from the image — mono input became a garbage 2-channel array, every later step ran on it. 4 of the 6 built-in presets affected. Now takes the corrected image, like `batch.py` always did.
- **GPU stacking ignored parameters** (`astraios/core/stacking.py`): The GPU path always returned a plain mean — MEDIAN integration and WEIGHTED_AVERAGE frame weights were silently discarded whenever a GPU was present, and `params.use_gpu` was ignored. Rejection kernels now return kept-masks and a shared `_gpu_integrate` applies the method/weights identically to CPU.
- **GPU sigma-clip centered on mean** (`astraios/core/stacking.py`): astropy's CPU `SigmaClip` centers on the median; the two paths rejected different pixels for the same kappa. GPU now median-centers.
- **Stacking crashed on color stacks** (`astraios/core/stacking.py`): ESD and MIN_MAX rejection indexed frames as `(N, H, W)` — every OSC stack hit `IndexError` on the CPU path, and MIN_MAX with 2 frames rejected everything into an all-black result. Both kernels now handle `(N, C, H, W)`.
- **Horizontal banding in tiled stacking** (`astraios/core/image_io.py`, `stacking.py`): Foreign float FITS tiles were min-max stretched by each tile's own range. Whole-file range is now computed once per frame and applied to all tiles.
- **AI Denoise never used its model** (`astraios/ai/inference/denoise.py`, `model_manager.py`): The bundled `cosmica_denoise_v1.pt` wasn't in the candidate list, its DenoiseUNet keys didn't fit a bare UNet, and the registry sha256/size were stale (re-download always failed integrity). All three fixed; architecture is now detected from the weights. "Full" tile size (0) and `overlap >= tile_size` no longer crash with `IndexError`.
- **Wavelet denoise GPU/CPU divergence** (`astraios/core/denoise.py`): GPU left the finest (most noise-dominated) scale unthresholded; now thresholded with the strongest factor like the CPU path.
- **Drago tonemap lifted blacks 30%** (`astraios/core/hdr_operators.py`): Missing the `log10(1+L)/log10(1+Lmax)` factor from Drago et al. 2003 — output was compressed into [0.30, 1].
- **pixel math crashed on GPU conditionals** (`astraios/core/pixel_math.py`): `T if T > 0.5 else X` took the truth value of a multi-element tensor; now uses `torch.where`. The module-global function-table swap (not thread-safe) is replaced by an explicit table parameter.
- **XISF writer produced unparseable files** (`astraios/core/image_io.py`): Header values with `&`/`<`/quotes weren't XML-escaped, and the attachment-offset patch desynced when the digit count changed. Escaping + fixed-point offset loop.
- **Re-saving FITS deleted astrometry** (`astraios/core/image_io.py`): `save_fits` copied a 9-key whitelist only; all WCS keywords (CRVAL/CD/SIP/…) are now preserved. `EXPTIME = 0.0` (bias) no longer treated as missing.
- **Preset round-trip lost enums** (`astraios/core/presets.py`): `NormalizeMethod`, `RegistrationMode`, `NormalizationMethod` were missing from the enum map — loaded presets silently switched tools to fallback behaviour. Unknown fields and corrupt JSON are now tolerated with warnings.
- **Click-and-crash, round 2** (`astraios/ui/main_window.py`, `tools_panel.py`, `project.py`): Implemented 10 missing panel/project APIs (Run Calibration, Debayer, LRGB, Continuum, Multi-Session, Blink incl. Shift+B, Macro record/stop, plate-solve callback).
- **Exit crash + stale results** (`astraios/ui/main_window.py`): `closeEvent` now stops running workers and timers; a generation counter drops results from superseded workers; Ctrl+O collision between Open Image button and Open Project action removed; export/macro/blink slots no longer terminate the app on I/O errors.
- **Project file corruption on crash** (`astraios/core/project.py`): Save is now atomic (temp file + fsync + rename); load reports which field is broken in hand-edited files.
- **Live stack crashed on mixed frame sizes** (`astraios/core/live_stack.py`); **multi-session crashed on mono+color mixes** (`astraios/core/multi_session.py`) — both now handled.
- **Star mask was a no-op** (`astraios/ui/main_window.py`): The generated mask was discarded after a success message; it is now registered and activated.
- **Degenerate parameter guards**: deconvolution `psf_fwhm <= 0` (all-NaN output) and `iterations <= 0` (silent no-op); HDR Mertens `sigma = 0` (all-white output).
- **Security**: astrometry.net API key/uploads/WCS downloads moved from plaintext HTTP to HTTPS.
- **CI**: New quality-gate workflow (ruff bug-classes + mypy + full test suite) runs on every push/PR; previously nothing ran between releases. Fixed `poetry install --with dev` → `--extras dev` in CONTRIBUTING.md.

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
