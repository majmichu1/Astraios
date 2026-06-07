# Cosmica — Comprehensive Bug Audit & Plan

> **Persistence file** — survives chat compaction. Read first if context lost.

## Locked Decisions
- **HDR core operators**: Reinhard + Drago + Core-blend (QComboBox in Smart Processor)
- **GPU migration**: ≥1.2× faster in ALL 3 image sizes (1MP, 8MP, 32MP) — benchmark gate
- **Commits**: one per fix (~85 atomic), for clean `git revert`
- **Image format**: float32 `[0,1]`, mono `(H,W)`, color `(C,H,W)` channels-first
- **GPU access**: always through `get_device_manager()` — never `torch.cuda.*` directly
- **Worker pattern**: `_start_worker(fn, *args, on_done=callback)` with `progress=None`
- **PyQt6 dialog return**: compare against `QDialog.DialogCode.Accepted` (int 1)

## Severity Counts
| Severity | Count |
|----------|-------|
| CRITICAL (click/crash/data corruption) | 10 |
| HIGH (broken feature, silent failure) | 27 |
| MEDIUM (CPU bottleneck, UX paper-cut) | 20 |
| LOW/cleanup | ~10 |
| **Total** | **~67** |

---

## Phase 1 — Click-and-Crash ✅ DONE (9 fixes, 10 regression tests, 859 pass)

| Bug | File | Fix |
|-----|------|-----|
| C1 | `live_stack_dialog.py:307` | Guard: `arr is None or not isinstance or arr.size == 0` |
| C2 | `smart_process_dialog.py` | Added `error` signal, `try/except` in `run()`, Cancel button, `request_cancel()` |
| C3 | `tools_panel.py:1540` + `denoise.py` | DenoiseMethod.TGV + MEDIAN, proper 3-way dispatch, TGV→wavelet fallback |
| C4 | `python_console.py` | Daemon thread timeout (default 5s), `set_timeout()` in namespace |
| C7 | `dso_catalog.py` | Circular FOV: `dra² + ddec² < half²` (was rectangular box → lost ~21% corners) |
| C9 | `processing_graph.py` | `update_params()`, `update_enabled()`, `_cache_params_hash` safety net |
| H13 | `channel_combine_dialog.py` | Weighted average for HOO palette (was naive `*0.5`) |
| H14 | `color_settings_dialog.py` | `QLabel("2.2")` → `QDoubleSpinBox(1.0–3.5)`, persisted via `get_config()` |
| H25-H27 | `main_window.py` | `_open_project` try/except `ValueError`, `_save_project` log at `info`, `_save_as` UI refresh |

Tests: `tests/test_ui/test_phase1_fixes.py` (10 regression tests)

---

## Phase 2 — Smart Processor Correctness 🔵 IN PROGRESS

| Bug | File | Status | Fix |
|-----|------|--------|-----|
| **C6** | `smart_processor.py:347` | ✅ DONE | `object_snr` uses `bg + 3σ` object mask (was P95 → background-biased). Object SNR = `(obj_median - bg) / noise` for M42-like objects |
| **C10** | `smart_processor.py` | ✅ DONE | HDR core protection → 3 selectable operators: Reinhard, Drago, Core-blend. Created `cosmica/core/hdr_operators.py` + QComboBox in dialog |
| **H3** | `smart_processor.py:1533` | ✅ DONE | Already correct (`max(min_midtone, min(0.5, midtone))` = proper floor) |
| **H8** | `plate_solve.py` ↔ `star_catalog.py` | ✅ DONE | Created `cosmica/core/wcs.py:normalise_wcs_dict()` — auto-normalizes between `ra_center/dec_center` ↔ `ra/dec`. Updated `main_window.py` + `color_calibration.py` callers |
| **M1** | `background.py` | ✅ DONE | Already uses GPU via `get_device_manager()` in all 3 functions with CPU fallback |
| **M5** | `tilt_analysis.py` | 🔵 PENDING | Check ellipticity threshold 0.30 consistency with `smart_processor.py` |
| **M15** | `scripting.py` | ✅ DONE | Added `mask_name: str | None` to `PipelineStep`, updated `record_step()` + `play_macro()` + `apply_pipeline_to_image()` with `masks` dict support |

---

## Phase 3 — Data Integrity (PENDING)

| Bug | File | Fix |
|-----|------|-----|
| **C5** | `color_management.py` | Bundle real ICC profiles in `cosmica/resources/icc/{sRGB,AdobeRGB,DisplayP3}.icc` |
| **C8** | `plate_solve.py` | Delete or `raise NotImplementedError` for local `plate_solve(scale_hint=...)` dead code |
| **H1** | `cosmic_clarity.py` | Platform-specific download URLs, `chmod +x`, atomic download with `.partial` |
| **H2** | `equipment.py` | Sexagesimal "05 35 17" via `astropy.coordinates.SkyCoord` |
| **H4** | `analysis/psf.py` | `curve_fit` bounds to prevent divergence |
| **H5** | `analysis/psf.py` | `aperture_radius/2.0` units fix (arcsec → pixels) |
| **H6** | `analysis/psf.py` | CPU radial sampling → vectorize |
| **H7** | `star_removal.py` | Rec.709 wrong for RGGB Bayer — fix assert |
| **H19** | `transforms.py` | `INTER_AREA` → `INTER_CUBIC` for upsize |
| **H20** | `drizzle.py` | NaN/inf guard |
| **H21** | `undo.py` | Ring buffer 50 full copies = 9.6GB for 8MP |
| **H22** | `fwhm_map.py` | Vectorize sub-aperture loop |
| **H23** | `fwhm_map.py` | Real FWHM measurement (was fabricated from ellipticity) |

---

## Phase 4 — GPU Migrations (benchmark-gated) (PENDING)

Create `tests/benchmarks/` with `pytest-benchmark`. Threshold: ≥1.2× faster on 1MP, 8MP, 32MP.

| Module | Expected win |
|--------|-------------|
| `local_normalization` (M6) | ~5× |
| `local_contrast` CLAHE | ~4× |
| `chromatic_aberration` | ~3× |
| `lrgb` Lab↔sRGB | ~3× |
| `cosmetic` median (M3) | ~3× |
| `denoise` NLM (M2) | ~2× |
| `lens_distortion` | ~2× |
| `analysis/psf` radial (H6) | ~2× |
| `aperture_photometry` (M19) | ~2× |

**Keep CPU** (transfer-bound): vignette, mosaic, hdr, mure, banding, tgv, channel_match, subframe, superbias.

---

## Phase 5 — UX/Worker-Thread (14 fixes) (PENDING)

| Bug | File | Fix |
|-----|------|-----|
| C2-fixup | `smart_process_dialog.py` | Cancel button wiring |
| H10 | `live_stack_dialog.py` | Preview worker thread |
| H11/H12 | `curves_widget.py` | Per-channel points cache |
| H24 | All dialogs | Migrate heavy ops to `_start_worker` (HDR/EZ/license/star_mask/channel_match/lens_dist/pixelmath) |
| M7 | `pixelmath_dialog.py` | Per-channel early-return |
| M8 | `tweaks_panel.py` | Log slider range (was 0–100) |
| M9 | `workflow_bar.py` | Transform tab reachable |
| M11 | `histogram.py` | NaN guards for QPainter |
| M12 | `image_canvas.py` | Trackpad horizontal zoom fix |
| M13 | `subframe_dialog.py` | Score button re-enable on error |
| M18 | `batch_preprocess_dialog.py` | Cancel waits for workers |

---

## Phase 6 — Plugin/Extension (PENDING)

M10 local plate_solve + CosmicClarity chmod +x, M14 manual white-ref PCC, M16 HTTP backoff, M17 vectorize superbias, D1–D5 cleanup.

---

## Phase 7 — PixInsight Feature Parity (PENDING)

SPCC Gaia→Teff, drizzle fallback, graph cache trust UI, mask in scripting, batch progress, subframe metrics, star reduction kernel UI, pixelmath highlighter, HOO live preview, SPCC fusion.

---

## Phase 8 — Build & Ship (PENDING)

ICC bundling, Windows <2GB, macOS Intel+ARM, Linux AppImage + `--headless` CI smoke test, CHANGELOG.md.

---

## Key File Locations

| File | Bugs |
|------|------|
| `cosmica/ai/smart_processor.py` | C6, C10, H3 |
| `cosmica/core/hdr_operators.py` | NEW — C10 operators |
| `cosmica/core/wcs.py` | NEW — H8 WCS normalization |
| `cosmica/core/batch.py` | M15 mask support |
| `cosmica/core/scripting.py` | M15 mask recording |
| `cosmica/core/color_calibration.py` | C5, H8, M14 |
| `cosmica/core/background.py` | M1 GPU path |
| `cosmica/core/analysis/tilt_analysis.py` | M5 threshold |
| `cosmica/ui/main_window.py` | C2, H8, H25-H27 |
| `cosmica/ui/dialogs/smart_process_dialog.py` | C2, C10 (HDR QComboBox) |
| `cosmica/ui/dialogs/live_stack_dialog.py` | C1, H10 |
| `cosmica/ui/panels/tools_panel.py` | C3 |
| `cosmica/ui/widgets/python_console.py` | C4 |
| `cosmica/core/dso_catalog.py` | C7 |
| `cosmica/core/processing_graph.py` | C9 |
| `cosmica/core/color_management.py` | C5 |
| `cosmica/core/plate_solve.py` | C8, H8 |
| `cosmica/core/equipment.py` | H2, D1 |
| `cosmica/core/undo.py` | H21 |
| `cosmica/core/transforms.py` | H19 |
| `cosmica/core/drizzle.py` | H20 |
| `cosmica/core/analysis/psf.py` | H4, H5, H6 |
| `cosmica/core/analysis/fwhm_map.py` | H22, H23 |
| `cosmica/ai/inference/cosmic_clarity.py` | H1 |
| `cosmica/ai/inference/star_removal.py` | H7 |

## Test Baseline
- 859 tests pass (was 717 originally; ~57s with `CUDA_VISIBLE_DEVICES=""`)
- CI: `pytest --ignore=tests/test_ui/` with `QT_QPA_PLATFORM=offscreen`
