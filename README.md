# Cosmica - Open Source AI Image Processing Suite for Astrophotography

**Professional astrophotography image processing. GPU-accelerated, free, and open source alternative to PixInsight.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/majmichu1/cosmica/actions/workflows/ci.yml/badge.svg)](https://github.com/majmichu1/cosmica/actions)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-donate-yellow)](https://buymeacoffee.com/majmichu)

<img width="2559" height="1600" alt="obraz" src="https://github.com/user-attachments/assets/a688865b-476d-4139-9cfc-4efbe89435a7" />

## Download

You don't need to install Python to use Cosmica. Download the standalone, ready-to-use versions (the AppImage is a portable CPU-only version; for full NVIDIA GPU acceleration, install via Poetry):

- **Windows:** [Download .exe (v0.1.9)](https://github.com/majmichu1/Cosmica/releases/latest)
- **Linux:** [Download .AppImage (v0.1.9)](https://github.com/majmichu1/Cosmica/releases/latest)

Bug reports and feedback are welcome in the Issues tab.

## Why Cosmica?

Cosmica is built as a modern workflow tool from calibration to export, featuring out-of-the-box AI integration.

| Feature | Cosmica | Siril | PixInsight |
|---|---|---|---|
| Price | Free (GPL v3) | Free | EUR 230+ |
| GPU Acceleration | Full PyTorch | Partial | Partial |
| AI Denoise / Sharpen | Built-in | No | Extra cost |
| Multi-Session Stacking | Yes | No | Yes |
| Spatially-Varying Deconvolution | Yes | No | Yes |
| Plate Solve + PCC | ASTAP / astrometry.net | Basic | Yes |
| Scripting | Python console | Python | JavaScript |
| Star Removal | StarNet v2 + morphological | No | Yes |
| Processing Graph (DAG) | Yes | No | Yes |
| Batch Preprocessing | Yes (full pipeline) | Yes | Yes |
| Live Stacking | Yes | Experimental | No |
| EZ Processing Scripts | Yes | Yes (preprocessing) | No |
| Plugin System | Yes | No | Yes |
| Non-Destructive Workflow | Yes (undo/redo graph) | No | No |

## Features

### Acquisition and Pre-Processing
- **Calibration** -- Master dark, flat, and bias creation with batch light frame calibration
- **Batch Preprocessing** -- Full PixInsight-style pipeline: folder scan, auto-matching by EXPTIME/FILTER/BINNING/CCD-TEMP, master calibration frame creation, calibration, cosmetic correction, registration, and stacking
- **Alignment** -- Star-based registration (1-pass, 2-pass refinement, triangle matching), FFT phase-correlation, and comet nucleus tracking
- **Stacking** -- Sigma-clip, winsorized sigma, linear fit, percentile clip, ESD, and min/max rejection; GPU-accelerated normalization
- **Drizzle Integration** -- 2x / 3x scale-up for undersampled data, GPU-accelerated
- **Multi-Session Stacking** -- Combine data from multiple nights with per-session adaptive weighting
- **Subframe Selector** -- Automatic frame scoring by FWHM, eccentricity, SNR, background, and star count
- **Debayer** -- RGGB, BGGR, GRBG, GBRG with VNG and other methods; auto-detection from FITS headers

### AI-Powered Tools
Local AI models download automatically on first use. No cloud required.

- **AI Denoise** -- Noise2Self U-Net trained on real astro images
- **AI Sharpen** -- Neural deconvolution for recovering fine detail
- **Star Removal** -- StarNet v2 network and morphological algorithm (works immediately, no model needed)
- **Super-Resolution** -- Upscale images with AI detail enhancement
- **CosmicClarity** -- Suite of pre-trained models for denoise, sharpen, satellite removal, and dark star enhancement
- **MureDenoise** -- Noise estimation using MURE (MUlti-Resolution Estimator) for optimal dark subtraction

### Detail Enhancement
- **Deconvolution** -- Richardson-Lucy with optional total-variation regularization and deringing protection
- **Spatially-Varying Deconvolution** -- Per-zone PSF measurement and blending for field curvature / coma correction
- **TGV Denoise** -- Total generalized variation (edge-preserving, non-AI) with Neumann boundary conditions
- **Wavelet Processing** -- Multi-scale decomposition, sharpening, and noise reduction via a trous algorithm
- **Local Contrast Enhancement** -- GPU-accelerated CLAHE
- **Unsharp Mask** -- Standard and advanced masking
- **Median Filter** -- Impulse noise removal

### Color and Calibration
- **Photometric Color Calibration (PCC)** -- Plate solve then match against Gaia DR3
- **SPCC** -- Spectrophotometric calibration with filter response curves
- **Background Extraction** -- Polynomial surface fitting, ABE (RBF-based), and DBE with per-pixel rejection
- **Background Neutralization** -- Robust color balancing from background samples
- **Color Calibration** -- Statistical and catalog-based correction
- **SCNR** -- Green noise reduction for narrowband and OSC images
- **Color Adjustment** -- Saturation, hue shift, vibrance
- **Curves** -- Per-channel curve editor with histogram overlay
- **Histogram Transform** -- Black point, midtone, white point with live preview
- **Generalized Hyperbolic Stretch (GHS)** -- Non-linear stretch preserving color
- **Arcsinh Stretch** -- Brightness-preserving stretch for deep-sky data
- **PixelMath** -- Full expression evaluator with syntax highlighting, expression history, per-channel apply, function reference, and create-new-image option

### Narrowband and Composition
- **Narrowband Combine** -- HOO, SHO, and custom palette mappings
- **LRGB Combine** -- Luminance-weighted RGB merging
- **Channel Combine** -- Custom channel mapping dialog
- **Continuum Subtraction** -- Remove broadband contamination from narrowband filters
- **HDR Composition** -- Multi-exposure blending using Mertens fusion

### Corrections and Utilities
- **Cosmetic Correction** -- Hot, cold, and dead pixel repair
- **Banding Reduction** -- Horizontal and vertical pattern removal
- **Chromatic Aberration Correction** -- Auto-detect and manual shift
- **Lens Distortion Correction** -- Radial distortion model with auto-estimation from star positions
- **Vignette Correction** -- Model-based flat-field emulation
- **Local Normalization** -- Background-matched frame normalization for mosaic and stacking
- **Star Reduction** -- Shrink star bloat without star removal
- **Morphology** -- Dilate, erode, open, close for star masks
- **PSF Measurement** -- Interactive FWHM, ellipticity, and angle from detected stars
- **Statistics** -- Per-channel min, max, mean, median, std, MAD, SNR, linearity detection, clipping detection
- **Noise Generation** -- Synthetic noise injection for testing and augmentation

### Plate Solving and Annotation
- **Plate Solve** -- ASTAP and astrometry.net with auto-fallback and API key management
- **WCS Overlay** -- Catalog star positions drawn on the image
- **DSO Annotation** -- Automatic deep-sky object labels from solved coordinates
- **Constellation Overlay** -- Constellation lines rendered from WCS solution

### Workflow and UI
- **Modern Dark Theme** -- Clean dark interface, designed for long nights
- **Processing Graph** -- Non-destructive DAG (directed acyclic graph) that records every operation; auto-refreshing history dialog
- **EZ Script Suite** -- One-click processing presets (OSC Quick Processing, Narrowband, Deep Sky Minimal, Luminance, Full Processing with ABE, Starless Processing)
- **4-Panel Layout** -- Project tree / Canvas + Histogram / Tools / Log
- **Split Before/After Preview** -- Draggable divider with live preview on every tool
- **Live Stacking** -- Real-time frame accumulation with adaptive normalization
- **Blink Comparator** -- A/B frame comparison at variable FPS
- **Interactive Histogram** -- Log scale, per-channel stats, clip indicators
- **Curve Editor** -- Per-channel control points with histogram backdrop
- **Macro Recorder** -- Record and playback processing steps
- **Python Console** -- Embedded scripting dock with live image access
- **Batch Processing** -- Unattended folder processing
- **Smart Processor** -- One-click automated workflow with quality checks
- **Equipment Profiles** -- Camera, telescope, and filter metadata for plate-scale calculations
- **ICC Color Management** -- Display profile-aware color rendering
- **Undo / Redo** -- Full history stack with display-reference matching
- **Presets** -- Save and recall tool settings
- **Plugin System** -- Extend Cosmica with Python plugins loaded from cosmica/plugins/

### File Support
- **Read:** FITS, XISF, TIFF, PNG, JPEG (auto-debayer for OSC)
- **Write:** FITS, XISF, TIFF (8/16-bit), PNG (8/16-bit), JPEG

## Getting Started (For Developers)

### Prerequisites
- Python 3.11-3.14
- Poetry (dependency management)

### Installation

```bash
# Clone the repository
git clone https://github.com/majmichu1/cosmica.git
cd cosmica

# Install with dev dependencies
poetry install --with dev

# Run the application
poetry run cosmica
```

### Building Standalone Binary

```bash
poetry install --with build
poetry run pyinstaller build/cosmica.spec
```

## Development

### Run Tests

```bash
poetry run pytest          # 859+ tests
poetry run pytest --ignore=tests/test_ui/   # core + AI tests only (no display needed)
```

### Run Linter

```bash
poetry run ruff check .
```

### Run Type Checker

```bash
poetry run mypy cosmica
```

## AI Model Training

Cosmica includes self-supervised AI training scripts. You can train your own denoise model on your astro images:

```bash
# Place your FITS files in astro_data/
mkdir -p astro_data
cp /path/to/your/*.fits astro_data/

# Train the denoise model
poetry run python scripts/train_denoise_model.py --input astro_data --epochs 30
```

The model uses Noise2Self -- a self-supervised approach that learns to denoise from noisy images alone, without needing clean reference images.

## Architecture

```
cosmica/
├── __main__.py          # Entry point
├── core/                # 66 image processing modules (GPU-accelerated)
├── ai/                  # AI inference, training, model management
├── ui/                  # PyQt6 interface (dialogs, panels, widgets)
├── updater/             # Auto-update via GitHub Releases
├── plugins/             # Plugin system (built-in + user)
├── resources/           # Static SVG icons
└── scripts/             # Training and utility scripts
```

Tests mirror the source layout under `tests/test_core/`, `tests/test_ai/` and `tests/test_ui/`.

## License

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0).

The GPL v3 is required because Cosmica uses PyQt6, which is licensed under GPL v3 for open-source use.

## Acknowledgments

- PyQt6 -- User interface framework
- PyTorch -- GPU-accelerated computation
- Astropy -- FITS file I/O and astronomical calculations
- Noise2Self -- Self-supervised denoising foundation
- StarNet -- Star removal network architecture
- All the open-source astronomical software that inspired this project
