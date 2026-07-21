# Astraios: Free, GPU-Accelerated Astrophotography Image Processing

**A modern, open-source alternative to PixInsight and Siril. Everything runs on your GPU, and it's free.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Installer validated](https://github.com/majmichu1/Astraios/actions/workflows/validate-installer.yml/badge.svg)](https://github.com/majmichu1/Astraios/actions/workflows/validate-installer.yml)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-donate-yellow)](https://buymeacoffee.com/majmichu)

<img width="2559" height="1600" alt="Astraios screenshot" src="https://github.com/user-attachments/assets/a688865b-476d-4139-9cfc-4efbe89435a7" />

## Download & Install

Astraios installs through a smart installer: no Python setup, no manual CUDA. It detects your GPU and installs the matching PyTorch automatically (NVIDIA cards get CUDA acceleration, everything else runs on CPU). The download itself is tiny; the heavy GPU libraries are fetched for your hardware on first install.

**[Download the latest release](https://github.com/majmichu1/Astraios/releases/latest)**

| Platform | What to do |
|----------|-----------|
| Windows | Download `Astraios-Setup-*.exe` and run it. |
| Linux | Download `install-astraios.sh`, then run `bash install-astraios.sh`. Works on Fedora/Bazzite, Ubuntu, Arch, etc. No `apt` or system Python required. |
| Linux (portable) | Download `Astraios-*-x86_64.AppImage`, then `chmod +x Astraios-*.AppImage && ./Astraios-*.AppImage`. Nothing is installed and no root is needed, which suits immutable systems like Bazzite and Silverblue. It still runs on the GPU: the first launch fetches the PyTorch matching your hardware into your home directory. Any normal desktop already has what it needs; a stripped-down or headless system may have to add the usual Qt graphics libraries (`libEGL`, `libGL`, `libxkbcommon-x11`), which are intentionally left to the host because they have to match your graphics driver. |

On first run the installer downloads PyTorch (a few minutes; it's the big piece), then Astraios launches from your Start Menu / application menu. Both installers are tested end-to-end on real Windows and Linux machines in CI on every change.

Astraios is in active development (v0.1.x-alpha), so expect rough edges and bugs. Reports in the [Issues](https://github.com/majmichu1/Astraios/issues) tab are very welcome.

Want the newest features before they reach an installer? The `main` branch is updated continuously and runs ahead of the packaged releases. You can run it from source (see [Run from Source](#run-from-source-developers)) to try new work as it lands, with the understanding that it is the development line: it is less stable than a release and some things may be unfinished or temporarily broken.

If you have an NVIDIA card, keep your GPU driver reasonably up to date, since the installer pulls CUDA 12.8 PyTorch (which covers older GPUs through the RTX 50-series).

## Why Astraios?

Astraios is built as a modern, end-to-end workflow tool, from calibration to export, with GPU acceleration baked in rather than bolted on. The table below reflects Siril 1.4 (Dec 2025) and current PixInsight including the paid third-party plugins most people add to it.

| Feature | Astraios | Siril | PixInsight |
|---|---|---|---|
| Price | Free (GPL v3) | Free (GPL v3) | Paid (~EUR 290) |
| GPU acceleration | Full (PyTorch) | Minimal | Some processes |
| One-click GPU installer | Yes | n/a | n/a |
| AI denoise | Built in and bundled (Noise2Self) | No | Paid plugin (NoiseXTerminator) |
| Plate solve + photometric color cal. | Yes (astrometry.net / ASTAP) | Yes (PCC + SPCC) | Yes (PCC + SPCC) |
| Multi-session stacking | Yes | Yes | Yes |
| Spatially-varying deconvolution | Yes | No | Paid plugin (BlurXTerminator) |
| Star removal | Morphological built-in; StarNet optional | StarNet (external binary) | StarXTerminator (paid) / StarNet |
| Scripting | Python console | Python | JavaScript (PJSR) |
| Non-destructive history graph | Yes | No | Process containers / history |
| Live stacking | Yes | Yes | No |

Where Astraios actually stands apart: everything runs on the GPU, the GPU install is one click, and spatially-varying deconvolution and AI denoise are built in and free rather than paid plugins.

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
- **SER Planetary Stacker** -- Lucky-imaging stacking of SER planetary, lunar, and solar videos: per-frame sharpness ranking, keep-best selection, alignment, and average / median / sigma-clip integration, streaming from disk
- **Multi-Frame Deconvolution** -- Jointly deconvolve all registered frames with per-frame PSFs for a sharper result than single-frame deconvolution on the stack
- **Dither Analysis** -- Measure dither spread, coverage, nearest-neighbour spacing, and walking-noise across a registered frame set
- **Pedestal** -- Add or remove a constant offset (per channel or global) around operations that dislike negative pixels

### Planetary and Solar
- **SER Viewer** -- Scrub and play SER planetary videos frame by frame with per-frame stats, and send any frame to the canvas
- **Planetary De-rotation** -- Undo a planet's rotation across a capture by reprojecting each frame through a body-frame longitude/latitude grid before stacking
- **Planet Projection** -- Reproject a planetary disc to an equirectangular longitude/latitude map or a re-oriented orthographic view, with automatic disc detection

### Science and Analysis
- **Exoplanet Transit Detector** -- Aperture photometry across a registered frame set with comparison stars and detrending, producing a light curve and transit verdict
- **Transient Hunter** -- Difference a new frame against a reference of the same field and classify candidates as new (supernova), vanished, or moved (asteroid, with motion vector)
- **Measure Magnitudes** -- Aperture photometry table with instrumental and calibrated magnitudes, zero point, and limiting magnitude, exportable as CSV
- **SNR Measurement** -- Signal-to-noise readout, overall and per channel, from selected background/signal regions or robust global statistics
- **Alt/Az Field Rotation** -- Field-rotation rate and total rotation over an exposure for alt-az mounts, plus parallactic angle

### AI and Advanced Denoising
Nothing runs in the cloud: models ship with Astraios and run on your own machine.

- **AI Denoise** -- a Noise2Self U-Net trained on real astro images, bundled with the app so it works with no download. It runs J-invariant inference with signal protection, so noise drops sharply while stars keep their brightness (measured on a synthetic field: 68% less background noise, 99% of star flux retained).
- **Classical denoising** -- TGV, wavelet, non-local means and chroma denoise are also built in, and are what the AI backend falls back to if a model is ever unavailable.
- **Star Removal** -- a built-in morphological remover that works immediately with no extra download, plus optional StarNet integration when you point Astraios at a StarNet binary you've installed
- **Bring your own models** -- in Preferences you can point Astraios at a StarNet binary, a denoise model, or a Cosmic Clarity model folder you already have

For sharpening, see the deconvolution tools below (Richardson-Lucy and spatially-varying deconvolution).

### Smart Processor (object-aware automation)
One click, and Astraios identifies the target (plate solve plus catalog / SIMBAD), figures out what it is and where it is in the frame, and applies a tailored pipeline: stretch, background, deconvolution, contrast, and colour all steered by the subject rather than blind whole-image heuristics. Per-object-type recipes, GPU throughout, with quality checks at every stage.

### Detail Enhancement
- **Deconvolution** -- Richardson-Lucy with optional total-variation regularization and deringing protection
- **Spatially-Varying Deconvolution** -- Per-zone PSF measurement and blending for field curvature / coma correction
- **TGV Denoise** -- Total generalized variation (edge-preserving, non-AI) with Neumann boundary conditions
- **Wavelet Processing** -- Multi-scale decomposition, sharpening, and noise reduction via a trous algorithm
- **Local Contrast Enhancement** -- GPU-accelerated CLAHE
- **Unsharp Mask** -- Standard and advanced masking
- **Median Filter** -- Impulse noise removal
- **WaveScale HDR** -- Wavelet-based recovery of detail inside bright cores of stretched images
- **WaveScale Dark Enhance** -- Multiscale deepening of faint dark structure (dust lanes, dark nebulae)
- **Texture and Clarity** -- Midtone-protected fine-detail and local-contrast punch, with smoothing at negative settings

### Color and Calibration
- **Photometric Color Calibration (PCC)** -- Plate solve then match against Gaia DR3
- **SPCC** -- Spectrophotometric calibration with filter response curves
- **SFCC** -- Spectral flux color calibration: physically integrates filter transmission x sensor QE x stellar flux to derive per-channel scales; ships with representative curves and imports your own vendor CSV data
- **Background Extraction** -- Polynomial surface fitting, ABE (RBF-based), and DBE with per-pixel rejection
- **Background Neutralization** -- Robust color balancing from background samples
- **Color Calibration** -- Statistical and catalog-based correction
- **SCNR** -- Green noise reduction for narrowband and OSC images
- **Color Adjustment** -- Saturation, hue shift, vibrance
- **Saturation by Hue** -- Per-colour-family saturation curves in HSV or perceptual Lab chroma
- **Selective Color** -- Adjust one colour family only (CMY/RGB shifts, luminance, chroma, contrast) with feathered selection
- **Selective Luminance** -- The same adjustments confined to a shadows / midtones / highlights band
- **Linear Fit** -- Match one image's levels onto a reference by a robust sigma-clipped slope and offset fit
- **Curves** -- Per-channel curve editor with histogram overlay
- **Histogram Transform** -- Black point, midtone, white point with live preview
- **Generalized Hyperbolic Stretch (GHS)** -- Non-linear stretch preserving color
- **Arcsinh Stretch** -- Brightness-preserving stretch for deep-sky data
- **PixelMath** -- Full expression evaluator with syntax highlighting, expression history, per-channel apply, function reference, and create-new-image option

### Narrowband and Composition
- **Narrowband Combine** -- HOO, SHO, and custom palette mappings
- **Perfect Palette Picker** -- Blend Ha/OIII/SII into 12 named false-color palettes (all SHO-family permutations plus Realistic 1/2 and Foraxx) or a free custom weight matrix, GPU-accelerated
- **Add Stars** -- Screen or additive recombine of an extracted star layer onto a starless image, with blend amount and mask support
- **LRGB Combine** -- Luminance-weighted RGB merging
- **Channel Combine** -- Custom channel mapping dialog
- **Continuum Subtraction** -- Remove broadband contamination from narrowband filters
- **Narrowband Normalization** -- Balance Ha / OIII / SII channels before palette combination (SHO / HSO / HOS / HOO)
- **NB Star Color** -- Replace the unnatural star colours of narrowband palettes with real RGB star colour
- **Luminance Recombine** -- LRGB finish by linear luminance scaling that preserves hue and chroma
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
- **Halo Reduction** -- Suppress the bright halos and glow rings around stars that stretching amplifies
- **Blemish Blaster and Clone Stamp** -- Click-to-heal dust motes and blemishes on the canvas; copy pixels between regions
- **Isophote Analysis** -- Fit elliptical isophotes to a galaxy (ellipticity, position angle, intensity profile) with model and residual and CSV export
- **Image Combine** -- Two-image pixel arithmetic (add, subtract, average, multiply, divide, screen, overlay, difference, min, max) with per-input weights
- **Copy Astrometry** -- Transfer a plate-solve (WCS) solution from one image's header to another

### Plate Solving and Annotation
- **Plate Solve** -- Offline local Gaia DR3 solving, plus ASTAP and astrometry.net, with auto-fallback and API key management
- **Gaia Catalog Manager** -- Download the local Gaia catalog bands (or reuse an existing Seti Astro Suite Pro catalog folder) for solving without internet
- **WCS Overlay** -- Catalog star positions drawn on the image
- **DSO Annotation** -- Automatic deep-sky object labels from solved coordinates
- **Constellation Overlay** -- Constellation lines rendered from WCS solution
- **Finder Chart** -- Annotated chart over a plate-solved image: compass, scale bar, field marker, grid, catalog markers, and an imaging-train field-of-view box for mosaic planning
- **What's In My Image** -- Identify every catalog object in a plate-solved field as a clickable table with pixel positions, and render the labels onto the image
- **What's In My Sky** -- Tonight's observing planner: transit time, maximum altitude, and hours visible for every catalog object from your location and date, with sun/moon rise-set and moon phase, fully offline
- **Minor Body Catalog** -- Asteroid and comet positions computed locally from downloadable orbital elements (Kepler propagation, no live network queries)

### Effects and Finishing
- **Layers** -- Photoshop-style layer stack with 18 blend modes, per-layer opacity, visibility, and masks, with a live composite preview and flatten
- **FX Tool** -- Orton glow, soft focus, bloom, vignette, film grain, and split toning
- **Diffraction Spikes** -- Synthetic spikes on the brightest stars (Newtonian, JWST-style, secondary sets, halos)
- **Nebula Flythrough** -- Render a cinematic zoom-into-the-image MP4, with star-parallax depth
- **Signature / Watermark** -- Text or logo signing with position, scale, rotation, and opacity

### Workflow and UI
- **Modern Dark Theme** -- Clean dark interface, designed for long nights
- **Processing History** -- Non-destructive, replayable history that records every operation; view, toggle, reorder, re-edit, and export as a macro
- **Hover Help** -- Every tool and setting carries a plain-language explanation on hover; sliders additionally teach what turning them up or down actually does, so full power stays approachable without trial and error
- **EZ Script Suite** -- One-click processing presets (OSC Quick Processing, Narrowband, Deep Sky Minimal, Luminance, Full Processing with ABE, Starless Processing)
- **4-Panel Layout** -- Project tree / Canvas + Histogram / Tools / Log
- **Split Before/After Preview** -- Draggable divider with live preview on every tool
- **Live Stacking** -- Real-time frame accumulation with adaptive normalization
- **Blink Comparator** -- A/B frame comparison at variable FPS
- **Macro Recorder** -- Record and playback processing steps
- **Python Console** -- Embedded scripting dock with live image access
- **Batch Processing** -- Unattended folder processing
- **Image Peeker** -- Cull a night's subs fast: auto-stretched thumbnails with per-frame stats (median, FWHM, eccentricity, star count) in a sortable grid
- **Batch Convert** -- Convert whole file sets between FITS, TIFF, PNG, JPEG, and XISF with bit-depth options
- **Batch Rename** -- Template renaming from FITS header tokens ({OBJECT}, {FILTER}, counters, filters) with a live dry-run preview
- **AstroBin Exporter** -- Group your light frames by night, filter, and exposure into the CSV AstroBin's acquisition importer expects
- **Export Curves (.acv)** -- Save the curve editor's points as a Photoshop-compatible .acv file
- **Equipment Profiles** -- Camera, telescope, and filter metadata for plate-scale calculations
- **ICC Color Management** -- Display profile-aware color rendering
- **Undo / Redo** -- Full history stack with display-reference matching
- **Plugin System** -- Extend Astraios with Python plugins loaded from `astraios/plugins/`

### File Support
- **Read:** FITS, XISF, TIFF, PNG, JPEG (auto-debayer for OSC)
- **Write:** FITS, XISF, TIFF (8/16-bit), PNG (8/16-bit), JPEG

## Run from Source (Developers)

### Prerequisites
- Python 3.11-3.14
- [Poetry](https://python-poetry.org/) (dependency management)

```bash
git clone https://github.com/majmichu1/Astraios.git
cd Astraios

poetry install --with dev      # install with dev dependencies
poetry run astraios            # run the application
```

`main` is the development line and is updated far more often than the tagged releases, so a `git pull` gives you the newest features and fixes long before they are packaged into an installer. That is the trade-off: `main` carries a `-dev` version, it is less stable than a release, and a feature may be half-finished or temporarily broken while it is being worked on. For a dependable setup, use the installer at the top of this page; if you run `main` and something breaks, a note in the Issues tab helps a lot.

On Linux/Windows with an NVIDIA GPU and a CUDA build of PyTorch installed, Astraios uses the GPU automatically (verify with the device shown in the log).

### Tests, Linting, Type Checks

```bash
poetry run pytest                              # 1000+ tests
poetry run pytest --ignore=tests/test_ui/      # core + AI only (no display needed)
poetry run ruff check astraios                 # lint
poetry run mypy astraios                        # type check
```

### How the installers are built
The release installers are produced by GitHub Actions (`.github/workflows/build.yml`):
the Windows installer (Inno Setup, `packaging/windows/`) and the Linux installer
script (`packaging/linux/install-astraios.sh`) both bundle [uv](https://github.com/astral-sh/uv)
and the app wheel, then install the GPU/CPU PyTorch at install time. They are
validated end-to-end on real runners by `validate-installer.yml`.

## AI Model Training

Astraios includes self-supervised AI training scripts. Train your own denoise model on your own data:

```bash
mkdir -p astro_data
cp /path/to/your/*.fits astro_data/
poetry run python scripts/train_denoise_model.py --input astro_data --epochs 30
```

The model uses **Noise2Self**: a self-supervised approach that learns to denoise from noisy images alone, with no clean reference images required.

## Architecture

```
astraios/
├── __main__.py          # Entry point
├── core/                # 60+ image processing modules (GPU-accelerated)
├── ai/                  # AI inference, training, model management
├── ui/                  # PyQt6 interface (dialogs, panels, widgets)
├── updater/             # Auto-update via GitHub Releases
├── plugins/             # Plugin system (built-in + user)
└── resources/           # Static assets (icons, catalog, recipes)
```

Tests mirror the source layout under `tests/test_core/`, `tests/test_ai/`, and `tests/test_ui/`.

## License

Licensed under the **GNU General Public License v3.0** (GPL-3.0), required because Astraios uses PyQt6 (GPL v3 for open-source use). All contributions must be GPL-3.0 compatible.

## Acknowledgments

- **PyQt6**: user interface framework
- **PyTorch**: GPU-accelerated computation
- **Astropy**: FITS I/O and astronomical calculations
- **Noise2Self**: self-supervised denoising foundation
- **StarNet**: star removal network architecture
- **[Seti Astro Suite Pro](https://github.com/setiastro/setiastrosuitepro)** (Franklin Marek, GPL-3.0): several tools are ported and adapted from its source under the shared GPL license, with attribution in each module and in CREDITS.md
- **uv**: the fast Python installer that powers the one-click installers
- All the open-source astronomical software that inspired this project
