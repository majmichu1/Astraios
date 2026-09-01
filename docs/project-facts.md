# Astraios project facts

Source of truth for every public statement about Astraios (README, website,
FAQ, structured data, `llms.txt`). When a fact changes, change it here first
and then everywhere it is repeated. Checked against the code on 2026-09-01.

## Definition

Astraios (full descriptive name: Astraios Astrophotography) is a free,
open-source desktop application for calibrating, registering, stacking and
processing deep-sky, planetary and solar astrophotography on Windows, Linux
and macOS. Supported operations use NVIDIA CUDA or Apple Metal acceleration
with automatic CPU fallback. Image processing and AI inference run locally;
explicitly selected online integrations such as astrometry.net transmit only
the data that service needs. Astraios is alpha software.

Intended users: amateur astrophotographers who want a guided, end-to-end
workflow (from raw subs to a finished image) without PixInsight's price or
learning curve, and who have, or do not have, a GPU.

Differentiator in one line: GPU-accelerated where it matters, with automatic
CPU fallback.

Former name: the project was called Cosmica until 2026-06-17. Releases older
than v0.1.x and some internal identifiers (for example the bundled model file
`cosmica_denoise_v1.pt`) keep that name.

Not related to: BigCode's Astraios language models, the EU ASTRAIOS research
project, or any other software called Astraios.

## Version and maturity

| Item | Value | Evidence |
|---|---|---|
| Version on `main` | 0.1.25 | `pyproject.toml`, `astraios/__init__.py` |
| Latest release | v0.1.25-alpha, 2026-08-25 | [Releases](https://github.com/majmichu1/Astraios/releases/latest) |
| Maturity | Alpha. Expect rough edges; report bugs | `Development Status :: 3 - Alpha` classifier |
| Release channel | Every tagged release is a normal (non-prerelease) GitHub release; the updater and installers read `/releases/latest` | `astraios/updater/auto_updater.py`, `packaging/linux/install-astraios.sh` |
| License | GNU GPL v3; `pyproject.toml` declares `GPL-3.0-or-later` | `LICENSE` (canonical text), `pyproject.toml` |
| Test suite | 2301 tests collected on 2026-09-01; CI runs the whole suite on a CPU-only Ubuntu runner | `.github/workflows/ci.yml` |

## Platforms and distribution

| Platform | Package | What it does |
|---|---|---|
| Windows 10/11 x64 | `Astraios-Setup-<version>.exe` (Inno Setup) | Installs a private Python environment with uv, detects an NVIDIA GPU with `nvidia-smi` and installs CUDA 12.8 PyTorch, otherwise CPU PyTorch. Start Menu shortcut. |
| Linux x86_64 | `install-astraios.sh` | Same detection and install into `~/.local`; desktop menu entry. Works on immutable distributions (no root, no system Python). |
| Linux x86_64 | `Astraios-<version>-x86_64.AppImage` | Portable; bundles Python, Qt and the science stack (about 330 MB), provisions the matching PyTorch into a per-user directory on first launch. |
| macOS 11 or later | `install-astraios-macos.sh` | Creates `Astraios.app` (`LSMinimumSystemVersion` 11.0). Apple Silicon uses the Metal (MPS) backend; Intel Macs run on the CPU. |
| Any | `astraios-<version>-py3-none-any.whl`, or `git clone` + Poetry | For developers; Python 3.11 to 3.14. |

PyTorch is downloaded at install time (CUDA build about 2 GB), so the
installer itself stays small and the GPU build matches the machine.

## Compatibility: CI-tested versus user-verified

CI-tested on every change to the packaging (`.github/workflows/validate-installer.yml`):

- Windows (windows-latest runner): installer bootstrap, venv, PyTorch, app import and offscreen GUI launch. CPU PyTorch only; the runner has no NVIDIA GPU.
- Linux (ubuntu-22.04): `install-astraios.sh` end to end, CPU PyTorch.
- macOS Apple Silicon (macos-latest, arm64): installer, `Astraios.app`, Metal (MPS) backend, and a check that every GPU tool produces the same pixels on Metal as on the CPU.
- macOS Intel (macos-15-intel, x86_64): installer and CPU path.

Not CI-tested: the NVIDIA CUDA path. No hosted runner has a GPU. It is
verified by the maintainer on an NVIDIA GeForce RTX 5060 Laptop GPU (7.7 GB
VRAM) under Fedora-based Linux (Bazzite), which is also where the processing
benchmarks and the Siril comparison were run. There are no external user
compatibility reports yet; the compatibility-report issue form collects
them.

## GPU, Metal and CPU behaviour

- Device selection is automatic: CUDA, then Apple Metal (MPS), then CPU
  (`astraios/core/device_manager.py`). `ASTRAIOS_FORCE_CPU=1` forces the CPU.
- AMD and Intel GPUs are not accelerated; those machines run the CPU path.
- Runs on the GPU when one is available: star detection, matching and
  warping for registration; pixel rejection and integration in stacking;
  drizzle; wavelet and chroma denoise; Richardson-Lucy and spatially varying
  deconvolution; wavelet sharpening, unsharp mask and convolutions;
  background extraction; stretches and curves; CLAHE; AI denoise and AI
  super-resolution inference.
- Runs on the CPU by design: master calibration frame combination
  (astropy/numpy), non-local-means denoise (OpenCV), plate solving (external
  solver or local catalog), catalog lookups, file I/O.
- Large images are processed in tiles so consumer VRAM (4 to 8 GB) is enough;
  on CUDA out-of-memory the operation falls back to the CPU.

## Local processing, network use, uploads, telemetry

Astraios collects no telemetry, analytics or crash reports. There is no
account, licence check or activation. Everything below is on demand and can
be avoided by not using the feature (or, for the update check, by turning it
off in Preferences).

| When | Where | What is sent |
|---|---|---|
| Startup (Preferences: Updates, on by default) | `api.github.com` | A request for the latest release; nothing about you or your images. |
| AI super-resolution, first use | `github.com/xinntao/Real-ESRGAN` releases | Downloads the Real-ESRGAN weights (about 67 MB), cached locally. |
| Gaia catalog download (plate solving, PCC/SPCC) | Backblaze bucket published by Seti Astro | Downloads catalog files once. |
| Object identification, Smart Processor | SIMBAD and VizieR TAP services (CDS Strasbourg) | Coordinates and object names from your plate solve. |
| Smart Processor reference mask (optional, best effort) | CDS hips2fits | Field centre, size and rotation of your solved frame; a survey cutout is downloaded. |
| Minor body catalog update | `raw.githubusercontent.com` (Seti Astro data) | Downloads orbital elements. |
| Plate solving with the astrometry.net backend selected and an API key set | `nova.astrometry.net` | Uploads a 16-bit mono FITS copy of the image being solved. This is the only feature that uploads image data, and only when you choose it. ASTAP and the local Gaia solver solve offline. |
| Update install | `github.com` release assets | Downloads the installer; verified against a `.sha256` sidecar when the release provides one. |

The bundled AI denoise model runs locally. StarNet and Cosmic Clarity models
are optional and only used if you point Astraios at files you installed.

## Formats

- Read: FITS (`.fit`, `.fits`, `.fts`), XISF, TIFF, PNG, JPEG. Bayer mosaics
  are debayered (RGGB, BGGR, GRBG, GBRG, pattern read from the header).
- Write: FITS, XISF, TIFF (8/16-bit), PNG (8/16-bit), JPEG.
- Video: SER for planetary, lunar and solar lucky imaging.
- Not supported: camera raw files (CR2, NEF, ARW, DNG). Convert them to
  FITS or TIFF first.
- FITS orientation: `ROWORDER` is honoured, so Siril files open right side up.

## Main capabilities

Full inventory with details: [docs/features.md](features.md).

- Pre-processing: calibration masters, batch preprocessing, cosmetic
  correction, subframe selection, debayer.
- Registration and stacking: star, triangle, FFT and comet alignment;
  sigma, winsorized, linear-fit, percentile, ESD and min/max rejection;
  drizzle; multi-session stacking; live stacking; SER lucky imaging.
- Gradient, colour and stretch: background extraction (polynomial, ABE,
  DBE), photometric and spectrophotometric colour calibration (PCC, SPCC,
  SFCC), SCNR, GHS, arcsinh, curves, histogram transform, PixelMath.
- Detail and noise: Richardson-Lucy, spatially varying and multi-frame
  deconvolution, wavelets, TGV, NLM and chroma denoise, AI denoise, AI
  super-resolution, star reduction, built-in star removal (StarNet optional).
- Narrowband: HOO/SHO palettes, normalization, continuum subtraction, star
  colour recovery.
- Science: aperture photometry, exoplanet transit light curves, transient
  detection, isophote fitting, SNR, plate solving, annotation.
- Workflow: Guided Processing wizard, Smart Processor, non-destructive
  history, macros, batch processing, Python console, plugins.

## AI features and what ships

| Tool | Model | Ships with the app | Fallback |
|---|---|---|---|
| AI Denoise | Noise2Self U-Net (`cosmica_denoise_v1.pt`) | Yes, bundled | Classical denoise (TGV, wavelet, NLM) |
| AI Super-Resolution | Real-ESRGAN x2/x4 | Downloaded on first use | Lanczos resampling |
| AI Sharpen | No trained model is published yet | No | Classical deconvolution |
| Star removal | Morphological remover | Yes | StarNet if you install it |
| Cosmic Clarity, StarNet | Third-party | No; optional, user-installed | Built-in tools |

AI inference runs on the local GPU or CPU; nothing is sent to a server.

## Known limitations (alpha)

- Alpha quality: workflows are complete but bugs are expected.
- No camera raw support (see Formats).
- GPU acceleration is NVIDIA CUDA and Apple Metal only.
- The CUDA path is verified on one machine, not in CI.
- AI Sharpen has no published model yet.
- Installers download about 2 GB of PyTorch on first install.
- Windows builds are not code-signed; SmartScreen may warn.
- The Windows installer, Linux script and AppImage are x86_64 only.

## Links

- Repository: https://github.com/majmichu1/Astraios
- Website: https://majmichu1.github.io/Astraios/
- Latest release: https://github.com/majmichu1/Astraios/releases/latest
- All releases: https://github.com/majmichu1/Astraios/releases
- Changelog: https://github.com/majmichu1/Astraios/blob/main/CHANGELOG.md
- Issues: https://github.com/majmichu1/Astraios/issues
- Discussions: https://github.com/majmichu1/Astraios/discussions
- Security policy: https://github.com/majmichu1/Astraios/blob/main/SECURITY.md

## Evidence in the repository

- Device selection and CPU fallback: `astraios/core/device_manager.py`
- GPU stacking and registration: `astraios/core/stacking.py`, `astraios/core/gpu_registration.py`
- Network calls: `astraios/updater/auto_updater.py`, `astraios/ai/inference/super_resolution.py`, `astraios/core/gaia_catalog.py`, `astraios/core/simbad_lookup.py`, `astraios/core/star_catalog.py`, `astraios/ai/reference_image.py`, `astraios/core/minor_body_catalog.py`, `astraios/core/plate_solve.py`
- Formats: `astraios/core/image_io.py`
- Models: `astraios/ai/model_manager.py`, `astraios/ai/models/`
- Installers and CI: `packaging/`, `.github/workflows/build.yml`, `.github/workflows/validate-installer.yml`, `.github/workflows/ci.yml`
- Benchmark and comparison method: `docs/BENCHMARKS.md`
