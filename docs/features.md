# Astraios features

The complete inventory of what Astraios does, grouped by task. Facts that
also appear on the website and in the README come from
[project-facts.md](project-facts.md). Parameter-level detail for the tools
panel is in [tools.md](tools.md); the recommended order of operations is in
[workflow.md](workflow.md).

Astraios picks CUDA, Apple Metal or CPU automatically. Tools marked (GPU)
run on the GPU when one is present and on the CPU otherwise; everything
else runs on the CPU by design.

## Acquisition and pre-processing

- **Calibration**: master dark, flat and bias creation, batch light-frame calibration.
- **Batch Preprocessing**: folder scan, automatic matching by EXPTIME, FILTER, BINNING and CCD-TEMP, master creation, calibration, registration and integration in one run.
- **Alignment** (GPU): star-based registration (1-pass, 2-pass refinement, triangle matching), FFT phase correlation, comet nucleus tracking. Every transform is verified before a frame is accepted; unregistrable frames are left out rather than stacked unaligned.
- **Stacking** (GPU): sigma-clip, winsorized sigma (default), linear fit, percentile clip, ESD and min/max rejection; additive, multiplicative, additive-plus-scaling and local normalization; registration padding is treated as missing data so edges do not darken.
- **Drizzle** (GPU): 2x and 3x, square and Gaussian kernels.
- **Multi-Session Stacking**: combine nights with per-session adaptive weighting.
- **Subframe Selector**: scoring by FWHM, eccentricity, SNR, background and star count; sigma, best-N and best-percent selection; scores cached in the project.
- **Debayer**: RGGB, BGGR, GRBG, GBRG with VNG and other methods; pattern read from FITS headers.
- **SER Planetary Stacker**: lucky imaging for SER planetary, lunar and solar video: per-frame sharpness ranking, keep-best selection, alignment and stacking.
- **Multi-Frame Deconvolution** (GPU): joint deconvolution of all registered frames with per-frame PSFs.
- **Dither Analysis**: dither spread, coverage, nearest-neighbour spacing and walking-noise across a registered set.
- **Pedestal**: add or remove a constant offset around operations that dislike negative pixels.

## Planetary and solar

- **SER Viewer**: scrub and play SER videos with per-frame statistics; send any frame to the canvas.
- **Planetary De-rotation**: undo a planet's rotation across a capture through a body-frame longitude/latitude grid before stacking.
- **Planet Projection**: equirectangular map or re-oriented orthographic view of a planetary disc, with automatic disc fitting.

## Science and analysis

- **Exoplanet Transit Detector**: aperture photometry across a registered set with comparison stars and detrending; light curve and transit fit.
- **Transient Hunter**: difference a new frame against a reference and classify candidates as new, vanished or moved.
- **Measure Magnitudes**: aperture photometry table with instrumental and calibrated magnitudes, zero point and limiting magnitude; CSV export.
- **SNR Measurement**: overall and per channel, from regions or robust global statistics.
- **Alt/Az Field Rotation**: rotation rate and total rotation for alt-az mounts, plus parallactic angle.
- **Isophote Analysis**: elliptical isophote fits (ellipticity, position angle, profile) with model, residual and CSV export.
- **Statistics**: per-channel min, max, mean, median, standard deviation, MAD, SNR, linearity and clipping detection.
- **PSF Measurement**: FWHM, ellipticity and angle from detected stars.

## AI and denoising

All models run on your machine, on the GPU when available.

- **AI Denoise** (GPU): a Noise2Self U-Net trained on real astro images, bundled with the app so it works with no download. J-invariant inference with star protection, tiled for large images.
- **AI Super-Resolution** (GPU): 2x or 4x upscaling through Real-ESRGAN, tiled to fit VRAM. The weights (about 67 MB) are downloaded from the upstream GitHub release on first use; if they cannot be fetched the tool falls back to Lanczos and says so.
- **AI Sharpen**: no trained model is published yet; the tool runs Richardson-Lucy deconvolution instead.
- **Classical denoising** (GPU except NLM): TGV, wavelet, non-local means (OpenCV) and chroma denoise; also the fallback when an AI model is unavailable.
- **Star Removal**: a built-in morphological remover, plus optional StarNet integration when you point Astraios at a StarNet binary in Preferences (StarNet runs as a separate process).
- **Bring your own models**: Preferences accepts a StarNet binary, a denoise model file or a Cosmic Clarity model folder.

## Smart Processor

One click: plate solve, identify the target (catalog and SIMBAD), work out what it is and where it sits in the frame, and apply a recipe for that object class (galaxy, emission nebula, globular cluster, and so on), with an optional survey-cutout mask from CDS hips2fits.

## Detail enhancement

- **Deconvolution** (GPU): Richardson-Lucy with total-variation regularization and deringing; Wiener.
- **Spatially-Varying Deconvolution** (GPU): per-zone PSF measurement and blending for field curvature and coma.
- **TGV Denoise** (GPU): total generalized variation, edge preserving.
- **Wavelet Processing** (GPU): a trous multi-scale decomposition, sharpening and noise reduction.
- **Local Contrast** (GPU): CLAHE with amount control.
- **Unsharp Mask** (GPU), **Median Filter** (GPU).
- **WaveScale HDR** and **WaveScale Dark Enhance**: wavelet recovery of bright cores and deepening of dark structure.
- **Texture and Clarity**: midtone-protected fine detail and local contrast.
- **Halo Reduction**, **Star Reduction**, **Frequency Separation**, **Background Grain** control.

## Colour and calibration

- **Photometric Color Calibration (PCC)**: plate solve, then match against Gaia DR3.
- **SPCC**: spectrophotometric calibration with sensor and filter response curves.
- **SFCC**: spectral flux colour calibration integrating filter transmission, sensor QE and stellar flux; ships with common sensor and filter curves.
- **Background Extraction** (GPU): polynomial surface, ABE (RBF) and DBE with per-pixel rejection.
- **Background Neutralization**, **Color Calibration** (statistical and catalog-based), **SCNR** (selectable channel), **Linear Fit** between images.
- **Color Adjustment** (saturation, hue, vibrance), **Saturation by Hue**, **Selective Color**, **Selective Luminance**.
- **Curves** (per channel, histogram overlay), **Histogram Transform**, **Generalized Hyperbolic Stretch**, **Arcsinh Stretch**, **Statistical Stretch**, **Star Stretch** (GPU).
- **PixelMath**: expression evaluator with syntax highlighting, history, per-channel apply, function reference and create-new-image.

## Narrowband and composition

- **Narrowband Combine**: HOO, SHO and custom mappings.
- **Perfect Palette Picker**: 12 named palettes (SHO family, Realistic 1/2, Foraxx) or free mixing.
- **Narrowband Normalization**, **Continuum Subtraction**, **NB Star Color** (real RGB star colour in a narrowband palette).
- **Add Stars**, **LRGB Combine**, **Luminance Recombine**, **Channel Combine**, **HDR Composition** (Mertens fusion).

## Corrections and utilities

- **Cosmetic Correction** (hot, cold, dead pixels), **Banding Reduction**, **Chromatic Aberration Correction**, **Lens Distortion Correction**, **Vignette Correction**, **Local Normalization**, **Morphology**.
- **Blemish Blaster and Clone Stamp**, **Image Combine** (two-image arithmetic with weights), **Copy Astrometry** (WCS transfer between images).
- **Transform**: crop, rotate, flip, resize, bin, invert.

## Plate solving and annotation

- **Plate Solve**: offline local Gaia DR3 solver, ASTAP, and astrometry.net (the only one that uploads an image, and only when selected with an API key); automatic fallback between them.
- **Gaia Catalog Manager**: download the catalog bands, or reuse an existing Seti Astro Suite Pro catalog folder.
- **WCS Overlay**, **DSO Annotation**, **Constellation Overlay**, **Finder Chart** (compass, scale bar, grid, markers, imaging-train field of view).
- **What's In My Image** (every catalog object in a solved field, clickable) and **What's In My Sky** (tonight's planner from your location).
- **Minor Body Catalog**: asteroid and comet positions computed locally from downloadable orbital elements.

## Effects and finishing

- **Layers**: layer stack with 18 blend modes, opacity, visibility and masks, live composite and flatten.
- **FX Tool**: Orton glow, soft focus, bloom, vignette, film grain, split toning.
- **Diffraction Spikes**, **Nebula Flythrough** (MP4 zoom with star parallax), **Signature / Watermark**.

## Workflow and interface

- **Guided Processing** (Ctrl+G): a wizard that walks the correct order (trim, gradient, colour, sharpen, denoise, stretch, saturation), suggests settings measured from your image, previews before committing, and can skip or step back.
- **Hover Help**: every tool and setting explains itself; sliders say what raising or lowering them does.
- **Processing History**: non-destructive, replayable; view, toggle, reorder, re-edit and export as a macro.
- **Smart telescope recognition**: Seestar, Dwarf, Vespera, Stellina and Unistellar frames are identified from their headers and routed to the guided workflow.
- **EZ Script Suite**: one-click presets (OSC quick, narrowband, deep-sky minimal, luminance, full with ABE, starless).
- **Split Before/After Preview** on every tool, **Blink Comparator**, **Image Peeker** (auto-stretched thumbnails with per-frame statistics).
- **Live Stacking**, **Batch Processing**, **Batch Convert** (FITS, TIFF, PNG, JPEG, XISF), **Batch Rename** from header tokens, **AstroBin Exporter**, **Export Curves (.acv)**.
- **Macro Recorder**, **Python Console** with live image access, **Plugin System** (`astraios/plugins/`).
- **Equipment Profiles**, **ICC Color Management** (display-profile aware), **Undo / Redo**, dark theme with bundled fonts.

## File support

- Read: FITS, XISF, TIFF, PNG, JPEG (OSC frames auto-debayered); SER video.
- Write: FITS, XISF, TIFF (8/16-bit), PNG (8/16-bit), JPEG.
- Not supported: camera raw files (CR2, NEF, ARW, DNG).
