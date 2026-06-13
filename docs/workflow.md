# Cosmica — Processing Workflow

A step-by-step guide to processing astrophotography images with Cosmica.

## Overview

The standard workflow for processing deep-sky astrophotography images follows this pipeline:

```
Calibration → Stacking → Stretch → Background → Color → Detail → Export
```

## Phase 1: Calibration

**Purpose**: Remove sensor artifacts and optical imperfections.

### Required Frames
- **Light frames**: Your actual astro images (50-200+ subs)
- **Dark frames**: Same exposure/temperature, cap on telescope (20-50)
- **Flat frames**: Evenly illuminated source, same optical train (20-50)
- **Bias frames**: Shortest exposure, cap on (50-100)

### Steps
1. **File → New Project** — create a project and organize frames
2. Import frames by type (Lights, Darks, Flats, Bias) into the Project Panel
3. **Process → Pre-Process → Calibration**
4. Wait for master frames to be created and lights to be calibrated

### What Happens
- Master Bias = median of all bias frames
- Master Dark = median of darks, bias-subtracted
- Master Flat = median of flats, dark/bias corrected, normalized
- Each light = (light - master_bias - master_dark) / master_flat

## Phase 2: Stacking

**Purpose**: Combine calibrated frames to improve SNR.

### Steps
1. **Subframe Selector** — score and reject bad frames (poor focus, clouds)
2. **Stacking** — sigma-clip rejection + average integration
3. Review the result — check for artifacts, satellite trails, etc.

### Parameters
- **Sigma Clip** (kappa=3.0): Rejects pixels > 3σ from mean
- **Winsorized Sigma**: Softer rejection, good for high-star-density
- **Linear Fit**: Best for images with varying background levels

## Phase 3: Stretch

**Purpose**: Transform linear (dark, flat-looking) data into a visible image.

### Steps
1. **Auto-Stretch** — start with default midtone=0.25
2. Adjust **Midtone** slider: higher = brighter shadows, lower = more contrast
3. For more control: **GHS** or **Histogram Transform**
4. Use **Curves** for fine tonal adjustments

### Tips
- Don't over-stretch — noise becomes visible
- Use linked RGB channels to maintain color balance
- Save your stretch as a **Preset** for similar images

## Phase 4: Background Extraction

**Purpose**: Remove light pollution gradients and vignetting.

### Steps
1. **Background Extraction** — automatic polynomial surface fitting
2. Or **ABE** (Automatic Background Extraction) — RBF-based
3. Check the residual — should be smooth with no structure
4. If stars are being affected, increase box size or use object-aware mode

### Tips
- For narrowband, background is often very uneven — use higher polynomial order
- If you have a flat field, vignette should already be corrected

## Phase 5: Color

**Purpose**: Achieve neutral white balance and remove color casts.

### Steps
1. **Color Calibration** — use G2V star reference or average neutralization
2. **SCNR** — remove green cast (common in OSC cameras)
3. **Color Adjustment** — fine-tune saturation and vibrance

### Tips
- Target a neutral gray background (R=G=B)
- Don't oversaturate — subtle is better
- Save color settings as a preset for reuse

## Phase 6: Detail Enhancement

**Purpose**: Sharpen and enhance fine detail without amplifying noise.

### Recommended Order
1. **Noise Reduction** first — clean up before sharpening
2. **Deconvolution** — use measured PSF FWHM for accurate sharpening
3. **Wavelet Sharpening** — multi-scale enhancement
4. **Local Contrast** — CLAHE for subtle pop

### Tips
- Always denoise before sharpening
- Monitor star sizes — don't let them grow
- Use masks to protect stars during heavy processing

## Phase 7: Export

**Purpose**: Save your processed image for sharing or further work.

### Steps
1. **File → Save Image As...** (or `Ctrl+Shift+S`)
2. Choose format:
   - **FITS**: For further processing, preserves full dynamic range
   - **TIFF 16-bit**: For printing, high-quality archival
   - **PNG 16-bit**: For web with good quality
   - **JPEG**: For sharing on social media (8-bit, lossy)

### Recommended Export Settings
| Use Case | Format | Bit Depth |
|----------|--------|-----------|
| Further processing | FITS | 32-bit float |
| Print / Archive | TIFF | 16-bit |
| Web portfolio | PNG | 16-bit |
| Social media | JPEG | 8-bit, quality 95% |

## Automated Workflow: Smart Processor

The **Smart Processor** (Ctrl+Shift+P) automates the entire pipeline:

1. Detects image type (OSC RGB, mono, narrowband)
2. Measures PSF and identifies the target object
3. Plans the optimal processing sequence
4. Executes background extraction, denoise, deconvolution, stretch
5. Delivers a finished result in one click

Simply load your image and click **Process → Smart Processor**.
