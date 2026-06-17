# Astraios — Tools Reference

Complete reference for all processing tools in the Tools Panel.

## Pre-Process Tab

### Calibration
Subtracts master dark, divides by master flat, subtracts master bias from light frames.
- **Inputs**: Light frames + calibration frames (Darks, Flats, Bias)
- **Key params**: None — fully automatic
- **Output**: Calibrated light frames

### Cosmetic Correction
Removes hot pixels, cold pixels, and dead pixels.
- **Hot Sigma** (default: 5.0): Sensitivity for hot pixel detection
- **Cold Sigma** (default: 5.0): Sensitivity for cold pixel detection
- **Detect Dead**: Enable dead pixel detection
- **Kernel Size** (default: 5): Neighborhood size for detection

### Banding Reduction
Reduces horizontal and vertical banding patterns.
- **Horizontal** / **Vertical**: Direction to correct
- **Amount** (default: 1.0): Strength of correction
- **Protection Sigma** (default: 3.0): Signal protection threshold

### Subframe Selector
Scores and weights frames for stacking quality.
- **FWHM Weight** (0.3): Importance of sharpness
- **Eccentricity Weight** (0.2): Importance of round stars
- **SNR Weight** (0.3): Importance of signal-to-noise
- **Stars Weight** (0.2): Importance of star count
- **Rejection Sigma** (2.0): Outlier rejection threshold

## Transform Tab

| Tool | Description |
|------|-------------|
| **Crop** | Manual crop by pixel coordinates |
| **Rotate** | 90°/180°/270° or arbitrary angle rotation |
| **Flip** | Horizontal, vertical, or both |
| **Resize** | Scale by factor or target dimensions |
| **Bin** | 2x2, 3x3 binning (average or sum) |
| **Invert** | Invert image polarity |

## Stacking Tab

Combine multiple frames into one with outlier rejection.
- **Rejection**: None / Sigma Clip / Winsorized Sigma / Linear Fit
- **Integration**: Average / Median
- **Kappa Low/High** (3.0): Rejection thresholds
- **Max Iterations** (5): Sigma clipping iterations
- **Winsorize Cutoff** (1.5): For winsorized rejection

## Stretch Tab

### Auto-Stretch
Automatic statistical stretch using shadow/midtone/highlight analysis.
- **Shadow Clip** (-2.8 MAD): Black point
- **Highlight Clip** (1.0): White point
- **Midtone** (0.25): Midtone balance
- **Linked**: Apply same stretch to all RGB channels

### Generalized Hyperbolic Stretch (GHS)
Parametric stretch with full control over the transfer function.
- **D** (5.0): Dynamic range parameter
- **b** (0.0): Brightness offset
- **SP** (0.0): Shadow protection
- **Shadow/Highlight Protection**: Protect extreme values

### Histogram Transform
Manual black/white/midtone control.
- **Black Point** (0.0): Input black
- **Midtone** (0.5): Gamma-like control
- **White Point** (1.0): Input white

### Curves
Interactive spline-based tonal curve editor.
- Separate curves for Master, Red, Green, Blue
- Click to add points, drag to move, right-click to delete

## Color Tab

### Color Calibration
White balance correction.
- **White Reference**: Average / G2V Star / Custom RGB
- **Neutralize Background**: Auto-detect and neutralize background color
- **Background Percentile** (10.0): Percentile for background sampling

### SCNR
Subtractive Chromatic Noise Reduction — removes green/magenta cast.
- **Method**: Average Neutral / Maximum Neutral
- **Amount** (1.0): Correction strength
- **Preserve Luminance**: Keep overall brightness

### Color Adjustment
Fine-tune saturation, hue, and vibrance.
- **Saturation** (1.0): Global saturation multiplier
- **Hue Shift** (0.0): Rotate hue
- **Vibrance** (0.0): Smart saturation (boosts muted colors more)

## Detail Tab

### Deconvolution
Richardson-Lucy deconvolution for sharpening.
- **PSF FWHM** (3.0): Measured star size
- **Iterations** (50): Deconvolution passes
- **Regularization** (0.001): Noise suppression
- **Deringing**: Suppress ringing artifacts

### Noise Reduction
NLM (Non-Local Means) or Wavelet-based denoising.
- **Method**: NLM / Wavelet
- **Strength** (0.5): Amount of denoising
- **Detail Preservation** (0.5): Protect edges
- **Chrominance Only**: Denoise color without affecting luminance

### Star Reduction
Reduce star size and brightness for nebula-focused images.
- **Amount** (0.5): Reduction strength
- **Iterations** (2): Passes
- **Protect Core**: Preserve bright star cores
- **Kernel Size** (3): Neighborhood size

### Wavelet Sharpening
Multi-scale sharpening via wavelet decomposition.
- **N Scales** (4): Number of wavelet levels
- **Scale Weights**: Per-level amplification
- **Residual Weight** (1.0): Base layer weight

### Local Contrast (CLAHE)
Contrast Limited Adaptive Histogram Equalization.
- **Clip Limit** (2.0): Contrast limit
- **Tile Size** (8): Grid size
- **Amount** (1.0): Blend factor

### Morphology
Morphological operations (Erode, Dilate, Open, Close).
- **Operation**: Erode / Dilate / Open / Close
- **Element**: Circle / Square / Diamond
- **Kernel Size** (3): Structuring element size
- **Iterations** (1): Number of passes

### Unsharp Mask
Classic sharpening via blurred difference.
- **Radius** (2.0): Blur radius
- **Amount** (0.5): Sharpening strength
- **Threshold** (0.0): Edge detection threshold

### Median Filter
Noise removal via median filtering.
- **Kernel Size** (3): Filter window

## AI Pro Tab (Pro License Required)

| Tool | Description |
|------|-------------|
| **AI Denoise** | Neural network-based noise reduction |
| **AI Sharpen** | Neural network-based deconvolution |
| **StarNet** | Star removal via StarNet v2 |
| **Smart Processor** | Fully automated end-to-end processing |
| **Equipment Profile** | Camera/telescope configuration |

## Utility Tab

| Tool | Description |
|------|-------------|
| **Background Extraction** | Polynomial surface fitting |
| **ABE** | Automatic Background Extraction (RBF) |
| **Vignette Correction** | Radial vignette removal |
| **Chromatic Aberration** | Color fringing correction |
| **Image Statistics** | Per-channel statistics display |
| **Star Mask** | Automatic star mask generation |
| **Narrowband Combine** | SHO/HOO/HOS palette mapping |
| **Pixel Math** | Custom expression evaluator |
| **Channel Operations** | Split/merge/extract channels |
| **HDR Composition** | Merge multiple exposures |
| **Batch Processing** | Process entire sessions (Pro) |
| **Macros** | Record/playback processing sequences |
