# Astraios — Getting Started Guide

Welcome to **Astraios**, a professional astrophotography image processing application.

## Installation

**Note:** Astraios is currently available as **source code only**. We are working on pre-compiled installers (.exe, .dmg, .AppImage) for a future release.

### From Source (Linux, macOS, Windows)

```bash
# 1. Clone the repository
git clone https://github.com/majmichu1/astraios.git
cd astraios

# 2. Install dependencies (requires Poetry)
# If you don't have Poetry: https://python-poetry.org/docs/#installation
poetry install

# 3. Run the application
poetry run astraios

## First Launch

When you first open Astraios, you'll see the main interface with four panels:

```
┌─────────────┬──────────────────────┬─────────────┐
│   Project   │     Image Canvas     │    Tools    │
│   Panel     │   + Histogram        │    Panel    │
│   (left)    │     (center)         │   (right)   │
├─────────────┴──────────────────────┴─────────────┤
│               Log Panel (bottom)                  │
└──────────────────────────────────────────────────┘
```

## Quick Start — Your First Image

### Step 1: Open an Image
- **File → Open Image** or press `Ctrl+I`
- Or **drag and drop** a FITS/XISF file onto the canvas

### Step 2: Stretch the Image
1. Go to the **Tools** panel (right)
2. Click the **Stretch** tab
3. Adjust the **Midtone** slider (default: 0.25)
4. Click **Apply**

### Step 3: Save Your Work
- **File → Save Image As...** or `Ctrl+Shift+S`
- Choose format: FITS (astronomy), TIFF (print), PNG (web)

## Basic Workflow

The recommended processing order for a typical astrophotography image:

```
1. Pre-Process
   ├── Calibration (Dark/Flat/Bias frames)
   ├── Cosmetic Correction (Hot/cold pixels)
   └── Subframe Selector (Frame scoring)

2. Stacking
   └── Sigma-Clip Rejection + Average Integration

3. Stretch
   ├── Auto-Stretch (midtone balance)
   └── Histogram Transform (manual control)

4. Background
   ├── Background Extraction (remove gradients)
   └── ABE (Automatic Background Extraction)

5. Color
   ├── Color Calibration (white balance)
   └── SCNR (remove green cast)

6. Detail
   ├── Deconvolution (sharpening)
   ├── Noise Reduction
   └── Local Contrast (CLAHE)

7. Export
   └── File → Export (TIFF 16-bit / PNG / JPEG)
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+N` | New Project |
| `Ctrl+O` | Open Project |
| `Ctrl+S` | Save Project |
| `Ctrl+I` | Open Image |
| `Ctrl+Shift+S` | Save/Export Image |
| `Ctrl+,` | Preferences |
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` | Redo |
| `Ctrl+0` | Fit to Window |
| `Ctrl+1` | Zoom 100% |
| `Ctrl+P` | Pixel Math |
| `Ctrl+B` | Batch Processing |
| `Ctrl+Shift+P` | Smart Processor |
| `Ctrl+Q` | Quit |

## Features

Astraios is currently **100% free and open-source**. All features are available to everyone.

### Processing Pipeline
- **Calibration**: Dark/Flat/Bias frame management.
- **Stacking**: Sigma-clip, winsorized sigma, and linear fit rejection.
- **Stretch**: Auto-stretch, GHS, and manual curves.
- **Tools**: Background extraction, SCNR, deconvolution, and more.

### AI Features (Coming Soon)
- **AI Denoise**: Self-supervised learning for superior noise reduction.
- **AI Sharpen**: Neural deconvolution for sharpening without ringing artifacts.

## Subframe Selector

The Subframe Selector scores each of your light frames on several quality metrics
and lets you reject the worst frames before stacking.  High-quality frames
receive a higher weight (or are included at all), which directly improves
signal-to-noise and sharpness in the final stack.

### Where to find it

Open the **Tools** panel on the right, go to the **Stacking** tab, and click
**Open Subframe Selector…**.  If you have aligned frames in your project they
will be pre-loaded automatically.

### Step-by-step usage

1. **Load frames** — click **Load Frames…** to pick individual files, or
   **Load Folder…** to import every FITS/TIFF in a directory.  File names
   appear in the table immediately.
2. **Adjust scoring weights** — in the *Scoring Weights* group box, set how
   much each metric contributes to the composite Quality Score (values should
   sum to 1.0; they are automatically normalised).
3. **Choose a rejection mode** — in the *Rejection / Selection* group box pick:
   - **Sigma rejection** — rejects frames whose score is more than *N* sigma
     below the mean.  Use the *Rejection sigma* spinbox (default 1.5).
   - **Keep best N frames** — keeps the top-N frames by quality score.
   - **Keep best N%** — keeps the top percentage of frames (e.g. 80 %).
4. **Score All Frames** — click the button.  A progress bar tracks scoring;
   thumbnails appear as each frame is processed.
5. **Review results** — the table colour-codes rows green (accepted) or red
   (rejected).  Click any column header to sort by that metric.
6. **Adjust and re-apply** — change the rejection mode or thresholds at any
   time, then click **Re-apply Filter** (or tweak a spinbox value) to update
   accept/reject status instantly without re-scoring.
7. **Use Accepted Frames** — click this button to send the accepted frame
   list to the stacking pipeline, then click **Run Stack** in the Stacking tab.

### What each metric means

| Column | Meaning |
|--------|---------|
| **FWHM** | Full-Width at Half-Maximum of detected stars, in pixels.  Lower is sharper. |
| **Eccentricity** | How elongated stars are (0 = perfect circle, 1 = line).  Lower is better. |
| **SNR** | Signal-to-noise ratio estimate.  Higher is better. |
| **Background** | Median sky background level (0–1).  Lower means darker skies. |
| **Stars** | Number of stars detected.  More stars usually means better tracking and focus. |
| **Score** | Composite quality score (0–1) computed from the weighted combination of the above.  Higher is better. |
| **Status** | *Accepted* or *Rejected* based on the current filter. |

### Column sorting

Click any column header to sort the table by that metric.  Click again to
reverse the sort order.  Numeric columns (FWHM, Eccentricity, SNR,
Background, Stars, Score) sort as numbers rather than strings.

### Score caching

Once you score a set of frames the results are saved into your project JSON
file automatically.  The next time you open the project and run **Weighted
Average** stacking, Astraios reuses the cached scores — no re-measurement
needed.  This is especially useful for large datasets where scoring takes
several minutes.

## Need Help?

- **Help → About Astraios** — version info
- **GitHub Issues** — [Report bugs](https://github.com/majmichu1/astraios/issues)
