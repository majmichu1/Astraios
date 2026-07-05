# Astraios — Credits & Open Source Acknowledgments

Astraios is built on the shoulders of many excellent open source projects.

## Core Dependencies

| Library | License | Usage |
|---------|---------|-------|
| [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) | GPL v3 | User interface framework |
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | GPU-accelerated computation |
| [NumPy](https://numpy.org/) | BSD-3-Clause | Array operations |
| [SciPy](https://scipy.org/) | BSD-3-Clause | Scientific computing, optimization |
| [Astropy](https://www.astropy.org/) | BSD-3-Clause | FITS file I/O, astronomical utilities |
| [ccdproc](https://ccdproc.readthedocs.io/) | BSD-3-Clause | CCD data processing primitives |
| [OpenCV](https://opencv.org/) | Apache 2.0 | Image alignment, contour detection |
| [Pillow](https://python-pillow.org/) | HPND | Common image format I/O |
| [Requests](https://docs.python-requests.org/) | Apache 2.0 | HTTP client for updates |
| [platformdirs](https://github.com/platformdirs/platformdirs) | MIT | Cross-platform user data directories |
| [packaging](https://packaging.pypa.io/) | Apache 2.0 / BSD-2-Clause | Version comparison |

## Development Dependencies

| Library | License | Usage |
|---------|---------|-------|
| [pytest](https://pytest.org/) | MIT | Test framework |
| [Ruff](https://github.com/astral-sh/ruff) | MIT | Linter and formatter |
| [PyInstaller](https://pyinstaller.org/) | GPL v2 (with exception) | Application packaging |
| [Poetry](https://python-poetry.org/) | MIT | Dependency management |

## Algorithm References

- **Sigma Clipping Rejection**: Standard kappa-sigma clipping as described in astronomical
  image processing literature.

- **Midtone Transfer Function (STF)**: Independent implementation of the mathematical MTF.

- **Background Extraction**: Polynomial surface fitting approach.

- **Noise2Self**: Self-supervised denoising architecture. See the original paper:
  *Krull et al., "Noise2Void — Learning Denoising from Single Noisy Images", CVPR 2019.*

## Seti Astro Suite Pro (Franklin Marek)

Several Astraios tools are ported and adapted from
[Seti Astro Suite Pro](https://github.com/setiastro/setiastrosuitepro),
Copyright Franklin Marek, licensed under GPL v3 (the same license as
Astraios). Each ported module carries an attribution header naming its
origin. Ported tools include the FX effects (Orton glow, star glow, soft
focus, split tone, film grain), diffraction spikes, saturation and chroma
hue curves, Halo-B-Gon halo reduction, WaveScale HDR, WaveScale Dark
Enhance, texture clarity, selective color and luminance, and pedestal
tools. The implementations were adapted to Astraios conventions
(channels-first float32 data, GPU execution through the device manager
with CPU fallbacks) while preserving the original processing math.
Sincere thanks to Franklin Marek for publishing this work under a
license that allows the astrophotography community to build on it.

## StarNet v2

StarNet v2 is licensed under GPL v3. When integrated, it will be:
- Isolated in a separate subprocess module
- Full GPL attribution provided
- Users will need to obtain StarNet separately
