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

## StarNet v2

StarNet v2 is licensed under GPL v3. When integrated, it will be:
- Isolated in a separate subprocess module
- Full GPL attribution provided
- Users will need to obtain StarNet separately
