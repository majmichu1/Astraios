# Installing Astraios

Every installer is small. It creates a private Python environment with
[uv](https://github.com/astral-sh/uv), detects your GPU and downloads the
matching PyTorch build, then installs Astraios into it. Nothing touches your
system Python and no root or administrator rights are needed on Linux and
macOS. Expect the PyTorch download to take a few minutes (about 2 GB for the
CUDA build, less for CPU).

Get the files from the [latest release](https://github.com/majmichu1/Astraios/releases/latest).
Each release asset has a `<name>.sha256` sidecar you can check with
`sha256sum -c` (Linux/macOS) or `Get-FileHash` (Windows) from the release
that introduced them onward.

## Windows 10/11 (x64)

1. Download `Astraios-Setup-<version>.exe` and run it.
2. The installer looks for an NVIDIA driver (`nvidia-smi`). If found, it installs CUDA 12.8 PyTorch (RTX 50-series and older); otherwise CPU PyTorch.
3. Launch Astraios from the Start Menu.

The installer is not code-signed, so Windows SmartScreen may show a warning;
choose "More info" and "Run anyway" if you downloaded it from the GitHub
release page. Uninstall from Settings > Apps.

## Linux (x86_64)

Installer script:

```bash
bash install-astraios.sh
```

Installs under `~/.local`, adds a desktop menu entry, and uses CUDA 12.8
PyTorch when `nvidia-smi` works, CPU PyTorch otherwise. Tested on Fedora,
Bazzite (immutable), Ubuntu and Arch; needs `curl` or `wget` only.

AppImage (portable):

```bash
chmod +x Astraios-<version>-x86_64.AppImage
./Astraios-<version>-x86_64.AppImage
```

The AppImage bundles Python, Qt and the science stack. PyTorch is not inside
(a CUDA build alone is larger than GitHub's asset limit); the first launch
provisions the matching build into a per-user directory. Needs `libfuse2`
like any AppImage.

## macOS 11 or later

```bash
bash install-astraios-macos.sh
```

Creates `Astraios.app` for Launchpad and Spotlight. Apple Silicon runs on
the GPU through PyTorch's Metal (MPS) backend; Intel Macs run on the CPU.
Both are validated on real Apple hardware in CI.

## From source (developers)

```bash
git clone https://github.com/majmichu1/Astraios.git
cd Astraios
poetry install --with dev
poetry run astraios
```

Python 3.11 to 3.14. Poetry installs the CPU PyTorch from PyPI; for CUDA,
install a CUDA build of `torch` into the same environment first, for example
from `https://download.pytorch.org/whl/cu128`. The device in use is printed
in the log panel at startup.

## Which GPU is used

Astraios picks CUDA, then Apple Metal, then CPU, automatically. AMD and Intel
GPUs are not accelerated. Set `ASTRAIOS_FORCE_CPU=1` to force the CPU. Large
images are processed in tiles so 4 to 8 GB of VRAM is enough; on a CUDA
out-of-memory error the operation falls back to the CPU.

## Updating

Astraios checks GitHub for a newer release at startup (Preferences > Updates
to turn it off) and can download and run the installer for you. Downloads are
verified against the release's `.sha256` sidecar when present. You can also
just run the newer installer over the old one.

## Troubleshooting

- The log panel prints the device at startup ("CUDA", "MPS" or "CPU"). If you
  expected CUDA and see CPU, update the NVIDIA driver and re-run the installer.
- Linux: if the desktop entry does not appear, log out and back in.
- Windows: if the app does not start, run the installer again; it repairs the
  environment.
- For anything else, open an issue with the log panel output attached, or use
  the [compatibility report](https://github.com/majmichu1/Astraios/issues/new?template=compatibility_report.yml)
  form to tell us what works on your hardware.
