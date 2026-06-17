<#
  Astraios first-run environment bootstrap (Windows).

  Runs once, at the end of installation. Creates an isolated Python environment
  next to the app, detects whether an NVIDIA GPU is present, and installs the
  matching PyTorch build (CUDA or CPU) plus Astraios and its dependencies.

  This is what lets the download stay tiny (~5 MB) while still delivering real
  GPU acceleration: the heavy, GPU-specific wheels are fetched on the user's own
  machine, for their own hardware — sidestepping GitHub's 2 GB release-asset cap
  and the "one frozen build can't fit every GPU" problem.

  Everything is logged to setup.log so a failed install can be diagnosed.
#>
param(
    # Install directory (Inno passes {app}); defaults to this script's folder.
    [string]$AppDir = $PSScriptRoot
)

$ErrorActionPreference = "Stop"
$log = Join-Path $AppDir "setup.log"

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $log -Value $line
}

try {
    Log "Astraios environment setup starting in $AppDir"

    $uv   = Join-Path $AppDir "uv.exe"
    $venv = Join-Path $AppDir ".venv"
    $py   = Join-Path $venv "Scripts\python.exe"
    $wheel = Get-ChildItem -Path $AppDir -Filter "astraios-*.whl" | Select-Object -First 1
    if (-not $wheel) { throw "No astraios wheel found in $AppDir" }
    Log "Wheel: $($wheel.Name)"

    if (-not (Test-Path $uv)) { throw "uv.exe not found in $AppDir" }

    # --- 1. Detect NVIDIA GPU ------------------------------------------------
    # Prefer the actual driver (nvidia-smi); fall back to the display adapter
    # name. Either signal is enough to justify the CUDA wheels.
    $hasNvidia = $false
    try {
        $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($smi) { & $smi.Source | Out-Null; if ($LASTEXITCODE -eq 0) { $hasNvidia = $true } }
    } catch { }
    if (-not $hasNvidia) {
        try {
            $gpu = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
                   Where-Object { $_.Name -match "NVIDIA" }
            if ($gpu) { $hasNvidia = $true }
        } catch { }
    }

    if ($hasNvidia) {
        # cu128 = CUDA 12.8 wheels: cover the newest cards (Blackwell / RTX 50xx)
        # and remain backward-compatible with older NVIDIA GPUs given a recent
        # driver. The wheels bundle the CUDA runtime, so the user needs only an
        # up-to-date driver, not a separate CUDA toolkit install.
        $torchIndex = "https://download.pytorch.org/whl/cu128"
        Log "NVIDIA GPU detected -> installing CUDA (cu128) PyTorch"
    } else {
        $torchIndex = "https://download.pytorch.org/whl/cpu"
        Log "No NVIDIA GPU -> installing CPU PyTorch"
    }

    # --- 2. Create the venv with a managed CPython ---------------------------
    # uv downloads a standalone Python 3.11 if the user has none — no system
    # Python required.
    if (-not (Test-Path $py)) {
        Log "Creating virtual environment (managed Python 3.11)..."
        & $uv venv --python 3.11 $venv
        if ($LASTEXITCODE -ne 0) { throw "uv venv failed ($LASTEXITCODE)" }
    } else {
        Log "Reusing existing virtual environment"
    }

    # --- 3. Install PyTorch from the chosen index FIRST ----------------------
    # Installing torch up front pins the correct GPU/CPU build; the subsequent
    # app install then sees torch already satisfied and won't pull the default
    # (CPU) wheel from PyPI over it.
    Log "Installing PyTorch (this is the large download)..."
    & $uv pip install --python $py torch torchvision --index-url $torchIndex
    if ($LASTEXITCODE -ne 0) { throw "torch install failed ($LASTEXITCODE)" }

    # --- 4. Install Astraios + remaining dependencies ------------------------
    # PyPI as the default index for everything else; the torch index stays as an
    # extra so torchvision's torch pin still resolves to the GPU build.
    Log "Installing Astraios and dependencies..."
    & $uv pip install --python $py $wheel.FullName --extra-index-url $torchIndex
    if ($LASTEXITCODE -ne 0) { throw "Astraios install failed ($LASTEXITCODE)" }

    # --- 5. Smoke test -------------------------------------------------------
    Log "Verifying install..."
    & $py -c "import astraios, torch; print('Astraios', astraios.__version__, '| torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { throw "import smoke test failed ($LASTEXITCODE)" }

    Log "Setup complete."
    exit 0
}
catch {
    Log "SETUP FAILED: $_"
    exit 1
}
