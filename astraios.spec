# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect torch data files (kernels, libs)
torch_datas = collect_data_files('torch', include_py_files=False)
# Collect astropy data
astropy_datas = collect_data_files('astropy')
# App resources and bundled model — paths are relative to spec file location.
extra_datas = [
    (str(Path('astraios') / 'resources'), str(Path('astraios') / 'resources')),
]
# Bundle any AI model weights that are actually present. The models are
# gitignored (downloaded from the CDN on first use), so they're absent in CI —
# PyInstaller errors out if a declared data file doesn't exist, so only add the
# ones that are there rather than hard-coding a single filename.
_models_dir = Path('astraios') / 'ai' / 'models'
if _models_dir.is_dir():
    for _m in _models_dir.glob('*.pt'):
        extra_datas.append((str(_m), str(_models_dir)))

hidden_imports = [
    # PyQt6
    'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets', 'PyQt6.QtOpenGL',
    'PyQt6.QtOpenGLWidgets', 'PyQt6.QtPrintSupport',
    # PyTorch — all backends must be present; torch/__init__.py imports them unconditionally
    'torch', 'torch.nn', 'torch.nn.functional', 'torch.cuda',
    'torch.backends', 'torch.backends.cuda', 'torch.backends.cudnn',
    'torch.backends.mkldnn', 'torch.backends.mkl', 'torch.backends.openmp',
    'torch.cuda._lazy_init',
    # numpy / scipy / astropy
    'numpy', 'scipy', 'scipy.ndimage', 'scipy.optimize', 'scipy.signal',
    'scipy.special._cdflib',
    'numpy.core._dtype_ctypes',
    'astropy', 'astropy.io.fits', 'astropy.wcs', 'astropy.stats',
    # image libs
    'cv2', 'PIL', 'PIL.Image', 'tifffile', 'rawpy',
    # astraios internals
    'astraios', 'astraios.core', 'astraios.ui', 'astraios.ai',
    'astraios.ai.inference.denoise', 'astraios.ai.inference.sharpen',
    'astraios.ai.models.denoise_model', 'astraios.ai.models.sharpen_model',
    'astraios.ai.models.unet',
    # misc
    'platformdirs', 'requests', 'packaging',
] + collect_submodules('astraios') + collect_submodules('PyQt6')

a = Analysis(
    ['astraios/__main__.py'],
    pathex=['.'],
    binaries=[],
    datas=torch_datas + astropy_datas + extra_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'matplotlib', 'notebook', 'IPython',
        # Drop bundled NVIDIA libs (~6GB) — app uses system CUDA drivers at runtime
        'nvidia', 'nvidia.cudnn', 'nvidia.cublas', 'nvidia.cuda_runtime',
        'nvidia.cuda_nvrtc', 'nvidia.cuda_cupti', 'nvidia.cufft',
        'nvidia.curand', 'nvidia.cusolver', 'nvidia.cusparse',
        'nvidia.nccl', 'nvidia.nvtx', 'nvidia.nvjitlink',
        # torch.backends.mkldnn and torch.cuda._lazy_init must NOT be excluded —
        # torch/__init__.py imports them unconditionally at startup.

        # --- AGRESYWNE ODCHUDZANIE LINUKSA (< 2GB limit) ---
        'PyQt5', 'PySide2', 'PySide6', 'wx',  # Nieużywane frameworki UI
        'scipy.datasets', # Zbędne gigabajty danych testowych
        'botocore', 'boto3', 'awscli', # Śmieci z chmury
        'triton', # Potężny kompilator GPU, na Linuksie potrafi ważyć setki MB
        'transformers', 'huggingface_hub', 'tensorboard', # Narzędzia ML, których nie używasz wprost
        'sympy', 'networkx', # Często dociągane przez torcha, a niepotrzebne przy prostej inferencji
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Astraios',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon='astraios/resources/icon.ico' if Path('astraios/resources/icon.ico').exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Astraios',
)
