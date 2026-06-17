# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Astraios — Windows build

import sys
from pathlib import Path

block_cipher = None
project_root = Path(SPECPATH).parent.parent

a = Analysis(
    [str(project_root / 'astraios' / '__main__.py')],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / 'astraios' / 'resources'), 'astraios/resources'),
    ],
    hiddenimports=[
        'astraios.core.device_manager',
        'astraios.core.image_io',
        'astraios.core.calibration',
        'astraios.core.stacking',
        'astraios.core.stretch',
        'astraios.core.background',
        'astraios.core.project',
        'astraios.ui.app',
        'astraios.ui.main_window',
        'astraios.ui.theme',
        'astraios.licensing.license_manager',
        'astraios.updater.auto_updater',
        'torch',
        'torchvision',
        'astropy',
        'ccdproc',
        'scipy',
        'cv2',
        'PIL',
        'PyQt6',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'notebook', 'jupyter'],
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
    console=False,  # GUI app, no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / 'astraios' / 'resources' / 'icons' / 'astraios.ico'),
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
