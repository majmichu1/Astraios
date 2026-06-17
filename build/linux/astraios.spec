# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Astraios — Linux build

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
    excludes=['tkinter', 'matplotlib', 'notebook', 'jupyter'],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='astraios',
    debug=False,
    strip=True,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=True,
    name='astraios',
)
