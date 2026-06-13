"""Narrowband Combining Dialog — assign filter images and choose palette mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from cosmica.core.image_io import load_image
from cosmica.core.narrowband import (
    NarrowbandPalette,
    NarrowbandParams,
    combine_narrowband,
)


class NarrowbandDialog(QDialog):
    """Dialog for combining narrowband filter images into a color composite."""

    result_ready = pyqtSignal(np.ndarray)  # emits combined RGB image

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Narrowband Combining")
        self.setMinimumWidth(450)

        self._images: dict[str, np.ndarray] = {}

        layout = QVBoxLayout(self)

        # Filter assignment
        assign_group = QGroupBox("Filter Assignment")
        assign_layout = QVBoxLayout(assign_group)

        info = QLabel("Load mono images for each narrowband filter channel.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #969696; font-size: 11px;")
        assign_layout.addWidget(info)

        self._ha_label = QLabel("Ha: (none)")
        self._btn_ha = QPushButton("Load Ha...")
        self._btn_ha.clicked.connect(lambda: self._load_filter("ha", self._ha_label))
        row = QHBoxLayout()
        row.addWidget(self._ha_label, 1)
        row.addWidget(self._btn_ha)
        assign_layout.addLayout(row)

        self._oiii_label = QLabel("OIII: (none)")
        self._btn_oiii = QPushButton("Load OIII...")
        self._btn_oiii.clicked.connect(lambda: self._load_filter("oiii", self._oiii_label))
        row = QHBoxLayout()
        row.addWidget(self._oiii_label, 1)
        row.addWidget(self._btn_oiii)
        assign_layout.addLayout(row)

        self._sii_label = QLabel("SII: (none)")
        self._btn_sii = QPushButton("Load SII...")
        self._btn_sii.clicked.connect(lambda: self._load_filter("sii", self._sii_label))
        row = QHBoxLayout()
        row.addWidget(self._sii_label, 1)
        row.addWidget(self._btn_sii)
        assign_layout.addLayout(row)

        layout.addWidget(assign_group)

        # Palette selection
        palette_group = QGroupBox("Palette")
        palette_layout = QVBoxLayout(palette_group)

        self._palette_combo = QComboBox()
        self._palette_combo.addItems([
            "SHO (Hubble Palette)",
            "HOO (Ha + OIII)",
            "HOS (Natural-ish)",
            "HSO",
            "OHS",
            "OSH",
            "Foraxx (dynamic)",
            "Dynamic SHO",
        ])
        self._palette_combo.setToolTip(
            "SHO: SII=Red, Ha=Green, OIII=Blue (Hubble)\n"
            "HOO: Ha=Red, OIII=Green+Blue\n"
            "HOS: Ha=Red, OIII=Green, SII=Blue\n"
            "HSO: Ha=Red, SII=Green, OIII=Blue\n"
            "OHS: OIII=Red, Ha=Green, SII=Blue\n"
            "OSH: OIII=Red, SII=Green, Ha=Blue\n"
            "Foraxx: R=Ha, B=OIII, green dynamically blends Ha/OIII\n"
            "Dynamic SHO: SHO with the dynamic green blend (R=SII)"
        )
        palette_layout.addWidget(self._palette_combo)
        layout.addWidget(palette_group)

        # Status
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #969696;")
        layout.addWidget(self._status_label)

        # Buttons
        btn_row = QHBoxLayout()
        self._btn_combine = QPushButton("Combine")
        self._btn_combine.setEnabled(False)
        self._btn_combine.clicked.connect(self._combine)
        btn_row.addWidget(self._btn_combine)

        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self._btn_cancel)
        layout.addLayout(btn_row)

    def _load_filter(self, key: str, label: QLabel):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Load {key.upper()} Image", "",
            "FITS (*.fit *.fits *.fts);;All Supported (*.fit *.fits *.fts *.xisf *.tif *.tiff *.png);;All (*)",
        )
        if not path:
            return
        try:
            img = load_image(path)
            data = img.data
            # Ensure mono
            if data.ndim == 3:
                data = np.mean(data, axis=0)
            self._images[key] = data
            label.setText(f"{key.upper()}: {Path(path).name}")
            self._update_combine_button()
        except Exception as e:
            self._status_label.setText(f"Error loading {key}: {e}")
            self._status_label.setStyleSheet("color: #ff6b6b;")

    def _update_combine_button(self):
        has_ha = "ha" in self._images
        self._btn_combine.setEnabled(has_ha)
        if has_ha:
            filters = [k.upper() for k in self._images]
            self._status_label.setText(f"Ready: {', '.join(filters)}")
            self._status_label.setStyleSheet("color: #69db7c;")

    def _combine(self):
        palette_map = {
            0: NarrowbandPalette.SHO,
            1: NarrowbandPalette.HOO,
            2: NarrowbandPalette.HOS,
            3: NarrowbandPalette.HSO,
            4: NarrowbandPalette.OHS,
            5: NarrowbandPalette.OSH,
            6: NarrowbandPalette.FORAXX,
            7: NarrowbandPalette.DYNAMIC_SHO,
        }
        params = NarrowbandParams(
            palette=palette_map.get(self._palette_combo.currentIndex(), NarrowbandPalette.SHO),
        )
        try:
            result = combine_narrowband(self._images, params)
            self.result_ready.emit(result)
            self.accept()
        except Exception as e:
            self._status_label.setText(f"Error: {e}")
            self._status_label.setStyleSheet("color: #ff6b6b;")
