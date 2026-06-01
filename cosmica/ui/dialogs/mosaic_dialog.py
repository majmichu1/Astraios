"""Mosaic Stitching Dialog — combine overlapping panels into a seamless mosaic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from cosmica.core.image_io import load_image
from cosmica.core.mosaic import (
    BlendMethod,
    MosaicParams,
    MosaicResult,
    NormalizeMethod,
    mosaic_stitch,
)


class MosaicWorker(QThread):
    """Runs mosaic stitching off the main thread."""

    progress = pyqtSignal(float, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, panels: list[np.ndarray], params: MosaicParams):
        super().__init__()
        self._panels = panels
        self._params = params

    def run(self):
        try:
            result = mosaic_stitch(
                self._panels,
                params=self._params,
                progress=self._emit_progress,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def _emit_progress(self, fraction: float, message: str):
        self.progress.emit(fraction, message)


class MosaicDialog(QDialog):
    """Dialog for mosaic stitching of multiple overlapping panels."""

    result_ready = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mosaic Stitching")
        self.setMinimumSize(500, 480)

        self._panel_paths: list[Path] = []
        self._worker: MosaicWorker | None = None

        layout = QVBoxLayout(self)

        # --- Blend method ---
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Blend method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(["Feather", "Multiband (Laplacian)", "Average"])
        method_row.addWidget(self._method_combo)
        layout.addLayout(method_row)

        # --- Feather width ---
        feather_row = QHBoxLayout()
        feather_row.addWidget(QLabel("Feather width (px):"))
        self._feather_spin = QSpinBox()
        self._feather_spin.setRange(0, 500)
        self._feather_spin.setValue(50)
        feather_row.addWidget(self._feather_spin)
        layout.addLayout(feather_row)

        # --- Photometric normalization ---
        norm_row = QHBoxLayout()
        self._normalize_check = QCheckBox("Photometric normalization")
        self._normalize_check.setChecked(True)
        self._normalize_check.toggled.connect(self._on_normalize_toggled)
        norm_row.addWidget(self._normalize_check)

        self._norm_method_combo = QComboBox()
        self._norm_method_combo.addItems(["Robust (median)", "Linear (mean)"])
        norm_row.addWidget(self._norm_method_combo)

        layout.addLayout(norm_row)

        self._per_channel_check = QCheckBox("Normalize each channel independently")
        self._per_channel_check.setChecked(True)
        layout.addWidget(self._per_channel_check)

        self._clip_check = QCheckBox("Clip negative values")
        self._clip_check.setChecked(True)
        layout.addWidget(self._clip_check)

        # --- Panel list ---
        layout.addWidget(QLabel("Panels:"))
        self._panel_list = QListWidget()
        layout.addWidget(self._panel_list)

        # --- Add / Remove buttons ---
        list_btn_row = QHBoxLayout()

        add_btn = QPushButton("Add Panels...")
        add_btn.clicked.connect(self._add_panels)
        list_btn_row.addWidget(add_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)
        list_btn_row.addWidget(remove_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all)
        list_btn_row.addWidget(clear_btn)

        layout.addLayout(list_btn_row)

        # --- Progress ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("Add at least 2 panels to begin")
        layout.addWidget(self._status_label)

        # --- Action buttons ---
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._run_btn = QPushButton("Stitch Mosaic")
        self._run_btn.clicked.connect(self._run)
        btn_row.addWidget(self._run_btn)

        layout.addLayout(btn_row)

        self._on_normalize_toggled(self._normalize_check.isChecked())

    def _on_normalize_toggled(self, checked: bool):
        self._norm_method_combo.setEnabled(checked)
        self._per_channel_check.setEnabled(checked)

    def _add_panels(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Panel Images",
            "",
            "All Supported (*.fit *.fits *.fts *.xisf *.tif *.tiff *.png);;All (*)",
        )
        for p in paths:
            path = Path(p)
            if path not in self._panel_paths:
                self._panel_paths.append(path)
                self._panel_list.addItem(path.name)
        self._update_status()

    def _remove_selected(self):
        for item in self._panel_list.selectedItems():
            idx = self._panel_list.row(item)
            self._panel_list.takeItem(idx)
            self._panel_paths.pop(idx)
        self._update_status()

    def _clear_all(self):
        self._panel_list.clear()
        self._panel_paths.clear()
        self._update_status()

    def _update_status(self):
        n = len(self._panel_paths)
        if n < 2:
            self._status_label.setText(f"{n} panel(s) loaded — need at least 2")
        else:
            self._status_label.setText(f"{n} panels ready to stitch")

    def _run(self):
        if len(self._panel_paths) < 2:
            self._status_label.setText("Need at least 2 panels")
            return

        self._run_btn.setEnabled(False)
        self._progress_bar.setValue(0)
        self._status_label.setText("Loading panels...")

        try:
            panels = [load_image(str(p)).data for p in self._panel_paths]
        except Exception as e:
            self._status_label.setText(f"Error loading panels: {e}")
            self._run_btn.setEnabled(True)
            return

        method_map = {
            0: BlendMethod.FEATHER,
            1: BlendMethod.MULTIBAND,
            2: BlendMethod.AVERAGE,
        }

        norm_map = {
            0: NormalizeMethod.ROBUST,
            1: NormalizeMethod.LINEAR,
        }

        params = MosaicParams(
            blend_method=method_map.get(
                self._method_combo.currentIndex(), BlendMethod.FEATHER
            ),
            feather_width=self._feather_spin.value(),
            normalize=norm_map.get(
                self._norm_method_combo.currentIndex(), NormalizeMethod.ROBUST
            ) if self._normalize_check.isChecked() else NormalizeMethod.NONE,
            norm_per_channel=self._per_channel_check.isChecked(),
            clip_negative=self._clip_check.isChecked(),
        )

        self._worker = MosaicWorker(panels, params)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, fraction: float, message: str):
        self._progress_bar.setValue(int(fraction * 100))
        self._status_label.setText(message)

    def _on_finished(self, result: MosaicResult):
        self._run_btn.setEnabled(True)
        self._progress_bar.setValue(100)
        self._status_label.setText(
            f"Mosaic complete — {result.n_panels} panels, "
            f"output {result.output_shape}"
        )
        self.result_ready.emit(result)
        self.accept()

    def _on_error(self, message: str):
        self._run_btn.setEnabled(True)
        self._status_label.setText(f"Error: {message}")
