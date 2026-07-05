"""Isophote Analysis dialog — fit elliptical isophotes to a galaxy image.

The fitting core is ported from Seti Astro Suite Pro (GPL-3.0, Franklin
Marek); this dialog drives astraios.core.isophote_analysis.
"""

from __future__ import annotations

import csv
import logging

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot

log = logging.getLogger(__name__)

_COLUMNS = ["sma", "eps", "pa_deg", "intens", "rms", "a3", "b3", "a4", "b4"]


class _FitWorker(QThread):
    """Runs fit_isophotes off the GUI thread (can be slow on large images)."""

    finished_ok = pyqtSignal(object)  # IsophoteResult
    failed = pyqtSignal(str)

    def __init__(self, data, params):
        super().__init__()
        self._data = data
        self._params = params

    def run(self):
        try:
            from astraios.core.isophote_analysis import fit_isophotes

            result = fit_isophotes(self._data, self._params)
            self.finished_ok.emit(result)
        except Exception as exc:
            log.exception("Isophote fit failed")
            self.failed.emit(str(exc))


class IsophoteDialog(QDialog):
    """Fit and inspect elliptical isophotes on a galaxy image."""

    # (ndarray, label) — connect to main_window's _display_preview_only.
    show_model_requested = pyqtSignal(object, str)
    show_residual_requested = pyqtSignal(object, str)

    def __init__(self, image_data: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Isophote Analysis")
        self.setMinimumSize(640, 580)
        self._data = image_data
        self._worker: _FitWorker | None = None
        self._result = None  # IsophoteResult once fitted

        h, w = image_data.shape[-2], image_data.shape[-1]

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Fits a family of concentric elliptical isophotes to a galaxy "
            "image (growing outward and inward from a seed radius), then "
            "builds a smooth 2D model and residual (image minus model) to "
            "reveal spiral arms, bars, and tidal features."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        geom = QGroupBox("Geometry")
        gform = QFormLayout(geom)

        self._center_check = QCheckBox("Use image center")
        self._center_check.setChecked(True)
        self._center_check.toggled.connect(self._update_center_enabled)
        gform.addRow(self._center_check)

        self._cx_spin = QDoubleSpinBox()
        self._cx_spin.setRange(0.0, 1_000_000.0)
        self._cx_spin.setValue(w / 2.0)
        self._cy_spin = QDoubleSpinBox()
        self._cy_spin.setRange(0.0, 1_000_000.0)
        self._cy_spin.setValue(h / 2.0)
        center_row = QHBoxLayout()
        center_row.addWidget(QLabel("x"))
        center_row.addWidget(self._cx_spin)
        center_row.addWidget(QLabel("y"))
        center_row.addWidget(self._cy_spin)
        center_row.addWidget(help_dot(
            "Initial ellipse center, in pixels. Ignored if \"Use image "
            "center\" is ticked."
        ))
        center_row.addStretch()
        gform.addRow("Center", center_row)

        self._sma0_spin = QDoubleSpinBox()
        self._sma0_spin.setRange(1.0, 100000.0)
        self._sma0_spin.setValue(20.0)
        gform.addRow("Seed SMA (px)", self._row(
            self._sma0_spin,
            "Seed semi-major axis. Fitting grows outward to Max SMA and "
            "inward starting from this ring.",
        ))

        self._maxsma_check = QCheckBox("Auto (min(H,W) / 1.2)")
        self._maxsma_check.setChecked(True)
        self._maxsma_check.toggled.connect(self._update_maxsma_enabled)
        gform.addRow(self._maxsma_check)
        self._maxsma_spin = QDoubleSpinBox()
        self._maxsma_spin.setRange(1.0, 100000.0)
        self._maxsma_spin.setValue(min(h, w) / 1.2)
        gform.addRow("Max SMA (px)", self._row(
            self._maxsma_spin, "Maximum semi-major axis to fit.",
        ))

        self._step_spin = QDoubleSpinBox()
        self._step_spin.setRange(0.01, 50.0)
        self._step_spin.setSingleStep(0.05)
        self._step_spin.setDecimals(3)
        self._step_spin.setValue(0.2)
        gform.addRow("Step", self._row(
            self._step_spin,
            "Relative growth factor between rings: ring i+1 = ring i * "
            "(1 + step).",
        ))

        self._eps0_spin = QDoubleSpinBox()
        self._eps0_spin.setRange(0.0, 0.95)
        self._eps0_spin.setSingleStep(0.05)
        self._eps0_spin.setValue(0.20)
        gform.addRow("Initial ellipticity", self._row(
            self._eps0_spin, "Initial ellipticity guess, 1 - b/a.",
        ))

        self._pa0_spin = QDoubleSpinBox()
        self._pa0_spin.setRange(-360.0, 360.0)
        self._pa0_spin.setValue(90.0)
        gform.addRow("Initial PA (deg)", self._row(
            self._pa0_spin,
            "Initial position angle guess, measured from the +x axis.",
        ))

        fix_row = QHBoxLayout()
        self._fix_center_check = QCheckBox("Fix center")
        self._fix_pa_check = QCheckBox("Fix PA")
        self._fix_eps_check = QCheckBox("Fix ellipticity")
        fix_row.addWidget(self._fix_center_check)
        fix_row.addWidget(self._fix_pa_check)
        fix_row.addWidget(self._fix_eps_check)
        fix_row.addWidget(help_dot(
            "Hold the corresponding geometry fixed across all radii "
            "instead of letting the fit refine it ring by ring."
        ))
        fix_row.addStretch()
        gform.addRow(fix_row)

        self._high_harmonics_check = QCheckBox("Fit high harmonics (a3/b3/a4/b4)")
        self._high_harmonics_check.setToolTip(
            "<qt>Fits (and, in the rendered model, adds back) 3rd/4th-order "
            "harmonics -- lets the model follow mild isophote twists.</qt>"
        )
        gform.addRow(self._high_harmonics_check)

        lay.addWidget(geom)

        wedge = QGroupBox("Wedge exclusion")
        wform = QFormLayout(wedge)
        self._wedge_check = QCheckBox("Exclude an angular wedge")
        self._wedge_check.setToolTip(
            "<qt>Excludes an angular wedge from the fit, e.g. to skip a "
            "dust lane, foreground star, or companion galaxy.</qt>"
        )
        wform.addRow(self._wedge_check)
        self._wedge_pa_spin = QDoubleSpinBox()
        self._wedge_pa_spin.setRange(-360.0, 360.0)
        self._wedge_pa_spin.setValue(0.0)
        wform.addRow("Wedge PA (deg)", self._wedge_pa_spin)
        self._wedge_width_spin = QDoubleSpinBox()
        self._wedge_width_spin.setRange(0.0, 180.0)
        self._wedge_width_spin.setValue(30.0)
        wform.addRow("Wedge width (deg)", self._wedge_width_spin)
        lay.addWidget(wedge)

        opts = QGroupBox("Options")
        oform = QFormLayout(opts)
        self._downsample_combo = QComboBox()
        self._downsample_combo.addItems(["1", "2", "4"])
        oform.addRow("Downsample", self._row(
            self._downsample_combo,
            "Block-mean downsample factor applied before fitting, for a "
            "faster, coarser fit; the result is scaled back up to full "
            "resolution.",
        ))
        self._normalize_check = QCheckBox("Normalize input (pre-fit stretch)")
        self._normalize_check.setToolTip(
            "<qt>Applies a simple brightness stretch before fitting, "
            "helping the fit see faint outskirts in linear data.</qt>"
        )
        oform.addRow(self._normalize_check)
        lay.addWidget(opts)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        btns = QHBoxLayout()
        self._fit_btn = QPushButton("Fit")
        self._fit_btn.clicked.connect(self._fit)
        btns.addWidget(self._fit_btn)
        self._model_btn = QPushButton("Show Model")
        self._model_btn.setEnabled(False)
        self._model_btn.clicked.connect(self._show_model)
        btns.addWidget(self._model_btn)
        self._residual_btn = QPushButton("Show Residual")
        self._residual_btn.setEnabled(False)
        self._residual_btn.clicked.connect(self._show_residual)
        btns.addWidget(self._residual_btn)
        self._export_btn = QPushButton("Export CSV...")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_csv)
        btns.addWidget(self._export_btn)
        lay.addLayout(btns)

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self._table, 1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        lay.addWidget(close_btn)

        self._update_center_enabled()
        self._update_maxsma_enabled()

    @staticmethod
    def _row(widget, tip: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(widget)
        row.addWidget(help_dot(tip))
        row.addStretch()
        return row

    def _update_center_enabled(self):
        enabled = not self._center_check.isChecked()
        self._cx_spin.setEnabled(enabled)
        self._cy_spin.setEnabled(enabled)

    def _update_maxsma_enabled(self):
        self._maxsma_spin.setEnabled(not self._maxsma_check.isChecked())

    def get_params(self):
        from astraios.core.isophote_analysis import IsophoteParams

        cx = None if self._center_check.isChecked() else float(self._cx_spin.value())
        cy = None if self._center_check.isChecked() else float(self._cy_spin.value())
        maxsma = None if self._maxsma_check.isChecked() else float(self._maxsma_spin.value())
        return IsophoteParams(
            cx=cx, cy=cy,
            sma0=float(self._sma0_spin.value()),
            maxsma=maxsma,
            step=float(self._step_spin.value()),
            eps0=float(self._eps0_spin.value()),
            pa0_deg=float(self._pa0_spin.value()),
            fix_center=self._fix_center_check.isChecked(),
            fix_pa=self._fix_pa_check.isChecked(),
            fix_eps=self._fix_eps_check.isChecked(),
            high_harmonics=self._high_harmonics_check.isChecked(),
            use_wedge=self._wedge_check.isChecked(),
            wedge_pa_deg=float(self._wedge_pa_spin.value()),
            wedge_width_deg=float(self._wedge_width_spin.value()),
            downsample=int(self._downsample_combo.currentText()),
            normalize_input=self._normalize_check.isChecked(),
            build_model=True,
        )

    def _fit(self):
        params = self.get_params()
        self._fit_btn.setEnabled(False)
        self._model_btn.setEnabled(False)
        self._residual_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._status.setText("Fitting isophotes...")

        self._worker = _FitWorker(self._data, params)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, result):
        self._result = result
        self._fit_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._populate_table(result)
        has_model = result.model is not None
        self._model_btn.setEnabled(has_model)
        self._residual_btn.setEnabled(result.residual is not None)
        self._export_btn.setEnabled(result.n_rings > 0)
        self._status.setText(f"Fitted {result.n_rings} ring(s).")

    def _on_fail(self, msg: str):
        self._fit_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status.setText(f"Fit failed: {msg}")

    def _populate_table(self, result):
        self._table.setRowCount(result.n_rings)
        for i in range(result.n_rings):
            values = [
                result.sma[i], result.eps[i], result.pa_deg[i],
                result.intens[i], result.intens_rms[i],
                result.a3[i], result.b3[i], result.a4[i], result.b4[i],
            ]
            for col, v in enumerate(values):
                item = QTableWidgetItem(f"{v:.4f}")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(i, col, item)

    def _show_model(self):
        if self._result is None or self._result.model is None:
            return
        self.show_model_requested.emit(self._result.model, "Isophote model")

    def _show_residual(self):
        if self._result is None or self._result.residual is None:
            return
        self.show_residual_requested.emit(self._result.residual, "Isophote residual")

    def _export_csv(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Isophote Table", "isophotes.csv", "CSV (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(_COLUMNS)
                r = self._result
                for i in range(r.n_rings):
                    writer.writerow([
                        r.sma[i], r.eps[i], r.pa_deg[i], r.intens[i], r.intens_rms[i],
                        r.a3[i], r.b3[i], r.a4[i], r.b4[i],
                    ])
            self._status.setText(f"Exported: {path}")
        except Exception as exc:
            self._status.setText(f"Export failed: {exc}")
