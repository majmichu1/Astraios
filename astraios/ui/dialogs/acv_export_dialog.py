"""ACV (Photoshop Curves) Exporter dialog.

Drives astraios.core.acv_export (write curve control points to a
Photoshop-compatible .acv binary curves file).

See astraios/core/acv_export.py's module docstring for provenance: this is
NOT ported from Seti Astro Suite Pro. SASpro's file named ``acv_exporter.py``
implements an unrelated "Astro Catalogue Viewer" image-export-by-catalog
feature and contains no Photoshop-curves logic to port. This dialog and its
core module are an independent implementation of Adobe's publicly
documented .acv format.
"""

from __future__ import annotations

import logging

from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot

log = logging.getLogger(__name__)

_DEFAULT_CURVES = [("Master", [(0.0, 0.0), (1.0, 1.0)])]


class ACVExportDialog(QDialog):
    """Export the current curve control points to a Photoshop .acv file."""

    def __init__(self, curves: list | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export .acv (Photoshop Curves)")
        self.setMinimumWidth(420)
        self._curves = list(curves) if curves else list(_DEFAULT_CURVES)

        lay = QVBoxLayout(self)
        intro_row = QHBoxLayout()
        intro = QLabel(
            "Writes the current curve's control points to a Photoshop-"
            "compatible .acv file that can be loaded in Photoshop's Curves "
            "dialog (or any tool that reads the standard .acv format)."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        intro_row.addWidget(intro, 1)
        intro_row.addWidget(help_dot(
            "The .acv format stores each channel's control points as "
            "(input, output) pairs scaled to 0-255, in curve order — "
            "Composite, then Red/Green/Blue for an RGB document. Photoshop "
            "supports 2-19 points per curve; exporting more than 19 points "
            "on any channel will fail."
        ))
        lay.addLayout(intro_row)

        summary_lines = [
            f"{name}: {len(points)} point(s)" for name, points in self._curves
        ]
        self._summary = QLabel("\n".join(summary_lines) or "(no curves)")
        self._summary.setWordWrap(True)
        lay.addWidget(self._summary)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self._save_btn = QPushButton("Save .acv...")
        self._save_btn.clicked.connect(self._save)
        self._save_btn.setEnabled(bool(self._curves))
        btns.addWidget(self._save_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

    def _save(self):
        from astraios.core.acv_export import export_acv

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Photoshop Curves", "curve.acv", "Photoshop Curves (*.acv)",
        )
        if not path:
            return
        try:
            out = export_acv(self._curves, path)
        except Exception as exc:
            log.exception("ACV export failed")
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._status.setText(f"Wrote: {out}")
        QMessageBox.information(self, "Exported", f"Wrote:\n{out}")
