"""Copy Astrometry dialog.

Drives astraios.core.copy_astrometry (transfer a WCS/SIP solution from a
plate-solved donor FITS header onto the current image's header), ported from
Seti Astro Suite Pro (GPL-3.0, Franklin Marek).

This is a header-only operation — no pixel data changes — so it runs
synchronously (no worker thread) and mutates the current image's ``.header``
dict in place on success.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot

log = logging.getLogger(__name__)


class CopyAstrometryDialog(QDialog):
    """Copy a plate-solved WCS solution from a donor FITS onto the current image."""

    applied = pyqtSignal()  # header updated in place; no image data to pass along

    def __init__(self, current_image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Copy Astrometry")
        self.setMinimumWidth(440)
        self._current_image = current_image
        self._source_header: dict | None = None

        lay = QVBoxLayout(self)
        intro_row = QHBoxLayout()
        intro = QLabel(
            "Copies a WCS (plate-solve) astrometric solution from a "
            "plate-solved source FITS onto the current image's header. "
            "Use this after stacking/mosaicking discards a per-frame "
            "solution, or to seed a tile with a neighbor's placement."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        intro_row.addWidget(intro, 1)
        intro_row.addWidget(help_dot(
            "Copies CTYPE/CRVAL/CRPIX/CD/PC/CDELT/SIP distortion keywords "
            "(and NAXIS1/2) from the source header onto the current "
            "image's header, replacing any existing WCS block there. All "
            "other header keys on the current image are left untouched."
        ))
        lay.addLayout(intro_row)

        src_row = QHBoxLayout()
        self._src_edit = QLineEdit()
        self._src_edit.setPlaceholderText("Load the plate-solved source FITS...")
        self._src_edit.setReadOnly(True)
        browse = QPushButton("Load...")
        browse.clicked.connect(self._browse)
        src_row.addWidget(self._src_edit, 1)
        src_row.addWidget(browse)
        lay.addLayout(src_row)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        btns = QHBoxLayout()
        self._apply_btn = QPushButton("Copy Astrometry")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply)
        btns.addWidget(self._apply_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Plate-Solved Source", "",
            "FITS (*.fit *.fits *.fts)"
        )
        if not path:
            return
        self._load_source(path)

    def _load_source(self, path: str):
        """Load ``path``'s FITS header and validate it carries a WCS solution."""
        from astraios.core.copy_astrometry import wcs_keywords_present

        try:
            from astropy.io import fits
            with fits.open(path) as hdul:
                header = dict(hdul[0].header)
        except Exception as exc:
            self._status.setText(f"Could not load: {exc}")
            self._source_header = None
            self._apply_btn.setEnabled(False)
            return

        if not wcs_keywords_present(header):
            self._status.setText(
                "No WCS solution found in that file (missing CRVAL1/CRVAL2) "
                "— pick a plate-solved image."
            )
            self._source_header = None
            self._apply_btn.setEnabled(False)
            return

        self._source_header = header
        self._src_edit.setText(Path(path).name)
        self._apply_btn.setEnabled(True)
        self._status.setText("")

    def _apply(self):
        from astraios.core.copy_astrometry import copy_astrometry

        if self._source_header is None or self._current_image is None:
            return
        try:
            target_header = getattr(self._current_image, "header", None) or {}
            new_header = copy_astrometry(self._source_header, target_header)
        except Exception as exc:
            log.exception("Copy astrometry failed")
            self._status.setText(f"Failed: {exc}")
            return

        self._current_image.header = new_header
        self._status.setText("Astrometric solution copied onto the current image.")
        self.applied.emit()
