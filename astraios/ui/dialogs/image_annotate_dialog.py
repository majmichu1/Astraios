"""Image Annotate ("What's In My Image") dialog.

Drives astraios.core.image_annotate (structured catalog-object identification
against a plate-solved image's WCS footprint: embedded Messier/NGC/IC deep-sky
objects, plus opt-in bright-star identification from a locally installed Gaia
DR3 catalog), ported from Seti Astro Suite Pro (GPL-3.0, Franklin Marek).

Rendering an annotated copy of the image reuses
``astraios.core.finder_chart.render_finder_chart`` — this dialog does not
reimplement any marker drawing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot, param_help

log = logging.getLogger(__name__)


class _NumericItem(QTableWidgetItem):
    """A table item that sorts by its underlying numeric value, not its text."""

    def __init__(self, value: float, text: str):
        super().__init__(text)
        self._value = float(value)

    def __lt__(self, other):
        if isinstance(other, _NumericItem):
            return self._value < other._value
        return super().__lt__(other)


class _IdentifyWorker(QThread):
    progress = pyqtSignal(float, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image_shape, wcs_header, params):
        super().__init__()
        self._image_shape = image_shape
        self._wcs_header = wcs_header
        self._params = params

    def run(self):
        try:
            from astraios.core.image_annotate import identify_objects
            result = identify_objects(
                self._image_shape, self._wcs_header, self._params,
                progress=lambda f, m: self.progress.emit(f, m),
            )
            self.done.emit(result)
        except Exception as exc:
            log.exception("Image annotate identification failed")
            self.failed.emit(str(exc))


class _RenderWorker(QThread):
    progress = pyqtSignal(float, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image, wcs_header, params, catalog_stars, dso_list):
        super().__init__()
        self._image = image
        self._wcs_header = wcs_header
        self._params = params
        self._catalog_stars = catalog_stars
        self._dso_list = dso_list

    def run(self):
        try:
            from astraios.core.finder_chart import render_finder_chart
            result = render_finder_chart(
                self._image, self._wcs_header, self._params,
                catalog_stars=self._catalog_stars, dso_list=self._dso_list,
                progress=lambda f, m: self.progress.emit(f, m),
            )
            self.done.emit(result)
        except Exception as exc:
            log.exception("Image annotate render failed")
            self.failed.emit(str(exc))


class ImageAnnotateDialog(QDialog):
    """Identify catalog objects inside a plate-solved image's footprint.

    Read-only with respect to the working image while identifying — the
    result is a structured object table. "Annotate onto image" additionally
    renders labeled markers over a copy of the image (via
    ``astraios.core.finder_chart``) and emits it through ``result_ready``.
    """

    result_ready = pyqtSignal(object)
    marker_requested = pyqtSignal(float, float, str)

    _COL_NAME, _COL_CATALOG, _COL_TYPE, _COL_MAG, _COL_SIZE, _COL_X, _COL_Y = range(7)

    def __init__(self, image: np.ndarray, parent=None, wcs_header: Any = None):
        super().__init__(parent)
        self.setWindowTitle("What's In My Image")
        self.setMinimumSize(640, 560)
        self._image = image
        self._wcs_header = wcs_header
        self._identify_worker: _IdentifyWorker | None = None
        self._render_worker: _RenderWorker | None = None
        self._results: list[Any] = []  # IdentifiedObject list; exposed for scripted verification

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Identifies deep-sky objects (and, optionally, bright stars) that "
            "fall inside the plate-solved image's field of view. Click a row "
            "to center the canvas on that object; use \"Annotate onto Image\" "
            "to render labeled markers over a copy of the image."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        if self._wcs_header is None:
            no_wcs = QLabel("No WCS solution available — plate solve first.")
            no_wcs.setStyleSheet("color: #e06666;")
            no_wcs.setWordWrap(True)
            lay.addWidget(no_wcs)

        form = QFormLayout()
        self._include_dso = QCheckBox("Deep-sky objects (Messier/NGC/IC)")
        self._include_dso.setChecked(True)
        form.addRow(*self._r("", self._include_dso,
                    "Match the embedded Messier/NGC/IC deep-sky object "
                    "catalog against the field. Always available, no "
                    "network or extra download required."))

        self._gaia_available = self._check_gaia_available()
        self._include_stars = QCheckBox("Bright stars (local Gaia DR3 catalog)")
        if self._gaia_available:
            self._include_stars.setChecked(False)
        else:
            self._include_stars.setChecked(False)
            self._include_stars.setEnabled(False)
            self._include_stars.setToolTip(
                "<qt>No local Gaia catalog installed. Install a magnitude "
                "band from Tools &gt; GAIA Catalog Manager to identify "
                "bright stars.</qt>"
            )
        form.addRow(*self._r("", self._include_stars,
                    "Identify bright stars from a locally installed Gaia "
                    "DR3 catalog (no network access — the catalog must "
                    "already be downloaded)."))

        self._star_mag_limit = QDoubleSpinBox()
        self._star_mag_limit.setRange(-2.0, 20.0)
        self._star_mag_limit.setValue(9.0)
        self._star_mag_limit.setEnabled(self._include_stars.isChecked())
        self._include_stars.toggled.connect(self._star_mag_limit.setEnabled)
        form.addRow(*self._r("Star magnitude limit", self._star_mag_limit, param_help(
            "Faintest Gaia G magnitude to include when bright-star "
            "identification is enabled.",
            higher="Includes fainter stars too, so more get identified "
                   "(and labeled/plotted).",
            lower="Only the brightest stars are included.",
        )))

        self._max_stars = QSpinBox()
        self._max_stars.setRange(1, 2000)
        self._max_stars.setValue(50)
        self._max_stars.setEnabled(self._include_stars.isChecked())
        self._include_stars.toggled.connect(self._max_stars.setEnabled)
        form.addRow(*self._r("Max stars", self._max_stars,
                    "Cap on the number of identified stars (brightest "
                    "first)."))

        self._label_bright_stars = QCheckBox("Label bright stars on annotated image")
        self._label_bright_stars.setChecked(True)
        self._label_bright_stars.setEnabled(self._include_stars.isChecked())
        self._include_stars.toggled.connect(self._label_bright_stars.setEnabled)
        form.addRow(*self._r("", self._label_bright_stars,
                    "When annotating onto the image, draw name labels next "
                    "to identified bright stars (identified deep-sky "
                    "objects are always labeled)."))
        lay.addLayout(form)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(
            ["Name", "Catalog", "Type", "Mag", "Size (')", "X", "Y"]
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSortingEnabled(True)
        self._table.cellClicked.connect(self._on_row_clicked)
        lay.addWidget(self._table, 1)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        lay.addWidget(self._summary)

        btns = QHBoxLayout()
        self._identify_btn = QPushButton("Identify")
        self._identify_btn.setEnabled(self._wcs_header is not None)
        self._identify_btn.clicked.connect(self._identify)
        btns.addWidget(self._identify_btn)
        self._annotate_btn = QPushButton("Annotate onto Image")
        self._annotate_btn.setEnabled(False)
        self._annotate_btn.clicked.connect(self._annotate)
        btns.addWidget(self._annotate_btn)
        btns.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

    @staticmethod
    def _r(label, widget, tip):
        row = QHBoxLayout()
        row.addWidget(widget)
        row.addWidget(help_dot(tip))
        row.addStretch()
        return label, row

    @staticmethod
    def _check_gaia_available() -> bool:
        try:
            from astraios.core.gaia_catalog import installed_files
            return bool(installed_files())
        except Exception:
            log.debug("Image annotate: Gaia catalog availability check failed", exc_info=True)
            return False

    # -- Identify ----------------------------------------------------------

    def _identify(self):
        from astraios.core.image_annotate import AnnotateParams

        if self._wcs_header is None:
            self._status.setText("No WCS solution — plate solve first.")
            return

        params = AnnotateParams(
            include_dso=self._include_dso.isChecked(),
            include_bright_stars=self._include_stars.isChecked() and self._gaia_available,
            star_mag_limit=float(self._star_mag_limit.value()),
            max_stars=int(self._max_stars.value()),
        )

        self._identify_btn.setEnabled(False)
        self._annotate_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
        self._status.setText("Identifying objects...")
        self._identify_worker = _IdentifyWorker(self._image.shape, self._wcs_header, params)
        self._identify_worker.progress.connect(self._on_identify_progress)
        self._identify_worker.done.connect(self._on_identify_done)
        self._identify_worker.failed.connect(self._on_identify_fail)
        self._identify_worker.start()

    def _on_identify_progress(self, fraction: float, message: str):
        self._progress.setValue(int(fraction * 100))
        self._status.setText(message)

    def _on_identify_done(self, results):
        self._progress.setVisible(False)
        self._identify_btn.setEnabled(True)
        self._results = list(results)

        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(self._results))
        for row, obj in enumerate(self._results):
            self._table.setItem(row, self._COL_NAME, QTableWidgetItem(obj.name))
            self._table.setItem(row, self._COL_CATALOG, QTableWidgetItem(obj.catalog))
            self._table.setItem(row, self._COL_TYPE, QTableWidgetItem(obj.type))
            mag_text = f"{obj.magnitude:.2f}" if obj.magnitude is not None else "-"
            mag_sort = obj.magnitude if obj.magnitude is not None else float("inf")
            self._table.setItem(row, self._COL_MAG, _NumericItem(mag_sort, mag_text))
            size_text = f"{obj.size:.1f}" if obj.size is not None else "-"
            self._table.setItem(
                row, self._COL_SIZE,
                _NumericItem(obj.size if obj.size is not None else -1.0, size_text),
            )
            self._table.setItem(row, self._COL_X, _NumericItem(obj.x, f"{obj.x:.1f}"))
            self._table.setItem(row, self._COL_Y, _NumericItem(obj.y, f"{obj.y:.1f}"))
        self._table.setSortingEnabled(True)
        self._table.resizeColumnsToContents()

        n = len(self._results)
        self._summary.setText(f"{n} object{'s' if n != 1 else ''} identified.")
        self._status.setText("Done.")
        self._annotate_btn.setEnabled(n > 0)

    def _on_identify_fail(self, msg):
        self._progress.setVisible(False)
        self._identify_btn.setEnabled(True)
        self._status.setText(f"Failed: {msg}")

    def _on_row_clicked(self, row: int, _column: int):
        if 0 <= row < len(self._results):
            obj = self._results[row]
            self.marker_requested.emit(float(obj.x), float(obj.y), obj.name)

    # -- Annotate onto image ------------------------------------------------

    def _annotate(self):
        from astraios.core.finder_chart import FinderChartParams
        from astraios.core.image_annotate import split_for_finder_chart

        if self._wcs_header is None or not self._results:
            return

        catalog_stars, dso_list = split_for_finder_chart(self._results)
        params = FinderChartParams(
            show_stars=bool(catalog_stars),
            show_star_labels=self._label_bright_stars.isChecked(),
            show_dso=bool(dso_list),
            show_dso_labels=True,
        )

        self._annotate_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
        self._status.setText("Rendering annotated image...")
        self._render_worker = _RenderWorker(
            self._image, self._wcs_header, params, catalog_stars, dso_list
        )
        self._render_worker.progress.connect(self._on_identify_progress)
        self._render_worker.done.connect(self._on_render_done)
        self._render_worker.failed.connect(self._on_render_fail)
        self._render_worker.start()

    def _on_render_done(self, result):
        self._progress.setVisible(False)
        self._annotate_btn.setEnabled(True)
        if not isinstance(result, np.ndarray):
            self._status.setText("No result produced.")
            return
        self._status.setText("Annotated image rendered.")
        self.result_ready.emit(result)

    def _on_render_fail(self, msg):
        self._progress.setVisible(False)
        self._annotate_btn.setEnabled(True)
        self._status.setText(f"Failed: {msg}")
