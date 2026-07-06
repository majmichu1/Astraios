"""Minor Body (asteroid/comet) Catalog dialog.

Drives astraios.core.minor_body_catalog (query a local/downloadable SQLite
catalog of osculating orbital elements, propagate positions with a local
Kepler solver, and filter to a field/time), ported and adapted from Seti
Astro Suite Pro (GPL-3.0, Franklin Marek). See that module's docstring for
the exact data source and offline-behavior notes — this dialog never blocks
on the network by default (``auto_download`` is opt-in via a checkbox here).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QDateTime, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDateTimeEdit,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot

log = logging.getLogger(__name__)


def _field_center_from_header(wcs_header: Any) -> tuple[float, float] | None:
    """Best-effort (ra_deg, dec_deg) field center from a FITS-header-like dict.

    Reads ``CRVAL1``/``CRVAL2`` directly rather than building a full astropy
    WCS + inverting pixel coordinates — good enough for a pre-fill default
    the user can freely edit, and works even for the minimal synthetic
    headers ``main_window._finder_chart_wcs_header`` builds for an in-app
    plate solve (which has no ``NAXIS*`` keywords).
    """
    if not wcs_header:
        return None
    try:
        ra = wcs_header.get("CRVAL1") if hasattr(wcs_header, "get") else wcs_header["CRVAL1"]
        dec = wcs_header.get("CRVAL2") if hasattr(wcs_header, "get") else wcs_header["CRVAL2"]
        if ra is None or dec is None:
            return None
        return float(ra), float(dec)
    except Exception:
        return None


class _Worker(QThread):
    done = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, ra_deg, dec_deg, radius_deg, time, params):
        super().__init__()
        self._ra_deg = ra_deg
        self._dec_deg = dec_deg
        self._radius_deg = radius_deg
        self._time = time
        self._params = params

    def run(self):
        try:
            from astraios.core.minor_body_catalog import query_minor_bodies
            result = query_minor_bodies(
                self._ra_deg, self._dec_deg, self._radius_deg, self._time, self._params
            )
            self.done.emit(result)
        except Exception as exc:
            log.exception("Minor body query failed")
            self.failed.emit(str(exc))


class MinorBodyDialog(QDialog):
    """Query asteroid/comet positions overlaid on a field, at a given time."""

    def __init__(self, parent=None, wcs_header: Any = None):
        super().__init__(parent)
        self.setWindowTitle("Minor Body Catalog")
        self.setMinimumSize(640, 560)
        self._wcs_header = wcs_header
        self._worker: _Worker | None = None
        self.results: list = []  # exposed for scripted verification

        outer = QVBoxLayout(self)
        intro = QLabel(
            "Looks up asteroid/comet positions for a field and time from a "
            "local orbital-element catalog (downloaded once, then computed "
            "offline) — not a live Horizons/MPC query. See the Query button's "
            "status line if the catalog isn't installed yet."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        outer.addWidget(intro)

        form = QFormLayout()
        center = _field_center_from_header(wcs_header)

        self._ra = QDoubleSpinBox()
        self._ra.setRange(0.0, 360.0)
        self._ra.setDecimals(4)
        self._ra.setSuffix(" deg")
        self._ra.setValue(center[0] if center else 0.0)
        form.addRow(*self._r("Field center RA", self._ra,
                    "Right ascension of the field center, in degrees "
                    "(pre-filled from the current plate solve, if any)."))

        self._dec = QDoubleSpinBox()
        self._dec.setRange(-90.0, 90.0)
        self._dec.setDecimals(4)
        self._dec.setSuffix(" deg")
        self._dec.setValue(center[1] if center else 0.0)
        form.addRow(*self._r("Field center Dec", self._dec,
                    "Declination of the field center, in degrees."))

        self._radius = QDoubleSpinBox()
        self._radius.setRange(0.01, 45.0)
        self._radius.setDecimals(2)
        self._radius.setSuffix(" deg")
        self._radius.setValue(1.0)
        form.addRow(*self._r("Search radius", self._radius,
                    "Cone-search radius around the field center. Keep this "
                    "close to your imaging train's actual field of view — a "
                    "large radius means propagating (and filtering) many "
                    "more candidate bodies."))

        self._time = QDateTimeEdit(QDateTime.currentDateTimeUtc())
        self._time.setCalendarPopup(True)
        self._time.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self._time.setTimeSpec(Qt.TimeSpec.UTC)
        form.addRow(*self._r("Observation time (UTC)", self._time,
                    "Date/time the positions are computed for — ideally the "
                    "sub's DATE-OBS/mid-exposure time."))

        self._include_asteroids = QCheckBox("Asteroids")
        self._include_asteroids.setChecked(True)
        self._include_comets = QCheckBox("Comets")
        self._include_comets.setChecked(True)
        kinds_row = QHBoxLayout()
        kinds_row.addWidget(self._include_asteroids)
        kinds_row.addWidget(self._include_comets)
        kinds_row.addWidget(help_dot(
            "Which minor-body tables to search. Untick one to speed up a "
            "search or if you're only interested in the other kind."))
        kinds_row.addStretch()
        form.addRow("Include:", kinds_row)

        self._allow_download = QCheckBox("Allow downloading the catalog if not installed")
        self._allow_download.setChecked(False)
        form.addRow(self._allow_download)

        outer.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._query_btn = QPushButton("Query")
        self._query_btn.setDefault(True)
        self._query_btn.clicked.connect(self._query)
        btn_row.addWidget(self._query_btn)
        btn_row.addStretch()
        outer.addLayout(btn_row)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        columns = [
            "Designation", "Kind", "RA (deg)", "Dec (deg)", "Mag", "Motion (\"/hr)", "PA (deg)",
        ]
        self._table = QTableWidget(0, len(columns))
        self._table.setHorizontalHeaderLabels(columns)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        outer.addWidget(self._table, 1)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        close_row.addWidget(close_btn)
        outer.addLayout(close_row)

    @staticmethod
    def _r(label, widget, tip):
        row = QHBoxLayout()
        row.addWidget(widget)
        row.addWidget(help_dot(tip))
        row.addStretch()
        return label, row

    def _query(self) -> None:
        if not (self._include_asteroids.isChecked() or self._include_comets.isChecked()):
            self._status.setText("Enable at least one of Asteroids/Comets.")
            return

        from astraios.core.minor_body_catalog import DEFAULT_DB_BASENAME, MinorBodyQueryParams

        params = MinorBodyQueryParams(
            include_asteroids=self._include_asteroids.isChecked(),
            include_comets=self._include_comets.isChecked(),
            auto_download=self._allow_download.isChecked(),
        )
        self._catalog_precheck_missing = (
            not (Path(params.data_dir) / DEFAULT_DB_BASENAME).is_file()
            and not self._allow_download.isChecked()
        )

        qt_dt = self._time.dateTime().toUTC()
        iso_time = qt_dt.toString("yyyy-MM-ddTHH:mm:ss")

        self._query_btn.setEnabled(False)
        self._status.setText("Querying...")
        self._table.setRowCount(0)

        self._worker = _Worker(
            self._ra.value(), self._dec.value(), self._radius.value(), iso_time, params
        )
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, result: list) -> None:
        self._query_btn.setEnabled(True)
        self.results = result

        if not result:
            if getattr(self, "_catalog_precheck_missing", False):
                self._status.setText(
                    "No results — the minor body catalog isn't installed locally "
                    "and downloading is disabled (offline mode). Tick "
                    "'Allow downloading the catalog' and query again to fetch it, "
                    "or place a pre-downloaded copy in "
                    "~/.astraios/minor_bodies/ yourself."
                )
            else:
                self._status.setText(
                    "No minor bodies found in this field/time (or the network "
                    "was unavailable while trying to fetch the catalog)."
                )
            return

        self._status.setText(f"{len(result)} object(s) found.")
        self._table.setRowCount(len(result))
        for row, body in enumerate(result):
            values = [
                body.designation,
                body.kind,
                f"{body.ra_deg:.5f}",
                f"{body.dec_deg:.5f}",
                f"{body.magnitude:.2f}" if body.magnitude is not None else "-",
                f"{body.motion_arcsec_per_hour:.2f}"
                if body.motion_arcsec_per_hour is not None else "-",
                f"{body.motion_position_angle_deg:.1f}"
                if body.motion_position_angle_deg is not None else "-",
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignLeft if col in (0, 1)
                    else Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self._table.setItem(row, col, item)

    def _on_fail(self, msg: str) -> None:
        self._query_btn.setEnabled(True)
        self._status.setText(f"Query failed: {msg}")
