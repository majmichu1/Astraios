"""AstroBin Exporter dialog.

Drives astraios.core.astrobin_export (scan a set of light frames' FITS
headers, group by observing night x filter x exposure, and write the CSV
AstroBin's acquisition-details importer expects), ported from Seti Astro
Suite Pro (GPL-3.0, Franklin Marek).

This is a header-only, CPU-bound I/O operation — no GPU, no pixel data — so
it runs synchronously (no worker thread); even large frame sets are just a
few thousand small FITS header reads.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot

log = logging.getLogger(__name__)

_TABLE_COLUMNS = [
    "date", "filter", "number", "duration", "gain", "iso", "binning",
    "sensorCooling", "fNumber", "darks", "flats", "flatDarks", "bias",
    "bortle", "meanSqm", "meanFwhm", "temperature",
]


class AstroBinExportDialog(QDialog):
    """Preview and export an AstroBin acquisition-details CSV for a set of
    light frames."""

    def __init__(self, parent=None, frame_paths: list | None = None):
        super().__init__(parent)
        self.setWindowTitle("AstroBin Exporter")
        self.setMinimumWidth(880)
        self.setMinimumHeight(560)
        self._frame_paths = list(frame_paths) if frame_paths else []
        self._records: list[dict] = []
        self._rows: list[dict] = []
        self._filter_map_edits: dict[str, QLineEdit] = {}

        lay = QVBoxLayout(self)
        intro_row = QHBoxLayout()
        intro = QLabel(
            "Scans the FITS headers of the loaded light frames and groups "
            "them by observing night, filter, and exposure length into the "
            "CSV format AstroBin's acquisition-details importer expects."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        intro_row.addWidget(intro, 1)
        intro_row.addWidget(help_dot(
            "AstroBin's filter field wants a numeric equipment-database ID, "
            "not a name — map your filter names below (find IDs at "
            "app.astrobin.com/equipment/explorer/filter). Darks/flats/"
            "flat-darks/bias/Bortle/mean SQM/mean FWHM are read from FITS "
            "headers when present (DARK/FLAT/FLATDARK/BIAS/BORTLE/MEAN_SQM/"
            "MEAN_FWHM); the defaults below only fill in for frames whose "
            "header is missing that value. 'Group nights noon-to-noon' "
            "keeps a session that crosses local midnight from being split "
            "into two separate rows."
        ))
        lay.addLayout(intro_row)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        # --- global fallback defaults ---
        defaults_box = QGroupBox("Global defaults (used only if a frame's header lacks the value)")
        grid = QGridLayout(defaults_box)

        self._fnum_edit = QLineEdit()
        self._fnum_edit.setPlaceholderText("e.g. 4.0")
        grid.addWidget(QLabel("f/number"), 0, 0)
        grid.addWidget(self._fnum_edit, 0, 1)

        self._darks_edit = QLineEdit()
        grid.addWidget(QLabel("Darks (#)"), 0, 2)
        grid.addWidget(self._darks_edit, 0, 3)

        self._flats_edit = QLineEdit()
        grid.addWidget(QLabel("Flats (#)"), 0, 4)
        grid.addWidget(self._flats_edit, 0, 5)

        self._flatdarks_edit = QLineEdit()
        grid.addWidget(QLabel("Flat-darks (#)"), 1, 0)
        grid.addWidget(self._flatdarks_edit, 1, 1)

        self._bias_edit = QLineEdit()
        grid.addWidget(QLabel("Bias (#)"), 1, 2)
        grid.addWidget(self._bias_edit, 1, 3)

        self._bortle_edit = QLineEdit()
        self._bortle_edit.setPlaceholderText("0-9")
        grid.addWidget(QLabel("Bortle"), 1, 4)
        grid.addWidget(self._bortle_edit, 1, 5)

        self._sqm_edit = QLineEdit()
        self._sqm_edit.setPlaceholderText("e.g. 21.30")
        grid.addWidget(QLabel("Mean SQM"), 2, 0)
        grid.addWidget(self._sqm_edit, 2, 1)

        self._fwhm_edit = QLineEdit()
        self._fwhm_edit.setPlaceholderText("e.g. 2.10")
        grid.addWidget(QLabel("Mean FWHM"), 2, 2)
        grid.addWidget(self._fwhm_edit, 2, 3)

        self._noon_cb = QCheckBox("Group nights noon -> noon (local time)")
        self._noon_cb.setChecked(True)
        grid.addWidget(self._noon_cb, 2, 4, 1, 2)

        for edit in (
            self._fnum_edit, self._darks_edit, self._flats_edit, self._flatdarks_edit,
            self._bias_edit, self._bortle_edit, self._sqm_edit, self._fwhm_edit,
        ):
            edit.textChanged.connect(self._recompute)
        self._noon_cb.toggled.connect(self._recompute)

        lay.addWidget(defaults_box)

        # --- filter ID mapping ---
        self._filter_box = QGroupBox("Filter -> AstroBin ID mapping")
        self._filter_grid = QGridLayout(self._filter_box)
        lay.addWidget(self._filter_box)

        # --- preview table ---
        self._table = QTableWidget()
        self._table.setColumnCount(len(_TABLE_COLUMNS))
        self._table.setHorizontalHeaderLabels(_TABLE_COLUMNS)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        lay.addWidget(self._table, 1)

        # --- actions ---
        act_row = QHBoxLayout()
        act_row.addStretch(1)
        self._save_btn = QPushButton("Save CSV...")
        self._save_btn.clicked.connect(self._save_csv)
        act_row.addWidget(self._save_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        act_row.addWidget(close_btn)
        lay.addLayout(act_row)

        self._load_headers()

    # ---------- header loading / recompute ----------
    def _load_headers(self):
        if not self._frame_paths:
            self._status.setText("No light frames provided.")
            self._save_btn.setEnabled(False)
            return

        from astraios.core.astrobin_export import read_frame_headers

        self._records = read_frame_headers(self._frame_paths)
        skipped = len(self._frame_paths) - len(self._records)
        msg = f"Loaded {len(self._records)} of {len(self._frame_paths)} frame(s)."
        if skipped:
            msg += f" ({skipped} could not be read.)"
        self._status.setText(msg)
        self._save_btn.setEnabled(bool(self._records))

        self._build_filter_map_editors()
        self._recompute()

    def _build_filter_map_editors(self):
        names = sorted({r.get("FILTER", "Unknown") for r in self._records if r.get("FILTER")})
        self._filter_map_edits.clear()
        # clear existing rows
        while self._filter_grid.count():
            item = self._filter_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for row, name in enumerate(names):
            self._filter_grid.addWidget(QLabel(name), row, 0)
            edit = QLineEdit()
            edit.setPlaceholderText("AstroBin numeric ID")
            edit.textChanged.connect(self._recompute)
            self._filter_grid.addWidget(edit, row, 1)
            self._filter_map_edits[name] = edit
        if not names:
            self._filter_grid.addWidget(QLabel("(no filters found)"), 0, 0)

    def _params(self):
        from astraios.core.astrobin_export import AstroBinExportParams

        filter_map = {
            name: edit.text().strip()
            for name, edit in self._filter_map_edits.items()
            if edit.text().strip().isdigit()
        }
        return AstroBinExportParams(
            fnumber=self._fnum_edit.text().strip() or "0",
            darks=self._darks_edit.text().strip(),
            flats=self._flats_edit.text().strip(),
            flat_darks=self._flatdarks_edit.text().strip(),
            bias=self._bias_edit.text().strip(),
            bortle=self._bortle_edit.text().strip(),
            mean_sqm=self._sqm_edit.text().strip(),
            mean_fwhm=self._fwhm_edit.text().strip(),
            noon_to_noon=self._noon_cb.isChecked(),
            filter_map=filter_map,
        )

    def _recompute(self):
        from astraios.core.astrobin_export import aggregate_astrobin_rows

        if not self._records:
            self._rows = []
        else:
            self._rows = aggregate_astrobin_rows(self._records, self._params())
        self._refresh_table()

    def _refresh_table(self):
        self._table.setRowCount(len(self._rows))
        for r, row in enumerate(self._rows):
            for c, key in enumerate(_TABLE_COLUMNS):
                item = QTableWidgetItem(str(row.get(key, "")))
                if key == "filter" and not str(row.get(key, "")).isdigit():
                    item.setForeground(Qt.GlobalColor.red)
                self._table.setItem(r, c, item)
        self._table.resizeColumnsToContents()

    # ---------- export ----------
    def _save_csv(self):
        from astraios.core.astrobin_export import export_astrobin_csv

        path, _ = QFileDialog.getSaveFileName(
            self, "Save AstroBin Acquisition CSV", "astrobin_acquisition.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return
        try:
            out = export_astrobin_csv(self._frame_paths, path, self._params())
        except Exception as exc:
            log.exception("AstroBin CSV export failed")
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QMessageBox.information(self, "Exported", f"Wrote:\n{out}")
