"""GAIA Catalog Manager dialog — download/inspect local Gaia DR3 catalog bands.

Drives astraios.core.gaia_catalog for the offline plate solver
(astraios.core.gaia_solver / astraios.core.plate_solve's 'gaia' backend).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot

log = logging.getLogger(__name__)

_SETTINGS_KEY = "gaia/catalog_dir"


class _CatalogDownloadCancelled(Exception):
    """Raised from the progress callback to unwind download_file() cleanly."""


def _human_size(num_bytes: float) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


class _BandDownloadWorker(QThread):
    """Downloads one Gaia catalog band off the GUI thread.

    Cancellation works by raising ``_CatalogDownloadCancelled`` from inside
    the progress callback (invoked synchronously in this thread by
    ``download_file``'s chunk loop) — the core module is never modified;
    ``download_file`` already unwinds cleanly on any exception raised from
    its ``progress`` callback (removes the partial ``.tmp`` file, re-raises).
    """

    progress = pyqtSignal(int, int, str)  # bytes_done, bytes_total, filename
    finished_ok = pyqtSignal(str)  # band_key
    failed = pyqtSignal(str, str)  # band_key, message
    cancelled = pyqtSignal(str)  # band_key

    def __init__(self, band_key: str, catalog_dir: Path):
        super().__init__()
        self._band_key = band_key
        self._catalog_dir = catalog_dir
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    def _on_progress(self, done: int, total: int, filename: str) -> None:
        if self._cancel_event.is_set():
            raise _CatalogDownloadCancelled()
        self.progress.emit(done, total, filename)

    def run(self) -> None:
        try:
            from astraios.core.gaia_catalog import download_band

            download_band(self._band_key, self._catalog_dir, progress=self._on_progress)
            self.finished_ok.emit(self._band_key)
        except _CatalogDownloadCancelled:
            self.cancelled.emit(self._band_key)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not a crash
            log.exception("Gaia catalog download failed for band %s", self._band_key)
            self.failed.emit(self._band_key, str(exc))


class GaiaCatalogDialog(QDialog):
    """Inspect and download local Gaia DR3 catalog bands for offline plate solving."""

    _COL_LABEL = 0
    _COL_INFO = 1
    _COL_STATUS = 2
    _COL_ACTION = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GAIA Catalog Manager")
        self.setMinimumSize(720, 460)
        self._settings = QSettings("Astraios", "Astraios")
        self._worker: _BandDownloadWorker | None = None
        self._downloading_row: int | None = None

        from astraios.core.gaia_catalog import GAIA_BANDS
        self._bands = GAIA_BANDS

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Downloads local Gaia DR3 magnitude bands so plate solving works "
            "fully offline, once installed — no astrometry.net upload, no "
            "internet needed at solve time. These are the same "
            "gaia_xp_*.sqlite files Seti Astro Suite Pro uses: if you already "
            "have an SASpro Gaia folder, point this at it instead of "
            "re-downloading."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Catalog folder:"))
        self._dir_edit = QLineEdit()
        self._dir_edit.setReadOnly(True)
        dir_row.addWidget(self._dir_edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(browse_btn)
        dir_row.addWidget(help_dot(
            "Where downloaded Gaia catalog files (gaia_xp_*.sqlite) live. "
            "Defaults to ~/.astraios/gaia. Reuse an existing Seti Astro "
            "Suite Pro Gaia folder here to skip re-downloading."
        ))
        lay.addLayout(dir_row)

        self._table = QTableWidget(len(self._bands), 4)
        self._table.setHorizontalHeaderLabels(["Band", "Size / Coverage", "Status", ""])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(self._COL_LABEL, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(self._COL_INFO, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(self._COL_STATUS, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(self._COL_ACTION, QHeaderView.ResizeMode.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        lay.addWidget(self._table, 1)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        btns = QHBoxLayout()
        self._cancel_btn = QPushButton("Cancel Download")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_download)
        btns.addWidget(self._cancel_btn)
        btns.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

        self._load_dir_from_settings()
        self._populate_table()

    # ── catalog directory ──────────────────────────────────────────────

    def _load_dir_from_settings(self) -> None:
        from astraios.core.gaia_catalog import default_gaia_dir

        raw = self._settings.value(_SETTINGS_KEY, "")
        raw = raw.strip() if isinstance(raw, str) else ""
        self._catalog_dir = Path(raw) if raw else default_gaia_dir()
        self._dir_edit.setText(str(self._catalog_dir))

    def catalog_dir(self) -> Path:
        return self._catalog_dir

    def _browse_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select Gaia Catalog Folder", str(self._catalog_dir)
        )
        if not folder:
            return
        self._catalog_dir = Path(folder)
        self._dir_edit.setText(str(self._catalog_dir))
        self._settings.setValue(_SETTINGS_KEY, str(self._catalog_dir))
        self._populate_table()

    # ── table ────────────────────────────────────────────────────────

    def _populate_table(self) -> None:
        from astraios.core.gaia_catalog import band_status

        for row, band in enumerate(self._bands):
            self._table.setItem(row, self._COL_LABEL, QTableWidgetItem(band.label))
            info_item = QTableWidgetItem(
                f"~{_human_size(band.est_size_mb * 1024 * 1024)}, {band.est_stars}\n"
                f"{band.description}"
            )
            info_item.setToolTip(band.description)
            self._table.setItem(row, self._COL_INFO, info_item)

            installed, missing = band_status(band, self._catalog_dir)
            if not missing:
                on_disk = sum(
                    (self._catalog_dir / f).stat().st_size
                    for f in installed
                    if (self._catalog_dir / f).exists()
                )
                status_text = f"Installed ({_human_size(on_disk)})"
            elif installed:
                on_disk = sum(
                    (self._catalog_dir / f).stat().st_size
                    for f in installed
                    if (self._catalog_dir / f).exists()
                )
                status_text = (
                    f"Partial: {len(installed)}/{len(band.filenames)} files "
                    f"({_human_size(on_disk)})"
                )
            else:
                status_text = "Not installed"
            status_item = QTableWidgetItem(status_text)
            self._table.setItem(row, self._COL_STATUS, status_item)

            btn = QPushButton("Re-download" if not missing else (
                "Download" if not installed else "Resume"
            ))
            btn.clicked.connect(lambda _checked, r=row: self._start_download(r))
            self._table.setCellWidget(row, self._COL_ACTION, btn)

        self._table.resizeRowsToContents()

    def _row_buttons(self):
        return [self._table.cellWidget(r, self._COL_ACTION) for r in range(self._table.rowCount())]

    # ── download ─────────────────────────────────────────────────────

    def _start_download(self, row: int) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._status.setText("A download is already in progress.")
            return

        band = self._bands[row]
        self._downloading_row = row
        for btn in self._row_buttons():
            if btn is not None:
                btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText(f"Downloading {band.label}...")

        self._catalog_dir.mkdir(parents=True, exist_ok=True)
        self._worker = _BandDownloadWorker(band.key, self._catalog_dir)
        queued = Qt.ConnectionType.QueuedConnection
        self._worker.progress.connect(self._on_download_progress, queued)
        self._worker.finished_ok.connect(self._on_download_done, queued)
        self._worker.failed.connect(self._on_download_failed, queued)
        self._worker.cancelled.connect(self._on_download_cancelled, queued)
        self._worker.start()

    def _cancel_download(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_cancel()
            self._status.setText("Cancelling...")

    def _on_download_progress(self, done: int, total: int, filename: str) -> None:
        if total > 0:
            self._progress.setValue(int(done / total * 100))
        self._status.setText(
            f"Downloading {filename}: {_human_size(done)}"
            + (f" / {_human_size(total)}" if total else "")
        )

    def _finish_download_ui(self) -> None:
        for btn in self._row_buttons():
            if btn is not None:
                btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._progress.setVisible(False)
        self._downloading_row = None
        self._worker = None

    def _on_download_done(self, band_key: str) -> None:
        self._status.setText(f"Download complete: {band_key}")
        self._finish_download_ui()
        self._populate_table()

    def _on_download_failed(self, band_key: str, message: str) -> None:
        self._status.setText(f"Download failed ({band_key}): {message}")
        self._finish_download_ui()
        self._populate_table()

    def _on_download_cancelled(self, band_key: str) -> None:
        self._status.setText(f"Download cancelled: {band_key}")
        self._finish_download_ui()
        self._populate_table()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_cancel()
            self._worker.wait(3000)
        super().closeEvent(event)
