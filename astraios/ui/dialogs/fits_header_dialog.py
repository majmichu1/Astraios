"""FITS Header Editor — view and edit FITS header keywords."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

# Keywords that should never be edited (structural)
_READONLY_KEYS = frozenset({
    "SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3",
    "EXTEND", "END", "BZERO", "BSCALE",
})


class FITSHeaderDialog(QDialog):
    """Dialog for viewing and editing FITS header key/value/comment triplets."""

    def __init__(self, header: dict, file_path: str | None = None, parent=None):
        super().__init__(parent)
        self._header = dict(header)
        self._file_path = file_path

        title = f"FITS Header — {Path(file_path).name}" if file_path else "FITS Header"
        self.setWindowTitle(title)
        self.setMinimumSize(720, 500)
        self.resize(820, 580)

        layout = QVBoxLayout(self)

        # Info
        info = QLabel(f"{len(header)} keywords  •  grey rows are read-only structural keys")
        info.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(info)

        # Table
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Keyword", "Value", "Comment"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table, 1)

        self._populate(header)

        # Buttons
        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add Keyword")
        btn_add.clicked.connect(self._add_row)
        btn_row.addWidget(btn_add)

        btn_del = QPushButton("Delete Selected")
        btn_del.clicked.connect(self._delete_selected)
        btn_row.addWidget(btn_del)

        btn_row.addStretch()

        if file_path:
            btn_save = QPushButton("Save to FITS")
            btn_save.setToolTip("Write edited header back to the FITS file")
            btn_save.clicked.connect(self._save_to_file)
            btn_row.addWidget(btn_save)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        btn_row.addWidget(buttons)
        layout.addLayout(btn_row)

    def _populate(self, header: dict):
        self._table.setRowCount(0)
        for key, value in header.items():
            if key in ("", "COMMENT", "HISTORY"):
                continue
            self._add_table_row(key, str(value), "")

    def _add_table_row(self, key: str, value: str, comment: str):
        row = self._table.rowCount()
        self._table.insertRow(row)
        readonly = key.upper() in _READONLY_KEYS

        key_item = QTableWidgetItem(key)
        val_item = QTableWidgetItem(value)
        com_item = QTableWidgetItem(comment)

        if readonly:
            for item in (key_item, val_item, com_item):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setForeground(Qt.GlobalColor.gray)
            key_item.setBackground(Qt.GlobalColor.darkGray)

        self._table.setItem(row, 0, key_item)
        self._table.setItem(row, 1, val_item)
        self._table.setItem(row, 2, com_item)

    def _add_row(self):
        self._add_table_row("NEW_KEY", "", "")
        self._table.scrollToBottom()
        self._table.editItem(self._table.item(self._table.rowCount() - 1, 0))

    def _delete_selected(self):
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()}, reverse=True)
        for row in rows:
            key = self._table.item(row, 0).text().upper()
            if key in _READONLY_KEYS:
                continue
            self._table.removeRow(row)

    def get_header(self) -> dict:
        """Return the edited header as a plain dict."""
        result = {}
        for row in range(self._table.rowCount()):
            key = self._table.item(row, 0).text().strip()
            val = self._table.item(row, 1).text().strip()
            if key:
                # Try to preserve numeric types
                try:
                    result[key] = int(val)
                except ValueError:
                    try:
                        result[key] = float(val)
                    except ValueError:
                        result[key] = val
        return result

    def _save_to_file(self):
        if not self._file_path:
            return
        try:
            from astropy.io import fits
            new_header = self.get_header()
            with fits.open(self._file_path, mode="update") as hdul:
                for hdu in hdul:
                    if hdu.data is not None:
                        for key, val in new_header.items():
                            if key.upper() not in _READONLY_KEYS:
                                try:
                                    hdu.header[key] = val
                                except Exception:
                                    pass
                        break
            QMessageBox.information(self, "Saved", f"Header written to {Path(self._file_path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save: {e}")
