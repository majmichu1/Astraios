"""Batch Rename Dialog — rename a set of files using a token template built
from FITS header values.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from astraios.core.batch_rename import (
    BatchRenameParams,
    batch_rename,
    find_collisions,
    plan_renames,
)
from astraios.ui.widgets.ui_kit import help_dot, param_help

log = logging.getLogger(__name__)

_DEFAULT_TEMPLATE = "LIGHT_{FILTER}_{EXPTIME:.0f}s_{DATE-OBS:%Y%m%d}_{#03}.{ext}"

_TOKEN_LEGEND = param_help(
    "Tokens usable in the filename template.",
    how=(
        "<b>{KEYWORD}</b> — any FITS header keyword, e.g. {OBJECT}, {FILTER} "
        "(missing keyword becomes an empty string).<br>"
        "<b>{KEYWORD:fmt}</b> — a numeric Python format spec, e.g. "
        "{EXPTIME:.0f}; or a date/time format for DATE-OBS/DATE "
        "({DATE-OBS:%Y%m%d}) and TIME-OBS/UTSTART/UTC-START "
        "({TIME-OBS:%H%M%S}).<br>"
        "<b>{#}</b> / <b>{#03}</b> — sequential counter, optionally "
        "zero-padded to a fixed width.<br>"
        "<b>{ext}</b> — the source file's extension, without the dot.<br>"
        "<b>Filters</b> — chained with ``|`` after a token body: "
        "<b>re:PATTERN</b> (regex search — first capture group, or the "
        "whole match), <b>upper</b>, <b>lower</b>, <b>slice:a:b</b>, "
        "<b>strip</b>. Example: {OBJECT|re:(\\w+)|upper}."
    ),
    default=f"e.g. {_DEFAULT_TEMPLATE}",
)


class BatchRenameDialog(QDialog):
    """Rename a chosen set of files using a FITS-header token template.

    The preview table updates live as the template/options change; nothing
    on disk is touched until "Apply" is pressed.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Rename")
        self.setMinimumSize(780, 560)

        self._paths: list[Path] = []
        self._planned: list[tuple[Path, Path]] = []

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Rename a set of files using a template built from their FITS "
            "header values. The preview updates live; nothing is renamed "
            "until you click Apply."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #8b949e;")
        layout.addWidget(intro)

        layout.addWidget(QLabel("Input files:"))
        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(100)
        layout.addWidget(self._file_list)

        file_row = QHBoxLayout()
        add_btn = QPushButton("Add Files...")
        add_btn.clicked.connect(self._add_files)
        file_row.addWidget(add_btn)
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_files)
        file_row.addWidget(clear_btn)
        file_row.addStretch(1)
        layout.addLayout(file_row)

        tmpl_row = QHBoxLayout()
        tmpl_row.addWidget(QLabel("Template:"))
        self._template_edit = QLineEdit(_DEFAULT_TEMPLATE)
        self._template_edit.textChanged.connect(self._refresh_preview)
        tmpl_row.addWidget(self._template_edit, 1)
        tmpl_row.addWidget(help_dot(_TOKEN_LEGEND))
        layout.addLayout(tmpl_row)

        opt_row = QHBoxLayout()
        self._lower_cb = QCheckBox("lowercase")
        self._lower_cb.toggled.connect(self._refresh_preview)
        opt_row.addWidget(self._lower_cb)

        self._slug_cb = QCheckBox("spaces→_ / strip unsafe chars")
        self._slug_cb.setChecked(True)
        self._slug_cb.toggled.connect(self._refresh_preview)
        opt_row.addWidget(self._slug_cb)

        self._keep_ext_cb = QCheckBox("append original extension if missing")
        self._keep_ext_cb.setChecked(True)
        self._keep_ext_cb.toggled.connect(self._refresh_preview)
        opt_row.addWidget(self._keep_ext_cb)

        opt_row.addSpacing(12)
        opt_row.addWidget(QLabel("Index start:"))
        self._index_start = QSpinBox()
        self._index_start.setRange(0, 999999)
        self._index_start.setValue(1)
        self._index_start.valueChanged.connect(self._refresh_preview)
        opt_row.addWidget(self._index_start)
        opt_row.addStretch(1)
        layout.addLayout(opt_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Destination (optional):"))
        self._dest_edit = QLineEdit()
        self._dest_edit.setPlaceholderText("Leave empty to rename in place")
        self._dest_edit.textChanged.connect(self._refresh_preview)
        out_row.addWidget(self._dest_edit, 1)
        dest_browse = QPushButton("Browse...")
        dest_browse.clicked.connect(self._browse_dest)
        out_row.addWidget(dest_browse)
        layout.addLayout(out_row)

        layout.addWidget(QLabel("Preview:"))
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Old name", "New name", "Status"])
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._table, 1)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._apply)
        btn_row.addWidget(self._apply_btn)
        btn_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    # ---------- file selection ----------

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select FITS Files", "",
            "FITS files (*.fit *.fits *.fts *.fz);;XISF files (*.xisf);;All files (*)",
        )
        for p in paths:
            path = Path(p)
            if path not in self._paths:
                self._paths.append(path)
                self._file_list.addItem(path.name)
        self._refresh_preview()

    def _clear_files(self):
        self._paths = []
        self._planned = []
        self._file_list.clear()
        self._table.setRowCount(0)
        self._status.setText("")

    def _browse_dest(self):
        start = self._dest_edit.text().strip() or ""
        d = QFileDialog.getExistingDirectory(self, "Choose Destination Folder", start)
        if d:
            self._dest_edit.setText(d)

    # ---------- params / preview ----------

    def _current_params(self) -> BatchRenameParams:
        dest = self._dest_edit.text().strip()
        return BatchRenameParams(
            lowercase=self._lower_cb.isChecked(),
            slugify=self._slug_cb.isChecked(),
            keep_ext=self._keep_ext_cb.isChecked(),
            index_start=self._index_start.value(),
            output_dir=dest or None,
        )

    def _refresh_preview(self):
        template = self._template_edit.text().strip()
        if not self._paths or not template:
            self._table.setRowCount(0)
            self._planned = []
            return

        params = self._current_params()
        try:
            self._planned = plan_renames(self._paths, template, params)
        except Exception as exc:
            self._status.setText(f"Template error: {exc}")
            self._table.setRowCount(0)
            self._planned = []
            return

        collisions = find_collisions(self._planned)
        collided_dsts = set(collisions.keys())

        self._table.setRowCount(len(self._planned))
        for row, (src, dst) in enumerate(self._planned):
            self._table.setItem(row, 0, QTableWidgetItem(src.name))
            self._table.setItem(row, 1, QTableWidgetItem(dst.name))
            if dst in collided_dsts:
                status = "name collision"
            elif dst != src and dst.exists():
                status = "will overwrite"
            else:
                status = "ok"
            status_item = QTableWidgetItem(status)
            if status != "ok":
                status_item.setForeground(Qt.GlobalColor.red)
            self._table.setItem(row, 2, status_item)

        if collisions:
            self._status.setText(
                f"{len(collisions)} name collision(s) detected — "
                "fix the template before applying."
            )
        else:
            self._status.setText(f"{len(self._planned)} file(s) ready to rename.")

    # ---------- apply ----------

    def _apply(self):
        template = self._template_edit.text().strip()
        if not self._paths:
            self._status.setText("Add at least one input file.")
            return
        if not template:
            self._status.setText("Enter a filename template.")
            return

        params = self._current_params()
        try:
            result = batch_rename(self._paths, template, params, dry_run=False)
        except ValueError as exc:
            QMessageBox.warning(self, "Batch Rename", str(exc))
            return
        except Exception as exc:
            log.exception("Batch rename failed")
            QMessageBox.critical(self, "Batch Rename", str(exc))
            return

        self._status.setText(f"Renamed {len(result)} file(s).")
        # Files have moved: track them under their new paths so the dialog
        # (and its preview) still reflects reality if the user renames again.
        self._paths = [dst for _src, dst in result]
        self._file_list.clear()
        for p in self._paths:
            self._file_list.addItem(p.name)
        self._refresh_preview()
