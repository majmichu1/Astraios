"""Batch Convert Dialog — convert a set of image files from one format to another.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from astraios.core.batch_convert import (
    ALLOWED_BIT_DEPTHS,
    INPUT_SUFFIXES,
    OUTPUT_FORMATS,
    BatchConvertParams,
    batch_convert,
)
from astraios.ui.widgets.ui_kit import help_dot, param_help

log = logging.getLogger(__name__)


class _ConvertWorker(QThread):
    """Runs batch_convert off the main thread."""

    progress = pyqtSignal(float, str)
    finished = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, paths: list[Path], output_dir: Path, params: BatchConvertParams):
        super().__init__()
        self._paths = paths
        self._output_dir = output_dir
        self._params = params
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            outputs = batch_convert(
                self._paths, self._output_dir, self._params, progress=self._emit_progress,
            )
            if not self._cancelled:
                self.finished.emit(outputs)
        except Exception as e:
            log.exception("Batch convert failed")
            self.error_occurred.emit(str(e))

    def _emit_progress(self, fraction: float, message: str):
        if self._cancelled:
            raise InterruptedError("Cancelled")
        self.progress.emit(fraction, message)


class BatchConvertDialog(QDialog):
    """Convert a chosen set of image files to another format and/or bit depth."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Convert")
        self.setMinimumSize(560, 480)

        self._paths: list[Path] = []
        self._worker: _ConvertWorker | None = None

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Convert a set of image files to another format (FITS, TIFF, PNG, "
            "JPEG, or XISF), optionally changing bit depth. RAW camera formats "
            "are not supported (Astraios has no RAW decoder)."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #8b949e;")
        layout.addWidget(intro)

        layout.addWidget(QLabel("Input files:"))
        self._file_list = QListWidget()
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

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output folder:"))
        self._out_edit = QLineEdit()
        out_row.addWidget(self._out_edit, 1)
        out_browse = QPushButton("Browse...")
        out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(out_browse)
        layout.addLayout(out_row)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Output format:"))
        self._fmt_combo = QComboBox()
        self._fmt_combo.addItems(list(OUTPUT_FORMATS))
        self._fmt_combo.currentIndexChanged.connect(self._refresh_depth_choices)
        fmt_row.addWidget(self._fmt_combo)
        fmt_row.addWidget(help_dot(param_help(
            "Destination file format for every converted file.",
            how="Written via Astraios's own image writer: FITS and XISF are "
                "always saved as 32-bit float; JPEG is always 8-bit.",
            default="TIFF is a safe, lossless default for further processing.",
        )))

        fmt_row.addSpacing(16)
        fmt_row.addWidget(QLabel("Bit depth:"))
        self._depth_combo = QComboBox()
        fmt_row.addWidget(self._depth_combo)
        fmt_row.addWidget(help_dot(param_help(
            "Output pixel bit depth, where the chosen format offers a choice.",
            how="TIFF and PNG can be written as 8-bit or 16-bit integer. "
                "FITS and XISF are always 32-bit float and JPEG is always "
                "8-bit, so no choice is offered for those.",
            higher="More precision retained (16-bit), larger file.",
            lower="Smaller file (8-bit), but risks banding on smooth "
                  "gradients (skies, faint nebulosity).",
            default="Auto picks 8-bit for PNG/JPEG and 16-bit for TIFF.",
        )))
        fmt_row.addStretch(1)
        layout.addLayout(fmt_row)

        opt_row = QHBoxLayout()
        self._skip_cb = QCheckBox("Skip existing")
        self._skip_cb.setChecked(True)
        opt_row.addWidget(self._skip_cb)
        opt_row.addWidget(help_dot(param_help(
            "If the destination file already exists, leave it untouched "
            "instead of overwriting it.",
            default="On: safe to re-run a batch after adding new files.",
        )))

        opt_row.addSpacing(16)
        opt_row.addWidget(QLabel("JPEG quality:"))
        self._quality_spin = QSpinBox()
        self._quality_spin.setRange(1, 100)
        self._quality_spin.setValue(95)
        opt_row.addWidget(self._quality_spin)
        opt_row.addWidget(help_dot(param_help(
            "JPEG compression quality (only used when the output format is JPEG).",
            higher="Less compression artifacting, larger file.",
            lower="Smaller file, more visible blocking/artifacts.",
            default="95 is visually lossless for most astrophotos.",
        )))
        opt_row.addStretch(1)
        layout.addLayout(opt_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)
        self._status = QLabel("Ready")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Convert")
        self._run_btn.clicked.connect(self._run)
        btn_row.addWidget(self._run_btn)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._refresh_depth_choices()

    # ---------- file / folder selection ----------

    def _add_files(self):
        patterns = " ".join(f"*{s}" for s in INPUT_SUFFIXES)
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Input Images", "", f"All Supported ({patterns});;All files (*)",
        )
        for p in paths:
            path = Path(p)
            if path not in self._paths:
                self._paths.append(path)
                self._file_list.addItem(path.name)

    def _clear_files(self):
        self._paths.clear()
        self._file_list.clear()

    def _browse_output(self):
        start = self._out_edit.text().strip() or ""
        d = QFileDialog.getExistingDirectory(self, "Choose Output Folder", start)
        if d:
            self._out_edit.setText(d)

    def _refresh_depth_choices(self):
        fmt = self._fmt_combo.currentText()
        self._depth_combo.blockSignals(True)
        self._depth_combo.clear()
        self._depth_combo.addItem("Auto", "auto")
        for depth in ALLOWED_BIT_DEPTHS.get(fmt, ()):
            self._depth_combo.addItem(f"{depth}-bit", depth)
        self._depth_combo.setCurrentIndex(0)
        self._depth_combo.blockSignals(False)

    # ---------- run / cancel ----------

    def _run(self):
        if not self._paths:
            self._status.setText("Add at least one input file.")
            return
        out_text = self._out_edit.text().strip()
        if not out_text:
            self._status.setText("Choose an output folder.")
            return

        params = BatchConvertParams(
            output_format=self._fmt_combo.currentText(),
            bit_depth=self._depth_combo.currentData(),
            jpeg_quality=self._quality_spin.value(),
            skip_existing=self._skip_cb.isChecked(),
        )

        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._status.setText("Converting...")
        self._progress.setValue(0)

        self._worker = _ConvertWorker(list(self._paths), Path(out_text), params)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(lambda: self._worker.deleteLater())
        self._worker.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
            self._status.setText("Cancelling...")
            self._cancel_btn.setEnabled(False)

    def _on_progress(self, fraction: float, message: str):
        self._progress.setValue(int(fraction * 100))
        self._status.setText(message)

    def _on_finished(self, outputs: list[Path]):
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._status.setText(f"Done: {len(outputs)}/{len(self._paths)} converted.")

    def _on_error(self, message: str):
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        if "Cancelled" in message:
            self._status.setText("Cancelled.")
        else:
            self._status.setText(f"Failed: {message}")
