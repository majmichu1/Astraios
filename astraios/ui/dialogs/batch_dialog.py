"""Batch Processing Dialog — apply a pipeline to multiple images."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from astraios.core.batch import (
    BatchResult,
    Pipeline,
    batch_process,
)


class BatchWorker(QThread):
    """Runs batch processing off the main thread."""

    progress = pyqtSignal(float, str)
    finished = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, input_paths, pipeline, output_dir, output_format):
        super().__init__()
        self._input_paths = input_paths
        self._pipeline = pipeline
        self._output_dir = output_dir
        self._output_format = output_format
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self._cancelled:
                return
            result = batch_process(
                self._input_paths,
                self._pipeline,
                self._output_dir,
                self._output_format,
                progress=self._emit_progress,
            )
            if not self._cancelled:
                self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _emit_progress(self, fraction: float, message: str):
        if self._cancelled:
            raise InterruptedError("Cancelled")
        self.progress.emit(fraction, message)


class BatchDialog(QDialog):
    """Dialog for batch processing images with a pipeline."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Processing")
        self.setMinimumSize(550, 500)

        self._input_paths: list[Path] = []
        self._output_dir: Path | None = None
        self._worker: BatchWorker | None = None

        layout = QVBoxLayout(self)

        # Input files
        layout.addWidget(QLabel("Input Images:"))
        self._input_list = QListWidget()
        layout.addWidget(self._input_list)

        input_row = QHBoxLayout()
        add_btn = QPushButton("Add Files...")
        add_btn.clicked.connect(self._add_files)
        input_row.addWidget(add_btn)
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_files)
        input_row.addWidget(clear_btn)
        layout.addLayout(input_row)

        # Pipeline steps
        layout.addWidget(QLabel("Processing Steps:"))
        self._step_list = QListWidget()
        layout.addWidget(self._step_list)

        step_row = QHBoxLayout()
        self._tool_combo = QComboBox()
        self._tool_combo.addItems([
            "auto_stretch", "ghs", "background_extraction",
            "cosmetic_correction", "banding_reduction", "histogram_transform",
            "scnr", "color_adjust", "denoise", "local_contrast",
        ])
        step_row.addWidget(self._tool_combo)

        add_step_btn = QPushButton("Add Step")
        add_step_btn.clicked.connect(self._add_step)
        step_row.addWidget(add_step_btn)

        remove_step_btn = QPushButton("Remove")
        remove_step_btn.clicked.connect(self._remove_step)
        step_row.addWidget(remove_step_btn)
        layout.addLayout(step_row)

        # Output directory
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        self._output_label = QLabel("(not set)")
        self._output_label.setStyleSheet("color: #969696;")
        out_row.addWidget(self._output_label, 1)
        out_btn = QPushButton("Choose...")
        out_btn.clicked.connect(self._choose_output)
        out_row.addWidget(out_btn)
        layout.addLayout(out_row)

        # Format
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Format:"))
        self._format_combo = QComboBox()
        self._format_combo.addItems(["fits", "xisf", "tif", "png"])
        fmt_row.addWidget(self._format_combo)
        layout.addLayout(fmt_row)

        # Progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        layout.addWidget(self._progress_bar)

        self._status = QLabel("Ready")
        layout.addWidget(self._status)

        # Run button
        self._run_btn = QPushButton("Run Batch")
        self._run_btn.clicked.connect(self._run)
        layout.addWidget(self._run_btn)

        self._pipeline = Pipeline(name="Batch Pipeline")

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Input Images", "",
            "All Supported (*.fit *.fits *.fts *.xisf *.tif *.tiff *.png);;All (*)",
        )
        for p in paths:
            path = Path(p)
            if path not in self._input_paths:
                self._input_paths.append(path)
                self._input_list.addItem(path.name)

    def _clear_files(self):
        self._input_paths.clear()
        self._input_list.clear()

    def _add_step(self):
        tool = self._tool_combo.currentText()
        step = self._pipeline.add_step(tool)
        self._step_list.addItem(f"{len(self._pipeline.steps)}. {tool}")

    def _remove_step(self):
        idx = self._step_list.currentRow()
        if idx >= 0:
            self._pipeline.remove_step(idx)
            self._step_list.takeItem(idx)
            # Refresh numbering
            self._step_list.clear()
            for i, s in enumerate(self._pipeline.steps):
                self._step_list.addItem(f"{i + 1}. {s.tool_name}")

    def _choose_output(self):
        d = QFileDialog.getExistingDirectory(self, "Choose Output Directory")
        if d:
            self._output_dir = Path(d)
            self._output_label.setText(d)
            self._output_label.setStyleSheet("color: #e0e0e0;")

    def _run(self):
        if not self._input_paths:
            self._status.setText("No input files selected")
            return
        if self._output_dir is None:
            self._status.setText("Choose an output directory first")
            return
        if not self._pipeline.steps:
            self._status.setText("Add at least one processing step")
            return

        # Keep the button ENABLED — it now acts as the Cancel button.
        self._set_button_mode(run_mode=False)
        self._status.setText("Processing...")

        self._worker = BatchWorker(
            self._input_paths,
            self._pipeline,
            self._output_dir,
            self._format_combo.currentText(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(lambda: self._worker.deleteLater())
        self._worker.start()

    def _set_button_mode(self, run_mode: bool):
        """Switch the Run/Cancel button between its two states."""
        self._run_btn.setEnabled(True)
        self._run_btn.setText("Run Batch" if run_mode else "Cancel")
        try:
            self._run_btn.clicked.disconnect()
        except TypeError:
            pass
        self._run_btn.clicked.connect(self._run if run_mode else self._cancel)

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
            self._status.setText("Cancelling...")
            # Prevent double-cancel; _on_finished/_on_error restores the button.
            self._run_btn.setEnabled(False)
        else:
            self._set_button_mode(run_mode=True)

    def _on_error(self, message: str):
        self._set_button_mode(run_mode=True)
        self._progress_bar.setValue(0)
        if "Cancelled" in message:
            self._status.setText("Batch cancelled")
        else:
            self._status.setText(f"Error: {message}")

    def _on_progress(self, fraction: float, message: str):
        self._progress_bar.setValue(int(fraction * 100))
        self._status.setText(message)

    def _on_finished(self, result: BatchResult):
        self._set_button_mode(run_mode=True)
        self._progress_bar.setValue(100)
        self._status.setText(
            f"Done: {result.n_processed} processed, {result.n_failed} failed"
        )
        if result.errors:
            error_text = "\n".join(str(e) for e in result.errors[:5])
            self._status.setText(f"{self._status.text()}\n{error_text}")
