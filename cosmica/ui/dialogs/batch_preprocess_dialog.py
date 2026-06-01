"""Batch Preprocessing Dialog — wizard for calibration → registration → stacking."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from cosmica.core.preprocessing import (
    CalibrationGroup,
    PreprocessingResult,
    run_preprocessing,
    scan_folder_for_frames,
)
from cosmica.core.stacking import RejectionMethod, StackingParams


def _combo_ss() -> str:
    return (
        "QComboBox { background: #161b22; color: #e0e0e0; border: 1px solid #30363d;"
        " border-radius: 4px; padding: 4px 8px; min-height: 24px; }"
        " QComboBox:hover { border-color: #58a6ff; }"
        " QComboBox::drop-down { border: none; padding-right: 8px; }"
        " QComboBox QAbstractItemView { background: #161b22; color: #e0e0e0;"
        " border: 1px solid #30363d; selection-background-color: #1f6feb; }"
    )


class _PreprocessWorker(QThread):
    progress = pyqtSignal(float, str)
    finished = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, light_paths, bias_paths, dark_paths, flat_paths,
                 output_dir, calibrate, register, stack, cosmetic,
                 stacking_params):
        super().__init__()
        self._light_paths = light_paths
        self._bias_paths = bias_paths
        self._dark_paths = dark_paths
        self._flat_paths = flat_paths
        self._output_dir = output_dir
        self._calibrate = calibrate
        self._register = register
        self._stack = stack
        self._cosmetic = cosmetic
        self._stacking_params = stacking_params
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self._cancelled:
                return
            result = run_preprocessing(
                self._light_paths,
                bias_paths=self._bias_paths,
                dark_paths=self._dark_paths,
                flat_paths=self._flat_paths,
                output_dir=self._output_dir,
                calibrate=self._calibrate,
                register=self._register,
                stack=self._stack,
                cosmetic=self._cosmetic,
                stacking_params=self._stacking_params,
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


class BatchPreprocessDialog(QDialog):
    """Wizard-style dialog for batch preprocessing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Preprocessing")
        self.setMinimumSize(720, 580)
        self.resize(780, 620)

        self._group = CalibrationGroup()
        self._output_dir: Path | None = None
        self._worker: _PreprocessWorker | None = None

        layout = QVBoxLayout(self)

        # ── Page stack ──────────────────────────────────────────────────
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_page1())  # 0: folder selection
        self._stack.addWidget(self._build_page2())  # 1: settings
        self._stack.addWidget(self._build_page3())  # 2: progress
        self._stack.addWidget(self._build_page4())  # 3: results
        layout.addWidget(self._stack)

        # ── Navigation buttons ──────────────────────────────────────────
        nav = QHBoxLayout()
        self._back_btn = QPushButton("← Back")
        self._back_btn.clicked.connect(self._go_back)
        self._back_btn.setVisible(False)
        nav.addWidget(self._back_btn)

        nav.addStretch()

        self._next_btn = QPushButton("Next →")
        self._next_btn.clicked.connect(self._go_next)
        nav.addWidget(self._next_btn)

        self._run_btn = QPushButton("Run Preprocessing")
        self._run_btn.clicked.connect(self._start_preprocessing)
        self._run_btn.setVisible(False)
        self._run_btn.setStyleSheet(
            "QPushButton { background: #2ea043; color: #fff; font-weight: bold;"
            " border: none; border-radius: 6px; padding: 8px 24px; }"
            " QPushButton:hover { background: #3fb950; }"
            " QPushButton:disabled { background: #21262d; color: #484f58; }"
        )
        nav.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._cancel_preprocessing)
        self._cancel_btn.setVisible(False)
        nav.addWidget(self._cancel_btn)

        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.accept)
        self._close_btn.setVisible(False)
        nav.addWidget(self._close_btn)

        layout.addLayout(nav)

        self._current_page = 0

    # ── Page builders ─────────────────────────────────────────────────────

    def _build_page1(self) -> QWidget:
        """Page 1: Folder selection with auto-detection."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        title = QLabel("Select Project Folder")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #e0e0e0;")
        layout.addWidget(title)

        desc = QLabel(
            "Choose a folder containing subdirectories named <b>lights</b>, "
            "<b>darks</b>, <b>flats</b>, <b>biases</b>.<br>"
            "FITS headers (IMAGETYP) are also used for auto-detection."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #8b949e; margin-bottom: 8px;")
        layout.addWidget(desc)

        # Folder picker row
        folder_row = QHBoxLayout()
        self._folder_path = QLineEdit()
        self._folder_path.setPlaceholderText("Choose a folder with calibration subfolders...")
        self._folder_path.setReadOnly(True)
        self._folder_path.setStyleSheet(
            "QLineEdit { background: #161b22; color: #c9d1d9; border: 1px solid #30363d;"
            " border-radius: 4px; padding: 6px 8px; }"
        )
        folder_row.addWidget(self._folder_path)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(browse_btn)
        layout.addLayout(folder_row)

        # Or manual file selection
        manual_group = QGroupBox("Or select files manually")
        manual_group.setStyleSheet(
            "QGroupBox { color: #8b949e; border: 1px solid #30363d; border-radius: 6px;"
            " margin-top: 12px; padding-top: 16px; }"
            " QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }"
        )
        manual_layout = QVBoxLayout(manual_group)

        def _add_file_row(label: str) -> tuple[QLineEdit, QPushButton, QLabel]:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(60)
            row.addWidget(lbl)
            le = QLineEdit()
            le.setReadOnly(True)
            le.setStyleSheet(
                "QLineEdit { background: #161b22; color: #8b949e; border: 1px solid #30363d;"
                " border-radius: 4px; padding: 4px 8px; }"
            )
            row.addWidget(le)
            count_label = QLabel("0 files")
            count_label.setStyleSheet("color: #8b949e; min-width: 60px;")
            row.addWidget(count_label)
            btn = QPushButton("Add...")
            row.addWidget(btn)
            manual_layout.addLayout(row)
            return le, btn, count_label

        self._lights_le, self._lights_btn, self._lights_count = _add_file_row("Lights")
        self._darks_le, self._darks_btn, self._darks_count = _add_file_row("Darks")
        self._flats_le, self._flats_btn, self._flats_count = _add_file_row("Flats")
        self._biases_le, self._biases_btn, self._biases_count = _add_file_row("Biases")

        self._lights_btn.clicked.connect(lambda: self._add_files("lights"))
        self._darks_btn.clicked.connect(lambda: self._add_files("darks"))
        self._flats_btn.clicked.connect(lambda: self._add_files("flats"))
        self._biases_btn.clicked.connect(lambda: self._add_files("biases"))

        layout.addWidget(manual_group)
        layout.addStretch()

        # What was detected
        self._detection_summary = QLabel("")
        self._detection_summary.setWordWrap(True)
        self._detection_summary.setStyleSheet("color: #8b949e; padding: 4px;")
        layout.addWidget(self._detection_summary)

        return page

    def _build_page2(self) -> QWidget:
        """Page 2: Processing settings."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        title = QLabel("Processing Settings")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #e0e0e0;")
        layout.addWidget(title)

        # Pipeline stages
        stage_group = QGroupBox("Pipeline Stages")
        stage_group.setStyleSheet(
            "QGroupBox { color: #e0e0e0; border: 1px solid #30363d; border-radius: 6px;"
            " margin-top: 12px; padding-top: 20px; }"
            " QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }"
        )
        stage_layout = QVBoxLayout(stage_group)
        self._cb_calibrate = QCheckBox("Calibrate frames (bias → dark → flat)")
        self._cb_calibrate.setChecked(True)
        stage_layout.addWidget(self._cb_calibrate)
        self._cb_cosmetic = QCheckBox("Cosmetic correction (hot/cold pixels)")
        self._cb_cosmetic.setChecked(True)
        stage_layout.addWidget(self._cb_cosmetic)
        self._cb_register = QCheckBox("Register (align) frames")
        self._cb_register.setChecked(True)
        stage_layout.addWidget(self._cb_register)
        self._cb_stack = QCheckBox("Stack frames")
        self._cb_stack.setChecked(True)
        stage_layout.addWidget(self._cb_stack)
        layout.addWidget(stage_group)

        # Stacking settings
        stack_group = QGroupBox("Stacking")
        stack_group.setStyleSheet(stage_group.styleSheet())
        stack_form = QFormLayout(stack_group)

        self._rejection_combo = QComboBox()
        for method in RejectionMethod:
            self._rejection_combo.addItem(method.name.replace("_", " ").title(), method)
        idx = self._rejection_combo.findData(RejectionMethod.SIGMA_CLIP)
        if idx >= 0:
            self._rejection_combo.setCurrentIndex(idx)
        self._rejection_combo.setStyleSheet(_combo_ss())
        stack_form.addRow("Rejection:", self._rejection_combo)

        kappa_row = QHBoxLayout()
        self._kappa_low = QSpinBox()
        self._kappa_low.setRange(1, 20)
        self._kappa_low.setValue(3)
        kappa_row.addWidget(QLabel("Low"))
        kappa_row.addWidget(self._kappa_low)
        self._kappa_high = QSpinBox()
        self._kappa_high.setRange(1, 20)
        self._kappa_high.setValue(3)
        kappa_row.addWidget(QLabel("High"))
        kappa_row.addWidget(self._kappa_high)
        stack_form.addRow("Kappa (σ):", kappa_row)

        self._use_gpu = QCheckBox("Use GPU acceleration")
        self._use_gpu.setChecked(True)
        stack_form.addRow("", self._use_gpu)

        layout.addWidget(stack_group)

        # Output
        out_group = QGroupBox("Output")
        out_group.setStyleSheet(stage_group.styleSheet())
        out_form = QFormLayout(out_group)

        out_dir_row = QHBoxLayout()
        self._out_dir_le = QLineEdit()
        self._out_dir_le.setPlaceholderText("Output directory...")
        self._out_dir_le.setStyleSheet(
            "QLineEdit { background: #161b22; color: #c9d1d9; border: 1px solid #30363d;"
            " border-radius: 4px; padding: 4px 8px; }"
        )
        out_dir_row.addWidget(self._out_dir_le)
        out_browse = QPushButton("Browse...")
        out_browse.clicked.connect(self._browse_output)
        out_dir_row.addWidget(out_browse)
        out_form.addRow("Directory:", out_dir_row)

        self._save_calibrated = QCheckBox("Save calibrated frames to subfolder")
        self._save_calibrated.setChecked(True)
        out_form.addRow("", self._save_calibrated)

        layout.addWidget(out_group)
        layout.addStretch()

        return page

    def _build_page3(self) -> QWidget:
        """Page 3: Progress display."""
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Processing...")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #e0e0e0;")
        layout.addWidget(title)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setStyleSheet(
            "QProgressBar { background: #21262d; border: 1px solid #30363d;"
            " border-radius: 4px; text-align: center; color: #c9d1d9; }"
            " QProgressBar::chunk { background: #2ea043; border-radius: 4px; }"
        )
        layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("Initializing...")
        self._progress_label.setStyleSheet("color: #8b949e;")
        layout.addWidget(self._progress_label)

        self._log_output = QTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setStyleSheet(
            "QTextEdit { background: #0d1117; color: #c9d1d9; border: 1px solid #30363d;"
            " border-radius: 4px; font-family: monospace; font-size: 11px; }"
        )
        self._log_output.setMinimumHeight(200)
        layout.addWidget(self._log_output)

        return page

    def _build_page4(self) -> QWidget:
        """Page 4: Results summary."""
        page = QWidget()
        layout = QVBoxLayout(page)

        title = QLabel("Preprocessing Complete")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #e0e0e0;")
        layout.addWidget(title)

        self._result_summary = QTextEdit()
        self._result_summary.setReadOnly(True)
        self._result_summary.setStyleSheet(
            "QTextEdit { background: #0d1117; color: #c9d1d9; border: 1px solid #30363d;"
            " border-radius: 4px; font-family: monospace; font-size: 12px; }"
        )
        layout.addWidget(self._result_summary)

        return page

    # ── Navigation ────────────────────────────────────────────────────────

    def _go_next(self):
        if self._current_page == 0 and not self._validate_page1():
            return
        self._current_page += 1
        self._stack.setCurrentIndex(self._current_page)
        self._update_nav()

    def _go_back(self):
        self._current_page -= 1
        self._stack.setCurrentIndex(self._current_page)
        self._update_nav()

    def _update_nav(self):
        self._back_btn.setVisible(self._current_page > 0 and self._current_page < 3)
        self._next_btn.setVisible(self._current_page == 0)
        self._run_btn.setVisible(self._current_page == 1)
        self._cancel_btn.setVisible(self._current_page == 2)
        self._close_btn.setVisible(self._current_page == 3)

    def _validate_page1(self) -> bool:
        if not self._group.lights:
            self._detection_summary.setStyleSheet("color: #f85149; padding: 4px;")
            self._detection_summary.setText(
                "No light frames found. Select a folder or add files manually."
            )
            return False
        return True

    # ── Folder / file selection ───────────────────────────────────────────

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if not folder:
            return
        folder_path = Path(folder)
        self._folder_path.setText(str(folder_path))

        self._group = scan_folder_for_frames(folder_path)
        self._update_counts()

        summary_parts = []
        if self._group.lights:
            summary_parts.append(f"• <b>{len(self._group.lights)} lights</b>")
        if self._group.darks:
            summary_parts.append(f"• <b>{len(self._group.darks)} darks</b>")
        if self._group.flats:
            summary_parts.append(f"• <b>{len(self._group.flats)} flats</b>")
        if self._group.biases:
            summary_parts.append(f"• <b>{len(self._group.biases)} biases</b>")

        if summary_parts:
            self._detection_summary.setStyleSheet("color: #2ea043; padding: 4px;")
            self._detection_summary.setText("Detected: " + ", ".join(summary_parts))
        else:
            self._detection_summary.setStyleSheet("color: #d29922; padding: 4px;")
            self._detection_summary.setText(
                "No frames detected in subfolders. Use manual file selection below."
            )

        # Set default output to project folder
        if self._out_dir_le.text() == "":
            self._out_dir_le.setText(str(folder_path / "preprocessed"))

    def _add_files(self, target: str):
        paths, _ = QFileDialog.getOpenFileNames(
            self, f"Select {target.title()} Files", "",
            "All Supported (*.fit *.fits *.fts *.xisf *.tif *.tiff *.png);;All (*)",
        )
        if not paths:
            return
        path_list = [Path(p) for p in paths]

        attr_map = {
            "lights": ("lights", self._lights_le, self._lights_count),
            "darks": ("darks", self._darks_le, self._darks_count),
            "flats": ("flats", self._flats_le, self._flats_count),
            "biases": ("biases", self._biases_le, self._biases_count),
        }
        attr, le, count_label = attr_map[target]

        existing = getattr(self._group, attr)
        for p in path_list:
            if p not in existing:
                existing.append(p)

        setattr(self._group, attr, existing)
        files_str = ", ".join(p.name for p in existing[:3])
        if len(existing) > 3:
            files_str += f"... (+{len(existing) - 3} more)"
        le.setText(files_str if existing else "")
        count_label.setText(f"{len(existing)} files")

    def _update_counts(self):
        self._lights_count.setText(f"{len(self._group.lights)} files")
        self._darks_count.setText(f"{len(self._group.darks)} files")
        self._flats_count.setText(f"{len(self._group.flats)} files")
        self._biases_count.setText(f"{len(self._group.biases)} files")

        for attr, le in [
            ("lights", self._lights_le),
            ("darks", self._darks_le),
            ("flats", self._flats_le),
            ("biases", self._biases_le),
        ]:
            paths = getattr(self._group, attr)
            if paths:
                files_str = ", ".join(p.name for p in paths[:3])
                if len(paths) > 3:
                    files_str += f"... (+{len(paths) - 3} more)"
                le.setText(files_str)
            else:
                le.setText("")

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Choose Output Directory")
        if d:
            self._output_dir = Path(d)
            self._out_dir_le.setText(d)

    # ── Run / cancel ──────────────────────────────────────────────────────

    def _start_preprocessing(self):
        out_dir_text = self._out_dir_le.text().strip()
        if out_dir_text:
            self._output_dir = Path(out_dir_text)

        stacking_params = StackingParams(
            rejection=self._rejection_combo.currentData(),
            kappa_low=self._kappa_low.value(),
            kappa_high=self._kappa_high.value(),
            use_gpu=self._use_gpu.isChecked(),
        )

        self._run_btn.setVisible(False)
        self._log_output.clear()
        self._log_output.append("Starting preprocessing pipeline...")

        self._worker = _PreprocessWorker(
            light_paths=list(self._group.lights),
            bias_paths=list(self._group.biases) if self._group.biases else None,
            dark_paths=list(self._group.darks) if self._group.darks else None,
            flat_paths=list(self._group.flats) if self._group.flats else None,
            output_dir=self._output_dir,
            calibrate=self._cb_calibrate.isChecked(),
            register=self._cb_register.isChecked(),
            stack=self._cb_stack.isChecked(),
            cosmetic=self._cb_cosmetic.isChecked(),
            stacking_params=stacking_params,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(lambda: self._worker.deleteLater())
        self._worker.start()

        self._current_page = 2
        self._stack.setCurrentIndex(self._current_page)
        self._update_nav()

    def _cancel_preprocessing(self):
        if self._worker:
            self._worker.cancel()
        self._log_output.append("Cancelling...")
        self._cancel_btn.setEnabled(False)

    def _on_progress(self, fraction: float, message: str):
        self._progress_bar.setValue(int(fraction * 100))
        self._progress_label.setText(message)

    def _on_error(self, message: str):
        self._log_output.append(f"<span style='color: #f85149;'>Error: {message}</span>")
        self._run_btn.setText("Run Preprocessing")
        self._run_btn.setVisible(True)
        self._cancel_btn.setVisible(False)

    def _on_finished(self, result: PreprocessingResult):
        self._progress_bar.setValue(100)
        self._progress_label.setText("Complete")

        summary = f"""
<b>Preprocessing Complete</b>
━━━━━━━━━━━━━━━━━━━━━
• Calibrated: {result.n_calibrated}
• Failed: {result.n_failed}
• Stacked: {'Yes' if result.stacked_image is not None else 'No'}
• Stack shape: {result.stacked_image.data.shape if result.stacked_image else 'N/A'}
"""
        if result.errors:
            summary += "\n<b>Errors:</b>\n"
            for err in result.errors[:10]:
                summary += f"  • {err}\n"
        self._result_summary.setHtml(summary)

        self._current_page = 3
        self._stack.setCurrentIndex(self._current_page)
        self._update_nav()
