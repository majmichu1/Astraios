"""Smart Processor Dialog — AI-driven adaptive processing UI."""

from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
)

from cosmica.ai.smart_processor import (
    InputType,
    SmartProcessor,
    SmartProcessorResult,
)
from cosmica.core.equipment import EquipmentProfile
from cosmica.ui.dialogs.equipment_dialog import EquipmentDialog


class SmartProcessWorker(QThread):
    """Runs Smart Processor off the main thread."""

    progress = pyqtSignal(float, str)
    finished = pyqtSignal(object)  # SmartProcessorResult

    def __init__(
        self,
        processor,
        data,
        fits_header,
        input_type_hint,
        target_name=None,
        ra_hint=None,
        dec_hint=None,
    ):
        super().__init__()
        self._processor = processor
        self._data = data
        self._fits_header = fits_header
        self._input_type_hint = input_type_hint
        self._target_name = target_name
        self._ra_hint = ra_hint
        self._dec_hint = dec_hint

    def run(self):
        result = self._processor.process(
            self._data,
            fits_header=self._fits_header,
            input_type_hint=self._input_type_hint,
            target_name=self._target_name,
            ra_hint=self._ra_hint,
            dec_hint=self._dec_hint,
            progress=self._emit_progress,
        )
        self.finished.emit(result)

    def _emit_progress(self, fraction: float, message: str):
        self.progress.emit(fraction, message)


class SmartProcessDialog(QDialog):
    """Dialog for AI-driven Smart Processing."""

    result_ready = pyqtSignal(object)  # emits SmartProcessorResult

    def __init__(
        self,
        parent=None,
        equipment: EquipmentProfile | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Smart Processor")
        self.setMinimumSize(600, 650)

        self._equipment = equipment
        self._worker: SmartProcessWorker | None = None
        self._result: SmartProcessorResult | None = None

        layout = QVBoxLayout(self)

        # --- Equipment section ---
        equip_group = QGroupBox("Equipment")
        equip_layout = QHBoxLayout(equip_group)

        self._equip_label = QLabel(self._equipment_summary())
        self._equip_label.setWordWrap(True)
        equip_layout.addWidget(self._equip_label, 1)

        equip_btn = QPushButton("Configure...")
        equip_btn.clicked.connect(self._open_equipment_dialog)
        equip_layout.addWidget(equip_btn)

        layout.addWidget(equip_group)

        # --- Target Information ---
        target_group = QGroupBox("Target Information (optional)")
        target_layout = QVBoxLayout(target_group)

        target_layout.addWidget(QLabel(
            "Help the Smart Processor identify your target for optimized processing."
        ))

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Target name:"))
        self._target_name_edit = QLineEdit()
        self._target_name_edit.setPlaceholderText("e.g. M42, NGC 7000, IC 1396...")
        name_row.addWidget(self._target_name_edit)
        target_layout.addLayout(name_row)

        coord_row = QHBoxLayout()
        coord_row.addWidget(QLabel("RA (deg):"))
        self._ra_spin = QDoubleSpinBox()
        self._ra_spin.setRange(0.0, 360.0)
        self._ra_spin.setDecimals(4)
        self._ra_spin.setSpecialValueText("auto")
        self._ra_spin.setValue(0.0)
        coord_row.addWidget(self._ra_spin)
        coord_row.addWidget(QLabel("Dec (deg):"))
        self._dec_spin = QDoubleSpinBox()
        self._dec_spin.setRange(-90.0, 90.0)
        self._dec_spin.setDecimals(4)
        self._dec_spin.setSpecialValueText("auto")
        self._dec_spin.setValue(0.0)
        coord_row.addWidget(self._dec_spin)
        target_layout.addLayout(coord_row)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Image type:"))
        self._type_combo = QComboBox()
        self._type_combo.addItems([
            "Auto-detect",
            "OSC / Color (RGB)",
            "Mono (Luminance)",
            "Narrowband SHO",
            "Narrowband HOO",
            "Dual Narrowband",
        ])
        type_row.addWidget(self._type_combo)
        target_layout.addLayout(type_row)

        layout.addWidget(target_group)

        # --- Options ---
        options_group = QGroupBox("Options")
        opt_layout = QVBoxLayout(options_group)

        self._object_aware_cb = QCheckBox("Object-aware background extraction")
        self._object_aware_cb.setChecked(True)
        opt_layout.addWidget(self._object_aware_cb)

        self._deconv_cb = QCheckBox("Enable deconvolution (when beneficial)")
        self._deconv_cb.setChecked(True)
        opt_layout.addWidget(self._deconv_cb)

        self._adaptive_cb = QCheckBox("Adaptive quality checks")
        self._adaptive_cb.setChecked(True)
        opt_layout.addWidget(self._adaptive_cb)

        layout.addWidget(options_group)

        # --- Progress ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("Ready — click 'Run' to start")
        layout.addWidget(self._status_label)

        # --- Log output ---
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(200)
        self._log_text.setStyleSheet(
            "font-family: monospace; font-size: 11px; "
            "background: #1a1a2e; color: #e0e0e0;"
        )
        log_layout.addWidget(self._log_text)
        layout.addWidget(log_group)

        # --- Results summary (initially hidden) ---
        self._results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(self._results_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._results_content = QLabel("")
        self._results_content.setWordWrap(True)
        scroll.setWidget(self._results_content)
        results_layout.addWidget(scroll)

        self._results_group.setVisible(False)
        layout.addWidget(self._results_group)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._run_btn = QPushButton("Run Smart Processor")
        self._run_btn.clicked.connect(self._run)
        btn_row.addWidget(self._run_btn)

        self._apply_btn = QPushButton("Apply Result")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply_result)
        btn_row.addWidget(self._apply_btn)

        layout.addLayout(btn_row)

    def set_image_data(self, data, fits_header=None):
        """Set the image data to process."""
        self._data = data
        self._fits_header = fits_header

    def _equipment_summary(self) -> str:
        if not self._equipment:
            return "No equipment profile set. Click 'Configure...' to select your equipment."
        cam = self._equipment.camera.name
        scope = self._equipment.telescope.name
        ps = self._equipment.plate_scale()
        n_filt = len(self._equipment.filters)
        return (
            f"Camera: {cam}\n"
            f"Telescope: {scope}\n"
            f"Plate scale: {ps:.2f} arcsec/px | "
            f"{n_filt} filter(s) configured"
        )

    def _open_equipment_dialog(self):
        dlg = EquipmentDialog(self, self._equipment)
        dlg.profile_ready.connect(self._on_equipment_set)
        dlg.exec()

    def _on_equipment_set(self, profile: EquipmentProfile):
        self._equipment = profile
        self._equip_label.setText(self._equipment_summary())

    def _run(self):
        if not hasattr(self, "_data") or self._data is None:
            self._status_label.setText("No image data loaded")
            return

        self._run_btn.setEnabled(False)
        self._apply_btn.setEnabled(False)
        self._results_group.setVisible(False)
        self._log_text.clear()
        self._progress_bar.setValue(0)

        # Gather target info from user inputs
        target_name = self._target_name_edit.text().strip() or None
        ra_hint = self._ra_spin.value() if self._ra_spin.value() > 0 else None
        dec_hint = self._dec_spin.value() if self._dec_spin.value() != 0 or (ra_hint is not None) else None

        # Image type override
        type_map = {
            0: None,  # Auto-detect
            1: InputType.OSC_RGB,
            2: InputType.MONO_LUMINANCE,
            3: InputType.NARROWBAND_SHO,
            4: InputType.NARROWBAND_HOO,
            5: InputType.DUAL_NARROWBAND,
        }
        input_type_hint = type_map.get(self._type_combo.currentIndex())

        processor = SmartProcessor(equipment=self._equipment)
        self._worker = SmartProcessWorker(
            processor,
            self._data,
            getattr(self, "_fits_header", None),
            input_type_hint,
            target_name=target_name,
            ra_hint=ra_hint,
            dec_hint=dec_hint,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, fraction: float, message: str):
        self._progress_bar.setValue(int(fraction * 100))
        self._status_label.setText(message)
        self._log_text.append(message)

    def _on_finished(self, result: SmartProcessorResult):
        self._result = result
        self._run_btn.setEnabled(True)
        self._apply_btn.setEnabled(True)
        self._progress_bar.setValue(100)

        # Compact status summary
        a = result.analysis
        passed = sum(1 for q in result.quality_checks if q.passed)
        total = len(result.quality_checks)
        target_str = f" | Target: {a.primary_target.id}" if a.primary_target else ""
        solve_str = " | Plate solve: OK" if (a.plate_solve_result and a.plate_solve_result.success) else ""
        self._status_label.setText(
            f"Done — QC: {passed}/{total} passed{target_str}{solve_str}"
        )

        # Show log
        self._log_text.clear()
        for msg in result.processing_log:
            self._log_text.append(msg)

        # Show results summary
        self._results_group.setVisible(True)
        lines = []

        # Analysis
        a = result.analysis
        lines.append(f"Input type: {a.input_type.name}")
        lines.append(f"Dimensions: {a.width}x{a.height}, {a.n_channels} channel(s)")
        lines.append(f"SNR: {a.median_snr:.1f}")
        lines.append(f"Dynamic range: {a.dynamic_range_stops:.1f} stops")

        if a.psf and a.psf.n_stars_used > 0:
            lines.append(
                f"PSF FWHM: {a.psf.fwhm:.2f} px "
                f"(ellipticity {a.psf.ellipticity:.3f}, "
                f"{a.psf.n_stars_used} stars)"
            )

        # Plate solve status
        if a.plate_solve_result and a.plate_solve_result.success:
            lines.append(
                f"Plate solve: ✓ SUCCESS — "
                f"RA={a.plate_solve_result.ra_center:.4f}°, "
                f"Dec={a.plate_solve_result.dec_center:.4f}°, "
                f"scale={a.plate_solve_result.pixel_scale:.2f}\"/px"
            )
        else:
            lines.append("Plate solve: ✗ No WCS solution (local solver needs reference catalog)")

        # Target identification
        if a.primary_target:
            t = a.primary_target
            names = f" ({', '.join(t.names[:2])})" if t.names else ""
            lines.append(f"Target identified: {t.id}{names}")
            lines.append(f"  Type: {t.object_type}, brightness: {t.brightness_class}")
            lines.append(f"  Angular size: {t.major_axis_arcmin:.0f}'×{t.minor_axis_arcmin:.0f}', "
                         f"constellation: {t.constellation}")
        else:
            lines.append("Target: Not identified (enter target name above for catalog lookup)")

        # Quality checks
        passed = sum(1 for q in result.quality_checks if q.passed)
        total = len(result.quality_checks)
        lines.append(f"\nQuality checks: {passed}/{total} passed")
        for qc in result.quality_checks:
            status = "PASS" if qc.passed else "ADJUSTED"
            lines.append(
                f"  [{status}] {qc.stage.value}: {qc.metric_name}="
                f"{qc.metric_value:.4f}"
            )
            if qc.adjustment:
                lines.append(f"    -> {qc.adjustment}")

        self._results_content.setText("\n".join(lines))

    def _apply_result(self):
        if self._result:
            self.result_ready.emit(self._result)
            self.accept()
