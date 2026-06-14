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
    error = pyqtSignal(str)  # error message

    def __init__(
        self,
        processor,
        data,
        fits_header,
        input_type_hint,
        target_name=None,
        ra_hint=None,
        dec_hint=None,
        wcs=None,
        enabled_stages=None,
        hdr_operator="core_blend",
        hdr_params=None,
        star_reduction=0.3,
        use_ai_denoise=True,
    ):
        super().__init__()
        self._processor = processor
        self._data = data
        self._fits_header = fits_header
        self._input_type_hint = input_type_hint
        self._target_name = target_name
        self._ra_hint = ra_hint
        self._dec_hint = dec_hint
        self._wcs = wcs
        self._enabled_stages = enabled_stages
        self._hdr_operator = hdr_operator
        self._hdr_params = hdr_params
        self._star_reduction = star_reduction
        self._use_ai_denoise = use_ai_denoise
        self._cancel_requested = False

    def request_cancel(self):
        self._cancel_requested = True
        self.requestInterruption()

    def run(self):
        try:
            result = self._processor.process(
                self._data,
                fits_header=self._fits_header,
                input_type_hint=self._input_type_hint,
                target_name=self._target_name,
                ra_hint=self._ra_hint,
                dec_hint=self._dec_hint,
                wcs_dict=self._wcs,
                enabled_stages=self._enabled_stages,
                hdr_operator=self._hdr_operator,
                hdr_params=self._hdr_params,
                star_reduction=self._star_reduction,
                use_ai_denoise=self._use_ai_denoise,
                progress=self._emit_progress,
            )
            if self._cancel_requested:
                self.error.emit("Cancelled")
                return
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")

    def _emit_progress(self, fraction: float, message: str):
        if self._cancel_requested:
            raise InterruptedError("Smart Process cancelled")
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

        # --- Processing Stages ---
        stages_group = QGroupBox("Processing Stages")
        stages_layout = QVBoxLayout(stages_group)

        self._stage_bg = QCheckBox("Background extraction")
        self._stage_bg.setChecked(True)
        stages_layout.addWidget(self._stage_bg)

        self._stage_denoise = QCheckBox("Noise reduction")
        self._stage_denoise.setChecked(True)
        stages_layout.addWidget(self._stage_denoise)

        ai_row = QHBoxLayout()
        ai_row.addSpacing(20)
        self._ai_denoise_cb = QCheckBox("Use AI denoise (falls back to wavelet if no model)")
        self._ai_denoise_cb.setChecked(True)
        self._ai_denoise_cb.setToolTip(
            "Denoise with the trained Noise2Self model when available — cleaner "
            "than classical wavelet. Automatically falls back to wavelet otherwise."
        )
        self._stage_denoise.toggled.connect(self._ai_denoise_cb.setEnabled)
        ai_row.addWidget(self._ai_denoise_cb)
        ai_row.addStretch(1)
        stages_layout.addLayout(ai_row)

        self._stage_deconv = QCheckBox("Deconvolution")
        self._stage_deconv.setChecked(True)
        stages_layout.addWidget(self._stage_deconv)

        self._stage_stretch = QCheckBox("Adaptive stretch")
        self._stage_stretch.setChecked(True)
        stages_layout.addWidget(self._stage_stretch)

        self._stage_lce = QCheckBox("Local contrast")
        self._stage_lce.setChecked(True)
        stages_layout.addWidget(self._stage_lce)

        self._stage_hdr = QCheckBox("HDR enhancement")
        self._stage_hdr.setChecked(True)
        stages_layout.addWidget(self._stage_hdr)

        self._stage_star_aware = QCheckBox("Star-aware (remove stars, enhance nebula, recombine)")
        self._stage_star_aware.setChecked(True)
        self._stage_star_aware.setToolTip(
            "Separates stars before enhancing the nebula so an aggressive stretch "
            "and local contrast don't bloat them, then screens the stars back in."
        )
        stages_layout.addWidget(self._stage_star_aware)

        sr_row = QHBoxLayout()
        sr_row.addSpacing(20)
        sr_row.addWidget(QLabel("Star reduction"))
        self._star_reduction_spin = QDoubleSpinBox()
        self._star_reduction_spin.setRange(0.0, 1.0)
        self._star_reduction_spin.setSingleStep(0.05)
        self._star_reduction_spin.setValue(0.3)
        self._star_reduction_spin.setToolTip(
            "How much to shrink the isolated stars before screening them back. "
            "0 = leave star sizes unchanged."
        )
        self._stage_star_aware.toggled.connect(self._star_reduction_spin.setEnabled)
        sr_row.addWidget(self._star_reduction_spin)
        sr_row.addStretch(1)
        stages_layout.addLayout(sr_row)

        hdr_op_layout = QHBoxLayout()
        hdr_op_layout.setContentsMargins(20, 0, 0, 0)
        hdr_op_layout.addWidget(QLabel("Operator:"))
        self._hdr_operator_combo = QComboBox()
        self._hdr_operator_combo.addItem("Core Blend", "core_blend")
        self._hdr_operator_combo.addItem("Reinhard Tonemap", "reinhard")
        self._hdr_operator_combo.addItem("Drago Tonemap", "drago")
        self._hdr_operator_combo.setCurrentIndex(0)
        hdr_op_layout.addWidget(self._hdr_operator_combo)
        hdr_op_layout.addStretch()
        stages_layout.addLayout(hdr_op_layout)

        self._object_aware_cb = QCheckBox("Object-aware background")
        self._object_aware_cb.setChecked(True)
        stages_layout.addWidget(self._object_aware_cb)

        self._adaptive_cb = QCheckBox("Adaptive quality checks")
        self._adaptive_cb.setChecked(True)
        stages_layout.addWidget(self._adaptive_cb)

        layout.addWidget(stages_group)

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

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        btn_row.addWidget(self._cancel_btn)

        self._apply_btn = QPushButton("Apply Result")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply_result)
        btn_row.addWidget(self._apply_btn)

        layout.addLayout(btn_row)

    def set_image_data(self, data, fits_header=None, wcs=None):
        """Set the image data to process."""
        self._data = data
        self._fits_header = fits_header
        self._wcs = wcs

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
        self._cancel_btn.setEnabled(True)
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

        # Collect enabled stages from checkboxes
        enabled_stages = set()
        if self._stage_bg.isChecked():
            enabled_stages.add("background")
        if self._stage_denoise.isChecked():
            enabled_stages.add("denoise")
        if self._stage_deconv.isChecked():
            enabled_stages.add("deconv")
        if self._stage_stretch.isChecked():
            enabled_stages.add("stretch")
        if self._stage_lce.isChecked():
            enabled_stages.add("local_contrast")
        if self._stage_hdr.isChecked():
            enabled_stages.add("hdr_merge")
        if self._stage_star_aware.isChecked():
            enabled_stages.add("star_aware")

        processor = SmartProcessor(equipment=self._equipment)
        hdr_op = self._hdr_operator_combo.currentData()
        self._worker = SmartProcessWorker(
            processor,
            self._data,
            getattr(self, "_fits_header", None),
            input_type_hint,
            target_name=target_name,
            ra_hint=ra_hint,
            dec_hint=dec_hint,
            wcs=getattr(self, "_wcs", None),
            enabled_stages=enabled_stages,
            hdr_operator=hdr_op,
            star_reduction=self._star_reduction_spin.value(),
            use_ai_denoise=self._ai_denoise_cb.isChecked(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_worker_error)
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
        self._cancel_btn.setEnabled(False)
        self._status_label.setText("Done")

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

    def _on_worker_error(self, message: str):
        self._run_btn.setEnabled(True)
        self._apply_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        self._progress_bar.setValue(0)
        self._status_label.setText(f"Error: {message}")
        self._log_text.append(f"ERROR: {message}")
        self._worker = None

    def _on_cancel_clicked(self):
        if self._worker is not None:
            self._cancel_btn.setEnabled(False)
            self._status_label.setText("Cancelling...")
            self._worker.request_cancel()

    def _apply_result(self):
        if self._result:
            self.result_ready.emit(self._result)
            self.accept()
