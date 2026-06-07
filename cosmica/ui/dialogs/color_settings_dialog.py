"""Color Settings Dialog — ICC profile selection, rendering intent, and conversion options."""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cosmica.core.color_management import (
    BUILTIN_PROFILES,
    RENDERING_INTENTS,
    ColorProfile,
    detect_monitor_profile,
    register_profile,
)

log = logging.getLogger(__name__)

SETTINGS_KEY_PROFILE = "color/profile_name"
SETTINGS_KEY_INTENT = "color/rendering_intent"
SETTINGS_KEY_GAMMA = "color/display_gamma"
SETTINGS_KEY_SOFTPROOF = "color/soft_proof_enabled"
SETTINGS_KEY_SOFTPROOF_PROFILE = "color/soft_proof_profile"


class ColorSettingsDialog(QDialog):
    """Dialog for managing ICC color profiles and display settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Color Management")
        self.setMinimumSize(480, 360)

        self._settings = QSettings("Cosmica", "Cosmica")
        self._monitor_profile: ColorProfile = detect_monitor_profile()
        self._profiles: dict[str, ColorProfile] = dict(BUILTIN_PROFILES)
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # ── Monitor profile ──────────────────────────────────────────────
        monitor_group = QGroupBox("Monitor Profile")
        mon_layout = QFormLayout(monitor_group)

        self._monitor_label = QLabel(self._monitor_profile.name)
        self._monitor_label.setStyleSheet("font-weight: bold;")
        mon_layout.addRow("Detected:", self._monitor_label)

        path_str = (
            str(self._monitor_profile.path)
            if self._monitor_profile.path
            else "sRGB (built-in)"
        )
        self._monitor_path_label = QLabel(path_str)
        self._monitor_path_label.setWordWrap(True)
        self._monitor_path_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        mon_layout.addRow("Path:", self._monitor_path_label)

        detect_btn = QPushButton("Re-detect Monitor Profile")
        detect_btn.clicked.connect(self._on_detect)
        mon_layout.addRow("", detect_btn)

        layout.addWidget(monitor_group)

        # ── Conversion settings ──────────────────────────────────────────
        conv_group = QGroupBox("Conversion Settings")
        conv_layout = QFormLayout(conv_group)

        self._profile_combo = QComboBox()
        for name in self._profiles:
            self._profile_combo.addItem(name)
        self._profile_combo.addItem("Custom (browse…)")
        self._profile_combo.currentTextChanged.connect(self._on_profile_changed)
        conv_layout.addRow("Working profile:", self._profile_combo)

        self._intent_combo = QComboBox()
        for label in RENDERING_INTENTS:
            self._intent_combo.addItem(label)
        self._intent_combo.setToolTip(
            "Perceptual: best for photographs\n"
            "Relative Colorimetric: preserve in-gamut colors (default)\n"
            "Saturation: vivid colors for graphics\n"
            "Absolute Colorimetric: simulate print output"
        )
        conv_layout.addRow("Rendering intent:", self._intent_combo)

        self._gamma_spin = QDoubleSpinBox()
        self._gamma_spin.setRange(1.0, 3.5)
        self._gamma_spin.setSingleStep(0.1)
        self._gamma_spin.setValue(self._load_gamma())
        self._gamma_spin.setDecimals(2)
        self._gamma_spin.setStyleSheet("font-weight: bold;")
        conv_layout.addRow("Display gamma:", self._gamma_spin)

        layout.addWidget(conv_group)

        # ── Soft-proofing ────────────────────────────────────────────────
        proof_group = QGroupBox("Soft-Proofing")
        proof_layout = QVBoxLayout(proof_group)

        self._softproof_cb = QCheckBox("Enable soft-proof simulation")
        proof_layout.addWidget(self._softproof_cb)

        sp_row = QWidget()
        sp_row_layout = QHBoxLayout(sp_row)
        sp_row_layout.setContentsMargins(0, 0, 0, 0)

        sp_row_layout.addWidget(QLabel("Proof profile:"))
        self._softproof_profile_combo = QComboBox()
        for name in self._profiles:
            self._softproof_profile_combo.addItem(name)
        self._softproof_profile_combo.addItem("Custom (browse…)")
        sp_row_layout.addWidget(self._softproof_profile_combo, 1)
        proof_layout.addWidget(sp_row)

        layout.addWidget(proof_group)

        # ── Buttons ──────────────────────────────────────────────────────
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _on_detect(self):
        """Re-detect monitor profile."""
        self._monitor_profile = detect_monitor_profile()
        self._monitor_label.setText(self._monitor_profile.name)
        path_str = (
            str(self._monitor_profile.path)
            if self._monitor_profile.path
            else "sRGB (built-in)"
        )
        self._monitor_path_label.setText(path_str)
        log.info("Detected monitor profile: %s", self._monitor_profile.name)

    def _on_profile_changed(self, name: str):
        if name == "Custom (browse…)":
            path, _ = QFileDialog.getOpenFileName(
                self, "Select ICC Profile", "",
                "ICC Profiles (*.icc *.icm);;All Files (*)",
            )
            if path:
                cp = register_profile(Path(path))
                if cp.is_valid():
                    self._profile_combo.blockSignals(True)
                    idx = self._profile_combo.count() - 1
                    self._profile_combo.insertItem(idx, cp.name)
                    self._profile_combo.setCurrentText(cp.name)
                    self._profile_combo.blockSignals(False)
                else:
                    QMessageBox.warning(self, "Invalid Profile",
                                        "Could not load the selected ICC profile.")
                    self._profile_combo.setCurrentText("sRGB")
            else:
                self._profile_combo.setCurrentText("sRGB")

    def _load_settings(self):
        profile_name = self._settings.value(SETTINGS_KEY_PROFILE, "sRGB", type=str)
        idx = self._profile_combo.findText(profile_name)
        if idx >= 0:
            self._profile_combo.setCurrentIndex(idx)

        intent_name = self._settings.value(SETTINGS_KEY_INTENT, "Perceptual", type=str)
        idx = self._intent_combo.findText(intent_name)
        if idx >= 0:
            self._intent_combo.setCurrentIndex(idx)

        checked = self._settings.value(SETTINGS_KEY_SOFTPROOF, False, type=bool)
        assert isinstance(checked, bool)
        self._softproof_cb.setChecked(checked)
        sp_name = self._settings.value(SETTINGS_KEY_SOFTPROOF_PROFILE, "sRGB", type=str)
        idx = self._softproof_profile_combo.findText(sp_name)
        if idx >= 0:
            self._softproof_profile_combo.setCurrentIndex(idx)

    def _on_accept(self):
        self._settings.setValue(
            SETTINGS_KEY_PROFILE, self._profile_combo.currentText()
        )
        self._settings.setValue(
            SETTINGS_KEY_INTENT, self._intent_combo.currentText()
        )
        self._settings.setValue(
            SETTINGS_KEY_SOFTPROOF, self._softproof_cb.isChecked()
        )
        self._settings.setValue(
            SETTINGS_KEY_SOFTPROOF_PROFILE, self._softproof_profile_combo.currentText()
        )
        self.accept()

    # ── Public accessors ────────────────────────────────────────────────

    def get_working_profile(self) -> ColorProfile:
        """Return the selected working color profile."""
        name = self._profile_combo.currentText()
        return self._profiles.get(name, list(self._profiles.values())[0])

    def get_rendering_intent(self) -> int:
        """Return the selected rendering intent as an ICC integer."""
        label = self._intent_combo.currentText()
        return RENDERING_INTENTS.get(label, 0)

    def get_display_gamma(self) -> float:
        """Return the configured display gamma."""
        return float(self._gamma_spin.value())

    def _load_gamma(self) -> float:
        """Load the saved display gamma from settings, or 2.2 default."""
        try:
            from cosmica.core.config import get_config
            cfg = get_config()
            return float(cfg.get(SETTINGS_KEY_GAMMA, 2.2))
        except Exception:
            return 2.2

    def accept(self):
        """Persist gamma when user clicks OK."""
        try:
            from cosmica.core.config import get_config
            get_config().set(SETTINGS_KEY_GAMMA, float(self._gamma_spin.value()))
        except Exception:
            pass
        super().accept()

    def is_soft_proof_enabled(self) -> bool:
        """Whether soft-proof simulation is active."""
        return bool(self._softproof_cb.isChecked())

    def get_soft_proof_profile(self) -> ColorProfile:
        """Return the profile used for soft-proof simulation."""
        name = self._softproof_profile_combo.currentText()
        return self._profiles.get(name, list(self._profiles.values())[0])
