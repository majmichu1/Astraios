"""Equipment Profile Dialog — select camera, telescope, and filters."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from astraios.core.equipment import (
    CameraProfile,
    EquipmentProfile,
    FilterProfile,
    TelescopeProfile,
    load_camera_database,
    load_filter_database,
    load_telescope_database,
)


class EquipmentDialog(QDialog):
    """Dialog for selecting and managing equipment profiles."""

    profile_ready = pyqtSignal(object)  # emits EquipmentProfile

    def __init__(self, parent=None, current_profile: EquipmentProfile | None = None):
        super().__init__(parent)
        self.setWindowTitle("Equipment Profile")
        self.setMinimumSize(500, 450)

        self._cameras: list[CameraProfile] = []
        self._telescopes: list[TelescopeProfile] = []
        self._filters: list[FilterProfile] = []
        self._current = current_profile

        layout = QVBoxLayout(self)

        # --- Camera ---
        cam_group = QGroupBox("Camera")
        cam_layout = QVBoxLayout(cam_group)
        self._camera_combo = QComboBox()
        cam_layout.addWidget(self._camera_combo)
        self._camera_info = QLabel("")
        self._camera_info.setStyleSheet("color: #aaa; font-size: 11px;")
        cam_layout.addWidget(self._camera_info)
        layout.addWidget(cam_group)

        # --- Telescope ---
        scope_group = QGroupBox("Telescope / Lens")
        scope_layout = QVBoxLayout(scope_group)
        self._telescope_combo = QComboBox()
        scope_layout.addWidget(self._telescope_combo)
        self._scope_info = QLabel("")
        self._scope_info.setStyleSheet("color: #aaa; font-size: 11px;")
        scope_layout.addWidget(self._scope_info)

        # Manual telescope entry (initially hidden)
        self._manual_scope_widget = QWidget()
        manual_layout = QVBoxLayout(self._manual_scope_widget)
        manual_layout.setContentsMargins(0, 4, 0, 0)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._manual_scope_name = QLineEdit()
        self._manual_scope_name.setPlaceholderText("e.g. My Refractor")
        name_row.addWidget(self._manual_scope_name, 1)
        manual_layout.addLayout(name_row)

        fl_row = QHBoxLayout()
        fl_row.addWidget(QLabel("Focal length (mm):"))
        self._manual_focal_spin = QDoubleSpinBox()
        self._manual_focal_spin.setRange(50, 10000)
        self._manual_focal_spin.setValue(1000)
        self._manual_focal_spin.setDecimals(1)
        fl_row.addWidget(self._manual_focal_spin, 1)
        manual_layout.addLayout(fl_row)

        ap_row = QHBoxLayout()
        ap_row.addWidget(QLabel("Aperture (mm):"))
        self._manual_aperture_spin = QDoubleSpinBox()
        self._manual_aperture_spin.setRange(10, 2000)
        self._manual_aperture_spin.setValue(200)
        self._manual_aperture_spin.setDecimals(1)
        ap_row.addWidget(self._manual_aperture_spin, 1)
        manual_layout.addLayout(ap_row)

        self._manual_scope_widget.setVisible(False)
        scope_layout.addWidget(self._manual_scope_widget)

        layout.addWidget(scope_group)

        # --- Filters ---
        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout(filter_group)

        filter_slots = ["Luminance", "Red", "Green", "Blue", "Ha", "OIII", "SII"]
        self._filter_combos: dict[str, QComboBox] = {}
        for slot in filter_slots:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{slot}:"))
            combo = QComboBox()
            combo.addItem("(none)")
            row.addWidget(combo, 1)
            filter_layout.addLayout(row)
            self._filter_combos[slot] = combo

        layout.addWidget(filter_group)

        # --- Computed info ---
        self._computed_label = QLabel("")
        self._computed_label.setStyleSheet("color: #80c0ff; font-size: 12px;")
        layout.addWidget(self._computed_label)

        # --- Buttons ---
        btn_row = QHBoxLayout()

        load_btn = QPushButton("Load Profile...")
        load_btn.clicked.connect(self._load_profile)
        btn_row.addWidget(load_btn)

        save_btn = QPushButton("Save Profile...")
        save_btn.clicked.connect(self._save_profile)
        btn_row.addWidget(save_btn)

        btn_row.addStretch()

        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply)
        btn_row.addWidget(apply_btn)

        layout.addLayout(btn_row)

        # Load databases
        self._load_databases()
        self._camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        self._telescope_combo.currentIndexChanged.connect(self._on_telescope_changed)
        self._manual_focal_spin.valueChanged.connect(lambda: self._update_info())
        self._manual_aperture_spin.valueChanged.connect(lambda: self._update_info())

    def _load_databases(self):
        try:
            self._cameras = load_camera_database()
            self._camera_combo.clear()
            for cam in self._cameras:
                self._camera_combo.addItem(cam.name)
        except Exception:
            self._camera_combo.addItem("(no camera database)")

        try:
            self._telescopes = load_telescope_database()
            self._telescope_combo.clear()
            for scope in self._telescopes:
                self._telescope_combo.addItem(scope.name)
        except Exception:
            self._telescope_combo.addItem("(no telescope database)")
        self._telescope_combo.addItem("-- Custom Entry --")

        try:
            self._filters = load_filter_database()
            for combo in self._filter_combos.values():
                for filt in self._filters:
                    combo.addItem(filt.name)
        except Exception:
            pass

        # Restore current profile selections if available
        if self._current:
            self._select_by_name(self._camera_combo, self._current.camera.name)
            self._restore_telescope(self._current.telescope)
            for slot, filt in self._current.filters.items():
                if slot in self._filter_combos:
                    self._select_by_name(self._filter_combos[slot], filt.name)

        self._update_info()

    def _select_by_name(self, combo: QComboBox, name: str):
        idx = combo.findText(name)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _restore_telescope(self, scope: TelescopeProfile) -> None:
        """Select *scope* in the combo, or restore it into the custom-entry
        fields when it is not a database telescope (e.g. a user-defined lens).

        Without this, loading a profile with a custom scope silently dropped it
        — findText() returns -1 for a name like 'Custom 268mm f/3.6', so the
        combo kept its default and the focal length/aperture were never filled.
        """
        idx = self._telescope_combo.findText(scope.name)
        if 0 <= idx < len(self._telescopes):
            self._telescope_combo.setCurrentIndex(idx)
            return
        # Not in the database → populate the manual fields and switch to custom.
        self._manual_scope_name.setText(scope.name)
        self._manual_focal_spin.setValue(scope.focal_length_mm)
        self._manual_aperture_spin.setValue(scope.aperture_mm)
        self._telescope_combo.setCurrentIndex(len(self._telescopes))  # "-- Custom Entry --"
        self._manual_scope_widget.setVisible(True)

    def _on_camera_changed(self):
        self._update_info()

    def _on_telescope_changed(self):
        is_custom = self._telescope_combo.currentIndex() == len(self._telescopes)
        self._manual_scope_widget.setVisible(is_custom)
        self._update_info()

    def _is_custom_scope(self) -> bool:
        return self._telescope_combo.currentIndex() == len(self._telescopes)

    def _get_custom_telescope(self) -> TelescopeProfile:
        fl = self._manual_focal_spin.value()
        ap = self._manual_aperture_spin.value()
        name = self._manual_scope_name.text().strip() or f"Custom {fl:.0f}mm f/{fl/ap:.1f}"
        return TelescopeProfile(
            name=name,
            aperture_mm=ap,
            focal_length_mm=fl,
            focal_ratio=fl / max(ap, 1),
            telescope_type="custom",
        )

    def _update_info(self):
        cam_idx = self._camera_combo.currentIndex()
        if 0 <= cam_idx < len(self._cameras):
            cam = self._cameras[cam_idx]
            cam_type = "Mono" if cam.is_mono else f"Color ({cam.bayer_pattern})"
            self._camera_info.setText(
                f"{cam.sensor} | {cam.pixel_size_um}um | "
                f"QE {cam.qe_peak:.0%} | {cam_type} | "
                f"{cam.resolution_x}x{cam.resolution_y}"
            )
        else:
            self._camera_info.setText("")

        scope_idx = self._telescope_combo.currentIndex()
        scope = None
        if self._is_custom_scope():
            scope = self._get_custom_telescope()
            self._scope_info.setText(
                f"{scope.aperture_mm:.0f}mm aperture | "
                f"{scope.focal_length_mm:.0f}mm FL | "
                f"f/{scope.focal_ratio:.1f} | custom"
            )
        elif 0 <= scope_idx < len(self._telescopes):
            scope = self._telescopes[scope_idx]
            self._scope_info.setText(
                f"{scope.aperture_mm:.0f}mm aperture | "
                f"{scope.focal_length_mm:.0f}mm FL | "
                f"f/{scope.focal_ratio:.1f} | {scope.telescope_type}"
            )
        else:
            self._scope_info.setText("")

        # Compute plate scale etc
        if 0 <= cam_idx < len(self._cameras) and scope is not None:
            cam = self._cameras[cam_idx]
            ps = 206.265 * cam.pixel_size_um / scope.focal_length_mm
            fov_w = ps * cam.resolution_x / 60.0
            fov_h = ps * cam.resolution_y / 60.0
            self._computed_label.setText(
                f"Plate scale: {ps:.2f} arcsec/px | "
                f"FOV: {fov_w:.1f}' x {fov_h:.1f}'"
            )

    def _build_profile(self) -> EquipmentProfile | None:
        cam_idx = self._camera_combo.currentIndex()
        if cam_idx < 0 or cam_idx >= len(self._cameras):
            return None

        if self._is_custom_scope():
            telescope = self._get_custom_telescope()
        else:
            scope_idx = self._telescope_combo.currentIndex()
            if scope_idx < 0 or scope_idx >= len(self._telescopes):
                return None
            telescope = self._telescopes[scope_idx]

        filters = {}
        for slot, combo in self._filter_combos.items():
            idx = combo.currentIndex()
            if idx > 0 and (idx - 1) < len(self._filters):
                filters[slot] = self._filters[idx - 1]

        return EquipmentProfile(
            camera=self._cameras[cam_idx],
            telescope=telescope,
            filters=filters,
        )

    def _apply(self):
        profile = self._build_profile()
        if profile:
            self.profile_ready.emit(profile)
            self.accept()

    def _save_profile(self):
        profile = self._build_profile()
        if not profile:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Equipment Profile", "equipment.json", "JSON (*.json)"
        )
        if not path:
            return
        # Qt on Linux does NOT auto-append the filter's extension, so a name like
        # "178mm" would be saved with no suffix — and then hidden by the
        # "*.json" filter on load, making it look like the save never happened.
        p = Path(path)
        if p.suffix.lower() != ".json":
            p = p.with_suffix(".json")
        try:
            profile.save(p)
            QMessageBox.information(
                self, "Profile Saved", f"Equipment profile saved to:\n{p}"
            )
        except Exception as exc:  # disk error, permissions, etc.
            QMessageBox.warning(
                self, "Save Failed", f"Could not save profile:\n{exc}"
            )

    def _load_profile(self):
        # Also surface extensionless legacy files (saved before the .json fix)
        # so users can still recover them.
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Equipment Profile", "", "JSON (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            profile = EquipmentProfile.load(Path(path))
            self._current = profile
            self._select_by_name(self._camera_combo, profile.camera.name)
            self._restore_telescope(profile.telescope)
            for slot, filt in profile.filters.items():
                if slot in self._filter_combos:
                    self._select_by_name(self._filter_combos[slot], filt.name)
            self._update_info()
        except Exception as exc:
            QMessageBox.warning(
                self, "Load Failed",
                f"Could not load profile from:\n{path}\n\n{exc}",
            )
