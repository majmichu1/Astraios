"""Regression: a profile with a CUSTOM telescope (e.g. a user-entered lens) was
dropped on load — findText() returns -1 for a name like 'Custom 268mm f/3.6',
so the combo kept its default and the focal length/aperture were never restored.
"""

from astraios.core.equipment import (
    CameraProfile,
    EquipmentProfile,
    TelescopeProfile,
)


def _profile_with_custom_scope() -> EquipmentProfile:
    cam = load_first_camera()
    scope = TelescopeProfile(
        name="Custom 268mm f/3.6",
        aperture_mm=75.0,
        focal_length_mm=268.0,
        focal_ratio=268.0 / 75.0,
        telescope_type="custom",
    )
    return EquipmentProfile(camera=cam, telescope=scope, filters={})


def load_first_camera() -> CameraProfile:
    from astraios.core.equipment import load_camera_database

    return load_camera_database()[0]


def test_custom_telescope_round_trips_through_dialog(qtbot):
    from astraios.ui.dialogs.equipment_dialog import EquipmentDialog

    prof = _profile_with_custom_scope()
    dlg = EquipmentDialog(current_profile=prof)

    # The custom scope must be reflected in the UI, not silently dropped.
    assert dlg._is_custom_scope()
    assert dlg._manual_focal_spin.value() == 268.0
    assert dlg._manual_aperture_spin.value() == 75.0
    assert dlg._manual_scope_name.text() == "Custom 268mm f/3.6"

    # And rebuilding the profile (what 'Apply' emits) must preserve the scope.
    rebuilt = dlg._build_profile()
    assert rebuilt is not None
    assert rebuilt.telescope.focal_length_mm == 268.0
    assert rebuilt.telescope.aperture_mm == 75.0
    assert abs(rebuilt.plate_scale() - prof.plate_scale()) < 1e-6


def test_database_telescope_still_selects_normally(qtbot):
    from astraios.core.equipment import load_telescope_database
    from astraios.ui.dialogs.equipment_dialog import EquipmentDialog

    scope = load_telescope_database()[0]
    prof = EquipmentProfile(camera=load_first_camera(), telescope=scope, filters={})
    dlg = EquipmentDialog(current_profile=prof)

    assert not dlg._is_custom_scope()
    assert dlg._telescope_combo.currentText() == scope.name
