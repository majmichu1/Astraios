"""Tests for camera-name matching and FITS-header auto-fill of pixel size."""

from astraios.core.equipment import detect_from_fits_header, match_camera_by_name


def test_pentax_present_and_matches():
    cam = match_camera_by_name("Pentax K-5 IIs")
    assert cam is not None
    assert cam.pixel_size_um == 4.78


def test_match_is_spacing_and_case_insensitive():
    for variant in ("PENTAX K-5 II s", "Pentax K-5IIs", "pentax k5iis"):
        cam = match_camera_by_name(variant)
        assert cam is not None and "Pentax" in cam.name


def test_unknown_camera_returns_none():
    assert match_camera_by_name("Totally Unknown Cam 9000") is None
    assert match_camera_by_name("") is None
    assert match_camera_by_name(None) is None


def test_header_autofills_pixel_size_for_unknown_gear():
    # Camera name in the header but no pixel-size keyword: should be filled in.
    info = detect_from_fits_header({"INSTRUME": "Pentax K-5 IIs", "FOCALLEN": 500})
    assert info.get("matched_camera") == "Pentax K-5 IIs"
    assert info.get("pixel_size_um") == 4.78


def test_header_pixel_size_keyword_wins():
    # An explicit pixel-size keyword is not overridden by the DB match.
    info = detect_from_fits_header({"INSTRUME": "Pentax K-5 IIs", "XPIXSZ": 9.9})
    assert info["pixel_size_um"] == 9.9
    assert "matched_camera" not in info
