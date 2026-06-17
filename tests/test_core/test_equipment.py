"""Tests for equipment profile system."""

import json

import numpy as np
import pytest

from astraios.core.equipment import (
    CameraProfile,
    EquipmentProfile,
    FilterProfile,
    TelescopeProfile,
    detect_from_fits_header,
    load_camera_database,
    load_filter_database,
    load_telescope_database,
)


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mono_camera() -> CameraProfile:
    """A typical mono cooled CMOS camera (ASI294MM-style)."""
    return CameraProfile(
        name="TestCam Mono 294",
        sensor="IMX294",
        pixel_size_um=4.63,
        read_noise_e=1.2,
        dark_current_e_per_s=0.002,
        full_well_e=63000,
        qe_peak=0.80,
        qe_curve=[
            (400.0, 0.40),
            (450.0, 0.60),
            (500.0, 0.75),
            (550.0, 0.80),
            (600.0, 0.72),
            (656.0, 0.65),
            (700.0, 0.45),
            (800.0, 0.15),
        ],
        bayer_pattern=None,
        resolution_x=4144,
        resolution_y=2822,
        camera_type="mono",
    )


@pytest.fixture
def color_camera() -> CameraProfile:
    """A typical one-shot-color CMOS camera."""
    return CameraProfile(
        name="TestCam Color 533",
        sensor="IMX533",
        pixel_size_um=3.76,
        read_noise_e=1.0,
        dark_current_e_per_s=0.003,
        full_well_e=50000,
        qe_peak=0.80,
        qe_curve=[
            (400.0, 0.35),
            (500.0, 0.70),
            (550.0, 0.80),
            (656.0, 0.60),
            (700.0, 0.40),
        ],
        bayer_pattern="RGGB",
        resolution_x=3008,
        resolution_y=3008,
        camera_type="color",
    )


@pytest.fixture
def telescope() -> TelescopeProfile:
    """An 80 mm f/6 refractor."""
    return TelescopeProfile(
        name="Test Refractor 80/480",
        aperture_mm=80.0,
        focal_length_mm=480.0,
        focal_ratio=6.0,
        telescope_type="refractor",
    )


@pytest.fixture
def broadband_filter() -> FilterProfile:
    """A luminance filter."""
    return FilterProfile(
        name="Luminance",
        filter_type="broadband",
        center_nm=None,
        bandwidth_nm=None,
        range_nm=(400.0, 700.0),
        peak_transmission=0.97,
        pass_bands=None,
    )


@pytest.fixture
def narrowband_filter() -> FilterProfile:
    """A 7 nm H-alpha narrowband filter."""
    return FilterProfile(
        name="Ha 7nm",
        filter_type="narrowband",
        center_nm=656.3,
        bandwidth_nm=7.0,
        range_nm=None,
        peak_transmission=0.92,
        pass_bands=None,
    )


@pytest.fixture
def dual_narrowband_filter() -> FilterProfile:
    """A dual-band Ha + OIII filter."""
    return FilterProfile(
        name="L-eXtreme",
        filter_type="dual_narrowband",
        center_nm=None,
        bandwidth_nm=None,
        range_nm=None,
        peak_transmission=0.90,
        pass_bands=[
            {"center_nm": 500.7, "bandwidth_nm": 7.0},
            {"center_nm": 656.3, "bandwidth_nm": 7.0},
        ],
    )


@pytest.fixture
def equipment(mono_camera, telescope, broadband_filter, narrowband_filter) -> EquipmentProfile:
    """A complete equipment profile with mono camera and two filters."""
    return EquipmentProfile(
        camera=mono_camera,
        telescope=telescope,
        filters={
            "L": broadband_filter,
            "Ha": narrowband_filter,
        },
    )


# ---------------------------------------------------------------------------
#  CameraProfile
# ---------------------------------------------------------------------------

class TestCameraProfile:
    def test_creation(self, mono_camera):
        assert mono_camera.name == "TestCam Mono 294"
        assert mono_camera.sensor == "IMX294"
        assert mono_camera.pixel_size_um == 4.63
        assert mono_camera.resolution_x == 4144
        assert mono_camera.resolution_y == 2822
        assert mono_camera.camera_type == "mono"

    def test_is_mono_true(self, mono_camera):
        assert mono_camera.is_mono is True

    def test_is_mono_false(self, color_camera):
        assert color_camera.is_mono is False

    def test_qe_at_exact_point(self, mono_camera):
        """QE at an exact curve point should return that point's QE value."""
        qe = mono_camera.qe_at(550.0)
        assert qe == pytest.approx(0.80, abs=1e-6)

    def test_qe_at_interpolated(self, mono_camera):
        """QE between two curve points should be linearly interpolated."""
        # Between (500, 0.75) and (550, 0.80): midpoint at 525 nm
        qe = mono_camera.qe_at(525.0)
        expected = 0.75 + 0.5 * (0.80 - 0.75)  # 0.775
        assert qe == pytest.approx(expected, abs=1e-6)

    def test_qe_at_below_range(self, mono_camera):
        """QE below the curve's minimum wavelength clamps to the first point."""
        qe = mono_camera.qe_at(300.0)
        assert qe == pytest.approx(0.40, abs=1e-6)  # first point in curve

    def test_qe_at_above_range(self, mono_camera):
        """QE above the curve's maximum wavelength clamps to the last point."""
        qe = mono_camera.qe_at(1000.0)
        assert qe == pytest.approx(0.15, abs=1e-6)  # last point in curve

    def test_qe_at_empty_curve(self):
        """Camera with empty qe_curve falls back to qe_peak."""
        cam = CameraProfile(
            name="Empty",
            sensor="X",
            pixel_size_um=3.0,
            read_noise_e=1.0,
            dark_current_e_per_s=0.001,
            full_well_e=30000,
            qe_peak=0.65,
            qe_curve=[],
            bayer_pattern=None,
            resolution_x=100,
            resolution_y=100,
            camera_type="mono",
        )
        assert cam.qe_at(550.0) == pytest.approx(0.65)

    def test_to_dict_from_dict_round_trip(self, mono_camera):
        d = mono_camera.to_dict()
        restored = CameraProfile.from_dict(d)
        assert restored.name == mono_camera.name
        assert restored.pixel_size_um == mono_camera.pixel_size_um
        assert restored.is_mono == mono_camera.is_mono
        assert restored.resolution_x == mono_camera.resolution_x
        assert restored.resolution_y == mono_camera.resolution_y
        assert len(restored.qe_curve) == len(mono_camera.qe_curve)
        # Verify QE curve values survive the round-trip
        for orig, rest in zip(mono_camera.qe_curve, restored.qe_curve):
            assert rest[0] == pytest.approx(orig[0])
            assert rest[1] == pytest.approx(orig[1])

    def test_to_dict_qe_curve_is_json_serializable(self, mono_camera):
        d = mono_camera.to_dict()
        # qe_curve should be list of lists (not tuples) for JSON
        for point in d["qe_curve"]:
            assert isinstance(point, list)
        # Should not raise
        json.dumps(d)


# ---------------------------------------------------------------------------
#  TelescopeProfile
# ---------------------------------------------------------------------------

class TestTelescopeProfile:
    def test_creation(self, telescope):
        assert telescope.name == "Test Refractor 80/480"
        assert telescope.aperture_mm == 80.0
        assert telescope.focal_length_mm == 480.0
        assert telescope.focal_ratio == 6.0
        assert telescope.telescope_type == "refractor"

    def test_to_dict_from_dict_round_trip(self, telescope):
        d = telescope.to_dict()
        restored = TelescopeProfile.from_dict(d)
        assert restored.name == telescope.name
        assert restored.aperture_mm == telescope.aperture_mm
        assert restored.focal_length_mm == telescope.focal_length_mm
        assert restored.focal_ratio == telescope.focal_ratio
        assert restored.telescope_type == telescope.telescope_type


# ---------------------------------------------------------------------------
#  FilterProfile
# ---------------------------------------------------------------------------

class TestFilterProfile:
    def test_creation_broadband(self, broadband_filter):
        assert broadband_filter.name == "Luminance"
        assert broadband_filter.filter_type == "broadband"
        assert broadband_filter.range_nm == (400.0, 700.0)
        assert broadband_filter.peak_transmission == 0.97

    def test_creation_narrowband(self, narrowband_filter):
        assert narrowband_filter.name == "Ha 7nm"
        assert narrowband_filter.filter_type == "narrowband"
        assert narrowband_filter.center_nm == 656.3
        assert narrowband_filter.bandwidth_nm == 7.0

    def test_effective_bandwidth_broadband(self, broadband_filter):
        """Broadband effective bandwidth is range_nm[1] - range_nm[0]."""
        assert broadband_filter.effective_bandwidth_nm == pytest.approx(300.0)

    def test_effective_bandwidth_narrowband(self, narrowband_filter):
        """Narrowband effective bandwidth is bandwidth_nm."""
        assert narrowband_filter.effective_bandwidth_nm == pytest.approx(7.0)

    def test_effective_bandwidth_dual_narrowband(self, dual_narrowband_filter):
        """Dual narrowband effective bandwidth is the sum of pass_band bandwidths."""
        assert dual_narrowband_filter.effective_bandwidth_nm == pytest.approx(14.0)

    def test_effective_bandwidth_no_data(self):
        """Filter with no bandwidth info returns 0."""
        f = FilterProfile(name="Empty", filter_type="broadband")
        assert f.effective_bandwidth_nm == pytest.approx(0.0)

    def test_to_dict_from_dict_round_trip_broadband(self, broadband_filter):
        d = broadband_filter.to_dict()
        restored = FilterProfile.from_dict(d)
        assert restored.name == broadband_filter.name
        assert restored.filter_type == broadband_filter.filter_type
        assert restored.range_nm == broadband_filter.range_nm
        assert restored.peak_transmission == broadband_filter.peak_transmission

    def test_to_dict_from_dict_round_trip_narrowband(self, narrowband_filter):
        d = narrowband_filter.to_dict()
        restored = FilterProfile.from_dict(d)
        assert restored.center_nm == narrowband_filter.center_nm
        assert restored.bandwidth_nm == narrowband_filter.bandwidth_nm

    def test_to_dict_from_dict_round_trip_dual(self, dual_narrowband_filter):
        d = dual_narrowband_filter.to_dict()
        restored = FilterProfile.from_dict(d)
        assert restored.pass_bands is not None
        assert len(restored.pass_bands) == 2
        assert restored.pass_bands[0]["center_nm"] == pytest.approx(500.7)
        assert restored.pass_bands[1]["bandwidth_nm"] == pytest.approx(7.0)

    def test_to_dict_range_nm_is_json_serializable(self, broadband_filter):
        d = broadband_filter.to_dict()
        # range_nm should be a list, not a tuple, for JSON
        assert isinstance(d["range_nm"], list)
        json.dumps(d)


# ---------------------------------------------------------------------------
#  EquipmentProfile
# ---------------------------------------------------------------------------

class TestEquipmentProfile:
    def test_plate_scale(self, equipment):
        """plate_scale = 206.265 * pixel_size_um / focal_length_mm."""
        expected = 206.265 * 4.63 / 480.0
        assert equipment.plate_scale() == pytest.approx(expected, rel=1e-6)

    def test_theoretical_resolution(self, equipment):
        """Rayleigh criterion: theta_arcsec = 1.22 * lambda_m / D_m * 206265."""
        wl = 550.0
        wavelength_m = wl * 1e-9
        aperture_m = 80.0 * 1e-3
        expected = 1.22 * wavelength_m / aperture_m * 206265.0
        result = equipment.theoretical_resolution(wl)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_theoretical_resolution_default_wavelength(self, equipment):
        """Default wavelength is 550 nm."""
        default_result = equipment.theoretical_resolution()
        explicit_result = equipment.theoretical_resolution(550.0)
        assert default_result == pytest.approx(explicit_result)

    def test_is_oversampled_with_measured_fwhm(self, equipment):
        """Over-sampled when plate_scale < fwhm / 2."""
        ps = equipment.plate_scale()  # ~1.99 arcsec/px
        # A seeing of 5 arcsec means FWHM/2 = 2.5 -- plate scale 1.99 < 2.5 -> oversampled
        assert equipment.is_oversampled(5.0) is True
        # A seeing of 2 arcsec means FWHM/2 = 1.0 -- plate scale 1.99 > 1.0 -> not oversampled
        assert equipment.is_oversampled(2.0) is False

    def test_is_oversampled_without_fwhm(self, equipment):
        """When no FWHM given, uses theoretical resolution."""
        result = equipment.is_oversampled()
        # Theoretical resolution for 80mm @ 550nm is ~1.4 arcsec
        # Plate scale ~1.99 arcsec/px.  1.99 < 1.4/2 = 0.7 -> False
        assert result is False

    def test_effective_throughput(self, equipment):
        """throughput = QE * transmission * bandwidth."""
        # Ha filter: center 656.3nm, bandwidth 7nm, transmission 0.92
        # QE at 656nm ~ 0.65 (from our fixture curve)
        result = equipment.effective_throughput("Ha")
        qe = equipment.camera.qe_at(656.3)
        expected = qe * 0.92 * 7.0
        assert result == pytest.approx(expected, rel=1e-4)

    def test_effective_throughput_missing_filter(self, equipment):
        with pytest.raises(KeyError, match="OIII"):
            equipment.effective_throughput("OIII")

    def test_field_of_view_arcmin(self, equipment):
        """FOV = plate_scale * resolution / 60."""
        ps = equipment.plate_scale()
        expected_x = ps * 4144 / 60.0
        expected_y = ps * 2822 / 60.0
        fov = equipment.field_of_view_arcmin()
        assert fov[0] == pytest.approx(expected_x, rel=1e-6)
        assert fov[1] == pytest.approx(expected_y, rel=1e-6)

    def test_save_load_round_trip(self, equipment, tmp_path):
        """Save to JSON and load back should produce identical data."""
        filepath = tmp_path / "test_profile.json"
        equipment.save(filepath)

        assert filepath.exists()

        loaded = EquipmentProfile.load(filepath)
        assert loaded.camera.name == equipment.camera.name
        assert loaded.camera.pixel_size_um == equipment.camera.pixel_size_um
        assert loaded.telescope.name == equipment.telescope.name
        assert loaded.telescope.focal_length_mm == equipment.telescope.focal_length_mm
        assert set(loaded.filters.keys()) == set(equipment.filters.keys())
        assert loaded.plate_scale() == pytest.approx(equipment.plate_scale())

    def test_save_creates_parent_dirs(self, equipment, tmp_path):
        filepath = tmp_path / "subdir" / "nested" / "profile.json"
        equipment.save(filepath)
        assert filepath.exists()

    def test_save_produces_valid_json(self, equipment, tmp_path):
        filepath = tmp_path / "profile.json"
        equipment.save(filepath)
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert "camera" in data
        assert "telescope" in data
        assert "filters" in data

    def test_to_dict_from_dict_round_trip(self, equipment):
        d = equipment.to_dict()
        restored = EquipmentProfile.from_dict(d)
        assert restored.camera.name == equipment.camera.name
        assert restored.telescope.aperture_mm == equipment.telescope.aperture_mm
        assert len(restored.filters) == len(equipment.filters)
        assert restored.plate_scale() == pytest.approx(equipment.plate_scale())

    def test_to_dict_structure(self, equipment):
        d = equipment.to_dict()
        assert isinstance(d["camera"], dict)
        assert isinstance(d["telescope"], dict)
        assert isinstance(d["filters"], dict)
        assert "L" in d["filters"]
        assert "Ha" in d["filters"]


# ---------------------------------------------------------------------------
#  Database loaders
# ---------------------------------------------------------------------------

class TestDatabaseLoaders:
    def test_load_camera_database(self):
        cameras = load_camera_database()
        assert isinstance(cameras, list)
        assert len(cameras) > 0
        for cam in cameras:
            assert isinstance(cam, CameraProfile)
            assert cam.name
            assert cam.pixel_size_um > 0

    def test_load_telescope_database(self):
        telescopes = load_telescope_database()
        assert isinstance(telescopes, list)
        assert len(telescopes) > 0
        for scope in telescopes:
            assert isinstance(scope, TelescopeProfile)
            assert scope.name
            assert scope.aperture_mm > 0
            assert scope.focal_length_mm > 0

    def test_load_filter_database(self):
        filters = load_filter_database()
        assert isinstance(filters, list)
        assert len(filters) > 0
        for f in filters:
            assert isinstance(f, FilterProfile)
            assert f.name
            assert f.filter_type in (
                "broadband",
                "narrowband",
                "dual_narrowband",
                "triple_narrowband",
                "light_pollution",
                "lp_broadband",
                "lp_narrowband",
                "none",
            )


# ---------------------------------------------------------------------------
#  FITS header detection
# ---------------------------------------------------------------------------

class TestDetectFromFitsHeader:
    def test_typical_header(self):
        header = {
            "INSTRUME": "ZWO ASI294MM Pro",
            "EXPTIME": 300.0,
            "GAIN": 120,
            "OFFSET": 30,
            "FILTER": "Ha",
            "CCD-TEMP": -10.0,
            "BAYERPAT": "",
            "XBINNING": 1,
            "YBINNING": 1,
            "FOCALLEN": 480.0,
            "APTDIA": 80.0,
            "XPIXSZ": 4.63,
            "IMAGETYP": "Light",
            "OBJECT": "M42",
            "RA": 83.8221,
            "DEC": -5.3911,
        }
        info = detect_from_fits_header(header)
        assert info["camera_name"] == "ZWO ASI294MM Pro"
        assert info["exposure_s"] == pytest.approx(300.0)
        assert info["gain"] == pytest.approx(120.0)
        assert info["offset"] == pytest.approx(30.0)
        assert info["filter"] == "Ha"
        assert info["temperature_c"] == pytest.approx(-10.0)
        assert info["binning"] == (1, 1)
        assert info["focal_length_mm"] == pytest.approx(480.0)
        assert info["aperture_mm"] == pytest.approx(80.0)
        assert info["pixel_size_um"] == pytest.approx(4.63)
        assert info["image_type"] == "light"
        assert info["object_name"] == "M42"
        assert info["ra"] == pytest.approx(83.8221)
        assert info["dec"] == pytest.approx(-5.3911)

    def test_minimal_header(self):
        """A header with only exposure time."""
        header = {"EXPTIME": 120.0}
        info = detect_from_fits_header(header)
        assert info["exposure_s"] == pytest.approx(120.0)
        assert "camera_name" not in info
        assert "filter" not in info

    def test_empty_header(self):
        info = detect_from_fits_header({})
        assert isinstance(info, dict)
        assert len(info) == 0

    def test_alternative_keywords(self):
        """Some capture software uses alternative FITS keywords."""
        header = {
            "CCD-NAME": "QHY600M",
            "EXPOSURE": 600.0,
            "EGAIN": 1.5,
            "FILTNAME": "OIII",
            "CCDTEMP": -20.0,
        }
        info = detect_from_fits_header(header)
        assert info["camera_name"] == "QHY600M"
        assert info["exposure_s"] == pytest.approx(600.0)
        assert info["gain"] == pytest.approx(1.5)
        assert info["filter"] == "OIII"
        assert info["temperature_c"] == pytest.approx(-20.0)

    def test_bayer_pattern_uppercase(self):
        header = {"BAYERPAT": "rggb"}
        info = detect_from_fits_header(header)
        assert info["bayer_pattern"] == "RGGB"
