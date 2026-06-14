"""Tests for the AI Smart Processor."""

import numpy as np
import pytest

from cosmica.ai.smart_processor import (
    ImageAnalysis,
    InputType,
    ProcessingPlan,
    QualityCheck,
    SmartProcessor,
    SmartProcessorResult,
)
from cosmica.core.equipment import (
    CameraProfile,
    EquipmentProfile,
    FilterProfile,
    TelescopeProfile,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _make_mono_image(h=64, w=64, mean=0.10, seed=42):
    """Create a small mono float32 image resembling unstretched astro data."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.01, (h, w)).astype(np.float32)
    image = np.full((h, w), mean, dtype=np.float32) + noise
    return np.clip(image, 0.0, 1.0)


def _make_color_image(h=64, w=64, mean=0.10, seed=42):
    """Create a small (3, H, W) float32 color image."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.01, (3, h, w)).astype(np.float32)
    image = np.full((3, h, w), mean, dtype=np.float32) + noise
    return np.clip(image, 0.0, 1.0)


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mono_camera() -> CameraProfile:
    return CameraProfile(
        name="TestCam Mono",
        sensor="TestSensor",
        pixel_size_um=4.63,
        read_noise_e=1.2,
        dark_current_e_per_s=0.002,
        full_well_e=63000,
        qe_peak=0.80,
        qe_curve=[
            (400.0, 0.40),
            (500.0, 0.75),
            (550.0, 0.80),
            (656.0, 0.65),
            (700.0, 0.45),
        ],
        bayer_pattern=None,
        resolution_x=4144,
        resolution_y=2822,
        camera_type="mono",
    )


@pytest.fixture
def telescope() -> TelescopeProfile:
    return TelescopeProfile(
        name="Test Scope",
        aperture_mm=80.0,
        focal_length_mm=480.0,
        focal_ratio=6.0,
        telescope_type="refractor",
    )


@pytest.fixture
def ha_filter() -> FilterProfile:
    return FilterProfile(
        name="Ha 7nm",
        filter_type="narrowband",
        center_nm=656.3,
        bandwidth_nm=7.0,
        peak_transmission=0.92,
    )


@pytest.fixture
def equipment(mono_camera, telescope, ha_filter) -> EquipmentProfile:
    return EquipmentProfile(
        camera=mono_camera,
        telescope=telescope,
        filters={"Ha": ha_filter},
    )


@pytest.fixture
def processor(equipment) -> SmartProcessor:
    return SmartProcessor(equipment=equipment, catalog=None)


@pytest.fixture
def processor_no_equipment() -> SmartProcessor:
    return SmartProcessor(equipment=None, catalog=None)


# ---------------------------------------------------------------------------
#  SmartProcessor creation
# ---------------------------------------------------------------------------

class TestSmartProcessorCreation:
    def test_create_with_equipment(self, equipment):
        sp = SmartProcessor(equipment=equipment)
        assert sp.equipment is equipment

    def test_create_without_equipment(self):
        sp = SmartProcessor()
        assert sp.equipment is None

    def test_catalog_defaults_when_none(self):
        sp = SmartProcessor(catalog=None)
        assert sp.catalog is not None


# ---------------------------------------------------------------------------
#  Input type detection
# ---------------------------------------------------------------------------

class TestInputTypeDetection:
    def test_mono_2d_detected_as_luminance(self, processor):
        data = _make_mono_image()
        assert data.ndim == 2
        result = processor.process(data, input_type_hint=None)
        assert result.analysis.input_type == InputType.MONO_LUMINANCE

    def test_color_3channel_detected_as_osc_rgb(self, processor_no_equipment):
        data = _make_color_image()
        assert data.shape[0] == 3
        result = processor_no_equipment.process(data, input_type_hint=None)
        assert result.analysis.input_type == InputType.OSC_RGB

    def test_input_type_hint_overrides_detection(self, processor):
        data = _make_color_image()
        result = processor.process(
            data, input_type_hint=InputType.NARROWBAND_SHO,
        )
        assert result.analysis.input_type == InputType.NARROWBAND_SHO


# ---------------------------------------------------------------------------
#  Mono image processing
# ---------------------------------------------------------------------------

class TestProcessMono:
    def test_result_type(self, processor):
        data = _make_mono_image()
        result = processor.process(data, input_type_hint=InputType.MONO_LUMINANCE)
        assert isinstance(result, SmartProcessorResult)

    def test_result_has_analysis(self, processor):
        data = _make_mono_image()
        result = processor.process(data, input_type_hint=InputType.MONO_LUMINANCE)
        assert isinstance(result.analysis, ImageAnalysis)
        assert result.analysis.n_channels == 1
        assert result.analysis.height == 64
        assert result.analysis.width == 64

    def test_result_has_plan(self, processor):
        data = _make_mono_image()
        result = processor.process(data, input_type_hint=InputType.MONO_LUMINANCE)
        assert isinstance(result.plan, ProcessingPlan)
        assert len(result.plan.channel_plans) == 1

    def test_result_image_shape_matches_input(self, processor):
        data = _make_mono_image(h=64, w=64)
        result = processor.process(data, input_type_hint=InputType.MONO_LUMINANCE)
        assert result.image.shape == data.shape

    def test_result_image_values_in_range(self, processor):
        data = _make_mono_image()
        result = processor.process(data, input_type_hint=InputType.MONO_LUMINANCE)
        assert result.image.min() >= 0.0
        assert result.image.max() <= 1.0

    def test_result_image_dtype(self, processor):
        data = _make_mono_image()
        result = processor.process(data, input_type_hint=InputType.MONO_LUMINANCE)
        assert result.image.dtype == np.float32

    def test_quality_checks_populated(self, processor):
        data = _make_mono_image()
        result = processor.process(data, input_type_hint=InputType.MONO_LUMINANCE)
        assert isinstance(result.quality_checks, list)
        assert len(result.quality_checks) > 0
        for qc in result.quality_checks:
            assert isinstance(qc, QualityCheck)

    def test_processing_log_populated(self, processor):
        data = _make_mono_image()
        result = processor.process(data, input_type_hint=InputType.MONO_LUMINANCE)
        assert isinstance(result.processing_log, list)
        assert len(result.processing_log) > 0

    def test_channel_plan_has_core_params(self, processor):
        """Each channel plan should include background, denoise, and stretch params."""
        data = _make_mono_image()
        result = processor.process(data, input_type_hint=InputType.MONO_LUMINANCE)
        cp = result.plan.channel_plans[0]
        assert cp.background_params is not None
        assert cp.denoise_params is not None
        assert cp.stretch_params is not None


# ---------------------------------------------------------------------------
#  Color image processing
# ---------------------------------------------------------------------------

class TestProcessColor:
    def test_result_image_shape_matches_input(self, processor):
        data = _make_color_image(h=64, w=64)
        result = processor.process(data, input_type_hint=InputType.OSC_RGB)
        assert result.image.shape == data.shape

    def test_three_channel_plans(self, processor):
        data = _make_color_image()
        result = processor.process(data, input_type_hint=InputType.OSC_RGB)
        assert len(result.plan.channel_plans) == 3

    def test_channel_plan_names(self, processor):
        data = _make_color_image()
        result = processor.process(data, input_type_hint=InputType.OSC_RGB)
        names = [cp.channel_name for cp in result.plan.channel_plans]
        assert names == ["R", "G", "B"]

    def test_result_values_in_range(self, processor):
        data = _make_color_image()
        result = processor.process(data, input_type_hint=InputType.OSC_RGB)
        assert result.image.min() >= 0.0
        assert result.image.max() <= 1.0

    def test_analysis_channels(self, processor):
        data = _make_color_image()
        result = processor.process(data, input_type_hint=InputType.OSC_RGB)
        assert result.analysis.n_channels == 3

    def test_each_channel_plan_has_core_params(self, processor):
        data = _make_color_image()
        result = processor.process(data, input_type_hint=InputType.OSC_RGB)
        for cp in result.plan.channel_plans:
            assert cp.background_params is not None
            assert cp.denoise_params is not None
            assert cp.stretch_params is not None


# ---------------------------------------------------------------------------
#  Progress callback
# ---------------------------------------------------------------------------

class TestProgressCallback:
    def test_progress_called(self, processor):
        data = _make_mono_image()
        calls = []

        def on_progress(fraction, message):
            calls.append((fraction, message))

        processor.process(
            data,
            input_type_hint=InputType.MONO_LUMINANCE,
            progress=on_progress,
        )
        assert len(calls) > 0

    def test_progress_starts_at_zero(self, processor):
        data = _make_mono_image()
        calls = []

        def on_progress(fraction, message):
            calls.append((fraction, message))

        processor.process(
            data,
            input_type_hint=InputType.MONO_LUMINANCE,
            progress=on_progress,
        )
        assert calls[0][0] == pytest.approx(0.0)

    def test_progress_ends_at_one(self, processor):
        data = _make_mono_image()
        calls = []

        def on_progress(fraction, message):
            calls.append((fraction, message))

        processor.process(
            data,
            input_type_hint=InputType.MONO_LUMINANCE,
            progress=on_progress,
        )
        assert calls[-1][0] == pytest.approx(1.0)

    def test_progress_fractions_non_decreasing(self, processor):
        data = _make_mono_image()
        fractions = []

        def on_progress(fraction, message):
            fractions.append(fraction)

        processor.process(
            data,
            input_type_hint=InputType.MONO_LUMINANCE,
            progress=on_progress,
        )
        for i in range(1, len(fractions)):
            assert fractions[i] >= fractions[i - 1]


# ---------------------------------------------------------------------------
#  Processing without equipment
# ---------------------------------------------------------------------------

class TestProcessWithoutEquipment:
    def test_mono_without_equipment(self, processor_no_equipment):
        data = _make_mono_image()
        result = processor_no_equipment.process(
            data, input_type_hint=InputType.MONO_LUMINANCE,
        )
        assert isinstance(result, SmartProcessorResult)
        assert result.image.shape == data.shape
        assert result.analysis.plate_scale_arcsec is None

    def test_color_without_equipment(self, processor_no_equipment):
        data = _make_color_image()
        result = processor_no_equipment.process(
            data, input_type_hint=InputType.OSC_RGB,
        )
        assert isinstance(result, SmartProcessorResult)
        assert result.image.shape == data.shape


# ---------------------------------------------------------------------------
#  Different image sizes
# ---------------------------------------------------------------------------

class TestImageSizes:
    def test_128x128_mono(self, processor):
        data = _make_mono_image(h=128, w=128)
        result = processor.process(data, input_type_hint=InputType.MONO_LUMINANCE)
        assert result.image.shape == (128, 128)

    def test_64x128_rectangular(self, processor):
        data = _make_mono_image(h=64, w=128)
        result = processor.process(data, input_type_hint=InputType.MONO_LUMINANCE)
        assert result.image.shape == (64, 128)


class TestNarrowbandPalettePreservation:
    """Narrowband channel intensity ratios are intentional (SHO/HOO palettes),
    so the per-channel signal-gain equalization must be skipped for them."""

    @staticmethod
    def _imbalanced_color(h=96, w=96):
        # Ha-dominant: ch0 strong, ch1/ch2 weak — like an SHO stack.
        rng = np.random.default_rng(0)
        img = np.zeros((3, h, w), np.float32)
        img[0] = np.clip(0.05 + rng.random((h, w)) * 0.4, 0, 1)   # strong
        img[1] = np.clip(0.02 + rng.random((h, w)) * 0.08, 0, 1)  # weak
        img[2] = np.clip(0.02 + rng.random((h, w)) * 0.06, 0, 1)  # weak
        return img.astype(np.float32)

    def test_gain_equalization_skipped_for_narrowband(self, processor_no_equipment):
        img = self._imbalanced_color()
        result = processor_no_equipment.process(img, input_type_hint=InputType.NARROWBAND_SHO)
        log = "\n".join(result.processing_log)
        assert "Skipping channel gain equalization" in log
        assert "Color gain correction" not in log

    def test_gain_equalization_applied_for_rgb(self, processor_no_equipment):
        img = self._imbalanced_color()
        result = processor_no_equipment.process(img, input_type_hint=InputType.OSC_RGB)
        log = "\n".join(result.processing_log)
        assert "Skipping channel gain equalization" not in log


class TestStarAwareProcessing:
    """Star-aware mode: separate stars, enhance the starless nebula, screen
    the stars back. On by default; must degrade gracefully and be disable-able."""

    @staticmethod
    def _nebula_with_stars(h=120, w=120):
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[0:h, 0:w]
        img = (0.06 * np.exp(-(((xx - w/2)**2 + (yy - h/2)**2) / (2 * 30**2)))).astype(np.float32)
        img = np.stack([img, img, img])
        img += (np.abs(rng.normal(0, 0.01, (3, h, w))) + 0.02).astype(np.float32)
        for _ in range(25):
            sy, sx = int(rng.integers(6, h - 6)), int(rng.integers(6, w - 6))
            img[:, sy-1:sy+2, sx-1:sx+2] += rng.uniform(0.3, 0.8)
        return np.clip(img, 0, 1).astype(np.float32)

    def test_star_aware_on_by_default(self, processor_no_equipment):
        img = self._nebula_with_stars()
        result = processor_no_equipment.process(img, input_type_hint=InputType.OSC_RGB)
        log = "\n".join(result.processing_log)
        assert result.plan.star_aware is True
        assert "Star separation" in log
        assert "Star-aware complete" in log
        assert result.image.shape == img.shape
        assert result.image.min() >= 0.0 and result.image.max() <= 1.0

    def test_star_aware_can_be_disabled(self, processor_no_equipment):
        img = self._nebula_with_stars()
        result = processor_no_equipment.process(
            img, input_type_hint=InputType.OSC_RGB,
            enabled_stages={"background", "denoise", "stretch", "local_contrast"},
        )
        assert result.plan.star_aware is False
        assert "Star separation" not in "\n".join(result.processing_log)

    def test_graceful_fallback_when_separation_fails(self, processor_no_equipment, monkeypatch):
        # If star removal returns None, the non-star-aware result is kept.
        monkeypatch.setattr(processor_no_equipment, "_separate_stars", lambda img: None)
        img = self._nebula_with_stars()
        result = processor_no_equipment.process(img, input_type_hint=InputType.OSC_RGB)
        assert result.image.shape == img.shape
        assert result.image.min() >= 0.0 and result.image.max() <= 1.0

    def test_separate_stars_uses_builtin_without_starnet(self, processor_no_equipment):
        # No StarNet binary in CI -> built-in morphological remover, valid output.
        img = self._nebula_with_stars()
        starless = processor_no_equipment._separate_stars(img)
        assert starless is not None
        assert starless.shape == img.shape
        # Starless should have lower peak (stars removed) than the original.
        assert float(starless.max()) <= float(img.max()) + 1e-6


class TestStarAwareStarReduction:
    """Star-aware mode can shrink the isolated star layer before recombining."""

    @staticmethod
    def _nebula_with_stars(h=120, w=120):
        rng = np.random.default_rng(1)
        yy, xx = np.mgrid[0:h, 0:w]
        img = (0.06 * np.exp(-(((xx - w/2)**2 + (yy - h/2)**2) / (2 * 30**2)))).astype(np.float32)
        img = np.stack([img, img, img])
        img += (np.abs(rng.normal(0, 0.01, (3, h, w))) + 0.02).astype(np.float32)
        for _ in range(30):
            sy, sx = int(rng.integers(6, h - 6)), int(rng.integers(6, w - 6))
            img[:, sy-2:sy+3, sx-2:sx+3] += rng.uniform(0.4, 0.9)
        return np.clip(img, 0, 1).astype(np.float32)

    def test_star_reduction_applied_by_default(self, processor_no_equipment):
        img = self._nebula_with_stars()
        result = processor_no_equipment.process(img, input_type_hint=InputType.OSC_RGB)
        assert "reduced star sizes" in "\n".join(result.processing_log)
        assert result.image.shape == img.shape

    def test_star_reduction_zero_skips(self, processor_no_equipment):
        img = self._nebula_with_stars()
        result = processor_no_equipment.process(
            img, input_type_hint=InputType.OSC_RGB, star_reduction=0.0,
        )
        assert "reduced star sizes" not in "\n".join(result.processing_log)

    def test_star_reduction_clamped(self, processor_no_equipment):
        img = self._nebula_with_stars()
        # Out-of-range values are clamped, not crash.
        result = processor_no_equipment.process(
            img, input_type_hint=InputType.OSC_RGB, star_reduction=5.0,
        )
        assert result.image.min() >= 0.0 and result.image.max() <= 1.0


def test_smart_dialog_wires_star_reduction(qtbot):
    from cosmica.ui.dialogs.smart_process_dialog import SmartProcessDialog

    dlg = SmartProcessDialog()
    assert dlg._stage_star_aware.isChecked()
    assert abs(dlg._star_reduction_spin.value() - 0.3) < 1e-9
    # Disabling star-aware disables the reduction control.
    dlg._stage_star_aware.setChecked(False)
    assert not dlg._star_reduction_spin.isEnabled()


class TestSPCCColorCalibration:
    """Photometric color calibration (SPCC) when plate-solved + Gaia available,
    with graceful statistical fallback otherwise."""

    @staticmethod
    def _color(h=80, w=80):
        rng = np.random.default_rng(3)
        img = np.clip(rng.random((3, h, w)) * 0.4 + 0.05, 0, 1).astype(np.float32)
        # Give it a colour cast so calibration has something to correct.
        img[0] *= 1.3
        return np.clip(img, 0, 1).astype(np.float32)

    _WCS = {"ra_center": 83.8, "dec_center": -5.39, "scale": 2.0}

    def test_spcc_applied_when_plate_solved_and_catalog(self, processor_no_equipment, monkeypatch):
        from cosmica.core import color_calibration, star_catalog
        from cosmica.core.color_calibration import ColorCalibrationResult

        fake_stars = [
            star_catalog.StarCatalogEntry(83.8 + i * 0.001, -5.39, 12.0, 12.3, 11.7, str(i))
            for i in range(20)
        ]
        monkeypatch.setattr(star_catalog, "query_gaia_dr3", lambda *a, **k: fake_stars)

        def fake_pcc(image, **kw):
            return ColorCalibrationResult(data=image.astype(np.float32),
                                          correction_factors=(1.1, 1.0, 0.9))
        monkeypatch.setattr(color_calibration, "photometric_color_calibrate", fake_pcc)

        result = processor_no_equipment.process(
            self._color(), input_type_hint=InputType.OSC_RGB, wcs_dict=self._WCS,
        )
        log = "\n".join(result.processing_log)
        assert result.plan.color_calibrated is True
        assert "SPCC: photometric calibration" in log
        assert "Skipping channel gain equalization (SPCC" in log

    def test_falls_back_to_statistical_when_no_plate_solve(self, processor_no_equipment):
        # No WCS, plate solve fails on synthetic data -> statistical balance.
        result = processor_no_equipment.process(self._color(), input_type_hint=InputType.OSC_RGB)
        assert result.plan.color_calibrated is False
        assert "Statistical colour balance (no SPCC" in "\n".join(result.processing_log)

    def test_falls_back_when_too_few_catalog_stars(self, processor_no_equipment, monkeypatch):
        from cosmica.core import star_catalog
        monkeypatch.setattr(star_catalog, "query_gaia_dr3", lambda *a, **k: [])
        result = processor_no_equipment.process(
            self._color(), input_type_hint=InputType.OSC_RGB, wcs_dict=self._WCS,
        )
        assert result.plan.color_calibrated is False
        log = "\n".join(result.processing_log)
        assert "Statistical colour balance" in log

    def test_no_spcc_for_mono(self, processor_no_equipment):
        mono = np.clip(np.abs(np.random.default_rng(0).normal(0, 0.05, (64, 64))), 0, 1).astype(np.float32)
        result = processor_no_equipment.process(
            mono, input_type_hint=InputType.MONO_LUMINANCE, wcs_dict=self._WCS,
        )
        assert result.plan.color_calibrated is False


class TestAIDenoise:
    """Noise reduction routes through the AI model (with wavelet fallback)."""

    @staticmethod
    def _mono():
        return np.clip(np.abs(np.random.default_rng(0).normal(0, 0.04, (64, 64))) + 0.05,
                       0, 1).astype(np.float32)

    def test_ai_denoise_routed_by_default(self, processor_no_equipment):
        result = processor_no_equipment.process(
            self._mono(), input_type_hint=InputType.MONO_LUMINANCE,
            enabled_stages={"denoise", "stretch"},
        )
        nr = [ln for ln in result.processing_log if "Noise reduction" in ln]
        assert nr and "(AI)" in nr[0]
        assert result.image.shape == (64, 64)

    def test_wavelet_when_ai_disabled(self, processor_no_equipment):
        result = processor_no_equipment.process(
            self._mono(), input_type_hint=InputType.MONO_LUMINANCE,
            enabled_stages={"denoise", "stretch"}, use_ai_denoise=False,
        )
        nr = [ln for ln in result.processing_log if "Noise reduction" in ln]
        assert nr and "(wavelet)" in nr[0]

    def test_run_denoise_falls_back_on_error(self, processor_no_equipment, monkeypatch):
        # If the AI path raises, it must fall back to wavelet, not crash.
        import cosmica.ai.inference.denoise as aidn
        monkeypatch.setattr(aidn, "ai_denoise", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        from cosmica.core.denoise import DenoiseParams
        out = processor_no_equipment._run_denoise(self._mono(), DenoiseParams())
        assert out.shape == (64, 64)


def test_smart_dialog_wires_ai_denoise(qtbot):
    from cosmica.ui.dialogs.smart_process_dialog import SmartProcessDialog

    dlg = SmartProcessDialog()
    assert dlg._ai_denoise_cb.isChecked()
    dlg._stage_denoise.setChecked(False)
    assert not dlg._ai_denoise_cb.isEnabled()
