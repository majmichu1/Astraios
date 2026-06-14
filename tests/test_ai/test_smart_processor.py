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
