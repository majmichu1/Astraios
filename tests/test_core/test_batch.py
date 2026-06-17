"""Tests for batch processing pipeline."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from astraios.core.batch import (
    BatchResult,
    Pipeline,
    PipelineStep,
    apply_pipeline_to_image,
    batch_process,
    register_tool,
    get_registered_tools,
    _TOOL_REGISTRY,
)


def _mono_image(h=64, w=64, value=0.5):
    return np.full((h, w), value, dtype=np.float32)


class TestPipelineStep:
    def test_defaults(self):
        step = PipelineStep(tool_name="auto_stretch")
        assert step.tool_name == "auto_stretch"
        assert step.params == {}
        assert step.enabled is True

    def test_with_params(self):
        step = PipelineStep(tool_name="denoise", params={"strength": 0.5})
        assert step.params["strength"] == 0.5


class TestPipeline:
    def test_add_step(self):
        p = Pipeline(name="Test")
        step = p.add_step("auto_stretch")
        assert len(p.steps) == 1
        assert step.tool_name == "auto_stretch"

    def test_add_step_with_params(self):
        p = Pipeline(name="Test")
        p.add_step("denoise", {"strength": 0.8})
        assert p.steps[0].params["strength"] == 0.8

    def test_remove_step(self):
        p = Pipeline(name="Test")
        p.add_step("step_a")
        p.add_step("step_b")
        p.add_step("step_c")
        p.remove_step(1)
        assert len(p.steps) == 2
        assert p.steps[0].tool_name == "step_a"
        assert p.steps[1].tool_name == "step_c"

    def test_remove_step_invalid_index(self):
        p = Pipeline(name="Test")
        p.add_step("step_a")
        p.remove_step(5)  # out of range, should be a no-op
        assert len(p.steps) == 1

    def test_move_step(self):
        p = Pipeline(name="Test")
        p.add_step("step_a")
        p.add_step("step_b")
        p.add_step("step_c")
        p.move_step(0, 2)
        assert p.steps[0].tool_name == "step_b"
        assert p.steps[1].tool_name == "step_c"
        assert p.steps[2].tool_name == "step_a"

    def test_move_step_invalid_index(self):
        p = Pipeline(name="Test")
        p.add_step("step_a")
        p.move_step(0, 5)  # out of range, should be a no-op
        assert p.steps[0].tool_name == "step_a"

    def test_to_dict(self):
        p = Pipeline(name="My Pipeline")
        p.add_step("auto_stretch")
        p.add_step("denoise", {"strength": 0.5})
        d = p.to_dict()
        assert d["name"] == "My Pipeline"
        assert len(d["steps"]) == 2
        assert d["steps"][0]["tool_name"] == "auto_stretch"
        assert d["steps"][1]["params"]["strength"] == 0.5

    def test_from_dict(self):
        d = {
            "name": "Loaded Pipeline",
            "steps": [
                {"tool_name": "auto_stretch", "params": {}, "enabled": True},
                {"tool_name": "denoise", "params": {"strength": 0.7}, "enabled": False},
            ],
        }
        p = Pipeline.from_dict(d)
        assert p.name == "Loaded Pipeline"
        assert len(p.steps) == 2
        assert p.steps[1].tool_name == "denoise"
        assert p.steps[1].params["strength"] == 0.7
        assert p.steps[1].enabled is False

    def test_to_dict_from_dict_roundtrip(self):
        p = Pipeline(name="Roundtrip Test")
        p.add_step("auto_stretch")
        p.add_step("scnr", {"amount": 0.8})
        step = p.add_step("denoise", {"strength": 0.3})
        step.enabled = False

        d = p.to_dict()
        p2 = Pipeline.from_dict(d)

        assert p2.name == p.name
        assert len(p2.steps) == len(p.steps)
        for s1, s2 in zip(p.steps, p2.steps):
            assert s1.tool_name == s2.tool_name
            assert s1.params == s2.params
            assert s1.enabled == s2.enabled


class TestToolRegistry:
    def setup_method(self):
        """Save and clear registry before each test."""
        self._saved = dict(_TOOL_REGISTRY)
        _TOOL_REGISTRY.clear()

    def teardown_method(self):
        """Restore registry after each test."""
        _TOOL_REGISTRY.clear()
        _TOOL_REGISTRY.update(self._saved)

    def test_register_tool(self):
        def my_tool(data):
            return data
        register_tool("my_tool", my_tool)
        tools = get_registered_tools()
        assert "my_tool" in tools

    def test_get_registered_tools_returns_copy(self):
        def my_tool(data):
            return data
        register_tool("my_tool", my_tool)
        tools = get_registered_tools()
        tools["extra"] = lambda d: d
        # Should not affect the internal registry
        assert "extra" not in get_registered_tools()


class TestApplyPipelineToImage:
    def setup_method(self):
        self._saved = dict(_TOOL_REGISTRY)
        _TOOL_REGISTRY.clear()

    def teardown_method(self):
        _TOOL_REGISTRY.clear()
        _TOOL_REGISTRY.update(self._saved)

    def test_empty_pipeline(self):
        data = _mono_image()
        p = Pipeline(name="Empty")
        # Register at least one tool so _register_default_tools is not called
        register_tool("dummy", lambda d: d)
        result = apply_pipeline_to_image(data, p)
        np.testing.assert_array_equal(result, data)

    def test_single_step(self):
        """A simple tool that doubles pixel values (clamped)."""
        def double_it(data):
            return np.clip(data * 2, 0, 1).astype(np.float32)

        register_tool("double", double_it)

        data = _mono_image(value=0.3)
        p = Pipeline(name="Double")
        p.add_step("double")

        result = apply_pipeline_to_image(data, p)
        expected = np.clip(data * 2, 0, 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_disabled_step_skipped(self):
        call_count = {"n": 0}
        def counting_tool(data):
            call_count["n"] += 1
            return data
        register_tool("counter", counting_tool)

        p = Pipeline(name="Test")
        step = p.add_step("counter")
        step.enabled = False

        data = _mono_image()
        apply_pipeline_to_image(data, p)
        assert call_count["n"] == 0

    def test_unknown_tool_skipped(self):
        register_tool("known", lambda d: d)
        p = Pipeline(name="Test")
        p.add_step("nonexistent_tool")
        data = _mono_image()
        # Should not raise, just skip the unknown tool
        result = apply_pipeline_to_image(data, p)
        np.testing.assert_array_equal(result, data)

    def test_multiple_steps_applied_in_order(self):
        """Steps should be applied sequentially."""
        def add_01(data):
            return np.clip(data + 0.1, 0, 1).astype(np.float32)

        def multiply_2(data):
            return np.clip(data * 2, 0, 1).astype(np.float32)

        register_tool("add_01", add_01)
        register_tool("multiply_2", multiply_2)

        data = _mono_image(value=0.2)
        p = Pipeline(name="Multi")
        p.add_step("add_01")
        p.add_step("multiply_2")

        result = apply_pipeline_to_image(data, p)
        # 0.2 + 0.1 = 0.3, then 0.3 * 2 = 0.6
        np.testing.assert_array_almost_equal(result, np.full_like(data, 0.6))

    def test_step_with_params(self):
        def add_value(data, amount=0.0):
            return np.clip(data + amount, 0, 1).astype(np.float32)

        register_tool("add_value", add_value)

        data = _mono_image(value=0.3)
        p = Pipeline(name="Params")
        p.add_step("add_value", {"amount": 0.15})

        result = apply_pipeline_to_image(data, p)
        np.testing.assert_array_almost_equal(result, np.full_like(data, 0.45))


class TestBatchProcess:
    def setup_method(self):
        self._saved = dict(_TOOL_REGISTRY)
        _TOOL_REGISTRY.clear()

    def teardown_method(self):
        _TOOL_REGISTRY.clear()
        _TOOL_REGISTRY.update(self._saved)

    def test_batch_process_with_temp_files(self, tmp_path):
        """End-to-end batch processing with real temp files."""
        from astraios.core.image_io import ImageData, save_fits

        # Register a simple tool
        def invert(data):
            return (1.0 - data).astype(np.float32)
        register_tool("invert", invert)

        # Create temp input files
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        output_dir = tmp_path / "outputs"

        input_paths = []
        for i in range(3):
            data = _mono_image(value=0.3 + i * 0.1)
            path = input_dir / f"image_{i}.fits"
            save_fits(ImageData(data=data, header={}), path)
            input_paths.append(path)

        pipeline = Pipeline(name="Batch Test")
        pipeline.add_step("invert")

        result = batch_process(
            input_paths=input_paths,
            pipeline=pipeline,
            output_dir=output_dir,
        )

        assert result.n_processed == 3
        assert result.n_failed == 0
        assert len(result.output_paths) == 3
        assert len(result.errors) == 0
        # Output files should exist
        for p in result.output_paths:
            assert p.exists()

    def test_batch_process_handles_errors(self, tmp_path):
        """Batch should continue when individual files fail."""
        register_tool("dummy", lambda d: d)

        pipeline = Pipeline(name="Error Test")
        pipeline.add_step("dummy")

        # Use non-existent input files
        input_paths = [
            tmp_path / "nonexistent_1.fits",
            tmp_path / "nonexistent_2.fits",
        ]
        output_dir = tmp_path / "outputs"

        result = batch_process(
            input_paths=input_paths,
            pipeline=pipeline,
            output_dir=output_dir,
        )

        assert result.n_processed == 0
        assert result.n_failed == 2
        assert len(result.errors) == 2

    def test_batch_process_creates_output_dir(self, tmp_path):
        """Output directory should be created if it does not exist."""
        register_tool("dummy", lambda d: d)

        output_dir = tmp_path / "deeply" / "nested" / "output"
        assert not output_dir.exists()

        pipeline = Pipeline(name="Dir Test")
        result = batch_process(
            input_paths=[],
            pipeline=pipeline,
            output_dir=output_dir,
        )
        assert output_dir.exists()

    def test_batch_process_progress_callback(self, tmp_path):
        register_tool("dummy", lambda d: d)

        output_dir = tmp_path / "outputs"
        pipeline = Pipeline(name="Progress Test")
        calls = []
        def progress(frac, msg):
            calls.append((frac, msg))

        batch_process(
            input_paths=[],
            pipeline=pipeline,
            output_dir=output_dir,
            progress=progress,
        )
        # Final progress call should be at 1.0
        assert any(frac == 1.0 for frac, _ in calls)


class TestNewToolsRegistered:
    """The feature tools added for SASpro parity must be batch/macro/scriptable."""

    def _color(self, h=48, w=48):
        rng = np.random.default_rng(0)
        return np.clip(rng.random((3, h, w)) * 0.4 + 0.1, 0, 1).astype(np.float32)

    def test_new_tools_in_registry(self):
        from astraios.core.batch import _register_default_tools

        _TOOL_REGISTRY.clear()
        _register_default_tools()
        tools = get_registered_tools()
        for name in (
            "arcsinh_stretch",
            "statistical_stretch",
            "star_stretch",
            "frequency_separation",
        ):
            assert name in tools, f"{name} not registered"

    def test_pipeline_runs_new_tools(self):
        img = self._color()
        p = Pipeline(name="features")
        p.add_step("statistical_stretch", {"target_median": 0.3})
        p.add_step("star_stretch", {"amount": 0.3, "color_boost": 1.2})
        p.add_step("arcsinh_stretch", {"stretch_factor": 8.0})
        out = apply_pipeline_to_image(img, p)
        assert out.shape == img.shape
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_frequency_separation_enum_param_from_string(self):
        # JSON-roundtripped macros pass the enum as a plain string.
        img = self._color()
        p = Pipeline(name="fs")
        p.add_step("frequency_separation", {"method": "DIVIDE", "sigma": 4.0, "hf_boost": 1.5})
        out = apply_pipeline_to_image(img, p)
        assert out.shape == img.shape
        assert out.min() >= 0.0 and out.max() <= 1.0


class TestMacroRoundTrip:
    """Interactive record -> save -> replay must work for the new tools."""

    def test_recorded_macro_replays(self):
        from astraios.core.scripting import MacroRecorder, play_macro

        rec = MacroRecorder()
        rec.start("feature macro")
        rec.record_step("statistical_stretch", {"target_median": 0.25, "linked": True})
        rec.record_step("frequency_separation", {"method": "SUBTRACT", "sigma": 4.0, "hf_boost": 1.3})
        rec.record_step("star_stretch", {"amount": 0.3, "color_boost": 1.2})
        macro = rec.stop()
        assert [s.tool_name for s in macro.steps] == [
            "statistical_stretch", "frequency_separation", "star_stretch",
        ]

        img = np.clip(np.random.default_rng(0).random((3, 48, 48)) * 0.4 + 0.1, 0, 1).astype(np.float32)
        out = play_macro(img, macro)
        assert out.shape == img.shape
        assert out.min() >= 0.0 and out.max() <= 1.0


class TestMorphologyToolRegistration:
    """Regression: the registered morphology tool used a default element 'DISK'
    that isn't a StructuringElement member (CIRCLE/SQUARE/DIAMOND) -> it raised
    KeyError on every call via batch/macro/scripting/graph."""

    def test_morphology_runs_with_defaults(self):
        from astraios.core.batch import get_registered_tools

        tool = get_registered_tools()["morphology"]
        img = np.random.default_rng(0).random((32, 32)).astype(np.float32)
        out = tool(img)
        assert out.shape == img.shape
        assert np.all(np.isfinite(out))

    def test_morphology_accepts_disk_alias_and_lowercase(self):
        from astraios.core.batch import get_registered_tools

        tool = get_registered_tools()["morphology"]
        img = np.random.default_rng(0).random((32, 32)).astype(np.float32)
        # 'DISK' alias and lowercase operation must not raise.
        assert tool(img, element="DISK", operation="dilate").shape == img.shape
        assert tool(img, element="square", operation="OPEN").shape == img.shape
