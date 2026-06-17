"""Tests for macro recording and playback."""

from pathlib import Path

import numpy as np
import pytest

from astraios.core.batch import Pipeline, PipelineStep, _TOOL_REGISTRY, register_tool
from astraios.core.scripting import MacroRecorder, load_macro, play_macro, save_macro


def _mono_image(h=64, w=64, value=0.5):
    return np.full((h, w), value, dtype=np.float32)


class TestMacroRecorder:
    def test_initial_state(self):
        recorder = MacroRecorder()
        assert recorder.is_recording is False
        assert recorder.step_count == 0

    def test_start_recording(self):
        recorder = MacroRecorder()
        recorder.start("Test Macro")
        assert recorder.is_recording is True
        assert recorder.step_count == 0

    def test_stop_recording(self):
        recorder = MacroRecorder()
        recorder.start("Test Macro")
        recorder.record_step("auto_stretch")
        macro = recorder.stop()
        assert recorder.is_recording is False
        assert isinstance(macro, Pipeline)
        assert macro.name == "Test Macro"
        assert len(macro.steps) == 1

    def test_record_step(self):
        recorder = MacroRecorder()
        recorder.start()
        recorder.record_step("auto_stretch")
        recorder.record_step("denoise", {"strength": 0.5})
        assert recorder.step_count == 2

    def test_record_step_not_recording(self):
        """Steps should be ignored when not recording."""
        recorder = MacroRecorder()
        recorder.record_step("auto_stretch")
        assert recorder.step_count == 0

    def test_record_step_params(self):
        recorder = MacroRecorder()
        recorder.start()
        recorder.record_step("scnr", {"amount": 0.8, "channel": "green"})
        macro = recorder.stop()
        assert macro.steps[0].tool_name == "scnr"
        assert macro.steps[0].params["amount"] == 0.8
        assert macro.steps[0].params["channel"] == "green"

    def test_discard(self):
        recorder = MacroRecorder()
        recorder.start("Will Discard")
        recorder.record_step("auto_stretch")
        recorder.record_step("denoise")
        recorder.discard()
        assert recorder.is_recording is False
        assert recorder.step_count == 0

    def test_start_resets_previous(self):
        """Starting a new recording should clear previous steps."""
        recorder = MacroRecorder()
        recorder.start("First")
        recorder.record_step("step_a")
        recorder.start("Second")
        assert recorder.step_count == 0

    def test_multiple_recordings(self):
        """Can record multiple macros sequentially."""
        recorder = MacroRecorder()

        recorder.start("Macro 1")
        recorder.record_step("step_a")
        macro1 = recorder.stop()

        recorder.start("Macro 2")
        recorder.record_step("step_b")
        recorder.record_step("step_c")
        macro2 = recorder.stop()

        assert len(macro1.steps) == 1
        assert len(macro2.steps) == 2
        assert macro1.name == "Macro 1"
        assert macro2.name == "Macro 2"


class TestSaveLoadMacro:
    def test_save_creates_file(self, tmp_path):
        macro = Pipeline(name="Save Test")
        macro.add_step("auto_stretch")
        path = tmp_path / "macro.json"
        save_macro(macro, path)
        assert path.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        macro = Pipeline(name="Nested")
        path = tmp_path / "deeply" / "nested" / "macro.json"
        save_macro(macro, path)
        assert path.exists()

    def test_load_macro(self, tmp_path):
        macro = Pipeline(name="Load Test")
        macro.add_step("auto_stretch")
        macro.add_step("denoise", {"strength": 0.5})
        path = tmp_path / "macro.json"
        save_macro(macro, path)

        loaded = load_macro(path)
        assert loaded.name == "Load Test"
        assert len(loaded.steps) == 2
        assert loaded.steps[0].tool_name == "auto_stretch"
        assert loaded.steps[1].params["strength"] == 0.5

    def test_save_load_roundtrip(self, tmp_path):
        """Full roundtrip: record -> save -> load -> verify."""
        recorder = MacroRecorder()
        recorder.start("Roundtrip")
        recorder.record_step("auto_stretch")
        recorder.record_step("scnr", {"amount": 0.6})
        recorder.record_step("denoise", {"strength": 0.3})
        macro = recorder.stop()

        path = tmp_path / "roundtrip.json"
        save_macro(macro, path)
        loaded = load_macro(path)

        assert loaded.name == macro.name
        assert len(loaded.steps) == len(macro.steps)
        for s1, s2 in zip(macro.steps, loaded.steps):
            assert s1.tool_name == s2.tool_name
            assert s1.params == s2.params
            assert s1.enabled == s2.enabled

    def test_load_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_macro(tmp_path / "nonexistent.json")


class TestPlayMacro:
    def setup_method(self):
        self._saved = dict(_TOOL_REGISTRY)
        _TOOL_REGISTRY.clear()

    def teardown_method(self):
        _TOOL_REGISTRY.clear()
        _TOOL_REGISTRY.update(self._saved)

    def test_play_empty_macro(self):
        """Empty macro should return a copy of the input."""
        data = _mono_image(value=0.4)
        macro = Pipeline(name="Empty")
        result = play_macro(data, macro)
        np.testing.assert_array_equal(result, data)
        assert result is not data  # should be a copy

    def test_play_macro_applies_steps(self):
        def add_01(data):
            return np.clip(data + 0.1, 0, 1).astype(np.float32)

        register_tool("add_01", add_01)

        data = _mono_image(value=0.3)
        macro = Pipeline(name="Add")
        macro.add_step("add_01")

        result = play_macro(data, macro)
        np.testing.assert_array_almost_equal(result, np.full_like(data, 0.4))

    def test_play_macro_skips_disabled(self):
        call_count = {"n": 0}
        def counting_tool(data):
            call_count["n"] += 1
            return data
        register_tool("counter", counting_tool)

        macro = Pipeline(name="Skip Disabled")
        step = macro.add_step("counter")
        step.enabled = False

        data = _mono_image()
        play_macro(data, macro)
        assert call_count["n"] == 0

    def test_play_macro_progress_callback(self):
        register_tool("noop", lambda d: d)

        data = _mono_image()
        macro = Pipeline(name="Progress")
        macro.add_step("noop")

        calls = []
        def progress(frac, msg):
            calls.append((frac, msg))

        play_macro(data, macro, progress=progress)
        assert len(calls) > 0
        assert any(frac == 1.0 for frac, _ in calls)

    def test_play_recorded_macro(self):
        """Record a macro, then play it back."""
        def invert(data):
            return (1.0 - data).astype(np.float32)

        register_tool("invert", invert)

        recorder = MacroRecorder()
        recorder.start("Invert Macro")
        recorder.record_step("invert")
        macro = recorder.stop()

        data = _mono_image(value=0.3)
        result = play_macro(data, macro)
        np.testing.assert_array_almost_equal(result, np.full_like(data, 0.7))

    def test_play_macro_full_workflow(self, tmp_path):
        """Full workflow: record -> save -> load -> play."""
        def double(data):
            return np.clip(data * 2, 0, 1).astype(np.float32)

        register_tool("double", double)

        # Record
        recorder = MacroRecorder()
        recorder.start("Full Workflow")
        recorder.record_step("double")
        macro = recorder.stop()

        # Save and load
        path = tmp_path / "workflow.json"
        save_macro(macro, path)
        loaded = load_macro(path)

        # Play
        data = _mono_image(value=0.25)
        result = play_macro(data, loaded)
        np.testing.assert_array_almost_equal(result, np.full_like(data, 0.5))
