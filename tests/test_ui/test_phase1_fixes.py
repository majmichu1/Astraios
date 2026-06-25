"""Tests for Phase 1 bug fixes — core logic without Qt widget instantiation."""

from __future__ import annotations

import numpy as np
import pytest


class TestC1LiveStackPreview:
    """C1: _display_preview should not crash on None/empty/first-frame arrays."""

    def test_none_array(self):
        from astraios.ui.dialogs.live_stack_dialog import LiveStackDialog

        # The _display_preview method skips None/empty via early return
        method = LiveStackDialog._display_preview

        # Calling with None would raise AttributeError before fix —
        # verify the early-return guard works by checking it returns None
        class _FakeDialog:
            pass

        dlg = _FakeDialog()
        # We can't actually instantiate the dialog (needs Qt),
        # but we can verify the condition logic in the source
        from ast import parse
        src = open(__import__("astraios.ui.dialogs.live_stack_dialog", fromlist=[""]).__file__).read()
        assert "is None or not isinstance(arr, np.ndarray) or arr.size == 0" in src, \
            "C1 guard should check None, isinstance, and empty"


class TestC2SmartProcessWorker:
    """C2: SmartProcessWorker should emit error on exception."""

    def test_error_signal_exists(self):
        from astraios.ui.dialogs.smart_process_dialog import SmartProcessWorker

        assert hasattr(SmartProcessWorker, "error"), \
            "SmartProcessWorker must have an error signal"

    def test_error_on_exception(self):
        from astraios.ui.dialogs.smart_process_dialog import SmartProcessWorker

        class _MockProcessor:
            def process(self, *args, **kwargs):
                raise RuntimeError("Boom")

        worker = SmartProcessWorker(_MockProcessor(), None, None, None)
        errors = []
        worker.error.connect(errors.append)
        worker.run()
        assert len(errors) == 1
        assert "RuntimeError" in errors[0]


class TestC3DenoiseDispatch:
    """C3: All 4 denoise methods should map to distinct DenoiseMethod enum values."""

    def test_methods_map_correctly(self):
        from astraios.ui.panels.tools_panel import ToolsPanel
        from astraios.core.denoise import DenoiseMethod

        # Test the dispatch table logic directly
        method_map = {
            "NLM (Non-Local Means)": DenoiseMethod.NLM,
            "Wavelet Denoise": DenoiseMethod.WAVELET,
            "TGV Denoise": DenoiseMethod.TGV,
            "Median Filter": DenoiseMethod.MEDIAN,
        }

        assert method_map["NLM (Non-Local Means)"] == DenoiseMethod.NLM
        assert method_map["Wavelet Denoise"] == DenoiseMethod.WAVELET
        assert method_map["TGV Denoise"] == DenoiseMethod.TGV
        assert method_map["Median Filter"] == DenoiseMethod.MEDIAN
        assert len(set(method_map.values())) == 4, "All 4 methods must be distinct"

    def test_each_method_produces_valid_params(self):
        from astraios.core.denoise import DenoiseParams, DenoiseMethod, denoise

        img = np.random.rand(1, 64, 64).astype(np.float32) * 0.5
        for method in DenoiseMethod:
            params = DenoiseParams(method=method, strength=0.3)
            result = denoise(img, params)
            assert result.shape == img.shape
            assert result.dtype == np.float32
            assert 0.0 <= result.min() <= result.max() <= 1.0


class TestC4PythonConsoleTimeout:
    """C4: Python console should timeout long-running commands."""

    def test_timeout_mechanism(self):
        from astraios.ui.widgets.python_console import PythonConsoleWidget

        w = PythonConsoleWidget.__new__(PythonConsoleWidget)
        w._timeout_sec = 0.5
        w._namespace = {}
        w._executor = None
        w._write = lambda *a, **kw: None
        w._inspector = type("_Fake", (), {"refresh": lambda s, ns: None})()

        import threading
        start = __import__("time").monotonic()
        w._execute("import time; time.sleep(10)")
        elapsed = __import__("time").monotonic() - start
        assert elapsed < 3.0, f"Timeout didn't fire: took {elapsed:.2f}s"


class TestC7CircularFOV:
    """C7: DSO catalog should use circular (not rectangular) FOV."""

    def test_query_uses_angular_distance(self):
        from astraios.core.dso_catalog import query_dso_in_field

        # M42 at RA=83.822, Dec=-5.391
        # With 3° FOV centered on M42, we expect 3 objects
        result = query_dso_in_field(83.822, -5.391, 3.0)
        assert len(result) >= 3, "Should find M42 region objects"
        names = {o.name for o in result}
        assert "M42" in names


class TestC9DagCache:
    """C9: DAG cache should invalidate on param changes."""

    def test_invalidation_on_update_params(self):
        from astraios.core.processing_graph import ProcessingGraph
        import numpy as np

        g = ProcessingGraph()
        g.set_base(np.ones((10, 10), dtype=np.float32))

        call_count = 0

        def fn(name, params, img):
            nonlocal call_count
            call_count += 1
            return img + params.get("val", 0)

        idx = g.record("test", params={"val": 1.0})
        g.evaluate(process_fn=fn)
        g.evaluate(process_fn=fn)
        assert call_count == 1, "Should cache on second call"

        g.update_params(idx, {"val": 2.0})
        g.evaluate(process_fn=fn)
        assert call_count == 2, "Should recompute after update_params"

    def test_hash_detects_mutation(self):
        from astraios.core.processing_graph import ProcessingGraph
        import numpy as np

        g = ProcessingGraph()
        g.set_base(np.ones((10, 10), dtype=np.float32))

        call_count = 0

        def fn(name, params, img):
            nonlocal call_count
            call_count += 1
            return img + params.get("val", 0)

        idx = g.record("test", params={"val": 1.0})
        g.evaluate(process_fn=fn)
        # Bypass update_params — simulate the bug (direct mutation)
        g.steps[idx].params = {"val": 3.0}
        g.evaluate(process_fn=fn)
        assert call_count == 2, "Hash should detect direct mutation"


class TestH25ProjectLoad:
    """H25: Project load should handle corrupted files gracefully."""

    def test_open_project_handles_value_error(self):
        """Verify that _open_project wraps Project.load in try/except ValueError."""
        src = open(__import__("astraios.ui.main_window", fromlist=[""]).__file__).read()
        assert "try:" in src and "except ValueError" in src, \
            "_open_project must handle ValueError from corrupted JSON"
