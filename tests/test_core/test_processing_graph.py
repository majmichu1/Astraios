"""Tests for the non-destructive linear processing history."""

import numpy as np

from astraios.core.processing_graph import HistoryStep, ProcessingGraph


def _add_fn():
    """A process_fn that adds params['val'] and counts invocations."""
    calls = {"n": 0}

    def fn(tool, params, img):
        calls["n"] += 1
        return img + float(params.get("val", 0.0))

    return fn, calls


def _base(v=0.0):
    g = ProcessingGraph()
    g.set_base(np.full((8, 8), v, dtype=np.float32))
    return g


def test_linear_chaining():
    # base 0 -> +1 -> +2 -> +3 == 6 everywhere (steps chain, not a star)
    g = _base(0.0)
    g.record("t", {"val": 1.0})
    g.record("t", {"val": 2.0})
    g.record("t", {"val": 3.0})
    fn, _ = _add_fn()
    out = g.evaluate(process_fn=fn)
    assert np.allclose(out, 6.0)


def test_evaluate_up_to_middle_stage():
    g = _base(0.0)
    g.record("t", {"val": 1.0})
    g.record("t", {"val": 2.0})
    g.record("t", {"val": 3.0})
    fn, _ = _add_fn()
    assert np.allclose(g.evaluate(up_to=0, process_fn=fn), 1.0)
    assert np.allclose(g.evaluate(up_to=1, process_fn=fn), 3.0)
    assert np.allclose(g.evaluate(up_to=2, process_fn=fn), 6.0)


def test_caching_no_recompute():
    g = _base(0.0)
    g.record("t", {"val": 1.0})
    fn, calls = _add_fn()
    g.evaluate(process_fn=fn)
    g.evaluate(process_fn=fn)
    assert calls["n"] == 1, "second evaluate should hit the cache"


def test_update_params_invalidates():
    g = _base(0.0)
    g.record("t", {"val": 1.0})
    fn, calls = _add_fn()
    g.evaluate(process_fn=fn)
    g.update_params(0, {"val": 5.0})
    out = g.evaluate(process_fn=fn)
    assert calls["n"] == 2
    assert np.allclose(out, 5.0)


def test_direct_param_mutation_detected():
    # Bypassing update_params (direct mutation) is still caught by the key hash.
    g = _base(0.0)
    g.record("t", {"val": 1.0})
    fn, calls = _add_fn()
    g.evaluate(process_fn=fn)
    g.steps[0].params = {"val": 9.0}
    g.evaluate(process_fn=fn)
    assert calls["n"] == 2


def test_toggle_step_recomputes_downstream():
    g = _base(0.0)
    g.record("t", {"val": 1.0})
    g.record("t", {"val": 2.0})
    fn, _ = _add_fn()
    assert np.allclose(g.evaluate(process_fn=fn), 3.0)
    g.set_enabled(0, False)  # disable +1
    assert np.allclose(g.evaluate(process_fn=fn), 2.0)
    g.set_enabled(0, True)
    assert np.allclose(g.evaluate(process_fn=fn), 3.0)


def test_remove_recomputes_and_respects_lock():
    g = _base(0.0)
    g.record("t", {"val": 1.0})
    g.record("t", {"val": 2.0})
    fn, _ = _add_fn()
    g.evaluate(process_fn=fn)
    assert g.remove(0) is True
    assert np.allclose(g.evaluate(process_fn=fn), 2.0)
    # Locked steps cannot be removed.
    g.set_locked(0, True)
    assert g.remove(0) is False


def test_reorder_changes_result_when_order_matters():
    # With a non-commutative op the order matters; reorder must recompute.
    g = _base(1.0)

    def fn(tool, params, img):
        if tool == "add":
            return img + float(params["v"])
        return img * float(params["v"])  # "mul"

    g.record("add", {"v": 1.0})  # (1+1)=2 then *3 = 6
    g.record("mul", {"v": 3.0})
    assert np.allclose(g.evaluate(process_fn=fn), 6.0)
    g.move(1, 0)  # mul first: (1*3)=3 then +1 = 4
    assert np.allclose(g.evaluate(process_fn=fn), 4.0)


def test_evaluate_returns_copy_not_cache_alias():
    g = _base(0.0)
    g.record("t", {"val": 1.0})
    fn, _ = _add_fn()
    out = g.evaluate(process_fn=fn)
    out += 100.0  # mutating the returned array must not corrupt the cache
    out2 = g.evaluate(process_fn=fn)
    assert np.allclose(out2, 1.0)


def test_base_only_returns_base_copy():
    g = _base(0.5)
    out = g.evaluate(process_fn=lambda *a: None)
    assert np.allclose(out, 0.5)
    out += 1.0
    assert np.allclose(g.base_image, 0.5)  # base untouched


def test_replayability_flags():
    g = _base(0.0)
    g.record("denoise", {"strength": 0.5})
    g.record("", {}, display_name="Manual edit")  # display-only
    assert g.steps[0].replayable is True
    assert g.steps[1].replayable is False
    assert g.is_replayable() is False
    assert g.is_replayable(up_to=0) is True


def test_to_pipeline_exports_enabled_replayable_steps():
    g = _base(0.0)
    g.record("denoise", {"strength": 0.5})
    g.record("scnr", {"amount": 0.8})
    g.record("", {}, display_name="display only")
    g.set_enabled(1, False)
    pipe = g.to_pipeline("My History")
    assert pipe.name == "My History"
    assert [s.tool_name for s in pipe.steps] == ["denoise"]  # enabled + replayable only


def test_to_from_dict_roundtrip():
    g = _base(0.0)
    g.record("denoise", {"strength": 0.5}, display_name="Denoise", mask_name="stars")
    g.record("scnr", {"amount": 0.8})
    g.set_enabled(1, False)
    g.set_locked(0, True)
    d = g.to_dict()
    g2 = ProcessingGraph.from_dict(d)
    assert [s.tool_name for s in g2.steps] == ["denoise", "scnr"]
    assert g2.steps[0].mask_name == "stars"
    assert g2.steps[0].locked is True
    assert g2.steps[1].enabled is False


def test_from_dict_migrates_legacy_v1_nodes():
    legacy = {
        "root_id": "base",
        "nodes": {
            "base": {"node_id": "base", "node_type": "BASE", "process_name": "base_image",
                     "params": {}, "enabled": True, "locked": False, "parent_ids": []},
            "curves_1": {"node_id": "curves_1", "node_type": "PROCESS",
                         "process_name": "Curves", "params": {}, "enabled": True,
                         "locked": False, "parent_ids": ["base"]},
        },
    }
    g = ProcessingGraph.from_dict(legacy)
    assert len(g.steps) == 1
    assert g.steps[0].display_name == "Curves"
    assert g.steps[0].replayable is False  # legacy nodes can't be replayed


def test_step_label_fallback():
    assert HistoryStep(tool_name="auto_stretch").label == "Auto Stretch"
    assert HistoryStep(tool_name="", display_name="Custom").label == "Custom"


def test_curves_nested_params_survive_json_and_replay():
    # Curves params are nested (per-channel control points), not a flat dict.
    # They must serialize to JSON and replay identically through the registry.
    import json

    from astraios.core.batch import get_registered_tools
    from astraios.core.curves import CurvePoints, CurvesParams

    tools = get_registered_tools()
    cp = CurvePoints()
    cp.points = [(0.0, 0.0), (0.3, 0.1), (0.7, 0.9), (1.0, 1.0)]
    import dataclasses

    flat = dataclasses.asdict(CurvesParams(master=cp))
    g = _base(0.0)
    g.set_base(np.clip(np.random.default_rng(0).random((3, 24, 24)).astype(np.float32), 0, 1))
    g.record("curves", flat, "Curves")
    g2 = ProcessingGraph.from_dict(json.loads(json.dumps(g.to_dict())))
    g2.set_base(g.base_image.copy())

    def pf(name, params, img):
        return tools[name](img, **(params or {})) if name in tools else img

    out_a = g.evaluate(process_fn=pf)
    out_b = g2.evaluate(process_fn=pf)
    assert out_a is not None and out_b is not None
    assert np.allclose(out_a, out_b)


def test_enum_params_survive_json_roundtrip():
    # Params hold enum members (e.g. DenoiseMethod); they must serialize to JSON
    # and rebuild as the real enum so a saved/reloaded history still replays.
    import json

    from astraios.core.denoise import DenoiseMethod

    g = _base(0.0)
    g.record("denoise", {"method": DenoiseMethod.WAVELET, "strength": 0.5}, "Denoise")
    d = g.to_dict()
    # Must be JSON-serializable (a project save would otherwise crash).
    blob = json.dumps(d)
    g2 = ProcessingGraph.from_dict(json.loads(blob))
    assert g2.steps[0].params["method"] is DenoiseMethod.WAVELET
    assert g2.steps[0].params["strength"] == 0.5
