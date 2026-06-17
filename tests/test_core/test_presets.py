"""Tests for the preset system."""

import pytest

from astraios.core.stretch import StretchParams
from astraios.core.denoise import DenoiseParams, DenoiseMethod
from astraios.core.presets import (
    delete_preset,
    deserialize_params,
    list_presets,
    load_default_presets,
    load_preset,
    register_tool,
    save_preset,
    serialize_params,
)


@pytest.fixture(autouse=True)
def register_tools():
    """Register tools before each test."""
    load_default_presets()


class TestSerializeParams:
    def test_serialize_simple_params(self):
        params = StretchParams(shadow_clip=-3.0, midtone=0.3, linked=False)
        data = serialize_params(params)
        assert data["shadow_clip"] == -3.0
        assert data["midtone"] == 0.3
        assert data["linked"] is False

    def test_serialize_enum_params(self):
        params = DenoiseParams(
            method=DenoiseMethod.NLM,
            strength=0.7,
            chrominance_only=True,
        )
        data = serialize_params(params)
        assert data["method"]["__enum__"] == "DenoiseMethod"
        assert data["method"]["name"] == "NLM"
        assert data["strength"] == 0.7

    def test_serialize_nested_dataclass(self):
        from astraios.core.curves import CurvesParams, CurvePoints

        params = CurvesParams(
            master=CurvePoints(points=[(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)]),
        )
        data = serialize_params(params)
        # Tuples are serialized with __tuple__ wrapper for round-trip fidelity
        assert len(data["master"]["points"]) == 3
        assert data["master"]["points"][0]["__tuple__"] == [0.0, 0.0]

    def test_roundtrip_nested_dataclass(self):
        from astraios.core.curves import CurvesParams, CurvePoints

        original = CurvesParams(
            master=CurvePoints(points=[(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)]),
            red=CurvePoints(points=[(0.0, 0.0), (1.0, 1.0)]),
            green=CurvePoints(points=[(0.0, 0.0), (1.0, 1.0)]),
            blue=CurvePoints(points=[(0.0, 0.0), (1.0, 1.0)]),
        )
        data = serialize_params(original)
        restored = deserialize_params(CurvesParams, data)
        assert len(restored.master.points) == 3
        assert restored.master.points[0] == (0.0, 0.0)


class TestDeserializeParams:
    def test_deserialize_simple_params(self):
        data = {"shadow_clip": -3.0, "midtone": 0.3, "linked": False, "highlight_clip": 1.0}
        params = deserialize_params(StretchParams, data)
        assert isinstance(params, StretchParams)
        assert params.shadow_clip == -3.0
        assert params.midtone == 0.3
        assert params.linked is False

    def test_deserialize_enum_params(self):
        data = {
            "method": {"__enum__": "DenoiseMethod", "name": "NLM"},
            "strength": 0.7,
            "detail_preservation": 0.5,
            "chrominance_only": True,
            "nlm_h": 10.0,
            "nlm_template_size": 7,
            "nlm_search_size": 21,
            "wavelet": "db4",
            "wavelet_levels": 4,
        }
        params = deserialize_params(DenoiseParams, data)
        assert isinstance(params, DenoiseParams)
        assert params.method == DenoiseMethod.NLM
        assert params.strength == 0.7


class TestSaveLoadPreset:
    def test_save_and_load_preset(self, tmp_path, monkeypatch):
        # Override user preset dir to tmp
        import astraios.core.presets as presets_mod
        monkeypatch.setattr(presets_mod, "get_user_preset_dir", lambda: tmp_path)

        save_preset("stretch", "My Default", StretchParams(midtone=0.3), "My custom stretch")
        result = load_preset("stretch", "My Default")
        assert isinstance(result, StretchParams)
        assert result.midtone == 0.3

    def test_load_nonexistent_preset(self):
        result = load_preset("stretch", "Does Not Exist")
        assert result is None

    def test_save_invalid_tool_raises(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            save_preset("fake_tool", "test", StretchParams())

    def test_delete_preset(self, tmp_path, monkeypatch):
        import astraios.core.presets as presets_mod
        monkeypatch.setattr(presets_mod, "get_user_preset_dir", lambda: tmp_path)

        save_preset("stretch", "To Delete", StretchParams())
        assert delete_preset("stretch", "To Delete") is True
        assert load_preset("stretch", "To Delete") is None

    def test_delete_system_preset_returns_false(self):
        # System presets shouldn't be deletable
        assert delete_preset("stretch", "some_system_preset") is False


class TestListPresets:
    def test_list_all_presets_empty_initially(self):
        # Should return at least empty list, not crash
        presets = list_presets()
        assert isinstance(presets, list)

    def test_list_presets_for_tool(self):
        presets = list_presets("stretch")
        assert isinstance(presets, list)

    def test_list_presets_after_save(self, tmp_path, monkeypatch):
        import astraios.core.presets as presets_mod
        monkeypatch.setattr(presets_mod, "get_user_preset_dir", lambda: tmp_path)

        save_preset("stretch", "List Test", StretchParams(), "Testing list")
        presets = list_presets("stretch")
        names = [p["name"] for p in presets]
        assert "List Test" in names
