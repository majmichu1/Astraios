"""Tests for the general tool-preset save/recall mechanism."""

from __future__ import annotations

import json

import pytest

from astraios.core.masks import MaskType
from astraios.core.tool_presets import (
    delete_preset,
    list_presets,
    load_preset,
    save_preset,
)


@pytest.fixture(autouse=True)
def _isolated_presets_dir(tmp_path, monkeypatch):
    """Redirect PRESETS_DIR to a temp dir so tests never touch ~/.astraios."""
    import astraios.core.tool_presets as tp

    monkeypatch.setattr(tp, "PRESETS_DIR", tmp_path / "presets")
    yield tmp_path


class TestRoundTrip:
    def test_save_and_load_simple_params(self):
        save_preset("my_tool", "default", {"amount": 0.5, "mode": "add"})
        loaded = load_preset("my_tool", "default")
        assert loaded == {"amount": 0.5, "mode": "add"}

    def test_save_and_load_nested_params(self):
        params = {"a": 1, "b": {"c": [1, 2, 3], "d": None}, "e": [True, False]}
        save_preset("my_tool", "nested", params)
        assert load_preset("my_tool", "nested") == params

    def test_save_and_load_enum_param_round_trips_exact_member(self):
        params = {"mask_type": MaskType.LUMINANCE, "radius": 3.0}
        save_preset("mask_tool", "lum", params)
        loaded = load_preset("mask_tool", "lum")
        assert loaded["mask_type"] is MaskType.LUMINANCE
        assert loaded["radius"] == 3.0

    def test_overwrite_existing_preset(self):
        save_preset("my_tool", "p", {"x": 1})
        save_preset("my_tool", "p", {"x": 2})
        assert load_preset("my_tool", "p") == {"x": 2}

    def test_multiple_presets_independent(self):
        save_preset("my_tool", "a", {"x": 1})
        save_preset("my_tool", "b", {"x": 2})
        assert load_preset("my_tool", "a") == {"x": 1}
        assert load_preset("my_tool", "b") == {"x": 2}

    def test_presets_isolated_per_tool(self):
        save_preset("tool_one", "p", {"x": 1})
        save_preset("tool_two", "p", {"x": 2})
        assert load_preset("tool_one", "p") == {"x": 1}
        assert load_preset("tool_two", "p") == {"x": 2}

    def test_on_disk_json_is_human_readable_and_enum_tagged(self, _isolated_presets_dir):
        save_preset("mask_tool", "lum", {"mask_type": MaskType.LUMINANCE})
        path = _isolated_presets_dir / "presets" / "mask_tool.json"
        assert path.exists()
        raw = json.loads(path.read_text())
        assert "__enum__" in raw["lum"]["mask_type"]
        assert raw["lum"]["mask_type"]["name"] == "LUMINANCE"


class TestListing:
    def test_list_presets_empty_when_no_file(self):
        assert list_presets("nonexistent_tool") == []

    def test_list_presets_returns_sorted_names(self):
        save_preset("my_tool", "zeta", {"x": 1})
        save_preset("my_tool", "alpha", {"x": 2})
        save_preset("my_tool", "mid", {"x": 3})
        assert list_presets("my_tool") == ["alpha", "mid", "zeta"]


class TestDeletion:
    def test_delete_removes_preset(self):
        save_preset("my_tool", "gone", {"x": 1})
        delete_preset("my_tool", "gone")
        assert list_presets("my_tool") == []
        with pytest.raises(KeyError):
            load_preset("my_tool", "gone")

    def test_delete_missing_preset_is_noop(self):
        # No file at all yet.
        delete_preset("my_tool", "never_existed")
        save_preset("my_tool", "keep", {"x": 1})
        # Deleting something that still doesn't exist shouldn't disturb "keep".
        delete_preset("my_tool", "also_never_existed")
        assert list_presets("my_tool") == ["keep"]

    def test_delete_one_preset_keeps_others(self):
        save_preset("my_tool", "a", {"x": 1})
        save_preset("my_tool", "b", {"x": 2})
        delete_preset("my_tool", "a")
        assert list_presets("my_tool") == ["b"]


class TestMissingFileBehavior:
    def test_load_missing_tool_raises_keyerror(self):
        with pytest.raises(KeyError):
            load_preset("no_such_tool", "no_such_preset")

    def test_load_missing_preset_name_raises_keyerror(self):
        save_preset("my_tool", "exists", {"x": 1})
        with pytest.raises(KeyError):
            load_preset("my_tool", "does_not_exist")


class TestValidation:
    def test_save_empty_preset_name_raises(self):
        with pytest.raises(ValueError):
            save_preset("my_tool", "", {"x": 1})

    def test_save_empty_tool_name_raises(self):
        with pytest.raises(ValueError):
            save_preset("", "preset", {"x": 1})

    def test_tool_name_with_unsafe_characters_is_sanitized(self, _isolated_presets_dir):
        save_preset("weird/tool:name", "p", {"x": 1})
        assert load_preset("weird/tool:name", "p") == {"x": 1}
        # Exactly one preset file should have been created (sanitized name).
        files = list((_isolated_presets_dir / "presets").glob("*.json"))
        assert len(files) == 1
