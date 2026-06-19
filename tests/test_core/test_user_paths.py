"""Tests for user-configured model/tool paths."""

from astraios.core import user_paths


def test_starnet_binary_present(tmp_path, monkeypatch):
    binpath = tmp_path / "StarNetv2CLI"
    binpath.write_text("x")
    monkeypatch.setattr(user_paths, "_get",
                        lambda key: str(binpath) if key == "starnet_path" else None)
    assert user_paths.starnet_binary() == binpath


def test_starnet_binary_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(user_paths, "_get", lambda key: str(tmp_path / "does_not_exist"))
    assert user_paths.starnet_binary() is None


def test_starnet_binary_unset(monkeypatch):
    monkeypatch.setattr(user_paths, "_get", lambda key: None)
    assert user_paths.starnet_binary() is None


def test_model_override(tmp_path, monkeypatch):
    model = tmp_path / "denoise.pt"
    model.write_text("x")
    monkeypatch.setattr(user_paths, "_get",
                        lambda key: str(model) if key == "denoise_model" else None)
    assert user_paths.model_override("denoise") == model
    assert user_paths.model_override("sharpen") is None


def test_cosmic_clarity_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(user_paths, "_get", lambda key: str(tmp_path))
    assert user_paths.cosmic_clarity_dir() == tmp_path
    # a file path is not a directory -> None
    f = tmp_path / "f.pt"
    f.write_text("x")
    monkeypatch.setattr(user_paths, "_get", lambda key: str(f))
    assert user_paths.cosmic_clarity_dir() is None


def test_get_handles_no_qt(monkeypatch):
    # If QSettings/Qt is unavailable, _get must return None, not raise.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "PyQt6.QtCore" or name.startswith("PyQt6"):
            raise ImportError("no Qt")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert user_paths._get("starnet_path") is None
