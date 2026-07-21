"""Tests for ModelManager — model caching, loading, and lifecycle."""

import pytest
import torch

from astraios.ai.model_manager import MODEL_REGISTRY, ModelInfo, ModelManager, ModelType
from astraios.ai.models.denoise_model import DenoiseUNet
from astraios.ai.models.sharpen_model import SharpenUNet


class TestModelType:
    def test_enum_members(self):
        assert ModelType.DENOISE is not None
        assert ModelType.SHARPEN is not None

    def test_enum_values_are_distinct(self):
        assert ModelType.DENOISE != ModelType.SHARPEN


class TestModelInfo:
    def test_dataclass_fields(self):
        info = ModelInfo(
            model_type=ModelType.DENOISE,
            filename="test.pt",
            version="1.0.0",
            sha256="abc123",
            size_bytes=1024,
            description="Test model",
        )
        assert info.model_type == ModelType.DENOISE
        assert info.filename == "test.pt"
        assert info.version == "1.0.0"
        assert info.sha256 == "abc123"
        assert info.size_bytes == 1024
        assert info.description == "Test model"


class TestModelManager:
    @pytest.fixture(autouse=True)
    def _isolate_from_bundled_models(self, tmp_path, monkeypatch):
        """Point the bundled-models directory at an empty temp dir.

        These tests describe the USER CACHE lifecycle (missing -> download ->
        delete). Astraios now ships a denoise model inside the package, and
        `get_model_path`/`is_available` correctly prefer that copy, so without
        this isolation the tests would be asserting against the real shipped
        file instead of the cache they set up.
        """
        import astraios.ai.model_manager as mm

        empty = tmp_path / "_no_bundled_models"
        empty.mkdir()
        monkeypatch.setattr(mm, "_BUNDLED_MODELS_DIR", empty)

    def test_creation_creates_directory(self, tmp_path):
        models_dir = tmp_path / "models"
        assert not models_dir.exists()
        manager = ModelManager(models_dir=models_dir)
        assert models_dir.exists()
        assert models_dir.is_dir()

    def test_models_dir_property(self, tmp_path):
        models_dir = tmp_path / "models"
        manager = ModelManager(models_dir=models_dir)
        assert manager.models_dir == models_dir

    def test_get_model_path_denoise(self, tmp_path):
        manager = ModelManager(models_dir=tmp_path)
        path = manager.get_model_path(ModelType.DENOISE)
        expected_filename = MODEL_REGISTRY[ModelType.DENOISE].filename
        assert path == tmp_path / expected_filename

    def test_get_model_path_sharpen(self, tmp_path):
        manager = ModelManager(models_dir=tmp_path)
        path = manager.get_model_path(ModelType.SHARPEN)
        expected_filename = MODEL_REGISTRY[ModelType.SHARPEN].filename
        assert path == tmp_path / expected_filename

    def test_is_available_returns_false_for_missing_model(self, tmp_path):
        manager = ModelManager(models_dir=tmp_path)
        assert manager.is_available(ModelType.DENOISE) is False
        assert manager.is_available(ModelType.SHARPEN) is False

    def test_needs_download_returns_true_for_missing_model(self, tmp_path):
        manager = ModelManager(models_dir=tmp_path)
        assert manager.needs_download(ModelType.DENOISE) is True
        assert manager.needs_download(ModelType.SHARPEN) is True

    def test_create_model_denoise(self, tmp_path):
        manager = ModelManager(models_dir=tmp_path)
        model = manager._create_model(ModelType.DENOISE)
        assert isinstance(model, DenoiseUNet)

    def test_create_model_sharpen(self, tmp_path):
        manager = ModelManager(models_dir=tmp_path)
        model = manager._create_model(ModelType.SHARPEN)
        assert isinstance(model, SharpenUNet)

    def test_list_models_returns_all_registered(self, tmp_path):
        manager = ModelManager(models_dir=tmp_path)
        models = manager.list_models()
        type_names = {m["type"] for m in models}
        assert "DENOISE" in type_names
        assert "SHARPEN" in type_names
        assert len(models) == len(MODEL_REGISTRY)

    def test_list_models_shows_not_available(self, tmp_path):
        manager = ModelManager(models_dir=tmp_path)
        models = manager.list_models()
        for m in models:
            assert m["available"] is False

    def test_list_models_has_expected_keys(self, tmp_path):
        manager = ModelManager(models_dir=tmp_path)
        models = manager.list_models()
        expected_keys = {"type", "description", "version", "filename", "available", "size_bytes"}
        for m in models:
            assert set(m.keys()) == expected_keys

    def test_get_cache_size_empty(self, tmp_path):
        manager = ModelManager(models_dir=tmp_path)
        assert manager.get_cache_size() == 0

    def test_save_and_load_model_denoise(self, tmp_path):
        """Save a dummy denoise model, then load it via ModelManager."""
        manager = ModelManager(models_dir=tmp_path)

        # Create a small model and save its state dict
        from astraios.ai.models.denoise_model import create_denoise_model

        original = create_denoise_model(
            in_channels=1, base_features=32, depth=4, use_noise_conditioning=True
        )
        model_path = manager.get_model_path(ModelType.DENOISE)
        torch.save(original.state_dict(), model_path)

        # Load via manager
        loaded = manager.load_model(ModelType.DENOISE, device="cpu")
        assert isinstance(loaded, DenoiseUNet)

        # Verify it is in eval mode
        assert not loaded.training

        # Verify weights match
        x = torch.rand(1, 1, 64, 64)
        original.eval()
        with torch.no_grad():
            out_orig = original(x)
            out_loaded = loaded(x)
        torch.testing.assert_close(out_orig, out_loaded)

    def test_save_and_load_model_sharpen(self, tmp_path):
        """Save a dummy sharpen model, then load it via ModelManager."""
        manager = ModelManager(models_dir=tmp_path)

        from astraios.ai.models.sharpen_model import create_sharpen_model

        original = create_sharpen_model(
            in_channels=1, base_features=32, depth=4, use_psf_conditioning=True
        )
        model_path = manager.get_model_path(ModelType.SHARPEN)
        torch.save(original.state_dict(), model_path)

        loaded = manager.load_model(ModelType.SHARPEN, device="cpu")
        assert isinstance(loaded, SharpenUNet)
        assert not loaded.training

        x = torch.rand(1, 1, 64, 64)
        original.eval()
        with torch.no_grad():
            out_orig = original(x)
            out_loaded = loaded(x)
        torch.testing.assert_close(out_orig, out_loaded)

    def test_load_model_raises_when_missing(self, tmp_path):
        manager = ModelManager(models_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            manager.load_model(ModelType.DENOISE)

    def test_delete_model(self, tmp_path):
        """Saving then deleting a model should remove the file."""
        manager = ModelManager(models_dir=tmp_path)

        from astraios.ai.models.denoise_model import create_denoise_model

        model = create_denoise_model(
            in_channels=1, base_features=32, depth=4
        )
        model_path = manager.get_model_path(ModelType.DENOISE)
        torch.save(model.state_dict(), model_path)
        assert model_path.exists()

        manager.delete_model(ModelType.DENOISE)
        assert not model_path.exists()

    def test_delete_model_nonexistent_does_not_raise(self, tmp_path):
        """Deleting a model that doesn't exist should not raise."""
        manager = ModelManager(models_dir=tmp_path)
        manager.delete_model(ModelType.DENOISE)  # Should not raise

    def test_get_cache_size_after_save(self, tmp_path):
        """Cache size should reflect saved model files."""
        manager = ModelManager(models_dir=tmp_path)

        from astraios.ai.models.denoise_model import create_denoise_model

        model = create_denoise_model(
            in_channels=1, base_features=32, depth=4
        )
        model_path = manager.get_model_path(ModelType.DENOISE)
        torch.save(model.state_dict(), model_path)

        size = manager.get_cache_size()
        assert size > 0
        assert size == model_path.stat().st_size


class TestBundledModelsAreProtected:
    """`delete_model` used to resolve through `get_model_path`, which prefers
    the bundled copy, so deleting a "cached" model removed the model shipped
    inside the installed package. It really happened: the first test run after
    a model was bundled silently deleted it from the working tree.
    """

    def test_delete_model_never_touches_the_bundled_copy(self, tmp_path, monkeypatch):
        import astraios.ai.model_manager as mm

        bundled_dir = tmp_path / "bundled"
        bundled_dir.mkdir()
        info = MODEL_REGISTRY[ModelType.DENOISE]
        bundled_file = bundled_dir / info.filename
        bundled_file.write_bytes(b"pretend weights")
        monkeypatch.setattr(mm, "_BUNDLED_MODELS_DIR", bundled_dir)

        manager = ModelManager(models_dir=tmp_path / "cache")
        manager.delete_model(ModelType.DENOISE)

        assert bundled_file.exists(), "delete_model removed the bundled model"

    def test_delete_model_still_removes_a_cached_copy(self, tmp_path, monkeypatch):
        import astraios.ai.model_manager as mm

        empty = tmp_path / "no_bundled"
        empty.mkdir()
        monkeypatch.setattr(mm, "_BUNDLED_MODELS_DIR", empty)

        cache = tmp_path / "cache"
        manager = ModelManager(models_dir=cache)
        info = MODEL_REGISTRY[ModelType.DENOISE]
        cached = cache / info.filename
        cached.write_bytes(b"downloaded weights")

        manager.delete_model(ModelType.DENOISE)
        assert not cached.exists(), "a genuinely cached model should still be deletable"

    def test_is_bundled_reports_the_shipped_denoise_model(self):
        # The real package: the denoise model ships with Astraios.
        manager = ModelManager()
        assert manager.is_bundled(ModelType.DENOISE) is True
