"""Model Manager — download, cache, and load AI model weights.

Handles first-use downloads from a CDN, local caching in the user's data
directory, integrity verification via SHA-256, and version management.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable

import torch
from platformdirs import user_data_dir

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


# Directory where bundled models ship with the package
_BUNDLED_MODELS_DIR = Path(__file__).parent / "models"

# Base URL for model downloads (used for large models not bundled in the package)
MODEL_CDN_BASE = "https://models.cosmica.app/v1"


class ModelType(Enum):
    DENOISE = auto()
    SHARPEN = auto()


@dataclass
class ModelInfo:
    """Metadata for a downloadable model."""

    model_type: ModelType
    filename: str
    version: str
    sha256: str
    size_bytes: int
    description: str


# Model registry — updated when new model versions are released
MODEL_REGISTRY: dict[ModelType, ModelInfo] = {
    ModelType.DENOISE: ModelInfo(
        model_type=ModelType.DENOISE,
        filename="cosmica_denoise_v1.pt",
        version="1.0.0",
        # Bundled with the package (see _BUNDLED_MODELS_DIR), so no download is
        # needed. Hash and size describe the shipped file; the previous values
        # described a model that was never published anywhere.
        sha256="a0a258d297674eed1d12783e728b2ea213304023cf776cd452742d708981c3bd",
        size_bytes=31114023,
        description="Astraios AI Denoiser v1 — Noise2Self U-Net for astronomical images",
    ),
    ModelType.SHARPEN: ModelInfo(
        model_type=ModelType.SHARPEN,
        filename="cosmica_sharpen_v1.pt",
        version="1.0.0",
        sha256="",  # populated after training
        size_bytes=0,
        description="U-Net sharpener for astronomical images",
    ),
}


class ModelManager:
    """Manages AI model downloads, caching, and loading.

    Models are stored in the user's data directory under ``models/``.
    On first use, models are downloaded from the CDN and verified.
    Subsequent loads use the cached file.
    """

    def __init__(self, models_dir: Path | None = None):
        if models_dir is None:
            self._models_dir = Path(user_data_dir("Astraios", "Astraios")) / "models"
        else:
            self._models_dir = models_dir
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._models_dir / "manifest.json"
        self._manifest = self._load_manifest()

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def get_model_path(self, model_type: ModelType) -> Path:
        """Get the local path for a model file.

        Checks the bundled package models first, then the user cache dir.
        """
        info = MODEL_REGISTRY.get(model_type)
        if info is None:
            raise ValueError(f"Unknown model type: {model_type}")
        bundled = _BUNDLED_MODELS_DIR / info.filename
        if bundled.exists():
            return bundled
        return self._models_dir / info.filename

    def get_info(self, model_type: ModelType) -> ModelInfo | None:
        """Get metadata for a model type."""
        return MODEL_REGISTRY.get(model_type)

    def get_cache_path(self, model_type: ModelType) -> Path | None:
        """Get the local path to a cached model, or None if not found."""
        path = self.get_model_path(model_type)
        if path.exists():
            return path
        return None

    def is_available(self, model_type: ModelType) -> bool:
        """Check if a model is available (bundled or downloaded).

        A model file already on disk is available regardless of registry
        state — an empty sha256 only means it cannot be DOWNLOADED (not yet
        published on the CDN), not that a locally-installed copy is unusable.
        """
        return self.get_model_path(model_type).exists()

    def needs_download(self, model_type: ModelType) -> bool:
        """Check if a model needs to be downloaded."""
        return not self.is_available(model_type)

    def download_model(
        self,
        model_type: ModelType,
        progress: ProgressCallback | None = None,
    ) -> Path:
        """Download a model from the CDN.

        Parameters
        ----------
        model_type : ModelType
            Which model to download.
        progress : callable, optional
            Progress callback receiving (fraction, message).

        Returns
        -------
        Path
            Local path to the downloaded model file.

        Raises
        ------
        RuntimeError
            If download fails or integrity check fails.
        """
        if progress is None:
            progress = _noop_progress

        info = MODEL_REGISTRY.get(model_type)
        if info is None:
            raise ValueError(f"Unknown model type: {model_type}")
        if not info.sha256:
            raise RuntimeError(
                f"{info.description} is not published on the CDN yet; "
                "place the file manually in the model cache to use it"
            )

        url = f"{MODEL_CDN_BASE}/{info.filename}"
        dest = self._models_dir / info.filename
        tmp = dest.with_suffix(".tmp")

        progress(0.0, f"Downloading {info.description}...")
        log.info("Downloading model from %s", url)

        try:
            import requests

            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            downloaded = 0

            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        frac = downloaded / total
                        progress(frac * 0.9, f"Downloading... {downloaded // 1024}KB")

        except Exception as e:
            if tmp.exists():
                tmp.unlink()
            raise RuntimeError(f"Download failed: {e}") from e

        # Verify integrity
        progress(0.9, "Verifying integrity...")
        if info.sha256:
            file_hash = _sha256_file(tmp)
            if file_hash != info.sha256:
                tmp.unlink()
                raise RuntimeError(
                    f"Integrity check failed: expected {info.sha256[:16]}..., "
                    f"got {file_hash[:16]}..."
                )

        # Move to final location
        shutil.move(str(tmp), str(dest))

        # Update manifest
        self._manifest[info.filename] = {
            "version": info.version,
            "sha256": info.sha256 or _sha256_file(dest),
            "size_bytes": dest.stat().st_size,
        }
        self._save_manifest()

        progress(1.0, f"{info.description} ready")
        log.info("Model downloaded: %s (%d bytes)", dest, dest.stat().st_size)
        return dest

    def load_model(
        self,
        model_type: ModelType,
        device: torch.device | str = "cpu",
    ) -> torch.nn.Module:
        """Load a model's weights into the appropriate architecture.

        Parameters
        ----------
        model_type : ModelType
            Which model to load.
        device : device or str
            Device to load the model onto.

        Returns
        -------
        nn.Module
            The model with loaded weights, in eval mode.

        Raises
        ------
        FileNotFoundError
            If the model hasn't been downloaded yet.
        """
        path = self.get_model_path(model_type)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Call download_model() first."
            )

        model = self._create_model(model_type)
        state_dict = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        log.info("Loaded model %s from %s", model_type.name, path)
        return model

    def ensure_model(
        self,
        model_type: ModelType,
        device: torch.device | str = "cpu",
        progress: ProgressCallback | None = None,
    ) -> torch.nn.Module:
        """Download if needed, then load a model.

        Convenience method that combines download + load.
        """
        if self.needs_download(model_type):
            self.download_model(model_type, progress=progress)
        return self.load_model(model_type, device=device)

    def is_bundled(self, model_type: ModelType) -> bool:
        """True if this model ships inside the package rather than being
        downloaded into the user cache."""
        info = MODEL_REGISTRY.get(model_type)
        if info is None:
            return False
        return (_BUNDLED_MODELS_DIR / info.filename).exists()

    def delete_model(self, model_type: ModelType) -> None:
        """Delete a DOWNLOADED model from the user cache.

        Only the user cache is touched. This used to call `get_model_path()`,
        which prefers the bundled copy, so deleting a "cached" model actually
        removed the model that ships inside the installed package -- gone
        until the user reinstalled, and it silently deleted bundled weights
        during the test run that first bundled one.
        """
        info = MODEL_REGISTRY.get(model_type)
        if info is None:
            raise ValueError(f"Unknown model type: {model_type}")
        path = self._models_dir / info.filename
        if path.exists():
            path.unlink()
            self._manifest.pop(info.filename, None)
            self._save_manifest()
            log.info("Deleted cached model: %s", path)
        elif self.is_bundled(model_type):
            log.info(
                "%s is bundled with Astraios and cannot be deleted; nothing to do",
                info.filename,
            )

    def get_cache_size(self) -> int:
        """Return total size of cached models in bytes."""
        total = 0
        for path in self._models_dir.iterdir():
            if path.is_file() and path.suffix == ".pt":
                total += path.stat().st_size
        return total

    def list_models(self) -> list[dict]:
        """List all registered models with their status."""
        result = []
        for model_type, info in MODEL_REGISTRY.items():
            available = self.is_available(model_type)
            path = self.get_model_path(model_type)
            result.append({
                "type": model_type.name,
                "description": info.description,
                "version": info.version,
                "filename": info.filename,
                "available": available,
                "size_bytes": path.stat().st_size if path.exists() else 0,
            })
        return result

    @staticmethod
    def _create_model(model_type: ModelType) -> torch.nn.Module:
        """Create the model architecture for a given type."""
        if model_type == ModelType.DENOISE:
            from astraios.ai.models.denoise_model import create_denoise_model
            return create_denoise_model()
        elif model_type == ModelType.SHARPEN:
            from astraios.ai.models.sharpen_model import create_sharpen_model
            return create_sharpen_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _load_manifest(self) -> dict:
        """Load the model manifest from disk."""
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_manifest(self) -> None:
        """Save the model manifest to disk."""
        with open(self._manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# Module-level singleton
_model_manager_instance: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """Return the global ModelManager singleton."""
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = ModelManager()
    return _model_manager_instance
