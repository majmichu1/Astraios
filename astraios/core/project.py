"""Project Management — create, save, load astrophotography projects."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from astraios.core.image_io import FrameType

log = logging.getLogger(__name__)

PROJECT_VERSION = 1
PROJECT_FILE = "astraios_project.json"


@dataclass
class FrameEntry:
    path: Path
    frame_type: FrameType
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "frame_type": self.frame_type.name,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FrameEntry:
        return cls(
            path=Path(d["path"]),
            frame_type=FrameType[d["frame_type"]],
            enabled=d.get("enabled", True),
            metadata=d.get("metadata", {}),
        )


@dataclass
class ProcessingStep:
    name: str
    timestamp: str
    params: dict[str, Any] = field(default_factory=dict)
    output_path: Path | None = None

    def to_dict(self) -> dict:
        d = {"name": self.name, "timestamp": self.timestamp, "params": self.params}
        if self.output_path:
            d["output_path"] = str(self.output_path)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ProcessingStep:
        return cls(
            name=d["name"],
            timestamp=d["timestamp"],
            params=d.get("params", {}),
            output_path=Path(d["output_path"]) if d.get("output_path") else None,
        )


@dataclass
class Project:
    name: str
    directory: Path
    frames: list[FrameEntry] = field(default_factory=list)
    history: list[ProcessingStep] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    frame_scores: dict = field(default_factory=dict)
    created: str = ""
    modified: str = ""

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().isoformat()
        if not self.modified:
            self.modified = self.created

    @property
    def project_file(self) -> Path:
        return self.directory / PROJECT_FILE

    @property
    def masters_dir(self) -> Path:
        d = self.directory / "masters"
        d.mkdir(exist_ok=True)
        return d

    @property
    def calibrated_dir(self) -> Path:
        d = self.directory / "calibrated"
        d.mkdir(exist_ok=True)
        return d

    @property
    def output_dir(self) -> Path:
        d = self.directory / "output"
        d.mkdir(exist_ok=True)
        return d

    def frames_by_type(self, frame_type: FrameType) -> list[FrameEntry]:
        return [f for f in self.frames if f.frame_type == frame_type and f.enabled]

    def add_frames(self, paths: list[Path], frame_type: FrameType) -> int:
        existing = {f.path for f in self.frames}
        added = 0
        for p in paths:
            p = Path(p).resolve()
            if p not in existing:
                self.frames.append(FrameEntry(path=p, frame_type=frame_type))
                added += 1
        self.touch()
        return added

    def remove_frame(self, path: Path) -> bool:
        path = Path(path).resolve()
        for i, f in enumerate(self.frames):
            if f.path == path:
                self.frames.pop(i)
                self.touch()
                return True
        return False

    def add_history(self, name: str, params: dict | None = None, output_path: Path | None = None):
        step = ProcessingStep(
            name=name,
            timestamp=datetime.now().isoformat(),
            params=params or {},
            output_path=output_path,
        )
        self.history.append(step)
        self.touch()

    def touch(self):
        self.modified = datetime.now().isoformat()

    def cache_frame_scores(self, scores_dict: dict) -> None:
        """Store per-frame quality scores keyed by absolute file path (str)."""
        self.frame_scores.update({str(k): v for k, v in scores_dict.items()})
        self.touch()

    def save(self):
        data = {
            "version": PROJECT_VERSION,
            "name": self.name,
            "created": self.created,
            "modified": self.modified,
            "settings": self.settings,
            "frames": [f.to_dict() for f in self.frames],
            "history": [h.to_dict() for h in self.history],
            "frame_scores": self.frame_scores,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        with open(self.project_file, "w") as f:
            json.dump(data, f, indent=2)
        log.info("Project saved: %s", self.project_file)

    @classmethod
    def load(cls, path: Path) -> Project:
        path = Path(path)
        if path.is_dir():
            path = path / PROJECT_FILE
        if not path.exists():
            raise FileNotFoundError(f"Project file not found: {path}")
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted project file: {path} — {e}") from e

        proj = cls(
            name=data["name"],
            directory=path.parent,
            created=data.get("created", ""),
            modified=data.get("modified", ""),
            settings=data.get("settings", {}),
        )
        proj.frames = [FrameEntry.from_dict(d) for d in data.get("frames", [])]
        proj.history = [ProcessingStep.from_dict(d) for d in data.get("history", [])]
        proj.frame_scores = data.get("frame_scores", {})
        log.info("Project loaded: %s (%d frames)", proj.name, len(proj.frames))
        proj._prune_missing_derived()
        return proj

    def _prune_missing_derived(self) -> None:
        """Remove derived frames (ALIGNED) whose files no longer exist on disk.

        Raw source frames (LIGHT/DARK/FLAT/BIAS) are kept even if missing — they
        may be on removable media. Derived frames are re-generated and safe to prune.
        """
        derived_types = {FrameType.ALIGNED}
        before = len(self.frames)
        self.frames = [
            f for f in self.frames
            if f.frame_type not in derived_types or f.path.exists()
        ]
        removed = before - len(self.frames)
        if removed:
            log.warning(
                "Pruned %d missing derived frame(s) from project (files not found on disk)",
                removed,
            )

    @classmethod
    def create(cls, name: str, directory: Path) -> Project:
        directory = Path(directory) / name
        directory.mkdir(parents=True, exist_ok=True)
        proj = cls(name=name, directory=directory)
        proj.save()
        return proj
