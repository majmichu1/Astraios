"""Tests for project management."""

from pathlib import Path

import pytest

from astraios.core.image_io import FrameType
from astraios.core.project import Project


class TestProject:
    def test_create_project(self, tmp_path):
        proj = Project.create("TestProject", tmp_path)
        assert proj.name == "TestProject"
        assert proj.project_file.exists()

    def test_save_and_load(self, tmp_path):
        proj = Project.create("SaveLoad", tmp_path)
        proj.add_frames([Path("/fake/light1.fits"), Path("/fake/light2.fits")], FrameType.LIGHT)
        proj.add_history("TestStep", {"param": 42})
        proj.save()

        loaded = Project.load(proj.project_file)
        assert loaded.name == "SaveLoad"
        assert len(loaded.frames) == 2
        assert len(loaded.history) == 1
        assert loaded.history[0].params["param"] == 42

    def test_add_frames(self, tmp_path):
        proj = Project.create("FrameTest", tmp_path)
        added = proj.add_frames(
            [Path("/a.fits"), Path("/b.fits"), Path("/c.fits")],
            FrameType.DARK,
        )
        assert added == 3
        assert len(proj.frames_by_type(FrameType.DARK)) == 3
        assert len(proj.frames_by_type(FrameType.LIGHT)) == 0

    def test_no_duplicate_frames(self, tmp_path):
        proj = Project.create("NoDup", tmp_path)
        proj.add_frames([Path("/a.fits")], FrameType.LIGHT)
        proj.add_frames([Path("/a.fits")], FrameType.LIGHT)
        assert len(proj.frames) == 1

    def test_remove_frame(self, tmp_path):
        proj = Project.create("Remove", tmp_path)
        proj.add_frames([Path("/a.fits"), Path("/b.fits")], FrameType.LIGHT)
        assert proj.remove_frame(Path("/a.fits"))
        assert len(proj.frames) == 1
        assert not proj.remove_frame(Path("/nonexistent.fits"))

    def test_directories(self, tmp_path):
        proj = Project.create("Dirs", tmp_path)
        assert proj.masters_dir.exists()
        assert proj.calibrated_dir.exists()
        assert proj.output_dir.exists()
