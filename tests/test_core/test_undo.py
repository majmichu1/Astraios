"""Tests for the undo/redo system."""

import numpy as np
import pytest

from astraios.core.image_io import ImageData
from astraios.core.undo import AstraiosUndoStack, ImageEditCommand


def _make_image(value: float = 0.5, shape=(10, 10)) -> ImageData:
    return ImageData(data=np.full(shape, value, dtype=np.float32))


class TestImageEditCommand:
    def test_undo_restores_before(self):
        before = _make_image(0.3)
        after = _make_image(0.7)
        target = [after]

        cmd = ImageEditCommand(before, after, "Test operation")
        cmd.set_target(target)

        cmd.undo()
        assert target[0].data[0, 0] == pytest.approx(0.3, abs=1e-6)

    def test_redo_restores_after(self):
        before = _make_image(0.3)
        after = _make_image(0.7)
        target = [after]

        cmd = ImageEditCommand(before, after, "Test operation")
        cmd.set_target(target)

        cmd.redo()
        assert target[0].data[0, 0] == pytest.approx(0.7, abs=1e-6)

    def test_undo_redo_cycle(self):
        before = _make_image(0.3)
        after = _make_image(0.7)
        target = [after]

        cmd = ImageEditCommand(before, after, "Test")
        cmd.set_target(target)

        cmd.undo()
        assert target[0].data[0, 0] == pytest.approx(0.3, abs=1e-6)

        cmd.redo()
        assert target[0].data[0, 0] == pytest.approx(0.7, abs=1e-6)

        cmd.undo()
        assert target[0].data[0, 0] == pytest.approx(0.3, abs=1e-6)


class TestAstraiosUndoStack:
    def test_push_and_undo(self):
        target = [_make_image(0.5)]
        stack = AstraiosUndoStack()
        stack.set_target(target)

        before = _make_image(0.3)
        after = _make_image(0.7)
        stack.push(before, after, "Test")

        assert stack.can_undo()
        assert not stack.can_redo()

        stack.undo()
        assert target[0].data[0, 0] == pytest.approx(0.3, abs=1e-6)
        assert stack.can_redo()

    def test_push_undo_redo(self):
        target = [_make_image(0.5)]
        stack = AstraiosUndoStack()
        stack.set_target(target)

        stack.push(_make_image(0.3), _make_image(0.5), "Op 1")
        stack.push(_make_image(0.5), _make_image(0.8), "Op 2")

        assert stack.count == 2

        # Undo Op 2 -> goes back to before=0.5
        stack.undo()
        assert target[0].data[0, 0] == pytest.approx(0.5, abs=1e-6)

        # Redo Op 2 -> goes back to after=0.8
        stack.redo()
        assert target[0].data[0, 0] == pytest.approx(0.8, abs=1e-6)

        # Undo Op 1 -> goes back to before=0.3
        stack.undo()
        stack.undo()
        assert target[0].data[0, 0] == pytest.approx(0.3, abs=1e-6)

    def test_can_undo_redo_initial(self):
        stack = AstraiosUndoStack()
        assert not stack.can_undo()
        assert not stack.can_redo()

    def test_clear(self):
        target = [_make_image(0.5)]
        stack = AstraiosUndoStack()
        stack.set_target(target)

        stack.push(_make_image(0.3), _make_image(0.5), "Op")
        assert stack.count > 0

        stack.clear()
        assert stack.count == 0

    def test_undo_text(self):
        target = [_make_image(0.5)]
        stack = AstraiosUndoStack()
        stack.set_target(target)

        stack.push(_make_image(0.3), _make_image(0.5), "Stretch")
        assert stack.undo_text() == "Stretch"

    def test_redo_text(self):
        target = [_make_image(0.5)]
        stack = AstraiosUndoStack()
        stack.set_target(target)

        stack.push(_make_image(0.3), _make_image(0.5), "Stretch")
        assert stack.redo_text() == ""

        stack.undo()
        assert stack.redo_text() == "Stretch"

    def test_multiple_operations(self):
        target = [_make_image(0.0)]
        stack = AstraiosUndoStack()
        stack.set_target(target)

        for i in range(1, 6):
            before = _make_image(float(i - 1) / 10)
            after = _make_image(float(i) / 10)
            stack.push(before, after, f"Step {i}")

        assert stack.count == 5
        assert stack.can_undo()

        # Undo all
        for _ in range(5):
            stack.undo()

        assert not stack.can_undo()
        # Redo all
        for _ in range(5):
            stack.redo()

        assert stack.can_undo()
        assert not stack.can_redo()
