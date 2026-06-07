"""Undo/Redo system — QUndoStack-based command history for image operations."""

from __future__ import annotations

import logging

from PyQt6.QtGui import QUndoCommand, QUndoStack

from cosmica.core.image_io import ImageData

log = logging.getLogger(__name__)

# Maximum number of undo steps to keep in memory
# Lowered from 50 — avoids 5+ GB for 8MP images (96 MB per copy × 50 = 4.8 GB)
MAX_UNDO_DEPTH = 20


class ImageEditCommand(QUndoCommand):
    """Generic undo/redo command for image transformations.

    Stores a copy of the image before and after the operation.
    For large images, this uses significant memory — the stack
    depth is limited by MAX_UNDO_DEPTH.

    Also stores the display stretch reference used for each state so that
    undo/redo can restore the exact same display brightness.
    """

    def __init__(
        self,
        before: ImageData,
        after: ImageData,
        description: str,
        before_display_ref=None,
        parent: QUndoCommand | None = None,
    ):
        super().__init__(description, parent)
        self._before = _safe_copy(before)
        self._after = _safe_copy(after)
        # Stretch references are small (thumbnail-sized) arrays — cheap to store
        self._before_display_ref = before_display_ref
        self._after_display_ref = None  # set after Apply display is computed
        self._target_ref: list[ImageData] | None = None  # set via set_target()
        self._display_ref_box: list | None = None  # [current_display_ref]

    def set_target(self, target_ref: list) -> None:
        """Set the mutable reference where undo/redo will swap the image."""
        self._target_ref = target_ref

    def set_display_ref_box(self, box: list) -> None:
        """Set the shared box that receives the display ref on undo/redo."""
        self._display_ref_box = box

    def undo(self) -> None:
        if self._target_ref is not None:
            self._target_ref[0] = self._before
        if self._display_ref_box is not None:
            self._display_ref_box[0] = self._before_display_ref
        log.debug("Undo: %s", self.text())

    def redo(self) -> None:
        if self._target_ref is not None:
            self._target_ref[0] = self._after
        if self._display_ref_box is not None:
            self._display_ref_box[0] = self._after_display_ref
        log.debug("Redo: %s", self.text())


def _safe_copy(image: ImageData) -> ImageData:
    """Create a deep copy of ImageData for undo storage."""
    return ImageData(
        data=image.data.copy(),
        header=dict(image.header),
        file_path=image.file_path,
        frame_type=image.frame_type,
    )


class CosmicaUndoStack:
    """High-level undo/redo manager wrapping QUndoStack.

    Provides a simple API:
    - push(before, after, description) — record an operation
    - undo() / redo() — navigate history
    - can_undo() / can_redo() — query state
    - clear() — reset history
    """

    def __init__(self, max_depth: int = MAX_UNDO_DEPTH):
        self._stack = QUndoStack()
        self._max_depth = max_depth
        self._target_ref: list[ImageData | None] = [None]
        # Shared box: undo/redo commands write the display ref here after executing
        self._display_ref_box: list = [None]

    def set_target(self, target_ref: list) -> None:
        """Set the mutable reference to the current image."""
        self._target_ref = target_ref

    def current_display_ref(self):
        """Return the stretch reference that should be used for the current display.

        Updated automatically whenever undo() or redo() is called.
        """
        return self._display_ref_box[0]

    def push(
        self,
        before: ImageData,
        after: ImageData,
        description: str,
        before_display_ref=None,
    ) -> None:
        """Push an operation onto the undo stack.

        Args:
            before: Image state before the operation.
            after: Image state after the operation.
            description: Human-readable description (e.g. "Stretch").
            before_display_ref: The stretch reference (numpy array or None) that was
                used to display the ``before`` state.  Restored on undo.
        """
        if self._stack.count() >= self._max_depth:
            self._stack.setUndoLimit(self._max_depth)

        cmd = ImageEditCommand(before, after, description, before_display_ref=before_display_ref)
        cmd.set_target(self._target_ref)
        cmd.set_display_ref_box(self._display_ref_box)
        self._stack.push(cmd)
        log.debug("Pushed undo command: %s (stack size: %d)", description, self._stack.count())

    def set_last_after_display_ref(self, ref) -> None:
        """Set the display ref for the *after* state of the most recent command.

        Call this after the Apply display has been computed so that redo can
        restore the exact same display brightness.
        """
        idx = self._stack.index() - 1
        if 0 <= idx < self._stack.count():
            cmd = self._stack.command(idx)
            if isinstance(cmd, ImageEditCommand):
                cmd._after_display_ref = ref

    def undo(self) -> bool:
        """Undo the last operation. Returns True if successful."""
        if self.can_undo():
            self._stack.undo()
            return True
        return False

    def redo(self) -> bool:
        """Redo the last undone operation. Returns True if successful."""
        if self.can_redo():
            self._stack.redo()
            return True
        return False

    def can_undo(self) -> bool:
        return self._stack.canUndo()

    def can_redo(self) -> bool:
        return self._stack.canRedo()

    def clear(self) -> None:
        self._stack.clear()
        log.debug("Undo stack cleared")

    @property
    def count(self) -> int:
        return self._stack.count()

    def undo_text(self) -> str:
        """Return description of the next undo action."""
        return self._stack.undoText()

    def redo_text(self) -> str:
        """Return description of the next redo action."""
        return self._stack.redoText()
