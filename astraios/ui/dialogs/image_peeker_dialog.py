"""Image Peeker Dialog — thumbnail grid + quick stats table for culling subs.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

See ``astraios/core/image_peek.py`` for a note on how this differs from
SASpro's single-image ``ImagePeekerDialogPro`` (a 100%-crop corner mosaic +
tilt/focal-plane/distortion analysis dispatch on the currently open image).
This dialog keeps the "Image Peeker" name and its quick-look spirit but is a
*multi-file* inspector: point it at a folder of subs and get an
auto-stretched thumbnail plus quick stats for each frame, so a night's subs
can be culled without opening every frame individually. UI idioms (QThread
worker, sortable stats table, thumbnail caching) follow
``subframe_dialog.py`` and ``blink_dialog.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from astraios.core.image_peek import FramePeek, ImagePeekParams, peek_frames
from astraios.ui.widgets.ui_kit import field_row, help_dot, param_help

_TILE_THUMB_SIZE = 140  # px — thumbnail edge inside the grid tile
_FILE_FILTER = "Images (*.fits *.fit *.fts *.FTS *.xisf *.tif *.tiff *.png *.jpg *.jpeg)"
_FOLDER_EXTS = ("*.fits", "*.fit", "*.fts", "*.FTS", "*.xisf", "*.tif", "*.tiff")


def _ndarray_to_qimage(arr: np.ndarray) -> QImage:
    """Convert an (h, w, 3) uint8 array to a QImage (safe to call off the GUI thread)."""
    arr = np.ascontiguousarray(arr)
    h, w, _ = arr.shape
    return QImage(arr.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888).copy()


class _NumericItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically instead of lexicographically."""

    def __lt__(self, other: QTableWidgetItem) -> bool:
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super().__lt__(other)


class _PeekWorker(QThread):
    """Run ``peek_frames`` off the GUI thread."""

    progress = pyqtSignal(float, str)
    finished_peek = pyqtSignal(object)  # list[FramePeek]
    error = pyqtSignal(str)

    def __init__(self, paths: list[str], params: ImagePeekParams):
        super().__init__()
        self._paths = paths
        self._params = params

    def run(self):
        try:
            results = peek_frames(
                self._paths,
                self._params,
                progress=lambda f, m: self.progress.emit(f, m),
            )
            self.finished_peek.emit(results)
        except Exception as exc:  # pragma: no cover - defensive
            self.error.emit(str(exc))


class _PeekTile(QFrame):
    """Clickable thumbnail tile with a short caption, shown in the grid."""

    clicked = pyqtSignal(str)

    def __init__(self, frame: FramePeek, parent=None):
        super().__init__(parent)
        self.path = frame.path
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QFrame { background: #161b22; border: 1px solid #30363d; border-radius: 4px; }"
            "QFrame:hover { border-color: #2ea043; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        thumb_label = QLabel()
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb_label.setFixedSize(_TILE_THUMB_SIZE, _TILE_THUMB_SIZE)
        qimg = _ndarray_to_qimage(frame.thumbnail)
        pix = QPixmap.fromImage(qimg).scaled(
            _TILE_THUMB_SIZE, _TILE_THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        thumb_label.setPixmap(pix)
        layout.addWidget(thumb_label)

        name = Path(frame.path).name
        stars_txt = f"{frame.n_stars}★" if frame.n_stars is not None else "—"
        fwhm_txt = f"FWHM {frame.fwhm:.1f}px" if frame.fwhm is not None else ""
        caption = QLabel(f"{name}\nmed {frame.median:.4f}  {fwhm_txt}  {stars_txt}")
        caption.setWordWrap(True)
        caption.setStyleSheet("color: #8b949e; font-size: 10px;")
        caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        caption.setFixedWidth(_TILE_THUMB_SIZE)
        layout.addWidget(caption)

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.path)
        super().mousePressEvent(event)


class ImagePeekerDialog(QDialog):
    """Quick multi-file thumbnail + stats inspector for culling light frames."""

    # Emitted when the user clicks a thumbnail tile or a table row's file cell.
    open_requested = pyqtSignal(str)

    _COL_FILE   = 0
    _COL_MIN    = 1
    _COL_MEDIAN = 2
    _COL_MEAN   = 3
    _COL_MAX    = 4
    _COL_STD    = 5
    _COL_FWHM   = 6
    _COL_ECC    = 7
    _COL_STARS  = 8
    _COL_FILTER = 9
    _COL_EXP    = 10
    _COL_DATE   = 11

    def __init__(self, parent=None, paths: list[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Image Peeker")
        self.setMinimumSize(1000, 700)

        self._paths: list[str] = list(paths) if paths else []
        self._frames: list[FramePeek] = []
        self._worker: _PeekWorker | None = None
        self._grid_cols = 0

        layout = QVBoxLayout(self)

        # ── Controls ──────────────────────────────────────────────────────
        controls = QHBoxLayout()

        add_files_btn = QPushButton("Add Files...")
        add_files_btn.clicked.connect(self._add_files)
        controls.addWidget(add_files_btn)

        add_folder_btn = QPushButton("Add Folder...")
        add_folder_btn.clicked.connect(self._add_folder)
        controls.addWidget(add_folder_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear)
        controls.addWidget(clear_btn)

        controls.addSpacing(16)

        self._thumb_size_spin = QSpinBox()
        self._thumb_size_spin.setRange(64, 512)
        self._thumb_size_spin.setValue(220)
        controls.addLayout(field_row(
            "Thumb size:", self._thumb_size_spin, label_width=70,
            help_text=param_help(
                "Longest edge of the generated preview thumbnail, in "
                "pixels.",
                higher="Larger, more detailed previews, at the cost of "
                       "slower peeking over large batches.",
                lower="Smaller previews that peek faster over large "
                      "batches.",
            ),
        ))

        self._measure_stars_check = QCheckBox("Measure stars")
        self._measure_stars_check.setChecked(True)
        controls.addWidget(self._measure_stars_check)
        controls.addWidget(help_dot(
            "Also measure FWHM, eccentricity, and star count per frame "
            "(reuses the same PSF fit as Subframe Selector). Disable for a "
            "faster pass when only the thumbnail and brightness stats matter."
        ))

        controls.addStretch()

        self._peek_btn = QPushButton("Peek")
        self._peek_btn.setEnabled(False)
        self._peek_btn.clicked.connect(self._run_peek)
        controls.addWidget(self._peek_btn)

        self._count_label = QLabel("No frames loaded")
        self._count_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        controls.addWidget(self._count_label)

        layout.addLayout(controls)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # ── Thumbnail grid + stats table ─────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)

        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setSpacing(6)
        self._grid_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._grid_container)
        splitter.addWidget(scroll)

        self._table = QTableWidget(0, 12)
        self._table.setHorizontalHeaderLabels([
            "File", "Min", "Median", "Mean", "Max", "Std",
            "FWHM", "Eccentricity", "Stars", "Filter", "Exposure (s)", "Date-Obs",
        ])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setSectionResizeMode(
            self._COL_FILE, QHeaderView.ResizeMode.Stretch
        )
        self._table.cellDoubleClicked.connect(self._on_table_row_activated)
        splitter.addWidget(self._table)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        if self._paths:
            self._set_paths(self._paths)

    # ── Loading paths ───────────────────────────────────────────────────────

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Light Frames", "", _FILE_FILTER)
        if paths:
            self._set_paths(self._paths + paths)

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder of Light Frames")
        if not folder:
            return
        p = Path(folder)
        found: list[str] = []
        for ext in _FOLDER_EXTS:
            found.extend(str(f) for f in sorted(p.glob(ext)))
        if found:
            self._set_paths(self._paths + found)

    def _clear(self):
        self._paths = []
        self._frames = []
        self._peek_btn.setEnabled(False)
        self._count_label.setText("No frames loaded")
        self._clear_grid()
        self._table.setRowCount(0)

    def _set_paths(self, paths: list[str]):
        # De-duplicate while preserving order.
        seen = set()
        ordered = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        self._paths = ordered
        n = len(self._paths)
        self._count_label.setText(f"{n} frame{'s' if n != 1 else ''} loaded")
        self._peek_btn.setEnabled(n > 0)

    # ── Running the peek ─────────────────────────────────────────────────────

    def _run_peek(self):
        if not self._paths:
            return
        params = ImagePeekParams(
            thumbnail_size=self._thumb_size_spin.value(),
            measure_stars=self._measure_stars_check.isChecked(),
        )
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._peek_btn.setEnabled(False)

        self._worker = _PeekWorker(self._paths, params)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_peek.connect(self._on_peeked)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, fraction: float, message: str):
        self._progress.setValue(int(fraction * 100))
        self._progress.setFormat(f"{message} (%p%)")

    def _on_error(self, message: str):
        self._progress.setVisible(False)
        self._peek_btn.setEnabled(True)
        self._count_label.setText(f"Error: {message}")

    def _on_peeked(self, frames: list[FramePeek]):
        self._frames = frames
        self._progress.setVisible(False)
        self._peek_btn.setEnabled(True)
        n_ok = len(frames)
        n_total = len(self._paths)
        skipped = n_total - n_ok
        note = f"  ({skipped} unreadable, skipped)" if skipped else ""
        self._count_label.setText(f"{n_ok}/{n_total} frames peeked{note}")
        self._populate_grid(frames, force=True)
        self._populate_table(frames)

    # ── Thumbnail grid ────────────────────────────────────────────────────

    def _clear_grid(self):
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _populate_grid(self, frames: list[FramePeek], force: bool = False):
        cols = max(1, self.width() // (_TILE_THUMB_SIZE + 20))
        if not force and cols == self._grid_cols:
            return
        self._grid_cols = cols
        self._clear_grid()
        for idx, frame in enumerate(frames):
            tile = _PeekTile(frame, parent=self._grid_container)
            tile.clicked.connect(self.open_requested.emit)
            row, col = divmod(idx, cols)
            self._grid_layout.addWidget(tile, row, col)

    # ── Stats table ───────────────────────────────────────────────────────

    def _populate_table(self, frames: list[FramePeek]):
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(frames))

        def _num(text: str) -> _NumericItem:
            it = _NumericItem(text)
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            return it

        for row, frame in enumerate(frames):
            file_item = QTableWidgetItem(Path(frame.path).name)
            file_item.setData(Qt.ItemDataRole.UserRole, frame.path)
            if frame.error:
                file_item.setForeground(QColor(220, 170, 90))
                file_item.setToolTip(frame.error)
            self._table.setItem(row, self._COL_FILE, file_item)

            self._table.setItem(row, self._COL_MIN, _num(f"{frame.min_val:.5f}"))
            self._table.setItem(row, self._COL_MEDIAN, _num(f"{frame.median:.5f}"))
            self._table.setItem(row, self._COL_MEAN, _num(f"{frame.mean:.5f}"))
            self._table.setItem(row, self._COL_MAX, _num(f"{frame.max_val:.5f}"))
            self._table.setItem(row, self._COL_STD, _num(f"{frame.std:.5f}"))

            fwhm_txt = f"{frame.fwhm:.2f}" if frame.fwhm is not None else "—"
            ecc_txt = f"{frame.eccentricity:.3f}" if frame.eccentricity is not None else "—"
            stars_txt = str(frame.n_stars) if frame.n_stars is not None else "—"
            self._table.setItem(row, self._COL_FWHM, _num(fwhm_txt))
            self._table.setItem(row, self._COL_ECC, _num(ecc_txt))
            self._table.setItem(row, self._COL_STARS, _num(stars_txt))

            self._table.setItem(row, self._COL_FILTER, QTableWidgetItem(frame.filter_name or "—"))
            exp_txt = f"{frame.exposure:.1f}" if frame.exposure is not None else "—"
            self._table.setItem(row, self._COL_EXP, _num(exp_txt))
            self._table.setItem(row, self._COL_DATE, QTableWidgetItem(frame.date_obs or "—"))

        self._table.setSortingEnabled(True)

    def _on_table_row_activated(self, row: int, _col: int):
        item = self._table.item(row, self._COL_FILE)
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.open_requested.emit(path)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self._frames:
            self._populate_grid(self._frames)
