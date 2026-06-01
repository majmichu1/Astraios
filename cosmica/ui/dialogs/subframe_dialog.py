"""Subframe Selector Dialog — score and reject light frames by quality.

Shows a thumbnail preview alongside per-frame statistics so the user can
visually verify which frames are sharp, round-star, and high-SNR before
passing them to stacking.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed as futures_as_completed
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from cosmica.core.subframe_selector import (
    SubframeScore,
    SubframeSelectorParams,
    filter_by_metric,
    score_subframes,
)

_THUMB_SIZE = 80   # px — thumbnail column width/height


def _make_thumbnail(path: str, size: int = _THUMB_SIZE) -> QPixmap:
    """Load an image, auto-stretch, and return a square QPixmap thumbnail.

    Downscales to 256px *before* stretching so auto_stretch runs on a tiny
    array — typically 20–50× faster than stretching the full image.
    """
    try:
        from cosmica.core.image_io import load_image
        from cosmica.core.stretch import StretchParams, auto_stretch
        img = load_image(path)
        data = img.data.astype(np.float32)

        # Fast downscale to at most 256px before expensive stretch
        _MAX = 256
        h, w = data.shape[-2], data.shape[-1]
        if max(h, w) > _MAX:
            factor = _MAX / max(h, w)
            new_h, new_w = max(1, int(h * factor)), max(1, int(w * factor))
            # Sliced sub-sampling — no scipy/torch needed, instant
            ry = max(1, h // new_h)
            rx = max(1, w // new_w)
            data = data[..., ::ry, ::rx][..., :new_h, :new_w]

        stretched = auto_stretch(data, StretchParams())
        if stretched.ndim == 2:
            rgb = np.stack([stretched] * 3, axis=-1)
        else:
            rgb = np.transpose(stretched, (1, 2, 0))
        rgb8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        h2, w2 = rgb8.shape[:2]
        s = min(h2, w2)
        y0, x0 = (h2 - s) // 2, (w2 - s) // 2
        crop = np.ascontiguousarray(rgb8[y0:y0+s, x0:x0+s])
        qimg = QImage(crop.tobytes(), s, s, s * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        return pix.scaled(size, size,
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
    except Exception:
        pix = QPixmap(size, size)
        pix.fill(QColor(40, 40, 40))
        return pix


class _ThumbLabel(QLabel):
    """Thumbnail cell widget that pops up a larger preview on hover."""

    _POPUP_SIZE = 300

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pixmap_full: QPixmap | None = None
        self._popup: QLabel | None = None

    def set_full_pixmap(self, pix: QPixmap) -> None:
        self._pixmap_full = pix

    def enterEvent(self, event):
        if self._pixmap_full is not None:
            popup = QLabel(self.window())
            popup.setWindowFlags(Qt.WindowType.ToolTip)
            scaled = self._pixmap_full.scaled(
                self._POPUP_SIZE, self._POPUP_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            popup.setPixmap(scaled)
            popup.setStyleSheet(
                "QLabel { background: #0d1117; border: 1px solid #30363d; padding: 4px; }"
            )
            popup.adjustSize()
            # Position near cursor, shifted so it doesn't cover the cell
            pos = QCursor.pos()
            popup.move(pos.x() + 16, pos.y() + 16)
            popup.show()
            self._popup = popup
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self._popup is not None:
            self._popup.close()
            self._popup = None
        super().leaveEvent(event)


class _NumericItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically."""

    def __lt__(self, other: QTableWidgetItem) -> bool:
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super().__lt__(other)


class _ScoreWorker(QThread):
    """Run subframe scoring off the main thread."""

    progress = pyqtSignal(float, str)
    # Emitted as each frame finishes measuring (row, raw metrics dict)
    frame_measured = pyqtSignal(int, dict)
    finished = pyqtSignal(list)       # list[SubframeScore] — emitted immediately after scoring
    thumbnail = pyqtSignal(int, object)  # row, QPixmap — emitted async after scoring
    error = pyqtSignal(str)

    def __init__(self, paths: list[str], params: SubframeSelectorParams):
        super().__init__()
        self._paths = paths
        self._params = params

    def run(self):
        try:
            def _frame_cb(idx: int, metrics: dict):
                self.frame_measured.emit(idx, metrics)

            scores = score_subframes(
                self._paths,
                self._params,
                progress=lambda f, m: self.progress.emit(f, m),
                frame_callback=_frame_cb,
            )

            # Emit scores immediately so the table shows accept/reject status
            self.finished.emit(scores)

            # Generate thumbnails in parallel (IO-bound) — emit each as it arrives
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(_make_thumbnail, s.file_path): row
                           for row, s in enumerate(scores)}
                for future in futures_as_completed(futures):
                    row = futures[future]
                    try:
                        pix = future.result()
                        self.thumbnail.emit(row, pix)
                    except Exception:
                        pass

        except Exception as exc:
            self.error.emit(str(exc))


class SubframeDialog(QDialog):
    """Dialog for scoring and selecting subframes with thumbnail previews."""

    accepted_frames = pyqtSignal(list, int)  # list[str] of accepted paths, total n_paths
    scores_ready = pyqtSignal(list)     # list[SubframeScore] — all scored frames

    # Column indices
    _COL_THUMB  = 0
    _COL_FILE   = 1
    _COL_FWHM   = 2
    _COL_ECC    = 3
    _COL_SNR    = 4
    _COL_BG     = 5
    _COL_STARS  = 6
    _COL_SCORE  = 7
    _COL_STATUS = 8

    def __init__(self, parent=None, preloaded_paths: list[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Subframe Selector")
        self.setMinimumSize(960, 680)

        self._scores: list[SubframeScore] = []
        self._worker = None
        self._paths: list[str] = []
        # Manual overrides: path → True (force accept) / False (force reject)
        self._manual_overrides: dict[str, bool] = {}

        layout = QVBoxLayout(self)

        # ── Scoring weights ───────────────────────────────────────────────────
        params_group = QGroupBox("Scoring Weights")
        p_layout = QHBoxLayout(params_group)

        self._fwhm_w = QDoubleSpinBox()
        self._fwhm_w.setRange(0, 1)
        self._fwhm_w.setValue(0.3)
        self._fwhm_w.setSingleStep(0.05)
        p_layout.addWidget(QLabel("FWHM:"))
        p_layout.addWidget(self._fwhm_w)

        self._ecc_w = QDoubleSpinBox()
        self._ecc_w.setRange(0, 1)
        self._ecc_w.setValue(0.2)
        self._ecc_w.setSingleStep(0.05)
        p_layout.addWidget(QLabel("Ecc:"))
        p_layout.addWidget(self._ecc_w)

        self._snr_w = QDoubleSpinBox()
        self._snr_w.setRange(0, 1)
        self._snr_w.setValue(0.3)
        self._snr_w.setSingleStep(0.05)
        p_layout.addWidget(QLabel("SNR:"))
        p_layout.addWidget(self._snr_w)

        self._stars_w = QDoubleSpinBox()
        self._stars_w.setRange(0, 1)
        self._stars_w.setValue(0.2)
        self._stars_w.setSingleStep(0.05)
        p_layout.addWidget(QLabel("Stars:"))
        p_layout.addWidget(self._stars_w)

        p_layout.addStretch()
        layout.addWidget(params_group)

        # ── Rejection / Selection controls ────────────────────────────────────
        filter_group = QGroupBox("Rejection / Selection")
        filter_h = QHBoxLayout(filter_group)

        filter_h.addWidget(QLabel("Mode:"))
        self._filter_mode_combo = QComboBox()
        self._filter_mode_combo.addItems([
            "Sigma rejection",
            "Keep best N frames",
            "Keep best N%",
        ])
        filter_h.addWidget(self._filter_mode_combo)
        filter_h.addSpacing(16)

        self._sigma_label = QLabel("Rejection sigma:")
        self._rejection_sigma_spin = QDoubleSpinBox()
        self._rejection_sigma_spin.setRange(0.5, 5.0)
        self._rejection_sigma_spin.setValue(1.5)
        self._rejection_sigma_spin.setSingleStep(0.5)
        filter_h.addWidget(self._sigma_label)
        filter_h.addWidget(self._rejection_sigma_spin)

        self._keep_n_label = QLabel("Keep N:")
        self._keep_n_spin = QSpinBox()
        self._keep_n_spin.setRange(1, 9999)
        self._keep_n_spin.setValue(20)
        filter_h.addWidget(self._keep_n_label)
        filter_h.addWidget(self._keep_n_spin)

        self._keep_pct_label = QLabel("Keep %:")
        self._keep_pct_spin = QDoubleSpinBox()
        self._keep_pct_spin.setRange(10.0, 100.0)
        self._keep_pct_spin.setValue(80.0)
        self._keep_pct_spin.setSingleStep(5.0)
        filter_h.addWidget(self._keep_pct_label)
        filter_h.addWidget(self._keep_pct_spin)

        filter_h.addStretch()

        reapply_btn = QPushButton("Re-apply Filter")
        reapply_btn.setToolTip("Re-apply the current filter to existing scores")
        reapply_btn.clicked.connect(self._reapply_filter)
        filter_h.addWidget(reapply_btn)

        layout.addWidget(filter_group)

        self._filter_mode_combo.currentIndexChanged.connect(self._on_filter_mode_changed)
        self._rejection_sigma_spin.valueChanged.connect(self._reapply_filter)
        self._keep_n_spin.valueChanged.connect(self._reapply_filter)
        self._keep_pct_spin.valueChanged.connect(self._reapply_filter)
        self._on_filter_mode_changed(0)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load Frames...")
        load_btn.clicked.connect(self._load_frames)
        btn_row.addWidget(load_btn)

        load_dir_btn = QPushButton("Load Folder...")
        load_dir_btn.setToolTip("Load all FITS files from a folder")
        load_dir_btn.clicked.connect(self._load_folder)
        btn_row.addWidget(load_dir_btn)

        self._clear_all_btn = QPushButton("Clear All")
        self._clear_all_btn.setToolTip("Remove all loaded frames and reset the table")
        self._clear_all_btn.setEnabled(False)
        self._clear_all_btn.clicked.connect(self._clear_all_frames)
        btn_row.addWidget(self._clear_all_btn)

        self._score_btn = QPushButton("Score All Frames")
        self._score_btn.setEnabled(False)
        self._score_btn.clicked.connect(self._run_scoring)
        btn_row.addWidget(self._score_btn)

        btn_row.addSpacing(16)

        # Manual accept/reject buttons
        self._accept_sel_btn = QPushButton("Force Accept")
        self._accept_sel_btn.setToolTip("Mark selected frames as accepted regardless of score")
        self._accept_sel_btn.setEnabled(False)
        self._accept_sel_btn.clicked.connect(self._force_accept_selected)
        btn_row.addWidget(self._accept_sel_btn)

        self._reject_sel_btn = QPushButton("Force Reject")
        self._reject_sel_btn.setToolTip("Mark selected frames as rejected (e.g. clouds, satellites)")
        self._reject_sel_btn.setEnabled(False)
        self._reject_sel_btn.clicked.connect(self._force_reject_selected)
        btn_row.addWidget(self._reject_sel_btn)

        self._clear_override_btn = QPushButton("Clear Override")
        self._clear_override_btn.setToolTip("Remove manual override and restore automatic score for selected frames")
        self._clear_override_btn.setEnabled(False)
        self._clear_override_btn.clicked.connect(self._clear_override_selected)
        btn_row.addWidget(self._clear_override_btn)

        btn_row.addStretch()

        self._frame_count_label = QLabel("No frames loaded")
        self._frame_count_label.setStyleSheet("color: #aaa; font-size: 11px;")
        btn_row.addWidget(self._frame_count_label)

        btn_row.addStretch()

        accept_btn = QPushButton("Use Accepted Frames")
        accept_btn.setToolTip("Send accepted frames to the stacking pipeline")
        accept_btn.clicked.connect(self._emit_accepted)
        btn_row.addWidget(accept_btn)

        layout.addLayout(btn_row)

        # ── Progress ──────────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # ── Results table ─────────────────────────────────────────────────────
        self._table = QTableWidget(0, 9)
        self._table.setHorizontalHeaderLabels([
            "Preview", "File", "FWHM", "Eccentricity", "SNR",
            "Background", "Stars", "Score", "Status",
        ])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSortingEnabled(True)
        self._table.selectionModel().selectionChanged.connect(self._on_selection_changed)

        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(self._COL_THUMB,  QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(self._COL_FILE,   QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(self._COL_STATUS, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(self._COL_THUMB,  _THUMB_SIZE + 8)
        self._table.setColumnWidth(self._COL_STATUS, 100)
        self._table.verticalHeader().setDefaultSectionSize(_THUMB_SIZE + 8)

        layout.addWidget(self._table, 1)

        # ── Summary label ─────────────────────────────────────────────────────
        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self._summary_label)

        if preloaded_paths:
            self._set_paths(preloaded_paths)

    # ── Filter controls ───────────────────────────────────────────────────────

    def _on_filter_mode_changed(self, index: int) -> None:
        is_sigma  = index == 0
        is_top_n  = index == 1
        is_top_pct = index == 2

        self._sigma_label.setVisible(is_sigma)
        self._rejection_sigma_spin.setVisible(is_sigma)
        self._keep_n_label.setVisible(is_top_n)
        self._keep_n_spin.setVisible(is_top_n)
        self._keep_pct_label.setVisible(is_top_pct)
        self._keep_pct_spin.setVisible(is_top_pct)

        if self._scores:
            self._reapply_filter()

    def _apply_filter(self, scores: list) -> list:
        mode_idx = self._filter_mode_combo.currentIndex()
        if mode_idx == 0:
            return filter_by_metric(scores, metric="quality_score", mode="sigma",
                                    sigma=self._rejection_sigma_spin.value())
        elif mode_idx == 1:
            return filter_by_metric(scores, metric="quality_score", mode="top_n",
                                    top_n=self._keep_n_spin.value())
        else:
            return filter_by_metric(scores, metric="quality_score", mode="top_percent",
                                    top_percent=self._keep_pct_spin.value())

    def _reapply_filter(self) -> None:
        if not self._scores:
            return
        filtered = self._apply_filter(self._scores)
        accepted_paths = {s.file_path for s in filtered if s.accepted}
        accepted_paths = self._apply_manual_overrides(accepted_paths)
        self._colour_rows(self._scores, accepted_paths)

    def _apply_manual_overrides(self, accepted_paths: set) -> set:
        """Apply manual force-accept / force-reject overrides to an accepted set."""
        result = set(accepted_paths)
        for path, override in self._manual_overrides.items():
            if override:
                result.add(path)
            else:
                result.discard(path)
        return result

    def _current_accepted_paths(self) -> set[str]:
        """Compute the current accepted set including manual overrides."""
        if not self._scores:
            return set()
        filtered = self._apply_filter(self._scores)
        accepted = {s.file_path for s in filtered if s.accepted}
        return self._apply_manual_overrides(accepted)

    # ── Manual override buttons ───────────────────────────────────────────────

    def _on_selection_changed(self):
        has_sel = bool(self._table.selectedRows()) if hasattr(self._table, "selectedRows") \
            else bool(self._table.selectionModel().selectedRows())
        for btn in (self._accept_sel_btn, self._reject_sel_btn, self._clear_override_btn):
            btn.setEnabled(has_sel and bool(self._scores))

    def _selected_paths(self) -> list[str]:
        """Return file paths for all selected rows."""
        paths = []
        seen_rows = set()
        for idx in self._table.selectionModel().selectedRows():
            row = idx.row()
            if row in seen_rows or row >= len(self._scores):
                continue
            seen_rows.add(row)
            paths.append(self._scores[row].file_path)
        return paths

    def _force_accept_selected(self):
        for path in self._selected_paths():
            self._manual_overrides[path] = True
        self._reapply_filter()

    def _force_reject_selected(self):
        for path in self._selected_paths():
            self._manual_overrides[path] = False
        self._reapply_filter()

    def _clear_override_selected(self):
        for path in self._selected_paths():
            self._manual_overrides.pop(path, None)
        self._reapply_filter()

    # ── Load ──────────────────────────────────────────────────────────────────

    def _load_frames(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Light Frames", "",
            "Images (*.fits *.fit *.fts *.FTS *.xisf *.tif *.tiff *.png)"
        )
        if paths:
            self._set_paths(paths)

    def _load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder of Light Frames")
        if not folder:
            return
        p = Path(folder)
        paths = []
        for ext in ("*.fits", "*.fit", "*.fts", "*.FTS", "*.xisf", "*.tif", "*.tiff"):
            paths.extend(str(f) for f in sorted(p.glob(ext)))
        if paths:
            self._set_paths(paths)

    def _clear_all_frames(self):
        self._paths = []
        self._scores = []
        self._manual_overrides.clear()
        self._table.setRowCount(0)
        self._score_btn.setEnabled(False)
        self._clear_all_btn.setEnabled(False)
        self._frame_count_label.setText("No frames loaded")
        self._summary_label.setText("")

    def _set_paths(self, paths: list[str]):
        self._paths = paths
        self._scores = []
        self._manual_overrides.clear()
        self._score_btn.setEnabled(True)
        self._clear_all_btn.setEnabled(True)
        self._summary_label.setText("")
        n = len(paths)
        self._frame_count_label.setText(f"{n} frame{'s' if n != 1 else ''} loaded")
        self._table.setSortingEnabled(False)
        self._table.setRowCount(n)
        for row, path in enumerate(paths):
            ph = QPixmap(_THUMB_SIZE, _THUMB_SIZE)
            ph.fill(QColor(30, 30, 30))
            lbl = _ThumbLabel()
            lbl.setPixmap(ph)
            self._table.setCellWidget(row, self._COL_THUMB, lbl)
            name_item = QTableWidgetItem(Path(path).name)
            name_item.setForeground(QColor(160, 160, 160))
            self._table.setItem(row, self._COL_FILE, name_item)
            # Blank placeholders for metric columns
            for col in (self._COL_FWHM, self._COL_ECC, self._COL_SNR,
                        self._COL_BG, self._COL_STARS, self._COL_SCORE):
                self._table.setItem(row, col, QTableWidgetItem("…"))
            self._table.setItem(row, self._COL_STATUS, QTableWidgetItem("Pending"))
        self._table.setSortingEnabled(True)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _run_scoring(self):
        if not self._paths:
            return
        params = SubframeSelectorParams(
            fwhm_weight=self._fwhm_w.value(),
            eccentricity_weight=self._ecc_w.value(),
            snr_weight=self._snr_w.value(),
            stars_weight=self._stars_w.value(),
            rejection_sigma=self._rejection_sigma_spin.value(),
        )

        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._score_btn.setEnabled(False)
        self._manual_overrides.clear()

        self._worker = _ScoreWorker(self._paths, params)
        self._worker.progress.connect(lambda f, _m: self._progress.setValue(int(f * 100)))
        self._worker.frame_measured.connect(self._on_frame_measured)
        self._worker.thumbnail.connect(self._on_thumbnail)
        self._worker.finished.connect(self._on_scored)
        self._worker.error.connect(lambda msg: self._progress.setFormat(f"Error: {msg}"))
        self._worker.start()

    def _on_frame_measured(self, row: int, metrics: dict):
        """Update a single row with raw metrics as soon as that frame finishes."""
        if row >= self._table.rowCount():
            return
        # Temporarily disable sorting so row stays in place during update
        sorting_was = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)

        dim = QColor(180, 180, 180)

        def _ni(text: str) -> _NumericItem:
            it = _NumericItem(text)
            it.setForeground(dim)
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            return it

        self._table.setItem(row, self._COL_FWHM,  _ni(f"{metrics.get('fwhm', 0):.2f}"))
        self._table.setItem(row, self._COL_ECC,   _ni(f"{metrics.get('eccentricity', 0):.3f}"))
        self._table.setItem(row, self._COL_SNR,   _ni(f"{metrics.get('snr', 0):.1f}"))
        self._table.setItem(row, self._COL_BG,    _ni(f"{metrics.get('background', 0):.4f}"))
        self._table.setItem(row, self._COL_STARS, _ni(str(metrics.get('n_stars', 0))))
        status_it = QTableWidgetItem("Measuring…")
        status_it.setForeground(QColor(180, 160, 80))
        status_it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(row, self._COL_STATUS, status_it)

        self._table.setSortingEnabled(sorting_was)

    def _on_thumbnail(self, row: int, pix: QPixmap):
        if row >= self._table.rowCount():
            return
        existing = self._table.cellWidget(row, self._COL_THUMB)
        if isinstance(existing, _ThumbLabel):
            existing.setPixmap(pix)
            existing.set_full_pixmap(pix)
        else:
            label = _ThumbLabel()
            label.setPixmap(pix)
            label.set_full_pixmap(pix)
            self._table.setCellWidget(row, self._COL_THUMB, label)

    def _on_scored(self, scores: list[SubframeScore]):
        self.scores_ready.emit(scores)
        self._scores = scores
        self._progress.setVisible(False)
        self._score_btn.setEnabled(True)

        filtered = self._apply_filter(scores)
        accepted_paths = {s.file_path for s in filtered if s.accepted}
        accepted_paths = self._apply_manual_overrides(accepted_paths)

        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(scores))

        for row, s in enumerate(scores):
            accepted = s.file_path in accepted_paths
            override = self._manual_overrides.get(s.file_path)
            self._set_row_scored(row, s, accepted, override)

        self._table.setSortingEnabled(True)
        self._update_summary(scores, accepted_paths)

    def _set_row_scored(self, row: int, s: SubframeScore, accepted: bool, override=None):
        """Fill a row with final scores and colour it accept/reject."""
        color = QColor(200, 255, 200) if accepted else QColor(255, 150, 150)
        bg    = QColor(30, 50, 30)   if accepted else QColor(50, 30, 30)

        def _item(text, numeric=False):
            it = _NumericItem(text) if numeric else QTableWidgetItem(text)
            it.setForeground(color)
            it.setBackground(bg)
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            return it

        self._table.setItem(row, self._COL_FILE,
                            _item(Path(s.file_path).name, numeric=False))
        self._table.item(row, self._COL_FILE).setTextAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._table.item(row, self._COL_FILE).setForeground(color)
        self._table.item(row, self._COL_FILE).setBackground(bg)
        self._table.setItem(row, self._COL_FWHM,  _item(f"{s.fwhm:.2f}", numeric=True))
        self._table.setItem(row, self._COL_ECC,   _item(f"{s.eccentricity:.3f}", numeric=True))
        self._table.setItem(row, self._COL_SNR,   _item(f"{s.snr:.1f}", numeric=True))
        self._table.setItem(row, self._COL_BG,    _item(f"{s.background:.4f}", numeric=True))
        self._table.setItem(row, self._COL_STARS, _item(str(s.n_stars), numeric=True))
        self._table.setItem(row, self._COL_SCORE, _item(f"{s.quality_score:.3f}", numeric=True))

        if override is True:
            status_text = "Accepted*"
        elif override is False:
            status_text = "Rejected*"
        else:
            status_text = "Accepted" if accepted else "Rejected"
        self._table.setItem(row, self._COL_STATUS, _item(status_text))

        widget = self._table.cellWidget(row, self._COL_THUMB)
        if widget:
            widget.setStyleSheet(f"background: {'#1a3a1a' if accepted else '#3a1a1a'};")

    def _colour_rows(self, scores: list[SubframeScore], accepted_paths: set) -> None:
        for row, s in enumerate(scores):
            if row >= self._table.rowCount():
                break
            accepted = s.file_path in accepted_paths
            override = self._manual_overrides.get(s.file_path)
            color = QColor(200, 255, 200) if accepted else QColor(255, 150, 150)
            bg    = QColor(30, 50, 30)   if accepted else QColor(50, 30, 30)

            for col in range(self._COL_FILE, self._table.columnCount()):
                it = self._table.item(row, col)
                if it:
                    it.setForeground(color)
                    it.setBackground(bg)

            status_it = self._table.item(row, self._COL_STATUS)
            if status_it:
                if override is True:
                    status_it.setText("Accepted*")
                elif override is False:
                    status_it.setText("Rejected*")
                else:
                    status_it.setText("Accepted" if accepted else "Rejected")

            widget = self._table.cellWidget(row, self._COL_THUMB)
            if widget:
                widget.setStyleSheet(f"background: {'#1a3a1a' if accepted else '#3a1a1a'};")

        self._update_summary(scores, accepted_paths)

    def _update_summary(self, scores: list[SubframeScore], accepted_paths: set) -> None:
        n_accepted = len(accepted_paths)
        total = len(scores)
        pct = 100 * n_accepted / max(total, 1)
        n_overridden = len(self._manual_overrides)
        override_note = f"  ({n_overridden} manual override{'s' if n_overridden != 1 else ''})" \
            if n_overridden else ""
        self._summary_label.setText(
            f"{n_accepted}/{total} frames accepted ({pct:.0f}%){override_note}  —  "
            f"green = accepted, red = rejected, * = manual override"
        )

    # ── Emit ──────────────────────────────────────────────────────────────────

    def _emit_accepted(self):
        if not self._scores:
            return
        accepted_paths = self._current_accepted_paths()
        accepted = [s.file_path for s in self._scores if s.file_path in accepted_paths]
        if accepted:
            self.accepted_frames.emit(accepted, len(self._paths))
            self.accept()
