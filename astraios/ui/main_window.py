"""Main Window — 4-panel layout: project(left), canvas(center), tools(right), log(bottom)."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSettings, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QDragEnterEvent, QDropEvent, QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import astraios
from astraios.core.abe import abe_extract
from astraios.core.background import extract_background
from astraios.core.background_neutralization import background_neutralization
from astraios.core.banding import banding_reduction
from astraios.core.calibration import (
    calibrate_lights_batch,
    create_master_bias,
    create_master_dark,
    create_master_flat,
)
from astraios.core.channels import extract_luminance, split_channels
from astraios.core.chromatic_aberration import correct_chromatic_aberration
from astraios.core.color_calibration import color_calibrate
from astraios.core.color_tools import color_adjust, scnr
from astraios.core.cosmetic import cosmetic_correction
from astraios.core.curves import CurvesParams, curves_transform
from astraios.core.deconvolution import (
    SpatialDeconvParams,
    richardson_lucy,
    richardson_lucy_spatial,
)
from astraios.core.denoise import denoise
from astraios.core.device_manager import get_device_manager
from astraios.core.equipment import EquipmentProfile
from astraios.core.filters import median_filter, unsharp_mask
from astraios.core.histogram_transform import histogram_transform
from astraios.core.image_io import (
    FrameType,
    ImageData,
    _guess_frame_type,
    auto_stretch_for_display_ref,
    load_image,
    save_image,
)
from astraios.core.local_contrast import local_contrast_enhance
from astraios.core.masks import Mask
from astraios.core.morphology import morphology_transform
from astraios.core.presets import load_default_presets
from astraios.core.project import Project
from astraios.core.scripting import MacroRecorder, load_macro, play_macro, save_macro
from astraios.core.stacking import (
    IntegrationMethod,
    StackingParams,
    align_frames,
    align_from_paths,
    stack_from_paths,
    stack_images,
)
from astraios.core.star_reduction import reduce_stars
from astraios.core.statistics import compute_image_statistics
from astraios.core.stretch import (
    arcsinh_stretch,
    auto_stretch,
    compute_histogram,
    generalized_hyperbolic_stretch,
)
from astraios.core.subframe_selector import SubframeSelectorParams
from astraios.core.transforms import (
    bin_image,
    crop,
    flip,
    invert,
    resize,
    rotate,
)
from astraios.core.undo import AstraiosUndoStack
from astraios.core.vignette import correct_vignette
from astraios.core.wcs import normalise_wcs_dict
from astraios.core.wavelets import wavelet_sharpen
from astraios.ui.panels.project_panel import ProjectPanel
from astraios.ui.panels.tools_panel import ToolsPanel
from astraios.ui.widgets.histogram import HistogramWidget
from astraios.ui.widgets.image_canvas import ImageCanvas
from astraios.ui.widgets.log_panel import LogPanel, QtLogHandler
from astraios.ui.widgets.tweaks_panel import TweaksPanel
from astraios.ui.widgets.workflow_bar import WorkflowBar

log = logging.getLogger(__name__)


def _align_with_optional_filter(
    paths,
    output_dir,
    stk_params=None,
    filename_prefix="aligned",
    progress=None,
):
    """Worker-thread wrapper: align frames from disk paths."""
    from pathlib import Path


    def _prog(frac, msg):
        if progress is not None:
            progress(frac, msg)

    _prog(0.0, f"Aligning {len(paths)} frames...")
    return align_from_paths(
        [Path(p) for p in paths],
        Path(output_dir),
        params=stk_params,
        filename_prefix=filename_prefix,
        progress=lambda f, m: _prog(f, m),
    )


def _score_and_stack_worker(
    aligned_paths,
    params,
    cached_scores=None,
    progress=None,
):
    """Worker-thread wrapper: score for Weighted Average weights + stack.

    If integration mode is WEIGHTED_AVERAGE, frames are scored and weights
    are assigned proportional to quality score.  Otherwise stacking runs
    directly — frame selection is done up-front via the Subframe Selector.

    cached_scores: optional dict mapping str path → score dict (from project JSON).
    When all frames have cached scores the scoring step is skipped entirely.
    """
    from astraios.core.stacking import IntegrationMethod
    from astraios.core.subframe_selector import SubframeSelectorParams, score_subframes

    def _prog(frac, msg):
        if progress:
            progress(frac, msg)

    final_paths = list(aligned_paths)
    needs_scoring = (
        hasattr(params, "integration") and params.integration == IntegrationMethod.WEIGHTED_AVERAGE
    )

    if needs_scoring:
        str_paths = [str(p) for p in final_paths]
        if cached_scores and all(p in cached_scores for p in str_paths):
            log.info(
                "Using cached frame scores for weighted stacking (%d frames)", len(str_paths)
            )
            weights = [
                max(float(cached_scores[p].get("quality_score", 0.5)), 1e-6)
                for p in str_paths
            ]
            _prog(0.45, "Using cached scores…")
        else:
            scores = score_subframes(
                str_paths,
                SubframeSelectorParams(),
                progress=lambda f, m: _prog(f * 0.45, m),
            )
            weights = [max(float(sc.quality_score), 1e-6) for sc in scores]
        log.info(
            "Frame weights: min=%.3f  max=%.3f  (%d frames)",
            min(weights), max(weights), len(weights),
        )
        params.frame_weights = weights

    stack_progress_start = 0.45 if needs_scoring else 0.0

    def _stack_prog(frac, msg):
        _prog(stack_progress_start + frac * (1.0 - stack_progress_start), msg)

    return stack_from_paths(final_paths, params=params, progress=_stack_prog)


class _ProcessingCancelled(BaseException):
    """Raised inside the worker thread when the user cancels an operation."""


class _AstraiosLogo(QWidget):
    """SVG-equivalent logo matching the HTML prototype."""

    def __init__(self, size: int = 18, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def paintEvent(self, event):
        from PyQt6.QtCore import QPointF, QRectF
        from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        accent = QColor("#2ea043")

        def _pen(alpha: int, width: float) -> QPen:
            c = QColor(accent)
            c.setAlpha(alpha)
            return QPen(c, width)

        p.setBrush(Qt.BrushStyle.NoBrush)
        # Outer ring
        p.setPen(_pen(255, 1.2))
        r1 = cx - 0.5
        p.drawEllipse(QRectF(cx - r1, cy - r1, r1 * 2, r1 * 2))
        # Middle ring (60% opacity)
        p.setPen(_pen(153, 0.8))
        r2 = cx * 0.5
        p.drawEllipse(QRectF(cx - r2, cy - r2, r2 * 2, r2 * 2))
        # Crosshair lines (30% opacity)
        p.setPen(_pen(77, 0.5))
        p.drawLine(QPointF(0, cy), QPointF(w, cy))
        p.drawLine(QPointF(cx, 0), QPointF(cx, h))
        # Centre dot
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(accent))
        p.drawEllipse(QRectF(cx - 1.5, cy - 1.5, 3, 3))
        p.end()


class ProcessingWorker(QThread):
    """Runs processing tasks off the main thread."""

    progress = pyqtSignal(float, str)
    finished = pyqtSignal(object)  # result
    error = pyqtSignal(str)
    elapsed = pyqtSignal(float)    # seconds
    cancelled = pyqtSignal()

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self._cancel_event = threading.Event()
        self._t0: float = 0.0

    def cancel(self):
        self._cancel_event.set()
        self.requestInterruption()

    def run(self):
        self._t0 = time.monotonic()
        try:
            self._kwargs["progress"] = self._emit_progress
            result = self._func(*self._args, **self._kwargs)
            self.elapsed.emit(time.monotonic() - self._t0)
            self.finished.emit(result)
        except _ProcessingCancelled:
            self.elapsed.emit(time.monotonic() - self._t0)
            self.cancelled.emit()
        except Exception as e:
            self.elapsed.emit(time.monotonic() - self._t0)
            log.exception("Processing error")
            self.error.emit(str(e))

    def _emit_progress(self, fraction: float, message: str):
        if self._cancel_event.is_set():
            raise _ProcessingCancelled()
        eta_str = ""
        if fraction > 0.02:
            elapsed = time.monotonic() - self._t0
            remaining = elapsed / fraction * (1.0 - fraction)
            if remaining >= 2:
                eta_str = (
                    f"  — ETA {remaining:.0f}s"
                    if remaining < 120
                    else f"  — ETA {remaining / 60:.1f}min"
                )
        self.progress.emit(fraction, message + eta_str)


class _MsFolderLoader(QThread):
    """Loads frames for a multi-session folder in the background."""

    progress = pyqtSignal(float, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, paths: list[str], parent=None):
        super().__init__(parent)
        self._paths = paths
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        from pathlib import Path

        from astraios.core.image_io import load_image
        loaded = []
        total = len(self._paths)
        for i, fpath in enumerate(self._paths):
            if self._cancelled:
                return
            try:
                img = load_image(fpath)
                if img is not None:
                    loaded.append(img)
            except Exception as exc:
                self.error.emit(f"Skipped {Path(fpath).name}: {exc}")
            if total > 1:
                self.progress.emit((i + 1) / total, f"Loading {Path(fpath).name}")
        if not self._cancelled:
            self.finished.emit(loaded)


class MainWindow(QMainWindow):
    """The main Astraios application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{astraios.__app_name__} v{astraios.__version__}")
        self._set_app_icon()
        self.setMinimumSize(1200, 750)
        self.resize(1600, 950)

        self._project: Project | None = None
        self._current_image: ImageData | None = None
        self._worker: ProcessingWorker | None = None
        self._master_bias: ImageData | None = None
        self._master_dark: ImageData | None = None
        self._master_flat: ImageData | None = None
        self._calibrated_lights: list[ImageData] = []
        # True when the current image has unsaved edits since the last export.
        self._dirty: bool = False

        # Mask registry + the mask currently applied to processing (None = whole image)
        self._masks: list[Mask] = []
        self._active_mask: Mask | None = None

        # Macro recording
        self._macro_recorder = MacroRecorder()
        self._current_macro = None  # last recorded/loaded Pipeline

        # Star layer extracted by StarNet (for the starless+stars blend workflow)
        self._extracted_stars: np.ndarray | None = None

        # Equipment profile for Smart Processor
        self._equipment_profile: EquipmentProfile | None = None

        # Undo/Redo stack
        self._undo_stack = AstraiosUndoStack()
        # Single-element list as mutable reference for undo commands
        self._image_ref: list[ImageData | None] = [None]
        self._undo_stack.set_target(self._image_ref)

        # Show the getting-started guide once, after the window is up.
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(500, self._maybe_first_run_welcome)

        # Cached downscaled image for live preview (recomputed in _display_image)
        self._preview_small_cache: tuple | None = None  # (small_array, scale)
        # Stretch reference used for the current display — None means use image itself
        self._preview_stretch_ref_cache = None  # np.ndarray | None

        # Processing Graph (non-destructive DAG)
        self._processing_graph = None
        self._skip_graph_auto_add = False

        # Preview debounce timers
        self._stretch_preview_timer = QTimer()
        self._stretch_preview_timer.setSingleShot(True)
        self._stretch_preview_timer.setInterval(100)
        self._stretch_preview_timer.timeout.connect(self._do_stretch_preview)
        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(100)
        self._preview_timer.timeout.connect(self._do_preview_requested)
        self._pending_preview_tool: str | None = None

        # Dynamic background extraction samples (image-space coords)
        self._bg_samples: list[tuple[float, float]] = []

        # WCS overlay data (image-space x, y, magnitude)
        self._wcs_overlay_stars: list[tuple[float, float, float]] = []
        self._constellation_segments: list[list[tuple[float, float]]] = []
        self._current_wcs: dict = {}   # last solved WCS dict

        # Python console dock (lazy init)
        self._python_console_dock = None

        # Multi-session stacking
        self._ms_sessions: list = []  # list of SessionGroup objects

        # Paths to saved aligned frames for memory-efficient stacking
        self._aligned_paths: list = []
        # Paths overridden by Subframe Selector dialog (subset of aligned frames)
        self._subframe_selected_paths: list[str] = []

        # Blink comparator
        self._blink_images: list = [None, None]  # [A, B] — display RGB uint8 arrays (H,W,3)
        self._blink_names: list[str] = ["", ""]
        self._blink_index = 0
        self._blink_timer = QTimer()
        self._blink_timer.timeout.connect(self._blink_tick)

        # Autosave every 5 minutes when a project is open
        self._autosave_timer = QTimer()
        self._autosave_timer.setInterval(5 * 60 * 1000)
        self._autosave_timer.timeout.connect(self._autosave_project)
        self._autosave_timer.start()

        # Register tools for preset system
        load_default_presets()
        from astraios.plugins.base import scan_plugins
        scan_plugins()

        self.setAcceptDrops(True)
        self._setup_menu()
        self._setup_toolbar()
        self._setup_ui()
        self._setup_logging()
        self._setup_statusbar()

    def _setup_menu(self):
        menu = self.menuBar()

        # ── Left corner widget: logo + version badge ──────────────────────────
        left_corner = QWidget()
        left_corner.setStyleSheet("background: transparent;")
        left_layout = QHBoxLayout(left_corner)
        left_layout.setContentsMargins(8, 0, 8, 0)
        left_layout.setSpacing(6)
        logo_widget = _AstraiosLogo(18)
        logo_label = QLabel("Astraios")
        logo_label.setStyleSheet(
            "color: #e6edf3; font-size: 13px; font-weight: 700; background: transparent;"
        )
        version_badge = QLabel(f"v{astraios.__version__}")
        version_badge.setStyleSheet(
            "color: #8b949e; font-size: 10px; padding: 1px 5px;"
        )
        left_layout.addWidget(logo_widget)
        left_layout.addWidget(logo_label)
        left_layout.addWidget(version_badge)
        menu.setCornerWidget(left_corner, Qt.Corner.TopLeftCorner)

        # ── Right corner widget: GPU chip + RAM chip + Export ─────────────────
        right_corner = QWidget()
        right_corner.setStyleSheet("background-color: transparent;")
        right_layout = QHBoxLayout(right_corner)
        right_layout.setContentsMargins(4, 0, 8, 0)
        right_layout.setSpacing(6)
        self._gpu_chip_label = QLabel("")
        self._gpu_chip_label.setStyleSheet(
            "color: #8b949e; font-size: 11px; padding: 1px 4px;"
        )
        self._ram_chip_label = QLabel("")
        self._ram_chip_label.setStyleSheet(
            "color: #8b949e; font-size: 11px; padding: 1px 4px;"
        )
        export_btn = QPushButton("⬆ Export Image…")
        export_btn.setStyleSheet(
            "QPushButton { color: #ffffff; background: #2ea043; border: none; border-radius: 4px;"
            " font-size: 11px; font-weight: 600; padding: 3px 10px; }"
            " QPushButton:hover { background: #3fb950; }"
        )
        export_btn.clicked.connect(self._save_image)
        right_layout.addWidget(self._gpu_chip_label)
        right_layout.addWidget(self._ram_chip_label)
        right_layout.addWidget(export_btn)
        menu.setCornerWidget(right_corner, Qt.Corner.TopRightCorner)

        # ── File menu ─────────────────────────────────────────────────────────
        file_menu = menu.addMenu("&File")

        self._new_proj_act = QAction("&New Project...", self)
        self._new_proj_act.setShortcut("Ctrl+N")
        self._new_proj_act.triggered.connect(self._new_project)
        file_menu.addAction(self._new_proj_act)

        self._open_proj_act = QAction("&Open Project...", self)
        self._open_proj_act.setShortcut("Ctrl+O")
        self._open_proj_act.triggered.connect(self._open_project)
        file_menu.addAction(self._open_proj_act)

        save_proj = QAction("&Save Project", self)
        save_proj.setShortcut("Ctrl+S")
        save_proj.triggered.connect(self._save_project)
        file_menu.addAction(save_proj)

        save_as_act = QAction("Save Project &As...", self)
        save_as_act.setShortcut("Ctrl+Shift+A")
        save_as_act.triggered.connect(self._save_project_as)
        file_menu.addAction(save_as_act)

        file_menu.addSeparator()

        self._open_img_act = QAction("Open &Image...", self)
        self._open_img_act.setShortcut("Ctrl+Shift+I")
        self._open_img_act.triggered.connect(self._open_image)
        file_menu.addAction(self._open_img_act)

        import_lights = QAction("Import &Lights...", self)
        import_lights.triggered.connect(self._on_import_lights)
        file_menu.addAction(import_lights)

        import_cal = QAction("Import &Calibration Frames...", self)
        import_cal.triggered.connect(self._on_import_calibration)
        file_menu.addAction(import_cal)

        file_menu.addSeparator()

        export_fits = QAction("Export as &FITS...", self)
        export_fits.triggered.connect(self._on_export_fits)
        file_menu.addAction(export_fits)

        export_tiff = QAction("Export as &TIFF...", self)
        export_tiff.triggered.connect(self._on_export_tiff)
        file_menu.addAction(export_tiff)

        export_png = QAction("Export as &PNG...", self)
        export_png.triggered.connect(self._on_export_png)
        file_menu.addAction(export_png)

        export_full = QAction("Export Image &As...", self)
        export_full.setShortcut("Ctrl+Shift+S")
        export_full.triggered.connect(self._save_image)
        file_menu.addAction(export_full)

        file_menu.addSeparator()

        prefs_act = QAction("&Preferences...", self)
        prefs_act.setShortcut("Ctrl+,")
        prefs_act.triggered.connect(self._show_preferences)
        file_menu.addAction(prefs_act)

        file_menu.addSeparator()

        quit_act = QAction("E&xit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # ── Edit menu ─────────────────────────────────────────────────────────
        edit_menu = menu.addMenu("&Edit")

        self._undo_act = QAction("&Undo", self)
        self._undo_act.setShortcut("Ctrl+Z")
        self._undo_act.triggered.connect(self._undo)
        self._undo_act.setEnabled(False)
        edit_menu.addAction(self._undo_act)

        self._redo_act = QAction("Re&do", self)
        self._redo_act.setShortcut("Ctrl+Shift+Z")
        self._redo_act.triggered.connect(self._redo)
        self._redo_act.setEnabled(False)
        edit_menu.addAction(self._redo_act)

        edit_menu.addSeparator()

        clear_undo = QAction("&Clear History", self)
        clear_undo.triggered.connect(self._clear_undo_history)
        edit_menu.addAction(clear_undo)

        edit_menu.addSeparator()

        stats_act = QAction("Image &Statistics...", self)
        stats_act.setShortcut("Ctrl+I")
        stats_act.triggered.connect(self._on_show_statistics)
        edit_menu.addAction(stats_act)

        fits_hdr = QAction("Edit FITS &Header...", self)
        fits_hdr.setShortcut("Ctrl+Shift+H")
        fits_hdr.triggered.connect(self._show_fits_header)
        edit_menu.addAction(fits_hdr)

        # ── View menu ─────────────────────────────────────────────────────────
        view_menu = menu.addMenu("&View")

        zoom_in_act = QAction("Zoom &In", self)
        zoom_in_act.setShortcut("+")
        zoom_in_act.triggered.connect(lambda: self._canvas.zoom_in())
        view_menu.addAction(zoom_in_act)
        self._zoom_in_act = zoom_in_act

        zoom_out_act = QAction("Zoom &Out", self)
        zoom_out_act.setShortcut("-")
        zoom_out_act.triggered.connect(lambda: self._canvas.zoom_out())
        view_menu.addAction(zoom_out_act)
        self._zoom_out_act = zoom_out_act

        fit_act = QAction("&Fit to Window", self)
        fit_act.setShortcut("F")
        fit_act.triggered.connect(lambda: self._canvas.fit_to_window())
        view_menu.addAction(fit_act)
        self._fit_act = fit_act

        fit_shortcut = QShortcut(QKeySequence("Ctrl+0"), self)
        fit_shortcut.activated.connect(lambda: self._canvas.fit_to_window())

        zoom100 = QAction("Zoom &100%", self)
        zoom100.setShortcut("1")
        zoom100.triggered.connect(lambda: self._canvas.zoom_to(1.0))
        view_menu.addAction(zoom100)

        zoom200 = QAction("Zoom &200%", self)
        zoom200.setShortcut("2")
        zoom200.triggered.connect(lambda: self._canvas.zoom_to(2.0))
        view_menu.addAction(zoom200)

        view_menu.addSeparator()

        split_act = QAction("&Before/After Split", self)
        split_act.setShortcut("B")
        split_act.setCheckable(True)
        split_act.toggled.connect(self._on_split_view_toggled)
        view_menu.addAction(split_act)
        self._split_act = split_act

        toggle_hist = QAction("Toggle &Histogram", self)
        toggle_hist.setShortcut("H")
        toggle_hist.triggered.connect(self._on_toggle_histogram)
        view_menu.addAction(toggle_hist)

        fullscreen_act = QAction("Full &Screen", self)
        fullscreen_act.setShortcut("F11")
        fullscreen_act.triggered.connect(self._on_fullscreen)
        view_menu.addAction(fullscreen_act)

        view_menu.addSeparator()

        history_act = QAction("&Processing History", self)
        history_act.setShortcut("Ctrl+H")
        history_act.triggered.connect(self._show_processing_graph)
        view_menu.addAction(history_act)

        view_menu.addSeparator()

        color_mgmt_act = QAction("&Color Management...", self)
        color_mgmt_act.triggered.connect(self._show_color_management)
        view_menu.addAction(color_mgmt_act)

        # ── Tools menu ────────────────────────────────────────────────────────
        tools_menu = menu.addMenu("&Tools")

        smart_act = QAction("&Smart Processor...", self)
        smart_act.setShortcut("Ctrl+Shift+P")
        smart_act.triggered.connect(self._show_smart_processor_dialog)
        tools_menu.addAction(smart_act)

        preprocess_act = QAction("&Batch Preprocessing...", self)
        preprocess_act.setShortcut("Ctrl+Shift+B")
        preprocess_act.triggered.connect(self._show_batch_preprocess_dialog)
        tools_menu.addAction(preprocess_act)

        batch_act = QAction("&Batch Processing...", self)
        batch_act.setShortcut("Ctrl+B")
        batch_act.triggered.connect(self._show_batch_dialog)
        tools_menu.addAction(batch_act)

        subframe_act = QAction("Su&bframe Selector...", self)
        subframe_act.triggered.connect(self._on_open_subframe_selector)
        tools_menu.addAction(subframe_act)

        blink_act = QAction("Blin&k Comparator", self)
        blink_act.triggered.connect(
            lambda: self._tools_panel._tab_widget.setCurrentIndex(
                self._tools_panel._tab_widget.count() - 1
            )
        )
        tools_menu.addAction(blink_act)

        blink_frame_act = QAction("Blin&k Frame Browser...", self)
        blink_frame_act.triggered.connect(self._show_blink_dialog)
        tools_menu.addAction(blink_frame_act)

        tools_menu.addSeparator()

        plate_act = QAction("&Plate Solve...", self)
        plate_act.triggered.connect(self._on_plate_solve_from_menu)
        tools_menu.addAction(plate_act)

        dso_act = QAction("&DSO Annotation", self)
        dso_act.triggered.connect(self._on_toggle_dso_overlay)
        tools_menu.addAction(dso_act)

        wcs_act = QAction("&WCS Overlay", self)
        wcs_act.triggered.connect(self._on_toggle_wcs_overlay)
        tools_menu.addAction(wcs_act)

        const_act = QAction("&Constellation Lines", self)
        const_act.triggered.connect(self._on_toggle_constellation_overlay)
        tools_menu.addAction(const_act)

        tools_menu.addSeparator()

        pm_act = QAction("&Pixel Math...", self)
        pm_act.setShortcut("Ctrl+P")
        pm_act.triggered.connect(self._show_pixelmath_dialog)
        tools_menu.addAction(pm_act)

        nb_act = QAction("&Narrowband Combine...", self)
        nb_act.triggered.connect(self._show_narrowband_dialog)
        tools_menu.addAction(nb_act)

        blend_act = QAction("Image &Blend...", self)
        blend_act.triggered.connect(self._show_blend_dialog)
        tools_menu.addAction(blend_act)

        hdr_act = QAction("&HDR Composition...", self)
        hdr_act.triggered.connect(self._show_hdr_dialog)
        tools_menu.addAction(hdr_act)

        create_mask = QAction("Create &Mask...", self)
        create_mask.triggered.connect(self._show_mask_dialog)
        tools_menu.addAction(create_mask)

        clear_mask = QAction("Clear Active Mask", self)
        clear_mask.triggered.connect(self._clear_active_mask)
        tools_menu.addAction(clear_mask)

        mosaic_act = QAction("&Mosaic Stitching...", self)
        mosaic_act.triggered.connect(self._show_mosaic_dialog)
        tools_menu.addAction(mosaic_act)

        tools_menu.addSeparator()

        ez_act = QAction("&EZ Script Suite...", self)
        ez_act.triggered.connect(self._show_ez_script_dialog)
        tools_menu.addAction(ez_act)

        tools_menu.addSeparator()

        chmatch_act = QAction("&Channel Match...", self)
        chmatch_act.triggered.connect(self._show_channel_match_dialog)
        tools_menu.addAction(chmatch_act)

        lens_act = QAction("&Lens Distortion Correction...", self)
        lens_act.triggered.connect(self._show_lens_distortion_dialog)
        tools_menu.addAction(lens_act)

        tools_menu.addSeparator()

        equip_act = QAction("&Equipment Profile...", self)
        equip_act.triggered.connect(self._show_equipment_dialog)
        tools_menu.addAction(equip_act)

        tools_menu.addSeparator()

        macro_start = QAction("Start &Recording", self)
        macro_start.triggered.connect(self._on_start_macro)
        tools_menu.addAction(macro_start)

        macro_stop = QAction("Sto&p Recording", self)
        macro_stop.triggered.connect(self._on_stop_macro)
        tools_menu.addAction(macro_stop)

        macro_play = QAction("P&lay Macro", self)
        macro_play.triggered.connect(self._on_play_macro)
        tools_menu.addAction(macro_play)

        tools_menu.addSeparator()

        live_stack_act = QAction("&Live Stack...", self)
        live_stack_act.triggered.connect(self._show_live_stack_dialog)
        tools_menu.addAction(live_stack_act)

        tools_menu.addSeparator()

        console_act = QAction("Python &Console", self)
        console_act.triggered.connect(self._on_open_python_console)
        tools_menu.addAction(console_act)

        # ── Help menu ─────────────────────────────────────────────────────────
        help_menu = menu.addMenu("&Help")

        docs_act = QAction("&Documentation", self)
        docs_act.triggered.connect(self._on_open_docs)
        help_menu.addAction(docs_act)

        getting_started = QAction("&Getting Started", self)
        getting_started.triggered.connect(lambda: self._show_welcome(first_run=False))
        help_menu.addAction(getting_started)

        workflow_act = QAction("Processing &Workflow (online)", self)
        workflow_act.triggered.connect(self._on_open_docs)
        help_menu.addAction(workflow_act)

        help_menu.addSeparator()

        about_act = QAction("&About Astraios", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

        dm = get_device_manager()
        device_act = QAction(f"GPU: {dm.info.name}", self)
        device_act.setEnabled(False)
        help_menu.addAction(device_act)

        help_menu.addSeparator()

        coffee_act = QAction("Buy Me a Coffee ☕", self)
        coffee_act.triggered.connect(self._on_buy_coffee)
        help_menu.addAction(coffee_act)

    def _setup_toolbar(self):
        """Quick-action toolbar below the menu bar."""
        tb = QToolBar("Quick Actions")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setObjectName("QuickActionToolbar")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        self._quick_toolbar = tb

        def _tbtn(symbol: str, tip: str, shortcut: str = "") -> QToolButton:
            btn = QToolButton()
            btn.setText(symbol)
            btn.setToolTip(tip)
            if shortcut:
                btn.setShortcut(shortcut)
            return btn

        # Primary action: a clearly-visible Open button so a new user knows
        # where to start (the menu's Ctrl+Shift+I was un-discoverable).
        open_btn = QToolButton()
        open_btn.setText("Open Image")
        open_btn.setToolTip("Open an image to start  (Ctrl+O)")
        open_btn.setShortcut("Ctrl+O")
        open_btn.clicked.connect(self._open_image)
        open_btn.setStyleSheet(
            "QToolButton { background: #2ea043; color: #ffffff; font-weight: 700; "
            "border-radius: 4px; padding: 3px 12px; } "
            "QToolButton:hover { background: #3fb950; }"
        )
        tb.addWidget(open_btn)
        tb.addSeparator()

        undo_btn = _tbtn("⎌", "Undo  Ctrl+Z")
        undo_btn.clicked.connect(self._undo)
        tb.addWidget(undo_btn)
        self._tb_undo_btn = undo_btn

        redo_btn = _tbtn("↷", "Redo  Ctrl+Shift+Z")
        redo_btn.clicked.connect(self._redo)
        tb.addWidget(redo_btn)
        self._tb_redo_btn = redo_btn

        tb.addSeparator()

        zoom_in_btn = _tbtn("⊕", "Zoom In  +")
        zoom_in_btn.clicked.connect(lambda: self._canvas.zoom_in())
        tb.addWidget(zoom_in_btn)

        zoom_out_btn = _tbtn("⊖", "Zoom Out  −")
        zoom_out_btn.clicked.connect(lambda: self._canvas.zoom_out())
        tb.addWidget(zoom_out_btn)

        fit_btn = _tbtn("⊡", "Fit to Window  F")
        fit_btn.clicked.connect(lambda: self._canvas.fit_to_window())
        tb.addWidget(fit_btn)

        tb.addSeparator()

        split_btn = _tbtn("⟺", "Before/After Split  B")
        split_btn.setCheckable(True)
        split_btn.toggled.connect(self._on_split_view_toggled)
        tb.addWidget(split_btn)
        self._tb_split_btn = split_btn

        tb.addSeparator()

        const_btn = _tbtn("✦", "Toggle Constellation Lines  C")
        const_btn.setCheckable(True)
        const_btn.setObjectName("const_tb_btn")
        const_btn.toggled.connect(self._on_toggle_constellation_overlay)
        tb.addWidget(const_btn)
        self._tb_const_btn = const_btn

        tb.addSeparator()

        history_btn = _tbtn("⧉", "Processing History  Ctrl+H")
        history_btn.clicked.connect(self._show_processing_graph)
        tb.addWidget(history_btn)

        smart_btn = QToolButton()
        _icon_path = Path(__file__).resolve().parent.parent / "resources" / "icons" / "lightning.svg"
        if _icon_path.exists():
            smart_btn.setIcon(__import__("PyQt6.QtGui", fromlist=["QIcon"]).QIcon(str(_icon_path)))
            smart_btn.setIconSize(__import__("PyQt6.QtCore", fromlist=["QSize"]).QSize(18, 18))
        else:
            smart_btn.setText("⚡")
        smart_btn.setToolTip("Smart Processor  Ctrl+Shift+P")
        smart_btn.clicked.connect(self._show_smart_processor_dialog)
        tb.addWidget(smart_btn)

        preprocess_btn = _tbtn("⊞ Preprocess", "Batch Preprocessing  Ctrl+Shift+B")
        preprocess_btn.clicked.connect(self._show_batch_preprocess_dialog)
        tb.addWidget(preprocess_btn)

        batch_btn = _tbtn("⊞ Batch", "Batch Processing  Ctrl+B")
        batch_btn.clicked.connect(self._show_batch_dialog)
        tb.addWidget(batch_btn)

        # Stretch to push status + Tweaks to right
        spacer = QWidget()
        spacer.setStyleSheet("background: transparent;")
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        # Operation status: "Ready" label + thin 4px progress bar
        self._tb_status_label = QLabel("Ready")
        self._tb_status_label.setStyleSheet(
            "color: #2ea043; font-size: 10px; font-family: monospace; padding: 0 4px;"
        )
        tb.addWidget(self._tb_status_label)

        self._tb_progress_bar = QProgressBar()
        self._tb_progress_bar.setFixedSize(120, 4)
        self._tb_progress_bar.setRange(0, 1000)
        self._tb_progress_bar.setValue(0)
        self._tb_progress_bar.setTextVisible(False)
        self._tb_progress_bar.setStyleSheet(
            "QProgressBar { background: #21262d; border: none; border-radius: 2px; }"
            " QProgressBar::chunk { background: #2ea043; border-radius: 2px; }"
        )
        tb.addWidget(self._tb_progress_bar)

        tb.addSeparator()

        tweaks_btn = _tbtn("⚙ Tweaks", "UI Tweaks")
        tweaks_btn.clicked.connect(self._on_toggle_tweaks)
        tb.addWidget(tweaks_btn)

        # Tweaks panel (created here, shown/hidden on demand)
        self._tweaks_panel = TweaksPanel(self)
        self._tweaks_panel.hide()
        self._tweaks_panel.accent_changed.connect(self._on_tweaks_accent_changed)
        self._tweaks_panel.workflow_visible.connect(self._on_tweaks_workflow_visible)
        self._tweaks_panel.log_visible.connect(self._on_tweaks_log_visible)
        self._tweaks_panel.log_height_changed.connect(self._on_tweaks_log_height)

    def _set_app_icon(self):
        icon_path = Path(__file__).resolve().parent.parent / "resources" / "icons" / "astraios.svg"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def _setup_ui(self):
        # Central widget with splitters
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Workflow pipeline bar
        self._workflow_bar = WorkflowBar()
        self._workflow_bar.step_clicked.connect(self._on_workflow_step_clicked)
        # TweaksPanel checkbox defaults to checked=True, so show by default
        main_layout.addWidget(self._workflow_bar)

        # Top: horizontal splitter (project | canvas | tools)
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Project panel
        self._project_panel = ProjectPanel()
        self._project_panel.frame_selected.connect(self._load_frame)
        self._project_panel.frames_imported.connect(self._on_frames_imported)
        self._project_panel.plate_solve_clicked.connect(self._on_plate_solve_from_menu)
        self._project_panel.dso_overlay_clicked.connect(self._on_toggle_dso_overlay)
        self._project_panel.show_statistics.connect(self._on_show_statistics)
        self._project_panel.show_fits_header.connect(self._show_fits_header)
        top_splitter.addWidget(self._project_panel)

        # Center: Canvas toolbar + Canvas + histogram
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        # ── Canvas toolbar ──────────────────────────────────────────────────
        canvas_tb = QFrame()
        canvas_tb.setFixedHeight(32)
        canvas_tb.setObjectName("CanvasToolbar")
        canvas_tb.setStyleSheet(
            "#CanvasToolbar { background: #161b22; border-bottom: 1px solid #30363d; }"
        )
        tb_layout = QHBoxLayout(canvas_tb)
        tb_layout.setContentsMargins(6, 0, 6, 0)
        tb_layout.setSpacing(2)

        def _ctb(text: str, tip: str = "", checkable: bool = False) -> QToolButton:
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setCheckable(checkable)
            return b

        zoom_out_tb = _ctb("−", "Zoom Out")
        self._canvas_zoom_label = QLabel("100%")
        self._canvas_zoom_label.setFixedWidth(44)
        self._canvas_zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas_zoom_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        zoom_in_tb = _ctb("+", "Zoom In")
        fit_tb = _ctb("Fit", "Fit to Window")
        one_to_one_tb = _ctb("1:1", "100% Zoom")
        tb_layout.addWidget(zoom_out_tb)
        tb_layout.addWidget(self._canvas_zoom_label)
        tb_layout.addWidget(zoom_in_tb)
        tb_layout.addWidget(fit_tb)
        tb_layout.addWidget(one_to_one_tb)

        _sep1 = QFrame()
        _sep1.setFrameShape(QFrame.Shape.VLine)
        _sep1.setStyleSheet("color: #30363d;")
        tb_layout.addWidget(_sep1)

        # After / Before / Split — manually exclusive
        self._view_btn_group = QButtonGroup(canvas_tb)
        self._view_btn_group.setExclusive(False)
        after_tb = _ctb("After", "Show processed image", checkable=True)
        before_tb = _ctb("Before", "Show original image", checkable=True)
        split_tb = _ctb("⟺ Split", "Before/After split view", checkable=True)
        after_tb.setChecked(True)
        for _b in (after_tb, before_tb, split_tb):
            self._view_btn_group.addButton(_b)
            tb_layout.addWidget(_b)

        _sep2 = QFrame()
        _sep2.setFrameShape(QFrame.Shape.VLine)
        _sep2.setStyleSheet("color: #30363d;")
        tb_layout.addWidget(_sep2)

        self._grid_tb = _ctb("Grid", "Toggle grid overlay", checkable=True)
        self._wcs_tb = _ctb("WCS", "Toggle WCS star overlay", checkable=True)
        tb_layout.addWidget(self._grid_tb)
        tb_layout.addWidget(self._wcs_tb)

        spacer_tb = QWidget()
        spacer_tb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb_layout.addWidget(spacer_tb)

        self._hist_toggle_tb = _ctb("▾ Histogram", "Show/hide histogram", checkable=True)
        self._hist_toggle_tb.setChecked(True)
        tb_layout.addWidget(self._hist_toggle_tb)

        fullscreen_tb = _ctb("⛶", "Fullscreen  F11")
        fullscreen_tb.clicked.connect(self._on_fullscreen)
        tb_layout.addWidget(fullscreen_tb)

        self._canvas_coord_label = QLabel("")
        self._canvas_coord_label.setStyleSheet(
            "color: #8b949e; font-size: 11px; font-family: monospace;"
        )
        tb_layout.addWidget(self._canvas_coord_label)

        center_layout.addWidget(canvas_tb)

        # ── Canvas ─────────────────────────────────────────────────────────
        self._canvas = ImageCanvas()
        self._canvas.cursor_position.connect(self._update_pixel_readout)
        self._canvas.cursor_position.connect(self._update_canvas_coord_label)
        center_layout.addWidget(self._canvas, 1)

        # ── Histogram container ─────────────────────────────────────────────
        self._hist_container = QWidget()
        hist_v = QVBoxLayout(self._hist_container)
        hist_v.setContentsMargins(4, 2, 4, 2)
        hist_v.setSpacing(2)

        hist_header = QWidget()
        hist_header_layout = QHBoxLayout(hist_header)
        hist_header_layout.setContentsMargins(0, 0, 0, 0)
        hist_header_layout.setSpacing(4)

        self._hist_channel_group = QButtonGroup(hist_header)
        for _ch in ("RGB", "R", "G", "B", "L"):
            ch_btn = QPushButton(_ch)
            ch_btn.setCheckable(True)
            ch_btn.setFixedWidth(32)
            ch_btn.setStyleSheet(
                "QPushButton { background: transparent; border: none;"
                " border-bottom: 2px solid transparent;"
                " color: #8b949e; font-size: 11px; padding: 2px 0; }"
                " QPushButton:checked { color: #e6edf3; border-bottom-color: #2ea043; }"
            )
            self._hist_channel_group.addButton(ch_btn)
            hist_header_layout.addWidget(ch_btn)
        self._hist_channel_group.buttons()[0].setChecked(True)

        hist_header_layout.addStretch()

        self._hist_stats_label = QLabel("")
        self._hist_stats_label.setStyleSheet(
            "color: #8b949e; font-size: 11px; font-family: monospace;"
        )
        hist_header_layout.addWidget(self._hist_stats_label)

        hist_v.addWidget(hist_header)

        self._histogram = HistogramWidget()
        hist_v.addWidget(self._histogram)

        center_layout.addWidget(self._hist_container)

        # ── Wire canvas toolbar signals ────────────────────────────────────
        zoom_out_tb.clicked.connect(self._canvas.zoom_out)
        zoom_in_tb.clicked.connect(self._canvas.zoom_in)
        fit_tb.clicked.connect(self._canvas.fit_to_window)
        one_to_one_tb.clicked.connect(lambda: self._canvas.zoom_to(1.0))
        self._canvas.zoom_changed.connect(
            lambda z: self._canvas_zoom_label.setText(f"{int(z * 100)}%")
        )

        def _on_view_mode_toggled(toggled_btn: QToolButton, checked: bool):
            if checked:
                for _b in self._view_btn_group.buttons():
                    if _b is not toggled_btn:
                        _b.setChecked(False)
            if after_tb.isChecked():
                self._canvas.set_view_mode("after")
            elif before_tb.isChecked():
                self._canvas.set_view_mode("before")
            elif split_tb.isChecked():
                self._canvas.set_view_mode("split")
            else:
                self._canvas.set_view_mode("after")

        after_tb.toggled.connect(lambda c: _on_view_mode_toggled(after_tb, c))
        before_tb.toggled.connect(lambda c: _on_view_mode_toggled(before_tb, c))
        split_tb.toggled.connect(lambda c: _on_view_mode_toggled(split_tb, c))

        self._grid_tb.toggled.connect(self._canvas.set_grid_visible)
        self._wcs_tb.toggled.connect(self._canvas.set_wcs_overlay_visible)
        self._hist_toggle_tb.toggled.connect(self._hist_container.setVisible)
        self._hist_channel_group.buttonClicked.connect(self._on_hist_channel_clicked)

        top_splitter.addWidget(center_widget)

        # Right: Tools panel
        self._tools_panel = ToolsPanel()
        self._connect_tool_signals()
        top_splitter.addWidget(self._tools_panel)

        # Set initial splitter sizes
        top_splitter.setSizes([250, 900, 320])

        # Vertical splitter (top panels | bottom log)
        self._v_splitter = QSplitter(Qt.Orientation.Vertical)
        self._v_splitter.addWidget(top_splitter)

        self._log_panel = LogPanel()
        self._v_splitter.addWidget(self._log_panel)
        self._v_splitter.setSizes([750, 120])

        main_layout.addWidget(self._v_splitter)

    def _connect_tool_signals(self):
        """Wire all tool panel signals to processing handlers."""
        tp = self._tools_panel

        # Canvas overlay signals
        self._canvas.undo_requested.connect(self._undo)
        self._canvas.redo_requested.connect(self._redo)
        self._canvas.export_requested.connect(self._save_image)

        # Tools panel undo/redo buttons
        tp.undo_requested.connect(self._undo)
        tp.redo_requested.connect(self._redo)

        # Existing signals
        tp.run_calibration.connect(self._on_run_calibration)
        tp.run_stacking.connect(self._on_run_stacking)
        tp.run_alignment.connect(self._on_run_alignment)
        tp.run_stretch.connect(self._on_run_stretch)
        tp.run_background.connect(self._on_run_background)
        tp.stretch_params_changed.connect(self._on_stretch_preview)

        # Phase A signals
        tp.run_cosmetic.connect(self._on_run_cosmetic)
        tp.run_banding.connect(self._on_run_banding)
        tp.run_histogram_transform.connect(self._on_run_histogram_transform)
        tp.run_curves.connect(self._on_run_curves)
        tp.run_scnr.connect(self._on_run_scnr)
        tp.run_color_adjust.connect(self._on_run_color_adjust)
        tp.run_deconvolution.connect(self._on_run_deconvolution)

        # Phase B signals
        tp.run_ghs.connect(self._on_run_ghs)
        tp.run_arcsinh_stretch.connect(self._on_run_arcsinh_stretch)
        tp.run_color_calibration.connect(self._on_run_color_calibration)
        tp.run_pcc.connect(self._on_run_pcc)
        tp.run_denoise.connect(self._on_run_denoise)
        tp.run_background_grain.connect(self._on_background_grain)
        tp.request_auto_denoise.connect(self._on_auto_denoise)
        tp.run_frequency_separation.connect(self._on_run_frequency_separation)
        tp.run_statistical_stretch.connect(self._on_run_statistical_stretch)
        tp.run_star_stretch.connect(self._on_run_star_stretch)
        tp.run_star_reduction.connect(self._on_run_star_reduction)
        tp.open_narrowband_dialog.connect(self._show_narrowband_dialog)
        tp.open_pixelmath_dialog.connect(self._show_pixelmath_dialog)
        tp.run_split_channels.connect(self._on_run_split_channels)
        tp.run_extract_luminance.connect(self._on_run_extract_luminance)

        # Phase C signals
        tp.run_wavelet_sharpen.connect(self._on_run_wavelet_sharpen)
        tp.run_local_contrast.connect(self._on_run_local_contrast)
        tp.run_morphology.connect(self._on_run_morphology)
        tp.open_hdr_dialog.connect(self._show_hdr_dialog)

        # Phase D signals
        tp.run_ai_denoise.connect(self._on_run_ai_denoise)
        tp.run_ai_sharpen.connect(self._on_run_ai_sharpen)
        tp.run_starnet.connect(self._on_run_starnet)
        tp.run_superbias.connect(self._on_run_superbias)
        tp.open_processing_graph.connect(self._show_processing_graph)
        tp.open_analysis_fwhm.connect(self._on_analysis_fwhm)
        tp.open_analysis_tilt.connect(self._on_analysis_tilt)
        tp.open_analysis_photometry.connect(self._on_analysis_photometry)
        tp.open_super_resolution.connect(self._on_run_ai_super_resolution)
        tp.open_batch_preprocess.connect(self._show_batch_preprocess_dialog)
        tp.open_batch_dialog.connect(self._show_batch_dialog)
        tp.start_macro_recording.connect(self._on_start_macro)
        tp.stop_macro_recording.connect(self._on_stop_macro)
        tp.play_macro.connect(self._on_play_macro)
        tp.save_macro.connect(self._on_save_macro)
        tp.load_macro.connect(self._on_load_macro)

        # Transform signals
        tp.start_crop_draw.connect(self._on_start_crop_draw)
        tp.run_crop.connect(self._on_run_crop)
        tp.run_rotate.connect(self._on_run_rotate)
        tp.run_flip.connect(self._on_run_flip)
        tp.run_resize.connect(self._on_run_resize)
        tp.run_bin.connect(self._on_run_bin)
        tp.run_invert.connect(self._on_run_invert)

        # New tool signals
        tp.run_unsharp_mask.connect(self._on_run_unsharp_mask)
        tp.run_median_filter.connect(self._on_run_median_filter)
        tp.run_abe.connect(self._on_run_abe)
        tp.run_vignette_correction.connect(self._on_run_vignette)
        tp.run_background_neutralization.connect(self._on_run_bg_neutralization)
        tp.run_chromatic_aberration.connect(self._on_run_ca)
        tp.show_image_statistics.connect(self._on_show_statistics)
        tp.edit_fits_header.connect(self._on_edit_fits_header)
        tp.open_star_mask_dialog.connect(self._on_open_star_mask)
        tp.open_subframe_selector.connect(self._on_open_subframe_selector)
        tp.measure_psf.connect(self._on_measure_psf)
        tp.run_continuum_subtraction.connect(self._on_run_continuum_subtraction)
        tp.toggle_sample_mode.connect(self._on_toggle_sample_mode)
        tp.clear_bg_samples.connect(self._on_clear_bg_samples)
        tp.add_bg_grid.connect(self._on_add_bg_grid)
        tp.toggle_wcs_overlay.connect(self._on_toggle_wcs_overlay)
        tp.toggle_dso_overlay.connect(self._on_toggle_dso_overlay)
        tp.toggle_constellation_overlay.connect(self._on_toggle_constellation_overlay)
        tp.open_python_console.connect(self._on_open_python_console)
        tp.run_mlt.connect(self._on_run_mlt)
        tp.run_lrgb_combine.connect(self._on_run_lrgb_combine)
        tp.run_spcc.connect(self._on_run_spcc)
        tp.open_channel_combine_dialog.connect(self._on_open_channel_combine)
        tp.run_debayer.connect(self._on_run_debayer)

        # Multi-session stacking
        tp.run_multi_session.connect(self._on_run_multi_session)
        tp.multi_session_add_folder.connect(self._on_ms_add_folder)
        tp.multi_session_clear.connect(self._on_ms_clear)

        # Blink comparator
        tp.blink_load_a.connect(lambda: self._blink_load_from_file(slot=0))
        tp.blink_load_b.connect(lambda: self._blink_load_from_file(slot=1))
        tp.blink_use_current_as_a.connect(lambda: self._blink_use_current(slot=0))
        tp.blink_use_current_as_b.connect(lambda: self._blink_use_current(slot=1))
        tp.blink_toggle.connect(self._on_blink_toggle)
        tp.blink_fps_changed.connect(self._on_blink_fps_changed)

        # Canvas sample signals
        self._canvas.sample_placed.connect(self._on_sample_placed)
        self._canvas.sample_removed.connect(self._on_sample_removed)
        self._canvas.crop_rect_selected.connect(self._on_crop_rect_selected)
        self._canvas.crop_mode_changed.connect(self._tools_panel.set_crop_draw_active)

        # Preview signals
        tp.preview_requested.connect(self._on_preview_requested)
        tp.preview_cancelled.connect(self._on_preview_cancelled)
        tp.curves_histogram_changed.connect(self._update_curves_histogram)
        tp.clip_points_changed.connect(self._histogram.set_clip_points)

        # Smart Processor signals
        tp.open_smart_processor.connect(self._show_smart_processor_dialog)
        tp.open_equipment_dialog.connect(self._show_equipment_dialog)
        tp.open_ez_scripts.connect(self._show_ez_script_dialog)
        tp.open_live_stack.connect(self._show_live_stack_dialog)
        tp.open_mosaic_dialog.connect(self._show_mosaic_dialog)

    def _setup_logging(self):
        handler = QtLogHandler(self._log_panel)
        handler.setLevel(logging.INFO)
        logging.getLogger("astraios").addHandler(handler)
        logging.getLogger("astraios").setLevel(logging.INFO)
        self._log_panel.log("Astraios started", "success")

        dm = get_device_manager()
        self._log_panel.log(f"Device: {dm.info.name} ({dm.backend.name})", "info")

    def _setup_statusbar(self):
        sb = self.statusBar()

        def _sb_label(text="", val: bool = False) -> QLabel:
            lbl = QLabel(text)
            color = "#e6edf3" if val else "#8b949e"
            font = "monospace" if val else "sans"
            lbl.setStyleSheet(f"color: {color}; font-size: 11px; font-family: {font}; padding: 0 6px 0 0;")
            return lbl

        self._status_filename_lbl = _sb_label("Image:")
        self._status_filename = _sb_label("", val=True)
        self._status_size_lbl = _sb_label("Size:")
        self._status_size = _sb_label("", val=True)
        self._status_depth_lbl = _sb_label("Depth:")
        self._status_depth = _sb_label("", val=True)
        self._status_channels_lbl = _sb_label("Ch:")
        self._status_channels = _sb_label("", val=True)
        self._status_history_lbl = _sb_label("Steps:")
        self._status_history = _sb_label("", val=True)
        for lbl in (
            self._status_filename_lbl, self._status_filename,
            self._status_size_lbl, self._status_size,
            self._status_depth_lbl, self._status_depth,
            self._status_channels_lbl, self._status_channels,
            self._status_history_lbl, self._status_history,
        ):
            sb.addWidget(lbl)

        self._preview_indicator = QLabel("")
        self._preview_indicator.setStyleSheet(
            "color: #00cc44; font-weight: bold; padding: 0 8px;"
        )
        self._vram_label = QLabel("")
        self._vram_label.setStyleSheet("color: #8b949e; font-size: 11px; padding: 0 8px;")
        self._cuda_badge = QLabel("")
        self._cuda_badge.setStyleSheet(
            "color: #2ea043; font-size: 10px; background: #1a4d2e; border: 1px solid #2ea043;"
            " border-radius: 3px; padding: 1px 5px;"
        )
        sb.addPermanentWidget(self._cuda_badge)
        sb.addPermanentWidget(self._vram_label)
        sb.addPermanentWidget(self._preview_indicator)
        sb.showMessage("Ready")

        self._vram_timer = QTimer(self)
        self._vram_timer.timeout.connect(self._update_vram_label)
        self._vram_timer.start(1000)
        self._update_vram_label()

    def _update_vram_label(self):
        try:
            import psutil as _psutil
            vm = _psutil.virtual_memory()
            ram_gb = (vm.total - vm.available) / 1024**3
            self._ram_chip_label.setText(f"RAM {ram_gb:.1f} GB")
        except Exception:
            log.debug("Could not update RAM label")

        try:
            dm = get_device_manager()
            if dm.device.type == "cuda":
                import torch as _torch
                free, total_bytes = _torch.cuda.mem_get_info(dm.device)
                total = total_bytes / 1024**3
                used = (total_bytes - free) / 1024**3
                vram_text = f"VRAM {used:.1f}/{total:.1f} GB"
                self._vram_label.setText(vram_text)
                _gpu_raw = dm.info.name
                for _pfx in ("NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", "AMD ", "Intel "):
                    if _gpu_raw.startswith(_pfx):
                        _gpu_raw = _gpu_raw[len(_pfx):]
                        break
                _gpu_short = " ".join(_gpu_raw.split()[:2])
                gpu_text = f"● {_gpu_short}"
                self._gpu_chip_label.setText(gpu_text)
                self._cuda_badge.setText("CUDA active")
                self._cuda_badge.show()
                self._log_panel.update_gpu_status(f"{gpu_text} · {vram_text}")
            elif dm.device.type == "mps":
                self._vram_label.setText("GPU: MPS")
                self._gpu_chip_label.setText("● Apple MPS")
                self._cuda_badge.setText("MPS")
                self._cuda_badge.show()
                self._log_panel.update_gpu_status("Apple MPS")
            else:
                self._vram_label.setText("CPU mode")
                self._gpu_chip_label.setText("CPU")
                self._cuda_badge.hide()
                self._log_panel.update_gpu_status("CPU mode")
        except Exception:
            self._vram_label.setText("")

    # ---------- File operations ----------

    def _new_project(self):
        name, ok = QInputDialog.getText(self, "New Project", "Project name:")
        if not ok or not name.strip():
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose Project Location")
        if not directory:
            return
        self._project = Project.create(name.strip(), Path(directory))
        self._project_panel.set_project(self._project)
        self._log_panel.log(f"Created project: {name}", "success")

    def _open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "Astraios Project (astraios_project.json)"
        )
        if not path:
            return
        try:
            self._project = Project.load(Path(path))
        except FileNotFoundError as e:
            QMessageBox.warning(self, "Project Not Found", str(e))
            self._log_panel.log(f"Open failed: {e}", "error")
            return
        except ValueError as e:
            QMessageBox.critical(
                self,
                "Corrupted Project",
                f"Could not parse project file:\n\n{e}\n\n"
                f"The file may be corrupted or from an incompatible version.",
            )
            self._log_panel.log(f"Open failed: {e}", "error")
            return
        except Exception as e:
            QMessageBox.critical(self, "Open Project Error", f"Unexpected error:\n\n{e}")
            self._log_panel.log(f"Open failed: {type(e).__name__}: {e}", "error")
            return
        self._project_panel.set_project(self._project)
        self._log_panel.log(f"Opened project: {self._project.name}", "success")

    def _save_project(self):
        """Helper to save current project safely."""
        if self._project:
            try:
                self._project.save()
                self._log_panel.log("Project saved", "info")
            except Exception as e:
                self._log_panel.log(f"Save failed: {type(e).__name__}: {e}", "error")
                QMessageBox.warning(self, "Save Failed", f"Could not save project:\n\n{e}")

    def _autosave_project(self):
        if self._project:
            try:
                self._project.save()
                log.debug("Project autosaved")
            except Exception as e:
                log.debug("Autosave failed: %s", e)

    def _open_image(self):
        if not self._maybe_discard_changes():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "All Supported (*.fit *.fits *.fts *.xisf *.tif *.tiff *.png);;FITS (*.fit *.fits *.fts);;XISF (*.xisf);;All (*)",
        )
        if path:
            self._load_frame(path)

    def _save_image(self):
        if self._current_image is None:
            return
        from astraios.ui.dialogs.export_dialog import ExportDialog

        src = getattr(self._current_image, "file_path", None)
        dialog = ExportDialog(self, source_name=Path(src).stem if src else "")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        params = dialog.get_export_params()
        try:
            save_image(
                self._current_image,
                path=params["path"],
                bit_depth=params["bit_depth"],
                jpeg_quality=params["jpeg_quality"],
            )
            self._log_panel.log(f"Image exported: {params['path']}", "success")
            self._dirty = False  # exported state is now the saved state
            self._workflow_bar.mark_complete(7)
        except Exception as e:
            log.exception("Export failed")
            self._log_panel.log(f"Export failed: {e}", "error")
            QMessageBox.critical(self, "Export Error", f"Failed to export image:\n{e}")

    def _show_fits_header(self):
        if self._current_image is None:
            self._log_panel.log("Load an image first", "warning")
            return
        from astraios.ui.dialogs.fits_header_dialog import FITSHeaderDialog
        path = getattr(self._current_image, "file_path", None)
        dlg = FITSHeaderDialog(self._current_image.header, file_path=path, parent=self)
        if dlg.exec():
            self._current_image.header.update(dlg.get_header())
            self._log_panel.log("FITS header updated", "success")

    def _show_color_management(self):
        from astraios.ui.dialogs.color_settings_dialog import ColorSettingsDialog

        dialog = ColorSettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._log_panel.log(
                f"Color profile: {dialog.get_working_profile().name}, "
                f"intent: {dialog._intent_combo.currentText()}",
                "info",
            )
            if dialog.is_soft_proof_enabled():
                self._log_panel.log(
                    f"Soft-proof enabled: {dialog.get_soft_proof_profile().name}",
                    "info",
                )
            self._log_panel.log("Color management settings updated", "success")

    def _show_preferences(self):
        from astraios.ui.dialogs.preferences_dialog import PreferencesDialog

        dialog = PreferencesDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            dialog.save()
            self._log_panel.log("Preferences saved", "success")
            self._apply_preferences(dialog.get_prefs())

    def _apply_preferences(self, prefs: dict):
        """Apply preference changes to running application."""
        # GPU device
        if prefs["processing"]["use_gpu"]:
            dev = get_device_manager()
            if dev.is_gpu:
                self._log_panel.log(f"GPU device: {dev.info.name}", "info")
            else:
                self._log_panel.log("GPU requested but not available, using CPU", "warning")

        # Pixel readout format
        fmt = prefs["appearance"]["pixel_readout_format"]
        self._pixel_format = fmt

        # Histogram log scale
        self._histogram.set_log_scale(prefs["appearance"]["histogram_log_scale"])

    def _show_mask_dialog(self):
        if self._current_image is None:
            self._log_panel.log("Load an image first to create masks", "warning")
            return
        from astraios.ui.dialogs.mask_dialog import MaskDialog

        dialog = MaskDialog(self._current_image.data, self)
        dialog.mask_created.connect(self._on_mask_created)
        dialog.exec()
        dialog.deleteLater()  # don't linger as a child holding image arrays

    def _on_mask_created(self, mask: Mask):
        self._masks.append(mask)
        self._active_mask = mask
        self._log_panel.log(
            f"Mask '{mask.name}' is now active — tools will affect only the masked "
            "area. Use Tools > Clear Active Mask to process the whole image again.",
            "success",
        )
        self._update_image_status()

    def _clear_active_mask(self):
        if self._active_mask is None:
            self._log_panel.log("No active mask", "info")
            return
        name = self._active_mask.name
        self._active_mask = None
        self._log_panel.log(f"Cleared active mask '{name}' — tools affect the whole image",
                            "info")
        self._update_image_status()

    def _show_narrowband_dialog(self):
        from astraios.ui.dialogs.narrowband_dialog import NarrowbandDialog

        dialog = NarrowbandDialog(self)
        dialog.result_ready.connect(self._on_narrowband_result)
        dialog.exec()

    def _on_narrowband_result(self, data):
        self._update_current_image(data, "Narrowband combine complete")
        if self._project:
            self._project.add_history("Narrowband Combine", {})

    def _show_blend_dialog(self):
        if self._current_image is None:
            self._log_panel.log("Load an image first to use as the blend base.", "warning")
            return
        from astraios.ui.dialogs.blend_dialog import BlendDialog

        dialog = BlendDialog(base_image=self._current_image.data, parent=self)
        if self._extracted_stars is not None:
            dialog.set_extracted_stars(self._extracted_stars)
        dialog.result_ready.connect(self._on_blend_result)
        dialog.exec()
        dialog.deleteLater()  # don't linger as a child holding image arrays

    def _on_blend_result(self, data):
        self._update_current_image(data, "Image blend complete")
        if self._project:
            self._project.add_history("Image Blend", {})

    def _show_mosaic_dialog(self):
        from astraios.ui.dialogs.mosaic_dialog import MosaicDialog

        dialog = MosaicDialog(self)
        dialog.result_ready.connect(self._on_mosaic_result)
        dialog.exec()
        dialog.deleteLater()  # don't linger as a child holding panel images

    def _on_mosaic_result(self, result):
        self._update_current_image(result.data, "Mosaic stitching complete")
        if self._project:
            self._project.add_history(
                "Mosaic Stitch",
                {"panels": result.n_panels, "output_shape": list(result.output_shape)},
            )

    def _show_pixelmath_dialog(self):
        if self._current_image is None:
            self._log_panel.log("Load an image first", "warning")
            return
        from astraios.ui.dialogs.pixelmath_dialog import PixelMathDialog

        available_images: dict[str, np.ndarray] = {}

        current_data_id = id(self._current_image.data)

        def _add_image(img: ImageData | None, fallback: str):
            if img is None:
                return
            if id(img.data) == current_data_id:
                return  # already available as T/R/G/B/L
            name = img.file_path.stem if img.file_path else fallback
            available_images[name] = img.data

        _add_image(self._master_bias, "master_bias")
        _add_image(self._master_dark, "master_dark")
        _add_image(self._master_flat, "master_flat")
        for i, cal in enumerate(self._calibrated_lights):
            _add_image(cal, f"calibrated_{i}")

        dialog = PixelMathDialog(
            self._current_image.data,
            self,
            available_images=available_images or None,
        )
        dialog.result_ready.connect(self._on_pixelmath_result)
        dialog.exec()
        dialog.deleteLater()  # don't linger as a child holding image arrays

    def _on_pixelmath_result(self, data):
        self._update_current_image(data, "Pixel Math applied")
        if self._project:
            self._project.add_history("Pixel Math", {})

    def _show_about(self):
        dm = get_device_manager()
        QMessageBox.about(
            self,
            "About Astraios",
            f"<h2>Astraios v{astraios.__version__}</h2>"
            f"<p>Professional astrophotography image processing</p>"
            f"<p>GPU: {dm.info.name} ({dm.backend.name})</p>"
            f"<p>&copy; 2024 Astraios Team</p>",
        )

    # ---------- New menu / toolbar handlers ----------

    def _save_project_as(self):
        """Save current project to a new location."""
        if self._project is None:
            QMessageBox.warning(self, "No Project", "No project is currently open.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Choose New Project Location")
        if not directory:
            return
        self._project.path = Path(directory) / self._project.name
        try:
            self._project.save()
        except Exception as e:
            self._log_panel.log(f"Save As failed: {type(e).__name__}: {e}", "error")
            QMessageBox.warning(self, "Save Failed", f"Could not save project:\n\n{e}")
            return
        new_path = self._project.path
        self._project_panel.set_project(self._project)
        self.setWindowTitle(f"{self._project.name} — Astraios [{new_path}]")
        self._log_panel.log(f"Project saved to: {new_path}", "success")

    def _on_import_lights(self):
        """Import light frames into current or new project."""
        if self._project is None:
            name, ok = QInputDialog.getText(self, "New Project", "Project name:")
            if not ok or not name.strip():
                return
            directory = QFileDialog.getExistingDirectory(self, "Choose Project Location")
            if not directory:
                return
            self._project = Project.create(name.strip(), Path(directory))
            self._project_panel.set_project(self._project)
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Light Frames",
            "",
            "Images (*.fit *.fits *.fts *.xisf *.tif *.tiff *.png);;All (*)",
        )
        if paths:
            for p in paths:
                self._project.add_frame(p, FrameType.LIGHT)
            self._project.save()
            self._project_panel.set_project(self._project)
            self._log_panel.log(f"Imported {len(paths)} light frames", "success")

    def _on_import_calibration(self):
        """Import calibration frames (bias/dark/flat) into current project."""
        if self._project is None:
            QMessageBox.warning(self, "No Project", "Open or create a project first.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Calibration Frames",
            "",
            "Images (*.fit *.fits *.fts *.xisf *.tif *.tiff);;All (*)",
        )
        if paths:
            type_counts: dict[str, int] = {}
            for p in paths:
                ft = _guess_frame_type({}, Path(p))
                if ft == FrameType.UNKNOWN:
                    ft = FrameType.FLAT  # safe default for unrecognized calibration frames
                self._project.add_frame(p, ft)
                type_counts[ft.name] = type_counts.get(ft.name, 0) + 1
            self._project.save()
            self._project_panel.set_project(self._project)
            summary = ", ".join(f"{v} {k.lower()}" for k, v in type_counts.items())
            self._log_panel.log(f"Imported {len(paths)} calibration frames: {summary}", "success")

    def _on_export_fits(self):
        if self._current_image is None:
            self._log_panel.log("No image to export", "warning")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export as FITS", "", "FITS (*.fits *.fit)"
        )
        if path:
            save_image(self._current_image, path=path)
            self._log_panel.log(f"Exported FITS: {Path(path).name}", "success")

    def _on_export_tiff(self):
        if self._current_image is None:
            self._log_panel.log("No image to export", "warning")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export as TIFF", "", "TIFF (*.tiff *.tif)"
        )
        if path:
            save_image(self._current_image, path=path)
            self._log_panel.log(f"Exported TIFF: {Path(path).name}", "success")

    def _on_export_png(self):
        if self._current_image is None:
            self._log_panel.log("No image to export", "warning")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export as PNG", "", "PNG (*.png)"
        )
        if path:
            save_image(self._current_image, path=path)
            self._log_panel.log(f"Exported PNG: {Path(path).name}", "success")

    def _on_toggle_histogram(self):
        self._histogram.setVisible(not self._histogram.isVisible())

    def _on_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _on_open_docs(self):
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl("https://github.com/majmichu1/Astraios"))

    def _on_buy_coffee(self):
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl("https://ko-fi.com/astraios"))

    def _on_plate_solve_from_menu(self):
        """Plate-solve the current image (ASTAP → astrometry.net → local)."""
        if self._current_image is None:
            self._log_panel.log("No image loaded to plate-solve", "error")
            return

        from PyQt6.QtCore import QSettings
        settings = QSettings("Astraios", "Astraios")
        raw = settings.value("platesolver/astrometry_api_key", "")
        api_key = raw.strip() or None

        image = self._current_image.data
        params = {"ra_hint": None, "dec_hint": None, "scale_hint": None}

        def _solve_work(img, progress=None):
            from astraios.core.plate_solve import plate_solve_auto, PlateSolveParams
            p = PlateSolveParams()
            result = plate_solve_auto(img, p, api_key=api_key, progress=progress)
            return result

        def _on_solve_done(result):
            if result.success:
                wcs = result.wcs_header or {}
                wcs = {} if wcs is None else wcs
                wcs = normalise_wcs_dict(wcs)
                self._current_wcs = wcs
                ra_h = int(result.ra_center / 15)
                ra_m = int((result.ra_center / 15 - ra_h) * 60)
                ra_s = ((result.ra_center / 15 - ra_h) * 60 - ra_m) * 60
                dec_d = int(result.dec_center)
                dec_m = int(abs(result.dec_center - dec_d) * 60)
                dec_s = (abs(result.dec_center - dec_d) * 60 - dec_m) * 60
                ra_str = f"{ra_h}h {ra_m:02d}m {ra_s:04.1f}s"
                dec_str = f"{dec_d}° {dec_m:02d}′ {dec_s:04.1f}″"
                scale_str = f"{result.pixel_scale:.2f}"
                pa_str = f"{result.rotation:.1f}"
                self._project_panel.set_wcs_info(ra_str, dec_str, scale_str, pa_str)
                self._log_panel.log(
                    f"Plate solve succeeded: RA={result.ra_center:.4f}° "
                    f"Dec={result.dec_center:.4f}° "
                    f"scale={result.pixel_scale:.2f}\"/px",
                    "success",
                )
                if self._project and self._current_image.file_path:
                    frame = self._project.get_frame(str(self._current_image.file_path))
                    if frame:
                        frame.wcs = wcs
                        self._save_project()
                # Fetch catalog stars and build overlays
                self._update_overlays_from_wcs(wcs)
            else:
                self._log_panel.log(
                    "Plate solve failed. Install ASTAP (astap_cli) or set an "
                    "astrometry.net API key in Preferences.", "error"
                )

        self._start_worker(_solve_work, image, on_done=_on_solve_done)

    # ---------- Toolbar / TweaksPanel handlers ----------

    def _on_split_view_toggled(self, checked: bool):
        """Toggle before/after split view on the canvas."""
        if checked and self._current_image is not None and not self._canvas.has_before():
            # No edit has been applied yet, so there's nothing to compare. Capture
            # the current view so the divider isn't empty, and say so.
            self._canvas.capture_before()
            self._log_panel.log(
                "Before/After: apply an edit first to see a difference.", "info"
            )
        self._canvas.set_split_mode(checked)

    def _on_toggle_tweaks(self):
        if self._tweaks_panel.isVisible():
            self._tweaks_panel.hide()
        else:
            self._tweaks_panel.position_near(self)
            self._tweaks_panel.show()
            self._tweaks_panel.raise_()

    def _on_tweaks_accent_changed(self, color_name: str):
        from astraios.ui import theme
        new_qss = theme.set_accent(color_name)
        QApplication.instance().setStyleSheet(new_qss)

    def _on_tweaks_workflow_visible(self, visible: bool):
        self._workflow_bar.setVisible(visible)

    def _on_tweaks_log_visible(self, visible: bool):
        self._log_panel.setVisible(visible)

    def _on_tweaks_log_height(self, height: int):
        sizes = self._v_splitter.sizes()
        if len(sizes) >= 2:
            total = sum(sizes)
            self._v_splitter.setSizes([total - height, height])

    # ---------- Workflow bar ----------

    def _on_workflow_step_clicked(self, idx: int):
        """Switch to the tools panel tab corresponding to the workflow step."""
        if idx == -1:
            self._save_image()
            return
        tp = self._tools_panel
        if hasattr(tp, "_tabs"):
            tp._tabs.setCurrentIndex(min(idx, tp._tabs.count() - 1))

    def _advance_workflow(self, completed_step: int):
        """Mark a step done and activate the next."""
        self._workflow_bar.mark_complete(completed_step)
        next_step = completed_step + 1
        if next_step <= 6:
            self._workflow_bar.set_current(next_step)

    # ---------- Unsaved-work protection ----------

    def _maybe_discard_changes(self) -> bool:
        """Ask before discarding unsaved edits. Returns True if it's OK to proceed
        (exported, discarded, or nothing to lose), False to abort the action."""
        if not self._dirty or self._current_image is None:
            return True
        from PyQt6.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setWindowTitle("Unsaved changes")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText("This image has edits that haven't been exported yet.")
        box.setInformativeText("Export them before continuing?")
        box.setStandardButtons(
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Save)
        choice = box.exec()
        if choice == QMessageBox.StandardButton.Save:
            self._save_image()
            return not self._dirty  # only proceed if the export actually completed
        if choice == QMessageBox.StandardButton.Discard:
            return True
        return False  # Cancel

    def closeEvent(self, event):
        if self._maybe_discard_changes():
            event.accept()
        else:
            event.ignore()

    # ---------- Onboarding ----------

    def _maybe_first_run_welcome(self):
        """Show the welcome guide once, on the first ever launch."""
        from PyQt6.QtCore import QSettings
        s = QSettings("Astraios", "Astraios")
        if not s.value("ui/welcome_shown", False, type=bool):
            self._show_welcome(first_run=True)
            s.setValue("ui/welcome_shown", True)

    def _show_welcome(self, first_run: bool = False):
        """A short 'how to start' guide (first launch + Help > Getting Started)."""
        from PyQt6.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setWindowTitle("Welcome to Astraios" if first_run else "Getting Started")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText("<b>Three steps to your first processed image</b>")
        box.setInformativeText(
            "<ol>"
            "<li><b>Open Image</b> — the green button at the top left, or drag a "
            "FITS/TIFF/XISF onto the window. Use a <i>linear</i> stacked file (not "
            "an already-stretched picture) for the best result.</li>"
            "<li><b>Process</b> — work left-to-right along the pipeline bar at the "
            "top (Background → Stretch → Colour → Detail); each step opens the "
            "matching tools on the right. Or click <b>Smart Processor</b> to do it "
            "all automatically.</li>"
            "<li><b>Export</b> — the last step on the pipeline bar saves your "
            "result.</li>"
            "</ol>"
            "You can reopen this any time from <b>Help &gt; Getting Started</b>."
        )
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    # ---------- Image display ----------

    @pyqtSlot(str)
    def _load_frame(self, path: str):
        if not Path(path).exists():
            self._log_panel.log(
                f"File not found (was it moved or deleted?): {Path(path).name}", "error"
            )
            return
        self._log_panel.log(f"Loading {Path(path).name}...", "info")

        def _load_work(path, progress=None):
            return load_image(path)

        def _on_loaded(image):
            # Reset undo history for the new image and size its depth to the
            # image (large images keep fewer steps so RAM stays bounded).
            self._undo_stack.configure_for_image(image.data.nbytes)
            self._update_undo_actions()
            self._display_image(image)
            self._log_panel.log(f"Loaded: {Path(path).name} ({image.shape_str})", "info")
            # A freshly loaded image is clean; reset edit/workflow/mask state.
            self._dirty = False
            self._active_mask = None
            self._workflow_bar.set_current(0)
            # The processing graph is created lazily (on the first edit, or when
            # the graph dialog is opened) so we don't hold an always-on full-res
            # copy of the base image (~780MB at 65MP) for sessions that never
            # touch it.
            self._processing_graph = None

        # Release the previous image's heavy auxiliary buffers before loading the
        # new frame so they don't coexist with it — at 65MP each is ~780MB. The
        # working image itself stays until the new one is ready, so a failed load
        # doesn't lose it.
        self._extracted_stars = None
        self._blink_images = [None, None]
        self._processing_graph = None
        import gc
        gc.collect()

        # Loading reports no fraction, so show a busy bar instead of a stuck 0%.
        self._log_panel.set_busy(True, f"Loading {Path(path).name}...")
        self._start_worker(_load_work, path, on_done=_on_loaded)

    def _display_image(self, image: ImageData, display_ref: "np.ndarray | None" = None):
        self._current_image = image
        self._image_ref[0] = image  # sync with undo ref

        # Downsample to at most 1024px before any stretch computation.
        # The canvas scales to fit the viewport anyway; applying MTF on 12M
        # pixels is 10-50× slower than on a 1024px thumbnail with no
        # visible quality difference.
        # Also cache the small image so live preview can reuse it without re-downscaling.
        small, _scale = self._downscale_for_preview(image.data)
        self._preview_small_cache = (small, _scale)

        import numpy as _np
        if display_ref is not None:
            small_ref, _ = self._downscale_for_preview(display_ref)
            if small.ndim == 2:
                hwc = _np.stack([small] * 3, axis=-1)
                ref_hwc = _np.stack([small_ref] * 3, axis=-1) if small_ref.ndim == 2 else _np.transpose(small_ref, (1, 2, 0))
            else:
                hwc = _np.transpose(small, (1, 2, 0))
                ref_hwc = _np.transpose(small_ref, (1, 2, 0)) if small_ref.ndim == 3 else _np.stack([small_ref] * 3, axis=-1)
            stretched = auto_stretch_for_display_ref(hwc, ref_hwc)
            rgb = _np.clip(stretched * 255, 0, 255).astype(_np.uint8)
            # Cache the stretch reference so subsequent tool previews match the display brightness
            self._preview_stretch_ref_cache = small_ref
        else:
            small_img = ImageData(data=small, header={})
            rgb = small_img.to_display(stretch=True)
            # No external reference — use the image itself for preview stretching
            self._preview_stretch_ref_cache = None

        self._canvas.set_image(rgb, image.data, display_scale=_scale)  # full-res data + scale for coord mapping
        fname = image.file_path.name if image.file_path else "Untitled"
        self._canvas.set_image_info(fname, image.shape_str)

        hist_data = compute_histogram(image.data)
        self._histogram.set_histogram_data(hist_data)
        self._update_hist_stats()
        self._update_curves_histogram(hist_data)
        self._sync_console_image()

    def _update_curves_histogram(self, hist_data: dict | None = None):
        """Push histogram data into the curve editor if the show-histogram checkbox is on."""
        tp = self._tools_panel
        if not tp.curves_histogram_visible:
            tp.curve_editor.set_histogram(None)
            return
        if hist_data is None:
            if self._current_image is None:
                return
            hist_data = compute_histogram(self._current_image.data)
        # Pick the channel matching the current curve selector
        channel_map = {0: "luminance", 1: "red", 2: "green", 3: "blue"}
        key = channel_map.get(tp.current_curve_channel, "luminance")
        counts = hist_data.get(key) or hist_data.get("gray")
        if counts is not None:
            import numpy as np
            tp.curve_editor.set_histogram(np.asarray(counts, dtype=np.float32))
        else:
            tp.curve_editor.set_histogram(None)

    def _push_undo(self, before: ImageData, after: ImageData, description: str):
        """Push an undo command and update the Undo/Redo action states."""
        self._undo_stack.push(
            before, after, description,
            before_display_ref=self._preview_stretch_ref_cache,
        )
        self._update_undo_actions()

    def _update_undo_actions(self):
        """Update the enabled state and text of undo/redo actions."""
        can_undo = self._undo_stack.can_undo()
        can_redo = self._undo_stack.can_redo()
        self._undo_act.setEnabled(can_undo)
        self._redo_act.setEnabled(can_redo)
        if hasattr(self, "_tb_undo_btn"):
            self._tb_undo_btn.setEnabled(can_undo)
        if hasattr(self, "_tb_redo_btn"):
            self._tb_redo_btn.setEnabled(can_redo)
        if hasattr(self, "_canvas"):
            self._canvas._overlay_undo_btn.setVisible(can_undo)
            self._canvas._overlay_redo_btn.setVisible(can_redo)
        undo_text = self._undo_stack.undo_text()
        redo_text = self._undo_stack.redo_text()
        self._undo_act.setText(f"&Undo ({undo_text})" if undo_text else "&Undo")
        self._redo_act.setText(f"Re&do ({redo_text})" if redo_text else "Re&do")

    def _undo(self):
        if self._undo_stack.undo():
            # Restore the exact stretch reference used when this state was first displayed
            ref = self._undo_stack.current_display_ref()
            self._display_image(self._image_ref[0], display_ref=ref)
            self._log_panel.log(f"Undid: {self._undo_stack.redo_text()}", "info")
            self._update_undo_actions()

    def _redo(self):
        if self._undo_stack.redo():
            ref = self._undo_stack.current_display_ref()
            self._display_image(self._image_ref[0], display_ref=ref)
            self._log_panel.log(f"Redid: {self._undo_stack.undo_text()}", "info")
            self._update_undo_actions()

    def _clear_undo_history(self):
        self._undo_stack.clear()
        self._update_undo_actions()
        self._log_panel.log("Undo history cleared", "info")

    def _update_current_image(
        self,
        data,
        message: str,
        undo_desc: str | None = None,
        geometric: bool = False,
        tool: str = "",
        tool_params: dict | None = None,
    ):
        """Replace current image data and update display, recording undo.

        geometric=True for ops that only change image extent (crop/rotate/flip/resize):
        those should re-stretch from the result's own statistics, not the pre-op reference.

        ``tool`` is the canonical registry name of the operation and ``tool_params``
        its parameters; when given, the step recorded in the non-destructive
        history is replayable (it can be re-evaluated, toggled, reordered and
        re-edited). When omitted, a display-only step is recorded so the history
        still shows the operation.
        """
        # If a live preview is running, keep it alive and re-trigger after update
        pending_preview = self._pending_preview_tool

        self._canvas.capture_before()   # save current render for Before/After compare
        before = self._current_image

        # Apply the active user mask centrally: only the masked region keeps the
        # new result; outside it reverts to the pre-op image. Doing it here means
        # every tool respects the mask without each handler needing to. Skipped
        # for geometric ops (the shape changes), and guarded on a shape match.
        if (self._active_mask is not None and not geometric and before is not None
                and before.data.shape == data.shape
                and self._active_mask.data.shape == data.shape[-2:]):
            from astraios.core.masks import apply_mask
            data = apply_mask(before.data, data, self._active_mask)
        # If the image size changed (a geometric op), the mask no longer fits.
        if (self._active_mask is not None
                and self._active_mask.data.shape != data.shape[-2:]):
            self._log_panel.log(
                f"Active mask '{self._active_mask.name}' cleared (image size changed)",
                "info",
            )
            self._active_mask = None
        # For pixel-value operations, anchor the display stretch to the pre-op image so
        # the executed result matches the live-preview brightness exactly.
        # For geometric operations the pixel distribution of the result may differ
        # (different crop region, flipped vignette, etc.) so re-stretch independently.
        display_ref = None if geometric or before is None else before.data
        image = ImageData(
            data=data,
            header=self._current_image.header.copy() if self._current_image else {},
            frame_type=self._current_image.frame_type if self._current_image else FrameType.RESULT,
        )
        if before is not None:
            desc = undo_desc if undo_desc else message
            self._push_undo(before, image, desc)
        self._display_image(image, display_ref=display_ref)
        # Store the after-display ref in the undo command so redo can match brightness
        self._undo_stack.set_last_after_display_ref(self._preview_stretch_ref_cache)
        self._log_panel.log(message, "success")
        self._update_image_status()

        # Record the operation in the non-destructive history, unless this update
        # is the history itself re-evaluating (which would recurse).
        if not self._skip_graph_auto_add and before is not None:
            if self._processing_graph is None:
                from astraios.core.processing_graph import ProcessingGraph
                self._processing_graph = ProcessingGraph()
                self._processing_graph.set_base(before.data)
            mask_name = self._active_mask.name if self._active_mask is not None else None
            self._processing_graph.record(
                tool_name=tool,
                params=dict(tool_params) if tool_params else {},
                display_name=undo_desc if undo_desc else message,
                mask_name=mask_name,
            )

        # The image now differs from the last saved/exported state.
        self._dirty = True
        # Reflect real progress on the workflow bar: the Tools Panel tab a tool
        # was run from is that operation's workflow step (the map is identity).
        tp = self._tools_panel
        if hasattr(tp, "_tabs"):
            step = tp._tabs.currentIndex()
            if 0 <= step <= 6:
                self._workflow_bar.mark_complete(step)

        if pending_preview is not None:
            # Re-run preview on the newly applied image (updates right side of split)
            self._preview_timer.start()
        else:
            self._on_preview_cancelled()

    def _update_image_status(self):
        """Refresh the status bar image info labels."""
        img = self._current_image
        if img is None:
            for lbl in (self._status_filename, self._status_size, self._status_depth,
                        self._status_channels, self._status_history):
                lbl.setText("")
            return
        fp = getattr(img, "file_path", None)
        name = Path(fp).name if fp else "unsaved"
        self._status_filename.setText(name)
        h, w = (img.data.shape[-2], img.data.shape[-1]) if img.data.ndim >= 2 else (0, 0)
        self._status_size.setText(f"{w} × {h}")
        depth = "32-bit float" if img.data.dtype == "float32" else str(img.data.dtype)
        self._status_depth.setText(depth)
        ch = "RGB" if img.data.ndim == 3 and img.data.shape[0] == 3 else "Mono"
        self._status_channels.setText(ch)
        idx = self._undo_stack.count
        mask_note = f" · mask: {self._active_mask.name}" if self._active_mask else ""
        self._status_history.setText(f"{idx} steps{mask_note}")

    @pyqtSlot(int, int, list)
    def _update_pixel_readout(self, x: int, y: int, values: list):
        if len(values) == 1:
            self.statusBar().showMessage(f"x={x} y={y}  |  L={values[0]:.5f}")
        elif len(values) >= 3:
            self.statusBar().showMessage(
                f"x={x} y={y}  |  R={values[0]:.5f}  G={values[1]:.5f}  B={values[2]:.5f}"
            )

    def _update_tb_progress(self, fraction: float, message: str):
        self._tb_progress_bar.setValue(int(fraction * 1000))
        self._tb_status_label.setText(message[:30] if message else "Working…")
        self._tb_status_label.setStyleSheet(
            "color: #d29922; font-size: 10px; font-family: monospace; padding: 0 4px;"
        )

    def _reset_tb_progress(self):
        self._tb_progress_bar.setValue(0)
        self._tb_status_label.setText("Ready")
        self._tb_status_label.setStyleSheet(
            "color: #2ea043; font-size: 10px; font-family: monospace; padding: 0 4px;"
        )

    def _update_canvas_coord_label(self, x: int, y: int, values: list):
        if len(values) == 1:
            self._canvas_coord_label.setText(f"x={x} y={y}  L={values[0]:.4f}")
        elif len(values) >= 3:
            self._canvas_coord_label.setText(
                f"x={x} y={y}  R={values[0]:.3f} G={values[1]:.3f} B={values[2]:.3f}"
            )

    def _on_hist_channel_clicked(self, btn):
        ch = btn.text()
        self._histogram.set_active_channel(ch)
        self._update_hist_stats()

    def _update_hist_stats(self):
        stats = self._histogram._get_stats()
        if stats is None:
            self._hist_stats_label.setText("")
        else:
            mean, median, sd, clip = stats
            self._hist_stats_label.setText(
                f"Mean {mean:.3f}  Med {median:.3f}  SD {sd:.3f}  Clip {clip:.1f}%"
            )

    # ---------- Drag and drop ----------

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
        supported = [
            p
            for p in paths
            if p.suffix.lower() in (".fit", ".fits", ".fts", ".xisf", ".tif", ".tiff", ".png")
        ]
        if not supported:
            if paths:
                self._log_panel.log(
                    f"Can't open {paths[0].name} — supported types: FITS, XISF, "
                    "TIFF, PNG", "warning",
                )
            return

        if self._project:
            # Add to project as lights by default
            self._on_frames_imported(supported, FrameType.LIGHT)
        else:
            # Just display the first one (would replace any unsaved edits)
            if not self._maybe_discard_changes():
                return
            self._load_frame(str(supported[0]))

    def _on_frames_imported(self, paths: list[Path], frame_type: FrameType):
        if self._project is None:
            self._log_panel.log("Create a project first (File > New Project)", "warning")
            return
        added = self._project.add_frames(paths, frame_type)
        self._project_panel.refresh()
        self._log_panel.log(f"Imported {added} {frame_type.name.lower()} frame(s)", "success")
        self._save_project()

    # ---------- Worker management ----------

    def _set_processing_locked(self, locked: bool) -> None:
        """Disable destructive UI while a background worker runs."""
        self._new_proj_act.setEnabled(not locked)
        self._open_proj_act.setEnabled(not locked)
        self._open_img_act.setEnabled(not locked)
        if locked:
            self._undo_act.setEnabled(False)
            self._redo_act.setEnabled(False)
            if hasattr(self, "_tb_undo_btn"):
                self._tb_undo_btn.setEnabled(False)
            if hasattr(self, "_tb_redo_btn"):
                self._tb_redo_btn.setEnabled(False)
            if hasattr(self, "_canvas"):
                self._canvas._overlay_undo_btn.setVisible(False)
                self._canvas._overlay_redo_btn.setVisible(False)
        else:
            self._update_undo_actions()

    def _on_worker_error(self, msg: str):
        """Surface a tool failure: log it AND show a modal so it isn't missed."""
        self._log_panel.log(f"Error: {msg}", "error")
        low = msg.lower()
        if "cancel" in low:
            return  # cancellation is not a failure
        if any(k in low for k in ("out of memory", "outofmemory", "cuda error",
                                  "cublas", "cudnn")):
            QMessageBox.warning(
                self,
                "Out of memory",
                "Your GPU ran out of memory processing this image.\n\n"
                "Try one of these:\n"
                "  • Enable 'Tiled inference' in the tool (smaller chunks)\n"
                "  • Work on a crop or a smaller image\n"
                "  • Close other GPU-heavy apps\n\n"
                "Some operations fall back to the CPU automatically.",
            )
        else:
            QMessageBox.critical(
                self, "Processing failed",
                f"The operation could not complete:\n\n{msg}",
            )

    def _start_worker(self, func, *args, on_done=None, **kwargs):
        if self._worker is not None:
            if self._worker.isRunning():
                self._log_panel.log("Cancelling previous task...", "warning")
                self._worker.cancel()
                if not self._worker.wait(5000):
                    self._worker.terminate()
                    self._worker.wait(1000)
                    self._log_panel.log("Previous task terminated", "warning")
            self._worker = None

        safe_args = tuple(
            a.copy() if isinstance(a, np.ndarray) else a for a in args
        )

        from PyQt6.QtCore import Qt as _Qt
        self._worker = ProcessingWorker(func, *safe_args, **kwargs)
        self._set_processing_locked(True)
        # QueuedConnection ensures slots run in the GUI thread even when signals
        # are emitted from the worker thread (prevents QTextEdit segfaults).
        self._worker.progress.connect(
            self._log_panel.update_progress, _Qt.ConnectionType.QueuedConnection
        )
        self._worker.progress.connect(
            self._update_tb_progress, _Qt.ConnectionType.QueuedConnection
        )
        self._worker.error.connect(
            self._on_worker_error, _Qt.ConnectionType.QueuedConnection,
        )
        self._worker.error.connect(
            lambda: self._log_panel.set_cancel_visible(False), _Qt.ConnectionType.QueuedConnection
        )
        self._worker.elapsed.connect(
            lambda secs: self._log_panel.log(
                f"Completed in {secs:.1f}s" if secs < 120
                else f"Completed in {secs / 60:.1f} min",
                "info",
            ),
            _Qt.ConnectionType.QueuedConnection,
        )
        self._worker.cancelled.connect(
            lambda: self._log_panel.log("Operation cancelled", "warning"),
            _Qt.ConnectionType.QueuedConnection,
        )
        self._worker.cancelled.connect(
            lambda: self._log_panel.reset_progress(), _Qt.ConnectionType.QueuedConnection
        )
        self._worker.cancelled.connect(
            lambda: self._log_panel.set_cancel_visible(False), _Qt.ConnectionType.QueuedConnection
        )
        if on_done:
            self._worker.finished.connect(on_done, _Qt.ConnectionType.QueuedConnection)
        self._worker.finished.connect(
            lambda: self._log_panel.reset_progress(), _Qt.ConnectionType.QueuedConnection
        )
        self._worker.finished.connect(
            lambda: self._log_panel.set_cancel_visible(False), _Qt.ConnectionType.QueuedConnection
        )
        self._worker.finished.connect(
            lambda _=None: self._reset_tb_progress(), _Qt.ConnectionType.QueuedConnection
        )
        self._worker.finished.connect(
            lambda _=None: self._set_processing_locked(False),
            _Qt.ConnectionType.QueuedConnection,
        )
        self._worker.finished.connect(
            lambda _=None: setattr(self, "_worker", None), _Qt.ConnectionType.QueuedConnection
        )
        self._worker.cancelled.connect(
            lambda: self._reset_tb_progress(), _Qt.ConnectionType.QueuedConnection
        )
        self._worker.cancelled.connect(
            lambda: self._set_processing_locked(False), _Qt.ConnectionType.QueuedConnection
        )
        self._worker.cancelled.connect(
            lambda: setattr(self, "_worker", None), _Qt.ConnectionType.QueuedConnection
        )
        self._worker.error.connect(
            lambda _=None: self._reset_tb_progress(), _Qt.ConnectionType.QueuedConnection
        )
        self._worker.error.connect(
            lambda _=None: self._set_processing_locked(False),
            _Qt.ConnectionType.QueuedConnection,
        )
        self._worker.error.connect(
            lambda _=None: setattr(self, "_worker", None), _Qt.ConnectionType.QueuedConnection
        )
        self._log_panel.set_cancel_visible(True)
        self._log_panel.cancel_requested.connect(self._worker.cancel, _Qt.ConnectionType.UniqueConnection)
        self._worker.start()

    # ---------- Processing operations ----------

    @pyqtSlot()
    def _on_run_calibration(self):
        if self._project is None:
            self._log_panel.log("No project loaded", "warning")
            return

        light_frames = self._project.frames_by_type(FrameType.LIGHT)
        if not light_frames:
            self._log_panel.log("No light frames to calibrate", "warning")
            return

        # Get calibration sources from panel (raw folders OR pre-made masters)
        cal_sources = self._tools_panel.get_calibration_sources()

        # Fall back to project-imported frames if panel has nothing configured
        bias_paths  = (cal_sources["bias_paths"]  or
                       [e.path for e in self._project.frames_by_type(FrameType.BIAS)])
        dark_paths  = (cal_sources["dark_paths"]  or
                       [e.path for e in self._project.frames_by_type(FrameType.DARK)])
        flat_paths  = (cal_sources["flat_paths"]  or
                       [e.path for e in self._project.frames_by_type(FrameType.FLAT)])

        bias_master_path  = cal_sources["bias_master"]
        dark_master_path  = cal_sources["dark_master"]
        flat_master_path  = cal_sources["flat_master"]

        n_bias = len(bias_paths) or (1 if bias_master_path else 0)
        n_dark = len(dark_paths) or (1 if dark_master_path else 0)
        n_flat = len(flat_paths) or (1 if flat_master_path else 0)
        self._log_panel.log(
            f"Starting calibration — {len(light_frames)} lights, "
            f"{n_bias} bias, {n_dark} dark, {n_flat} flat", "info"
        )
        self._start_worker(
            self._calibration_pipeline,
            bias_paths, dark_paths, flat_paths,
            [e.path for e in light_frames],
            bias_master_path, dark_master_path, flat_master_path,
            on_done=self._on_calibration_done,
        )

    @staticmethod
    def _calibration_pipeline(
        bias_paths, dark_paths, flat_paths, light_paths,
        bias_master_path=None, dark_master_path=None, flat_master_path=None,
        progress=None,
    ):
        results = {}
        prog = progress or (lambda f, m: None)

        # ── Master bias ───────────────────────────────────────────────────────
        master_bias = None
        if bias_master_path:
            from astraios.core.image_io import load_image
            master_bias = load_image(bias_master_path)
            prog(0.05, f"Loaded master bias: {Path(bias_master_path).name}")
            results["master_bias"] = master_bias
        elif bias_paths:
            prog(0.0, f"Creating master bias from {len(bias_paths)} frames…")
            r = create_master_bias(bias_paths, progress=lambda f, m: prog(f * 0.15, m))
            master_bias = r.master
            results["master_bias"] = master_bias

        # ── Master dark ───────────────────────────────────────────────────────
        master_dark = None
        if dark_master_path:
            from astraios.core.image_io import load_image
            master_dark = load_image(dark_master_path)
            prog(0.15, f"Loaded master dark: {Path(dark_master_path).name}")
            results["master_dark"] = master_dark
        elif dark_paths:
            prog(0.15, f"Creating master dark from {len(dark_paths)} frames…")
            r = create_master_dark(
                dark_paths, master_bias=master_bias,
                progress=lambda f, m: prog(0.15 + f * 0.2, m),
            )
            master_dark = r.master
            results["master_dark"] = master_dark

        # ── Master flat ───────────────────────────────────────────────────────
        master_flat = None
        if flat_master_path:
            from astraios.core.image_io import load_image
            master_flat = load_image(flat_master_path)
            prog(0.35, f"Loaded master flat: {Path(flat_master_path).name}")
            results["master_flat"] = master_flat
        elif flat_paths:
            prog(0.35, f"Creating master flat from {len(flat_paths)} frames…")
            r = create_master_flat(
                flat_paths, master_bias=master_bias, master_dark=master_dark,
                progress=lambda f, m: prog(0.35 + f * 0.2, m),
            )
            master_flat = r.master
            results["master_flat"] = master_flat

        # ── Calibrate lights ──────────────────────────────────────────────────
        prog(0.55, f"Calibrating {len(light_paths)} light frames…")
        calibrated = calibrate_lights_batch(
            light_paths,
            master_bias=master_bias,
            master_dark=master_dark,
            master_flat=master_flat,
            progress=lambda f, m: prog(0.55 + f * 0.45, m),
        )
        results["calibrated"] = calibrated
        return results

    @pyqtSlot(object)
    def _on_calibration_done(self, results: dict):
        self._master_bias = results.get("master_bias")
        self._master_dark = results.get("master_dark")
        self._master_flat = results.get("master_flat")
        self._calibrated_lights = results.get("calibrated", [])

        n = len(self._calibrated_lights)
        self._log_panel.log(f"Calibration complete: {n} lights calibrated", "success")

        if self._calibrated_lights:
            self._display_image(self._calibrated_lights[0])

        if self._project:
            self._project.add_history("Calibration", {"n_lights": n})
            self._save_project()

    def _get_raw_light_paths(self) -> list[Path] | None:
        """Return raw light frame paths after user confirmation.

        Shows a warning dialog about uncalibrated frames on first use.
        Returns None if cancelled or no lights in project.
        """
        if self._calibrated_lights:
            return [img.file_path for img in self._calibrated_lights
                    if img.file_path is not None]

        if not self._project:
            self._log_panel.log("No project loaded", "error")
            return None

        light_frames = [f for f in self._project.frames if f.frame_type == FrameType.LIGHT]
        if not light_frames:
            self._log_panel.log("No light frames in project", "error")
            return None

        settings = QSettings("Astraios", "Astraios")
        if not settings.value("stacking/raw_warning_acknowledged", False, type=bool):
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Uncalibrated Frames")
            msg.setText("No calibrated frames found. Do you want to align/stack raw frames?")
            msg.setInformativeText(
                "Stacking raw frames without calibration (bias/dark/flat correction) "
                "may result in artifacts. It is recommended to calibrate frames first."
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg.setDefaultButton(QMessageBox.StandardButton.No)

            cb = QCheckBox("Do not show this warning again")
            msg.setCheckBox(cb)

            if msg.exec() == QMessageBox.StandardButton.No:
                return None

            if cb.isChecked():
                settings.setValue("stacking/raw_warning_acknowledged", True)

        return [f.path for f in light_frames]

    def _get_light_paths(self) -> list | None:
        """Return raw light frame paths from project without loading them into RAM.

        Returns None if the user cancels or there are no lights.
        """
        if not self._project:
            self._log_panel.log("No project loaded", "error")
            return None
        light_frames = [f for f in self._project.frames if f.frame_type == FrameType.LIGHT]
        if not light_frames:
            self._log_panel.log("No light frames in project", "error")
            return None

        settings = QSettings("Astraios", "Astraios")
        if not settings.value("stacking/raw_warning_acknowledged", False, type=bool):
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Uncalibrated Frames")
            msg.setText("No calibrated frames found. Do you want to align/stack raw frames?")
            msg.setInformativeText(
                "Stacking raw frames without calibration (bias/dark/flat correction) "
                "may result in artifacts. It is recommended to calibrate frames first."
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg.setDefaultButton(QMessageBox.StandardButton.No)
            cb = QCheckBox("Do not show this warning again")
            msg.setCheckBox(cb)
            if msg.exec() == QMessageBox.StandardButton.No:
                return None
            if cb.isChecked():
                settings.setValue("stacking/raw_warning_acknowledged", True)

        return [f.path for f in light_frames]

    @pyqtSlot()
    def _on_run_alignment(self):
        """Start alignment.

        If calibrated lights are in memory, use in-memory align_frames (fast, small N).
        Otherwise stream from disk via align_from_paths (safe for 100+ frames).
        """
        params_dict = None

        if self._calibrated_lights:
            # Calibrated lights already in RAM — use in-memory path
            lights = self._calibrated_lights
            self._tools_panel.set_ref_frame_max(len(lights))
            params_dict = self._tools_panel.get_alignment_params()
            stk_params = StackingParams(
                registration_mode=params_dict["mode"],
                reference_frame_index=params_dict["reference_frame_index"],
                comet_nucleus_radius=params_dict.get("comet_nucleus_radius", 15),
            )
            self._log_panel.log(
                f"Starting alignment ({params_dict['mode'].name}) — "
                f"{len(lights)} calibrated frames (in-memory)...",
                "info",
            )
            self._start_worker(
                align_frames,
                lights,
                params=stk_params,
                on_done=self._on_alignment_done,
            )
        else:
            # Raw project frames — stream from disk to avoid OOM
            if self._subframe_selected_paths:
                paths = [Path(p) for p in self._subframe_selected_paths]
                self._log_panel.log(
                    f"Using {len(paths)} subframe-selected frames for alignment", "info"
                )
            else:
                paths = self._get_light_paths()
            if not paths:
                return
            params_dict = self._tools_panel.get_alignment_params()
            stk_params = StackingParams(
                registration_mode=params_dict["mode"],
                reference_frame_index=params_dict["reference_frame_index"],
                comet_nucleus_radius=params_dict.get("comet_nucleus_radius", 15),
            )

            if not self._project:
                # Offer to create a project so the output directory can be determined
                from PyQt6.QtWidgets import QInputDialog
                name, ok = QInputDialog.getText(
                    self, "Create Project",
                    "No project is loaded. Enter a project name to create one now:",
                )
                if not ok or not name.strip():
                    return
                directory = QFileDialog.getExistingDirectory(self, "Choose Project Location")
                if not directory:
                    return
                self._project = Project.create(name.strip(), Path(directory))
                self._project_panel.set_project(self._project)
                self._log_panel.log(f"Created project: {name}", "success")
            aligned_dir = os.path.join(self._project.directory, "aligned")
            import re as _re
            raw_name = getattr(self._project, "name", None) or "frame"
            file_prefix = _re.sub(r"[^\w\-]", "_", raw_name) + "_reg"

            self._log_panel.log(
                f"Starting alignment ({params_dict['mode'].name}) — "
                f"{len(paths)} raw frames (streaming from disk, low-RAM)...",
                "info",
            )
            self._start_worker(
                _align_with_optional_filter,
                paths,
                aligned_dir,
                stk_params=stk_params,
                filename_prefix=file_prefix,
                on_done=self._on_path_alignment_done,
            )

    def _on_path_alignment_done(self, result_paths: list):
        """Callback when path-based alignment finishes.

        result_paths is already a list[Path] written to disk by align_from_paths.
        """
        if not result_paths:
            self._log_panel.log("Alignment failed: no frames produced", "error")
            return

        from pathlib import Path
        result_paths = [Path(p) for p in result_paths if p is not None]

        self._log_panel.log(
            f"Alignment complete: {len(result_paths)} frames saved to 'aligned/' folder",
            "info",
        )

        # Register aligned frames in project
        if self._project:
            for p in result_paths:
                if p.exists():
                    self._project.remove_frame(p)
            self._project.add_frames([p for p in result_paths if p.exists()], FrameType.ALIGNED)
            self._aligned_paths = result_paths

            if hasattr(self, "_project_panel"):
                self._project_panel.refresh()
        else:
            self._aligned_paths = result_paths

        # Clear subframe selection — stacking should now use the freshly aligned _aligned_paths
        self._subframe_selected_paths = []
        self._tools_panel.set_subframe_count(0, 0)

        # Display first aligned frame
        try:
            first = load_image(str(result_paths[0]))
            self._display_image(first)
        except Exception as exc:
            log.warning("Could not display first aligned frame: %s", exc)

        self._log_panel.log(
            f"Alignment complete: {len(result_paths)} frames aligned. Ready to stack.",
            "success",
        )
        if self._project:
            self._project.add_history("Alignment", {"n_frames": len(result_paths)})
            self._save_project()
        try:
            import gc as _gc; _gc.collect()
            get_device_manager().empty_cache()
        except Exception:
            log.debug("GPU cache flush failed")

    def _on_alignment_done(self, aligned_lights: list):
        """Callback when alignment finishes."""
        if not aligned_lights:
            self._log_panel.log("Alignment failed: no frames aligned", "error")
            return

        # Save aligned frames to project folder FIRST, then free from RAM
        project_dir = self._project.directory if self._project else None
        aligned_paths: list[Path] = []
        if project_dir:
            aligned_dir = os.path.join(project_dir, "aligned")
            os.makedirs(aligned_dir, exist_ok=True)

            for i, img in enumerate(aligned_lights):
                filename = f"aligned_{i + 1:03d}.fits"
                filepath = Path(os.path.join(aligned_dir, filename))
                try:
                    from astraios.core.image_io import save_image
                    save_image(img, str(filepath))
                    aligned_paths.append(filepath)
                except Exception as e:
                    log.warning(f"Failed to save aligned frame {filename}: {e}")

            self._log_panel.log(
                f"Saved {len(aligned_lights)} aligned frames to 'aligned/' folder", "info"
            )

        # Display reference frame before freeing memory
        ref_display = aligned_lights[0]
        self._display_image(ref_display)
        n_aligned = len(aligned_lights)

        # *** Free aligned_lights from RAM and VRAM — stacking reads from disk ***
        del aligned_lights
        import gc as _gc; _gc.collect()
        try:
            get_device_manager().empty_cache()
        except Exception:
            log.debug("GPU cache flush failed")

        if project_dir and aligned_paths:
            # Register aligned frames in the project (REGISTERED section)
            for p in aligned_paths:
                if p.exists():
                    self._project.remove_frame(p)
            self._project.add_frames([p for p in aligned_paths if p.exists()], FrameType.ALIGNED)
            # Store paths for subsequent stack-from-disk
            self._aligned_paths = aligned_paths

            # Refresh file tree
            if hasattr(self, "_project_panel"):
                self._project_panel.refresh()
        else:
            self._aligned_paths = []

        # Clear subframe selection — stacking should now use the freshly aligned _aligned_paths
        self._subframe_selected_paths = []
        self._tools_panel.set_subframe_count(0, 0)

        self._log_panel.log(
            f"Alignment complete: {n_aligned} frames aligned. Ready to stack.",
            "success",
        )
        if self._project:
            self._project.add_history("Alignment", {"n_frames": n_aligned})
            self._save_project()

    def _on_run_stacking(self):

        # Subframe Selector can override the aligned path list with a quality-filtered subset.
        if self._subframe_selected_paths:
            aligned_paths = [Path(p) for p in self._subframe_selected_paths]
            self._log_panel.log(
                f"Using {len(aligned_paths)} subframe-selected frames for stacking", "info"
            )
        else:
            aligned_paths = getattr(self, "_aligned_paths", [])

        if not aligned_paths and self._project:
            aligned_paths = [
                e.path for e in self._project.frames_by_type(FrameType.ALIGNED)
                if e.path.exists()
            ]

        if aligned_paths:
            params = self._tools_panel.get_stacking_params()
            n = len(aligned_paths)
            self._log_panel.log(f"Stacking {n} aligned frames…", "info")
            cached = getattr(self._project, "frame_scores", {}) if self._project else {}
            self._start_worker(
                _score_and_stack_worker,
                aligned_paths,
                params,
                cached,
                on_done=self._on_stacking_done,
            )
            return

        # No aligned frames on disk — prefer path-based align+stack to avoid loading
        # all frames into RAM on the main thread. Fall back to in-memory only when
        # calibrated lights are already loaded (small dataset path).
        light_paths = self._get_light_paths() if not self._calibrated_lights else None

        if light_paths is not None:
            # Disk-based align + stack (memory-safe for large datasets)
            params = self._tools_panel.get_stacking_params()
            drizzle_enabled, drizzle_params = self._tools_panel.get_drizzle_params()
            if drizzle_enabled:
                self._log_panel.log("Drizzle requires in-memory path; loading frames…", "info")
                # Fall through to load lights below
            else:
                output_dir = str(self._project.directory / "aligned") if self._project else None
                if output_dir is None:
                    self._log_panel.log("No project directory for alignment output", "error")
                    return
                self._log_panel.log(
                    f"Aligning + stacking {len(light_paths)} frames (disk path)…", "info"
                )
                self._start_worker(
                    _align_with_optional_filter,
                    light_paths,
                    output_dir,
                    stk_params=params,
                    on_done=self._on_path_alignment_done,
                )
                return

        light_paths = self._get_raw_light_paths()
        if light_paths is None:
            return

        params = self._tools_panel.get_stacking_params()
        drizzle_enabled, drizzle_params = self._tools_panel.get_drizzle_params()
        self._log_panel.log(f"Loading and stacking {len(light_paths)} frames…", "info")

        def _load_and_stack_work(paths, stk_params, progress=None):
            from pathlib import Path

            from astraios.core.image_io import load_image
            loaded = []
            total = len(paths)
            for i, p in enumerate(paths):
                try:
                    img = load_image(p)
                    if img is not None:
                        loaded.append(img)
                except Exception as e:
                    log.warning("Failed to load %s: %s", p, e)
                if progress and total > 1:
                    progress((i + 1) / total * 0.3, f"Loading {Path(p).name}")
            if not loaded:
                raise RuntimeError("Failed to load any light frames")

            if drizzle_enabled:
                import numpy as _np

                from astraios.core.drizzle import drizzle_integrate
                arrays = [f if isinstance(f, _np.ndarray) else f.data for f in loaded]
                if progress:
                    progress(0.35, "Drizzle integrating…")
                return drizzle_integrate(
                    arrays, params=drizzle_params,
                    progress=lambda f, m: progress(0.35 + f * 0.65, m) if progress else None,
                )
            else:
                from astraios.core.subframe_selector import score_subframes

                def _prog(f, m):
                    if progress:
                        progress(f, m)

                if (
                    hasattr(stk_params, "integration")
                    and stk_params.integration == IntegrationMethod.WEIGHTED_AVERAGE
                ):
                    paths_list = [str(img.file_path) for img in loaded if img.file_path is not None]
                    if len(paths_list) == len(loaded):
                        scores = score_subframes(
                            paths_list, SubframeSelectorParams(),
                            progress=lambda f, m: _prog(f * 0.4, m),
                        )
                        stk_params.frame_weights = [
                            max(float(sc.quality_score), 1e-6) for sc in scores
                        ]

                _prog(0.4, f"Stacking {len(loaded)} frames…")
                return stack_images(
                    loaded, stk_params, align=True,
                    progress=lambda f, m: _prog(0.4 + f * 0.6, m),
                )

        def _on_load_and_stack_done(result):
            if drizzle_enabled:
                import numpy as _np

                from astraios.core.image_io import ImageData
                img = ImageData(data=result.data.astype(_np.float32), header={})
                self._display_image(img)
                self._log_panel.log(
                    f"Drizzle complete: {result.n_frames} frames, "
                    f"{result.output_scale}× scale ({img.shape_str})",
                    "success",
                )
                if self._project:
                    self._project.add_history(
                        "Drizzle",
                        {"n_frames": result.n_frames, "scale": result.output_scale},
                    )
                    self._save_project()
            else:
                self._on_stacking_done(result)

        self._start_worker(
            _load_and_stack_work, light_paths, params,
            on_done=_on_load_and_stack_done,
        )

    @pyqtSlot(object)
    def _on_stacking_done(self, result):
        self._display_image(result.image)
        # Flush GPU allocator — alignment tensors accumulate during stack+align
        try:
            import gc as _gc; _gc.collect()
            get_device_manager().empty_cache()
        except Exception:
            log.debug("GPU cache flush failed")
        self._log_panel.log(
            f"Stacking complete: {result.n_frames} frames, {result.total_rejected} pixels rejected",
            "success",
        )
        if self._project:
            self._project.add_history("Stacking", {"n_frames": result.n_frames})
            self._save_project()
            # Auto-save the integrated image to the project output folder.
            try:
                out_dir = self._project.output_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                import re as _re
                safe_name = _re.sub(r"[^\w\-]", "_", self._project.name)
                out_path = out_dir / f"{safe_name}_integrated.fits"
                save_image(result.image, str(out_path))
                self._log_panel.log(f"Integrated image saved: {out_path}", "success")
            except Exception as exc:
                log.warning("Could not auto-save integrated image: %s", exc)
                self._log_panel.log(f"Auto-save failed: {exc}", "warning")

    # ── Multi-session stacking ────────────────────────────────────────────────

    @pyqtSlot()
    def _on_ms_add_folder(self):
        """Let the user pick a folder of light frames as a new session."""
        from PyQt6.QtWidgets import QFileDialog, QInputDialog
        folder = QFileDialog.getExistingDirectory(self, "Select Session Folder", "")
        if not folder:
            return
        import glob as _glob
        from pathlib import Path
        extensions = ("*.fits", "*.fit", "*.fts", "*.xisf", "*.FITS", "*.FIT", "*.FTS")
        files = []
        for ext in extensions:
            files.extend(_glob.glob(str(Path(folder) / ext)))
        if not files:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No frames", f"No FITS/XISF files found in:\n{folder}")
            return

        # Ask user for a session name (default = folder name)
        default_name = Path(folder).name
        name, ok = QInputDialog.getText(self, "Session Name", "Name for this session:", text=default_name)
        if not ok or not name.strip():
            name = default_name

        # Load frames in background
        from PyQt6.QtCore import Qt as _Qt
        self._ms_pending_name = name.strip()
        self._ms_folder_loader = _MsFolderLoader(sorted(files))
        self._ms_folder_loader.progress.connect(
            self._log_panel.update_progress, _Qt.ConnectionType.QueuedConnection
        )
        self._ms_folder_loader.error.connect(
            lambda msg: self._log_panel.log(msg, "warning"),
            _Qt.ConnectionType.QueuedConnection,
        )
        self._ms_folder_loader.finished.connect(
            self._on_ms_folder_loaded, _Qt.ConnectionType.QueuedConnection
        )
        self._set_processing_locked(True)
        self._ms_folder_loader.start()

    @pyqtSlot(list)
    def _on_ms_folder_loaded(self, loaded: list):
        """Process frames loaded by _MsFolderLoader."""
        self._set_processing_locked(False)
        self._log_panel.reset_progress()

        name = self._ms_pending_name
        if not loaded:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Load failed", "Could not load any frames from the folder.")
            return

        from astraios.core.multi_session import SessionGroup
        total_time = 0.0
        for img in loaded:
            exp = (img.header or {}).get("EXPTIME", (img.header or {}).get("EXPOSURE", None))
            if exp is not None:
                try:
                    total_time += float(exp)
                except (ValueError, TypeError):
                    pass
        session = SessionGroup(
            frames=loaded,
            name=name,
            integration_time=total_time if total_time > 0 else None,
        )
        self._ms_sessions.append(session)
        self._tools_panel.ms_add_session(name, len(loaded))
        self._log_panel.log(f"Session '{name}': {len(loaded)} frames loaded", "success")

    @pyqtSlot()
    def _on_ms_clear(self):
        self._ms_sessions.clear()

    @pyqtSlot()
    def _on_run_multi_session(self):
        if len(self._ms_sessions) < 2:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, "Multi-Session",
                "Please add at least 2 sessions using 'Add Session…' before stacking."
            )
            return

        ms_params_dict = self._tools_panel.get_multi_session_params()
        stacking_params = self._tools_panel.get_stacking_params()

        from astraios.core.multi_session import MultiSessionParams

        ms_params = MultiSessionParams(
            per_session_params=stacking_params,
            weight_mode=ms_params_dict["weight_mode"],
            normalize_background=ms_params_dict["normalize_background"],
            align_sub_stacks=ms_params_dict["align_sub_stacks"],
        )

        sessions_snapshot = list(self._ms_sessions)
        n = len(sessions_snapshot)
        names = [s.name for s in sessions_snapshot]
        self._log_panel.log(
            f"Multi-session stacking: {n} sessions ({', '.join(names)})…", "info"
        )

        def _ms_work(sessions, params, progress=None):
            from astraios.core.multi_session import stack_multi_session as _ms
            return _ms(sessions, params, progress=progress or (lambda f, m: None))

        self._start_worker(_ms_work, sessions_snapshot, ms_params,
                           on_done=self._on_multi_session_done)

    @pyqtSlot(object)
    def _on_multi_session_done(self, result):
        self._display_image(result.image)
        total_frames = sum(r.n_frames for r in result.sub_stacks)
        weight_info = ", ".join(
            f"{name}: {w:.2f}"
            for name, w in zip(result.session_names, result.weights)
        )
        self._log_panel.log(
            f"Multi-session complete: {result.n_sessions} sessions, "
            f"{total_frames} total frames  [weights: {weight_info}]",
            "success",
        )
        if self._project:
            self._project.add_history(
                "Multi-Session Stack",
                {"n_sessions": result.n_sessions, "n_frames": total_frames},
            )
            self._save_project()

    @pyqtSlot()
    def _on_run_stretch(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_stretch_params()
        _p = params

        def _work(data, progress=None):
            return auto_stretch(data, _p)

        def _done(stretched):
            self._update_current_image(stretched, "Stretch applied")
            if self._project:
                self._project.add_history(
                    "Auto-Stretch", {"midtone": _p.midtone, "shadow_clip": _p.shadow_clip}
                )
            self._macro_recorder.record_step(
                "auto_stretch", {"midtone": _p.midtone, "shadow_clip": _p.shadow_clip}
            )
            self._tools_panel.reset_stretch_params()

        self._start_worker(_work, self._current_image.data, on_done=_done)

    def _on_stretch_preview(self):
        if self._current_image is None:
            return
        if not self._tools_panel.split_preview_enabled:
            self._on_preview_cancelled()
            return
        self._preview_indicator.setText("● Live Preview: Auto Stretch")
        self._stretch_preview_timer.start()

    def _do_stretch_preview(self):
        if self._current_image is None:
            return
        import numpy as np
        small, _scale = self._downscale_for_preview(self._current_image.data)
        params = self._tools_panel.get_stretch_params()
        stretched = auto_stretch(small, params)
        if stretched.size == 0:
            return
        # Mirror the Apply display path exactly:
        #   Apply → auto_stretch_for_display_ref(stretched_result, before.data)
        # Use current image (small) as the "before" reference — same as what Apply would use.
        if stretched.ndim == 3:
            after_hwc = np.transpose(stretched, (1, 2, 0))
            ref_hwc   = np.transpose(small,     (1, 2, 0))
        else:
            after_hwc = np.stack([stretched, stretched, stretched], axis=-1)
            ref_hwc   = np.stack([small,     small,     small],     axis=-1)
        after_disp = auto_stretch_for_display_ref(after_hwc, ref_hwc)
        after_rgb  = np.clip(after_disp * 255, 0, 255).astype(np.uint8)
        self._canvas.set_after_image(after_rgb)
        self._canvas.set_split_mode(True)

    def _downscale_for_preview(self, data):
        """Downscale image data so longest side is at most 1024 px."""
        import cv2
        import numpy as np

        if data.ndim == 2:
            h, w = data.shape
        else:
            c, h, w = data.shape
        longest = max(h, w)
        if longest <= 1024:
            return data.copy(), 1.0
        scale = 1024.0 / longest
        new_w, new_h = int(w * scale), int(h * scale)
        if data.ndim == 2:
            return cv2.resize(data, (new_w, new_h), interpolation=cv2.INTER_AREA), scale
        channels = [
            cv2.resize(data[ch], (new_w, new_h), interpolation=cv2.INTER_AREA) for ch in range(c)
        ]
        return np.stack(channels, axis=0), scale

    @pyqtSlot(str)
    def _on_preview_requested(self, tool_name: str):
        """Run the named tool on a downscaled copy and show split preview."""
        self._pending_preview_tool = tool_name
        self._preview_timer.start()
        label = tool_name.replace("_", " ").title()
        self._preview_indicator.setText(f"● Live Preview: {label}")

    def _do_preview_requested(self):
        import numpy as np
        tool_name = self._pending_preview_tool
        if tool_name is None or self._current_image is None:
            return

        # Use cached downscaled image — avoids re-downscaling 144MB on every slider drag.
        if self._preview_small_cache is not None:
            small, _scale = self._preview_small_cache
        else:
            small, _scale = self._downscale_for_preview(self._current_image.data)

        try:
            result = self._run_tool_preview(tool_name, small)
        except Exception as e:
            self._log_panel.log(f"Preview failed for {tool_name}: {e}", "warning")
            return

        if result is None or result.size == 0:
            return

        # Convert to HWC. Always use small (current image) as the stretch reference
        # so the preview reflects what the tool does to the current image — avoids
        # black previews caused by stale bright refs from previous operations.
        if result.ndim == 2:
            after_hwc = np.stack([result, result, result], axis=-1)
            ref_hwc = np.stack([small, small, small], axis=-1) if small.ndim == 2 else np.transpose(small, (1, 2, 0))
        else:
            after_hwc = np.transpose(result, (1, 2, 0))
            ref_hwc = np.stack([small, small, small], axis=-1) if small.ndim == 2 else np.transpose(small, (1, 2, 0))
        after_disp = auto_stretch_for_display_ref(after_hwc, ref_hwc)
        after_rgb = np.clip(after_disp * 255, 0, 255).astype(np.uint8)
        self._canvas.set_after_image(after_rgb)
        self._canvas.set_split_mode(True)

    def _run_tool_preview(self, tool_name: str, data):
        """Execute a tool on the given data and return the result array."""
        tp = self._tools_panel
        if tool_name == "cosmetic":
            r = cosmetic_correction(data, tp.get_cosmetic_params())
            return r.data
        elif tool_name == "banding":
            return banding_reduction(data, tp.get_banding_params())
        elif tool_name == "deconvolution":
            params = tp.get_deconvolution_params()
            if isinstance(params, SpatialDeconvParams):
                return richardson_lucy_spatial(data, params=params)
            return richardson_lucy(data, params=params)
        elif tool_name == "denoise":
            return denoise(data, tp.get_denoise_params())
        elif tool_name == "frequency_separation":
            from astraios.core.frequency_separation import frequency_separation
            return frequency_separation(data, tp.get_frequency_separation_params())
        elif tool_name == "star_stretch":
            from astraios.core.star_stretch import star_stretch
            return star_stretch(data, tp.get_star_stretch_params())
        elif tool_name == "statistical_stretch":
            from astraios.core.stretch import statistical_stretch
            return statistical_stretch(data, tp.get_statistical_stretch_params())
        elif tool_name == "scnr":
            return scnr(data, tp.get_scnr_params())
        elif tool_name == "color_adjust":
            return color_adjust(data, tp.get_color_adjust_params())
        elif tool_name == "wavelet":
            return wavelet_sharpen(data, tp.get_wavelet_params())
        elif tool_name == "mlt":
            return wavelet_sharpen(data, tp.get_mlt_params())
        elif tool_name == "local_contrast":
            return local_contrast_enhance(data, tp.get_local_contrast_params())
        elif tool_name == "unsharp_mask":
            return unsharp_mask(data, tp.get_unsharp_mask_params())
        elif tool_name == "median_filter":
            return median_filter(data, tp.get_median_filter_params())
        elif tool_name == "histogram_transform":
            return histogram_transform(data, tp.get_histogram_transform_params())
        elif tool_name == "ghs":
            return generalized_hyperbolic_stretch(data, tp.get_ghs_params())
        elif tool_name == "arcsinh_stretch":
            return arcsinh_stretch(data, tp.get_arcsinh_params())
        elif tool_name == "curves":
            return curves_transform(data, tp.get_curves_params())
        return None

    @pyqtSlot()
    def _on_preview_cancelled(self):
        """Clear the split preview and restore normal view."""
        self._pending_preview_tool = None
        self._preview_timer.stop()
        self._stretch_preview_timer.stop()
        self._canvas.set_view_mode("after")
        self._canvas.clear_after_image()
        self._preview_indicator.setText("")

    # ---------- Transform operations ----------

    @pyqtSlot()
    def _on_start_crop_draw(self):
        """Toggle interactive crop-draw mode on the canvas."""
        currently_active = getattr(self._canvas, '_crop_mode', False)
        self._canvas.set_crop_mode(not currently_active)

    @pyqtSlot(int, int, int, int)
    def _on_crop_rect_selected(self, x: int, y: int, w: int, h: int):
        """Called when user finishes drawing a crop rectangle on the canvas.

        Coordinates are already in full-res space (the canvas applies _display_scale).
        """
        self._tools_panel.set_crop_from_rect(x, y, w, h)
        self._log_panel.log(f"Crop region set: x={x}, y={y}, w={w}, h={h}", "info")

    def _on_run_crop(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_crop_params()
        result = crop(self._current_image.data, params)
        self._update_current_image(result, f"Cropped to {result.shape[-1]}x{result.shape[-2]}", geometric=True)
        if self._project:
            self._project.add_history(
                "Crop",
                {"x": params.x, "y": params.y, "width": params.width, "height": params.height},
            )
        self._macro_recorder.record_step(
            "crop", {"x": params.x, "y": params.y, "width": params.width, "height": params.height}
        )

    @pyqtSlot()
    def _on_run_rotate(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_rotate_params()
        result = rotate(self._current_image.data, params)
        self._update_current_image(result, f"Rotated ({params.angle.name})", geometric=True)
        if self._project:
            self._project.add_history("Rotate", {"angle": params.angle.name})
        self._macro_recorder.record_step("rotate", {"angle": params.angle.name})

    @pyqtSlot()
    def _on_run_flip(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_flip_params()
        result = flip(self._current_image.data, params)
        self._update_current_image(result, f"Flipped ({params.axis.name})", geometric=True)
        if self._project:
            self._project.add_history("Flip", {"axis": params.axis.name})
        self._macro_recorder.record_step("flip", {"axis": params.axis.name})

    @pyqtSlot()
    def _on_run_resize(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_resize_params()
        result = resize(self._current_image.data, params)
        self._update_current_image(result, f"Resized to {result.shape[-1]}x{result.shape[-2]}", geometric=True)
        if self._project:
            self._project.add_history("Resize", {"scale": params.scale})
        self._macro_recorder.record_step("resize", {"scale": params.scale})

    @pyqtSlot()
    def _on_run_bin(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_bin_params()
        result = bin_image(self._current_image.data, params)
        self._update_current_image(
            result, f"Binned {params.factor}x{params.factor} ({params.mode.name})"
        )
        if self._project:
            self._project.add_history("Bin", {"factor": params.factor, "mode": params.mode.name})
        self._macro_recorder.record_step("bin", {"factor": params.factor})

    @pyqtSlot()
    def _on_run_invert(self):
        if self._current_image is None:
            return
        result = invert(self._current_image.data)
        self._update_current_image(result, "Image inverted", tool="invert")
        if self._project:
            self._project.add_history("Invert", {})
        self._macro_recorder.record_step("invert")

    # ---------- New tool operations ----------

    @pyqtSlot()
    def _on_run_unsharp_mask(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_unsharp_mask_params()
        self._log_panel.log(
            f"Applying Unsharp Mask (r={params.radius}, a={params.amount})...", "info"
        )
        _p = params

        def _work(data, progress=None):
            return unsharp_mask(data, _p)

        def _done(result):
            self._update_current_image(result, "Unsharp mask applied")
            if self._project:
                self._project.add_history(
                    "Unsharp Mask", {"radius": _p.radius, "amount": _p.amount}
                )
            self._macro_recorder.record_step(
                "unsharp_mask", {"radius": _p.radius, "amount": _p.amount}
            )

        self._start_worker(_work, self._current_image.data, on_done=_done)

    @pyqtSlot()
    def _on_run_median_filter(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_median_filter_params()
        self._log_panel.log(f"Applying Median Filter (k={params.kernel_size})...", "info")
        _p = params

        def _work(data, progress=None):
            return median_filter(data, _p)

        def _done(result):
            self._update_current_image(result, "Median filter applied")
            if self._project:
                self._project.add_history("Median Filter", {"kernel_size": _p.kernel_size})

        self._start_worker(_work, self._current_image.data, on_done=_done)

    @pyqtSlot()
    def _on_run_abe(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_abe_params()
        self._log_panel.log("Running ABE (RBF background extraction)...", "info")
        self._start_worker(
            abe_extract,
            self._current_image.data,
            params=params,
            on_done=self._on_abe_done,
        )

    @pyqtSlot(object)
    def _on_abe_done(self, result):
        corrected, bg_model = result
        self._update_current_image(corrected, "ABE background extraction complete")
        if self._project:
            self._project.add_history("ABE", {})

    @pyqtSlot()
    def _on_run_vignette(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_vignette_params()
        self._log_panel.log("Applying vignette correction...", "info")
        _p = params

        def _work(data, progress=None):
            return correct_vignette(data, _p)

        def _done(result):
            self._update_current_image(result, "Vignette correction applied")
            if self._project:
                self._project.add_history(
                    "Vignette Correction", {"strength": _p.strength, "falloff": _p.falloff}
                )

        self._start_worker(_work, self._current_image.data, on_done=_done)

    @pyqtSlot()
    def _on_run_ca(self):
        if self._current_image is None:
            return
        if self._current_image.data.ndim != 3 or self._current_image.data.shape[0] < 3:
            self._log_panel.log("Chromatic aberration requires a color image", "error")
            return
        params = self._tools_panel.get_ca_params()
        self._log_panel.log("Correcting chromatic aberration...", "info")
        _p = params

        def _work(data, progress=None):
            return correct_chromatic_aberration(data, _p)

        def _done(result):
            self._update_current_image(result, "Chromatic aberration corrected")
            if self._project:
                self._project.add_history("CA Correction", {"auto": _p.auto_detect})

        self._start_worker(_work, self._current_image.data, on_done=_done)

    @pyqtSlot()
    def _on_show_statistics(self):
        if self._current_image is None:
            self._log_panel.log("Load an image first", "warning")
            return
        from astraios.ui.dialogs.statistics_dialog import StatisticsDialog

        stats = compute_image_statistics(self._current_image.data)
        dialog = StatisticsDialog(stats, self)
        dialog.exec()

    @pyqtSlot()
    def _on_edit_fits_header(self):
        if self._current_image is None:
            self._log_panel.log("Load an image first", "warning")
            return
        from astraios.ui.dialogs.fits_header_dialog import FITSHeaderDialog

        path = getattr(self._current_image, "file_path", None)
        dlg = FITSHeaderDialog(self._current_image.header, file_path=path, parent=self)
        if dlg.exec() == FITSHeaderDialog.DialogCode.Accepted:
            self._current_image.header = dlg.get_header()
            self._log_panel.log("FITS header updated", "success")

    @pyqtSlot()
    def _on_measure_psf(self):
        if self._current_image is None:
            return
        from astraios.core.psf import measure_psf

        self._log_panel.log("Measuring PSF from stars…", "info")
        _cutout = int(self._tools_panel._psf_cutout_spin.value())
        _force_cpu = self._tools_panel._psf_force_cpu.isChecked()

        def _psf_work(data, progress=None):
            # Use lower min_flux so measurement works on stretched images too
            return measure_psf(
                data,
                cutout_radius=_cutout,
                force_cpu=_force_cpu,
                min_flux=0.05,
                max_flux=0.99,
            )

        def _on_psf_done(result):
            if result is None or result.n_stars_used == 0:
                self._log_panel.log("PSF measurement failed: no stars found", "warning")
                return
            self._tools_panel.set_psf_result(
                result.fwhm, result.fwhm_x, result.fwhm_y,
                result.ellipticity, result.theta,
                result.n_stars_used, result.fwhm_std,
            )
            self._tools_panel.set_psf_fwhm(result.fwhm)
            self._log_panel.log(
                f"PSF: FWHM={result.fwhm:.2f}px  ellipticity={result.ellipticity:.2f}"
                f"  ({result.n_stars_used} stars)",
                "success",
            )

        self._start_worker(_psf_work, self._current_image.data, on_done=_on_psf_done)

    @pyqtSlot()
    def _on_run_continuum_subtraction(self):
        if self._current_image is None:
            return
        from pathlib import Path as _Path

        from PyQt6.QtWidgets import QFileDialog

        from astraios.core.image_io import load_image
        from astraios.core.narrowband import continuum_subtraction

        bb_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Broadband (Continuum) Image",
            "",
            "Images (*.fits *.fit *.fts *.xisf *.tif *.tiff *.png *.jpg)",
        )
        if not bb_path:
            return

        scale = self._tools_panel.get_continuum_scale()
        self._log_panel.log(
            f"Continuum subtraction: scale={scale:.3f}, broadband={_Path(bb_path).name}", "info"
        )

        def _cont_work(nb_data, bb_path_str, scale, progress=None):
            import numpy as _np
            bb_img = load_image(bb_path_str)
            bb_ch = _np.mean(bb_img.data, axis=0) if bb_img.data.ndim == 3 else bb_img.data
            nb_ch = _np.mean(nb_data, axis=0) if nb_data.ndim == 3 else nb_data
            result_ch = continuum_subtraction(nb_ch, bb_ch, scale)
            # Preserve original dimensionality: broadcast mono result to 3D if needed
            if nb_data.ndim == 3:
                return _np.stack([result_ch, result_ch, result_ch], axis=0)
            return result_ch

        def _on_cont_done(result_ch):
            self._update_current_image(result_ch, "Continuum subtraction applied")

        self._start_worker(
            _cont_work,
            self._current_image.data,
            bb_path,
            scale,
            on_done=_on_cont_done,
        )

    # ── Dynamic background sample placement ──────────────────────────────────

    @pyqtSlot(bool)
    def _on_toggle_sample_mode(self, enabled: bool):
        self._canvas.set_sample_mode(enabled)
        if not enabled:
            self._canvas.set_sample_points(self._bg_samples)

    @pyqtSlot(float, float)
    def _on_sample_placed(self, x: float, y: float):
        self._bg_samples.append((x, y))
        self._canvas.set_sample_points(self._bg_samples)
        self._tools_panel.set_bg_sample_count(len(self._bg_samples))

    @pyqtSlot(float, float)
    def _on_sample_removed(self, x: float, y: float):
        if not self._bg_samples:
            return
        import math as _math
        nearest_idx = min(
            range(len(self._bg_samples)),
            key=lambda i: _math.hypot(self._bg_samples[i][0] - x, self._bg_samples[i][1] - y),
        )
        self._bg_samples.pop(nearest_idx)
        self._canvas.set_sample_points(self._bg_samples)
        self._tools_panel.set_bg_sample_count(len(self._bg_samples))

    @pyqtSlot()
    def _on_clear_bg_samples(self):
        self._bg_samples.clear()
        self._canvas.clear_sample_points()
        self._tools_panel.set_bg_sample_count(0)

    @pyqtSlot(int, int, int)
    def _on_add_bg_grid(self, rows: int, cols: int, box_size: int):
        """Auto-place background sample points in an evenly-spaced grid."""
        if self._current_image is None:
            return
        data = self._current_image.data
        h = data.shape[-2]
        w = data.shape[-1]
        margin = box_size // 2 + 2

        import numpy as _np
        ys = _np.linspace(margin, h - margin - 1, rows).astype(int)
        xs = _np.linspace(margin, w - margin - 1, cols).astype(int)
        added = 0
        for y in ys:
            for x in xs:
                pt = (float(x), float(y))
                if pt not in self._bg_samples:
                    self._bg_samples.append(pt)
                    added += 1

        self._canvas.set_sample_points(self._bg_samples)
        self._tools_panel.set_bg_sample_count(len(self._bg_samples))
        self._log_panel.log(
            f"Added {added} grid sample{'s' if added != 1 else ''} "
            f"({rows}×{cols}, {len(self._bg_samples)} total)", "info"
        )

    # ── WCS overlay ──────────────────────────────────────────────────────────

    @pyqtSlot(bool)
    def _on_toggle_wcs_overlay(self, enabled: bool):
        if enabled and not self._wcs_overlay_stars:
            self._log_panel.log(
                "No WCS data available — run Solve & Calibrate (PCC) first", "warning"
            )
            return
        self._canvas.set_wcs_overlay_visible(enabled)

    def _update_wcs_overlay(self, wcs: dict, catalog_stars: list):
        """Project catalog stars to image pixel coordinates and store for overlay."""
        if not wcs or not catalog_stars:
            return
        if self._current_image is None:
            return

        import numpy as _np

        from astraios.core.color_calibration import _make_pixel_to_sky

        h = self._current_image.data.shape[-2] if self._current_image.data.ndim == 3 else self._current_image.data.shape[0]
        w = self._current_image.data.shape[-1] if self._current_image.data.ndim == 3 else self._current_image.data.shape[1]

        sky_fn = _make_pixel_to_sky(wcs, w, h)
        scale_deg = (wcs.get("scale") or 1.0) / 3600.0
        match_radius = max(scale_deg * 10, 0.005)

        overlay = []
        for cat in catalog_stars:
            if cat.g_mag is None:
                continue
            ra0, dec0 = sky_fn(w / 2, h / 2)
            cos_dec = _np.cos(_np.radians(dec0))
            dra = (cat.ra_deg - ra0) * cos_dec
            ddec = cat.dec_deg - dec0
            px = w / 2 + dra / max(scale_deg, 1e-10)
            py = h / 2 - ddec / max(scale_deg, 1e-10)
            if 0 <= px < w and 0 <= py < h:
                bp_rp = (float(cat.bp_mag) - float(cat.rp_mag)) if (cat.bp_mag is not None and cat.rp_mag is not None) else 0.0
                overlay.append((float(px), float(py), float(cat.g_mag), bp_rp))

        self._wcs_overlay_stars = overlay
        self._current_wcs = wcs
        self._canvas.set_overlay_stars(overlay)
        # Auto-populate DSO and constellation annotations
        self._update_dso_annotations(wcs)
        self._update_constellation_overlay(wcs)

    def _update_overlays_from_wcs(self, wcs: dict):
        """Query Gaia star catalog and update all WCS-based overlays."""
        if not wcs or self._current_image is None:
            return
        try:
            from astraios.core.star_catalog import query_gaia_dr3
            ra_center = wcs.get("ra_center", 0.0)
            dec_center = wcs.get("dec_center", 0.0)
            catalog_stars = query_gaia_dr3(ra_center, dec_center, radius_deg=0.5)
            self._log_panel.log(f"Gaia DR3: {len(catalog_stars)} stars retrieved", "info")
            self._update_wcs_overlay(wcs, list(catalog_stars))
        except Exception as e:
            self._log_panel.log(f"Catalog query skipped: {e}", "info")

    def _update_dso_annotations(self, wcs: dict):
        """Project DSO catalog entries to image pixel coordinates and push to canvas."""
        if not wcs or self._current_image is None:
            return
        import numpy as _np

        from astraios.core.dso_catalog import query_dso_in_field

        h = self._current_image.data.shape[-2] if self._current_image.data.ndim == 3 else self._current_image.data.shape[0]
        w = self._current_image.data.shape[-1] if self._current_image.data.ndim == 3 else self._current_image.data.shape[1]

        ra_center = wcs.get("ra_center", 0.0)
        dec_center = wcs.get("dec_center", 0.0)
        scale_deg = (wcs.get("scale") or 1.0) / 3600.0
        fov_deg = max(w, h) * scale_deg * 1.5

        dsos = query_dso_in_field(ra_center, dec_center, fov_deg)
        cos_dec = _np.cos(_np.radians(dec_center))
        annotations = []
        for dso in dsos:
            dra = (dso.ra_deg - ra_center) * cos_dec
            ddec = dso.dec_deg - dec_center
            px = w / 2 + dra / max(scale_deg, 1e-10)
            py = h / 2 - ddec / max(scale_deg, 1e-10)
            if -50 <= px < w + 50 and -50 <= py < h + 50:
                annotations.append((float(px), float(py), dso.name, dso.type_code))

        self._canvas.set_dso_annotations(annotations)

    def _update_constellation_overlay(self, wcs: dict):
        """Project constellation line segments to image pixel coords."""
        if not wcs or self._current_image is None:
            return
        import numpy as _np

        from astraios.core.constellations import CONSTELLATION_LINES

        h = self._current_image.data.shape[-2] if self._current_image.data.ndim == 3 else self._current_image.data.shape[0]
        w = self._current_image.data.shape[-1] if self._current_image.data.ndim == 3 else self._current_image.data.shape[1]

        ra_center = wcs.get("ra_center", 0.0)
        dec_center = wcs.get("dec_center", 0.0)
        scale_deg = (wcs.get("scale") or 1.0) / 3600.0
        cos_dec = _np.cos(_np.radians(dec_center))
        segments = []
        for cl in CONSTELLATION_LINES:
            for seg in cl.segments:
                projected = []
                for ra_deg, dec_deg in seg:
                    dra = (ra_deg - ra_center) * cos_dec
                    ddec = dec_deg - dec_center
                    px = w / 2 + dra / max(scale_deg, 1e-10)
                    py = h / 2 - ddec / max(scale_deg, 1e-10)
                    if -1000 <= px < w + 1000 and -1000 <= py < h + 1000:
                        projected.append((float(px), float(py)))
                if len(projected) >= 2:
                    segments.append(projected)

        self._constellation_segments = segments
        self._canvas.set_constellation_lines(segments)

    def _on_toggle_dso_overlay(self, enabled=None):
        """Toggle DSO annotation overlay visibility on the canvas."""
        if enabled is None:
            current = getattr(self._canvas, '_show_dso_overlay', False)
            enabled = not current
        if enabled and not getattr(self._canvas, '_dso_annotations', None):
            self._log_panel.log(
                "No DSO annotations available — run Solve & Calibrate (PCC) first", "warning"
            )
            return
        self._canvas.set_dso_overlay_visible(enabled)

    def _on_toggle_constellation_overlay(self, enabled=None):
        """Toggle constellation line overlay visibility."""
        if enabled is None:
            current = getattr(self._canvas, '_show_constellation_overlay', False)
            enabled = not current
        if enabled and not self._constellation_segments and hasattr(self, '_current_wcs'):
            self._update_constellation_overlay(self._current_wcs)
        self._canvas.set_constellation_overlay_visible(enabled)
        if hasattr(self, "_tb_const_btn"):
            self._tb_const_btn.blockSignals(True)
            self._tb_const_btn.setChecked(enabled)
            self._tb_const_btn.blockSignals(False)

    # ── Python console ────────────────────────────────────────────────────────

    @pyqtSlot()
    def _on_open_python_console(self):
        from PyQt6.QtWidgets import QDockWidget

        from astraios.ui.widgets.python_console import PythonConsoleWidget

        if self._python_console_dock is None:
            console = PythonConsoleWidget()
            console.image_updated.connect(self._on_console_image_updated)
            console.image_preview.connect(self._on_console_image_preview)
            dock = QDockWidget("Python Console", self)
            dock.setWidget(console)
            dock.setMinimumWidth(560)
            dock.setMinimumHeight(340)
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
            self._python_console_dock = dock

        self._python_console_dock.show()
        self._python_console_dock.raise_()
        # Inject current image
        self._sync_console_image()

    def _sync_console_image(self):
        if self._python_console_dock is None:
            return
        console = self._python_console_dock.widget()
        if console and self._current_image is not None:
            console.set_image(self._current_image)

    @pyqtSlot(object)
    def _on_console_image_updated(self, arr):
        import numpy as _np
        if not isinstance(arr, _np.ndarray):
            self._log_panel.log("apply() requires a numpy ndarray", "error")
            return
        self._update_current_image(arr, "Image updated from Python console")

    @pyqtSlot(object)
    def _on_console_image_preview(self, arr):
        """Display array on canvas as a temporary preview without replacing current image."""
        import numpy as _np
        if not isinstance(arr, _np.ndarray):
            return
        arr = _np.clip(arr, 0, 1).astype(_np.float32)
        from astraios.core.image_io import ImageData
        preview = ImageData(data=arr, header={})
        self._display_image(preview)

    @pyqtSlot()
    def _on_open_star_mask(self):
        if self._current_image is None:
            return
        from astraios.ui.dialogs.star_mask_dialog import StarMaskDialog

        dialog = StarMaskDialog(self._current_image.data, self)
        dialog.mask_ready.connect(self._on_star_mask_ready)
        dialog.exec()
        dialog.deleteLater()  # don't linger as a child holding image arrays

    @pyqtSlot(object)
    def _on_star_mask_ready(self, mask):
        self._log_panel.log("Star mask generated", "success")

    @pyqtSlot()
    def _show_live_stack_dialog(self):
        from astraios.ui.dialogs.live_stack_dialog import LiveStackDialog

        dialog = LiveStackDialog(self)
        dialog.exec()

    def _show_blink_dialog(self):
        from astraios.ui.dialogs.blink_dialog import BlinkDialog

        paths: list[str] = []
        if self._aligned_paths:
            paths = [str(p) for p in self._aligned_paths]
        elif self._project:
            lights = [e.path for e in self._project.frames_by_type(FrameType.LIGHT) if e.path.exists()]
            paths = [str(p) for p in lights]
            aligned = [e.path for e in self._project.frames_by_type(FrameType.ALIGNED) if e.path.exists()]
            if aligned:
                paths = [str(p) for p in aligned]

        dialog = BlinkDialog(frame_paths=paths or None, parent=self)
        dialog.exec()

    @pyqtSlot()
    def _on_open_subframe_selector(self):
        from astraios.ui.dialogs.subframe_dialog import SubframeDialog

        # Pre-populate with aligned frames from the project, or light frame paths.
        preloaded: list[str] = []
        if self._subframe_selected_paths:
            preloaded = list(self._subframe_selected_paths)
        elif self._aligned_paths:
            preloaded = [str(p) for p in self._aligned_paths]
        elif self._project:
            aligned = [e.path for e in self._project.frames_by_type(FrameType.ALIGNED) if e.path.exists()]
            if aligned:
                preloaded = [str(p) for p in aligned]
            else:
                lights = [e.path for e in self._project.frames_by_type(FrameType.LIGHT) if e.path.exists()]
                preloaded = [str(p) for p in lights]

        dialog = SubframeDialog(self, preloaded_paths=preloaded or None)
        dialog.accepted_frames.connect(self._on_subframe_selection_done)
        dialog.scores_ready.connect(self._on_subframe_scores_ready)
        dialog.exec()

    @pyqtSlot(list, int)
    def _on_subframe_selection_done(self, accepted_paths: list[str], n_total: int):
        """Receive accepted frame paths from SubframeDialog and feed them into stacking."""
        if not accepted_paths:
            return
        self._log_panel.log(
            f"Subframe selector: {len(accepted_paths)}/{n_total} frames accepted — ready to stack",
            "success",
        )
        self._subframe_selected_paths = accepted_paths
        self._tools_panel.set_subframe_count(len(accepted_paths), n_total)

    @pyqtSlot(list)
    def _on_subframe_scores_ready(self, scores) -> None:
        """Cache scored frame metrics in the project JSON."""
        if not self._project:
            return
        scores_dict = {
            s.file_path: {
                "fwhm": s.fwhm,
                "eccentricity": s.eccentricity,
                "snr": s.snr,
                "background": s.background,
                "n_stars": s.n_stars,
                "quality_score": s.quality_score,
            }
            for s in scores
        }
        self._project.cache_frame_scores(scores_dict)
        self._save_project()
        self._log_panel.log(f"Cached scores for {len(scores_dict)} frames in project", "info")

    @pyqtSlot()
    def _show_ez_script_dialog(self):
        from astraios.ui.dialogs.ez_script_dialog import EZScriptDialog

        def _provider():
            return self._current_image.data if self._current_image else None

        dialog = EZScriptDialog(self, image_provider=_provider)
        dialog.exec()
        result = dialog.dialog_result()
        if result is not None:
            self._update_current_image(result, "EZ Script Suite complete")

    @pyqtSlot()
    def _show_channel_match_dialog(self):
        if self._current_image is None:
            return
        data = self._current_image.data
        if data.ndim != 3 or data.shape[0] < 3:
            self._log_panel.log("Channel Match requires a colour (RGB) image.", "warning")
            return
        from astraios.core.channel_match import ChannelMatchParams, align_channels

        params = ChannelMatchParams()
        # align_channels expects channel-last (H, W, 3); internal format is (C, H, W).
        rgb_hwc = np.ascontiguousarray(np.transpose(data[:3], (1, 2, 0)))
        aligned = align_channels(rgb_hwc, params)
        result = np.transpose(aligned, (2, 0, 1)).astype(np.float32)
        self._update_current_image(result, "Channel Match complete")

    @pyqtSlot()
    def _show_lens_distortion_dialog(self):
        if self._current_image is None:
            self._log_panel.log("No image loaded", "warning")
            return
        from astraios.core.lens_distortion import LensDistortionParams, correct_distortion

        params = LensDistortionParams()
        data = self._current_image.data
        if data.ndim == 3:
            # correct_distortion expects (H, W) or (H, W, C); internal format is (C, H, W).
            hwc = np.ascontiguousarray(np.transpose(data, (1, 2, 0)))
            corrected = correct_distortion(hwc, params)
            result = np.transpose(corrected, (2, 0, 1)).astype(np.float32)
        else:
            result = correct_distortion(data, params)
        self._update_current_image(result, "Lens distortion correction applied")

    @pyqtSlot()
    def _on_run_background(self):
        if self._current_image is None:
            return
        # Convert image-space sample coords to integer (row, col) tuples
        manual_pts = [(int(round(y)), int(round(x))) for x, y in self._bg_samples]
        params = self._tools_panel.get_background_params(manual_points=manual_pts)
        n = len(manual_pts)
        msg = f"Extracting background ({n} manual sample{'s' if n != 1 else ''})..." if n else "Extracting background..."
        self._log_panel.log(msg, "info")

        def _bg_work(data, progress=None):
            return extract_background(data, params)

        self._start_worker(_bg_work, self._current_image.data, on_done=self._on_background_done)

    @pyqtSlot(object)
    def _on_background_done(self, result):
        corrected, bg_model = result
        self._update_current_image(
            corrected, "Background extraction complete", tool="background_extraction"
        )
        params = self._tools_panel.get_background_params()
        if self._project:
            self._project.add_history(
                "Background Extraction",
                {
                    "grid_size": params.grid_size,
                    "polynomial_order": params.polynomial_order,
                },
            )
        self._macro_recorder.record_step("background_extraction")

    def _on_run_bg_neutralization(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_background_neutralization_params()
        self._log_panel.log(
            f"Running background neutralization (percentile={params.percentile:.1f}%)…", "info"
        )

        def _work(data, progress=None):
            from astraios.core.background_neutralization import _noop_progress
            return background_neutralization(data, params, progress=progress or _noop_progress)

        self._start_worker(_work, self._current_image.data, on_done=self._on_bg_neutralization_done)

    @pyqtSlot(object)
    def _on_bg_neutralization_done(self, result):
        self._update_current_image(result, "Background neutralization complete")
        if self._project:
            params = self._tools_panel.get_background_neutralization_params()
            self._project.add_history(
                "Background Neutralization",
                {"percentile": params.percentile, "amount": params.amount},
            )
        self._macro_recorder.record_step("background_neutralization")


    # ---------- Phase A processing operations ----------

    @pyqtSlot()
    def _on_run_cosmetic(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_cosmetic_params()
        self._log_panel.log("Running cosmetic correction...", "info")

        def _work(data, progress=None):
            return cosmetic_correction(data, params)

        def _done(result):
            self._update_current_image(
                result.data,
                f"Cosmetic correction: {result.total_corrected} pixels fixed "
                f"({result.hot_pixels} hot, {result.cold_pixels} cold, {result.dead_pixels} dead)",
            )
            if self._project:
                self._project.add_history(
                    "Cosmetic Correction",
                    {"hot": result.hot_pixels, "cold": result.cold_pixels, "dead": result.dead_pixels},
                )

        self._start_worker(_work, self._current_image.data, on_done=_done)

    @pyqtSlot()
    def _on_run_banding(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_banding_params()
        self._log_panel.log("Running banding reduction...", "info")
        _p = params

        def _work(data, progress=None):
            return banding_reduction(data, _p)

        def _done(result):
            self._update_current_image(result, "Banding reduction complete")
            if self._project:
                self._project.add_history(
                    "Banding Reduction", {"horizontal": _p.horizontal, "vertical": _p.vertical}
                )

        self._start_worker(_work, self._current_image.data, on_done=_done)

    @pyqtSlot()
    def _on_run_histogram_transform(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_histogram_transform_params()
        self._log_panel.log("Applying histogram transform...", "info")
        _p = params

        def _ht_work(data, progress=None):
            return histogram_transform(data, _p)

        def _ht_done(result):
            self._update_current_image(result, "Histogram transform applied")
            if self._project:
                self._project.add_history(
                    "Histogram Transform",
                    {"black_point": _p.black_point, "midtone": _p.midtone, "white_point": _p.white_point},
                )
            self._macro_recorder.record_step(
                "histogram_transform",
                {"black_point": _p.black_point, "midtone": _p.midtone, "white_point": _p.white_point},
            )
            self._tools_panel.reset_histogram_transform_params()

        self._start_worker(_ht_work, self._current_image.data, on_done=_ht_done)

    @pyqtSlot()
    def _on_run_curves(self):
        if self._current_image is None:
            return
        params = CurvesParams()
        params.master = self._tools_panel.curve_editor.curve
        self._log_panel.log("Applying curves...", "info")
        _p = params

        def _curves_work(data, progress=None):
            return curves_transform(data, _p)

        def _curves_done(result):
            self._update_current_image(result, "Curves applied")
            if self._project:
                self._project.add_history("Curves", {})

        self._start_worker(_curves_work, self._current_image.data, on_done=_curves_done)

    @pyqtSlot()
    def _on_run_scnr(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_scnr_params()
        self._log_panel.log("Applying SCNR...", "info")
        _p = params

        def _scnr_work(data, progress=None):
            return scnr(data, _p)

        def _scnr_done(result):
            self._update_current_image(
                result, "SCNR applied",
                tool="scnr", tool_params={"method": _p.method, "amount": _p.amount},
            )
            if self._project:
                self._project.add_history("SCNR", {"method": _p.method.name, "amount": _p.amount})
            self._macro_recorder.record_step("scnr", {"amount": _p.amount})

        self._start_worker(_scnr_work, self._current_image.data, on_done=_scnr_done)

    @pyqtSlot()
    def _on_run_color_adjust(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_color_adjust_params()
        self._log_panel.log("Applying color adjustment...", "info")
        _p = params

        def _ca_work(data, progress=None):
            return color_adjust(data, _p)

        def _ca_done(result):
            self._update_current_image(result, "Color adjustment applied")
            if self._project:
                self._project.add_history(
                    "Color Adjustment",
                    {"saturation": _p.saturation, "hue_shift": _p.hue_shift, "vibrance": _p.vibrance},
                )

        self._start_worker(_ca_work, self._current_image.data, on_done=_ca_done)

    @pyqtSlot()
    def _on_run_deconvolution(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_deconvolution_params()
        if isinstance(params, SpatialDeconvParams):
            self._log_panel.log(
                f"Running spatial deconvolution (3x3 zones, "
                f"fallback FWHM={params.fallback_fwhm}, "
                f"{params.iterations} iterations)...",
                "info",
            )
            self._start_worker(
                richardson_lucy_spatial,
                self._current_image.data,
                params=params,
                on_done=self._on_deconvolution_done,
            )
        else:
            self._log_panel.log(
                f"Running deconvolution (FWHM={params.psf_fwhm}, "
                f"{params.iterations} iterations)...",
                "info",
            )
            self._start_worker(
                richardson_lucy,
                self._current_image.data,
                params=params,
                on_done=self._on_deconvolution_done,
            )

    @pyqtSlot(object)
    def _on_deconvolution_done(self, result):
        self._update_current_image(result, "Deconvolution complete")
        if self._project:
            self._project.add_history("Deconvolution", {})

    # ---------- Phase B processing operations ----------

    @pyqtSlot()
    def _on_run_ghs(self):
        import numpy as np
        if self._current_image is None:
            return
        params = self._tools_panel.get_ghs_params()
        data = self._current_image.data
        # Warn if image appears already stretched (GHS is designed for linear data)
        sample = data[0] if data.ndim == 3 else data
        median_val = float(np.median(sample[sample > 0])) if np.any(sample > 0) else 0.0
        if median_val > 0.1:
            self._log_panel.log(
                f"Warning: image median={median_val:.3f} — GHS works best on linear (unstretched) data",
                "warning",
            )
        self._log_panel.log(f"Applying GHS (D={params.D})...", "info")
        _p = params

        def _ghs_work(d, progress=None):
            return generalized_hyperbolic_stretch(d, _p)

        def _ghs_done(result):
            self._update_current_image(result, "GHS applied")
            if self._project:
                self._project.add_history("GHS", {"D": _p.D, "b": _p.b, "SP": _p.SP})
            self._tools_panel.reset_ghs_params()

        self._start_worker(_ghs_work, data, on_done=_ghs_done)

    @pyqtSlot()
    def _on_run_arcsinh_stretch(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_arcsinh_params()
        self._log_panel.log(f"Applying Arcsinh Stretch (β={params.stretch_factor})...", "info")
        _p = params

        def _work(d, progress=None):
            return arcsinh_stretch(d, _p)

        def _done(result):
            self._update_current_image(result, "Arcsinh Stretch applied")
            if self._project:
                self._project.add_history(
                    "Arcsinh Stretch",
                    {"stretch_factor": _p.stretch_factor, "black_point": _p.black_point},
                )

        self._start_worker(_work, self._current_image.data, on_done=_done)

    @pyqtSlot()
    def _on_run_color_calibration(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_color_calibration_params()
        self._log_panel.log("Running color calibration...", "info")
        _p = params

        def _work(data, progress=None):
            return color_calibrate(data, _p)

        def _done(result):
            factors = result.correction_factors
            self._update_current_image(
                result.data,
                f"Color calibration complete (R={factors[0]:.3f}, G={factors[1]:.3f}, B={factors[2]:.3f})",
            )
            if self._project:
                self._project.add_history("Color Calibration", {"factors": list(factors)})

        self._start_worker(_work, self._current_image.data, on_done=_done)

    @pyqtSlot()
    def _on_run_pcc(self):
        if self._current_image is None:
            return

        if self._current_image.data.ndim != 3 or self._current_image.data.shape[0] < 3:
            self._log_panel.log("PCC requires a color (RGB) image", "error")
            return

        params = self._tools_panel.get_pcc_params()
        self._log_panel.log("Starting Photometric Color Calibration (PCC)...", "info")

        # Get astrometry.net key from QSettings
        from PyQt6.QtCore import QSettings
        settings = QSettings("Astraios", "Astraios")
        api_key = settings.value("platesolver/astrometry_api_key", "")
        api_key = api_key.strip() or None

        image_data = self._current_image.data
        file_path = self._current_image.file_path
        ra_hint = params.get("ra_hint")
        dec_hint = params.get("dec_hint")
        solver = params.get("solver", "auto")

        def _pcc_work(data, progress=None):
            import tempfile
            from pathlib import Path as _Path

            from astraios.core.color_calibration import (
                ColorCalibrationParams,
                photometric_color_calibrate,
            )
            from astraios.core.star_catalog import (
                plate_solve_astap,
                plate_solve_astrometry_net,
                plate_solve_auto,
                query_gaia_dr3,
            )

            # Determine the FITS path to give to the plate solver.
            # If the image has no saved file, write a temp FITS.
            solve_path = file_path
            tmp_fits = None
            if solve_path is None or not _Path(str(solve_path)).exists():
                try:
                    from astraios.core.image_io import ImageData, save_image
                    tmp_file = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
                    tmp_fits = _Path(tmp_file.name)
                    tmp_file.close()
                    save_image(ImageData(data=data, header={}), tmp_fits)
                    solve_path = tmp_fits
                except Exception as e:
                    log.warning("Could not save temp FITS for plate solving: %s", e)
                    solve_path = None

            wcs = None
            if solve_path is not None:
                if progress:
                    progress(0.1, "Plate solving...")
                if solver == "astap":
                    wcs = plate_solve_astap(solve_path, ra_hint, dec_hint)
                elif solver == "astrometry_net":
                    if api_key:
                        wcs = plate_solve_astrometry_net(solve_path, api_key, ra_hint, dec_hint)
                    else:
                        log.warning("No astrometry.net API key set in Preferences")
                else:  # auto
                    wcs = plate_solve_auto(solve_path, api_key, ra_hint, dec_hint)
                if wcs:
                    wcs = normalise_wcs_dict(wcs)

            if tmp_fits is not None:
                tmp_fits.unlink(missing_ok=True)
                # Also clean up ASTAP output files
                for ext in (".wcs", ".ini"):
                    tmp_fits.with_suffix(ext).unlink(missing_ok=True)

            if progress:
                progress(0.5, "Querying Gaia DR3 catalog...")

            catalog_stars = []
            effective_ra = ra_hint
            effective_dec = dec_hint
            if wcs:
                effective_ra = effective_ra or wcs.get("ra")
                effective_dec = effective_dec or wcs.get("dec")

            if effective_ra is not None and effective_dec is not None:
                catalog_stars = query_gaia_dr3(effective_ra, effective_dec, radius_deg=0.5)
                log.info("Gaia DR3: %d stars retrieved", len(catalog_stars))

            if progress:
                progress(0.75, "Computing color correction...")

            if wcs is None and not catalog_stars:
                log.warning("No plate solution and no catalog — falling back to statistical calibration")
                result = __import__("astraios.core.color_calibration", fromlist=["color_calibrate"]).color_calibrate(
                    data, ColorCalibrationParams()
                )
            else:
                result = photometric_color_calibrate(
                    data,
                    catalog_stars=catalog_stars if catalog_stars else None,
                    wcs=wcs,
                )

            if progress:
                progress(1.0, "Done")
            return result, wcs, catalog_stars

        def _on_pcc_done(payload):
            result, solved_wcs, stars = payload
            factors = result.correction_factors
            self._update_current_image(
                result.data,
                f"PCC complete (R={factors[0]:.3f}, G={factors[1]:.3f}, B={factors[2]:.3f})",
            )
            if solved_wcs or stars:
                self._update_wcs_overlay(solved_wcs or {}, stars or [])
            if self._project:
                self._project.add_history(
                    "Photometric Color Calibration", {"factors": list(factors)}
                )

        self._start_worker(_pcc_work, image_data, on_done=_on_pcc_done)

    @pyqtSlot()
    def _on_run_spcc(self):
        if self._current_image is None:
            return
        if not self._wcs_overlay_stars:
            self._log_panel.log(
                "SPCC requires plate solve data — run Solve & Calibrate (PCC) first", "warning"
            )
            return
        if self._current_image.data.ndim != 3 or self._current_image.data.shape[0] != 3:
            self._log_panel.log("SPCC requires a 3-channel RGB image", "warning")
            return

        from astraios.core.spcc import spcc_calibrate
        params = self._tools_panel.get_spcc_params()
        catalog = self._wcs_overlay_stars
        catalog_with_color = [(x, y, bp_rp) for x, y, _mag, bp_rp in catalog]

        self._log_panel.log(
            f"Running SPCC ({params.filter_name}, {len(catalog_with_color)} catalog stars)…", "info"
        )

        def _spcc_work(data, progress=None):
            return spcc_calibrate(data, catalog_with_color, params=params,
                                  progress=progress or (lambda f, m: None))

        self._start_worker(_spcc_work, self._current_image.data,
                           on_done=lambda r: self._update_current_image(r, "SPCC complete"))

    @pyqtSlot()
    def _on_auto_denoise(self):
        """Measure image noise and set a recommended denoise Amount."""
        if self._current_image is None:
            self._log_panel.log("Auto denoise: no image loaded.", "warn")
            return
        from astraios.core.denoise import recommend_strength

        strength, sigma, snr = recommend_strength(self._current_image.data)
        self._tools_panel.set_denoise_amount(strength)
        self._tools_panel.set_denoise_noise_readout(sigma, snr)
        self._log_panel.log(
            f"Measured noise σ={sigma:.4f}, SNR={snr:.1f} → Amount set to {strength:.2f}",
            "info",
        )

    def _on_run_denoise(self):
        if self._current_image is None:
            return

        if self._tools_panel.is_tgv_denoise_selected():
            from astraios.core.tgv_denoise import tgv_denoise
            tgv_params = self._tools_panel.get_tgv_params()
            self._log_panel.log(
                f"Running TGV denoising (strength={tgv_params.strength:.2f}, "
                f"{tgv_params.n_iter} iters)...", "info"
            )
            self._start_worker(tgv_denoise, self._current_image.data, tgv_params,
                               on_done=lambda r: self._update_current_image(r, "TGV denoising complete"))
            return

        params = self._tools_panel.get_denoise_params()
        self._log_panel.log(f"Running noise reduction ({params.method.name})...", "info")
        _p = params

        def _work(data, progress=None):
            return denoise(data, _p)

        def _done(result):
            self._update_current_image(
                result, f"Noise reduction complete ({_p.method.name})",
                tool="denoise",
                tool_params={
                    "method": _p.method, "strength": _p.strength,
                    "detail_preservation": _p.detail_preservation,
                    "wavelet": _p.wavelet, "wavelet_levels": _p.wavelet_levels,
                },
            )
            if self._project:
                self._project.add_history(
                    "Denoise", {"method": _p.method.name, "strength": _p.strength}
                )

        self._start_worker(_work, self._current_image.data, on_done=_done)

    def _on_background_grain(self):
        if self._current_image is None:
            self._log_panel.log("No image loaded", "warning")
            return
        from astraios.core.luma_denoise import LumaDenoiseParams, denoise_background_luma

        strength = self._tools_panel._bg_grain_strength.value()
        self._log_panel.log(
            f"Reducing background grain (strength={strength:.2f})...", "info"
        )

        def _work(data, progress=None):
            return denoise_background_luma(data, LumaDenoiseParams(strength=strength))

        def _done(result):
            self._update_current_image(result, "Background grain reduced")
            if self._project:
                self._project.add_history("Background Grain", {"strength": strength})

        self._start_worker(_work, self._current_image.data, on_done=_done)

    @pyqtSlot()
    def _on_run_frequency_separation(self):
        if self._current_image is None:
            return
        from astraios.core.frequency_separation import frequency_separation

        params = self._tools_panel.get_frequency_separation_params()
        self._log_panel.log(
            f"Frequency separation (σ={params.sigma:.0f}, detail×{params.hf_boost:.2f}, "
            f"smooth={params.lf_smooth:.0f}, {params.method.name.lower()})...",
            "info",
        )
        _p = params

        def _work(data, progress=None):
            return frequency_separation(data, _p, progress=progress)

        def _done(result):
            self._update_current_image(result, "Frequency separation complete")
            step_params = {
                "sigma": _p.sigma, "method": _p.method.name,
                "hf_boost": _p.hf_boost, "lf_smooth": _p.lf_smooth,
            }
            if self._project:
                self._project.add_history("FrequencySeparation", step_params)
            self._macro_recorder.record_step("frequency_separation", step_params)

        self._start_worker(_work, self._current_image.data, on_done=_done)

    @pyqtSlot()
    def _on_run_statistical_stretch(self):
        if self._current_image is None:
            return
        from astraios.core.stretch import statistical_stretch

        params = self._tools_panel.get_statistical_stretch_params()
        self._log_panel.log(
            f"Statistical stretch (target median={params.target_median:.2f}, "
            f"{'linked' if params.linked else 'per-channel'})...",
            "info",
        )
        _p = params

        def _work(data, progress=None):
            return statistical_stretch(data, _p)

        def _done(result):
            self._update_current_image(result, "Statistical stretch complete")
            step_params = {
                "target_median": _p.target_median,
                "shadow_clip": _p.shadow_clip,
                "linked": _p.linked,
            }
            if self._project:
                self._project.add_history("StatisticalStretch", step_params)
            self._macro_recorder.record_step("statistical_stretch", step_params)

        self._start_worker(_work, self._current_image.data, on_done=_done)

    @pyqtSlot()
    def _on_run_star_stretch(self):
        if self._current_image is None:
            return
        from astraios.core.star_stretch import star_stretch

        params = self._tools_panel.get_star_stretch_params()
        self._log_panel.log(
            f"Star stretch (amount={params.amount:.2f}, colour×{params.color_boost:.2f})...",
            "info",
        )
        _p = params

        def _work(data, progress=None):
            return star_stretch(data, _p, progress=progress)

        def _done(result):
            self._update_current_image(result, "Star stretch complete")
            step_params = {"amount": _p.amount, "color_boost": _p.color_boost}
            if self._project:
                self._project.add_history("StarStretch", step_params)
            self._macro_recorder.record_step("star_stretch", step_params)

        self._start_worker(_work, self._current_image.data, on_done=_done)

    @pyqtSlot()
    def _on_run_star_reduction(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_star_reduction_params()
        self._log_panel.log("Running star reduction...", "info")
        _p = params

        def _work(data, progress=None):
            return reduce_stars(data, params=_p)

        def _done(result):
            self._update_current_image(result, "Star reduction complete")
            if self._project:
                self._project.add_history(
                    "Star Reduction", {"amount": _p.amount, "iterations": _p.iterations}
                )

        self._start_worker(_work, self._current_image.data, on_done=_done)

    @pyqtSlot()
    def _on_run_split_channels(self):
        if self._current_image is None:
            return
        data = self._current_image.data
        if data.ndim != 3:
            self._log_panel.log("Split channels requires a color image", "warning")
            return
        channels = split_channels(data)
        # Display the first channel (R), log that others are available
        self._update_current_image(
            channels[0], f"Channel split: showing Red ({len(channels)} channels)"
        )
        if self._project:
            self._project.add_history("Split Channels", {"n_channels": len(channels)})

    @pyqtSlot()
    def _on_run_extract_luminance(self):
        if self._current_image is None:
            return
        lum = extract_luminance(self._current_image.data)
        self._update_current_image(lum, "Luminance extracted")
        if self._project:
            self._project.add_history("Extract Luminance", {})

    # ---------- Phase C processing operations ----------

    @pyqtSlot()
    def _on_run_wavelet_sharpen(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_wavelet_params()
        self._log_panel.log(f"Running wavelet sharpening ({params.n_scales} scales)...", "info")
        _p = params

        def _work(data, progress=None):
            return wavelet_sharpen(data, _p)

        def _done(result):
            self._update_current_image(result, "Wavelet sharpening complete")
            if self._project:
                self._project.add_history(
                    "Wavelet Sharpen", {"n_scales": _p.n_scales, "scale_weights": _p.scale_weights}
                )
            self._macro_recorder.record_step(
                "wavelet_sharpen", {"n_scales": _p.n_scales, "scale_weights": _p.scale_weights}
            )

        self._start_worker(_work, self._current_image.data, on_done=_done)

    @pyqtSlot()
    def _on_run_mlt(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_mlt_params()
        n_thresh = sum(1 for t in params.noise_thresholds if t > 0)
        self._log_panel.log(
            f"Running MLT ({params.n_scales} scales, {n_thresh} denoise bands)…", "info"
        )
        _p = params

        def _work(data, progress=None):
            return wavelet_sharpen(data, _p)

        def _done(result):
            self._update_current_image(result, "MLT complete")
            if self._project:
                self._project.add_history("MLT", {"n_scales": _p.n_scales})

        self._start_worker(_work, self._current_image.data, on_done=_done)

    @pyqtSlot()
    def _on_run_lrgb_combine(self):
        if self._current_image is None:
            return
        from PyQt6.QtWidgets import QFileDialog

        from astraios.core.image_io import load_image
        from astraios.core.lrgb import lrgb_combine

        # Validate current image is mono (luminance)
        data = self._current_image.data
        if data.ndim == 3 and data.shape[0] == 3:
            self._log_panel.log(
                "LRGB Combine: current image should be the Luminance (mono) image. "
                "Load your L image first.", "warning"
            )

        rgb_path, _ = QFileDialog.getOpenFileName(
            self, "Select RGB Color Image", "",
            "Images (*.fits *.fit *.fts *.xisf *.tif *.tiff *.png *.jpg)"
        )
        if not rgb_path:
            return

        params = self._tools_panel.get_lrgb_params()
        lum_data = data
        self._log_panel.log(
            f"LRGB Combine: L weight={params.luminance_weight}, "
            f"sat boost={params.saturation_boost}…", "info"
        )

        def _lrgb_work(lum, rgb_path_str, p, progress=None):
            import numpy as _np
            rgb_img = load_image(rgb_path_str)
            rgb = rgb_img.data
            if rgb.ndim == 2:
                rgb = _np.stack([rgb, rgb, rgb], axis=0)
            return lrgb_combine(lum, rgb, params=p,
                                progress=progress or (lambda f, m: None))

        def _on_lrgb_done(result):
            self._update_current_image(result, "LRGB combine complete")
            if self._project:
                self._project.add_history("LRGB Combine", {
                    "lum_weight": params.luminance_weight,
                    "sat_boost": params.saturation_boost,
                })

        self._start_worker(_lrgb_work, lum_data, rgb_path, params, on_done=_on_lrgb_done)

    @pyqtSlot()
    def _on_open_channel_combine(self):
        from astraios.ui.dialogs.channel_combine_dialog import ChannelCombineDialog

        dlg = ChannelCombineDialog(
            current_image=self._current_image,
            parent=self,
        )
        if dlg.exec() and dlg.result_data() is not None:
            rgb = dlg.result_data()
            self._update_current_image(rgb, "Channel combine complete")
            if self._project:
                self._project.add_history("Channel Combine", {"palette": dlg._palette_combo.currentText()})
            self._log_panel.log(
                f"Combined channels: palette={dlg._palette_combo.currentText()}, "
                f"shape={rgb.shape}", "info"
            )
        # The dialog stays alive as a child of this window after exec(); free its
        # full-res channel cache and schedule its deletion so repeated opens don't
        # accumulate hundreds of MB each.
        dlg.cleanup()
        dlg.deleteLater()

    @pyqtSlot()
    def _on_run_debayer(self):
        if self._current_image is None:
            return
        from astraios.core.debayer import debayer as _debayer
        from astraios.core.debayer import detect_bayer_pattern

        data = self._current_image.data
        if data.ndim == 3:
            self._log_panel.log(
                "Image is already color (3 channels) — debayer not needed.", "warning"
            )
            return

        p = self._tools_panel.get_debayer_params()
        pattern = p["pattern"]
        method = p["method"]

        # Auto-detect from header if user left on Auto-detect
        if not pattern:
            pattern = detect_bayer_pattern(self._current_image.header)
            if not pattern:
                self._log_panel.log(
                    "No BAYERPAT in FITS header and no pattern selected. "
                    "Select RGGB / BGGR / GRBG / GBRG manually.", "error"
                )
                return

        self._log_panel.log(f"Debayering: pattern={pattern} method={method}…", "info")

        def _work(d, pat, meth, progress=None):
            return _debayer(d, pattern=pat, method=meth)

        def _done(result):
            self._update_current_image(result, f"Debayer ({pattern}) complete")
            if self._project:
                self._project.add_history("Debayer", {"pattern": pattern, "method": method})
            self._log_panel.log(
                f"Debayer complete: {pattern} → shape {result.shape}", "success"
            )

        self._start_worker(_work, data, pattern, method, on_done=_done)

    @pyqtSlot()
    def _on_run_local_contrast(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_local_contrast_params()
        self._log_panel.log("Running local contrast enhancement...", "info")
        _p = params

        def _work(data, progress=None):
            return local_contrast_enhance(data, _p)

        def _done(result):
            self._update_current_image(result, "Local contrast enhancement complete")
            if self._project:
                self._project.add_history(
                    "Local Contrast", {"clip_limit": _p.clip_limit, "amount": _p.amount}
                )
            self._macro_recorder.record_step(
                "local_contrast",
                {"clip_limit": _p.clip_limit, "tile_size": _p.tile_size, "amount": _p.amount},
            )

        self._start_worker(_work, self._current_image.data, on_done=_done)

    @pyqtSlot()
    def _on_run_morphology(self):
        if self._current_image is None:
            return
        params = self._tools_panel.get_morphology_params()
        self._log_panel.log(f"Running morphology ({params.operation.name})...", "info")
        _p = params

        def _work(data, progress=None):
            return morphology_transform(data, _p)

        def _done(result):
            self._update_current_image(result, f"Morphology {_p.operation.name} complete")
            if self._project:
                self._project.add_history(
                    "Morphology", {"operation": _p.operation.name, "kernel_size": _p.kernel_size}
                )
            self._macro_recorder.record_step(
                "morphology",
                {"operation": _p.operation.name, "kernel_size": _p.kernel_size, "iterations": _p.iterations},
            )

        self._start_worker(_work, self._current_image.data, on_done=_done)

    def _show_hdr_dialog(self):
        from astraios.ui.dialogs.hdr_dialog import HDRDialog

        dialog = HDRDialog(self)
        dialog.result_ready.connect(self._on_hdr_result)
        dialog.exec()

    def _on_hdr_result(self, data):
        self._update_current_image(data, "HDR composition complete")
        if self._project:
            self._project.add_history("HDR Composition", {})

    # ---------- Phase D processing operations ----------

    @pyqtSlot()
    def _on_run_ai_denoise(self):
        if self._current_image is None:
            self._log_panel.log("No image loaded", "warning")
            return

        backend = self._tools_panel.get_ai_denoise_backend()
        if "CosmicClarity" in backend:
            params = self._tools_panel.get_cosmic_clarity_denoise_params()
            self._log_panel.log("Running AI Denoise (CosmicClarity)...", "info")
            from astraios.ai.inference.cosmic_clarity import apply as cc_apply
            self._start_worker(
                cc_apply,
                self._current_image.data,
                params=params,
                on_done=self._on_ai_denoise_done,
            )
        else:
            params = self._tools_panel.get_ai_denoise_params()
            self._log_panel.log("Running AI Denoise (Noise2Self)...", "info")
            from astraios.ai.inference.denoise import ai_denoise
            self._start_worker(
                ai_denoise,
                self._current_image.data,
                params=params,
                on_done=self._on_ai_denoise_done,
            )

    @pyqtSlot(object)
    def _on_ai_denoise_done(self, result):
        self._update_current_image(result, "AI Denoise complete")
        if self._project:
            self._project.add_history("AI Denoise", {})
        self._macro_recorder.record_step("ai_denoise")

    @pyqtSlot()
    def _on_run_ai_sharpen(self):
        if self._current_image is None:
            self._log_panel.log("No image loaded", "warning")
            return

        backend = self._tools_panel.get_ai_sharpen_backend()
        if "CosmicClarity" in backend:
            params = self._tools_panel.get_cosmic_clarity_sharpen_params()
            self._log_panel.log("Running AI Sharpen (CosmicClarity)...", "info")
            from astraios.ai.inference.cosmic_clarity import apply as cc_apply
            self._start_worker(
                cc_apply,
                self._current_image.data,
                params=params,
                on_done=self._on_ai_sharpen_done,
            )
        else:
            params = self._tools_panel.get_ai_sharpen_params()
            self._log_panel.log("Running AI Sharpen (Richardson-Lucy)...", "info")
            from astraios.ai.inference.sharpen import ai_sharpen
            self._start_worker(
                ai_sharpen,
                self._current_image.data,
                params=params,
                on_done=self._on_ai_sharpen_done,
            )

    @pyqtSlot(object)
    def _on_ai_sharpen_done(self, result):
        self._update_current_image(result, "AI Sharpen complete")
        if self._project:
            self._project.add_history("AI Sharpen", {})
        self._macro_recorder.record_step("ai_sharpen")

    @pyqtSlot()
    def _on_run_superbias(self):
        self._log_panel.log("Creating SuperBias from loaded bias frames...", "info")
        self._log_panel.log("Load bias frames via Project panel first.", "info")

    def _show_processing_graph(self):
        if self._current_image is None:
            self._log_panel.log("Load an image first", "warning")
            return
        from astraios.core.processing_graph import ProcessingGraph
        from astraios.ui.dialogs.processing_graph_dialog import ProcessingGraphDialog

        if not hasattr(self, "_processing_graph") or self._processing_graph is None:
            self._processing_graph = ProcessingGraph()
            self._processing_graph.set_base(self._current_image.data)

        dialog = ProcessingGraphDialog(self, self._processing_graph)
        dialog.view_stage.connect(self._on_history_view_stage)
        dialog.history_changed.connect(self._on_history_changed)
        dialog.export_macro.connect(self._on_history_export_macro)
        dialog.exec()

    def _on_history_view_stage(self, index: int):
        """Preview the image at history step *index* (-1 = base) on the canvas.

        Pure preview: it does not alter the working image or the undo stack.
        """
        if getattr(self, "_processing_graph", None) is None or self._current_image is None:
            return
        self._skip_graph_auto_add = True
        try:
            img = self._processing_graph.evaluate(
                up_to=index, process_fn=self._process_graph_step
            )
        finally:
            self._skip_graph_auto_add = False
        if img is None:
            self._log_panel.log(
                "Cannot reconstruct this stage (an earlier step is not replayable yet)",
                "warning",
            )
            return
        preview = ImageData(
            data=img,
            header=self._current_image.header.copy(),
            frame_type=self._current_image.frame_type,
        )
        self._display_image(preview)

    def _on_history_changed(self):
        """Recompute the final result after the history was edited, and commit it."""
        if getattr(self, "_processing_graph", None) is None:
            return
        self._skip_graph_auto_add = True
        try:
            result = self._processing_graph.evaluate(process_fn=self._process_graph_step)
            if result is not None:
                self._update_current_image(result, "Processing history updated")
        finally:
            self._skip_graph_auto_add = False

    def _on_history_export_macro(self):
        """Export the replayable history steps as a reusable macro."""
        if getattr(self, "_processing_graph", None) is None:
            return
        pipeline = self._processing_graph.to_pipeline("History Macro")
        if not pipeline.steps:
            self._log_panel.log(
                "No replayable steps to export yet (display-only history).", "warning"
            )
            return
        from pathlib import Path

        from PyQt6.QtWidgets import QFileDialog

        from astraios.core.scripting import save_macro

        path, _ = QFileDialog.getSaveFileName(
            self, "Export History as Macro", "history_macro.json", "Macro (*.json)"
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        save_macro(pipeline, Path(path))
        self._log_panel.log(
            f"Exported {len(pipeline.steps)} step(s) to {Path(path).name}", "success"
        )

    def _process_graph_step(self, process_name: str, params: dict, image):
        """Execute a single processing graph step.

        Delegates to the shared batch/macro tool registry so the
        processing graph, batch pipelines and recorded macros all run through
        one execution path. Falls back to the legacy hardcoded names (and
        finally a no-op) for description-style nodes that predate the registry.
        """
        from astraios.core.batch import get_registered_tools

        tools = get_registered_tools()
        func = tools.get(process_name)
        if func is not None:
            return func(image, **(params or {}))

        # Legacy fallbacks for older graphs / description-named nodes.
        from astraios.core.background import extract_background
        from astraios.core.denoise import DenoiseParams, denoise
        from astraios.core.filters import unsharp_mask
        from astraios.core.stretch import arcsinh_stretch

        if process_name == "background":
            bg = extract_background(image)
            return image - bg[1]
        elif process_name == "denoise":
            return denoise(image, DenoiseParams(**params) if params else None)
        elif process_name == "stretch":
            return arcsinh_stretch(image, None)
        elif process_name == "unsharp_mask":
            return unsharp_mask(image, None)
        return image

    def _on_analysis_fwhm(self):
        if self._current_image is None:
            return
        data = self._current_image.data

        def _work(img, progress=None):
            from astraios.core.analysis.fwhm_map import compute_fwhm_map
            return compute_fwhm_map(img)

        def _done(result):
            self._log_panel.log(
                f"FWHM map: mean={result.mean_fwhm:.2f}px, std={result.std_fwhm:.2f}px, "
                f"tilt={'YES' if result.tilt_detected else 'no'} ({result.tilt_angle:.0f}°)",
                "info",
            )

        self._start_worker(_work, data, on_done=_done)

    def _on_analysis_tilt(self):
        if self._current_image is None:
            return
        data = self._current_image.data

        def _work(img, progress=None):
            from astraios.core.analysis.tilt_analysis import analyze_tilt
            return analyze_tilt(img)

        def _done(result):
            self._log_panel.log(
                f"Tilt analysis: coma={result.coma_detected}, "
                f"astigmatism={result.astigmatism_detected}, "
                f"tilt={result.tilt_detected}",
                "info",
            )
            self._log_panel.log(result.summary, "info")

        self._start_worker(_work, data, on_done=_done)

    def _on_analysis_photometry(self):
        if self._current_image is None:
            return
        data = self._current_image.data

        def _work(img, progress=None):
            from astraios.core.analysis.aperture_photometry import run_photometry
            return run_photometry(img)

        def _done(result):
            self._log_panel.log(
                f"Photometry: {result.n_sources} sources detected, "
                f"mean flux={float(np.mean(result.flux)) if result.n_sources else 0:.1f}",
                "info",
            )

        self._start_worker(_work, data, on_done=_done)

    def _on_run_ai_super_resolution(self):
        if self._current_image is None:
            return
        scale_text = self._tools_panel._sr_scale.currentText()
        scale = int(scale_text.replace("×", ""))
        tile_text = self._tools_panel._sr_tile.currentText()
        tile = 0 if tile_text == "Full" else int(tile_text)

        self._log_panel.log(f"Upscaling image {scale}× using AI super-resolution...", "info")

        from astraios.ai.inference.super_resolution import SuperResParams, upscale

        params = SuperResParams(scale=scale, tile_size=tile)

        self._start_worker(
            lambda data, progress=None: upscale(data, params),
            self._current_image.data,
            on_done=lambda result: self._update_current_image(
                result, f"Super-resolution {scale}× complete"
            ),
        )

    def _on_run_starnet(self):
        if self._current_image is None:
            return
        backend = self._tools_panel._star_removal_path.currentText()
        threshold = self._tools_panel._star_threshold.value()

        if "StarNet" in backend and backend != "Built-in (starrem2k13)":
            self._log_panel.log("Running StarNet star removal...", "info")
            from astraios.ai.inference.starnet import find_starnet_binary, run_starnet

            starnet_path = find_starnet_binary()
            if starnet_path is None and "Auto" in backend:
                self._log_panel.log("StarNet binary not found, trying built-in...", "info")
                self._run_starrem_builtin(threshold)
                return
            elif starnet_path is None:
                self._log_panel.log(
                    "StarNet binary not found. Download from starnetastro.com "
                    "or switch to Built-in backend.", "warning"
                )
                self._prompt_starnet_missing()
                return

            self._start_worker(
                lambda data, progress=None: run_starnet(data, starnet_path=starnet_path),
                self._current_image.data,
                on_done=self._on_starnet_done,
            )
        else:
            self._run_starrem_builtin(threshold)

    def _prompt_starnet_missing(self):
        """Tell the user StarNet isn't installed and offer to set its path."""
        from PyQt6.QtWidgets import QMessageBox

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("StarNet not found")
        box.setText("StarNet v2 was not found on this system.")
        box.setInformativeText(
            "Download the command-line version (StarNetv2CLI) from "
            "starnetastro.com, then point Astraios at the executable under "
            "Preferences > AI Models. You can also switch the Star Removal "
            "backend to Built-in to remove stars without StarNet."
        )
        open_prefs = box.addButton("Open Preferences…", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is open_prefs:
            self._show_preferences()

    def _run_starrem_builtin(self, threshold=0.5):
        self._log_panel.log("Running built-in star removal...", "info")
        from astraios.ai.inference.star_removal import remove_stars_builtin

        self._start_worker(
            lambda data, progress=None: remove_stars_builtin(data, threshold),
            self._current_image.data,
            on_done=self._on_starnet_builtin_done,
        )

    @pyqtSlot(object)
    def _on_starnet_done(self, result):
        if not result.success:
            self._log_panel.log(f"StarNet failed: {result.message}", "error")
            return
        self._update_current_image(result.starless, "StarNet complete: stars removed")
        if result.stars_only is not None:
            self._extracted_stars = result.stars_only
            self._log_panel.log(
                "Extracted star layer kept — stretch the starless image, then "
                "Tools → Image Blend (Screen) to add the stars back.",
                "info",
            )
        if self._project:
            self._project.add_history("StarNet", {})
        self._macro_recorder.record_step("starnet")

    @pyqtSlot(object)
    def _on_starnet_builtin_done(self, result):
        self._update_current_image(result, "Star removal complete (built-in)")
        if self._project:
            self._project.add_history("star_removal_builtin", {})

    def _show_batch_preprocess_dialog(self):
        from astraios.ui.dialogs.batch_preprocess_dialog import BatchPreprocessDialog

        dialog = BatchPreprocessDialog(self)
        dialog.exec()

    def _show_batch_dialog(self):
        from astraios.ui.dialogs.batch_dialog import BatchDialog

        dialog = BatchDialog(self)
        dialog.exec()

    # ---------- Macro operations ----------

    @pyqtSlot()
    def _on_start_macro(self):
        self._macro_recorder.start("User Macro")
        self._tools_panel.set_macro_recording(True)
        self._log_panel.log("Macro recording started", "info")

    @pyqtSlot()
    def _on_stop_macro(self):
        self._current_macro = self._macro_recorder.stop()
        self._tools_panel.set_macro_recording(False)
        n = len(self._current_macro.steps)
        self._log_panel.log(f"Macro recording stopped: {n} steps captured", "success")

    @pyqtSlot()
    def _on_play_macro(self):
        if self._current_image is None:
            self._log_panel.log("Load an image first", "warning")
            return
        if self._current_macro is None or len(self._current_macro.steps) == 0:
            self._log_panel.log("No macro recorded or loaded", "warning")
            return
        macro = self._current_macro
        self._log_panel.log(
            f"Playing macro: {macro.name} ({len(macro.steps)} steps)...",
            "info",
        )

        def _play_work(data, progress=None):
            return play_macro(data, macro, progress=progress)

        def _play_done(result):
            self._update_current_image(result, "Macro playback complete")
            if self._project:
                self._project.add_history("Play Macro", {"name": macro.name})

        self._start_worker(_play_work, self._current_image.data, on_done=_play_done)

    @pyqtSlot()
    def _on_save_macro(self):
        if self._current_macro is None:
            self._log_panel.log("No macro to save", "warning")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Macro", "", "Astraios Macro (*.json)")
        if path:
            save_macro(self._current_macro, Path(path))
            self._log_panel.log(f"Macro saved: {path}", "success")

    @pyqtSlot()
    def _on_load_macro(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Macro", "", "Astraios Macro (*.json)")
        if path:
            self._current_macro = load_macro(Path(path))
            self._log_panel.log(
                f"Macro loaded: {self._current_macro.name} ({len(self._current_macro.steps)} steps)",
                "success",
            )

    # ---------- Smart Processor ----------

    def _show_equipment_dialog(self):
        from astraios.ui.dialogs.equipment_dialog import EquipmentDialog

        dlg = EquipmentDialog(self, self._equipment_profile)
        dlg.profile_ready.connect(self._on_equipment_set)
        dlg.exec()

    def _on_equipment_set(self, profile: EquipmentProfile):
        self._equipment_profile = profile
        self._log_panel.log(
            f"Equipment set: {profile.camera.name} + {profile.telescope.name} "
            f"({profile.plate_scale():.2f} arcsec/px)",
            "success",
        )

    def _show_smart_processor_dialog(self):
        from astraios.ui.dialogs.smart_process_dialog import SmartProcessDialog

        if self._current_image is None:
            self._log_panel.log("Load an image first", "warning")
            return

        dlg = SmartProcessDialog(self, equipment=self._equipment_profile)
        dlg.set_image_data(
            self._current_image.data,
            fits_header=getattr(self._current_image, "header", None),
            wcs=getattr(self, "_current_wcs", None),
        )
        dlg.result_ready.connect(self._on_smart_processor_result)
        dlg.exec()

    def _on_smart_processor_result(self, result):
        n_checks = len(result.quality_checks)
        n_passed = sum(1 for q in result.quality_checks if q.passed)
        target_info = ""
        if result.analysis.primary_target:
            target_info = f" Target: {result.analysis.primary_target.id}."

        self._update_current_image(
            result.image,
            f"Smart Processor complete.{target_info} Quality: {n_passed}/{n_checks} checks passed.",
        )
        if self._project:
            self._project.add_history(
                "Smart Processor",
                {
                    "quality_checks": f"{n_passed}/{n_checks}",
                },
            )

    # ── Blink Comparator ──────────────────────────────────────────────────────

    def _blink_load_from_file(self, slot: int):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Load Blink Image {'A' if slot == 0 else 'B'}", "",
            "Images (*.fits *.fit *.fts *.xisf *.tif *.tiff *.png *.jpg)"
        )
        if not path:
            return
        img = load_image(path)
        self._blink_images[slot] = self._make_display_rgb(img.data)
        name = Path(path).name
        self._blink_names[slot] = name
        self._tools_panel.set_blink_slot_label("a" if slot == 0 else "b", name)
        self._log_panel.log(f"Blink {'A' if slot == 0 else 'B'}: loaded {name}", "info")

    def _blink_use_current(self, slot: int):
        if self._current_image is None:
            self._log_panel.log("No image loaded", "warning")
            return
        self._blink_images[slot] = self._make_display_rgb(self._current_image.data)
        name = getattr(self._current_image, "path", None)
        name = Path(name).name if name else "current image"
        self._blink_names[slot] = name
        self._tools_panel.set_blink_slot_label("a" if slot == 0 else "b", name)
        self._log_panel.log(f"Blink {'A' if slot == 0 else 'B'}: set to {name}", "info")

    def _make_display_rgb(self, data) -> "np.ndarray":
        """Convert image data to display-ready uint8 RGB (H,W,3) array."""
        import numpy as _np

        from astraios.core.stretch import StretchParams, auto_stretch
        stretched = auto_stretch(data, StretchParams())
        if stretched.ndim == 2:
            rgb = _np.stack([stretched] * 3, axis=-1)
        else:
            # (3, H, W) → (H, W, 3)
            rgb = _np.transpose(stretched, (1, 2, 0))
        return (_np.clip(rgb, 0, 1) * 255).astype(_np.uint8)

    @pyqtSlot(bool)
    def _on_blink_toggle(self, enabled: bool):
        if enabled:
            if self._blink_images[0] is None or self._blink_images[1] is None:
                self._log_panel.log(
                    "Blink Comparator: load both Image A and Image B first", "warning"
                )
                self._tools_panel.reset_blink_toggle()
                return
            fps = max(1, self._tools_panel._blink_fps_spin.value())
            self._blink_index = 0
            self._blink_timer.start(1000 // fps)
            self._log_panel.log(f"Blink Comparator started ({fps} fps)", "info")
        else:
            self._blink_timer.stop()
            # Restore the original current image on canvas
            if self._current_image is not None:
                self._display_image(self._current_image)
            self._log_panel.log("Blink Comparator stopped", "info")

    @pyqtSlot(int)
    def _on_blink_fps_changed(self, fps: int):
        if self._blink_timer.isActive():
            self._blink_timer.setInterval(1000 // fps)

    def _blink_tick(self):
        img = self._blink_images[self._blink_index]
        if img is not None:
            self._canvas.set_image(img)
            slot_name = "A" if self._blink_index == 0 else "B"
            self.statusBar().showMessage(
                f"Blink: {slot_name} — {self._blink_names[self._blink_index]}"
            )
        self._blink_index = 1 - self._blink_index

    def keyPressEvent(self, event):
        """Global keyboard shortcuts."""
        from PyQt6.QtCore import Qt as _Qt
        if (event.key() == _Qt.Key.Key_B
                and not event.isAutoRepeat()
                and event.modifiers() == _Qt.KeyboardModifier.NoModifier):
            btn = self._tools_panel._blink_toggle_btn
            btn.setChecked(not btn.isChecked())
            return
        super().keyPressEvent(event)
