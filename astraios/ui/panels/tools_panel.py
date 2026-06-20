"""tools_panel.py — Astraios Tools Panel (PyQt6 redesign).

Drop-in replacement for the existing astraios/ui/panels/tools_panel.py.
All signals and getter/setter methods are identical to the original.
Visual style matches the HTML prototype exactly using ui_kit widgets.
"""
from __future__ import annotations

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from astraios.ai.inference.denoise import AIDenoiseParams
from astraios.core.abe import ABEParams
from astraios.core.background import BackgroundParams
from astraios.core.background_neutralization import BackgroundNeutralizationParams
from astraios.core.banding import BandingParams
from astraios.core.color_calibration import ColorCalibrationParams
from astraios.core.color_tools import ColorAdjustParams, SCNRParams
from astraios.core.cosmetic import CosmeticParams
from astraios.core.curves import CurvesParams
from astraios.core.deconvolution import DeconvolutionParams, SpatialDeconvParams
from astraios.core.denoise import DenoiseParams
from astraios.core.filters import UnsharpMaskParams
from astraios.core.histogram_transform import HistogramTransformParams
from astraios.core.local_contrast import LocalContrastParams
from astraios.core.morphology import MorphologyParams
from astraios.core.stacking import (
    IntegrationMethod,
    NormalizationMethod,
    RegistrationMode,
    RejectionMethod,
    StackingParams,
)
from astraios.core.star_reduction import StarReductionParams
from astraios.core.stretch import ArcsinhStretchParams, GHSParams, StretchParams
from astraios.core.transforms import (
    BinParams,
    CropParams,
    FlipParams,
    ResizeParams,
    RotateParams,
)
from astraios.core.vignette import VignetteParams
from astraios.core.wavelets import WaveletParams
from astraios.ui.widgets.curves_widget import CurveEditor
from astraios.ui.widgets.ui_kit import (
    ACCENT, ACCENT_DARK, ACCENT_HOVER, ACCENT_PURPLE, BG_HOVER, BG_PRIMARY, BG_SECONDARY,
    BG_TERTIARY, BLUE, BORDER, FONT_MONO, ORANGE, RED,
    TEXT_PRIMARY, TEXT_SECONDARY,
    CollapsibleSection, InfoLabel, RunBtn, SliderRow,
    divider, field_row, make_label, scrollable_tab,
    styled_check, styled_combo, styled_spin,
)

# Tab-bar stylesheet
_TAB_SS = f"""
QTabWidget::pane {{
    border: none; background: {BG_PRIMARY};
}}
QTabBar {{
    background: {BG_PRIMARY};
}}
QTabBar::tab {{
    background: {BG_PRIMARY}; color: {TEXT_SECONDARY};
    padding: 7px 10px; font-size: 10px; font-weight: 600;
    border: none; border-bottom: 2px solid transparent;
    min-width: 0;
}}
QTabBar::tab:selected {{
    color: {ACCENT}; border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover:!selected {{
    color: {TEXT_PRIMARY};
}}
QTabBar::scroller {{
    width: 52px;
}}
QTabBar QToolButton {{
    background-color: {BG_TERTIARY}; border: 1px solid {BORDER};
    border-radius: 4px; color: {TEXT_PRIMARY};
    width: 22px; height: 22px;
    padding: 0px; margin: 1px;
    font-size: 12px;
}}
QTabBar QToolButton:hover {{
    background-color: {BG_HOVER}; color: #ffffff;
}}
"""

_BOTTOM_SS = f"""
QWidget#tools_bottom {{
    background: {BG_SECONDARY};
    border-top: 1px solid {BORDER};
}}
"""


class ToolsPanel(QWidget):
    """Right-side tabbed processing controls."""

    # ── Signals (identical to original) ──────────────────
    run_calibration          = pyqtSignal()
    run_stacking             = pyqtSignal()
    run_alignment            = pyqtSignal()
    run_stretch              = pyqtSignal()
    run_statistical_stretch  = pyqtSignal()
    run_background           = pyqtSignal()
    stretch_params_changed   = pyqtSignal()
    run_cosmetic             = pyqtSignal()
    run_banding              = pyqtSignal()
    run_histogram_transform  = pyqtSignal()
    run_curves               = pyqtSignal()
    run_scnr                 = pyqtSignal()
    run_color_adjust         = pyqtSignal()
    run_deconvolution        = pyqtSignal()
    run_ghs                  = pyqtSignal()
    run_arcsinh_stretch      = pyqtSignal()
    run_star_stretch         = pyqtSignal()
    run_color_calibration    = pyqtSignal()
    run_pcc                  = pyqtSignal()
    run_denoise              = pyqtSignal()
    request_auto_denoise     = pyqtSignal()
    run_frequency_separation = pyqtSignal()
    run_star_reduction       = pyqtSignal()
    open_narrowband_dialog   = pyqtSignal()
    open_pixelmath_dialog    = pyqtSignal()
    run_split_channels       = pyqtSignal()
    run_extract_luminance    = pyqtSignal()
    run_wavelet_sharpen      = pyqtSignal()
    run_local_contrast       = pyqtSignal()
    run_morphology           = pyqtSignal()
    open_hdr_dialog          = pyqtSignal()
    run_ai_denoise           = pyqtSignal()
    run_ai_sharpen           = pyqtSignal()
    run_starnet              = pyqtSignal()
    open_batch_preprocess    = pyqtSignal()
    open_batch_dialog        = pyqtSignal()
    start_macro_recording    = pyqtSignal()
    stop_macro_recording     = pyqtSignal()
    play_macro               = pyqtSignal()
    save_macro               = pyqtSignal()
    load_macro               = pyqtSignal()
    run_unsharp_mask         = pyqtSignal()
    run_median_filter        = pyqtSignal()
    run_abe                  = pyqtSignal()
    run_vignette_correction  = pyqtSignal()
    run_chromatic_aberration = pyqtSignal()
    show_image_statistics    = pyqtSignal()
    edit_fits_header         = pyqtSignal()
    curves_histogram_changed = pyqtSignal()
    measure_psf              = pyqtSignal()
    run_continuum_subtraction= pyqtSignal()
    toggle_sample_mode       = pyqtSignal(bool)
    clear_bg_samples         = pyqtSignal()
    add_bg_grid              = pyqtSignal(int, int, int)
    toggle_wcs_overlay       = pyqtSignal(bool)
    run_background_neutralization = pyqtSignal()
    open_python_console      = pyqtSignal()
    run_mlt                  = pyqtSignal()
    run_lrgb_combine         = pyqtSignal()
    run_spcc                 = pyqtSignal()
    toggle_dso_overlay       = pyqtSignal(bool)
    toggle_constellation_overlay = pyqtSignal(bool)
    open_star_mask_dialog    = pyqtSignal()
    open_subframe_selector   = pyqtSignal()
    blink_load_a             = pyqtSignal()
    blink_load_b             = pyqtSignal()
    blink_use_current_as_a   = pyqtSignal()
    blink_use_current_as_b   = pyqtSignal()
    blink_toggle             = pyqtSignal(bool)
    blink_fps_changed        = pyqtSignal(int)
    start_crop_draw          = pyqtSignal()
    run_crop                 = pyqtSignal()
    run_rotate               = pyqtSignal()
    run_flip                 = pyqtSignal()
    run_resize               = pyqtSignal()
    run_bin                  = pyqtSignal()
    run_invert               = pyqtSignal()
    preview_requested        = pyqtSignal(str)
    preview_cancelled        = pyqtSignal()
    run_multi_session        = pyqtSignal()
    multi_session_add_folder = pyqtSignal()
    multi_session_clear      = pyqtSignal()
    open_channel_combine_dialog = pyqtSignal()
    run_debayer              = pyqtSignal()
    clip_points_changed      = pyqtSignal(float, float)
    open_smart_processor     = pyqtSignal()
    open_equipment_dialog    = pyqtSignal()
    open_analysis_fwhm       = pyqtSignal()
    open_analysis_tilt       = pyqtSignal()
    open_analysis_photometry = pyqtSignal()
    open_ez_scripts          = pyqtSignal()
    open_live_stack          = pyqtSignal()
    open_mosaic_dialog       = pyqtSignal()
    open_processing_graph    = pyqtSignal()
    open_super_resolution    = pyqtSignal()
    run_superbias            = pyqtSignal()
    undo_requested           = pyqtSignal()
    redo_requested           = pyqtSignal()

    # ── Init ─────────────────────────────────────────────
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(260)
        self.setMaximumWidth(420)
        self.setStyleSheet(f"background: {BG_SECONDARY};")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Tool search — filters/expands matching sections across all tabs, so a
        # user can find a tool by name or keyword instead of hunting 9 tabs.
        from PyQt6.QtWidgets import QLineEdit
        self._tool_search = QLineEdit()
        self._tool_search.setPlaceholderText("Search tools…")
        self._tool_search.setClearButtonEnabled(True)
        self._tool_search.textChanged.connect(self._filter_tools)
        self._tool_search.setStyleSheet(
            "QLineEdit { background: #0d1117; color: #e6edf3; border: 1px solid "
            "#30363d; border-radius: 4px; padding: 5px 8px; margin: 4px; }"
        )
        outer.addWidget(self._tool_search)

        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)
        self._tabs.setUsesScrollButtons(True)
        self._tabs.tabBar().setExpanding(False)
        self._tabs.setStyleSheet(_TAB_SS)
        outer.addWidget(self._tabs)

        # bottom preset/undo bar
        bottom = QWidget()
        bottom.setObjectName("tools_bottom")
        bottom.setFixedHeight(34)
        bottom.setStyleSheet(_BOTTOM_SS)
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(8, 4, 8, 4)
        bl.setSpacing(4)

        self._btn_load_preset = RunBtn("↙ Load Preset", flat=True)
        self._btn_load_preset.setFixedHeight(24)
        self._btn_load_preset.clicked.connect(self._on_load_preset)
        bl.addWidget(self._btn_load_preset)
        self._btn_save_preset = RunBtn("↗ Save Preset", flat=True)
        self._btn_save_preset.setFixedHeight(24)
        self._btn_save_preset.clicked.connect(self._on_save_preset)
        bl.addWidget(self._btn_save_preset)
        bl.addStretch()
        self._btn_undo = RunBtn("↩", flat=True)
        self._btn_undo.setFixedWidth(32)
        self._btn_undo.setFixedHeight(24)
        self._btn_undo.setToolTip("Undo (Ctrl+Z)")
        self._btn_redo = RunBtn("↪", flat=True)
        self._btn_redo.setFixedWidth(32)
        self._btn_redo.setFixedHeight(24)
        self._btn_redo.setToolTip("Redo (Ctrl+Y)")
        self._btn_undo.clicked.connect(self.undo_requested)
        self._btn_redo.clicked.connect(self.redo_requested)
        bl.addWidget(self._btn_undo)
        bl.addWidget(self._btn_redo)
        outer.addWidget(bottom)

        # build tabs
        self._build_preprocess_tab()
        self._build_stacking_tab()
        self._build_background_tab()
        self._build_stretch_tab()
        self._build_transform_tab()
        self._build_color_tab()
        self._build_detail_tab()
        self._build_ai_tab()
        self._build_utility_tab()

        QTimer.singleShot(0, self._fix_tab_scroll_buttons)

    def _filter_tools(self, text: str):
        """Filter tool sections by the search box: show + expand matches across
        all tabs; an empty query restores the default collapsed/expanded state."""
        from astraios.ui.widgets.ui_kit import CollapsibleSection

        q = text.strip().lower()
        sections = self._tabs.findChildren(CollapsibleSection)
        if not q:
            for sec in sections:
                sec.setVisible(True)
                sec.set_open(getattr(sec, "_default_open", False))
            return
        for sec in sections:
            hit = sec.matches(q)
            sec.setVisible(hit)
            if hit:
                sec.set_open(True)

    def _fix_tab_scroll_buttons(self):
        from PyQt6.QtCore import Qt

        tb = self._tabs.tabBar()
        for btn in tb.findChildren(QToolButton):
            if btn.arrowType() == Qt.ArrowType.LeftArrow:
                btn.setText("◀")
                btn.setArrowType(Qt.ArrowType.NoArrow)
            elif btn.arrowType() == Qt.ArrowType.RightArrow:
                btn.setText("▶")
                btn.setArrowType(Qt.ArrowType.NoArrow)

    # ── TAB 1: Pre-Process ────────────────────────────────
    def _build_preprocess_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Calibration
        cal = CollapsibleSection("Calibration", accent=True)
        cal.add_info(
            "Create masters from raw frame folders or use pre-made masters."
        )
        self._cal_bias_label = cal.add_status_label("Bias: none")
        bl, _ = self._cal_frame_row(cal, "bias")
        self._cal_dark_label = cal.add_status_label("Dark: none")
        self._cal_frame_row(cal, "dark")
        self._cal_flat_label = cal.add_status_label("Flat: none")
        self._cal_frame_row(cal, "flat")
        cal.add_info("Light frames: add via Project panel → Import Lights")
        cal.add_run("▶ Run Calibration", self.run_calibration.emit)
        lay.addWidget(cal)

        # SuperBias
        sb = CollapsibleSection("SuperBias")
        sb.add_info("Statistically optimized bias — reduces read noise using spatial redundancy.")
        sb.add_run("▶ Create SuperBias", self.run_superbias.emit)
        lay.addWidget(sb)

        # Cosmetic
        cos = CollapsibleSection("Cosmetic Correction")
        cos.add_info("Detect and remove hot, cold, and dead pixels.")
        self._hot_sigma = cos.add_slider("Hot sigma", 5.0, 1.0, 20.0, 0.5, 1)
        self._cold_sigma = cos.add_slider("Cold sigma", 5.0, 1.0, 20.0, 0.5, 1)
        self._dead_pixel_check = cos.add_check("Detect dead pixels (value=0)", True)
        cos.add_run("▶ Apply Cosmetic Correction", self.run_cosmetic.emit)
        lay.addWidget(cos)

        # Debayer
        deb = CollapsibleSection("Debayer (OSC / Color Camera)")
        deb.add_info("Convert raw Bayer mosaic to color image.")
        self._debayer_pattern_combo = deb.add_combo(
            "Pattern",
            ["Auto-detect", "RGGB", "BGGR", "GRBG", "GBRG"],
        )
        self._debayer_method_combo = deb.add_combo(
            "Method",
            ["VNG (best quality)", "Edge-Aware (EA)",
             "Superpixel (2× bin)", "Bilinear (fastest)"],
        )
        deb.add_run("▶ Apply Debayer", self.run_debayer.emit)
        lay.addWidget(deb)

        self._tabs.addTab(scrollable_tab(lay), "⬡  Pre-Process")

    def _cal_frame_row(self, sec: CollapsibleSection, frame_type: str):
        rl = QHBoxLayout()
        rl.setSpacing(4)
        bf = RunBtn("Folder…", flat=True)
        bm = RunBtn("Master…", flat=True)
        bf.setFixedHeight(26)
        bm.setFixedHeight(26)
        rl.addWidget(bf)
        rl.addWidget(bm)
        sec.add_layout(rl)
        return rl, (bf, bm)

    # ── TAB 2: Stacking ───────────────────────────────────
    def _build_stacking_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Subframe Selector (moved here from Pre-Process)
        sub = CollapsibleSection("Subframe Selector")
        sub.add_info("Score and reject frames by FWHM, eccentricity, SNR, star count.")
        self._subframe_count_label = sub.add_status_label("No subframe selection active")
        sub.add_run("⊞ Open Subframe Selector…", self.open_subframe_selector.emit, flat=True)
        lay.addWidget(sub)

        # Registration
        reg = CollapsibleSection("Registration (Alignment)", accent=True)
        reg.add_info("Detect stars and align frames to a reference.")
        self._reg_mode_combo = reg.add_combo(
            "Mode",
            ["Star (1-Pass)", "Star (2-Pass)", "Triangle Match",
             "FFT Translation", "Comet"],
            "Star (2-Pass)",
        )
        self._star_sens_spin = reg.add_slider("Star sensitivity", 5.0, 1.0, 20.0, 0.5, 1)
        self._max_shift_spin = reg.add_spin("Max distance (px)", 10, 500, 50, 10)
        self._ransac_thresh_spin = reg.add_spin("RANSAC threshold", 1.0, 10.0, 3.0, 0.5, 1)
        self._ref_frame_combo = reg.add_combo(
            "Reference frame",
            ["Auto (best quality)", "First frame", "Last frame", "Specific frame #"],
        )
        reg.add_run("▶ Align Frames", self.run_alignment.emit, flat=True)
        lay.addWidget(reg)

        # Integration
        integ = CollapsibleSection("Integration (Stacking)", accent=True)
        integ.add_info("Combine aligned frames using rejection to increase SNR.")
        self._rejection_combo = integ.add_combo(
            "Rejection",
            ["Sigma Clipping", "Winsorized Sigma", "Linear Fit",
             "Percentile Clip", "ESD (Generalized)", "Min/Max", "None"],
        )
        self._norm_combo = integ.add_combo(
            "Normalization",
            ["Additive + Scaling", "Linear Fit", "Local", "Additive", "Multiplicative", "None"],
            current="Additive + Scaling",
        )
        self._integration_combo = integ.add_combo(
            "Integration",
            ["Average", "Median", "Weighted Average"],
        )
        self._kappa_spin = integ.add_slider("Kappa (σ)", 3.0, 0.5, 10.0, 0.1, 1)
        integ.add_run("▶ Stack Images", self.run_stacking.emit)
        lay.addWidget(integ)

        # Drizzle
        drz = CollapsibleSection("Drizzle Integration")
        self._drizzle_check = drz.add_check("Enable Drizzle")
        self._drizzle_scale_combo = drz.add_combo("Output scale", ["2× (recommended)", "3×"])
        self._drizzle_drop_spin = drz.add_slider("Drop shrink", 0.7, 0.5, 1.0, 0.05, 2)
        lay.addWidget(drz)

        # Multi-session
        ms = CollapsibleSection("Multi-Session Integration")
        ms.add_info("Stack frames from different telescopes, cameras, or nights.")
        self._ms_session_list = QListWidget()
        self._ms_session_list.setFixedHeight(80)
        self._ms_session_list.setStyleSheet(f"""
            QListWidget {{
                background: {BG_TERTIARY}; color: {TEXT_PRIMARY};
                border: 1px solid {BORDER}; border-radius: 4px;
                font-size: 11px;
            }}
        """)
        ms.add_widget(self._ms_session_list)
        btns = ms.add_btn_row([("+ Add Session…", True), ("Clear All", True)])
        btns[0].clicked.connect(self.multi_session_add_folder.emit)
        btns[1].clicked.connect(self.multi_session_clear.emit)
        self._ms_weight_combo = ms.add_combo(
            "Weighting",
            ["SNR (recommended)", "Integration time", "Equal weight"],
        )
        self._ms_normalize_check = ms.add_check("Normalize background", True)
        self._ms_align_check     = ms.add_check("Align sub-stacks", True)
        self._btn_ms_stack = ms.add_run("▶ Stack All Sessions", self.run_multi_session.emit)
        self._btn_ms_stack.setEnabled(False)
        lay.addWidget(ms)

        # Live Stack
        live = CollapsibleSection("Live Stack")
        live.add_info("Real-time frame accumulation with live preview.")
        live.add_run("▶ Open Live Stack…", self.open_live_stack.emit, flat=True)
        lay.addWidget(live)

        # Batch
        batch = CollapsibleSection("Batch Processing")
        batch.add_info("Full calibration → registration → stacking pipeline.")
        batch.add_run("⊞ Batch Preprocess…", self.open_batch_preprocess.emit, flat=True)
        batch.add_info("Apply a pipeline to multiple processed images.")
        batch.add_run("⊞ Open Batch Dialog…", self.open_batch_dialog.emit, flat=True)
        lay.addWidget(batch)

        # Mosaic
        mosaic = CollapsibleSection("Mosaic Stitching")
        mosaic.add_info("Combine overlapping panels into a seamless mosaic.")
        mosaic.add_run("⊞ Open Mosaic Dialog…", self.open_mosaic_dialog.emit, flat=True)
        lay.addWidget(mosaic)

        self._tabs.addTab(scrollable_tab(lay), "⧉  Stacking")

    # ── TAB 3: Background ─────────────────────────────────
    def _build_background_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Background Extraction
        bg = CollapsibleSection("Background Extraction", accent=True)
        bg.add_info("Remove light pollution gradients.")
        self._bg_grid_spin  = bg.add_spin("Grid size", 4, 32, 8)
        self._bg_order_spin = bg.add_spin("Poly order", 1, 6, 3)

        # auto-grid row
        grid_row = QHBoxLayout()
        grid_row.setSpacing(6)
        for lbl_text, attr in [("Rows", "_bg_grid_rows_spin"), ("Cols", "_bg_grid_cols_spin")]:
            sub_lay = QVBoxLayout()
            sub_lay.setSpacing(2)
            sub_lay.addWidget(make_label(lbl_text, TEXT_SECONDARY, 10))
            spin = styled_spin(2, 20, 5)
            setattr(self, attr, spin)
            sub_lay.addWidget(spin)
            grid_row.addLayout(sub_lay)
        bg.add_layout(grid_row)

        self._bg_box_size_spin = bg.add_spin("Box size (px)", 8, 256, 64, 8)
        self._bg_tolerance = bg.add_slider("Tolerance (σ)", 2.5, 1.0, 5.0, 0.5, 1)
        btn_grid = bg.add_run(
            "⊞ Add Auto-Grid Samples",
            lambda: self.add_bg_grid.emit(
                int(self._bg_grid_rows_spin.value()),
                int(self._bg_grid_cols_spin.value()),
                int(self._bg_box_size_spin.value()),
            ),
            flat=True,
        )

        self._btn_place_samples = RunBtn("Place Samples", flat=True)
        self._btn_place_samples.setCheckable(True)
        self._btn_place_samples.toggled.connect(self.toggle_sample_mode.emit)
        bg.add_widget(self._btn_place_samples)

        self._bg_sample_label = bg.add_status_label("0 manual samples")
        btn_clear = bg.add_run("Clear Samples", self.clear_bg_samples.emit, flat=True)
        bg.add_run("▶ Extract Background", self.run_background.emit)
        lay.addWidget(bg)

        # ABE
        abe = CollapsibleSection("ABE (Advanced)")
        abe.add_info("Background extraction using polynomial or RBF surface fitting.")
        self._abe_grid_spin   = abe.add_spin("Grid size", 5, 30, 10)
        self._abe_model_combo = abe.add_combo("Model", ["Polynomial (recommended)", "RBF"])
        self._abe_degree_spin = abe.add_spin("Poly degree", 1, 5, 2)
        self._abe_kernel_combo = abe.add_combo(
            "RBF kernel", ["Thin Plate Spline", "Multiquadric", "Gaussian"]
        )
        self._abe_mode_combo  = abe.add_combo("Mode", ["Subtraction", "Division"])
        abe.add_run("▶ Run ABE", self.run_abe.emit)
        lay.addWidget(abe)

        # Background Neutralization (NEW)
        bn = CollapsibleSection("Background Neutralization")
        bn.add_info(
            "Shift sky background to neutral zero per-channel. "
            "Equivalent to PixInsight BackgroundNeutralization (statistical mode)."
        )
        self._bn_percentile = bn.add_slider("Percentile", 2.0, 0.5, 10.0, 0.5, 1)
        self._bn_amount     = bn.add_slider("Amount", 1.0, 0.0, 1.0, 0.05, 2)
        self._bn_protect    = bn.add_slider("Protect bright", 0.5, 0.0, 1.0, 0.05, 2)
        bn.add_run("▶ Apply Background Neutralization",
                   self.run_background_neutralization.emit)
        lay.addWidget(bn)

        # Vignette
        vig = CollapsibleSection("Vignette Correction")
        vig.add_info("Remove optical vignetting toward image edges.")
        self._vignette_amount = vig.add_slider("Amount", 0.3, 0.0, 1.0, 0.05, 2)
        self._vignette_radius = vig.add_slider("Radius", 0.8, 0.3, 1.0, 0.05, 2)
        vig.add_run("▶ Correct Vignette", self.run_vignette_correction.emit)
        lay.addWidget(vig)

        # Banding
        band = CollapsibleSection("Banding Reduction")
        band.add_info("Remove horizontal/vertical banding from CMOS sensors.")
        self._banding_amount = band.add_slider("Amount", 1.0, 0.1, 3.0, 0.1, 1)
        self._banding_dir_combo = band.add_combo(
            "Direction", ["Horizontal", "Vertical", "Both"]
        )
        band.add_run("▶ Reduce Banding", self.run_banding.emit)
        lay.addWidget(band)

        self._tabs.addTab(scrollable_tab(lay), "◫  Background")

    # ── TAB 4: Stretch ────────────────────────────────────
    def _build_stretch_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Auto-stretch
        aut = CollapsibleSection("Auto-Stretch", accent=True)
        aut.add_info("Statistical midtone stretch.")
        self._midtone_slider = aut.add_slider("Midtone", 0.25, 0.01, 0.99, 0.01, 2, 0.25)
        self._midtone_slider.value_changed.connect(lambda _: self.stretch_params_changed.emit())
        self._shadow_spin = aut.add_spin("Shadow clip", -10.0, 0.0, -2.8, 0.1, 1)
        self._linked_check = aut.add_check("Link RGB channels", True)
        self._split_check  = aut.add_check("Before/After split preview")
        self._split_check.toggled.connect(lambda _: self.stretch_params_changed.emit())
        btns = aut.add_btn_row([("▶ Apply Stretch", False), ("Reset", True)])
        btns[0].clicked.connect(self.run_stretch.emit)
        btns[1].clicked.connect(lambda: self._midtone_slider.setValue(0.25))
        lay.addWidget(aut)

        # Statistical Stretch (target median)
        sst2 = CollapsibleSection("Statistical Stretch")
        sst2.add_info(
            "Stretch the background to a chosen target level — the midtone is "
            "solved for you. Lower target = darker background."
        )
        self._statstretch_target = sst2.add_slider("Target median", 0.25, 0.05, 0.6, 0.01, 2)
        self._statstretch_shadow = sst2.add_spin("Shadow clip", -10.0, 0.0, -2.8, 0.1, 1)
        self._statstretch_linked = sst2.add_check("Link RGB channels", True)
        self._statstretch_preview_check = sst2.add_check("Live split preview")
        self._statstretch_target.value_changed.connect(
            lambda _: self._fire_preview("statistical_stretch", self._statstretch_preview_check)
        )
        self._statstretch_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("statistical_stretch") if on
            else self.preview_cancelled.emit()
        )
        sst2.add_run("▶ Apply Statistical Stretch", self.run_statistical_stretch.emit)
        lay.addWidget(sst2)

        # Arcsinh (NEW)
        arc = CollapsibleSection("Arcsinh Stretch")
        arc.add_info(
            "Lupton et al. 2004 — linear-to-arcsinh ramp. Preserves star colours "
            "better than log; reveals faint nebulosity without blowing out stars."
        )
        self._arcsinh_factor_spin = arc.add_spin(
            "Stretch factor β", 0.1, 1000.0, 10.0, 1.0, 1
        )
        self._arcsinh_bp_spin = arc.add_spin("Black point", 0.0, 0.5, 0.0, 0.001, 4)
        self._arcsinh_linked_check = arc.add_check("Linked RGB", True)
        self._arcsinh_preview_check = arc.add_check("Live split preview")
        self._arcsinh_factor_spin.valueChanged.connect(
            lambda _: self._fire_preview("arcsinh_stretch", self._arcsinh_preview_check)
        )
        self._arcsinh_bp_spin.valueChanged.connect(
            lambda _: self._fire_preview("arcsinh_stretch", self._arcsinh_preview_check)
        )
        self._arcsinh_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("arcsinh_stretch") if on
            else self.preview_cancelled.emit()
        )
        btns = arc.add_btn_row([("▶ Apply Arcsinh Stretch", False), ("Reset", True)])
        btns[0].clicked.connect(self.run_arcsinh_stretch.emit)
        btns[1].clicked.connect(lambda: (
            self._arcsinh_factor_spin.setValue(10.0),
            self._arcsinh_bp_spin.setValue(0.0),
        ))
        lay.addWidget(arc)

        # Star Stretch
        sst = CollapsibleSection("Star Stretch")
        sst.add_info(
            "Colour-preserving stretch for star layers — lifts faint stars while "
            "keeping their hue. Best run on an extracted star image, then screened back."
        )
        self._star_stretch_amount = sst.add_slider("Amount", 0.2, 0.0, 1.0, 0.05, 2)
        self._star_stretch_color = sst.add_slider("Colour boost", 1.0, 0.0, 3.0, 0.05, 2)
        self._star_stretch_preview_check = sst.add_check("Live split preview")
        for _sl in (self._star_stretch_amount, self._star_stretch_color):
            _sl.value_changed.connect(
                lambda _, s=self._star_stretch_preview_check: self._fire_preview("star_stretch", s)
            )
        self._star_stretch_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("star_stretch") if on
            else self.preview_cancelled.emit()
        )
        sst.add_run("▶ Apply Star Stretch", self.run_star_stretch.emit)
        lay.addWidget(sst)

        # GHS
        ghs = CollapsibleSection("Generalized Hyperbolic Stretch")
        ghs.add_info("Advanced non-linear stretch.")
        self._ghs_d_spin  = ghs.add_spin("Stretch (D)",   0.0, 20.0, 5.0, 0.5, 1)
        self._ghs_b_spin  = ghs.add_spin("Asymmetry (b)", -5.0, 5.0, 0.0, 0.1, 1)
        self._ghs_sp_spin = ghs.add_spin("Sym. point",    0.0,  1.0, 0.0, 0.05, 3)
        self._ghs_shadow_slider    = ghs.add_slider("Shadow prot.",    0.0, 0.0, 1.0, 0.01, 2)
        self._ghs_highlight_slider = ghs.add_slider("Highlight prot.", 0.0, 0.0, 1.0, 0.01, 2)
        self._ghs_preview_check = ghs.add_check("Live split preview")
        for _ghs_spin in (self._ghs_d_spin, self._ghs_b_spin, self._ghs_sp_spin):
            _ghs_spin.valueChanged.connect(
                lambda _, s=self._ghs_preview_check: self._fire_preview("ghs", s)
            )
        for _ghs_sl in (self._ghs_shadow_slider, self._ghs_highlight_slider):
            _ghs_sl.value_changed.connect(
                lambda _, s=self._ghs_preview_check: self._fire_preview("ghs", s)
            )
        self._ghs_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("ghs") if on
            else self.preview_cancelled.emit()
        )
        btns = ghs.add_btn_row([("▶ Apply GHS", False), ("Reset", True)])
        btns[0].clicked.connect(self.run_ghs.emit)
        lay.addWidget(ghs)

        # Histogram Transform
        ht = CollapsibleSection("Histogram Transform")
        ht.add_info("Black point, midtone, and white point adjustment.")
        self._ht_black_spin  = ht.add_spin("Black point", 0.0, 0.99, 0.0, 0.01, 3)
        self._ht_midtone_slider = ht.add_slider("Midtone",  0.5, 0.01, 0.99, 0.01, 2, 0.5)
        self._ht_white_spin  = ht.add_spin("White point", 0.01, 1.0, 1.0, 0.01, 3)
        self._ht_black_spin.valueChanged.connect(lambda _: self._emit_clip_points())
        self._ht_white_spin.valueChanged.connect(lambda _: self._emit_clip_points())
        self._ht_preview_check = ht.add_check("Live split preview")
        self._ht_black_spin.valueChanged.connect(
            lambda _: self._fire_preview("histogram_transform", self._ht_preview_check)
        )
        self._ht_midtone_slider.value_changed.connect(
            lambda _: self._fire_preview("histogram_transform", self._ht_preview_check)
        )
        self._ht_white_spin.valueChanged.connect(
            lambda _: self._fire_preview("histogram_transform", self._ht_preview_check)
        )
        self._ht_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("histogram_transform") if on
            else self.preview_cancelled.emit()
        )
        btns = ht.add_btn_row([("▶ Apply HT", False), ("Reset", True)])
        btns[0].clicked.connect(self.run_histogram_transform.emit)
        lay.addWidget(ht)

        # Curves
        crv = CollapsibleSection("Curves")
        crv.add_info("Click to add points, drag to adjust. Right-click to remove.")
        self._curve_channel_combo = crv.add_combo(
            "Channel", ["Master (L)", "Red", "Green", "Blue"]
        )
        self._curve_channel_combo.currentIndexChanged.connect(
            lambda _: self.curves_histogram_changed.emit()
        )
        self._curve_editor = CurveEditor()
        self._curve_editor.setMinimumHeight(180)
        crv.add_widget(self._curve_editor)
        self._curves_histogram_check = crv.add_check("Show histogram")
        self._curves_histogram_check.stateChanged.connect(
            lambda _: self.curves_histogram_changed.emit()
        )
        btns = crv.add_btn_row([("▶ Apply Curves", False), ("Reset", True)])
        btns[0].clicked.connect(self.run_curves.emit)
        btns[1].clicked.connect(self._curve_editor.reset)
        lay.addWidget(crv)

        self._tabs.addTab(scrollable_tab(lay), "◑  Stretch")

    # ── TAB 5: Transform ──────────────────────────────────
    def _build_transform_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Crop
        crop = CollapsibleSection("Crop", accent=True)
        crop.add_info("Crop image to a rectangular region.")
        self._btn_crop_draw = RunBtn("✏ Draw on Image…", flat=True)
        self._btn_crop_draw.setCheckable(True)
        self._btn_crop_draw.toggled.connect(
            lambda on: self.start_crop_draw.emit() if on else None
        )
        crop.add_widget(self._btn_crop_draw)
        self._crop_x_spin = crop.add_spin("X offset", 0, 99999, 0)
        self._crop_y_spin = crop.add_spin("Y offset", 0, 99999, 0)
        self._crop_w_spin = crop.add_spin("Width",    0, 99999, 0)
        self._crop_h_spin = crop.add_spin("Height",   0, 99999, 0)
        crop.add_run("▶ Apply Crop", self.run_crop.emit)
        lay.addWidget(crop)

        # Rotate
        rot = CollapsibleSection("Rotate")
        self._rotate_combo = rot.add_combo(
            "Preset", ["90° CW", "180°", "270° CW", "Custom angle"]
        )
        self._rotate_angle_spin = rot.add_spin("Custom angle", -360, 360, 0, 0.1, 1, "°")
        self._rotate_expand_check = rot.add_check("Expand canvas", True)
        rot.add_run("▶ Apply Rotation", self.run_rotate.emit)
        lay.addWidget(rot)

        # Flip
        flp = CollapsibleSection("Flip")
        self._flip_combo = flp.add_combo("Axis", ["Horizontal", "Vertical", "Both"])
        flp.add_run("▶ Apply Flip", self.run_flip.emit)
        lay.addWidget(flp)

        # Resize
        rsz = CollapsibleSection("Resize / Resample")
        self._resize_scale_spin = rsz.add_spin("Scale", 0.1, 10.0, 1.0, 0.1, 2)
        self._resize_interp_combo = rsz.add_combo(
            "Interpolation", ["Lanczos", "Bicubic", "Bilinear", "Nearest"]
        )
        rsz.add_run("▶ Apply Resize", self.run_resize.emit)
        lay.addWidget(rsz)

        # Bin
        bn = CollapsibleSection("Bin")
        bn.add_info("Combine pixels to increase SNR at lower resolution.")
        self._bin_factor_combo = bn.add_combo("Factor", ["2x2", "3x3", "4x4"])
        self._bin_mode_combo   = bn.add_combo("Mode",   ["Average", "Sum"])
        bn.add_run("▶ Apply Bin", self.run_bin.emit)
        lay.addWidget(bn)

        # Invert
        inv = CollapsibleSection("Invert")
        inv.add_info("Invert all pixel values (1 − image).")
        inv.add_run("▶ Invert Image", self.run_invert.emit)
        lay.addWidget(inv)

        self._tabs.addTab(scrollable_tab(lay), "⟳  Transform")

    # ── TAB 6: Color ──────────────────────────────────────
    def _build_color_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # SCNR
        scnr = CollapsibleSection("SCNR (Green Noise Removal)", accent=True)
        scnr.add_info("Remove color noise, typically excess green channel.")
        self._scnr_target_combo  = scnr.add_combo("Target",  ["Green", "Red", "Blue"])
        self._scnr_method_combo  = scnr.add_combo(
            "Method",
            ["Average Neutral", "Maximum Neutral", "Additive-Subtractive Mask"],
        )
        self._scnr_amount = scnr.add_slider("Amount", 0.5, 0.0, 1.0, 0.01, 2)
        self._scnr_preview_check = scnr.add_check("Live split preview")
        self._scnr_amount.value_changed.connect(
            lambda _: self._fire_preview("scnr", self._scnr_preview_check)
        )
        self._scnr_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("scnr") if on
            else self.preview_cancelled.emit()
        )
        scnr.add_run("▶ Apply SCNR", self.run_scnr.emit)
        lay.addWidget(scnr)

        # Color Adjust
        ca = CollapsibleSection("Color Adjustment")
        self._hue_slider        = ca.add_slider("Hue shift",   0, -180, 180, 1, 0)
        self._sat_slider        = ca.add_slider("Saturation",  0, -100, 100, 1, 0)
        self._vibrance_slider   = ca.add_slider("Vibrance",    0, -100, 100, 1, 0)
        ca.add_run("▶ Apply Color Adjust", self.run_color_adjust.emit)
        lay.addWidget(ca)

        # Color Calibration
        cc = CollapsibleSection("Color Calibration")
        cc.add_info("White balance using background reference or star colours.")
        self._cc_method_combo = cc.add_combo(
            "Method",
            ["Background reference", "Photometric (SPCC)", "Manual RGB"],
        )
        cc.add_info("Manual RGB multipliers (only for Manual RGB method).")
        rgb_row = QHBoxLayout()
        rgb_row.setSpacing(6)
        self._cc_r_spin = styled_spin(0.1, 5.0, 1.0, 0.01, 2, "")
        self._cc_g_spin = styled_spin(0.1, 5.0, 1.0, 0.01, 2, "")
        self._cc_b_spin = styled_spin(0.1, 5.0, 1.0, 0.01, 2, "")
        for label, spin in [
            ("R", self._cc_r_spin), ("G", self._cc_g_spin), ("B", self._cc_b_spin),
        ]:
            sub = QHBoxLayout()
            sub.setSpacing(2)
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #aaa; font-size: 10px;")
            sub.addWidget(lbl)
            sub.addWidget(spin)
            rgb_row.addLayout(sub)
        cc.body.addLayout(rgb_row)
        self._cc_rgb_row = rgb_row
        self._cc_method_combo.currentIndexChanged.connect(self._toggle_cc_manual_rgb)
        self._toggle_cc_manual_rgb()
        btns = cc.add_btn_row([("Pick BG Reference", True), ("Calibrate", False)])
        btns[1].clicked.connect(self.run_color_calibration.emit)
        lay.addWidget(cc)

        # PCC — Photometric Color Calibration (plate solve + Gaia DR3)
        pcc = CollapsibleSection("PCC (Photometric)")
        pcc.add_info("Plate-solve + Gaia DR3 star catalog white balance.")
        self._pcc_solver_combo = pcc.add_combo(
            "Solver", ["Auto", "ASTAP", "Astrometry.net"]
        )
        self._pcc_ra_spin  = pcc.add_spin("RA hint (°)",  0.0, 360.0, 0.0, 0.001, 3)
        self._pcc_dec_spin = pcc.add_spin("Dec hint (°)", -90.0, 90.0, 0.0, 0.001, 3)
        pcc.add_info("Leave RA/Dec at 0 to read from FITS header.")
        pcc.add_run("▶ Run PCC", self.run_pcc.emit)
        lay.addWidget(pcc)

        # SPCC — Spectrophotometric Color Calibration (sensor QE + filter curves)
        spcc = CollapsibleSection("SPCC (Spectrophotometric)")
        spcc.add_info("Sensor QE + filter transmission curves for precise white balance.")
        self._spcc_filter_combo  = spcc.add_combo(
            "Filter set", ["Broadband (L/R/G/B)", "Narrowband Ha/OIII/SII", "Custom"]
        )
        self._spcc_camera_combo  = spcc.add_combo(
            "Camera", ["ZWO ASI2600MM Pro", "QHY268M", "ZWO ASI533MC Pro"]
        )
        spcc.add_run("▶ Run SPCC", self.run_spcc.emit)
        lay.addWidget(spcc)
        self._populate_spcc_cameras()

        # Narrowband
        nb = CollapsibleSection("Narrowband Tools")
        nb.add_info("SHO/HOO/HaRGB palette mapping, continuum subtraction, blending.")
        nb.add_run("⊞ Open Narrowband Dialog…", self.open_narrowband_dialog.emit, flat=True)
        nb.add_run("▶ Continuum Subtraction", self.run_continuum_subtraction.emit, flat=True)
        lay.addWidget(nb)

        # LRGB / Channels
        lc = CollapsibleSection("LRGB / Channel Combine")
        lc.add_info("Combine luminance and colour channels.")
        btns = lc.add_btn_row([("LRGB Combine…", True), ("Channel Combine…", True)])
        btns[0].clicked.connect(self.run_lrgb_combine.emit)
        btns[1].clicked.connect(self.open_channel_combine_dialog.emit)
        lc.add_divider()
        lc.add_run("▶ Extract Luminance", self.run_extract_luminance.emit, flat=True)
        lc.add_run("▶ Split Channels", self.run_split_channels.emit, flat=True)
        lay.addWidget(lc)

        self._tabs.addTab(scrollable_tab(lay), "◈  Color")

    # ── TAB 7: Detail ─────────────────────────────────────
    def _build_detail_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Deconvolution
        dec = CollapsibleSection("Deconvolution", accent=True)
        dec.add_info("Restore fine detail lost to seeing or tracking.")
        self._deconv_method_combo = dec.add_combo(
            "Method", ["Richardson-Lucy", "Blind (Spatial)", "Wiener"]
        )
        self._deconv_psf_spin   = dec.add_spin("PSF FWHM (px)", 0.5, 20.0, 3.0, 0.1, 1)
        self._deconv_iter       = dec.add_slider("Iterations", 30, 1, 200, 1, 0)
        self._deconv_reg        = dec.add_spin("Regularization", 0.0, 0.1, 0.001, 0.001, 4)
        self._deconv_deringing  = dec.add_check("Deringing protection", True)
        self._deconv_dering_amt = dec.add_slider("Deringing amount", 0.5, 0.0, 1.0, 0.05, 2)
        btns = dec.add_btn_row([("Measure PSF", True), ("Star Mask", True)])
        btns[0].clicked.connect(self.measure_psf.emit)
        btns[1].clicked.connect(self.open_star_mask_dialog.emit)
        self._deconv_preview_check = dec.add_check("Live split preview")
        for _w in (self._deconv_iter, self._deconv_dering_amt):
            _w.value_changed.connect(
                lambda _: self._fire_preview("deconvolution", self._deconv_preview_check)
            )
        self._deconv_psf_spin.valueChanged.connect(
            lambda _: self._fire_preview("deconvolution", self._deconv_preview_check)
        )
        self._deconv_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("deconvolution") if on
            else self.preview_cancelled.emit()
        )
        dec.add_run("▶ Apply Deconvolution", self.run_deconvolution.emit)
        lay.addWidget(dec)

        # PSF Measurement (expanded results)
        psf = CollapsibleSection("PSF Measurement")
        psf.add_info("Detect stars, fit 2D Gaussians, report FWHM and ellipticity.")
        # results grid
        self._psf_result_labels: dict[str, QLabel] = {}
        metrics = [
            ("FWHM", "—"), ("FWHM X", "—"), ("FWHM Y", "—"),
            ("Ellipticity", "—"), ("Rotation", "—"), ("Stars used", "—"),
            ("FWHM σ", "—"),
        ]
        from PyQt6.QtWidgets import QGridLayout
        grid_w = QWidget()
        grid_w.setStyleSheet(
            f"background: {BG_TERTIARY}; border-radius: 5px; border: 1px solid {BORDER};"
        )
        grid_lay = QGridLayout(grid_w)
        grid_lay.setContentsMargins(8, 8, 8, 8)
        grid_lay.setSpacing(4)
        for i, (name, default) in enumerate(metrics):
            col = (i % 2) * 2
            row = i // 2
            grid_lay.addWidget(make_label(name, TEXT_SECONDARY, 9), row * 2, col)
            val_lbl = make_label(default, ACCENT, 11, mono=True)
            self._psf_result_labels[name] = val_lbl
            grid_lay.addWidget(val_lbl, row * 2 + 1, col)
        psf.add_widget(grid_w)
        self._psf_cutout_spin = psf.add_spin("Cutout radius", 6, 32, 12)
        self._psf_force_cpu   = psf.add_check("Force CPU (for parallel use)")
        psf.add_run("▶ Measure PSF", self.measure_psf.emit)
        lay.addWidget(psf)

        # Noise Reduction
        dnz = CollapsibleSection("Noise Reduction")
        self._denoise_method_combo = dnz.add_combo(
            "Method",
            ["TGV Denoise", "NLM (Non-Local Means)", "Wavelet Denoise", "Median Filter"],
        )
        self._denoise_amount     = dnz.add_slider("Amount",     0.5, 0.0, 1.0, 0.05, 2)
        self._denoise_lum        = dnz.add_slider("Luminance",  0.7, 0.0, 1.0, 0.05, 2)
        self._denoise_chrom      = dnz.add_slider("Chrominance",0.5, 0.0, 1.0, 0.05, 2)
        _auto_btn = dnz.add_btn_row([("🎯 Auto (measure noise)", False)])[0]
        _auto_btn.clicked.connect(self.request_auto_denoise.emit)
        self._denoise_noise_label = dnz.add_status_label("Noise: not measured")
        self._denoise_preview_check = dnz.add_check("Live split preview")
        for _sl in (self._denoise_amount, self._denoise_lum, self._denoise_chrom):
            _sl.value_changed.connect(
                lambda _, s=self._denoise_preview_check: self._fire_preview("denoise", s)
            )
        self._denoise_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("denoise") if on
            else self.preview_cancelled.emit()
        )
        dnz.add_run("▶ Apply Denoise", self.run_denoise.emit)
        lay.addWidget(dnz)

        # Star Reduction
        sr = CollapsibleSection("Star Reduction")
        sr.add_info("Reduce star halos to reveal faint nebula details.")
        self._star_reduction_amount = sr.add_slider("Amount (%)", 50, 0, 100, 1, 0)
        self._star_reduction_kernel = sr.add_combo(
            "Kernel", ["Elliptical", "Circular", "Square", "Diamond"]
        )
        self._star_reduction_iters = sr.add_slider("Iterations", 2, 1, 10, 1, 0)
        self._star_reduction_protect = sr.add_check("Protect core", True)
        sr.add_run("▶ Reduce Stars", self.run_star_reduction.emit)
        lay.addWidget(sr)

        # Wavelets / MLT
        wav = CollapsibleSection("Wavelets / MLT")
        wav.add_info("Multi-scale sharpening with per-layer control.")
        self._wavelet_layers = wav.add_slider("Layers", 5, 2, 8, 1, 0)
        self._wavelet_layer_sliders: list[SliderRow] = []
        defaults = [0.3, 0.3, 0.0, 0.0, 0.0]
        for i in range(5):
            s = wav.add_slider(f"Layer {i+1}", defaults[i], 0.0, 2.0, 0.1, 1)
            self._wavelet_layer_sliders.append(s)
        self._wav_preview_check = wav.add_check("Live split preview")
        for _sl in self._wavelet_layer_sliders:
            _sl.value_changed.connect(
                lambda _, s=self._wav_preview_check: self._fire_preview("wavelet", s)
            )
        self._wav_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("wavelet") if on
            else self.preview_cancelled.emit()
        )
        btns = wav.add_btn_row([("▶ Wavelets", False), ("▶ MLT", False)])
        btns[0].clicked.connect(self.run_wavelet_sharpen.emit)
        btns[1].clicked.connect(self.run_mlt.emit)
        lay.addWidget(wav)

        # Frequency Separation
        fs = CollapsibleSection("Frequency Separation")
        fs.add_info("Split into structure (LF) + detail (HF). Boost detail or smooth colour/gradients independently.")
        self._fs_method = fs.add_combo("Method", ["Subtract (linear)", "Divide (ratio)"])
        self._fs_sigma = fs.add_slider("Split radius", 5.0, 1.0, 50.0, 1.0, 1)
        self._fs_hf_boost = fs.add_slider("Detail boost", 1.0, 0.0, 3.0, 0.05, 2)
        self._fs_lf_smooth = fs.add_slider("Smooth structure", 0.0, 0.0, 30.0, 1.0, 1)
        self._fs_preview_check = fs.add_check("Live split preview")
        for _sl in (self._fs_sigma, self._fs_hf_boost, self._fs_lf_smooth):
            _sl.value_changed.connect(
                lambda _, s=self._fs_preview_check: self._fire_preview("frequency_separation", s)
            )
        self._fs_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("frequency_separation") if on
            else self.preview_cancelled.emit()
        )
        fs.add_run("▶ Apply Frequency Separation", self.run_frequency_separation.emit)
        lay.addWidget(fs)

        # CLAHE
        clh = CollapsibleSection("Local Contrast / CLAHE")
        self._clahe_clip  = clh.add_slider("Clip limit", 2.0, 0.5, 10.0, 0.5, 1)
        self._clahe_tiles = clh.add_slider("Tile size",  8,   4,   32,   1,   0)
        self._clahe_preview_check = clh.add_check("Live split preview")
        for _sl in (self._clahe_clip, self._clahe_tiles):
            _sl.value_changed.connect(
                lambda _, s=self._clahe_preview_check: self._fire_preview("local_contrast", s)
            )
        self._clahe_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("local_contrast") if on
            else self.preview_cancelled.emit()
        )
        clh.add_run("▶ Apply CLAHE", self.run_local_contrast.emit)
        lay.addWidget(clh)

        # Unsharp Mask
        um = CollapsibleSection("Unsharp Mask")
        self._um_radius    = um.add_slider("Radius (px)", 1.5, 0.5, 10.0, 0.5, 1)
        self._um_amount    = um.add_slider("Amount",      0.5, 0.0,  2.0, 0.05, 2)
        self._um_threshold = um.add_slider("Threshold",   0.0, 0.0,  0.1, 0.005, 3)
        self._um_preview_check = um.add_check("Live split preview")
        for _sl in (self._um_radius, self._um_amount, self._um_threshold):
            _sl.value_changed.connect(
                lambda _, s=self._um_preview_check: self._fire_preview("unsharp_mask", s)
            )
        self._um_preview_check.toggled.connect(
            lambda on: self.preview_requested.emit("unsharp_mask") if on
            else self.preview_cancelled.emit()
        )
        um.add_run("▶ Apply Unsharp Mask", self.run_unsharp_mask.emit)
        lay.addWidget(um)

        # Morphology
        mor = CollapsibleSection("Morphology")
        self._morph_op     = mor.add_combo(
            "Operation", ["Erosion", "Dilation", "Opening", "Closing", "Gradient"]
        )
        self._morph_kernel = mor.add_combo("Kernel", ["Disk", "Square", "Diamond"])
        self._morph_iters  = mor.add_slider("Iterations", 1, 1, 10, 1, 0)
        mor.add_run("▶ Apply Morphology", self.run_morphology.emit)
        lay.addWidget(mor)

        # Chromatic Aberration
        ca = CollapsibleSection("Chromatic Aberration")
        ca.add_info("Correct lateral colour fringing at image edges.")
        ca.add_run("▶ Correct CA", self.run_chromatic_aberration.emit)
        lay.addWidget(ca)

        self._tabs.addTab(scrollable_tab(lay), "◎  Detail")

    # ── TAB 8: AI Tools ───────────────────────────────────
    def _build_ai_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Credit card
        credit = QLabel(
            "AI models powered by CosmicClarity — "
            "MIT-licensed astro AI by Franklin Marek (Seti Astro)"
        )
        credit.setWordWrap(True)
        credit.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 10px; "
            "padding: 6px 8px; background: #1a1a2e; border-radius: 5px;"
        )
        lay.addWidget(credit)

        # Status card
        status_w = QWidget()
        status_w.setStyleSheet(
            f"background: {BG_SECONDARY}; border: 1px solid {BORDER}; border-radius: 6px;"
        )
        sl = QVBoxLayout(status_w)
        sl.setContentsMargins(10, 10, 10, 10)
        sl.setSpacing(6)
        hdr_row = QHBoxLayout()
        hdr_row.addWidget(make_label("AI Model Status", TEXT_SECONDARY, 10, bold=True))
        hdr_row.addStretch()
        gpu_lbl = QLabel("GPU")
        gpu_lbl.setStyleSheet(
            f"background: {ACCENT_PURPLE}; color: #fff; font-size: 10px; "
            "font-weight: 700; border-radius: 8px; padding: 1px 8px;"
        )
        hdr_row.addWidget(gpu_lbl)
        sl.addLayout(hdr_row)

        _model_statuses = [
            ("AI Denoise (Noise2Self)",       "ready"),
            ("AI Sharpen (Richardson-Lucy)",  "ready"),
            ("Star Removal (built-in)",       "ready"),
            ("StarNet (external binary)",     "optional"),
            ("CosmicClarity (your models)",   "optional"),
        ]
        _status_style = {
            "ready":    (ACCENT_DARK, ACCENT, "Ready"),
            "training": ("#2d1f00", ORANGE,  "Training…"),
            "planned":  ("#1c1c2e", ACCENT_PURPLE, "Planned"),
            "optional": ("#1c1c2e", ACCENT_PURPLE, "Optional"),
        }
        for name, status in _model_statuses:
            row = QHBoxLayout()
            row.addWidget(make_label(name, TEXT_PRIMARY, 11))
            row.addStretch()
            bg, col, lbl_text = _status_style[status]
            badge = QLabel(lbl_text)
            badge.setStyleSheet(
                f"background: {bg}; color: {col}; font-size: 10px; "
                "font-weight: 600; border-radius: 8px; padding: 1px 8px;"
            )
            row.addWidget(badge)
            sl.addLayout(row)
        lay.addWidget(status_w)

        # AI Denoise
        den = CollapsibleSection("AI Denoise", accent=True)
        den.add_info(
            "Noise2Self: a self-supervised denoiser trained on real astro images, "
            "built in and ready. CosmicClarity uses your own Cosmic Clarity models "
            "(set their folder in Preferences); it does nothing until you do."
        )
        self._ai_denoise_backend = den.add_combo(
            "Backend",
            ["Noise2Self (built-in)", "CosmicClarity (your models)"],
            current="Noise2Self (built-in)",
        )
        self._ai_denoise_strength = den.add_slider("Strength", 0.7, 0.0, 1.0, 0.05, 2)
        self._ai_tile_combo       = den.add_combo("Tile size", ["128", "256", "512", "Full"])
        self._ai_star_protect     = den.add_check("Protect stars (star mask)", True)
        self._ai_tiled_check      = den.add_check("Tiled inference (reduces VRAM)", True)
        den.add_run("▶ Apply AI Denoise", self.run_ai_denoise.emit)
        lay.addWidget(den)

        # AI Sharpen
        shr = CollapsibleSection("AI Sharpen")
        shr.add_info(
            "Richardson-Lucy deconvolution, built in and ready. CosmicClarity uses "
            "your own Cosmic Clarity models (set their folder in Preferences); it "
            "does nothing until you do."
        )
        self._ai_sharpen_backend = shr.add_combo(
            "Backend",
            ["Richardson-Lucy (built-in)", "CosmicClarity (your models)"],
            current="Richardson-Lucy (built-in)",
        )
        self._ai_sharpen_strength = shr.add_slider("Strength", 0.5, 0.0, 1.0, 0.05, 2)
        shr.add_run("▶ Apply AI Sharpen", self.run_ai_sharpen.emit)
        lay.addWidget(shr)

        # Star Removal
        star = CollapsibleSection("Star Removal", accent=True)
        star.add_info("Remove stars from image using deep learning.")
        self._star_removal_path = star.add_combo(
            "Backend",
            ["Auto (StarNet v2 preferred)", "StarNet v2", "Built-in (starrem2k13)"],
            current="Auto (StarNet v2 preferred)",
        )
        self._star_threshold = star.add_slider("Threshold", 0.5, 0.1, 0.9, 0.05, 2)
        self._star_protect_bg = star.add_check("Protect background detail", True)
        star.add_info(
            "StarNet v2 requires manual download from starnetastro.com. "
            "Built-in (starrem2k13) works out of the box — run download script first.",
        )
        star.add_run("▶ Remove Stars", self.run_starnet.emit)
        lay.addWidget(star)

        # AI Super-Resolution
        sr = CollapsibleSection("AI Super-Resolution")
        sr.add_info("Upscale images with learned detail synthesis (Real-ESRGAN).")
        self._sr_scale = sr.add_combo("Scale", ["2×", "4×"], current="2×")
        self._sr_tile = sr.add_combo("Tile size", ["512", "1024", "Full"], current="512")
        sr.add_run("▶ Upscale", self.open_super_resolution.emit)
        lay.addWidget(sr)

        # Train
        train = CollapsibleSection("Train Your Own Models")
        train.add_info("Self-supervised training on your own astro images.")
        train.add_code_block(
            "poetry run python scripts/\ntrain_denoise_model.py\n--input astro_data --epochs 30"
        )
        train.add_run("Open Training Guide…", flat=True)
        lay.addWidget(train)

        self._tabs.addTab(scrollable_tab(lay), "✦  AI Tools")

    # ── TAB 9: Utility ────────────────────────────────────
    def _build_utility_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Pixel Math
        pm = CollapsibleSection("Pixel Math", accent=True)
        pm.add_info("Custom per-pixel math. Variables: R, G, B, L, img1, img2.")
        self._pixelmath_expr = QTextEdit()
        self._pixelmath_expr.setPlaceholderText("R * 0.5 + B * 0.5")
        self._pixelmath_expr.setFixedHeight(54)
        self._pixelmath_expr.setStyleSheet(
            f"background: {BG_TERTIARY}; color: {ACCENT}; "
            f"border: 1px solid {BORDER}; border-radius: 5px; "
            f"padding: 4px 8px; font-family: {FONT_MONO}; font-size: 11px;"
        )
        pm.add_widget(self._pixelmath_expr)
        pm.add_run("⊞ Open Pixel Math Dialog…", self.open_pixelmath_dialog.emit, flat=True)
        lay.addWidget(pm)

        # EZ Scripts
        ez = CollapsibleSection("EZ Script Suite")
        ez.add_info("One-click processing presets (OSC, Narrowband, Luminance, etc.)")
        ez.add_run("⊞ Open EZ Scripts…", self.open_ez_scripts.emit, flat=True)
        lay.addWidget(ez)

        # HDR
        hdr = CollapsibleSection("HDR Composition")
        hdr.add_info("Merge differently-exposed images for extended dynamic range.")
        hdr.add_run("⊞ Open HDR Dialog…", self.open_hdr_dialog.emit, flat=True)
        lay.addWidget(hdr)

        # Blink
        blink = CollapsibleSection("Blink Comparator")
        blink.add_info("Rapidly alternate between two images to spot differences.")
        btns = blink.add_btn_row([("Load A…", True), ("Load B…", True)])
        btns[0].clicked.connect(self.blink_load_a.emit)
        btns[1].clicked.connect(self.blink_load_b.emit)
        btns2 = blink.add_btn_row([("Current → A", True), ("Current → B", True)])
        btns2[0].clicked.connect(self.blink_use_current_as_a.emit)
        btns2[1].clicked.connect(self.blink_use_current_as_b.emit)
        self._blink_fps = blink.add_slider("FPS", 2, 1, 10, 1, 0)
        self._blink_fps.value_changed.connect(lambda v: self.blink_fps_changed.emit(int(v)))
        self._btn_blink_toggle = blink.add_run("▶ Start Blinking")
        self._blink_active = False
        self._btn_blink_toggle.clicked.connect(self._on_blink_toggle)
        lay.addWidget(blink)

        # Macros
        mac = CollapsibleSection("Macros / Scripting")
        mac.add_info("Record, edit, and replay processing sequences.")
        btns = mac.add_btn_row([("⏺ Record", True), ("⏹ Stop", True), ("▶ Play", True)])
        btns[0].clicked.connect(self.start_macro_recording.emit)
        btns[1].clicked.connect(self.stop_macro_recording.emit)
        btns[2].clicked.connect(self.play_macro.emit)
        btns2 = mac.add_btn_row([("Save Macro…", True), ("Load Macro…", True)])
        btns2[0].clicked.connect(self.save_macro.emit)
        btns2[1].clicked.connect(self.load_macro.emit)
        lay.addWidget(mac)

        # Processing History
        ph = CollapsibleSection("Processing History (Non-Destructive)")
        ph.add_info("View and edit processing steps. Changes cascade and invalidate downstream steps.")
        ph.add_run("⊞ Open Processing History…", self.open_processing_graph.emit, flat=True)
        lay.addWidget(ph)

        # Analysis
        an = CollapsibleSection("Analysis Tools")
        an.add_info("Diagnose optical quality (FWHM, tilt, aberrations).")
        btns = an.add_btn_row([("FWHM Map", True), ("Tilt/Aberrations", True), ("Photometry", True)])
        btns[0].clicked.connect(self.open_analysis_fwhm.emit)
        btns[1].clicked.connect(self.open_analysis_tilt.emit)
        btns[2].clicked.connect(self.open_analysis_photometry.emit)
        lay.addWidget(an)

        # Python Console
        con = CollapsibleSection("Python Console")
        con.add_info("Full Python access to the Astraios core API.")
        con.add_run("⊞ Open Python Console…", self.open_python_console.emit, flat=True)
        lay.addWidget(con)

        # Overlays
        ov = CollapsibleSection("Overlays")
        ov.add_info("Toggle WCS, DSO, and constellation overlays on the canvas.")
        btns = ov.add_btn_row([("WCS Overlay", False), ("DSO Annotations", False), ("Constellations", False)])
        for b in btns:
            b.setCheckable(True)
        btns[0].clicked.connect(self.toggle_wcs_overlay)
        btns[1].clicked.connect(self.toggle_dso_overlay)
        btns[2].clicked.connect(self.toggle_constellation_overlay)
        lay.addWidget(ov)

        # Statistics
        stats = CollapsibleSection("Image Statistics")
        stats.add_info("Mean, median, SD, min, max, histogram percentiles.")
        stats.add_run("⊞ Show Statistics…", self.show_image_statistics.emit, flat=True)
        lay.addWidget(stats)

        # FITS Header
        hdr = CollapsibleSection("FITS Header")
        hdr.add_info("View and edit image FITS metadata.")
        hdr.add_run("⊞ Edit FITS Header…", self.edit_fits_header.emit, flat=True)
        lay.addWidget(hdr)

        self._tabs.addTab(scrollable_tab(lay), "⚙  Utility")

    # ── Internal helpers ──────────────────────────────────

    def _populate_spcc_cameras(self) -> None:
        """Load all cameras from the equipment database into the SPCC camera combo."""
        try:
            from astraios.core.equipment import load_camera_database
            cameras = load_camera_database()
            self._spcc_camera_combo.clear()
            for cam in cameras:
                self._spcc_camera_combo.addItem(cam.name)
        except Exception:
            pass  # keep the hardcoded defaults if database fails to load

    def _fire_preview(self, tool_name: str, check: "QCheckBox") -> None:
        if check.isChecked():
            self.preview_requested.emit(tool_name)

    def _emit_clip_points(self):
        self.clip_points_changed.emit(
            float(self._ht_black_spin.value()),
            float(self._ht_white_spin.value()),
        )

    def _on_blink_toggle(self):
        self._blink_active = not self._blink_active
        self.blink_toggle.emit(self._blink_active)
        self._btn_blink_toggle.setText(
            "⏹ Stop Blinking" if self._blink_active else "▶ Start Blinking"
        )
        self._btn_blink_toggle.setStyleSheet(
            self._btn_blink_toggle.styleSheet().replace(
                ACCENT, "#c93030" if self._blink_active else ACCENT
            )
        )

    # ── Public setters (called from main_window) ──────────

    def set_calibration_status(
        self,
        bias: str | None,
        dark: str | None,
        flat: str | None,
    ) -> None:
        self._cal_bias_label.setText(f"Bias: {bias or 'none'}")
        self._cal_dark_label.setText(f"Dark: {dark or 'none'}")
        self._cal_flat_label.setText(f"Flat: {flat or 'none'}")

    def set_bg_sample_count(self, n: int) -> None:
        self._bg_sample_label.setText(f"{n} manual sample{'s' if n != 1 else ''}")

    def set_psf_result(self, fwhm: float, fwhm_x: float, fwhm_y: float,
                       ellipticity: float, theta: float,
                       n_stars: int, fwhm_std: float) -> None:
        updates = {
            "FWHM": f"{fwhm:.2f} px",
            "FWHM X": f"{fwhm_x:.2f} px",
            "FWHM Y": f"{fwhm_y:.2f} px",
            "Ellipticity": f"{ellipticity:.3f}",
            "Rotation": f"{theta:.1f}°",
            "Stars used": str(n_stars),
            "FWHM σ": f"{fwhm_std:.2f} px",
        }
        for key, val in updates.items():
            if key in self._psf_result_labels:
                self._psf_result_labels[key].setText(val)

    def set_crop_draw_active(self, active: bool) -> None:
        """Sync the Draw on Image button with the canvas crop mode state."""
        self._btn_crop_draw.blockSignals(True)
        self._btn_crop_draw.setChecked(active)
        self._btn_crop_draw.blockSignals(False)

    def set_crop_from_rect(self, x: int, y: int, w: int, h: int) -> None:
        self._crop_x_spin.setValue(x)
        self._crop_y_spin.setValue(y)
        self._crop_w_spin.setValue(w)
        self._crop_h_spin.setValue(h)

    def set_subframe_count(self, n_selected: int, n_total: int) -> None:
        if n_selected == 0:
            self._subframe_count_label.setText("No subframe selection active")
        else:
            self._subframe_count_label.setText(
                f"{n_selected} / {n_total} frames selected"
            )

    def add_multi_session(self, label: str) -> None:
        self._ms_session_list.addItem(label)
        self._btn_ms_stack.setEnabled(self._ms_session_list.count() > 0)

    def clear_multi_sessions(self) -> None:
        self._ms_session_list.clear()
        self._btn_ms_stack.setEnabled(False)

    # ── Public getters (called from main_window) ──────────

    def _parse_norm(self, text: str) -> NormalizationMethod:
        nmap = {
            "Additive + Scaling": NormalizationMethod.ADDITIVE_SCALING,
            # "Linear Fit" is a rejection method, not a normalization one;
            # normalize_stack_linear_fit() is just an alias for ADDITIVE_SCALING.
            "Linear Fit":        NormalizationMethod.ADDITIVE_SCALING,
            "Local":             NormalizationMethod.LOCAL,
            "Additive":          NormalizationMethod.ADDITIVE,
            "Multiplicative":    NormalizationMethod.MULTIPLICATIVE,
            "None":              NormalizationMethod.NONE,
        }
        return nmap.get(text, NormalizationMethod.ADDITIVE_SCALING)

    def get_stacking_params(self) -> StackingParams:
        rejection_map = {
            "Sigma Clipping":     RejectionMethod.SIGMA_CLIP,
            "Winsorized Sigma":   RejectionMethod.WINSORIZED_SIGMA,
            "Linear Fit":        RejectionMethod.LINEAR_FIT,
            "Percentile Clip":   RejectionMethod.PERCENTILE_CLIP,
            "ESD (Generalized)": RejectionMethod.ESD,
            "Min/Max":           RejectionMethod.MIN_MAX,
            "None":              RejectionMethod.NONE,
        }
        integ_map = {
            "Average":          IntegrationMethod.AVERAGE,
            "Median":           IntegrationMethod.MEDIAN,
            "Weighted Average": IntegrationMethod.WEIGHTED_AVERAGE,
        }
        kappa = float(self._kappa_spin.value())
        return StackingParams(
            rejection=rejection_map.get(
                self._rejection_combo.currentText(), RejectionMethod.SIGMA_CLIP
            ),
            integration=integ_map.get(
                self._integration_combo.currentText(), IntegrationMethod.AVERAGE
            ),
            normalization=self._parse_norm(self._norm_combo.currentText()),
            kappa_low=kappa,
            kappa_high=kappa,
        )

    def get_alignment_params(self) -> dict:
        mode_map = {
            "Star (1-Pass)":    RegistrationMode.STAR_1_PASS,
            "Star (2-Pass)":    RegistrationMode.STAR_2_PASS,
            "Triangle Match":   RegistrationMode.TRIANGLE,
            "FFT Translation":  RegistrationMode.FFT_TRANSLATION,
            "Comet":            RegistrationMode.COMET,
        }
        ref_map = {
            "Auto (best quality)": -1,   # -1 = auto-select highest-variance frame
            "First frame":          0,
            "Last frame":          -2,   # -2 = last frame (special sentinel in stacking.py)
            "Specific frame #":     0,
        }
        mode = mode_map.get(
            self._reg_mode_combo.currentText(), RegistrationMode.STAR_2_PASS
        )
        ref_idx = ref_map.get(self._ref_frame_combo.currentText(), 0)
        return {
            "mode": mode,
            "reference_frame_index": ref_idx,
            "star_sensitivity": float(self._star_sens_spin.value()),
            "max_shift": int(self._max_shift_spin.value()),
            "ransac_threshold": float(self._ransac_thresh_spin.value()),
        }

    def get_stretch_params(self) -> StretchParams:
        return StretchParams(
            midtone=self._midtone_slider.value(),
            shadow_clip=float(self._shadow_spin.value()),
            linked=self._linked_check.isChecked(),
        )

    def get_ghs_params(self) -> GHSParams:
        return GHSParams(
            D=float(self._ghs_d_spin.value()),
            b=float(self._ghs_b_spin.value()),
            SP=float(self._ghs_sp_spin.value()),
            shadow_protection=self._ghs_shadow_slider.value(),
            highlight_protection=self._ghs_highlight_slider.value(),
        )

    def get_arcsinh_params(self) -> ArcsinhStretchParams:
        return ArcsinhStretchParams(
            stretch_factor=float(self._arcsinh_factor_spin.value()),
            black_point=float(self._arcsinh_bp_spin.value()),
            linked=self._arcsinh_linked_check.isChecked(),
        )

    def get_histogram_transform_params(self) -> HistogramTransformParams:
        return HistogramTransformParams(
            black_point=float(self._ht_black_spin.value()),
            midtone=self._ht_midtone_slider.value(),
            white_point=float(self._ht_white_spin.value()),
        )

    def reset_histogram_transform_params(self) -> None:
        self._ht_black_spin.setValue(0.0)
        self._ht_midtone_slider.setValue(0.5)
        self._ht_white_spin.setValue(1.0)

    def get_background_params(self, manual_points: list | None = None) -> BackgroundParams:
        return BackgroundParams(
            grid_size=int(self._bg_grid_spin.value()),
            polynomial_order=int(self._bg_order_spin.value()),
            per_pixel_sigma=self._bg_tolerance.value(),
            manual_points=manual_points or [],
        )

    def get_background_neutralization_params(self) -> BackgroundNeutralizationParams:
        return BackgroundNeutralizationParams(
            percentile=self._bn_percentile.value(),
            amount=self._bn_amount.value(),
            protect_bright=self._bn_protect.value(),
        )

    def get_cosmetic_params(self) -> CosmeticParams:
        return CosmeticParams(
            hot_sigma=self._hot_sigma.value(),
            cold_sigma=self._cold_sigma.value(),
            detect_dead=self._dead_pixel_check.isChecked(),
        )

    def get_vignette_params(self) -> VignetteParams:
        return VignetteParams(
            strength=self._vignette_amount.value(),
            radius=self._vignette_radius.value(),
        )

    def get_banding_params(self) -> BandingParams:
        direction = self._banding_dir_combo.currentText().lower()
        return BandingParams(
            horizontal=(direction in ("horizontal", "both")),
            vertical=(direction in ("vertical", "both")),
            amount=self._banding_amount.value(),
        )

    def get_ai_denoise_params(self) -> AIDenoiseParams:
        tile_map = {"128": 128, "256": 256, "512": 512, "Full": 0}
        return AIDenoiseParams(
            strength=self._ai_denoise_strength.value(),
            tile_size=tile_map.get(self._ai_tile_combo.currentText(), 256),
            protect_stars=0.8 if self._ai_star_protect.isChecked() else 0.0,
        )

    def get_ai_denoise_backend(self) -> str:
        return self._ai_denoise_backend.currentText()

    def get_ai_sharpen_backend(self) -> str:
        return self._ai_sharpen_backend.currentText()

    def get_cosmic_clarity_denoise_params(self):
        from astraios.ai.inference.cosmic_clarity import CosmicClarityParams
        tile_map = {"128": 128, "256": 256, "512": 512, "Full": 0}
        return CosmicClarityParams(
            model="denoise",
            strength=self._ai_denoise_strength.value(),
            tile_size=tile_map.get(self._ai_tile_combo.currentText(), 256),
            keep_original_size=True,
        )

    def get_cosmic_clarity_sharpen_params(self):
        from astraios.ai.inference.cosmic_clarity import CosmicClarityParams
        return CosmicClarityParams(
            model="sharpen",
            strength=self._ai_sharpen_strength.value(),
            tile_size=512,
            keep_original_size=True,
        )

    def get_ai_sharpen_params(self):
        from astraios.ai.inference.sharpen import AISharpenParams
        return AISharpenParams(
            strength=self._ai_sharpen_strength.value(),
            tile_size=512,
        )

    def get_deconvolution_params(self) -> "DeconvolutionParams | SpatialDeconvParams":
        method = self._deconv_method_combo.currentText()
        iters = int(self._deconv_iter.value())
        reg = float(self._deconv_reg.value())
        if method == "Blind (Spatial)":
            return SpatialDeconvParams(
                iterations=iters,
                regularization=reg,
            )
        return DeconvolutionParams(
            psf_fwhm=float(self._deconv_psf_spin.value()),
            iterations=iters,
            regularization=reg,
            deringing=self._deconv_deringing.isChecked(),
            deringing_amount=self._deconv_dering_amt.value(),
        )

    def set_psf_fwhm(self, fwhm: float) -> None:
        """Auto-populate the PSF FWHM field from a Measure PSF result."""
        self._deconv_psf_spin.setValue(round(fwhm, 1))

    def is_tgv_denoise_selected(self) -> bool:
        """Check if TGV Denoise is the selected method."""
        return self._denoise_method_combo.currentText() == "TGV Denoise"

    def get_tgv_params(self):
        """Return TGVParams from the current slider values."""
        from astraios.core.tgv_denoise import TGVParams
        return TGVParams(
            strength=self._denoise_amount.value(),
            n_iter=150,
        )

    def get_denoise_params(self) -> DenoiseParams:
        from astraios.core.denoise import DenoiseMethod
        method_map = {
            "NLM (Non-Local Means)": DenoiseMethod.NLM,
            "Wavelet Denoise": DenoiseMethod.WAVELET,
            "TGV Denoise": DenoiseMethod.TGV,
            "Median Filter": DenoiseMethod.MEDIAN,
        }
        method = method_map.get(self._denoise_method_combo.currentText(), DenoiseMethod.WAVELET)
        return DenoiseParams(
            method=method,
            strength=self._denoise_amount.value(),
            detail_preservation=self._denoise_lum.value(),
            chrominance_only=(self._denoise_chrom.value() > 0.5),
        )

    def set_denoise_amount(self, value: float) -> None:
        """Set the denoise Amount slider (used by the Auto measurement)."""
        self._denoise_amount.setValue(value)

    def set_denoise_noise_readout(self, sigma: float, snr: float) -> None:
        """Display the measured noise sigma / SNR under the denoise controls."""
        self._denoise_noise_label.setText(f"Noise σ={sigma:.4f}   SNR={snr:.1f}")

    def get_frequency_separation_params(self):
        from astraios.core.frequency_separation import (
            FrequencySeparationParams,
            SeparationMethod,
        )

        method = (
            SeparationMethod.DIVIDE
            if self._fs_method.currentText().startswith("Divide")
            else SeparationMethod.SUBTRACT
        )
        return FrequencySeparationParams(
            sigma=self._fs_sigma.value(),
            method=method,
            hf_boost=self._fs_hf_boost.value(),
            lf_smooth=self._fs_lf_smooth.value(),
        )

    def get_star_stretch_params(self):
        from astraios.core.star_stretch import StarStretchParams

        return StarStretchParams(
            amount=self._star_stretch_amount.value(),
            color_boost=self._star_stretch_color.value(),
        )

    def get_statistical_stretch_params(self):
        from astraios.core.stretch import StatisticalStretchParams

        return StatisticalStretchParams(
            target_median=self._statstretch_target.value(),
            shadow_clip=self._statstretch_shadow.value(),
            linked=self._statstretch_linked.isChecked(),
        )

    def get_star_reduction_params(self) -> StarReductionParams:
        from astraios.core.morphology import StructuringElement
        kernel_map = {
            "Elliptical": StructuringElement.CIRCLE,
            "Circular": StructuringElement.CIRCLE,
            "Square": StructuringElement.SQUARE,
            "Diamond": StructuringElement.DIAMOND,
        }
        return StarReductionParams(
            amount=self._star_reduction_amount.value() / 100.0,
            iterations=int(self._star_reduction_iters.value()),
            protect_core=self._star_reduction_protect.isChecked(),
            kernel_type=kernel_map.get(self._star_reduction_kernel.currentText(), StructuringElement.CIRCLE),
        )

    def get_unsharp_mask_params(self) -> UnsharpMaskParams:
        return UnsharpMaskParams(
            radius=self._um_radius.value(),
            amount=self._um_amount.value(),
            threshold=self._um_threshold.value(),
        )

    def get_local_contrast_params(self) -> LocalContrastParams:
        return LocalContrastParams(
            clip_limit=self._clahe_clip.value(),
            tile_size=int(self._clahe_tiles.value()),
        )

    def get_scnr_params(self) -> SCNRParams:
        from astraios.core.color_tools import SCNRMethod
        method_map = {
            "Average Neutral": SCNRMethod.AVERAGE_NEUTRAL,
            "Maximum Neutral": SCNRMethod.MAXIMUM_NEUTRAL,
        }
        return SCNRParams(
            method=method_map.get(
                self._scnr_method_combo.currentText(), SCNRMethod.AVERAGE_NEUTRAL
            ),
            amount=self._scnr_amount.value(),
        )

    def get_color_adjust_params(self) -> ColorAdjustParams:
        # ColorAdjustParams fields are saturation (multiplier, 1.0=neutral),
        # hue_shift (degrees) and vibrance (0-1). The sliders are centred on 0.
        return ColorAdjustParams(
            hue_shift=float(self._hue_slider.value()),
            saturation=1.0 + self._sat_slider.value() / 100.0,
            vibrance=max(0.0, self._vibrance_slider.value() / 100.0),
        )

    def get_curves_params(self) -> CurvesParams:
        # CurvesParams holds a CurvePoints per channel (master/red/green/blue);
        # apply the editor's points to the channel selected in the combo.
        from astraios.core.curves import CurvePoints

        cp = CurvePoints(points=list(self._curve_editor.get_points()))
        params = CurvesParams()
        channel = self._curve_channel_combo.currentText()
        if channel == "Red":
            params.red = cp
        elif channel == "Green":
            params.green = cp
        elif channel == "Blue":
            params.blue = cp
        else:  # "Master (L)"
            params.master = cp
        return params

    def get_wavelet_params(self) -> WaveletParams:
        n = int(self._wavelet_layers.value())
        weights = [s.value() for s in self._wavelet_layer_sliders[:n]]
        return WaveletParams(
            n_scales=n,
            scale_weights=weights,
        )

    def get_crop_params(self) -> CropParams:
        # CropParams uses width/height == 0 to mean "full remaining"; the old
        # ``or None`` turned 0 into None and broke ``width > 0`` comparisons.
        return CropParams(
            x=int(self._crop_x_spin.value()),
            y=int(self._crop_y_spin.value()),
            width=int(self._crop_w_spin.value()),
            height=int(self._crop_h_spin.value()),
        )

    def get_rotate_params(self) -> RotateParams:
        from astraios.core.transforms import RotateAngle

        angle_map = {
            "90° CW": RotateAngle.CW_90,
            "180°": RotateAngle.CW_180,
            "270° CW": RotateAngle.CW_270,
        }
        text = self._rotate_combo.currentText()
        if text in angle_map:
            return RotateParams(
                angle=angle_map[text], expand=self._rotate_expand_check.isChecked()
            )
        return RotateParams(
            angle=RotateAngle.ARBITRARY,
            arbitrary_degrees=float(self._rotate_angle_spin.value()),
            expand=self._rotate_expand_check.isChecked(),
        )

    def get_flip_params(self) -> FlipParams:
        from astraios.core.transforms import FlipAxis

        axis_map = {
            "horizontal": FlipAxis.HORIZONTAL,
            "vertical": FlipAxis.VERTICAL,
            "both": FlipAxis.BOTH,
        }
        return FlipParams(
            axis=axis_map.get(self._flip_combo.currentText().lower(), FlipAxis.HORIZONTAL)
        )

    def get_resize_params(self) -> ResizeParams:
        return ResizeParams(
            scale=float(self._resize_scale_spin.value()),
            interpolation=self._resize_interp_combo.currentText(),
        )

    def get_bin_params(self) -> BinParams:
        from astraios.core.transforms import BinMode

        factor_map = {"2x2": 2, "3x3": 3, "4x4": 4}
        mode_map = {"average": BinMode.AVERAGE, "sum": BinMode.SUM}
        return BinParams(
            factor=factor_map.get(self._bin_factor_combo.currentText(), 2),
            mode=mode_map.get(self._bin_mode_combo.currentText().lower(), BinMode.AVERAGE),
        )

    def get_abe_params(self) -> ABEParams:
        return ABEParams(
            grid_size=int(self._abe_grid_spin.value()),
            model_type=self._abe_model_combo.currentText(),
            polynomial_degree=int(self._abe_degree_spin.value()),
            rbf_kernel=self._abe_kernel_combo.currentText(),
            correction_mode=self._abe_mode_combo.currentText().lower(),
        )

    def get_morphology_params(self) -> MorphologyParams:
        from astraios.core.morphology import MorphOp, StructuringElement

        op_map = {
            "Erosion": MorphOp.ERODE, "Dilation": MorphOp.DILATE,
            "Opening": MorphOp.OPEN, "Closing": MorphOp.CLOSE,
            "Gradient": MorphOp.DILATE,
        }
        el_map = {
            "Disk": StructuringElement.CIRCLE, "Square": StructuringElement.SQUARE,
            "Diamond": StructuringElement.DIAMOND,
        }
        return MorphologyParams(
            operation=op_map.get(self._morph_op.currentText(), MorphOp.ERODE),
            element=el_map.get(self._morph_kernel.currentText(), StructuringElement.CIRCLE),
            iterations=int(self._morph_iters.value()),
        )

    def get_median_filter_params(self):
        from astraios.core.filters import MedianFilterParams

        # Median strength rides on the denoise "Amount" slider (no dedicated UI).
        kernel = 3 + 2 * int(round(self._denoise_amount.value() * 3))  # 3,5,7,9
        return MedianFilterParams(kernel_size=kernel)

    def get_mlt_params(self) -> WaveletParams:
        # MLT (multiscale linear transform) reuses the wavelet controls.
        return self.get_wavelet_params()

    def get_ca_params(self):
        from astraios.core.chromatic_aberration import CAParams

        return CAParams(auto_detect=True)

    def reset_stretch_params(self) -> None:
        self._midtone_slider.setValue(0.25)

    def reset_ghs_params(self) -> None:
        pass  # GHS sliders intentionally retain their values after applying

    def get_color_calibration_params(self) -> ColorCalibrationParams:
        method_map = {
            "Background reference": "average",
            "Photometric (SPCC)": "G2V",
            "Manual RGB": "custom",
        }
        white_ref = method_map.get(self._cc_method_combo.currentText(), "average")
        custom_rgb = (
            float(self._cc_r_spin.value()),
            float(self._cc_g_spin.value()),
            float(self._cc_b_spin.value()),
        )
        return ColorCalibrationParams(white_reference=white_ref, custom_rgb=custom_rgb)

    def get_drizzle_params(self) -> tuple[bool, "DrizzleParams"]:
        from astraios.core.drizzle import DrizzleParams
        enabled = self._drizzle_check.isChecked()
        text = self._drizzle_scale_combo.currentText()
        scale = 3 if text.startswith("3") else 2
        params = DrizzleParams(scale=scale, drop_shrink=float(self._drizzle_drop_spin.value()))
        return (enabled, params)

    def get_spcc_params(self):
        """Return SPCCParams from the SPCC UI section."""
        from astraios.core.spcc import SPCCParams
        filter_map = {
            "Broadband (L/R/G/B)": "Mono + Baader LRGB",
            "Narrowband Ha/OIII/SII": "OSC (no filter)",
            "Custom": "OSC (no filter)",
        }
        filter_name = filter_map.get(self._spcc_filter_combo.currentText(), "OSC (no filter)")
        return SPCCParams(filter_name=filter_name)

    def _toggle_cc_manual_rgb(self):
        visible = self._cc_method_combo.currentText() == "Manual RGB"
        for i in range(self._cc_rgb_row.count()):
            item = self._cc_rgb_row.itemAt(i)
            if item and item.layout():
                for j in range(item.layout().count()):
                    w = item.layout().itemAt(j).widget()
                    if w:
                        w.setVisible(visible)

    def _on_save_preset(self):
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self, "Save Preset",
            "Batch presets can be saved via the Batch Processing dialog.\n"
            "Per-section preset support coming in a future update.",
        )

    def _on_load_preset(self):
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self, "Load Preset",
            "Batch presets can be loaded via the Batch Processing dialog.\n"
            "Per-section preset support coming in a future update.",
        )

    def get_pcc_params(self) -> dict:
        solver_map = {"Auto": "auto", "ASTAP": "astap", "Astrometry.net": "astrometry_net"}
        ra = float(self._pcc_ra_spin.value())
        dec = float(self._pcc_dec_spin.value())
        return {
            "solver": solver_map.get(self._pcc_solver_combo.currentText(), "auto"),
            "ra_hint": ra if ra != 0.0 else None,
            "dec_hint": dec if dec != 0.0 else None,
        }

    # ── Compatibility properties used by main_window ──────
    @property
    def split_preview_enabled(self) -> bool:
        return self._split_check.isChecked()

    @property
    def curve_editor(self) -> "CurveEditor":
        return self._curve_editor

    @property
    def curves_histogram_visible(self) -> bool:
        return self._curves_histogram_check.isChecked()

    @property
    def current_curve_channel(self) -> int:
        return self._curve_channel_combo.currentIndex()
