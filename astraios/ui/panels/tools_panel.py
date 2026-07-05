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
    run_multiframe_deconv    = pyqtSignal()
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
    run_background_grain     = pyqtSignal()
    request_auto_denoise     = pyqtSignal()
    run_frequency_separation = pyqtSignal()
    run_star_reduction       = pyqtSignal()
    run_fx                   = pyqtSignal()
    run_diffraction_spikes   = pyqtSignal()
    run_sat_chroma           = pyqtSignal()
    run_halo_reduction       = pyqtSignal()
    run_wavescale_hdr        = pyqtSignal()
    run_wavescale_dark       = pyqtSignal()
    run_texture_clarity      = pyqtSignal()
    run_selective_color      = pyqtSignal()
    run_selective_luma       = pyqtSignal()
    run_pedestal             = pyqtSignal()
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
    blemish_mode_toggled     = pyqtSignal(bool)
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
        self._build_effects_tab()
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
        cal = CollapsibleSection(
            "Calibration", accent=True,
            help_text="Creates master bias, dark, and flat frames from your raw "
                      "calibration sub-frames (or lets you load pre-made masters), "
                      "then applies them to your light frames. Bias removes fixed "
                      "read noise, darks remove thermal noise and hot pixels, and "
                      "flats correct vignetting and dust shadows. Run this first, "
                      "before registration and stacking.",
        )
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
        sb = CollapsibleSection(
            "SuperBias",
            help_text="Builds a lower-noise master bias from your bias sub-frames by "
                      "separating out the sensor's fixed column pattern from the "
                      "random per-pixel read noise, then recombining them — the "
                      "result is as clean as averaging hundreds of plain bias frames "
                      "from as few as 15-20. Use it instead of a plain bias master "
                      "when you only have a handful of bias frames, or your camera "
                      "shows banding.",
        )
        sb.add_info("Statistically optimized bias — reduces read noise using spatial redundancy.")
        sb.add_run("▶ Create SuperBias", self.run_superbias.emit)
        lay.addWidget(sb)

        # Cosmetic
        cos = CollapsibleSection(
            "Cosmetic Correction",
            help_text="Finds individual defective sensor pixels — hot (stuck "
                      "bright), cold (stuck dark), and dead (stuck at zero) — by "
                      "comparing each pixel to its local neighborhood, then "
                      "replaces them with the local median. A matching dark frame "
                      "already removes most hot pixels during calibration; this "
                      "mops up what is left, or covers cameras with no darks.",
        )
        cos.add_info("Detect and remove hot, cold, and dead pixels.")
        self._hot_sigma = cos.add_slider(
            "Hot sigma", 5.0, 1.0, 20.0, 0.5, 1,
            help_text="How many times brighter than its neighbors a pixel must be "
                      "to be flagged as hot. Lower catches more pixels (more "
                      "aggressive); higher flags only the most obvious defects.",
        )
        self._cold_sigma = cos.add_slider(
            "Cold sigma", 5.0, 1.0, 20.0, 0.5, 1,
            help_text="How many times darker than its neighbors a pixel must be "
                      "to be flagged as cold. Lower catches more pixels; higher "
                      "flags only the most obvious defects.",
        )
        self._dead_pixel_check = cos.add_check(
            "Detect dead pixels (value=0)", True,
            help_text="Also flags any pixel whose value is exactly zero as dead "
                      "and repairs it with the local median.",
        )
        cos.add_run("▶ Apply Cosmetic Correction", self.run_cosmetic.emit)
        lay.addWidget(cos)

        # Debayer
        deb = CollapsibleSection(
            "Debayer (OSC / Color Camera)",
            help_text="Converts a raw one-shot-color camera's Bayer mosaic — one "
                      "red, green, or blue filter over each pixel — into a full "
                      "RGB image by interpolating the missing colors at every "
                      "pixel. Not needed for mono cameras. Run this once, right "
                      "after loading OSC raw frames, before calibration and "
                      "stacking, unless your capture software already debayered.",
        )
        deb.add_info("Convert raw Bayer mosaic to color image.")
        self._debayer_pattern_combo = deb.add_combo(
            "Pattern",
            ["Auto-detect", "RGGB", "BGGR", "GRBG", "GBRG"],
            help_text="The color filter layout on the sensor. Auto-detect reads "
                      "it from the FITS header; if the result looks tinted "
                      "magenta or green, try the other patterns.",
        )
        self._debayer_method_combo = deb.add_combo(
            "Method",
            ["VNG (best quality)", "Edge-Aware (EA)",
             "Superpixel (2× bin)", "Bilinear (fastest)"],
            help_text="VNG and Edge-Aware interpolate the missing colors most "
                      "accurately (recommended). Superpixel bins each 2x2 color "
                      "quad into one pixel — half the resolution but no "
                      "interpolation artifacts, good for very noisy or "
                      "underexposed subs. Bilinear is fastest but softer.",
        )
        deb.add_run("▶ Apply Debayer", self.run_debayer.emit)
        lay.addWidget(deb)

        # Pedestal (ported from Seti Astro Suite Pro, GPL-3.0)
        ped = CollapsibleSection(
            "Pedestal",
            help_text="Adds or removes a constant offset from the whole "
                      "image. Add a small pedestal before operations that "
                      "dislike negative pixels; remove one to restore the "
                      "true black level afterwards.",
        )
        self._ped_mode_combo = ped.add_combo(
            "Mode", ["Add", "Remove"],
            help_text="Add raises every pixel by the amount. Remove "
                      "subtracts the measured (or given) offset back out.",
        )
        self._ped_amount = ped.add_spin(
            "Amount", 0.0, 0.5, 0.0, 0.001, 4,
            help_text="Offset to add, in the 0-1 image scale. Typical "
                      "values are 0.001 to 0.01. In Remove mode, leave at 0 "
                      "to auto-detect the offset from the darkest pixels.",
        )
        self._ped_per_channel = ped.add_check(
            "Per channel", True,
            help_text="Ticked: each color channel is measured and treated "
                      "separately. Unticked: one global value for all.",
        )
        self._ped_clip = ped.add_check(
            "Clip to 0-1", True,
            help_text="Keeps the result inside the valid range. Untick "
                      "only if a following tool expects negative pixels.",
        )
        ped.add_run("▶ Apply Pedestal", self.run_pedestal.emit)
        lay.addWidget(ped)

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
        sub = CollapsibleSection(
            "Subframe Selector",
            help_text="Scores every light frame by star sharpness (FWHM), star "
                      "roundness (eccentricity), background noise (SNR), and star "
                      "count, then lets you reject the worst ones before "
                      "stacking. Clouds, wind gusts, and tracking blips usually "
                      "stand out as clear outliers. Run after calibration and "
                      "before Registration.",
        )
        sub.add_info("Score and reject frames by FWHM, eccentricity, SNR, star count.")
        self._subframe_count_label = sub.add_status_label("No subframe selection active")
        sub.add_run("⊞ Open Subframe Selector…", self.open_subframe_selector.emit, flat=True)
        lay.addWidget(sub)

        # Registration
        reg = CollapsibleSection(
            "Registration (Alignment)", accent=True,
            help_text="Detects stars in every frame and warps each one so its "
                      "stars line up with a reference frame, pixel for pixel, "
                      "before stacking. Run after calibration (and subframe "
                      "selection), before Integration.",
        )
        reg.add_info("Detect stars and align frames to a reference.")
        self._reg_mode_combo = reg.add_combo(
            "Mode",
            ["Star (1-Pass)", "Star (2-Pass)", "Triangle Match",
             "FFT Translation", "Comet"],
            "Star (2-Pass)",
            help_text="How frames are matched and aligned. Star (2-Pass) is the "
                      "accurate default: a star match plus a tight refinement "
                      "pass. Star (1-Pass) is faster, slightly less precise. "
                      "Triangle Match tolerates large rotation or scale "
                      "differences, e.g. combining sessions from different "
                      "equipment. FFT Translation only corrects x/y shift, not "
                      "rotation — fine for an alt-az mount with a field "
                      "de-rotator. Comet tracks a moving comet's nucleus "
                      "instead of the stars.",
        )
        self._star_sens_spin = reg.add_slider(
            "Star sensitivity", 5.0, 1.0, 20.0, 0.5, 1,
            help_text="How faint a star can be and still be used for alignment. "
                      "Higher finds more, fainter stars, useful on sparse "
                      "fields; lower keeps only the brightest, most reliable "
                      "stars, useful on noisy or trailed frames.",
        )
        self._max_shift_spin = reg.add_spin(
            "Max distance (px)", 10, 500, 50, 10,
            help_text="A star match is discarded if it implies a shift larger "
                      "than this many pixels. Raise it if the mount drifted a "
                      "lot between subs; lower it to reject bad matches faster.",
        )
        self._ransac_thresh_spin = reg.add_spin(
            "RANSAC threshold", 1.0, 10.0, 3.0, 0.5, 1,
            help_text="How far a star position may deviate from the fitted "
                      "alignment model and still count as a good match, in "
                      "pixels. Lower is stricter; higher tolerates noisier star "
                      "detections.",
        )
        self._ref_frame_combo = reg.add_combo(
            "Reference frame",
            ["Auto (best quality)", "First frame", "Last frame", "Specific frame #"],
            help_text="Which frame all the others are aligned to. Auto picks "
                      "the sharpest/highest-quality frame for you; First and "
                      "Last use a fixed choice; Specific frame # lets you pick "
                      "by number.",
        )
        # Index input for "Specific frame #" (1-based, shown only when chosen;
        # the option used to silently behave like "First frame").
        self._ref_frame_spin = styled_spin(1, 9999, 1, 1, 0, "")
        self._ref_frame_row = QWidget()
        self._ref_frame_row.setLayout(field_row(
            "Frame #", self._ref_frame_spin, 110,
            help_text="The 1-based frame number to align every other frame to.",
        ))
        self._ref_frame_row.setVisible(False)
        reg.add_widget(self._ref_frame_row)
        self._ref_frame_combo.currentTextChanged.connect(
            lambda t: self._ref_frame_row.setVisible(t == "Specific frame #")
        )
        reg.add_run("▶ Align Frames", self.run_alignment.emit, flat=True)
        lay.addWidget(reg)

        # Integration
        integ = CollapsibleSection(
            "Integration (Stacking)", accent=True,
            help_text="Combines all the aligned frames into one image, rejecting "
                      "outlier pixels — satellite trails, cosmic ray hits, plane "
                      "streaks — before averaging so they don't leave marks in "
                      "the final stack. Run this last, after Registration.",
        )
        integ.add_info("Combine aligned frames using rejection to increase SNR.")
        self._rejection_combo = integ.add_combo(
            "Rejection",
            ["Sigma Clipping", "Winsorized Sigma", "Linear Fit",
             "Percentile Clip", "ESD (Generalized)", "Min/Max", "None"],
            help_text="How outlier pixels (satellite trails, cosmic rays) are "
                      "thrown out before combining. Sigma Clipping rejects "
                      "pixels far from the per-pixel average, measured in units "
                      "of noise (sigma) — the standard choice. Winsorized Sigma "
                      "is more robust with few frames: outliers are pulled in to "
                      "the clip limit instead of dropped entirely. Linear Fit "
                      "models each pixel's brightness trend across frames before "
                      "clipping — good with many frames and changing sky glow. "
                      "Percentile Clip drops a fixed top/bottom percentage per "
                      "pixel. ESD is a statistical test that can catch several "
                      "trails through the same pixel. Min/Max simply drops the "
                      "highest and lowest values. None disables rejection "
                      "(fastest, but trails and cosmic rays stay in).",
        )
        self._norm_combo = integ.add_combo(
            "Normalization",
            ["Additive + Scaling", "Linear Fit", "Local", "Additive", "Multiplicative", "None"],
            current="Additive + Scaling",
            help_text="Matches brightness and background level between frames "
                      "before combining, so sky-glow differences between subs "
                      "(moonlight, transparency, altitude) don't show up as a "
                      "patchy stack. Additive + Scaling matches both offset and "
                      "gain robustly (recommended default). Local also corrects "
                      "background differences that vary across the frame, like a "
                      "light dome on one side. Additive matches offset only; "
                      "Multiplicative matches gain only; None applies no "
                      "correction, only safe for frames shot under identical "
                      "conditions.",
        )
        self._integration_combo = integ.add_combo(
            "Integration",
            ["Average", "Median", "Weighted Average"],
            help_text="How the surviving pixel values are combined into one. "
                      "Average (mean) gives the best signal-to-noise ratio. "
                      "Median is more robust if rejection missed an outlier, at "
                      "a small SNR cost. Weighted Average favors frames with "
                      "better measured quality (SNR/FWHM) over weaker ones.",
        )
        self._kappa_spin = integ.add_slider(
            "Kappa (σ)", 3.0, 0.5, 10.0, 0.1, 1,
            help_text="The sigma multiplier used for rejection: how many "
                      "noise-sigmas a pixel must deviate from the average before "
                      "it is thrown out. Lower rejects more aggressively (risks "
                      "removing real faint signal); higher keeps more data but "
                      "lets more trails and artifacts slip through.",
        )
        integ.add_run("▶ Stack Images", self.run_stacking.emit)
        lay.addWidget(integ)

        # Multi-Frame Deconvolution
        mfd = CollapsibleSection(
            "Multi-Frame Deconvolution",
            help_text="Jointly deconvolves every REGISTERED (aligned) frame "
                      "against one shared sharp image, instead of stacking "
                      "first and sharpening after — frames with a tighter PSF "
                      "or a cleaner background pull harder on the result, "
                      "typically resolving finer detail than stack-then-"
                      "sharpen. An alternative or complement to Integration; "
                      "needs at least 2 aligned frames and no plate solve.",
        )
        mfd.add_info("Joint Richardson-Lucy deconvolution across all aligned frames.")
        self._mfd_iterations_spin = mfd.add_spin(
            "Iterations", 1, 200, 20, 1,
            help_text="Maximum number of joint Richardson-Lucy iterations to "
                      "run. More iterations sharpen further but risk ringing "
                      "and take longer; Early stop below usually cuts this "
                      "short automatically.",
        )
        self._mfd_min_iterations_spin = mfd.add_spin(
            "Min iterations", 1, 50, 3, 1,
            help_text="Minimum iterations to run before Early stop is allowed "
                      "to trigger, so the solve doesn't quit before it has "
                      "made meaningful progress.",
        )
        self._mfd_rho_combo = mfd.add_combo(
            "Residual loss", ["Huber (robust)", "L2 (classic)"],
            help_text="How pixel residuals are weighted each iteration. Huber "
                      "(robust) downweights outliers — satellite trails, hot "
                      "pixels, residual noise — so they don't drag the solve "
                      "around; L2 (classic) is the textbook Richardson-Lucy "
                      "update with no outlier protection.",
        )
        self._mfd_color_mode_combo = mfd.add_combo(
            "Color mode", ["Luma (fast)", "Per-channel"],
            help_text="Luma (fast) deconvolves a single shared-luminance "
                      "plane — quicker, no per-channel color shift. "
                      "Per-channel solves R, G, and B independently with the "
                      "same per-frame PSF — slower, but corrects any "
                      "per-channel blur difference (e.g. atmospheric "
                      "dispersion).",
        )
        self._mfd_seed_mode_combo = mfd.add_combo(
            "Seed image", ["Robust (sigma-clip)", "Median", "Mean", "Integrated (current image)"],
            help_text="How the initial estimate is built before iterating. "
                      "Robust (sigma-clip) rejects outliers across frames "
                      "first — the safest default. Median and Mean are "
                      "simpler combines. Integrated (current image) starts "
                      "from whatever image is currently displayed (e.g. an "
                      "existing stack) instead of recombining the aligned "
                      "frames.",
        )
        self._mfd_kappa_spin = mfd.add_slider(
            "Kappa", 2.0, 1.0, 10.0, 0.1, 1,
            help_text="Clamps the per-pixel multiplicative update each "
                      "iteration to [1/kappa, kappa], limiting overshoot and "
                      "ringing. Lower is more conservative; higher lets the "
                      "solve move faster per iteration.",
        )
        self._mfd_relaxation_spin = mfd.add_slider(
            "Relaxation", 0.7, 0.1, 1.0, 0.05, 2,
            help_text="Damping factor blending each iteration's raw update "
                      "into the estimate. 1.0 takes the full step every "
                      "iteration (fastest, more prone to ringing); lower "
                      "values damp the update for a smoother, more stable "
                      "convergence.",
        )
        self._mfd_early_stop_check = mfd.add_check(
            "Early stop", True,
            help_text="Stops iterating once the per-iteration update size and "
                      "relative change both plateau, instead of always "
                      "running the full iteration count.",
        )
        self._mfd_super_res_combo = mfd.add_combo(
            "Super-resolution", ["1x (native)", "2x", "3x"],
            help_text="Drizzle-like output upsampling factor. 1x solves at "
                      "native resolution. 2x/3x reconstruct a finer output "
                      "grid from the same frames — sharper if your optics "
                      "undersample the stars, but slower and more VRAM-"
                      "hungry.",
        )
        self._mfd_low_vram_check = mfd.add_check(
            "Low VRAM mode",
            help_text="Releases the GPU memory cache after every frame "
                      "instead of only between iterations — slower, but keeps "
                      "peak VRAM use lower on constrained GPUs.",
        )
        mfd.add_run("▶ Run Multi-Frame Deconvolution", self.run_multiframe_deconv.emit)
        lay.addWidget(mfd)

        # Drizzle
        drz = CollapsibleSection(
            "Drizzle Integration",
            help_text="Reconstructs a higher-resolution stack from many "
                      "undersampled, dithered frames by mapping each input "
                      "pixel onto a finer output grid (the technique used on "
                      "Hubble data). Only helps if your subs were dithered — "
                      "moved slightly between exposures — and your optics "
                      "undersample the stars; otherwise it just makes a bigger, "
                      "noisier image.",
        )
        self._drizzle_check = drz.add_check(
            "Enable Drizzle",
            help_text="Use drizzle integration for the next stack instead of a "
                      "normal 1:1 combine.",
        )
        self._drizzle_scale_combo = drz.add_combo(
            "Output scale", ["2× (recommended)", "3×"],
            help_text="Output image size relative to the originals. Only go "
                      "beyond 2x if the data is significantly undersampled and "
                      "well dithered, or the result will look soft and grainy.",
        )
        self._drizzle_drop_spin = drz.add_slider(
            "Drop shrink", 0.7, 0.5, 1.0, 0.05, 2,
            help_text="Fraction of each input pixel's footprint (\"pixfrac\") "
                      "dropped onto the output grid. Smaller sharpens detail "
                      "further but needs more, well-dithered frames to avoid "
                      "gaps; values near 1.0 behave like a normal resample and "
                      "are safer with limited dithering.",
        )
        lay.addWidget(drz)

        # Multi-session
        ms = CollapsibleSection(
            "Multi-Session Integration",
            help_text="Stacks light frames captured with different telescopes, "
                      "cameras, or on different nights into one combined image, "
                      "normalizing each session's background level before "
                      "combining and optionally re-aligning each sub-stack to a "
                      "common reference.",
        )
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
            help_text="How much influence each session gets in the final "
                      "combine. SNR favors your cleanest sessions. Integration "
                      "time favors sessions with more total exposure. Equal "
                      "weight treats every session the same regardless of "
                      "quality or length.",
        )
        self._ms_normalize_check = ms.add_check(
            "Normalize background", True,
            help_text="Matches sky background level across sessions before "
                      "combining, so a brighter or hazier night does not skew "
                      "the result.",
        )
        self._ms_align_check     = ms.add_check(
            "Align sub-stacks", True,
            help_text="Re-registers each session's stack to a common reference "
                      "before the final combine, correcting for framing "
                      "differences between sessions (different mount, rotation, "
                      "field of view).",
        )
        self._btn_ms_stack = ms.add_run("▶ Stack All Sessions", self.run_multi_session.emit)
        self._btn_ms_stack.setEnabled(False)
        lay.addWidget(ms)

        # Live Stack
        live = CollapsibleSection(
            "Live Stack",
            help_text="Accumulates and displays frames in real time as your "
                      "camera captures them, so you can judge focus, framing, "
                      "and sky conditions during the session instead of after "
                      "it. Opens a separate live-view window.",
        )
        live.add_info("Real-time frame accumulation with live preview.")
        live.add_run("▶ Open Live Stack…", self.open_live_stack.emit, flat=True)
        lay.addWidget(live)

        # Batch
        batch = CollapsibleSection(
            "Batch Processing",
            help_text="Runs calibration, registration, and stacking unattended "
                      "over many target folders, or applies a saved processing "
                      "pipeline to a batch of already-processed images. Use it "
                      "once you have settings dialed in and want to repeat them "
                      "without babysitting each target.",
        )
        batch.add_info("Full calibration → registration → stacking pipeline.")
        batch.add_run("⊞ Batch Preprocess…", self.open_batch_preprocess.emit, flat=True)
        batch.add_info("Apply a pipeline to multiple processed images.")
        batch.add_run("⊞ Open Batch Dialog…", self.open_batch_dialog.emit, flat=True)
        lay.addWidget(batch)

        # Mosaic
        mosaic = CollapsibleSection(
            "Mosaic Stitching",
            help_text="Combines several overlapping imaging panels — a "
                      "multi-pane wide-field target — into one seamless image "
                      "by matching stars in the overlap regions and blending "
                      "the seams. Each panel should already be stacked.",
        )
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
        bg = CollapsibleSection(
            "Remove Gradient (Background Extraction)", accent=True,
            help_text="Models and subtracts large-scale light-pollution and "
                      "skyglow gradients using a grid of sampled background "
                      "points fit to a smooth polynomial surface. Run after "
                      "stacking, before stretching — gradients are far easier "
                      "to model while the data is still linear (unstretched).",
        )
        bg.add_info("Remove light pollution gradients.")
        self._bg_grid_spin  = bg.add_spin(
            "Grid size", 4, 32, 8,
            help_text="Number of sample points along each axis of the sampling "
                      "grid (NxN). More points can follow finer gradient shapes "
                      "but risk sampling into nebulosity.",
        )
        self._bg_order_spin = bg.add_spin(
            "Poly order", 1, 6, 3,
            help_text="Degree of the polynomial surface fit to the samples. "
                      "Higher orders follow more complex gradients but risk "
                      "eating into the target signal if pushed too high; start "
                      "at 2-3.",
        )

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

        self._bg_box_size_spin = bg.add_spin(
            "Box size (px)", 8, 256, 64, 8,
            help_text="Size, in pixels, of the measurement box placed at each "
                      "auto-grid sample point.",
        )
        self._bg_tolerance = bg.add_slider(
            "Tolerance (σ)", 2.5, 1.0, 5.0, 0.5, 1,
            help_text="How far a sample point's measured value may be from the "
                      "local median, in noise-sigma, before it is rejected as "
                      "contaminated by a star or nebulosity.",
        )
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
        abe = CollapsibleSection(
            "Light Pollution Removal (ABE)",
            help_text="A second gradient remover, using a grid of sample points "
                      "fit with a polynomial or radial-basis-function (RBF) "
                      "surface, then subtracted or divided out. Reach for this "
                      "instead of Remove Gradient when you need an RBF surface "
                      "for a complex, non-polynomial gradient, or a divisive "
                      "correction for flat-field-style errors.",
        )
        abe.add_info("Background extraction using polynomial or RBF surface fitting.")
        self._abe_grid_spin   = abe.add_spin(
            "Grid size", 5, 30, 10,
            help_text="Number of sample points per axis. More points follow "
                      "finer gradient detail but risk landing on target signal.",
        )
        self._abe_model_combo = abe.add_combo(
            "Model", ["Polynomial (recommended)", "RBF"],
            help_text="Polynomial fits a smooth mathematical surface across the "
                      "whole frame — stable and free of edge artifacts "
                      "(recommended default). RBF can follow complex, irregular "
                      "gradients but is more sensitive to how the sample points "
                      "are placed near the image edges.",
        )
        self._abe_degree_spin = abe.add_spin(
            "Poly degree", 1, 5, 2,
            help_text="Degree of the polynomial surface. Higher can follow "
                      "stronger gradients but risks eating into faint "
                      "nebulosity; start low and raise it only if a gradient "
                      "remains.",
        )
        self._abe_kernel_combo = abe.add_combo(
            "RBF kernel", ["Thin Plate Spline", "Multiquadric", "Gaussian"],
            help_text="Shape of the interpolation function used between sample "
                      "points when Model is set to RBF. Thin Plate Spline gives "
                      "the smoothest surface; Multiquadric and Gaussian are "
                      "alternatives for stubborn gradients.",
        )
        self._abe_mode_combo  = abe.add_combo(
            "Mode", ["Subtraction", "Division"],
            help_text="Subtraction removes an additive gradient — the usual "
                      "case for light pollution and skyglow. Division removes "
                      "a multiplicative gradient, such as flat-field-style "
                      "errors from a bad or missing flat.",
        )
        abe.add_run("▶ Run ABE", self.run_abe.emit)
        lay.addWidget(abe)

        # Background Neutralization (NEW)
        bn = CollapsibleSection(
            "Background Neutralization",
            help_text="Shifts the sky background to a neutral zero level, per "
                      "color channel, using the darkest percentile of pixels as "
                      "a reference. Removes a color cast left by light "
                      "pollution or an imperfect color balance. Run on linear "
                      "(unstretched) data, before or after color calibration, "
                      "whenever the sky background isn't a neutral black.",
        )
        bn.add_info(
            "Shift sky background to neutral zero per-channel. "
            "Equivalent to PixInsight BackgroundNeutralization (statistical mode)."
        )
        self._bn_percentile = bn.add_slider(
            "Percentile", 2.0, 0.5, 10.0, 0.5, 1,
            help_text="The darkest X% of pixels used to estimate the sky "
                      "background level per channel. Lower samples only the "
                      "very darkest sky — safer on frames with a lot of "
                      "extended nebulosity.",
        )
        self._bn_amount     = bn.add_slider(
            "Amount", 1.0, 0.0, 1.0, 0.05, 2,
            help_text="Blend strength of the correction. 1.0 applies the full "
                      "measured shift; lower values apply it partially.",
        )
        self._bn_protect    = bn.add_slider(
            "Protect bright", 0.5, 0.0, 1.0, 0.05, 2,
            help_text="Ignores pixels brighter than this fraction of the "
                      "image's peak value when measuring the background, so "
                      "bright nebulosity or stars don't skew the sky estimate.",
        )
        bn.add_run("▶ Apply Background Neutralization",
                   self.run_background_neutralization.emit)
        lay.addWidget(bn)

        # Vignette
        vig = CollapsibleSection(
            "Vignette Correction",
            help_text="Removes optical vignetting — radial darkening toward the "
                      "corners caused by the optical train — by generating a "
                      "synthetic radial brightness model and dividing it out. "
                      "Use this when you don't have (or trust) a real flat "
                      "frame; a proper flat calibration is usually more "
                      "accurate.",
        )
        vig.add_info("Remove optical vignetting toward image edges.")
        self._vignette_amount = vig.add_slider(
            "Amount", 0.3, 0.0, 1.0, 0.05, 2,
            help_text="Strength of the correction. Higher brightens the "
                      "corners more; too high can overcorrect and brighten them "
                      "past the center brightness.",
        )
        self._vignette_radius = vig.add_slider(
            "Radius", 0.8, 0.3, 1.0, 0.05, 2,
            help_text="How far from the center the vignetting model extends, "
                      "as a fraction of the frame. Smaller assumes the "
                      "darkening starts closer to the center.",
        )
        vig.add_run("▶ Correct Vignette", self.run_vignette_correction.emit)
        lay.addWidget(vig)

        # Banding
        band = CollapsibleSection(
            "Banding Reduction",
            help_text="Removes faint horizontal or vertical stripes caused by "
                      "CMOS sensor readout, by measuring and subtracting the "
                      "per-row or per-column offset relative to its neighbors.",
        )
        band.add_info("Remove horizontal/vertical banding from CMOS sensors.")
        self._banding_amount = band.add_slider(
            "Amount", 1.0, 0.1, 3.0, 0.1, 1,
            help_text="Strength of the correction. Higher removes more "
                      "banding but can also flatten very faint real signal "
                      "that happens to run along the same direction.",
        )
        self._banding_dir_combo = band.add_combo(
            "Direction", ["Horizontal", "Vertical", "Both"],
            help_text="Which banding direction to remove: Horizontal corrects "
                      "row-to-row offsets, Vertical corrects column-to-column "
                      "offsets, Both removes both directions.",
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
        aut = CollapsibleSection(
            "Auto-Stretch", accent=True,
            help_text="Applies a nonlinear midtone stretch to make faint "
                      "nebulosity visible on screen, computed from the image's "
                      "own statistics (background median and noise) — the same "
                      "idea as PixInsight's Screen Transfer Function. Usually "
                      "the first stretch you reach for on a freshly stacked, "
                      "linear image.",
        )
        aut.add_info("Statistical midtone stretch.")
        self._midtone_slider = aut.add_slider(
            "Midtone", 0.25, 0.01, 0.99, 0.01, 2, 0.25,
            help_text="Target midtone brightness after the stretch. Higher "
                      "pulls the midtones brighter, revealing more faint "
                      "detail at the cost of a busier-looking image; lower "
                      "keeps the image darker and more contrasty.",
        )
        self._midtone_slider.value_changed.connect(lambda _: self.stretch_params_changed.emit())
        self._shadow_spin = aut.add_spin(
            "Shadow clip", -10.0, 0.0, -2.8, 0.1, 1,
            help_text="Where the black point is set, in noise-sigma below the "
                      "background median. More negative sets black further "
                      "below the sky, keeping more of the noise floor visible; "
                      "less negative (closer to 0) clips more of the "
                      "background to pure black.",
        )
        self._linked_check = aut.add_check(
            "Link RGB channels", True,
            help_text="Stretch all three color channels with the same curve, "
                      "preserving color balance. Untick to stretch each "
                      "channel independently, which can shift colors.",
        )
        self._split_check  = aut.add_check("Show before/after preview")
        self._split_check.toggled.connect(lambda _: self.stretch_params_changed.emit())
        btns = aut.add_btn_row([("▶ Apply Stretch", False), ("Reset", True)])
        btns[0].clicked.connect(self.run_stretch.emit)
        btns[1].clicked.connect(lambda: self._midtone_slider.setValue(0.25))
        lay.addWidget(aut)

        # Statistical Stretch (target median)
        sst2 = CollapsibleSection(
            "Statistical Stretch",
            help_text="Like Auto-Stretch, but instead of setting the midtone "
                      "directly you pick the background brightness you want and "
                      "the midtone curve is solved for you. Simpler to reason "
                      "about: choose how dark the sky should look, done.",
        )
        sst2.add_info(
            "Stretch the background to a chosen target level — the midtone is "
            "solved for you. Lower target = darker background."
        )
        self._statstretch_target = sst2.add_slider(
            "Target median", 0.25, 0.05, 0.6, 0.01, 2,
            help_text="The background brightness wanted after stretching, as a "
                      "fraction of full scale. Lower keeps the sky darker and "
                      "more contrasty; higher lifts the background brighter, "
                      "revealing more faint signal but looking flatter.",
        )
        self._statstretch_shadow = sst2.add_spin(
            "Shadow clip", -10.0, 0.0, -2.8, 0.1, 1,
            help_text="Where the black point is set, in noise-sigma below the "
                      "background median. More negative keeps more of the "
                      "noise floor visible; less negative clips more of the "
                      "background to pure black.",
        )
        self._statstretch_linked = sst2.add_check(
            "Link RGB channels", True,
            help_text="Stretch all three color channels with the same curve, "
                      "preserving color balance. Untick to stretch each "
                      "channel independently, which can shift colors.",
        )
        self._statstretch_preview_check = sst2.add_check("Show before/after preview")
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
        arc = CollapsibleSection(
            "Arcsinh Stretch",
            help_text="A linear-to-arcsinh stretch (Lupton et al. 2004) that "
                      "compresses bright stars and galaxy cores less "
                      "aggressively than a log stretch, so stars keep their "
                      "color while faint nebulosity is still pulled up. A good "
                      "alternative to GHS or Auto-Stretch when stars are "
                      "washing out to white.",
        )
        arc.add_info(
            "Lupton et al. 2004 — linear-to-arcsinh ramp. Preserves star colours "
            "better than log; reveals faint nebulosity without blowing out stars."
        )
        self._arcsinh_factor_spin = arc.add_spin(
            "Stretch factor β", 0.1, 1000.0, 10.0, 1.0, 1,
            help_text="How aggressively mid-to-bright signal is compressed. "
                      "Higher pulls up faint nebulosity more strongly but "
                      "compresses bright stars and cores further toward white.",
        )
        self._arcsinh_bp_spin = arc.add_spin(
            "Black point", 0.0, 0.5, 0.0, 0.001, 4,
            help_text="Input level mapped to zero before stretching. Raise it "
                      "slightly if the background floor isn't perfectly black, "
                      "to remove residual skyglow before the stretch.",
        )
        self._arcsinh_linked_check = arc.add_check(
            "Linked RGB", True,
            help_text="Stretch all three color channels with the same curve, "
                      "preserving color balance. Untick to stretch each "
                      "channel independently, which can shift colors.",
        )
        self._arcsinh_preview_check = arc.add_check("Show before/after preview")
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
        sst = CollapsibleSection(
            "Star Stretch",
            help_text="A colour-preserving arcsinh-style stretch tuned for "
                      "star-only layers: lifts faint stars while keeping their "
                      "natural hue, instead of blowing them white. Best run on "
                      "a star-only image made with Star Removal, then screened "
                      "back over the starless background.",
        )
        sst.add_info(
            "Colour-preserving stretch for star layers — lifts faint stars while "
            "keeping their hue. Best run on an extracted star image, then screened back."
        )
        self._star_stretch_amount = sst.add_slider(
            "Amount", 0.2, 0.0, 1.0, 0.05, 2,
            help_text="Stretch strength. Higher lifts fainter stars more; "
                      "internally this raises the arcsinh stretch factor from "
                      "barely any lift toward a very strong pull.",
        )
        self._star_stretch_color = sst.add_slider(
            "Colour boost", 1.0, 0.0, 3.0, 0.05, 2,
            help_text="Saturation multiplier applied to the stars after "
                      "stretching. 1.0 leaves color unchanged, higher enriches "
                      "star color, lower desaturates. No effect on mono images.",
        )
        self._star_stretch_preview_check = sst.add_check("Show before/after preview")
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
        ghs = CollapsibleSection(
            "Hyperbolic Stretch (GHS)",
            help_text="Generalized Hyperbolic Stretch: a more flexible "
                      "nonlinear stretch than Auto-Stretch, with independent "
                      "control over stretch strength, symmetry, and how "
                      "strongly shadows and highlights are protected from "
                      "being pushed. Popular in PixInsight for fine-tuned, "
                      "repeatable stretches.",
        )
        ghs.add_info("Advanced non-linear stretch.")
        self._ghs_d_spin  = ghs.add_spin(
            "Stretch (D)",   0.0, 20.0, 5.0, 0.5, 1,
            help_text="Overall stretch intensity. 0 leaves the image "
                      "unchanged; higher applies a stronger nonlinear stretch.",
        )
        self._ghs_b_spin  = ghs.add_spin(
            "Asymmetry (b)", -5.0, 5.0, 0.0, 0.1, 1,
            help_text="Shapes the stretch curve asymmetrically around the "
                      "symmetry point: negative compresses highlights more, "
                      "positive compresses shadows more, 0 is symmetric.",
        )
        self._ghs_sp_spin = ghs.add_spin(
            "Sym. point",    0.0,  1.0, 0.0, 0.05, 3,
            help_text="The pivot brightness (0-1) the stretch is centered on. "
                      "Set it to the background level to concentrate the "
                      "stretch on the sky and fainter signal.",
        )
        self._ghs_shadow_slider    = ghs.add_slider(
            "Shadow prot.",    0.0, 0.0, 1.0, 0.01, 2,
            help_text="Protects the darkest pixels from being stretched, "
                      "reducing how much background noise gets amplified. "
                      "0 = no protection, 1 = shadows barely move.",
        )
        self._ghs_highlight_slider = ghs.add_slider(
            "Highlight prot.", 0.0, 0.0, 1.0, 0.01, 2,
            help_text="Protects the brightest pixels (star cores) from being "
                      "pushed further, reducing clipping to white. 0 = no "
                      "protection, 1 = highlights barely move.",
        )
        self._ghs_preview_check = ghs.add_check("Show before/after preview")
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
        btns[1].clicked.connect(lambda: (
            self._ghs_d_spin.setValue(5.0),
            self._ghs_b_spin.setValue(0.0),
            self._ghs_sp_spin.setValue(0.0),
            self._ghs_shadow_slider.setValue(0.0),
            self._ghs_highlight_slider.setValue(0.0),
        ))
        lay.addWidget(ghs)

        # Histogram Transform
        ht = CollapsibleSection(
            "Histogram Transform",
            help_text="Classic manual black point / midtone / white point "
                      "levels adjustment — the hands-on version of "
                      "Auto-Stretch: you choose where black, the midtone "
                      "curve, and white land instead of the tool computing "
                      "them from statistics.",
        )
        ht.add_info("Black point, midtone, and white point adjustment.")
        self._ht_black_spin  = ht.add_spin(
            "Black point", 0.0, 0.99, 0.0, 0.01, 3,
            help_text="Input level mapped to pure black. Raise it to crush "
                      "the background darker; anything below this clips to 0.",
        )
        self._ht_midtone_slider = ht.add_slider(
            "Midtone",  0.5, 0.01, 0.99, 0.01, 2, 0.5,
            help_text="Brightness of the midtones. Higher brightens the "
                      "middle of the tonal range without moving the black or "
                      "white points.",
        )
        self._ht_white_spin  = ht.add_spin(
            "White point", 0.01, 1.0, 1.0, 0.01, 3,
            help_text="Input level mapped to pure white. Lower it to "
                      "brighten highlights faster; anything above this clips "
                      "to 1. Useful for a final contrast boost after "
                      "stretching.",
        )
        self._ht_black_spin.valueChanged.connect(lambda _: self._emit_clip_points())
        self._ht_white_spin.valueChanged.connect(lambda _: self._emit_clip_points())
        self._ht_preview_check = ht.add_check("Show before/after preview")
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
        btns[1].clicked.connect(self.reset_histogram_transform_params)
        lay.addWidget(ht)

        # Curves
        crv = CollapsibleSection(
            "Curves",
            help_text="A freeform tone curve: click to add control points, "
                      "drag to reshape, right-click to remove. The most "
                      "flexible stretch and contrast tool — anything "
                      "Auto-Stretch, Histogram Transform, or GHS can do can "
                      "also be shaped here by hand.",
        )
        crv.add_info("Click to add points, drag to adjust. Right-click to remove.")
        self._curve_channel_combo = crv.add_combo(
            "Channel", ["Master (L)", "Red", "Green", "Blue"],
            help_text="Which channel the curve applies to. Master (L) affects "
                      "overall brightness/contrast; Red, Green, and Blue let "
                      "you correct color balance or push a single channel's "
                      "contrast.",
        )
        self._curve_channel_combo.currentIndexChanged.connect(
            lambda _: self.curves_histogram_changed.emit()
        )
        self._curve_editor = CurveEditor()
        self._curve_editor.setMinimumHeight(180)
        crv.add_widget(self._curve_editor)
        self._curves_histogram_check = crv.add_check(
            "Show histogram",
            help_text="Overlays the image's brightness histogram behind the "
                      "curve, to help place points relative to where the data "
                      "actually lies.",
        )
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
        crop = CollapsibleSection(
            "Crop", accent=True,
            help_text="Crops the image to a rectangular region, drawn on the "
                      "canvas or typed as exact pixel coordinates. Use it to "
                      "trim the soft, uneven edges stacking leaves behind, or "
                      "to frame a smaller area of interest.",
        )
        crop.add_info("Crop image to a rectangular region.")
        self._btn_crop_draw = RunBtn("✏ Draw on Image…", flat=True)
        self._btn_crop_draw.setCheckable(True)
        self._btn_crop_draw.toggled.connect(
            lambda on: self.start_crop_draw.emit() if on else None
        )
        crop.add_widget(self._btn_crop_draw)
        self._crop_x_spin = crop.add_spin(
            "X offset", 0, 99999, 0,
            help_text="Left edge of the crop rectangle, in pixels from the "
                      "image's left side.",
        )
        self._crop_y_spin = crop.add_spin(
            "Y offset", 0, 99999, 0,
            help_text="Top edge of the crop rectangle, in pixels from the "
                      "image's top.",
        )
        self._crop_w_spin = crop.add_spin(
            "Width",    0, 99999, 0,
            help_text="Width of the crop rectangle in pixels. 0 uses "
                      "everything remaining to the right of the offset.",
        )
        self._crop_h_spin = crop.add_spin(
            "Height",   0, 99999, 0,
            help_text="Height of the crop rectangle in pixels. 0 uses "
                      "everything remaining below the offset.",
        )
        crop.add_run("▶ Apply Crop", self.run_crop.emit)
        lay.addWidget(crop)

        # Rotate
        rot = CollapsibleSection(
            "Rotate",
            help_text="Rotates the whole image by a fixed 90-degree step or "
                      "an arbitrary angle.",
        )
        self._rotate_combo = rot.add_combo(
            "Preset", ["90° CW", "180°", "270° CW", "Custom angle"],
            help_text="90/180/270 rotate losslessly with no interpolation. "
                      "Custom angle rotates by any angle typed below, which "
                      "resamples the image and can add a slight softness.",
        )
        self._rotate_angle_spin = rot.add_spin(
            "Custom angle", -360, 360, 0, 0.1, 1, "°",
            help_text="Rotation angle in degrees, used only when Preset is "
                      "set to Custom angle. Positive rotates clockwise.",
        )
        self._rotate_expand_check = rot.add_check(
            "Expand canvas", True,
            help_text="Grows the canvas so no corners are cut off by the "
                      "rotation. Unticked, the image keeps its original size "
                      "and the corners get cropped away.",
        )
        rot.add_run("▶ Apply Rotation", self.run_rotate.emit)
        lay.addWidget(rot)

        # Flip
        flp = CollapsibleSection(
            "Flip",
            help_text="Mirrors the image horizontally, vertically, or both — "
                      "useful for matching orientation between different "
                      "optical trains (star diagonal vs. none) or fixing an "
                      "upside-down capture.",
        )
        self._flip_combo = flp.add_combo(
            "Axis", ["Horizontal", "Vertical", "Both"],
            help_text="Horizontal flips left/right, Vertical flips top/bottom, "
                      "Both does both at once (equivalent to a 180-degree "
                      "rotation without resampling).",
        )
        flp.add_run("▶ Apply Flip", self.run_flip.emit)
        lay.addWidget(flp)

        # Resize
        rsz = CollapsibleSection(
            "Resize / Resample",
            help_text="Resamples the image to a different pixel size by a "
                      "scale factor. Use it to shrink an oversampled image for "
                      "sharing, or, more rarely, to enlarge one — enlarging "
                      "cannot invent real detail.",
        )
        self._resize_scale_spin = rsz.add_spin(
            "Scale", 0.1, 10.0, 1.0, 0.1, 2,
            help_text="Multiplier applied to both width and height. Below 1 "
                      "shrinks the image; above 1 enlarges it.",
        )
        self._resize_interp_combo = rsz.add_combo(
            "Interpolation", ["Lanczos", "Bicubic", "Bilinear", "Nearest"],
            help_text="Algorithm used to compute the new pixel values. "
                      "Lanczos gives the sharpest results for both shrinking "
                      "and enlarging (recommended). Bicubic is a good, faster "
                      "general choice. Bilinear is faster but softer. Nearest "
                      "keeps hard pixel edges with no blending, useful for "
                      "masks, not photos.",
        )
        rsz.add_run("▶ Apply Resize", self.run_resize.emit)
        lay.addWidget(rsz)

        # Bin
        bn = CollapsibleSection(
            "Bin",
            help_text="Reduces resolution by averaging or summing blocks of "
                      "neighboring pixels, trading detail for a real "
                      "signal-to-noise gain — the digital equivalent of "
                      "on-sensor hardware binning. Useful on oversampled data "
                      "or to rescue a noisy stack.",
        )
        bn.add_info("Combine pixels to increase SNR at lower resolution.")
        self._bin_factor_combo = bn.add_combo(
            "Factor", ["2x2", "3x3", "4x4"],
            help_text="Block size to combine: 2x2 halves the resolution, 3x3 "
                      "and 4x4 reduce it further.",
        )
        self._bin_mode_combo   = bn.add_combo(
            "Mode",   ["Average", "Sum"],
            help_text="Average keeps the same brightness scale and improves "
                      "SNR (recommended for finished images). Sum adds the "
                      "values together, brightening the result — matches how "
                      "a camera's hardware binning works on raw data.",
        )
        bn.add_run("▶ Apply Bin", self.run_bin.emit)
        lay.addWidget(bn)

        # Invert
        inv = CollapsibleSection(
            "Invert",
            help_text="Inverts every pixel value (1 minus the value), turning "
                      "a bright nebula on a dark sky into a dark shape on a "
                      "bright sky. Occasionally used to make faint dark "
                      "nebulae or dust lanes easier to see and trace.",
        )
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
        scnr = CollapsibleSection(
            "Remove Green Cast (SCNR)", accent=True,
            help_text="Removes an excess-green (or red/blue) color cast, most "
                      "often seen on broadband OSC/RGB cameras under "
                      "light-polluted skies. Replaces the target channel using "
                      "the other two instead of just scaling it down, so stars "
                      "keep a natural color.",
        )
        scnr.add_info("Remove color noise, typically excess green channel.")
        self._scnr_target_combo  = scnr.add_combo(
            "Target",  ["Green", "Red", "Blue"],
            help_text="Which channel gets neutralized. Green is the usual "
                      "cast to remove; Red/Blue handle unusual filter or "
                      "camera color casts.",
        )
        # Only the two implemented SCNR methods are offered; the previous
        # third entry silently ran Average Neutral.
        self._scnr_method_combo  = scnr.add_combo(
            "Method",
            ["Average Neutral", "Maximum Neutral"],
            help_text="Average Neutral replaces the target channel with the "
                      "average of the other two — balanced, moderate removal. "
                      "Maximum Neutral replaces it with whichever of the other "
                      "two is larger at each pixel — stronger removal, but can "
                      "shift color balance more.",
        )
        self._scnr_amount = scnr.add_slider(
            "Amount", 0.5, 0.0, 1.0, 0.01, 2,
            help_text="How much of the correction is applied. 0 leaves the "
                      "image unchanged, 1 applies the full neutralization.",
        )
        self._scnr_preview_check = scnr.add_check("Show before/after preview")
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
        ca = CollapsibleSection(
            "Color Adjustment",
            help_text="Global hue, saturation, and vibrance adjustment for the "
                      "whole image — a quick, coarse color grade. For "
                      "per-color-family or per-brightness-band control, use "
                      "Saturation by Hue or Selective Color instead.",
        )
        self._hue_slider        = ca.add_slider(
            "Hue shift",   0, -180, 180, 1, 0,
            help_text="Rotates every color around the color wheel by this "
                      "many degrees. 0 leaves hues unchanged.",
        )
        self._sat_slider        = ca.add_slider(
            "Saturation",  0, -100, 100, 1, 0,
            help_text="Overall color intensity. Negative washes colors toward "
                      "gray, 0 is unchanged, positive intensifies colors.",
        )
        # Vibrance is boost-only in the core (0-1); a negative range here was
        # an inert half of the slider.
        self._vibrance_slider   = ca.add_slider(
            "Vibrance",    0, 0, 100, 1, 0,
            help_text="Boosts only the less-saturated colors, leaving "
                      "already-vivid colors alone — a gentler way to add "
                      "punch without oversaturating stars. 0 = no boost.",
        )
        ca.add_run("▶ Apply Color Adjust", self.run_color_adjust.emit)
        lay.addWidget(ca)

        # Saturation / Chroma hue curves (ported from Seti Astro Suite Pro, GPL-3.0)
        sat = CollapsibleSection(
            "Saturation by Hue",
            help_text="Boost or mute the saturation of each color "
                      "independently: push the reds of an emission nebula "
                      "without oversaturating stars, or tame just the "
                      "greens. Each slider is a multiplier for that color "
                      "family (1.0 = unchanged).",
        )
        self._satc_mode_combo = sat.add_combo(
            "Color space", ["Saturation (HSV)", "Chroma (Lab)"],
            help_text="HSV saturation is fast and punchy but can shift hues "
                      "slightly. Lab chroma is perceptually cleaner and "
                      "keeps hues stable; use it for careful color work.",
        )
        self._satc_band_sliders = {}
        for band, hue, tip in [
            ("Red", 0, "Emission nebulae (H-alpha), red stars."),
            ("Yellow", 60, "Star cores, sodium regions."),
            ("Green", 120, "Usually noise in astro images; often muted."),
            ("Cyan", 180, "OIII regions, planetary nebula shells."),
            ("Blue", 240, "Reflection nebulae, hot stars."),
            ("Magenta", 300, "Mixed Ha+OIII edges, galaxy cores."),
        ]:
            self._satc_band_sliders[hue] = sat.add_slider(
                band, 1.0, 0.0, 3.0, 0.05, 2,
                help_text=f"Saturation multiplier for {band.lower()} tones. "
                          f"{tip} 0 removes the color, 1 keeps it, "
                          "3 triples it.",
            )
        self._satc_strength = sat.add_slider(
            "Master strength", 1.0, 0.0, 3.0, 0.05, 2,
            help_text="Global multiplier applied on top of all the band "
                      "sliders. 1.0 = use band values as-is.",
        )
        sat.add_run("▶ Apply Saturation", self.run_sat_chroma.emit)
        lay.addWidget(sat)

        # Selective Color (ported from Seti Astro Suite Pro, GPL-3.0)
        selc = CollapsibleSection(
            "Selective Color",
            help_text="Adjust ONE color family without touching the rest: "
                      "pick the family, then shift its color balance, "
                      "brightness, and saturation. Like Photoshop's "
                      "Selective Color, tuned for astro images.",
        )
        self._selc_family_combo = selc.add_combo(
            "Color family",
            ["Reds", "Yellows", "Greens", "Cyans", "Blues", "Magentas"],
            help_text="Which colors are selected for adjustment. Reds "
                      "covers H-alpha nebulosity, Cyans covers OIII, Blues "
                      "covers reflection nebulae.",
        )
        self._selc_smooth = selc.add_slider(
            "Edge feather", 10.0, 0.0, 60.0, 1.0, 0,
            help_text="How softly the selection fades at the edge of the "
                      "color family, in hue degrees. Higher avoids hard "
                      "color seams.",
        )
        self._selc_intensity = selc.add_slider(
            "Intensity", 1.0, 0.0, 2.0, 0.05, 2,
            help_text="Master strength of all adjustments below.",
        )
        selc.add_divider()
        self._selc_adjust = {}
        for name, tip in [
            ("cyan", "Shift the selected colors toward cyan (+) or red (-)."),
            ("magenta", "Shift toward magenta (+) or green (-)."),
            ("yellow", "Shift toward yellow (+) or blue (-)."),
            ("red", "Add (+) or remove (-) red."),
            ("green", "Add (+) or remove (-) green."),
            ("blue", "Add (+) or remove (-) blue."),
            ("luminance", "Brighten (+) or darken (-) the selected colors."),
            ("chroma", "Increase (+) or mute (-) the colorfulness of the "
                       "selection without changing its brightness."),
            ("contrast", "Add (+) or reduce (-) contrast inside the "
                         "selection."),
        ]:
            self._selc_adjust[name] = selc.add_slider(
                name.capitalize(), 0.0, -1.0, 1.0, 0.01, 2, help_text=tip,
            )
        selc.add_divider()
        self._selc_min_chroma = selc.add_slider(
            "Ignore gray below", 0.05, 0.0, 0.5, 0.01, 2,
            help_text="Pixels less colorful than this are never selected, "
                      "protecting the gray background from color shifts.",
        )
        self._selc_edge_blur = selc.add_slider(
            "Mask blur", 0.0, 0.0, 20.0, 0.5, 1,
            help_text="Blurs the selection mask by this many pixels for "
                      "seamless transitions.",
        )
        selc.add_run("▶ Apply Selective Color", self.run_selective_color.emit)
        lay.addWidget(selc)

        # Selective Luminance (ported from Seti Astro Suite Pro, GPL-3.0)
        sell = CollapsibleSection(
            "Selective Luminance",
            help_text="Adjust only a brightness band: color-correct just "
                      "the shadows, add contrast to just the midtones, or "
                      "desaturate only the highlights.",
        )
        self._sell_range_combo = sell.add_combo(
            "Range", ["Shadows", "Midtones", "Highlights", "Custom"],
            help_text="Which brightness band is selected. Custom uses the "
                      "Low/High values below.",
        )
        self._sell_lo = sell.add_slider(
            "Low", 0.0, 0.0, 1.0, 0.01, 2,
            help_text="Lower edge of the custom brightness band (0 = "
                      "black).",
        )
        self._sell_hi = sell.add_slider(
            "High", 0.25, 0.0, 1.0, 0.01, 2,
            help_text="Upper edge of the custom brightness band (1 = "
                      "white).",
        )
        self._sell_smooth = sell.add_slider(
            "Edge feather", 0.05, 0.0, 0.5, 0.01, 2,
            help_text="How softly the selection fades at the band edges.",
        )
        self._sell_intensity = sell.add_slider(
            "Intensity", 1.0, 0.0, 2.0, 0.05, 2,
            help_text="Master strength of all adjustments below.",
        )
        sell.add_divider()
        self._sell_adjust = {}
        for name, tip in [
            ("cyan", "Shift the selected tones toward cyan (+) or red (-)."),
            ("magenta", "Shift toward magenta (+) or green (-)."),
            ("yellow", "Shift toward yellow (+) or blue (-)."),
            ("red", "Add (+) or remove (-) red."),
            ("green", "Add (+) or remove (-) green."),
            ("blue", "Add (+) or remove (-) blue."),
            ("luminance", "Brighten (+) or darken (-) the selected band."),
            ("chroma", "Increase (+) or mute (-) colorfulness in the band."),
            ("contrast", "Add (+) or reduce (-) contrast inside the band."),
        ]:
            self._sell_adjust[name] = sell.add_slider(
                name.capitalize(), 0.0, -1.0, 1.0, 0.01, 2, help_text=tip,
            )
        self._sell_edge_blur = sell.add_slider(
            "Mask blur", 5.0, 0.0, 20.0, 0.5, 1,
            help_text="Blurs the selection mask by this many pixels so the "
                      "band transition is invisible.",
        )
        sell.add_run("▶ Apply Selective Luminance", self.run_selective_luma.emit)
        lay.addWidget(sell)

        # Color Calibration
        cc = CollapsibleSection(
            "Color Calibration",
            help_text="Corrects the overall color balance (white balance) so "
                      "the sky background reads neutral and star/nebula "
                      "colors look realistic, using one of several reference "
                      "methods.",
        )
        cc.add_info("White balance using background reference or star colours.")
        self._cc_method_combo = cc.add_combo(
            "Method",
            ["Background reference", "Photometric (SPCC)", "Manual RGB"],
            help_text="Background reference measures the sky background and "
                      "forces it neutral — fast, no plate solve needed. "
                      "Photometric (SPCC) plate-solves the field and uses real "
                      "catalog star colors for accurate, repeatable results "
                      "(see the SPCC section below). Manual RGB lets you type "
                      "your own per-channel multipliers.",
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
        # (A "Pick BG Reference" button used to sit here, wired to nothing —
        # the calibrator finds the background from the darkest pixels itself.)
        cc.add_run("▶ Calibrate", self.run_color_calibration.emit)
        lay.addWidget(cc)

        # PCC — Photometric Color Calibration (plate solve + Gaia DR3)
        pcc = CollapsibleSection(
            "PCC (Photometric)",
            help_text="Plate-solves the image to know exactly which stars are "
                      "in the field, then uses their known catalog magnitudes "
                      "(Gaia DR3) to compute an accurate color correction — a "
                      "reliable way to get real, repeatable star and nebula "
                      "colors without guessing a white balance by eye.",
        )
        pcc.add_info("Plate-solve + Gaia DR3 star catalog white balance.")
        self._pcc_solver_combo = pcc.add_combo(
            "Solver", ["Auto", "GAIA (offline)", "ASTAP", "Astrometry.net"],
            help_text="Which plate-solving engine identifies the field. Auto "
                      "tries the available solvers in order; ASTAP and "
                      "Astrometry.net are external tools that must be "
                      "installed separately. GAIA (offline) solves without "
                      "an internet connection once a local Gaia catalog band "
                      "has been downloaded (Tools -> GAIA Catalog Manager).",
        )
        self._pcc_ra_spin  = pcc.add_spin(
            "RA hint (°)",  0.0, 360.0, 0.0, 0.001, 3,
            help_text="Approximate field-center right ascension in degrees, "
                      "to help the solver search a smaller area and solve "
                      "faster. Leave at 0 to read it from the FITS header.",
        )
        self._pcc_dec_spin = pcc.add_spin(
            "Dec hint (°)", -90.0, 90.0, 0.0, 0.001, 3,
            help_text="Approximate field-center declination in degrees, to "
                      "help the solver search a smaller area and solve "
                      "faster. Leave at 0 to read it from the FITS header.",
        )
        pcc.add_info("Leave RA/Dec at 0 to read from FITS header.")
        pcc.add_run("▶ Run PCC", self.run_pcc.emit)
        lay.addWidget(pcc)

        # SPCC — Spectrophotometric Color Calibration (sensor QE + filter curves)
        spcc = CollapsibleSection(
            "SPCC (Spectrophotometric)",
            help_text="Like PCC, but models the actual filter transmission "
                      "curves and camera sensor response, then fits the color "
                      "correction from real stellar spectra — the most "
                      "physically accurate calibration available. Also "
                      "requires the image to be plate-solved first.",
        )
        spcc.add_info("Sensor QE + filter transmission curves for precise white balance.")
        # Only filter sets with real response curves in spcc.py are offered;
        # the previous "Narrowband" and "Custom" entries silently ran the
        # OSC broadband calibration.
        self._spcc_filter_combo  = spcc.add_combo(
            "Filter set", ["OSC (no filter)", "Broadband (L/R/G/B)"],
            help_text="Which filters captured the data. OSC (no filter) is "
                      "for one-shot-color cameras; Broadband (L/R/G/B) is for "
                      "mono cameras with standard luminance/RGB filters.",
        )
        self._spcc_camera_combo  = spcc.add_combo(
            "Camera", ["ZWO ASI2600MM Pro", "QHY268M", "ZWO ASI533MC Pro"],
            help_text="The camera model used, so its measured sensor "
                      "quantum-efficiency curve feeds the color fit. Pick the "
                      "closest match if your exact camera isn't listed.",
        )
        spcc.add_run("▶ Run SPCC", self.run_spcc.emit)
        lay.addWidget(spcc)
        self._populate_spcc_cameras()

        # Narrowband
        nb = CollapsibleSection(
            "Narrowband Tools",
            help_text="Combine narrowband filter data (Ha, OIII, SII) into "
                      "false-color palettes such as SHO (Hubble palette) or "
                      "HOO, and subtract continuum starlight from a narrowband "
                      "channel to isolate the pure emission signal.",
        )
        nb.add_info("SHO/HOO/HaRGB palette mapping, continuum subtraction, blending.")
        nb.add_run("⊞ Open Narrowband Dialog…", self.open_narrowband_dialog.emit, flat=True)
        nb.add_run("▶ Continuum Subtraction", self.run_continuum_subtraction.emit, flat=True)
        lay.addWidget(nb)

        # LRGB / Channels
        lc = CollapsibleSection(
            "LRGB / Channel Combine",
            help_text="Combine separate channel images into one: LRGB Combine "
                      "merges a sharp luminance frame with lower-resolution "
                      "color data; Channel Combine builds a color image from "
                      "any three separate channel images. Extract Luminance "
                      "and Split Channels do the reverse, pulling channels "
                      "back apart.",
        )
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
        dec = CollapsibleSection(
            "Deconvolution", accent=True,
            help_text="Sharpens fine detail blurred by atmospheric seeing or "
                      "tracking error by mathematically reversing that blur, "
                      "using an estimate of the point spread function (PSF) — "
                      "how a single star's light was smeared out. Needs a "
                      "reasonable PSF FWHM (use Measure PSF) and works best on "
                      "clean, low-noise data; too many iterations or too "
                      "little deringing can add ring-shaped halos around "
                      "bright stars.",
        )
        dec.add_info("Restore fine detail lost to seeing or tracking.")
        self._deconv_method_combo = dec.add_combo(
            "Method", ["Richardson-Lucy", "Blind (Spatial)", "Wiener"],
            help_text="Richardson-Lucy is the standard iterative method and "
                      "needs an accurate PSF FWHM below. Blind (Spatial) "
                      "estimates the PSF from the image itself, useful when "
                      "there's no good star to measure. Wiener is a faster, "
                      "single-pass method with less fine control.",
        )
        self._deconv_psf_spin   = dec.add_spin(
            "PSF FWHM (px)", 0.5, 20.0, 3.0, 0.1, 1,
            help_text="How wide a typical star's blur is, in pixels (full "
                      "width at half maximum). Use Measure PSF for an accurate "
                      "value instead of guessing — a wrong FWHM gives poor "
                      "sharpening or ringing.",
        )
        self._deconv_iter       = dec.add_slider(
            "Iterations", 30, 1, 200, 1, 0,
            help_text="Number of refinement passes. More sharpens further but "
                      "amplifies noise and ringing faster than it reveals real "
                      "detail; watch the preview and stop once artifacts "
                      "appear.",
        )
        self._deconv_reg        = dec.add_spin(
            "Regularization", 0.0, 0.1, 0.001, 0.001, 4,
            help_text="Smoothing applied between iterations to suppress noise "
                      "amplification. 0 disables it (sharper but noisier); "
                      "higher keeps the result cleaner but softer.",
        )
        self._deconv_deringing  = dec.add_check(
            "Deringing protection", True,
            help_text="Detects star cores and dampens sharpening near them, "
                      "preventing the dark or bright rings deconvolution can "
                      "leave around bright stars.",
        )
        self._deconv_dering_amt = dec.add_slider(
            "Deringing amount", 0.5, 0.0, 1.0, 0.05, 2,
            help_text="Strength of the deringing protection. Higher protects "
                      "star cores more but can leave them slightly softer than "
                      "the rest of the image.",
        )
        btns = dec.add_btn_row([("Measure PSF", True), ("Star Mask", True)])
        btns[0].clicked.connect(self.measure_psf.emit)
        btns[1].clicked.connect(self.open_star_mask_dialog.emit)
        self._deconv_preview_check = dec.add_check("Show before/after preview")
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
        psf = CollapsibleSection(
            "PSF Measurement",
            help_text="Detects stars, fits a 2D Gaussian to each one, and "
                      "reports the average star width (FWHM), roundness "
                      "(ellipticity), and orientation — a quick check of focus "
                      "and tracking/optics quality, and the tool that feeds an "
                      "accurate PSF FWHM into Deconvolution.",
        )
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
        self._psf_cutout_spin = psf.add_spin(
            "Cutout radius", 6, 32, 12,
            help_text="Half-size, in pixels, of the box cropped around each "
                      "star for the Gaussian fit. Increase for large, bloated "
                      "stars; too large can pull in neighboring stars and "
                      "skew the fit.",
        )
        self._psf_force_cpu   = psf.add_check(
            "Force CPU (for parallel use)",
            help_text="Runs the measurement on the CPU instead of the GPU. "
                      "Use this to measure PSF while the GPU is busy with "
                      "another operation running at the same time.",
        )
        psf.add_run("▶ Measure PSF", self.measure_psf.emit)
        lay.addWidget(psf)

        # Noise Reduction
        dnz = CollapsibleSection(
            "Noise Reduction",
            help_text="Reduces random pixel noise while trying to preserve "
                      "real detail and star shapes. Try Wavelet or TGV first "
                      "on a stacked astro image, and use Auto to measure the "
                      "image's noise level and pick a sensible starting "
                      "strength.",
        )
        self._denoise_method_combo = dnz.add_combo(
            "Method",
            ["TGV Denoise", "NLM (Non-Local Means)", "Wavelet Denoise", "Median Filter"],
            help_text="TGV Denoise: edge-preserving smoothing, a good general "
                      "default. NLM (Non-Local Means): compares whole patches "
                      "across the image, strong on faint smooth backgrounds "
                      "but slower. Wavelet Denoise: removes noise scale by "
                      "scale, preserves fine structure well. Median Filter: "
                      "simple and fast, best on isolated speckle noise, "
                      "softest on fine detail.",
        )
        self._denoise_amount     = dnz.add_slider(
            "Amount",     0.5, 0.0, 1.0, 0.05, 2,
            help_text="Overall denoise strength. Higher removes more noise "
                      "but risks smoothing away faint real detail.",
        )
        self._denoise_lum        = dnz.add_slider(
            "Luminance",  0.7, 0.0, 1.0, 0.05, 2,
            help_text="How much of the original detail is blended back in. "
                      "Higher preserves more fine detail (lighter denoising); "
                      "lower gives a stronger, smoother result.",
        )
        self._denoise_chrom      = dnz.add_slider(
            "Chrominance",0.5, 0.0, 1.0, 0.05, 2,
            help_text="Above the halfway point, only color noise is removed "
                      "and luminance detail is left untouched; below halfway, "
                      "luminance and color noise are both reduced together.",
        )
        _auto_btn = dnz.add_btn_row([("🎯 Auto (measure noise)", False)])[0]
        _auto_btn.clicked.connect(self.request_auto_denoise.emit)
        self._denoise_noise_label = dnz.add_status_label("Noise: not measured")
        self._denoise_preview_check = dnz.add_check("Show before/after preview")
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

        # Background Grain — post-stretch luminance grain in the dark sky only
        bgg = CollapsibleSection(
            "Background Grain",
            help_text="Smooths the mid-tone luminance grain in the dark sky "
                      "that the black point can't reach, since it only crushes "
                      "the shadows. Confined to the background, so the subject "
                      "and stars stay sharp. Apply after stretching.",
        )
        bgg.add_info(
            "Smooths the mid-tone luminance grain in the dark sky that the black "
            "point can't reach (it only crushes the shadows). Confined to the "
            "background, so the subject and stars stay sharp. Apply after stretching."
        )
        self._bg_grain_strength = bgg.add_slider(
            "Strength", 0.5, 0.0, 1.0, 0.05, 2,
            help_text="How strongly the background grain is smoothed. Higher "
                      "smooths more but can look plasticky if the background "
                      "has real faint structure.",
        )
        bgg.add_run("▶ Reduce Background Grain", self.run_background_grain.emit)
        lay.addWidget(bgg)

        # Star Reduction
        sr = CollapsibleSection(
            "Star Reduction",
            help_text="Shrinks star sizes using morphological erosion inside "
                      "an automatically detected star mask, so stars take up "
                      "less of the frame and faint nebula detail around and "
                      "beneath them becomes easier to see. Run after "
                      "stretching; protect the core to avoid a doughnut look "
                      "on bright stars.",
        )
        sr.add_info("Reduce star halos to reveal faint nebula details.")
        self._star_reduction_amount = sr.add_slider(
            "Amount (%)", 50, 0, 100, 1, 0,
            help_text="How much the stars are shrunk. Higher reduces each "
                      "star's size further.",
        )
        self._star_reduction_kernel = sr.add_combo(
            "Kernel", ["Elliptical", "Circular", "Square", "Diamond"],
            help_text="Shape of the erosion kernel used to shrink stars. "
                      "Elliptical/Circular give the most natural round stars; "
                      "Square and Diamond are more aggressive and can leave "
                      "stars looking less round.",
        )
        self._star_reduction_iters = sr.add_slider(
            "Iterations", 2, 1, 10, 1, 0,
            help_text="Number of erosion passes. More passes shrink stars "
                      "further but take longer and can distort star shapes.",
        )
        self._star_reduction_protect = sr.add_check(
            "Protect core", True,
            help_text="Keeps the brightest center of each star intact while "
                      "shrinking its outer halo, avoiding a hollow, "
                      "doughnut-shaped star.",
        )
        sr.add_run("▶ Reduce Stars", self.run_star_reduction.emit)
        lay.addWidget(sr)

        # Wavelets / MLT
        wav = CollapsibleSection(
            "Wavelets / MLT",
            help_text="Multi-scale sharpening: splits the image into layers "
                      "from fine detail to large structure (a trous wavelet "
                      "transform) and lets you boost or suppress each layer "
                      "independently. Wavelets applies the layer gains "
                      "directly; MLT (Multi-scale Linear Transform) "
                      "reconstructs with the same layers for a different "
                      "blend. Layer 1 is the finest detail, higher layers are "
                      "progressively coarser structure.",
        )
        wav.add_info("Multi-scale sharpening with per-layer control.")
        self._wavelet_layers = wav.add_slider(
            "Layers", 5, 2, 8, 1, 0,
            help_text="How many wavelet scales the image is decomposed into. "
                      "More layers reach larger structures but take longer.",
        )
        self._wavelet_layer_sliders: list[SliderRow] = []
        defaults = [0.3, 0.3, 0.0, 0.0, 0.0]
        for i in range(5):
            s = wav.add_slider(
                f"Layer {i+1}", defaults[i], 0.0, 2.0, 0.1, 1,
                help_text="Gain for this wavelet scale's detail. 0 removes "
                          "that scale entirely, 1 leaves it unchanged, above "
                          "1 boosts/sharpens it. Lower layer numbers are "
                          "finer detail (and noise); higher numbers are "
                          "coarser structure.",
            )
            self._wavelet_layer_sliders.append(s)
        self._wav_preview_check = wav.add_check("Show before/after preview")
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
        fs = CollapsibleSection(
            "Frequency Separation",
            help_text="Splits the image into a low-frequency layer (large-"
                      "scale structure, color, gradients) and a high-"
                      "frequency layer (fine detail, edges, star cores), so "
                      "you can process them independently — smooth color "
                      "mottle in the low layer without touching detail, or "
                      "boost detail in the high layer without amplifying "
                      "color noise.",
        )
        fs.add_info("Split into structure (LF) + detail (HF). Boost detail or smooth colour/gradients independently.")
        self._fs_method = fs.add_combo(
            "Method", ["Subtract (linear)", "Divide (ratio)"],
            help_text="Subtract (linear): detail = image minus structure, "
                      "recombined by adding back. Divide (ratio): detail = "
                      "image divided by structure, recombined by multiplying "
                      "back; keeps detail contrast consistent in both shadows "
                      "and highlights, often better on stretched data.",
        )
        self._fs_sigma = fs.add_slider(
            "Split radius", 5.0, 1.0, 50.0, 1.0, 1,
            help_text="Blur radius, in pixels, that defines the low-frequency "
                      "structure layer. Larger separates off coarser "
                      "structure, leaving more into the fine-detail layer; "
                      "smaller keeps more in the structure layer.",
        )
        self._fs_hf_boost = fs.add_slider(
            "Detail boost", 1.0, 0.0, 3.0, 0.05, 2,
            help_text="Multiplies the fine-detail layer before recombining. "
                      "Above 1 sharpens detail, below 1 softens it, 1 leaves "
                      "it unchanged.",
        )
        self._fs_lf_smooth = fs.add_slider(
            "Smooth structure", 0.0, 0.0, 30.0, 1.0, 1,
            help_text="Blurs the structure/color layer by this many pixels "
                      "before recombining, smoothing large-scale color mottle "
                      "or gradients without touching fine detail. 0 leaves it "
                      "untouched.",
        )
        self._fs_preview_check = fs.add_check("Show before/after preview")
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
        clh = CollapsibleSection(
            "Local Contrast / CLAHE",
            help_text="Contrast Limited Adaptive Histogram Equalization: "
                      "boosts local contrast in small tiles across the image "
                      "independently, revealing faint structure in both dim "
                      "and bright regions at once, while a clip limit stops "
                      "noise from being over-amplified in flat areas. Applied "
                      "to luminance only; color is preserved.",
        )
        self._clahe_clip  = clh.add_slider(
            "Clip limit", 2.0, 0.5, 10.0, 0.5, 1,
            help_text="Caps how much contrast can be boosted within each "
                      "tile before the excess spreads to neighboring tiles "
                      "instead. Higher allows a stronger local contrast boost "
                      "but amplifies more noise.",
        )
        self._clahe_tiles = clh.add_slider(
            "Tile size",  8,   4,   32,   1,   0,
            help_text="Number of contrast tiles along each axis of the grid. "
                      "More tiles boost finer local detail but can look "
                      "patchy; fewer tiles give a smoother, more "
                      "global-looking boost.",
        )
        self._clahe_preview_check = clh.add_check("Show before/after preview")
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
        um = CollapsibleSection(
            "Unsharp Mask",
            help_text="Classic sharpening: blurs a copy of the image, "
                      "subtracts it from the original to isolate edges, then "
                      "adds that edge layer back to increase local contrast "
                      "at edges. Simpler and faster than Deconvolution for a "
                      "quick contrast and detail boost.",
        )
        self._um_radius    = um.add_slider(
            "Radius (px)", 1.5, 0.5, 10.0, 0.5, 1,
            help_text="Size, in pixels, of the blur used to find edges. "
                      "Smaller sharpens fine detail; larger sharpens coarser "
                      "structure.",
        )
        self._um_amount    = um.add_slider(
            "Amount",      0.5, 0.0,  2.0, 0.05, 2,
            help_text="Strength of the sharpening. 0 has no effect; higher "
                      "increases edge contrast more but can create dark or "
                      "bright fringes if pushed too far.",
        )
        self._um_threshold = um.add_slider(
            "Threshold",   0.0, 0.0,  0.1, 0.005, 3,
            help_text="Minimum brightness difference needed before an edge "
                      "is sharpened. Raise it to avoid amplifying noise in "
                      "smooth, flat sky.",
        )
        self._um_preview_check = um.add_check("Show before/after preview")
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
        mor = CollapsibleSection(
            "Morphology",
            help_text="Structural pixel operations borrowed from computer "
                      "vision, mostly useful for cleaning up masks or "
                      "exaggerating/attenuating small bright features: "
                      "shrink, grow, remove small specks, fill small gaps, or "
                      "extract just the edges between regions.",
        )
        self._morph_op     = mor.add_combo(
            "Operation", ["Erosion", "Dilation", "Opening", "Closing", "Gradient"],
            help_text="Erosion shrinks bright regions and removes small "
                      "bright specks. Dilation grows bright regions and fills "
                      "small dark gaps. Opening (erode then dilate) removes "
                      "small bright noise while keeping larger shapes intact. "
                      "Closing (dilate then erode) fills small dark holes "
                      "while keeping larger shapes intact. Gradient "
                      "(dilation minus erosion) extracts just the edges "
                      "between regions.",
        )
        self._morph_kernel = mor.add_combo(
            "Kernel", ["Disk", "Square", "Diamond"],
            help_text="Shape of the neighborhood used for the operation. "
                      "Disk gives the most natural, round result; Square and "
                      "Diamond give straighter or diagonal edges.",
        )
        self._morph_iters  = mor.add_slider(
            "Iterations", 1, 1, 10, 1, 0,
            help_text="How many times the operation is repeated. More "
                      "iterations produce a stronger effect.",
        )
        mor.add_run("▶ Apply Morphology", self.run_morphology.emit)
        lay.addWidget(mor)

        # Chromatic Aberration
        ca = CollapsibleSection(
            "Chromatic Aberration",
            help_text="Corrects lateral chromatic aberration — red and blue "
                      "channels shifted slightly relative to green near the "
                      "frame edges, a common refractor and wide-field "
                      "artifact — by detecting the shift from star positions "
                      "and nudging the channels back into alignment.",
        )
        ca.add_info("Correct lateral colour fringing at image edges.")
        ca.add_run("▶ Correct CA", self.run_chromatic_aberration.emit)
        lay.addWidget(ca)

        # Halo-B-Gon (ported from Seti Astro Suite Pro, GPL-3.0)
        halo = CollapsibleSection(
            "Halo Reduction (Halo-B-Gon)",
            help_text="Shrinks the bright halos and glow rings around stars "
                      "that stretching amplifies, without eating the star "
                      "cores. Run it on stretched images; enable the linear "
                      "option only for unstretched data.",
        )
        self._halo_level_combo = halo.add_combo(
            "Strength", ["Extra Low", "Low", "Medium", "High"], current="Low",
            help_text="How aggressively halos are darkened. Start Low; "
                      "High can dim faint nebulosity near bright stars.",
        )
        self._halo_linear_check = halo.add_check(
            "Linear (unstretched) data",
            help_text="Tick when the image has NOT been stretched yet. The "
                      "tool then boosts the data internally to find halos.",
        )
        halo.add_run("▶ Reduce Halos", self.run_halo_reduction.emit)
        lay.addWidget(halo)

        # WaveScale HDR (ported from Seti Astro Suite Pro, GPL-3.0)
        wshdr = CollapsibleSection(
            "WaveScale HDR",
            help_text="Recovers detail inside bright cores (galaxy centers, "
                      "the Orion Trapezium) on already-stretched images by "
                      "boosting local contrast at several wavelet scales "
                      "while taming the overall highlight brightness.",
        )
        self._wshdr_scales = wshdr.add_spin(
            "Scales", 2, 10, 5,
            help_text="How many detail sizes are processed. More scales "
                      "reach larger structures but take longer.",
        )
        self._wshdr_compression = wshdr.add_slider(
            "Strength", 1.5, 0.1, 5.0, 0.05, 2,
            help_text="Local contrast boost inside the bright regions. "
                      "Higher digs out more core detail.",
        )
        self._wshdr_mask_gamma = wshdr.add_slider(
            "Focus", 5.0, 0.1, 10.0, 0.1, 1,
            help_text="Concentrates the effect on only the brightest areas. "
                      "Higher = tighter around the core; lower spreads the "
                      "effect into midtones.",
        )
        self._wshdr_decay = wshdr.add_slider(
            "Scale falloff", 0.5, 0.1, 1.0, 0.05, 2,
            help_text="How quickly the boost fades from fine to coarse "
                      "scales. Lower keeps it on fine detail only.",
        )
        wshdr.add_run("▶ Apply WaveScale HDR", self.run_wavescale_hdr.emit)
        lay.addWidget(wshdr)

        # WaveScale Dark Enhance (ported from Seti Astro Suite Pro, GPL-3.0)
        wsde = CollapsibleSection(
            "WaveScale Dark Enhance",
            help_text="Deepens and reveals faint dark structure: dust "
                      "lanes, dark nebulae, galaxy arm shadows. The "
                      "opposite of HDR: it works on the darkest parts of "
                      "the image.",
        )
        self._wsde_scales = wsde.add_spin(
            "Scales", 2, 10, 6,
            help_text="How many detail sizes are analyzed for dark "
                      "structure.",
        )
        self._wsde_boost = wsde.add_slider(
            "Boost", 5.0, 0.1, 10.0, 0.1, 1,
            help_text="Strength of the dark-detail enhancement. 1.0 does "
                      "nothing; higher makes dust lanes more pronounced.",
        )
        self._wsde_mask_gamma = wsde.add_slider(
            "Focus", 1.0, 0.1, 10.0, 0.1, 1,
            help_text="Concentrates the effect on only the faintest dips. "
                      "Raise it if midtones start darkening.",
        )
        self._wsde_iterations = wsde.add_spin(
            "Iterations", 1, 10, 2,
            help_text="Number of enhancement passes. Each pass re-measures "
                      "what is dark. More passes = stronger but slower.",
        )
        self._wsde_decay = wsde.add_slider(
            "Scale falloff", 0.5, 0.1, 1.0, 0.05, 2,
            help_text="How quickly the boost fades toward coarse scales. "
                      "The finest scale is always skipped to avoid boosting "
                      "noise.",
        )
        wsde.add_run("▶ Enhance Dark Structure", self.run_wavescale_dark.emit)
        lay.addWidget(wsde)

        # Texture & Clarity (ported from Seti Astro Suite Pro, GPL-3.0)
        txc = CollapsibleSection(
            "Texture and Clarity",
            help_text="Two midtone punch controls in one: Texture sharpens "
                      "fine surface detail, Clarity adds larger local "
                      "contrast, both without touching star cores or "
                      "shadows as hard as a normal sharpen. Negative values "
                      "smooth instead.",
        )
        self._txc_texture_amount = txc.add_slider(
            "Texture", 0.0, -1.0, 1.0, 0.01, 2,
            help_text="Fine-detail strength. Positive crispens surface "
                      "texture; negative gives a silky smoothing.",
        )
        self._txc_texture_radius = txc.add_slider(
            "Texture radius", 1.0, 0.1, 10.0, 0.1, 1,
            help_text="Size of the detail treated as texture, in pixels.",
        )
        self._txc_clarity_amount = txc.add_slider(
            "Clarity", 0.0, -1.0, 1.0, 0.01, 2,
            help_text="Local contrast strength at a larger scale. Positive "
                      "adds punch; negative gives a soft, dreamy look.",
        )
        self._txc_clarity_radius = txc.add_slider(
            "Clarity radius", 3.0, 0.1, 10.0, 0.1, 1,
            help_text="Size of the local contrast neighborhood, in pixels.",
        )
        self._txc_mask_strength = txc.add_slider(
            "Midtone protection", 1.0, 0.0, 1.0, 0.05, 2,
            help_text="1 = classic behavior (effect confined to midtones, "
                      "shadows and highlights protected). 0 = apply "
                      "everywhere.",
        )
        txc.add_run("▶ Apply Texture and Clarity", self.run_texture_clarity.emit)
        lay.addWidget(txc)

        # Blemish Blaster (ported from Seti Astro Suite Pro, GPL-3.0)
        blm = CollapsibleSection(
            "Blemish Blaster",
            help_text="Heals a small circular blemish -- a satellite trail "
                      "nick, hot pixel cluster, or plane streak -- by "
                      "sampling neighbor patches around the click point and "
                      "blending in whichever ones best match the "
                      "surrounding background, feathered at the edge of "
                      "the brush.",
        )
        blm.add_info("Click blemishes on the image to heal them locally.")
        self._blemish_radius = blm.add_spin(
            "Radius (px)", 1, 900, 12,
            help_text="Brush radius in pixels.",
        )
        self._blemish_feather = blm.add_slider(
            "Feather", 0.5, 0.0, 1.0, 0.05, 2,
            help_text="Edge softness. 0 = hard-edged disc; 1 = the "
                      "correction fades in smoothly all the way from the "
                      "brush edge to its center.",
        )
        self._blemish_opacity = blm.add_slider(
            "Opacity", 1.0, 0.0, 1.0, 0.05, 2,
            help_text="Blend strength of the healed value over the "
                      "original. 1.0 fully replaces; lower values "
                      "partially blend the healed patch back with the "
                      "original pixels.",
        )
        self._blemish_toggle_btn = RunBtn("Heal on click", flat=True)
        self._blemish_toggle_btn.setCheckable(True)
        self._blemish_toggle_btn.toggled.connect(self.blemish_mode_toggled.emit)
        blm.add_widget(self._blemish_toggle_btn)
        blm.add_info("Tick, then click blemishes on the image. Untick when done.")
        lay.addWidget(blm)

        self._tabs.addTab(scrollable_tab(lay), "◎  Detail")

    # ── TAB: Effects ──────────────────────────────────────
    def _build_effects_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # FX Tool (ported from Seti Astro Suite Pro, GPL-3.0)
        fx = CollapsibleSection(
            "FX Tool", accent=True,
            help_text="Artistic finishing effects: Orton glow, soft focus, "
                      "bloom, vignette, film grain, and split toning. Pick an "
                      "effect, tune its sliders, apply. Only the sliders for "
                      "the chosen effect are shown.",
        )
        self._fx_effect_combo = fx.add_combo(
            "Effect",
            ["Orton Glow", "Soft Focus", "Bloom", "Vignette", "Film Grain", "Split Tone"],
            help_text="Orton Glow: dreamy bright glow. Soft Focus: gentle "
                      "blur overlay. Bloom: glow only around highlights. "
                      "Vignette: darkened corners. Film Grain: analog "
                      "texture. Split Tone: tint shadows and highlights "
                      "different colors.",
        )
        self._fx_blur_radius = fx.add_slider(
            "Glow radius", 15.0, 1.0, 100.0, 1.0, 0,
            help_text="Size of the glow/blur in pixels. Bigger = softer, "
                      "dreamier halo.",
        )
        self._fx_opacity = fx.add_slider(
            "Opacity", 0.5, 0.0, 1.0, 0.01, 2,
            help_text="How strongly the effect is mixed into the image. "
                      "0 = off, 1 = full effect.",
        )
        self._fx_glow_brightness = fx.add_slider(
            "Glow brightness", 1.4, 0.5, 3.0, 0.05, 2,
            help_text="Brightens the glow layer before blending. Higher = "
                      "more luminous glow.",
        )
        # In a hideable container so the label disappears with the combo
        # when another effect is selected.
        self._fx_blend_combo = styled_combo(["Screen", "Soft Light", "Lighten"])
        self._fx_blend_row = QWidget()
        self._fx_blend_row.setLayout(field_row(
            "Blend mode", self._fx_blend_combo, 110,
            help_text="Screen is brightest and hazy, Soft Light is gentler "
                      "and protects shadows, Lighten shifts color the least.",
        ))
        fx.add_widget(self._fx_blend_row)
        self._fx_highlight_protect = fx.add_slider(
            "Highlight protect", 0.5, 0.0, 1.0, 0.01, 2,
            help_text="Fades the glow near already-bright areas so star "
                      "cores and galaxy cores don't blow out.",
        )
        self._fx_luma_recovery = fx.add_slider(
            "Luma recovery", 0.7, 0.0, 1.0, 0.01, 2,
            help_text="Pulls overall brightness back toward the original "
                      "after a Screen blend, preventing a washed-out look.",
        )
        self._fx_bloom_threshold = fx.add_slider(
            "Bloom threshold", 0.7, 0.0, 1.0, 0.01, 2,
            help_text="Only pixels brighter than this get the bloom glow. "
                      "Lower = more of the image blooms.",
        )
        self._fx_bloom_brightness = fx.add_slider(
            "Bloom brightness", 1.5, 0.5, 3.0, 0.05, 2,
            help_text="Brightness boost of the isolated highlights before "
                      "they are blended back.",
        )
        self._fx_vignette_amount = fx.add_slider(
            "Vignette amount", 0.4, 0.0, 1.0, 0.01, 2,
            help_text="How dark the corners get. 0 = none, 1 = black "
                      "corners.",
        )
        self._fx_vignette_radius = fx.add_slider(
            "Vignette radius", 0.7, 0.1, 1.5, 0.01, 2,
            help_text="Distance from center where darkening starts. Smaller "
                      "= vignette reaches further into the frame.",
        )
        self._fx_vignette_softness = fx.add_slider(
            "Vignette softness", 0.5, 0.05, 1.0, 0.01, 2,
            help_text="Softness of the transition into the dark corners.",
        )
        self._fx_grain_intensity = fx.add_slider(
            "Grain intensity", 0.1, 0.0, 1.0, 0.01, 2,
            help_text="Strength of the film grain texture.",
        )
        self._fx_grain_size = fx.add_slider(
            "Grain size", 1.0, 0.0, 8.0, 0.5, 1,
            help_text="Clump size of the grain. 0 is the finest grain; "
                      "higher values look like faster, chunkier film stock.",
        )
        self._fx_grain_mono = fx.add_check(
            "Monochrome grain", True,
            help_text="Ticked: the same grain on all channels (classic "
                      "black-and-white film look). Unticked: colored grain.",
        )
        self._fx_shadow_hue = fx.add_slider(
            "Shadow hue", 220.0, 0.0, 360.0, 1.0, 0,
            help_text="Tint color for the dark areas, as a hue angle "
                      "(0 red, 120 green, 220 blue...).",
        )
        self._fx_highlight_hue = fx.add_slider(
            "Highlight hue", 40.0, 0.0, 360.0, 1.0, 0,
            help_text="Tint color for the bright areas. Classic teal-orange "
                      "is shadows 220, highlights 40.",
        )
        self._fx_tone_balance = fx.add_slider(
            "Tone balance", 0.0, -1.0, 1.0, 0.01, 2,
            help_text="Shifts the split point: negative tints more of the "
                      "image as shadows, positive as highlights.",
        )
        self._fx_tone_strength = fx.add_slider(
            "Tone strength", 0.3, 0.0, 1.0, 0.01, 2,
            help_text="Overall strength of the split-tone tint.",
        )
        fx.add_run("▶ Apply FX", self.run_fx.emit)
        lay.addWidget(fx)
        self._fx_rows_by_effect = {
            "Orton Glow": [self._fx_blur_radius, self._fx_opacity,
                           self._fx_glow_brightness, self._fx_blend_row,
                           self._fx_highlight_protect, self._fx_luma_recovery],
            "Soft Focus": [self._fx_blur_radius, self._fx_opacity],
            "Bloom": [self._fx_blur_radius, self._fx_opacity,
                      self._fx_bloom_threshold, self._fx_bloom_brightness,
                      self._fx_luma_recovery],
            "Vignette": [self._fx_vignette_amount, self._fx_vignette_radius,
                         self._fx_vignette_softness],
            "Film Grain": [self._fx_grain_intensity, self._fx_grain_size,
                           self._fx_grain_mono],
            "Split Tone": [self._fx_shadow_hue, self._fx_highlight_hue,
                           self._fx_tone_balance, self._fx_tone_strength],
        }
        self._fx_effect_combo.currentTextChanged.connect(self._update_fx_rows)
        self._update_fx_rows()

        # Diffraction Spikes (ported from Seti Astro Suite Pro, GPL-3.0)
        sp = CollapsibleSection(
            "Diffraction Spikes",
            help_text="Adds synthetic diffraction spikes to the brightest "
                      "stars, like a Newtonian's spider vanes or a JWST "
                      "look. Stars are detected automatically; you choose "
                      "how many get spikes and how they look.",
        )
        self._spike_amount = sp.add_slider(
            "Stars with spikes", 10.0, 1.0, 100.0, 1.0, 0,
            help_text="Percentage of the brightest detected stars that "
                      "receive spikes. Keep it low: real optics only spike "
                      "the bright ones.",
        )
        self._spike_quantity = sp.add_spin(
            "Spikes per star", 2, 12, 4,
            help_text="Number of primary spikes. 4 = Newtonian cross, "
                      "6 = JWST style.",
        )
        self._spike_length = sp.add_slider(
            "Length", 1.0, 0.1, 5.0, 0.05, 2,
            help_text="Base spike length, scaled by each star's size.",
        )
        self._spike_angle = sp.add_slider(
            "Angle", 0.0, 0.0, 180.0, 1.0, 0,
            help_text="Rotates the whole spike pattern, in degrees.",
        )
        self._spike_intensity = sp.add_slider(
            "Intensity", 0.8, 0.0, 1.0, 0.01, 2,
            help_text="Opacity of the primary spikes.",
        )
        self._spike_width = sp.add_slider(
            "Width", 1.0, 0.2, 4.0, 0.05, 2,
            help_text="Thickness of the spikes.",
        )
        self._spike_sharpness = sp.add_slider(
            "Sharpness", 1.0, 0.2, 4.0, 0.05, 2,
            help_text="How quickly a spike fades along its length. Higher "
                      "= shorter bright core with a long faint tail.",
        )
        sp.add_divider()
        self._spike_secondary_intensity = sp.add_slider(
            "Secondary intensity", 0.0, 0.0, 1.0, 0.01, 2,
            help_text="Opacity of a second, offset spike set (0 disables "
                      "it). Adds the thin cross JWST images show.",
        )
        self._spike_secondary_length = sp.add_slider(
            "Secondary length", 0.5, 0.1, 3.0, 0.05, 2,
            help_text="Length of the secondary spikes relative to the star.",
        )
        self._spike_secondary_offset = sp.add_slider(
            "Secondary offset", 45.0, 0.0, 180.0, 1.0, 0,
            help_text="Angle between the primary and secondary spike sets.",
        )
        self._spike_flare_intensity = sp.add_slider(
            "Soft flare", 0.3, 0.0, 1.0, 0.01, 2,
            help_text="A soft round glow under the spikes that sells the "
                      "effect. 0 disables it.",
        )
        self._spike_halo_check = sp.add_check(
            "Diffraction halo ring",
            help_text="Adds a colored diffraction ring around spiked stars.",
        )
        self._spike_rainbow_check = sp.add_check(
            "Rainbow spikes",
            help_text="Overlays subtle chromatic (rainbow) structure along "
                      "the primary spikes, like real diffraction.",
        )
        sp.add_run("▶ Render Spikes", self.run_diffraction_spikes.emit)
        lay.addWidget(sp)

        lay.addStretch()
        self._tabs.addTab(scrollable_tab(lay), "✧  Effects")

    def _update_fx_rows(self):
        """Show only the sliders that belong to the selected FX effect."""
        current = self._fx_effect_combo.currentText()
        all_rows = set()
        for rows in self._fx_rows_by_effect.values():
            all_rows.update(rows)
        visible = set(self._fx_rows_by_effect.get(current, []))
        for row in all_rows:
            row.setVisible(row in visible)

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
        den = CollapsibleSection(
            "AI Denoise", accent=True,
            help_text="Runs a trained denoising model over the image in GPU "
                      "tiles, instead of a classical noise-reduction filter. "
                      "Noise2Self is a self-supervised model trained on real "
                      "astro data and works out of the box; CosmicClarity uses "
                      "your own separately obtained Cosmic Clarity models and "
                      "does nothing until you point Preferences at them.",
        )
        den.add_info(
            "Noise2Self: a self-supervised denoiser trained on real astro images, "
            "built in and ready. CosmicClarity uses your own Cosmic Clarity models "
            "(set their folder in Preferences); it does nothing until you do."
        )
        self._ai_denoise_backend = den.add_combo(
            "Backend",
            ["Noise2Self (built-in)", "CosmicClarity (your models)"],
            current="Noise2Self (built-in)",
            help_text="Which denoising model runs. Noise2Self is built in and "
                      "ready. CosmicClarity uses your own downloaded Cosmic "
                      "Clarity model files — set their folder in Preferences "
                      "> AI Models first.",
        )
        self._ai_denoise_strength = den.add_slider(
            "Strength", 0.7, 0.0, 1.0, 0.05, 2,
            help_text="How strongly the model's denoising is blended in. "
                      "Higher removes more noise but risks softening faint "
                      "real detail.",
        )
        self._ai_tile_combo       = den.add_combo(
            "Tile size", ["128", "256", "512", "Full"],
            help_text="Splits the image into tiles of this size for GPU "
                      "inference, keeping VRAM usage bounded on large images. "
                      "Full processes the whole image at once (needs the "
                      "most VRAM); smaller tiles use less memory.",
        )
        self._ai_star_protect     = den.add_check(
            "Protect stars (star mask)", True,
            help_text="Blends the original, undenoised image back in at "
                      "detected star positions, keeping star cores sharp "
                      "instead of softened by the denoiser.",
        )
        self._ai_tiled_check      = den.add_check(
            "Tiled inference (reduces VRAM)", True,
            help_text="Processes the image in smaller tiles instead of all "
                      "at once, to stay within GPU memory limits on large "
                      "images. Turn off only with ample VRAM to spare.",
        )
        den.add_run("▶ Apply AI Denoise", self.run_ai_denoise.emit)
        lay.addWidget(den)

        # AI Sharpen
        shr = CollapsibleSection(
            "AI Sharpen",
            help_text="Runs a sharpening engine over the image instead of "
                      "manually tuning Deconvolution or Unsharp Mask. "
                      "Richardson-Lucy (built-in) runs an automatic RL "
                      "deconvolution pass; CosmicClarity uses your own "
                      "separately obtained Cosmic Clarity models and does "
                      "nothing until you point Preferences at them.",
        )
        shr.add_info(
            "Richardson-Lucy deconvolution, built in and ready. CosmicClarity uses "
            "your own Cosmic Clarity models (set their folder in Preferences); it "
            "does nothing until you do."
        )
        self._ai_sharpen_backend = shr.add_combo(
            "Backend",
            ["Richardson-Lucy (built-in)", "CosmicClarity (your models)"],
            current="Richardson-Lucy (built-in)",
            help_text="Which sharpening engine runs. Richardson-Lucy "
                      "(built-in) works out of the box. CosmicClarity uses "
                      "your own downloaded Cosmic Clarity model files — set "
                      "their folder in Preferences > AI Models first.",
        )
        self._ai_sharpen_strength = shr.add_slider(
            "Strength", 0.5, 0.0, 1.0, 0.05, 2,
            help_text="How strongly the sharpening is applied. Higher "
                      "sharpens more but risks noise amplification and "
                      "ringing, the same trade-off as manual Deconvolution.",
        )
        shr.add_run("▶ Apply AI Sharpen", self.run_ai_sharpen.emit)
        lay.addWidget(shr)

        # Star Removal
        star = CollapsibleSection(
            "Star Removal", accent=True,
            help_text="Separates stars from the rest of the image into two "
                      "layers using a neural network, so you can process the "
                      "starless background (stretch, denoise, boost "
                      "nebulosity) without also over-processing the stars, "
                      "then add the stars back afterward.",
        )
        star.add_info("Remove stars from image using deep learning.")
        self._star_removal_path = star.add_combo(
            "Backend",
            ["Auto (StarNet v2 preferred)", "StarNet v2", "Built-in (starrem2k13)"],
            current="Auto (StarNet v2 preferred)",
            help_text="Which star-removal engine runs. Built-in "
                      "(starrem2k13) works with no download required. "
                      "StarNet v2 needs the external StarNetv2CLI binary "
                      "installed separately but can give cleaner results on "
                      "dense star fields; Auto prefers StarNet v2 when it's "
                      "available and falls back to the built-in model.",
        )
        self._star_threshold = star.add_slider(
            "Threshold", 0.5, 0.1, 0.9, 0.05, 2,
            help_text="How aggressively pixels are classified as star versus "
                      "background. Higher removes more of each star, "
                      "including its dimmer outer halo; lower is more "
                      "conservative and leaves more of the star behind.",
        )
        self._star_protect_bg = star.add_check(
            "Protect background detail", True,
            help_text="Tries to preserve faint nebulosity that overlaps "
                      "star positions instead of removing it along with the "
                      "star.",
        )
        star.add_info(
            "Built-in works out of the box (no download). For best results on "
            "dense fields, install StarNet v2 (StarNetv2CLI) from starnetastro.com "
            "and set its path in Preferences > AI Models.",
        )
        star.add_run("▶ Remove Stars", self.run_starnet.emit)
        lay.addWidget(star)

        # AI Super-Resolution
        sr = CollapsibleSection(
            "AI Super-Resolution",
            help_text="Upscales the image using a Real-ESRGAN neural network "
                      "that synthesizes plausible fine detail, rather than "
                      "just interpolating pixels like Resize/Resample. Can "
                      "make a low-resolution image look sharper for display, "
                      "but the added detail is not new real signal — do not "
                      "use it ahead of scientific measurement.",
        )
        sr.add_info("Upscale images with learned detail synthesis (Real-ESRGAN).")
        self._sr_scale = sr.add_combo(
            "Scale", ["2×", "4×"], current="2×",
            help_text="Output size multiplier. 4x takes noticeably longer "
                      "and can look artificial on already-soft or noisy "
                      "input.",
        )
        self._sr_tile = sr.add_combo(
            "Tile size", ["512", "1024", "Full"], current="512",
            help_text="Splits the image into tiles of this size for GPU "
                      "inference, to stay within VRAM limits. Full processes "
                      "the whole image at once and needs the most VRAM.",
        )
        sr.add_run("▶ Upscale", self.open_super_resolution.emit)
        lay.addWidget(sr)

        # Train
        train = CollapsibleSection(
            "Train Your Own Models",
            help_text="Fine-tunes the built-in AI denoiser on your own raw "
                      "light frames using self-supervised training "
                      "(Noise2Self), so the model learns your camera's "
                      "specific noise characteristics. No clean reference "
                      "images are needed. See the training guide button for "
                      "the exact command.",
        )
        train.add_info("Self-supervised training on your own astro images.")
        train.add_code_block(
            "poetry run python scripts/\ntrain_denoise_model.py\n--input astro_data --epochs 30"
        )
        train.add_run("Open Training Guide…", self._show_training_guide, flat=True)
        lay.addWidget(train)

        self._tabs.addTab(scrollable_tab(lay), "✦  AI Tools")

    def get_fx_params(self):
        from astraios.core.fx_effects import BlendMode, FXEffect, FXParams

        effect_map = {
            "Orton Glow": FXEffect.ORTON_GLOW, "Soft Focus": FXEffect.SOFT_FOCUS,
            "Bloom": FXEffect.BLOOM, "Vignette": FXEffect.VIGNETTE,
            "Film Grain": FXEffect.FILM_GRAIN, "Split Tone": FXEffect.SPLIT_TONE,
        }
        blend_map = {
            "Screen": BlendMode.SCREEN, "Soft Light": BlendMode.SOFT_LIGHT,
            "Lighten": BlendMode.LIGHTEN,
        }
        return FXParams(
            effect=effect_map.get(self._fx_effect_combo.currentText(),
                                  FXEffect.ORTON_GLOW),
            blur_radius=float(self._fx_blur_radius.value()),
            opacity=float(self._fx_opacity.value()),
            glow_brightness=float(self._fx_glow_brightness.value()),
            blend_mode=blend_map.get(self._fx_blend_combo.currentText(),
                                     BlendMode.SCREEN),
            highlight_protect=float(self._fx_highlight_protect.value()),
            luma_recovery=float(self._fx_luma_recovery.value()),
            bloom_threshold=float(self._fx_bloom_threshold.value()),
            bloom_brightness=float(self._fx_bloom_brightness.value()),
            vignette_amount=float(self._fx_vignette_amount.value()),
            vignette_radius=float(self._fx_vignette_radius.value()),
            vignette_softness=float(self._fx_vignette_softness.value()),
            grain_intensity=float(self._fx_grain_intensity.value()),
            grain_size=float(self._fx_grain_size.value()),
            grain_mono=self._fx_grain_mono.isChecked(),
            shadow_hue=float(self._fx_shadow_hue.value()),
            highlight_hue=float(self._fx_highlight_hue.value()),
            tone_balance=float(self._fx_tone_balance.value()),
            tone_strength=float(self._fx_tone_strength.value()),
        )

    def get_diffraction_spike_params(self):
        from astraios.core.diffraction_spikes import DiffractionSpikeParams

        return DiffractionSpikeParams(
            star_amount=float(self._spike_amount.value()),
            quantity=int(self._spike_quantity.value()),
            length=float(self._spike_length.value()),
            angle=float(self._spike_angle.value()),
            intensity=float(self._spike_intensity.value()),
            spike_width=float(self._spike_width.value()),
            sharpness=float(self._spike_sharpness.value()),
            secondary_intensity=float(self._spike_secondary_intensity.value()),
            secondary_length=float(self._spike_secondary_length.value()),
            secondary_offset=float(self._spike_secondary_offset.value()),
            soft_flare_intensity=float(self._spike_flare_intensity.value()),
            enable_halo=self._spike_halo_check.isChecked(),
            enable_rainbow=self._spike_rainbow_check.isChecked(),
        )

    def get_sat_chroma_params(self):
        from astraios.core.sat_chroma import SatChromaMode, SatChromaParams

        mode = (SatChromaMode.CHROMA_LAB
                if "Lab" in self._satc_mode_combo.currentText()
                else SatChromaMode.SATURATION_HSV)
        # Band sliders become curve control points; duplicate the 0-degree
        # value at 360 so the curve wraps cleanly.
        points = [(float(h), float(s.value()))
                  for h, s in sorted(self._satc_band_sliders.items())]
        points.append((360.0, points[0][1]))
        return SatChromaParams(
            mode=mode,
            curve_points=points,
            strength=float(self._satc_strength.value()),
        )

    def get_halo_reduction_params(self):
        from astraios.core.halo_reduction import (
            HaloReductionLevel,
            HaloReductionParams,
        )

        level_map = {
            "Extra Low": HaloReductionLevel.EXTRA_LOW,
            "Low": HaloReductionLevel.LOW,
            "Medium": HaloReductionLevel.MEDIUM,
            "High": HaloReductionLevel.HIGH,
        }
        return HaloReductionParams(
            reduction_level=level_map.get(self._halo_level_combo.currentText(),
                                          HaloReductionLevel.LOW),
            is_linear=self._halo_linear_check.isChecked(),
        )

    def get_wavescale_hdr_params(self):
        from astraios.core.wavescale_hdr import WaveScaleHDRParams

        return WaveScaleHDRParams(
            n_scales=int(self._wshdr_scales.value()),
            compression_factor=float(self._wshdr_compression.value()),
            mask_gamma=float(self._wshdr_mask_gamma.value()),
            decay_rate=float(self._wshdr_decay.value()),
        )

    def get_wavescale_dark_params(self):
        from astraios.core.wavescale_dark_enhance import WaveScaleDarkEnhanceParams

        return WaveScaleDarkEnhanceParams(
            n_scales=int(self._wsde_scales.value()),
            boost_factor=float(self._wsde_boost.value()),
            mask_gamma=float(self._wsde_mask_gamma.value()),
            iterations=int(self._wsde_iterations.value()),
            decay_rate=float(self._wsde_decay.value()),
        )

    def get_texture_clarity_params(self):
        from astraios.core.texture_clarity import TextureClarityParams

        return TextureClarityParams(
            texture_amount=float(self._txc_texture_amount.value()),
            texture_radius=float(self._txc_texture_radius.value()),
            clarity_amount=float(self._txc_clarity_amount.value()),
            clarity_radius=float(self._txc_clarity_radius.value()),
            mask_strength=float(self._txc_mask_strength.value()),
        )

    def get_blemish_params(self):
        from astraios.core.blemish import BlemishParams

        return BlemishParams(
            radius=int(self._blemish_radius.value()),
            feather=float(self._blemish_feather.value()),
            opacity=float(self._blemish_opacity.value()),
        )

    _SELC_FAMILY_ARCS = {
        "Reds": [(330.0, 30.0)],
        "Yellows": [(30.0, 90.0)],
        "Greens": [(90.0, 150.0)],
        "Cyans": [(150.0, 210.0)],
        "Blues": [(210.0, 270.0)],
        "Magentas": [(270.0, 330.0)],
    }

    def get_selective_color_params(self):
        from astraios.core.selective_adjust import SelectiveColorParams

        adj = {k: float(s.value()) for k, s in self._selc_adjust.items()}
        return SelectiveColorParams(
            hue_ranges=self._SELC_FAMILY_ARCS.get(
                self._selc_family_combo.currentText(), [(330.0, 30.0)]
            ),
            smooth_deg=float(self._selc_smooth.value()),
            intensity=float(self._selc_intensity.value()),
            min_chroma=float(self._selc_min_chroma.value()),
            edge_blur=float(self._selc_edge_blur.value()),
            **adj,
        )

    def get_selective_luma_params(self):
        from astraios.core.selective_adjust import SelectiveLumaParams

        ranges = {
            "Shadows": (0.0, 0.25),
            "Midtones": (0.25, 0.75),
            "Highlights": (0.75, 1.0),
        }
        choice = self._sell_range_combo.currentText()
        if choice == "Custom":
            lo, hi = float(self._sell_lo.value()), float(self._sell_hi.value())
        else:
            lo, hi = ranges.get(choice, (0.0, 0.25))
        adj = {k: float(s.value()) for k, s in self._sell_adjust.items()}
        return SelectiveLumaParams(
            lo=lo, hi=hi,
            smooth=float(self._sell_smooth.value()),
            intensity=float(self._sell_intensity.value()),
            edge_blur=float(self._sell_edge_blur.value()),
            **adj,
        )

    def get_pedestal_params(self):
        from astraios.core.pedestal import PedestalParams

        mode = "remove" if self._ped_mode_combo.currentText() == "Remove" else "add"
        amount = float(self._ped_amount.value())
        return PedestalParams(
            mode=mode,
            per_channel=self._ped_per_channel.isChecked(),
            amount=amount,
            # In Remove mode, 0 means auto-detect (core expects None for that)
            remove_amount=(amount if mode == "remove" and amount > 0 else None),
            clip=self._ped_clip.isChecked(),
        )

    def _show_training_guide(self):
        """Explain how to train a personal denoise model (button was dead)."""
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.information(
            self,
            "Train Your Own Denoise Model",
            "Astraios can fine-tune the AI denoiser on your own raw subs\n"
            "using self-supervised Noise2Self training (no clean targets\n"
            "needed).\n\n"
            "1. Collect a folder of your raw light frames (FITS).\n"
            "2. From the Astraios source checkout, run:\n\n"
            "   poetry run python scripts/train_denoise_model.py \\\n"
            "       --input /path/to/your/lights --epochs 30\n\n"
            "3. Point Preferences > AI Models > Denoise model at the\n"
            "   resulting .pt file.\n\n"
            "Training benefits from a CUDA GPU; expect roughly an hour\n"
            "for 30 epochs on a mid-range card.",
        )

    # ── TAB 9: Utility ────────────────────────────────────
    def _build_utility_tab(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # Pixel Math
        pm = CollapsibleSection(
            "Pixel Math", accent=True,
            help_text="Write a custom mathematical expression evaluated per "
                      "pixel, for combining, masking, or building synthetic "
                      "channels beyond what a fixed dialog offers. Variables: "
                      "R, G, B, L for the current image's channels/luminance, "
                      "img1/img2 for a second loaded image.",
        )
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
        ez = CollapsibleSection(
            "EZ Script Suite",
            help_text="One-click preset processing chains for common "
                      "workflows (OSC, Narrowband, Luminance-only, etc.) — "
                      "runs a pre-built sequence of steps instead of applying "
                      "each tool by hand.",
        )
        ez.add_info("One-click processing presets (OSC, Narrowband, Luminance, etc.)")
        ez.add_run("⊞ Open EZ Scripts…", self.open_ez_scripts.emit, flat=True)
        lay.addWidget(ez)

        # HDR
        hdr = CollapsibleSection(
            "HDR Composition",
            help_text="Merges several exposures of the same target taken at "
                      "different exposure lengths into one image with "
                      "extended dynamic range, so both the faint outer "
                      "regions and a bright core (a galaxy nucleus, the "
                      "Trapezium) are well exposed together.",
        )
        hdr.add_info("Merge differently-exposed images for extended dynamic range.")
        hdr.add_run("⊞ Open HDR Dialog…", self.open_hdr_dialog.emit, flat=True)
        lay.addWidget(hdr)

        # Blink
        blink = CollapsibleSection(
            "Blink Comparator",
            help_text="Rapidly alternates between two loaded images so your "
                      "eye catches differences a static side-by-side "
                      "comparison would miss — useful for spotting a moving "
                      "asteroid or comet, judging before/after processing, or "
                      "checking frame-to-frame focus and tracking changes.",
        )
        blink.add_info("Rapidly alternate between two images to spot differences.")
        btns = blink.add_btn_row([("Load A…", True), ("Load B…", True)])
        btns[0].clicked.connect(self.blink_load_a.emit)
        btns[1].clicked.connect(self.blink_load_b.emit)
        btns2 = blink.add_btn_row([("Current → A", True), ("Current → B", True)])
        btns2[0].clicked.connect(self.blink_use_current_as_a.emit)
        btns2[1].clicked.connect(self.blink_use_current_as_b.emit)
        self._blink_fps = blink.add_slider(
            "FPS", 2, 1, 10, 1, 0,
            help_text="How many times per second the display swaps between "
                      "image A and B while blinking.",
        )
        self._blink_fps.value_changed.connect(lambda v: self.blink_fps_changed.emit(int(v)))
        self._btn_blink_toggle = blink.add_run("▶ Start Blinking")
        self._blink_active = False
        self._btn_blink_toggle.clicked.connect(self._on_blink_toggle)
        lay.addWidget(blink)

        # Macros
        mac = CollapsibleSection(
            "Macros / Scripting",
            help_text="Records a sequence of tool actions as you perform "
                      "them, then replays that exact sequence on the same or "
                      "a different image later — a way to repeat a "
                      "processing recipe without redoing every step by hand.",
        )
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
        ph = CollapsibleSection(
            "Processing History (Non-Destructive)",
            help_text="Shows every processing step applied so far as an "
                      "editable, replayable list. Editing or removing an "
                      "earlier step automatically re-runs everything "
                      "downstream of it, so you can revisit an old decision "
                      "without starting over.",
        )
        ph.add_info("View and edit processing steps. Changes cascade and invalidate downstream steps.")
        ph.add_run("⊞ Open Processing History…", self.open_processing_graph.emit, flat=True)
        lay.addWidget(ph)

        # Analysis
        an = CollapsibleSection(
            "Analysis Tools",
            help_text="Diagnostic tools for judging optical and tracking "
                      "quality: FWHM Map shows star sharpness across the "
                      "field (spotting tilt or coma), Tilt/Aberrations "
                      "compares corner-to-corner focus, and Photometry "
                      "measures star brightness for variable-star or "
                      "exoplanet work.",
        )
        an.add_info("Diagnose optical quality (FWHM, tilt, aberrations).")
        btns = an.add_btn_row([("FWHM Map", True), ("Tilt/Aberrations", True), ("Photometry", True)])
        btns[0].clicked.connect(self.open_analysis_fwhm.emit)
        btns[1].clicked.connect(self.open_analysis_tilt.emit)
        btns[2].clicked.connect(self.open_analysis_photometry.emit)
        lay.addWidget(an)

        # Python Console
        con = CollapsibleSection(
            "Python Console",
            help_text="Opens a live Python console with full access to the "
                      "Astraios core processing API, for scripting custom "
                      "operations, batch automation, or experiments beyond "
                      "what the UI exposes.",
        )
        con.add_info("Full Python access to the Astraios core API.")
        con.add_run("⊞ Open Python Console…", self.open_python_console.emit, flat=True)
        lay.addWidget(con)

        # Overlays
        ov = CollapsibleSection(
            "Overlays",
            help_text="Toggles informational overlays drawn on top of the "
                      "canvas: WCS shows the coordinate grid from "
                      "plate-solving, DSO Annotations labels known deep-sky "
                      "objects in the field, and Constellations draws "
                      "constellation lines and names.",
        )
        ov.add_info("Toggle WCS, DSO, and constellation overlays on the canvas.")
        btns = ov.add_btn_row([("WCS Overlay", False), ("DSO Annotations", False), ("Constellations", False)])
        for b in btns:
            b.setCheckable(True)
        btns[0].clicked.connect(self.toggle_wcs_overlay)
        btns[1].clicked.connect(self.toggle_dso_overlay)
        btns[2].clicked.connect(self.toggle_constellation_overlay)
        lay.addWidget(ov)

        # Statistics
        stats = CollapsibleSection(
            "Image Statistics",
            help_text="Displays numeric statistics for the current image — "
                      "mean, median, standard deviation, min/max, and "
                      "histogram percentiles — for judging exposure, "
                      "background level, and dynamic range without "
                      "eyeballing the histogram.",
        )
        stats.add_info("Mean, median, SD, min, max, histogram percentiles.")
        stats.add_run("⊞ Show Statistics…", self.show_image_statistics.emit, flat=True)
        lay.addWidget(stats)

        # FITS Header
        hdr = CollapsibleSection(
            "FITS Header",
            help_text="View and edit the FITS metadata keywords stored with "
                      "the image — exposure time, filter, gain, coordinates, "
                      "and so on. Editing values here changes what's "
                      "recorded in the file, not the pixel data.",
        )
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

    def get_multiframe_deconv_params(self):
        from astraios.core.multiframe_deconv import MultiFrameDeconvParams

        rho_map = {"Huber (robust)": "huber", "L2 (classic)": "l2"}
        color_map = {"Luma (fast)": "luma", "Per-channel": "perchannel"}
        seed_map = {
            "Robust (sigma-clip)": "robust",
            "Median": "median",
            "Mean": "mean",
            "Integrated (current image)": "integrated",
        }
        sr_map = {"1x (native)": 1, "2x": 2, "3x": 3}

        return MultiFrameDeconvParams(
            iterations=int(self._mfd_iterations_spin.value()),
            min_iterations=int(self._mfd_min_iterations_spin.value()),
            rho=rho_map.get(self._mfd_rho_combo.currentText(), "huber"),
            color_mode=color_map.get(self._mfd_color_mode_combo.currentText(), "luma"),
            seed_mode=seed_map.get(self._mfd_seed_mode_combo.currentText(), "robust"),
            kappa=float(self._mfd_kappa_spin.value()),
            relaxation=float(self._mfd_relaxation_spin.value()),
            early_stop=self._mfd_early_stop_check.isChecked(),
            super_resolution=sr_map.get(self._mfd_super_res_combo.currentText(), 1),
            low_vram=self._mfd_low_vram_check.isChecked(),
        )

    def set_ref_frame_max(self, n: int):
        """Cap the 'Specific frame #' input at the loaded frame count.

        main_window has always called this before alignment, but the method
        did not exist — aligning in-memory calibrated lights crashed with
        AttributeError before registration even started.
        """
        if n >= 1:
            self._ref_frame_spin.setMaximum(n)

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
        }
        mode = mode_map.get(
            self._reg_mode_combo.currentText(), RegistrationMode.STAR_2_PASS
        )
        ref_choice = self._ref_frame_combo.currentText()
        if ref_choice == "Specific frame #":
            ref_idx = int(self._ref_frame_spin.value()) - 1  # UI is 1-based
        else:
            ref_idx = ref_map.get(ref_choice, 0)
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
        from astraios.core.transforms import InterpolationMethod

        interp_map = {
            "lanczos": InterpolationMethod.LANCZOS,
            "bicubic": InterpolationMethod.BICUBIC,
            "bilinear": InterpolationMethod.BILINEAR,
            "nearest": InterpolationMethod.NEAREST,
        }
        return ResizeParams(
            scale=float(self._resize_scale_spin.value()),
            interpolation=interp_map.get(
                self._resize_interp_combo.currentText().lower(),
                InterpolationMethod.LANCZOS,
            ),
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
        # abe.py compares lowercase tokens; passing the display text meant
        # "RBF" never matched "rbf" and RBF mode silently ran the polynomial.
        kernel_map = {
            "Thin Plate Spline": "thin_plate_spline",
            "Multiquadric": "multiquadric",
            "Gaussian": "gaussian",
        }
        model = "rbf" if self._abe_model_combo.currentText() == "RBF" else "polynomial"
        return ABEParams(
            grid_size=int(self._abe_grid_spin.value()),
            model_type=model,
            polynomial_degree=int(self._abe_degree_spin.value()),
            rbf_kernel=kernel_map.get(
                self._abe_kernel_combo.currentText(), "thin_plate_spline"
            ),
            correction_mode=self._abe_mode_combo.currentText().lower(),
        )

    def get_morphology_params(self) -> MorphologyParams:
        from astraios.core.morphology import MorphOp, StructuringElement

        op_map = {
            "Erosion": MorphOp.ERODE, "Dilation": MorphOp.DILATE,
            "Opening": MorphOp.OPEN, "Closing": MorphOp.CLOSE,
            "Gradient": MorphOp.GRADIENT,
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
            "OSC (no filter)": "OSC (no filter)",
            "Broadband (L/R/G/B)": "Mono + Baader LRGB",
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
        solver_map = {
            "Auto": "auto",
            "GAIA (offline)": "gaia",
            "ASTAP": "astap",
            "Astrometry.net": "astrometry_net",
        }
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
