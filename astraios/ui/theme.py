"""Dark theme stylesheet for Astraios.

The visual language is the Cosmica UI design handoff (Space Grotesk for the
interface, JetBrains Mono for numbers, GitHub-dark palette, green accent) and
this module is the single source of truth for it. Everything here is measured
against the design prototype: sizes, radii, weights and colours are the
prototype's, not approximations.

The button language is the prototype's ``Btn`` component: a button is flat
(tertiary background, hairline border) unless it is the primary action of a
form, which is accent green. In Qt that maps onto ``QPushButton`` for the flat
default, ``QPushButton:default`` (the dialog's default button) and
``QPushButton[accent="true"]`` for the green primary.
"""

from __future__ import annotations

from pathlib import Path

_ICONS_DIR = Path(__file__).resolve().parent.parent / "resources" / "icons"
FONTS_DIR = Path(__file__).resolve().parent.parent / "resources" / "fonts"

# ── Design tokens (mirrors components/shared.jsx in the design handoff) ──
ACCENT = "#2ea043"
ACCENT_HOVER = "#3fb950"
ACCENT_DARK = "#1a4d2e"
ACCENT_PURPLE = "#8957e5"
ACCENT_BLUE = "#388bfd"
RED = "#f85149"
ORANGE = "#d29922"
BG_PRIMARY = "#0d1117"     # Window, canvas chrome, section bodies
BG_SECONDARY = "#161b22"   # Panels, bars, section headers
BG_TERTIARY = "#21262d"    # Inputs, flat buttons, chips
BG_HOVER = "#30363d"       # Hover state
BORDER = "#30363d"         # Every hairline
TEXT_PRIMARY = "#e6edf3"
TEXT_SECONDARY = "#8b949e"
TEXT_DIM = "#4a5260"       # Future workflow steps, disabled text

FONT_SANS = '"Space Grotesk", "Inter", "Segoe UI", "Roboto", "Ubuntu", sans-serif'
FONT_MONO = '"JetBrains Mono", "Fira Code", "Cascadia Code", "Consolas", monospace'

# Swatch options for accent color theming
ACCENT_COLORS: dict[str, tuple[str, str, str]] = {
    # name: (accent, accent_hover, accent_dark)
    "green":  ("#2ea043", "#3fb950", "#1a4d2e"),
    "blue":   ("#388bfd", "#58a6ff", "#0d3a6b"),
    "purple": ("#8957e5", "#a371f7", "#3b1d6e"),
    "gold":   ("#d29922", "#e3b341", "#6e4f00"),
    "red":    ("#f85149", "#ff7b72", "#6e1c19"),
}


def _icon(name: str) -> str:
    """Resolve icon path."""
    return (_ICONS_DIR / f"{name}.svg").as_posix()


def load_bundled_fonts() -> list[str]:
    """Register the fonts the design is built on with the application.

    The design specifies Space Grotesk and JetBrains Mono, and until these
    were bundled the stylesheet only *asked* for them: on a machine without
    them (most machines) every fallback in the list was tried and the UI
    rendered in whatever the OS offered. Both fonts are SIL Open Font
    License, which the GPL-3.0 build can ship. Returns the families loaded.
    """
    from PyQt6.QtGui import QFontDatabase

    families: list[str] = []
    if not FONTS_DIR.is_dir():
        return families
    for path in sorted(FONTS_DIR.glob("*.ttf")):
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id >= 0:
            for fam in QFontDatabase.applicationFontFamilies(font_id):
                if fam not in families:
                    families.append(fam)
    return families


def _build_stylesheet(accent: str, accent_hover: str, accent_dark: str) -> str:
    """Build the full QSS stylesheet with the given accent colors."""
    check_icon = _icon("check")
    chevron = _icon("chevron-down")
    radio_dot = _icon("radio-dot")
    return f"""
/* ═══════════════════════════════════════════════════════
   Astraios Dark Theme
   ═══════════════════════════════════════════════════════ */

/* ── Global ─────────────────────────────────────────── */
QWidget {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
    font-family: {FONT_SANS};
    font-size: 12px;
}}

QMainWindow, QDialog {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
}}

/* Transparent background on non-container widgets so parent backgrounds
   show through correctly (prevents the "highlight block" artefact). */
QLabel, QCheckBox, QRadioButton {{
    background-color: transparent;
    border: none;
}}

QLabel:disabled, QCheckBox:disabled, QRadioButton:disabled {{
    color: {TEXT_DIM};
}}

/* ── Menu Bar (36px) ────────────────────────────────── */
QMenuBar {{
    background-color: {BG_SECONDARY};
    color: {TEXT_PRIMARY};
    border-bottom: 1px solid {BORDER};
    padding: 0px 8px;
    spacing: 0px;
    min-height: 36px;
}}

QMenuBar::item {{
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 12px;
    background: transparent;
}}

QMenuBar::item:selected, QMenuBar::item:pressed {{
    background-color: {BG_HOVER};
}}

/* ── Dropdown Menus ─────────────────────────────────── */
QMenu {{
    background-color: {BG_SECONDARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px;
}}

QMenu::item {{
    padding: 5px 24px 5px 12px;
    border-radius: 4px;
    font-size: 12px;
    min-width: 200px;
}}

QMenu::item:selected {{
    background-color: {BG_HOVER};
}}

QMenu::item:disabled {{
    color: {TEXT_DIM};
}}

QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 3px 4px;
}}

QMenu::icon {{
    margin-right: 6px;
}}

QMenu::indicator {{
    width: 14px;
    height: 14px;
    margin-left: 6px;
}}

QMenu::indicator:checked {{
    image: url({check_icon});
    background: {accent};
    border-radius: 3px;
}}

/* ── Toolbars (30px quick toolbar) ──────────────────── */
QToolBar {{
    background-color: {BG_SECONDARY};
    border: none;
    border-bottom: 1px solid {BORDER};
    spacing: 2px;
    padding: 2px 8px;
    min-height: 30px;
}}

QToolBar::separator {{
    width: 1px;
    background-color: {BORDER};
    margin: 6px 3px;
}}

QToolButton {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 2px 7px;
    color: {TEXT_SECONDARY};
    font-size: 13px;
    min-height: 20px;
}}

QToolButton:hover {{
    background-color: {BG_HOVER};
    color: {TEXT_PRIMARY};
}}

QToolButton:pressed, QToolButton:checked {{
    background-color: {accent_dark};
    border-color: {accent};
    color: {accent};
}}

QToolButton:disabled {{
    color: {TEXT_DIM};
}}

QToolButton::menu-indicator {{
    image: none;
}}

/* ── Buttons (design "Btn": flat by default, accent opt-in) ── */
QPushButton {{
    background-color: {BG_TERTIARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: {BG_HOVER};
}}

QPushButton:pressed {{
    background-color: {accent_dark};
}}

QPushButton:checked {{
    background-color: {accent_dark};
    border-color: {accent};
}}

QPushButton:default, QPushButton[accent="true"] {{
    background-color: {accent};
    color: #ffffff;
    border: 1px solid transparent;
    font-weight: 600;
}}

QPushButton:default:hover, QPushButton[accent="true"]:hover {{
    background-color: {accent_hover};
}}

QPushButton:default:pressed, QPushButton[accent="true"]:pressed {{
    background-color: {accent_dark};
}}

QPushButton[danger="true"] {{
    background-color: #8b0000;
    color: #ffffff;
    border: 1px solid transparent;
    font-weight: 600;
}}

QPushButton[danger="true"]:hover {{
    background-color: #c93030;
}}

QPushButton[small="true"] {{
    padding: 3px 10px;
    font-size: 11px;
}}

QPushButton[flat="true"], QPushButton#flatButton {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
}}

QPushButton[flat="true"]:hover {{
    background-color: {BG_HOVER};
}}

QPushButton:disabled {{
    background-color: {BG_TERTIARY};
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER};
}}

/* ── Inputs ─────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateEdit, QTimeEdit, QDateTimeEdit {{
    background-color: {BG_TERTIARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 4px 8px;
    font-size: 12px;
    selection-background-color: {accent_dark};
    selection-color: {TEXT_PRIMARY};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QDateEdit:focus, QTimeEdit:focus, QDateTimeEdit:focus {{
    border-color: {accent};
}}

QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {TEXT_DIM};
}}

QLineEdit[readOnly="true"] {{
    color: {TEXT_SECONDARY};
}}

/* Numeric inputs are monospaced and right-aligned like the prototype's
   NumberInput; the spin arrows are gone, wheel and arrow keys still work. */
QSpinBox, QDoubleSpinBox {{
    font-family: {FONT_MONO};
    font-size: 11px;
}}

QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 0px;
    height: 0px;
    border: none;
}}

QComboBox {{
    padding-right: 24px;
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 22px;
    border: none;
    background: transparent;
}}

QComboBox::down-arrow {{
    image: url({chevron});
    width: 10px;
    height: 10px;
}}

QComboBox QAbstractItemView {{
    background-color: {BG_SECONDARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px;
    outline: 0;
    selection-background-color: {BG_HOVER};
    selection-color: {TEXT_PRIMARY};
}}

QComboBox QAbstractItemView::item {{
    padding: 5px 8px;
    border-radius: 4px;
    min-height: 22px;
}}

QComboBox QAbstractItemView::item:selected {{
    background-color: {BG_HOVER};
}}

/* ── Text areas ─────────────────────────────────────── */
QTextEdit, QPlainTextEdit {{
    background-color: {BG_TERTIARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 4px 6px;
    selection-background-color: {accent_dark};
    font-family: {FONT_MONO};
    font-size: 11px;
}}

QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {accent};
}}

QTextBrowser {{
    font-family: {FONT_SANS};
    font-size: 12px;
}}

/* ── Checkboxes / radios ────────────────────────────── */
QCheckBox {{
    spacing: 6px;
    font-size: 12px;
}}

QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1.5px solid {BORDER};
    border-radius: 3px;
    background-color: {BG_TERTIARY};
}}

QCheckBox::indicator:hover {{
    border-color: {accent};
}}

QCheckBox::indicator:checked {{
    background-color: {accent};
    border-color: {accent};
    image: url({check_icon});
}}

QCheckBox::indicator:disabled {{
    border-color: {BORDER};
    background-color: {BG_SECONDARY};
}}

QRadioButton {{
    spacing: 6px;
    font-size: 12px;
}}

QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border: 1.5px solid {BORDER};
    border-radius: 8px;
    background-color: {BG_TERTIARY};
}}

QRadioButton::indicator:hover {{
    border-color: {accent};
}}

QRadioButton::indicator:checked {{
    background-color: {accent};
    border-color: {accent};
    image: url({radio_dot});
}}

/* ── Sliders ────────────────────────────────────────── */
QSlider {{
    background: transparent;
    border: none;
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {BG_TERTIARY};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {accent};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    border: 2px solid {BG_PRIMARY};
}}

QSlider::handle:horizontal:hover {{
    background: {accent_hover};
}}

QSlider::sub-page:horizontal {{
    background: {accent};
    border-radius: 2px;
}}

QSlider::groove:vertical {{
    width: 4px;
    background: {BG_TERTIARY};
    border-radius: 2px;
}}

QSlider::handle:vertical {{
    background: {accent};
    width: 14px;
    height: 14px;
    margin: 0 -5px;
    border-radius: 7px;
    border: 2px solid {BG_PRIMARY};
}}

/* ── Progress Bar ───────────────────────────────────── */
QProgressBar {{
    background-color: {BG_TERTIARY};
    border: none;
    border-radius: 3px;
    text-align: center;
    color: {TEXT_PRIMARY};
    font-family: {FONT_MONO};
    font-size: 10px;
    min-height: 6px;
}}

QProgressBar::chunk {{
    background-color: {accent};
    border-radius: 3px;
}}

/* ── Scrollbars (6px, track-less) ───────────────────── */
QScrollBar:vertical {{
    background: transparent;
    width: 6px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background-color: {BG_HOVER};
    min-height: 20px;
    border-radius: 3px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {TEXT_SECONDARY};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 6px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background-color: {BG_HOVER};
    min-width: 20px;
    border-radius: 3px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {TEXT_SECONDARY};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
}}

QScrollArea {{
    border: none;
    background: transparent;
}}

QAbstractScrollArea::corner {{
    background: transparent;
}}

/* ── Tabs (dialogs; the Tools panel draws its own strip) ── */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background-color: {BG_PRIMARY};
    top: -1px;
}}

QTabBar {{
    background: transparent;
}}

QTabBar::tab {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    padding: 7px 12px;
    border: none;
    border-bottom: 2px solid transparent;
    margin-right: 2px;
    font-size: 11px;
    font-weight: 600;
}}

QTabBar::tab:selected {{
    color: {accent};
    border-bottom: 2px solid {accent};
}}

QTabBar::tab:hover:!selected {{
    color: {TEXT_PRIMARY};
}}

QTabBar::scroller {{
    width: 40px;
}}

QTabBar QToolButton {{
    background-color: {BG_TERTIARY};
    border: 1px solid {BORDER};
    border-radius: 4px;
    color: {TEXT_PRIMARY};
    padding: 0px;
    margin: 1px;
}}

/* ── Trees, Lists, Tables ───────────────────────────── */
QTreeWidget, QListWidget, QTableWidget, QTreeView, QListView, QTableView {{
    background-color: {BG_SECONDARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    alternate-background-color: {BG_PRIMARY};
    selection-background-color: {accent_dark};
    selection-color: {TEXT_PRIMARY};
    outline: none;
    padding: 2px;
    font-size: 11px;
}}

QTreeWidget::item, QListWidget::item, QTreeView::item, QListView::item {{
    padding: 4px 6px;
    border-radius: 4px;
}}

QTreeWidget::item:hover, QListWidget::item:hover, QTreeView::item:hover, QListView::item:hover {{
    background-color: {BG_HOVER};
}}

QTreeWidget::item:selected, QListWidget::item:selected,
QTreeView::item:selected, QListView::item:selected {{
    background-color: {accent_dark};
    color: {TEXT_PRIMARY};
}}

QTableWidget::item:selected, QTableView::item:selected {{
    background-color: {accent_dark};
    color: {TEXT_PRIMARY};
}}

QHeaderView::section {{
    background-color: {BG_SECONDARY};
    color: {TEXT_SECONDARY};
    padding: 5px 8px;
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
    font-size: 10px;
    text-transform: uppercase;
}}

QTableCornerButton::section {{
    background-color: {BG_SECONDARY};
    border: none;
}}

/* ── Group Boxes ────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 16px;
    padding-top: 10px;
    font-weight: 600;
    font-size: 11px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: {TEXT_SECONDARY};
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}

/* ── Splitters (hairline, accent on hover) ──────────── */
QSplitter::handle {{
    background-color: {BORDER};
}}

QSplitter::handle:horizontal {{
    width: 1px;
}}

QSplitter::handle:vertical {{
    height: 1px;
}}

QSplitter::handle:hover {{
    background-color: {accent};
}}

/* ── Status Bar (22px) ──────────────────────────────── */
QStatusBar {{
    background-color: {BG_SECONDARY};
    color: {TEXT_SECONDARY};
    font-size: 10px;
    border-top: 1px solid {BORDER};
    padding: 0px 12px;
    min-height: 22px;
}}

QStatusBar::item {{
    border: none;
}}

/* ── Dock Widgets ───────────────────────────────────── */
QDockWidget {{
    color: {TEXT_PRIMARY};
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}

QDockWidget::title {{
    background-color: {BG_SECONDARY};
    padding: 6px 10px;
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}

/* ── Tooltips ───────────────────────────────────────── */
QToolTip {{
    background-color: {BG_SECONDARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11px;
}}

/* ── Message boxes / dialogs ────────────────────────── */
QMessageBox {{
    background-color: {BG_PRIMARY};
}}

QMessageBox QLabel {{
    font-size: 12px;
}}

QDialogButtonBox QPushButton {{
    min-width: 72px;
}}

/* ── Pro Feature Accent ─────────────────────────────── */
QPushButton#proButton {{
    background-color: {ACCENT_PURPLE};
    color: #ffffff;
    border: 1px solid transparent;
}}

QPushButton#proButton:hover {{
    background-color: #a371f7;
}}

/* ── Canvas area (image display background) ─────────── */
#ImageCanvas {{
    background-color: #000000;
    border: none;
}}

/* Canvas toolbar buttons are the design's "Btn small flat" */
#CanvasToolbar {{
    background-color: {BG_SECONDARY};
    border-bottom: 1px solid {BORDER};
}}

#CanvasToolbar QToolButton {{
    background-color: {BG_TERTIARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 2px 10px;
    color: {TEXT_PRIMARY};
    font-size: 11px;
    font-weight: 500;
    min-height: 18px;
}}

#CanvasToolbar QToolButton:hover {{
    background-color: {BG_HOVER};
}}

#CanvasToolbar QToolButton:checked {{
    background-color: {accent_dark};
    border-color: {BORDER};
    color: {TEXT_PRIMARY};
}}

#CanvasToolbar QToolButton:pressed {{
    background-color: {accent_dark};
}}

/* Toggles whose state is already in their label (the histogram
   disclosure) stay flat when checked instead of lighting up green. */
#CanvasToolbar QToolButton[quiet="true"]:checked {{
    background-color: {BG_TERTIARY};
    border-color: {BORDER};
}}

/* Histogram strip under the canvas */
#HistogramPanel {{
    background-color: {BG_SECONDARY};
    border-top: 1px solid {BORDER};
}}

#HistogramPanel QPushButton {{
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    color: {TEXT_SECONDARY};
    font-size: 10px;
    font-weight: 600;
    padding: 3px 10px;
}}

#HistogramPanel QPushButton:hover {{
    color: {TEXT_PRIMARY};
}}

#HistogramPanel QPushButton:checked {{
    color: {accent};
    border-bottom: 2px solid {accent};
}}

/* Pill chips in the menu bar (GPU / RAM) */
QLabel#StatusChip {{
    background-color: {BG_TERTIARY};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 2px 10px;
    color: {TEXT_SECONDARY};
    font-family: {FONT_MONO};
    font-size: 10px;
}}

/* ── Workflow Bar ───────────────────────────────────── */
#WorkflowBar {{
    background-color: {BG_SECONDARY};
    border-bottom: 1px solid {BORDER};
}}

/* ── Tweaks Panel ───────────────────────────────────── */
#TweaksPanel {{
    background-color: {BG_SECONDARY};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
"""


def get_dark_theme() -> str:
    """Return the complete dark theme stylesheet with default accent."""
    return _build_stylesheet(ACCENT, ACCENT_HOVER, ACCENT_DARK)


def set_accent(color_name: str) -> str:
    """Return a stylesheet with the given accent color applied.

    Parameters
    ----------
    color_name : str
        One of the keys in ACCENT_COLORS ('green', 'blue', 'purple', 'gold', 'red')
        OR a raw hex color string (e.g. '#388bfd').
    """
    if color_name in ACCENT_COLORS:
        accent, accent_hover, accent_dark = ACCENT_COLORS[color_name]
    else:
        # Raw hex — derive hover/dark by darkening slightly
        accent = color_name
        accent_hover = color_name
        accent_dark = color_name
    return _build_stylesheet(accent, accent_hover, accent_dark)


# Backward compatibility
DARK_THEME = get_dark_theme()
