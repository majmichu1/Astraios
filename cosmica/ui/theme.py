"""Dark theme stylesheet for Cosmica — professional-grade dark theme."""

from __future__ import annotations

from pathlib import Path

_ICONS_DIR = Path(__file__).resolve().parent.parent / "resources" / "icons"

# Accent palette — GitHub-inspired green with refined tones
ACCENT = "#2ea043"
ACCENT_HOVER = "#3fb950"
ACCENT_DARK = "#1a4d2e"
BG_PRIMARY = "#0d1117"     # Deepest background
BG_SECONDARY = "#161b22"   # Panels, sidebars
BG_TERTIARY = "#21262d"    # Inputs, buttons
BG_HOVER = "#30363d"       # Hover state
BORDER = "#30363d"         # Borders
TEXT_PRIMARY = "#e6edf3"   # Primary text
TEXT_SECONDARY = "#8b949e" # Secondary text, hints

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


def _build_stylesheet(accent: str, accent_hover: str, accent_dark: str) -> str:
    """Build the full QSS stylesheet with the given accent colors."""
    return f"""
/* ═══════════════════════════════════════════════════════
   Cosmica Dark Theme — Professional Astrophotography UI
   ═══════════════════════════════════════════════════════ */

/* ── Global ─────────────────────────────────────────── */
QMainWindow, QDialog {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
}}

QWidget {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
    font-family: "Space Grotesk", "Inter", "Segoe UI", "Roboto", "Ubuntu", sans-serif;
    font-size: 13px;
}}

/* Transparent background on non-container widgets so parent backgrounds
   show through correctly — prevents the "highlight block" artefact */
QLabel, QCheckBox, QRadioButton {{
    background-color: transparent;
    border: none;
}}

/* ── Menu Bar ───────────────────────────────────────── */
QMenuBar {{
    background-color: {BG_SECONDARY};
    color: {TEXT_PRIMARY};
    border-bottom: 1px solid {BORDER};
    padding: 0px 4px;
    spacing: 2px;
    min-height: 32px;
}}

QMenuBar::item {{
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
}}

QMenuBar::item:selected {{
    background-color: {BG_HOVER};
}}

QMenuBar::item:pressed {{
    background-color: {accent_dark};
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
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
    font-size: 12px;
}}

QMenu::item:selected {{
    background-color: {BG_HOVER};
}}

QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 8px;
}}

QMenu::icon {{
    margin-right: 6px;
}}

QMenu::shortcut {{
    color: {TEXT_SECONDARY};
    margin-left: 24px;
    font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", "Consolas", monospace;
    font-size: 11px;
}}

/* ── Toolbars ───────────────────────────────────────── */
QToolBar {{
    background-color: {BG_SECONDARY};
    border: none;
    border-bottom: 1px solid {BORDER};
    spacing: 1px;
    padding: 2px 6px;
}}

QToolBar::separator {{
    width: 1px;
    background-color: {BORDER};
    margin: 4px 4px;
}}

QToolButton {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 3px 6px;
    color: {TEXT_PRIMARY};
    font-size: 13px;
    min-width: 24px;
    min-height: 22px;
}}

QToolButton:hover {{
    background-color: {BG_HOVER};
    border-color: {BORDER};
}}

QToolButton:pressed {{
    background-color: {accent_dark};
}}

QToolButton:checked {{
    background-color: {accent_dark};
    border-color: {accent};
    color: {accent};
}}

QToolButton:disabled {{
    color: {TEXT_SECONDARY};
}}

/* ── Status Bar ─────────────────────────────────────── */
QStatusBar {{
    background-color: {BG_SECONDARY};
    color: {TEXT_SECONDARY};
    font-size: 11px;
    font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", "Consolas", monospace;
    border-top: 1px solid {BORDER};
    padding: 1px 8px;
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
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* ── Tabs ───────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    background-color: {BG_PRIMARY};
    top: -1px;
}}

QTabBar::tab {{
    background-color: {BG_SECONDARY};
    color: {TEXT_SECONDARY};
    padding: 8px 16px;
    border: 1px solid {BORDER};
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    margin-right: 2px;
    font-weight: 500;
}}

QTabBar::tab:selected {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
    border-bottom: 2px solid {accent};
}}

QTabBar::tab:hover:!selected {{
    background-color: {BG_HOVER};
    color: {TEXT_PRIMARY};
}}

QTabBar::scroller {{
    width: 52px;
}}

QTabBar QToolButton {{
    background-color: {BG_TERTIARY};
    border: 1px solid {BORDER};
    border-radius: 4px;
    color: {TEXT_PRIMARY};
    width: 22px;
    height: 22px;
    padding: 0px;
    margin: 1px;
    font-size: 12px;
}}

QTabBar QToolButton:hover {{
    background-color: {BG_HOVER};
    color: #ffffff;
}}

/* ── Trees, Lists, Tables ───────────────────────────── */
QTreeWidget, QListWidget, QTableWidget {{
    background-color: {BG_SECONDARY};
    border: 1px solid {BORDER};
    border-radius: 4px;
    alternate-background-color: {BG_PRIMARY};
    selection-background-color: {accent_dark};
    selection-color: {TEXT_PRIMARY};
    outline: none;
    padding: 2px;
}}

QTreeWidget::item, QListWidget::item {{
    padding: 4px 6px;
    border-radius: 3px;
}}

QTreeWidget::item:hover, QListWidget::item:hover {{
    background-color: {BG_HOVER};
}}

QHeaderView::section {{
    background-color: {BG_SECONDARY};
    color: {TEXT_SECONDARY};
    padding: 6px 8px;
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
}}

/* ── Buttons ────────────────────────────────────────── */
QPushButton {{
    background-color: {accent};
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 13px;
}}

QPushButton:hover {{
    background-color: {accent_hover};
}}

QPushButton:pressed {{
    background-color: {accent_dark};
}}

QPushButton:disabled {{
    background-color: {BG_TERTIARY};
    color: {TEXT_SECONDARY};
}}

QPushButton[flat="true"], QPushButton#flatButton {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 4px;
}}

QPushButton[flat="true"]:hover {{
    background-color: {BG_HOVER};
}}

/* ── Inputs ─────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {BG_TERTIARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: {accent_dark};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {accent};
    outline: none;
}}

QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}

QComboBox QAbstractItemView {{
    background-color: {BG_SECONDARY};
    color: {TEXT_PRIMARY};
    selection-background-color: {accent_dark};
    border: 1px solid {BORDER};
    border-radius: 4px;
}}

/* ── Sliders ────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 4px;
    background: {BG_TERTIARY};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {accent};
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
    border: 2px solid {BG_PRIMARY};
}}

QSlider::handle:horizontal:hover {{
    background: {accent_hover};
}}

QSlider::sub-page:horizontal {{
    background: {accent};
    border-radius: 2px;
}}

/* ── Progress Bar ───────────────────────────────────── */
QProgressBar {{
    background-color: {BG_TERTIARY};
    border: none;
    border-radius: 4px;
    text-align: center;
    color: {TEXT_PRIMARY};
    height: 20px;
    font-weight: 500;
}}

QProgressBar::chunk {{
    background-color: {accent};
    border-radius: 4px;
}}

/* ── Scrollbars ─────────────────────────────────────── */
QScrollBar:vertical {{
    background-color: {BG_PRIMARY};
    width: 8px;
    margin: 0;
    border-radius: 4px;
}}

QScrollBar::handle:vertical {{
    background-color: {BG_HOVER};
    min-height: 20px;
    border-radius: 4px;
    margin: 2px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {TEXT_SECONDARY};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {BG_PRIMARY};
    height: 8px;
    margin: 0;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal {{
    background-color: {BG_HOVER};
    min-width: 20px;
    border-radius: 4px;
    margin: 2px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {TEXT_SECONDARY};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── Tooltips ───────────────────────────────────────── */
QToolTip {{
    background-color: {BG_SECONDARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
}}

/* ── Splitters ──────────────────────────────────────── */
QSplitter::handle {{
    background-color: {BORDER};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

QSplitter::handle:hover {{
    background-color: {accent};
}}

/* ── Group Boxes ────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 14px;
    font-weight: 600;
    font-size: 12px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {TEXT_SECONDARY};
}}

/* ── Checkboxes ─────────────────────────────────────── */
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 1.5px solid {BORDER};
    border-radius: 4px;
    background-color: {BG_TERTIARY};
}}

QCheckBox::indicator:checked {{
    background-color: {accent};
    border-color: {accent};
    image: url({_icon('check')});
}}

QCheckBox::indicator:hover {{
    border-color: {accent};
}}

/* ── Text Edits ─────────────────────────────────────── */
QTextEdit, QPlainTextEdit {{
    background-color: {BG_PRIMARY};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 4px;
    selection-background-color: {accent_dark};
    font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
}}

/* ── Pro Feature Accent ─────────────────────────────── */
QPushButton#proButton {{
    background-color: #8957e5;
    color: #ffffff;
}}

QPushButton#proButton:hover {{
    background-color: #a371f7;
}}

/* ── Canvas area (image display background) ─────────── */
#ImageCanvas {{
    background-color: #000000;
    border: 1px solid {BORDER};
}}

/* ── Workflow Bar ───────────────────────────────────── */
#WorkflowBar {{
    background-color: {BG_SECONDARY};
    border-bottom: 1px solid {BORDER};
}}

#WorkflowStep {{
    background-color: transparent;
    border: none;
    border-radius: 0;
    color: {TEXT_SECONDARY};
    font-size: 11px;
    padding: 4px 8px;
    text-align: center;
}}

#WorkflowStep:hover {{
    color: {TEXT_PRIMARY};
    background-color: {BG_HOVER};
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
