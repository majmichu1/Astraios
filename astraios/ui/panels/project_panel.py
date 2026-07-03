"""Project Panel — left sidebar showing project files and history."""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from astraios.core.image_io import FrameType
from astraios.core.project import Project
from astraios.ui.theme import (
    ACCENT,
    ACCENT_DARK,
    BG_HOVER,
    BG_SECONDARY,
    BG_TERTIARY,
    BORDER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

log = logging.getLogger(__name__)

FRAME_TYPE_LABELS = {
    FrameType.LIGHT: "Light Frames",
    FrameType.DARK: "Dark Frames",
    FrameType.FLAT: "Flat Frames",
    FrameType.BIAS: "Bias / Offset",
    FrameType.MASTER_DARK: "Master Dark",
    FrameType.MASTER_FLAT: "Master Flat",
    FrameType.MASTER_BIAS: "Master Bias",
    FrameType.CALIBRATED: "Calibrated Lights",
    FrameType.ALIGNED: "Aligned Frames",
    FrameType.RESULT: "Results",
}

_TYPE_COLOR = {
    "LIGHT":  (ACCENT,    ACCENT_DARK),
    "DARK":   (TEXT_SECONDARY, "#1c1c1c"),
    "FLAT":   ("#388bfd", "#0d1a2d"),
    "BIAS":   ("#d29922", "#2d1f00"),
    "CALIBRATED": ("#3fb950", "#0d2818"),
    "ALIGNED": ("#58a6ff", "#0d2040"),
    "RESULT": ("#a371f7", "#2d1060"),
}

FILE_FILTERS = (
    "FITS Files (*.fit *.fits *.fts);;"
    "XISF Files (*.xisf);;"
    "TIFF Files (*.tif *.tiff);;"
    "All Files (*)"
)

_FILTER_TABS = ["ALL", "LIGHT", "DARK", "FLAT"]


class _FrameRow(QFrame):
    """Single frame row widget in the frame list."""

    clicked = pyqtSignal(str)  # emits file path
    context_requested = pyqtSignal(str)  # emits file path

    def __init__(self, entry, parent=None):
        super().__init__(parent)
        self._path = str(entry.path)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._selected = False
        self._apply_style(False)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda _: self.context_requested.emit(self._path))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(1)

        # Top row: filename + type badge
        top_row = QHBoxLayout()
        top_row.setSpacing(4)

        fname = entry.path.name
        name_lbl = QLabel(fname)
        name_lbl.setStyleSheet(
            f"color: {'#ff6b6b' if not entry.path.exists() else TEXT_PRIMARY};"
            "font-family: 'JetBrains Mono', 'Cascadia Code', monospace; font-size: 10px;"
        )
        name_lbl.setToolTip(str(entry.path))
        name_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        top_row.addWidget(name_lbl)

        type_name = entry.frame_type.name
        t_color, t_bg = _TYPE_COLOR.get(type_name, (TEXT_SECONDARY, BG_TERTIARY))
        badge = QLabel(type_name)
        badge.setStyleSheet(
            f"color: {t_color}; background-color: {t_bg};"
            "font-size: 8px; font-weight: 700; padding: 1px 5px;"
            "border-radius: 8px;"
        )
        badge.setFixedHeight(14)
        top_row.addWidget(badge)
        layout.addLayout(top_row)

        # Bottom row: SNR, FWHM, warning
        meta = entry.metadata
        snr = meta.get("snr") or meta.get("quality_score")
        fwhm = meta.get("fwhm")
        if snr is not None or fwhm is not None:
            bot_row = QHBoxLayout()
            bot_row.setSpacing(8)
            if snr is not None:
                snr_lbl = QLabel(
                    f'SNR <span style="color:#388bfd;font-family:monospace">{snr:.1f}</span>'
                )
                snr_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; background-color: transparent;")
                bot_row.addWidget(snr_lbl)
            if fwhm is not None:
                fwhm_color = "#f85149" if fwhm > 4.0 else TEXT_PRIMARY
                fwhm_lbl = QLabel(
                    f'FWHM <span style="color:{fwhm_color};font-family:monospace">{fwhm:.1f}"</span>'
                )
                fwhm_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; background-color: transparent;")
                bot_row.addWidget(fwhm_lbl)
            if fwhm is not None and fwhm > 4.0:
                warn = QLabel("⚠")
                warn.setStyleSheet("color: #d29922; font-size: 10px; background-color: transparent;")
                bot_row.addWidget(warn)
            bot_row.addStretch()
            layout.addLayout(bot_row)

    def _apply_style(self, selected: bool):
        if selected:
            self.setStyleSheet(
                f"QFrame {{ background-color: {ACCENT_DARK}; border: 1px solid {ACCENT};"
                "border-radius: 5px; }"
            )
        else:
            self.setStyleSheet(
                "QFrame { background-color: transparent; border: 1px solid transparent;"
                "border-radius: 5px; }"
            )

    def set_selected(self, sel: bool):
        self._selected = sel
        self._apply_style(sel)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._path)
        super().mousePressEvent(event)

    def enterEvent(self, event):
        if not self._selected:
            self.setStyleSheet(
                f"QFrame {{ background-color: {BG_HOVER}; border: 1px solid transparent;"
                "border-radius: 5px; }"
            )
        super().enterEvent(event)

    def leaveEvent(self, event):
        if not self._selected:
            self._apply_style(False)
        super().leaveEvent(event)


class ProjectPanel(QWidget):
    """Left panel: file list with filter/search, plate solve, history."""

    frame_selected = pyqtSignal(str)    # file path
    frames_imported = pyqtSignal(list, object)  # paths, FrameType
    plate_solve_clicked = pyqtSignal()
    dso_overlay_clicked = pyqtSignal()
    show_statistics = pyqtSignal()
    show_fits_header = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self.setMaximumWidth(400)
        self.setStyleSheet(f"QWidget {{ background-color: {BG_SECONDARY}; }}")

        self._project: Project | None = None
        self._active_filter = "ALL"
        self._search_text = ""
        self._selected_path: str | None = None
        self._frame_rows: list[_FrameRow] = []
        # Fallback source for the History tree when no project is loaded
        # (main window supplies the non-destructive editing history).
        self._fallback_history_provider = None
        self._history_signature: tuple = ()

        # WCS info cache
        self._wcs_ra = ""
        self._wcs_dec = ""
        self._wcs_scale = ""
        self._wcs_pa = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        header = QWidget()
        header.setStyleSheet(f"QWidget#pp_header {{ background-color: {BG_SECONDARY}; }}")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(12, 10, 12, 8)
        header_layout.setSpacing(6)

        # Title row
        title_row = QHBoxLayout()
        title_lbl = QLabel("PROJECT")
        title_lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 700;"
        )
        title_row.addWidget(title_lbl)
        title_row.addStretch()
        btn_new = QPushButton("＋")
        btn_new.setFixedSize(22, 22)
        btn_new.setToolTip("New Project")
        btn_new.setStyleSheet(
            f"QPushButton {{ color: {TEXT_SECONDARY}; background: transparent; border: none;"
            "font-size: 13px; }"
            f"QPushButton:hover {{ color: {TEXT_PRIMARY}; }}"
        )
        btn_open = QPushButton("⌂")
        btn_open.setFixedSize(22, 22)
        btn_open.setToolTip("Open Project")
        btn_open.setStyleSheet(btn_new.styleSheet())
        title_row.addWidget(btn_new)
        title_row.addWidget(btn_open)
        header_layout.addLayout(title_row)

        # Project name box
        self._name_box = QWidget()
        self._name_box.setStyleSheet(
            f"border: 1px solid {BORDER}; border-radius: 5px;"
        )
        name_box_layout = QVBoxLayout(self._name_box)
        name_box_layout.setContentsMargins(8, 5, 8, 5)
        name_box_layout.setSpacing(1)
        self._proj_name_lbl = QLabel("No Project")
        self._proj_name_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; background-color: transparent;")
        self._proj_stats_lbl = QLabel("")
        self._proj_stats_lbl.setStyleSheet(
            f"color: {ACCENT}; font-size: 10px;"
            "font-family: 'JetBrains Mono', 'Cascadia Code', monospace;"
        )
        name_box_layout.addWidget(self._proj_name_lbl)
        name_box_layout.addWidget(self._proj_stats_lbl)
        header_layout.addWidget(self._name_box)

        # Import buttons
        import_row = QHBoxLayout()
        import_row.setSpacing(4)
        btn_lights = QPushButton("+ Lights")
        btn_lights.setFixedHeight(26)
        btn_lights.setStyleSheet(
            f"QPushButton {{ background-color: {ACCENT}; color: #fff; border: none;"
            "border-radius: 4px; font-size: 10px; font-weight: 600; }"
            f"QPushButton:hover {{ background-color: #3fb950; }}"
        )
        btn_lights.clicked.connect(lambda: self._import_frames(FrameType.LIGHT))
        btn_cals = QPushButton("+ Cals")
        btn_cals.setFixedHeight(26)
        btn_cals.setStyleSheet(
            f"QPushButton {{ color: {TEXT_PRIMARY}; background: transparent;"
            f"border: 1px solid {BORDER}; border-radius: 4px; font-size: 10px; }}"
            f"QPushButton:hover {{ background-color: {BG_HOVER}; }}"
        )
        btn_cals.clicked.connect(self._import_cals)
        btn_auto = QPushButton("Auto…")
        btn_auto.setFixedHeight(26)
        btn_auto.setToolTip("Auto-import folder (detect frame types from FITS headers)")
        btn_auto.setStyleSheet(btn_cals.styleSheet())
        btn_auto.clicked.connect(self._import_folder_auto)
        import_row.addWidget(btn_lights)
        import_row.addWidget(btn_cals)
        import_row.addWidget(btn_auto)
        header_layout.addLayout(import_row)

        outer.addWidget(header)
        outer.addWidget(self._separator())

        # ── Filter tabs ────────────────────────────────────────────────────────
        filter_bar = QWidget()
        filter_bar.setFixedHeight(30)
        filter_bar.setStyleSheet(f"QWidget {{ background-color: {BG_SECONDARY}; }}")
        filter_layout = QHBoxLayout(filter_bar)
        filter_layout.setContentsMargins(8, 0, 8, 0)
        filter_layout.setSpacing(0)
        self._filter_btns: dict[str, QPushButton] = {}
        for f in _FILTER_TABS:
            btn = QPushButton(f)
            btn.setFixedHeight(28)
            btn.setStyleSheet(self._filter_btn_style(f == "ALL"))
            btn.clicked.connect(lambda checked, fv=f: self._set_filter(fv))
            self._filter_btns[f] = btn
            filter_layout.addWidget(btn)
        outer.addWidget(filter_bar)
        outer.addWidget(self._separator())

        # ── Search ────────────────────────────────────────────────────────────
        search_wrap = QWidget()
        search_wrap.setStyleSheet(f"QWidget {{ background-color: {BG_SECONDARY}; }}")
        search_layout = QHBoxLayout(search_wrap)
        search_layout.setContentsMargins(8, 6, 8, 6)
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search frames…")
        self._search_edit.setFixedHeight(26)
        self._search_edit.setStyleSheet(
            f"QLineEdit {{ background-color: {BG_TERTIARY}; color: {TEXT_PRIMARY};"
            f"border: 1px solid {BORDER}; border-radius: 5px; padding: 3px 8px;"
            "font-size: 11px; }"
            f"QLineEdit:focus {{ border-color: {ACCENT}; }}"
        )
        self._search_edit.textChanged.connect(self._on_search)
        search_layout.addWidget(self._search_edit)
        outer.addWidget(search_wrap)
        outer.addWidget(self._separator())

        # ── Frame list ────────────────────────────────────────────────────────
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            f"QScrollBar:vertical {{ background: transparent; width: 6px; }}"
            f"QScrollBar::handle:vertical {{ background: {BG_HOVER}; border-radius: 3px; }}"
        )
        self._frame_list_widget = QWidget()
        self._frame_list_widget.setStyleSheet("background: transparent;")
        self._frame_list_layout = QVBoxLayout(self._frame_list_widget)
        self._frame_list_layout.setContentsMargins(4, 4, 4, 4)
        self._frame_list_layout.setSpacing(2)
        self._frame_list_layout.addStretch()
        scroll_area.setWidget(self._frame_list_widget)
        outer.addWidget(scroll_area, 1)

        outer.addWidget(self._separator())

        # ── Plate Solve section ───────────────────────────────────────────────
        plate_section = QWidget()
        plate_section.setStyleSheet(f"QWidget {{ background-color: {BG_SECONDARY}; }}")
        plate_layout = QVBoxLayout(plate_section)
        plate_layout.setContentsMargins(8, 8, 8, 6)
        plate_layout.setSpacing(4)

        plate_hdr = QLabel("PLATE SOLVE")
        plate_hdr.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 10px; font-weight: 600; background-color: transparent;")
        plate_layout.addWidget(plate_hdr)

        plate_btn_row = QHBoxLayout()
        plate_btn_row.setSpacing(4)
        btn_solve = QPushButton("Solve WCS")
        btn_solve.setMinimumHeight(28)
        btn_solve.setStyleSheet(btn_cals.styleSheet().replace("font-size: 10px;", "font-size: 10px;"))
        btn_solve.clicked.connect(self.plate_solve_clicked)
        btn_dso = QPushButton("DSO Overlay")
        btn_dso.setMinimumHeight(28)
        btn_dso.setStyleSheet(btn_cals.styleSheet())
        btn_dso.clicked.connect(self.dso_overlay_clicked)
        plate_btn_row.addWidget(btn_solve)
        plate_btn_row.addWidget(btn_dso)
        plate_layout.addLayout(plate_btn_row)

        self._wcs_info = QLabel("")
        self._wcs_info.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 9px;"
            "font-family: 'JetBrains Mono', 'Cascadia Code', monospace;"
        )
        self._wcs_info.setWordWrap(True)
        plate_layout.addWidget(self._wcs_info)

        outer.addWidget(plate_section)
        outer.addWidget(self._separator())

        # ── Stats section ─────────────────────────────────────────────────────
        stats_section = QWidget()
        stats_section.setStyleSheet(f"QWidget {{ background-color: {BG_SECONDARY}; }}")
        stats_layout = QHBoxLayout(stats_section)
        stats_layout.setContentsMargins(8, 6, 8, 8)
        stats_layout.setSpacing(4)
        btn_stats = QPushButton("Statistics")
        btn_stats.setMinimumHeight(28)
        btn_stats.setStyleSheet(btn_cals.styleSheet())
        btn_stats.clicked.connect(self.show_statistics)
        btn_header = QPushButton("FITS Header")
        btn_header.setMinimumHeight(28)
        btn_header.setStyleSheet(btn_cals.styleSheet())
        btn_header.clicked.connect(self.show_fits_header)
        stats_layout.addWidget(btn_stats)
        stats_layout.addWidget(btn_header)
        outer.addWidget(stats_section)

        outer.addWidget(self._separator())

        # ── History tab (collapsed at bottom) ─────────────────────────────────
        self._history_tree = QTreeWidget()
        self._history_tree.setHeaderLabels(["Step", "Time"])
        self._history_tree.setMaximumHeight(120)
        self._history_tree.setStyleSheet(
            f"QTreeWidget {{ background: {BG_SECONDARY}; border: none; font-size: 10px; }}"
        )
        self._history_tree.setVisible(False)

        hist_hdr_row = QHBoxLayout()
        self._hist_toggle = QPushButton("▶ History")
        self._hist_toggle.setStyleSheet(
            f"QPushButton {{ color: {TEXT_SECONDARY}; background: transparent; border: none;"
            "font-size: 10px; text-align: left; padding: 4px 8px; }"
            f"QPushButton:hover {{ color: {TEXT_PRIMARY}; }}"
        )
        self._hist_toggle.clicked.connect(self._toggle_history)
        hist_hdr_row.addWidget(self._hist_toggle)
        hist_hdr_row.addStretch()

        hist_container = QWidget()
        hist_container.setStyleSheet(f"background: {BG_SECONDARY};")
        hist_vbox = QVBoxLayout(hist_container)
        hist_vbox.setContentsMargins(0, 0, 0, 0)
        hist_vbox.setSpacing(0)
        hist_vbox.addLayout(hist_hdr_row)
        hist_vbox.addWidget(self._history_tree)
        outer.addWidget(hist_container)

        # Keep the History tree current: operations record history from 50+
        # call sites, none of which know about this panel — poll cheaply
        # instead of wiring every one.
        from PyQt6.QtCore import QTimer
        self._history_timer = QTimer(self)
        self._history_timer.setInterval(1500)
        self._history_timer.timeout.connect(self._refresh_history_if_changed)
        self._history_timer.start()

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _separator() -> QFrame:
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {BORDER};")
        return sep

    def _filter_btn_style(self, active: bool) -> str:
        color = ACCENT if active else TEXT_SECONDARY
        border = f"2px solid {ACCENT}" if active else "2px solid transparent"
        return (
            f"QPushButton {{ padding: 4px 0; font-size: 10px; font-weight: 600;"
            f"background: none; border: none; border-bottom: {border}; color: {color};"
            "border-radius: 0; }"
            f"QPushButton:hover {{ color: {TEXT_PRIMARY}; }}"
        )

    def _set_filter(self, f: str):
        self._active_filter = f
        for name, btn in self._filter_btns.items():
            btn.setStyleSheet(self._filter_btn_style(name == f))
        self._rebuild_frame_list()

    def _on_search(self, text: str):
        self._search_text = text
        self._rebuild_frame_list()

    def _toggle_history(self):
        visible = not self._history_tree.isVisible()
        self._history_tree.setVisible(visible)
        self._hist_toggle.setText(("▼" if visible else "▶") + " History")

    # ── Frame list ───────────────────────────────────────────────────────────

    def _rebuild_frame_list(self):
        """Clear and repopulate the frame list according to current filter/search."""
        for row in self._frame_rows:
            row.deleteLater()
        self._frame_rows.clear()

        # Remove all items except the trailing stretch
        while self._frame_list_layout.count() > 1:
            item = self._frame_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if self._project is None:
            self._insert_empty_hint(
                "No project yet.\n\nFile > New Project to stack multiple frames,\n"
                "or just Open Image to process a single stacked file."
            )
            return

        search = self._search_text.lower()
        for entry in self._project.frames:
            type_name = entry.frame_type.name
            if self._active_filter != "ALL" and type_name != self._active_filter:
                continue
            if search and search not in entry.path.name.lower():
                continue
            row = _FrameRow(entry)
            if self._selected_path and str(entry.path) == self._selected_path:
                row.set_selected(True)
            row.clicked.connect(self._on_row_clicked)
            row.context_requested.connect(self._on_context_menu)
            self._frame_rows.append(row)
            self._frame_list_layout.insertWidget(
                self._frame_list_layout.count() - 1, row
            )

        if not self._frame_rows:
            self._insert_empty_hint(
                "No frames here yet.\nClick '+ Lights' below, or drag FITS files in."
            )

    def _insert_empty_hint(self, text: str):
        """Show a centered dim hint in the frame-list area when it's empty."""
        hint = QLabel(text)
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 10px; padding: 20px 8px;"
        )
        self._frame_list_layout.insertWidget(
            self._frame_list_layout.count() - 1, hint
        )

    def _on_row_clicked(self, path: str):
        self._selected_path = path
        for row in self._frame_rows:
            row.set_selected(row._path == path)
        self.frame_selected.emit(path)

    def _on_context_menu(self, path: str):
        menu = QMenu(self)
        open_action = menu.addAction("Open in Canvas")
        remove_action = menu.addAction("Remove from Project")
        action = menu.exec(self.cursor().pos())
        if action == open_action:
            self.frame_selected.emit(path)
        elif action == remove_action and self._project:
            self._project.remove_frame(Path(path))
            self.refresh()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_project(self, project: Project | None):
        self._project = project
        self._selected_path = None
        if project:
            self._proj_name_lbl.setText(project.name)
            lights = [e for e in project.frames if e.frame_type == FrameType.LIGHT]
            good = sum(1 for e in lights if e.path.exists())
            self._proj_stats_lbl.setText(f"{good}/{len(lights)} frames")
        else:
            self._proj_name_lbl.setText("No Project")
            self._proj_stats_lbl.setText("")
        self.refresh()

    def refresh(self):
        self._rebuild_frame_list()
        self.refresh_history()

        # Update stats label
        if self._project:
            lights = [e for e in self._project.frames if e.frame_type == FrameType.LIGHT]
            good = sum(1 for e in lights if e.path.exists())
            self._proj_stats_lbl.setText(f"{good}/{len(lights)} frames")

    def set_fallback_history_provider(self, provider):
        """Set a zero-arg callable returning [(name, time_str), ...] used for
        the History tree when no project is loaded (e.g. the editing history
        of a directly-opened image)."""
        self._fallback_history_provider = provider

    def _history_entries(self) -> list[tuple[str, str]]:
        if self._project is not None:
            entries = []
            for step in self._project.history:
                ts = (step.timestamp.split("T")[-1][:8]
                      if "T" in step.timestamp else step.timestamp)
                entries.append((step.name, ts))
            return entries
        if self._fallback_history_provider is not None:
            try:
                return list(self._fallback_history_provider())
            except Exception:
                log.debug("Fallback history provider failed", exc_info=True)
        return []

    def refresh_history(self):
        """Rebuild the History tree from the current source."""
        entries = self._history_entries()
        self._history_signature = (len(entries), entries[-1] if entries else None)
        self._history_tree.clear()
        for name, ts in entries:
            QTreeWidgetItem(self._history_tree, [name, ts])
        # Newest step visible without scrolling
        if entries:
            self._history_tree.scrollToBottom()

    def _refresh_history_if_changed(self):
        entries = self._history_entries()
        sig = (len(entries), entries[-1] if entries else None)
        if sig != self._history_signature:
            self.refresh_history()

    def set_wcs_info(self, ra: str, dec: str, scale: str, pa: str):
        """Update plate solve result display."""
        self._wcs_ra = ra
        self._wcs_dec = dec
        self._wcs_scale = scale
        self._wcs_pa = pa
        if ra and dec:
            self._wcs_info.setText(
                f"RA {ra} · Dec {dec}\nScale {scale}″/px · PA {pa}°"
            )
        else:
            self._wcs_info.setText("")

    # ── Import helpers ────────────────────────────────────────────────────────

    def _import_frames(self, frame_type: FrameType):
        paths, _ = QFileDialog.getOpenFileNames(
            self, f"Import {frame_type.name.title()} Frames", "", FILE_FILTERS,
        )
        if paths:
            self.frames_imported.emit([Path(p) for p in paths], frame_type)

    def _import_cals(self):
        """Pop a small menu to pick calibration frame type."""
        menu = QMenu(self)
        for label, ft in [
            ("Darks", FrameType.DARK),
            ("Flats", FrameType.FLAT),
            ("Bias / Offset", FrameType.BIAS),
        ]:
            menu.addAction(label, lambda checked=False, f=ft: self._import_frames(f))
        menu.exec(self.cursor().pos())

    def _import_folder_auto(self):
        from PyQt6.QtWidgets import QMessageBox

        from astraios.core.image_io import _guess_frame_type

        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Auto-Import", "")
        if not folder:
            return

        folder = Path(folder)
        extensions = ("*.fits", "*.fit", "*.fts", "*.xisf",
                      "*.FITS", "*.FIT", "*.FTS", "*.XISF")
        files = []
        for ext in extensions:
            files.extend(folder.glob(ext))
            files.extend(folder.rglob(ext))
        files = sorted(set(files))

        if not files:
            QMessageBox.information(self, "Auto-Import", f"No FITS/XISF files found in:\n{folder}")
            return

        groups: dict[FrameType, list[Path]] = {}
        unknown: list[Path] = []

        for path in files:
            try:
                header = {}
                if path.suffix.lower() in (".fits", ".fit", ".fts"):
                    try:
                        from astropy.io import fits as _fits
                        with _fits.open(str(path), memmap=True) as hdul:
                            for hdu in hdul:
                                if hdu.data is not None:
                                    header = dict(hdu.header)
                                    break
                    except Exception:
                        pass
                ft = _guess_frame_type(header, path)
                if ft == FrameType.UNKNOWN:
                    unknown.append(path)
                else:
                    groups.setdefault(ft, []).append(path)
            except Exception:
                unknown.append(path)

        total = 0
        for ft, paths in groups.items():
            self.frames_imported.emit(paths, ft)
            total += len(paths)

        if unknown:
            reply = QMessageBox.question(
                self, "Auto-Import — Unknown Frames",
                f"{len(unknown)} frame(s) could not be auto-detected.\nImport as Light frames?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.frames_imported.emit(unknown, FrameType.LIGHT)
                total += len(unknown)

        summary = ", ".join(
            f"{len(v)} {FRAME_TYPE_LABELS.get(k, k.name)}"
            for k, v in groups.items()
        )
        QMessageBox.information(
            self, "Auto-Import Complete",
            f"Imported {total} frames:\n{summary or 'none detected'}"
            + (f"\n{len(unknown)} unknown" if unknown else ""),
        )
