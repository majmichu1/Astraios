"""Preferences dialog — application settings with QSettings persistence."""

from __future__ import annotations

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Default values
from astraios.ui.widgets.ui_kit import form_label, param_help

DEFAULTS = {
    # Processing
    "processing/tiled_processing": True,
    "processing/tile_size": 1024,
    "processing/use_gpu": True,
    "processing/max_threads": 0,  # 0 = let PyTorch use every core
    # Paths
    "paths/default_import_dir": "",
    "paths/default_export_dir": "",
    "paths/model_cache_dir": "",
    # AI models
    "ai/auto_download_models": True,
    # User-provided models / external tools (so they aren't re-downloaded)
    "models/starnet_path": "",
    "models/denoise_model": "",
    "models/cosmic_clarity_dir": "",
    # Appearance
    "appearance/split_preview_max": 1024,
    "appearance/histogram_log_scale": True,
    "appearance/pixel_readout_format": "float",  # float, percent
    # Plate solving
    "platesolver/astap_path": "",
    "platesolver/auto_solve": False,
    "platesolver/astrometry_api_key": "",
    # Auto-update
    "update/check_on_startup": True,
}


def load_prefs() -> dict:
    """The saved preferences in the same shape ``PreferencesDialog.get_prefs``
    returns, so the main window can apply them at startup without opening
    the dialog. Until this existed the saved values were only applied after
    the user pressed OK in the dialog, so a restart silently reverted them."""
    s = QSettings("Astraios", "Astraios")

    def get(key, default):
        val = s.value(key, DEFAULTS.get(key, default))
        if isinstance(default, bool):
            return str(val).lower() in ("true", "1")
        if isinstance(default, int):
            try:
                return int(val)
            except (TypeError, ValueError):
                return default
        return "" if val is None else str(val)

    return {
        "processing": {
            "use_gpu": get("processing/use_gpu", True),
            "tiled_processing": get("processing/tiled_processing", True),
            "tile_size": get("processing/tile_size", 1024),
            "max_threads": get("processing/max_threads", 0),
        },
        "paths": {
            "default_import_dir": get("paths/default_import_dir", ""),
            "default_export_dir": get("paths/default_export_dir", ""),
            "model_cache_dir": get("paths/model_cache_dir", ""),
        },
        "ai": {"auto_download_models": get("ai/auto_download_models", True)},
        "appearance": {
            "split_preview_max": get("appearance/split_preview_max", 1024),
            "histogram_log_scale": get("appearance/histogram_log_scale", True),
            "pixel_readout_format": get("appearance/pixel_readout_format", "float"),
        },
        "platesolver": {
            "auto_solve": get("platesolver/auto_solve", False),
            "astap_path": get("platesolver/astap_path", ""),
            "astrometry_api_key": get("platesolver/astrometry_api_key", ""),
        },
        "update": {"check_on_startup": get("update/check_on_startup", True)},
    }


class PreferencesDialog(QDialog):
    """Application preferences dialog with tabbed settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumSize(640, 500)
        self._settings = QSettings("Astraios", "Astraios")
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        tabs = QTabWidget()

        # --- Processing tab ---
        proc_tab = QWidget()
        proc_layout = QFormLayout(proc_tab)

        self._use_gpu = QCheckBox("Use GPU acceleration (CUDA/MPS)")
        self._use_gpu.setToolTip("<qt>" + param_help(
            "Runs the processing on the graphics card when one is available.",
            how="Most tools are many times faster on the GPU. Turn this off "
                "only to compare results or to work around a driver problem. "
                "Takes effect the next time Astraios starts.",
        ) + "</qt>")
        proc_layout.addRow("", self._use_gpu)

        self._tiled = QCheckBox("Tiled processing (for large images)")
        self._tiled.setToolTip("<qt>" + param_help(
            "Processes very large frames (over 24 megapixels) piece by piece "
            "to keep memory use bounded.",
            how="Only pixel-by-pixel stages are tiled, so the result is "
                "identical to processing the whole frame at once. Turn it off "
                "on a machine with plenty of RAM to skip the small overhead.",
        ) + "</qt>")
        proc_layout.addRow("", self._tiled)

        self._tile_size = QSpinBox()
        self._tile_size.setRange(256, 4096)
        self._tile_size.setSingleStep(256)
        self._tile_size.setSuffix(" px")
        proc_layout.addRow(form_label("Tile size:", param_help(
            "Edge length of one tile when tiled processing is active.",
            how="A 2048 px colour tile needs about 50 MB of working memory.",
            higher="Fewer, larger tiles; slightly faster, more memory.",
            lower="More, smaller tiles; less memory per step.",
        )), self._tile_size)

        self._max_threads = QSpinBox()
        self._max_threads.setRange(0, 64)
        self._max_threads.setSpecialValueText("Auto (all cores)")
        self._max_threads.setSuffix(" threads")
        proc_layout.addRow(form_label("CPU threads:", param_help(
            "How many processor cores the CPU stages may use.",
            how="Auto lets PyTorch use every core, which is fastest. Set a "
                "number to keep the machine responsive for other work while "
                "a long stack runs.",
        )), self._max_threads)

        proc_layout.addRow(None, QLabel())  # spacer

        gpu_info = QLabel(
            "<span style='color: #8b949e;'>GPU acceleration requires PyTorch with "
            "CUDA (NVIDIA) or MPS (Apple Silicon). Falls back to CPU automatically.</span>"
        )
        gpu_info.setWordWrap(True)
        proc_layout.addRow("", gpu_info)

        tabs.addTab(proc_tab, "⚙ Processing")

        # --- Paths tab ---
        paths_tab = QWidget()
        paths_layout = QFormLayout(paths_tab)

        self._import_dir = QLineEdit()
        self._import_dir.setPlaceholderText("System default")
        import_browse = QPushButton("Browse...")
        import_browse.clicked.connect(lambda: self._browse_dir(self._import_dir))
        import_layout = QHBoxLayout()
        import_layout.addWidget(self._import_dir)
        import_layout.addWidget(import_browse)
        paths_layout.addRow(form_label("Default import folder:", param_help(
            "Where the Open and Import dialogs start.",
            how="Leave blank to start in the system default folder.",
        )), import_layout)

        self._export_dir = QLineEdit()
        self._export_dir.setPlaceholderText("Same as source image")
        export_browse = QPushButton("Browse...")
        export_browse.clicked.connect(lambda: self._browse_dir(self._export_dir))
        export_layout = QHBoxLayout()
        export_layout.addWidget(self._export_dir)
        export_layout.addWidget(export_browse)
        paths_layout.addRow(form_label("Default export folder:", param_help(
            "Where the Export dialog starts when nothing has been exported yet.",
            how="After the first export, the dialog remembers the last folder "
                "you used instead.",
        )), export_layout)

        self._model_cache = QLineEdit()
        self._model_cache.setPlaceholderText("~/.local/share/Astraios/models")
        model_browse = QPushButton("Browse...")
        model_browse.clicked.connect(lambda: self._browse_dir(self._model_cache))
        model_cache_layout = QHBoxLayout()
        model_cache_layout.addWidget(self._model_cache)
        model_cache_layout.addWidget(model_browse)
        paths_layout.addRow(form_label("AI model cache:", param_help(
            "Folder where downloaded AI models are kept.",
            how="Leave blank for the default location. Applies to models "
                "downloaded after the change.",
        )), model_cache_layout)

        tabs.addTab(paths_tab, "📁 Paths")

        # --- AI Models tab ---
        ai_tab = QWidget()
        ai_layout = QFormLayout(ai_tab)

        self._auto_download = QCheckBox("Download models when needed")
        self._auto_download.setToolTip("<qt>" + param_help(
            "Fetches an AI model the first time a tool needs it.",
            how="With this off, a tool whose model is missing stops with a "
                "message instead of downloading. Useful on a metered "
                "connection.",
        ) + "</qt>")
        ai_layout.addRow("", self._auto_download)

        ai_layout.addRow(None, QLabel())

        ai_info = QLabel(
            "<span style='color: #8b949e;'>AI models are downloaded on first use. "
            "Typical size: 50–200 MB per model. Requires internet connection.</span>"
        )
        ai_info.setWordWrap(True)
        ai_layout.addRow("", ai_info)

        ai_layout.addRow(None, QLabel())
        own_models = QLabel(
            "<b>Use models you already have</b><br>"
            "<span style='color: #8b949e;'>Point Astraios at a StarNet binary, a denoise "
            "model, or a Cosmic Clarity model folder you've installed, so they aren't "
            "downloaded again. Leave blank to use the built-in defaults.</span>"
        )
        own_models.setWordWrap(True)
        ai_layout.addRow("", own_models)

        self._starnet_path = QLineEdit()
        self._starnet_path.setPlaceholderText("e.g. ~/StarNet/StarNetv2CLI")
        sn_browse = QPushButton("Browse...")
        sn_browse.clicked.connect(lambda: self._browse_file(self._starnet_path))
        sn_layout = QHBoxLayout()
        sn_layout.addWidget(self._starnet_path)
        sn_layout.addWidget(sn_browse)
        ai_layout.addRow(form_label("StarNet binary:", param_help(
            "Your own StarNet++ command-line executable.",
            how="Star removal runs it as a separate program. Point this at "
                "StarNetv2CLI (or starnet++) wherever you installed it.",
        )), sn_layout)

        self._denoise_model = QLineEdit()
        self._denoise_model.setPlaceholderText("a .pt denoise model (optional)")
        dn_browse = QPushButton("Browse...")
        dn_browse.clicked.connect(lambda: self._browse_file(self._denoise_model))
        dn_layout = QHBoxLayout()
        dn_layout.addWidget(self._denoise_model)
        dn_layout.addWidget(dn_browse)
        ai_layout.addRow(form_label("AI denoise model:", param_help(
            "A .pt weights file to use instead of the built-in denoiser.",
            how="For a model you trained yourself with the scripts in the "
                "repository. Leave blank to use the bundled one.",
        )), dn_layout)

        self._cosmic_clarity_dir = QLineEdit()
        self._cosmic_clarity_dir.setPlaceholderText("your Cosmic Clarity model folder")
        cc_browse = QPushButton("Browse...")
        cc_browse.clicked.connect(lambda: self._browse_dir(self._cosmic_clarity_dir))
        cc_layout = QHBoxLayout()
        cc_layout.addWidget(self._cosmic_clarity_dir)
        cc_layout.addWidget(cc_browse)
        ai_layout.addRow(form_label("Cosmic Clarity folder:", param_help(
            "The folder holding your Cosmic Clarity model files.",
            how="Enables the Cosmic Clarity backend in AI Denoise and AI "
                "Sharpen. The models are not bundled; download them from "
                "the Seti Astro site.",
        )), cc_layout)

        tabs.addTab(ai_tab, "🤖 AI Models")

        # --- Appearance tab ---
        app_tab = QWidget()
        app_layout = QFormLayout(app_tab)

        self._preview_max = QSpinBox()
        self._preview_max.setRange(512, 2048)
        self._preview_max.setSingleStep(256)
        self._preview_max.setSuffix(" px")
        app_layout.addRow(form_label("Live preview size:", param_help(
            "Longest side of the reduced copy used for live before/after "
            "previews, in pixels.",
            higher="Sharper preview, slower slider response.",
            lower="Snappier sliders; fine detail is judged on Apply.",
            default="1024 px keeps most tools interactive.",
        )), self._preview_max)

        self._hist_log = QCheckBox("Use log scale for histogram")
        self._hist_log.setToolTip("<qt>" + param_help(
            "Draws the histogram with a logarithmic height.",
            how="Astro images have almost every pixel near black; a linear "
                "histogram is one spike. Log scale shows the faint tail and "
                "the star highlights as well.",
        ) + "</qt>")
        app_layout.addRow("", self._hist_log)

        self._pixel_format = QComboBox()
        self._pixel_format.addItems(["Float (0.0–1.0)", "Percent (0–100%)", "16-bit (0–65535)"])
        app_layout.addRow(form_label("Pixel readout format:", param_help(
            "How the value under the cursor is shown in the canvas toolbar.",
            how="Float is the 0 to 1 scale the tools use; percent and 16-bit "
                "match what other programs display.",
        )), self._pixel_format)

        tabs.addTab(app_tab, "🎨 Appearance")

        # --- Plate Solver tab ---
        ps_tab = QWidget()
        ps_layout = QFormLayout(ps_tab)

        self._auto_solve = QCheckBox("Plate solve automatically when an image opens")
        self._auto_solve.setToolTip("<qt>" + param_help(
            "Runs the plate solver on every image that opens without sky "
            "coordinates in its header.",
            how="Gives you the WCS overlay, object labels and colour "
                "calibration by catalog without a click. Costs a few seconds "
                "per image, and needs ASTAP or an internet connection.",
        ) + "</qt>")
        ps_layout.addRow("", self._auto_solve)

        self._astap_path = QLineEdit()
        self._astap_path.setPlaceholderText("astap_cli, if it is not on your PATH")
        as_browse = QPushButton("Browse...")
        as_browse.clicked.connect(lambda: self._browse_file(self._astap_path))
        as_layout = QHBoxLayout()
        as_layout.addWidget(self._astap_path)
        as_layout.addWidget(as_browse)
        ps_layout.addRow(form_label("ASTAP executable:", param_help(
            "The ASTAP command-line solver, for offline plate solving.",
            how="Astraios looks for astap_cli on the PATH first. On Windows "
                "and macOS it usually installs elsewhere, so point this at it "
                "(astap_cli.exe inside the ASTAP folder).",
        )), as_layout)

        self._astrometry_api_key = QLineEdit()
        self._astrometry_api_key.setPlaceholderText("Get free key at nova.astrometry.net")
        self._astrometry_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        ps_layout.addRow(form_label("Astrometry.net API key:", param_help(
            "Key for the online solver at nova.astrometry.net, used when "
            "no local solver succeeds.",
            how="Free; sign in on the site and copy the key from your "
                "profile. Solving online takes 30 seconds to a few minutes.",
        )), self._astrometry_api_key)

        ps_layout.addRow(None, QLabel())

        ps_info = QLabel(
            "<span style='color: #8b949e;'>Solving tries the offline Gaia catalog, "
            "then ASTAP, then nova.astrometry.net. Anything that works is enough; "
            "a free key from "
            "<a href='https://nova.astrometry.net' style='color: #58a6ff;'>nova.astrometry.net</a>"
            " is the easiest start.</span>"
        )
        ps_info.setWordWrap(True)
        ps_info.setOpenExternalLinks(True)
        ps_layout.addRow("", ps_info)

        tabs.addTab(ps_tab, "🔭 Plate Solver")

        # --- Update tab ---
        upd_tab = QWidget()
        upd_layout = QFormLayout(upd_tab)

        self._check_update = QCheckBox("Check for updates on startup")
        self._check_update.setToolTip("<qt>" + param_help(
            "Looks up the latest release on GitHub a few seconds after "
            "startup and tells you if it is newer.",
            how="One small web request; nothing is downloaded or installed "
                "by itself.",
        ) + "</qt>")
        upd_layout.addRow("", self._check_update)

        upd_layout.addRow(None, QLabel())

        upd_info = QLabel(
            "<span style='color: #8b949e;'>Astraios only tells you about a new "
            "version. Installing it is the same as the first install: run the "
            "installer for your system again.</span>"
        )
        upd_info.setWordWrap(True)
        upd_layout.addRow("", upd_info)

        tabs.addTab(upd_tab, "🔄 Updates")

        layout.addWidget(tabs)

        # Buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        btn_box.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self._restore_defaults
        )
        layout.addWidget(btn_box)

    def _browse_dir(self, line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            line_edit.setText(path)

    def _browse_file(self, line_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if path:
            line_edit.setText(path)

    def _load_settings(self):
        """Load settings from QSettings into widgets."""
        self._use_gpu.setChecked(self._get("processing/use_gpu", True))
        self._tiled.setChecked(self._get("processing/tiled_processing", True))
        self._tile_size.setValue(self._get("processing/tile_size", 1024))
        self._max_threads.setValue(self._get("processing/max_threads", 0))

        self._import_dir.setText(self._get("paths/default_import_dir", ""))
        self._export_dir.setText(self._get("paths/default_export_dir", ""))
        self._model_cache.setText(self._get("paths/model_cache_dir", ""))

        self._auto_download.setChecked(self._get("ai/auto_download_models", True))

        self._starnet_path.setText(self._get("models/starnet_path", ""))
        self._denoise_model.setText(self._get("models/denoise_model", ""))
        self._cosmic_clarity_dir.setText(self._get("models/cosmic_clarity_dir", ""))

        self._preview_max.setValue(self._get("appearance/split_preview_max", 1024))
        self._hist_log.setChecked(self._get("appearance/histogram_log_scale", True))
        pixel_fmt = self._get("appearance/pixel_readout_format", "float")
        pixel_idx = {"float": 0, "percent": 1, "16bit": 2}.get(pixel_fmt, 0)
        self._pixel_format.setCurrentIndex(pixel_idx)

        self._auto_solve.setChecked(self._get("platesolver/auto_solve", False))
        self._astap_path.setText(self._get("platesolver/astap_path", ""))
        self._astrometry_api_key.setText(self._get("platesolver/astrometry_api_key", ""))

        self._check_update.setChecked(self._get("update/check_on_startup", True))

    def _get(self, key: str, default):
        """Get a setting value from QSettings."""
        val = self._settings.value(key)
        if val is None:
            return default
        if isinstance(default, bool):
            return val in (True, "true", "1", "True")
        if isinstance(default, int):
            try:
                return int(val)
            except (ValueError, TypeError):
                return default
        return val

    def _restore_defaults(self):
        """Reset all settings to defaults."""
        keys_to_remove = [
            "processing/use_gpu", "processing/tiled_processing",
            "processing/tile_size", "processing/tile_overlap", "processing/max_threads",
            "paths/default_import_dir", "paths/default_export_dir", "paths/model_cache_dir",
            "ai/auto_download_models", "ai/model_quality",
            "models/starnet_path", "models/denoise_model", "models/cosmic_clarity_dir",
            "appearance/split_preview_max", "appearance/histogram_log_scale",
            "appearance/pixel_readout_format",
            "platesolver/auto_solve", "platesolver/astrometry_net_path",
            "platesolver/astrometry_api_key",
            "update/check_on_startup",
            "update/auto_download",
        ]
        for key in keys_to_remove:
            self._settings.remove(key)
        self._settings.sync()
        self._load_settings()

    def save(self):
        """Save current widget values to QSettings."""
        self._settings.setValue("processing/use_gpu", self._use_gpu.isChecked())
        self._settings.setValue("processing/tiled_processing", self._tiled.isChecked())
        self._settings.setValue("processing/tile_size", self._tile_size.value())
        self._settings.setValue("processing/max_threads", self._max_threads.value())

        self._settings.setValue("paths/default_import_dir", self._import_dir.text())
        self._settings.setValue("paths/default_export_dir", self._export_dir.text())
        self._settings.setValue("paths/model_cache_dir", self._model_cache.text())

        self._settings.setValue("ai/auto_download_models", self._auto_download.isChecked())

        self._settings.setValue("models/starnet_path", self._starnet_path.text().strip())
        self._settings.setValue("models/denoise_model", self._denoise_model.text().strip())
        self._settings.setValue(
            "models/cosmic_clarity_dir", self._cosmic_clarity_dir.text().strip()
        )

        self._settings.setValue("appearance/split_preview_max", self._preview_max.value())
        self._settings.setValue("appearance/histogram_log_scale", self._hist_log.isChecked())
        pixel_map = {0: "float", 1: "percent", 2: "16bit"}
        self._settings.setValue(
            "appearance/pixel_readout_format",
            pixel_map.get(self._pixel_format.currentIndex(), "float"),
        )

        self._settings.setValue("platesolver/auto_solve", self._auto_solve.isChecked())
        self._settings.setValue("platesolver/astap_path", self._astap_path.text().strip())
        self._settings.setValue("platesolver/astrometry_api_key", self._astrometry_api_key.text().strip())

        self._settings.setValue("update/check_on_startup", self._check_update.isChecked())

        self._settings.sync()

    def get_prefs(self) -> dict:
        """Return all preferences as a nested dict."""
        return {
            "processing": {
                "use_gpu": self._use_gpu.isChecked(),
                "tiled_processing": self._tiled.isChecked(),
                "tile_size": self._tile_size.value(),
                "max_threads": self._max_threads.value(),
            },
            "paths": {
                "default_import_dir": self._import_dir.text(),
                "default_export_dir": self._export_dir.text(),
                "model_cache_dir": self._model_cache.text(),
            },
            "ai": {
                "auto_download_models": self._auto_download.isChecked(),
            },
            "appearance": {
                "split_preview_max": self._preview_max.value(),
                "histogram_log_scale": self._hist_log.isChecked(),
                "pixel_readout_format": {0: "float", 1: "percent", 2: "16bit"}.get(
                    self._pixel_format.currentIndex(), "float"
                ),
            },
            "platesolver": {
                "auto_solve": self._auto_solve.isChecked(),
                "astap_path": self._astap_path.text().strip(),
                "astrometry_api_key": self._astrometry_api_key.text(),
            },
            "update": {
                "check_on_startup": self._check_update.isChecked(),
            },
        }
