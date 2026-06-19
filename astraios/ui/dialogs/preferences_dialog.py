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
DEFAULTS = {
    # Processing
    "processing/tiled_processing": True,
    "processing/tile_size": 1024,
    "processing/tile_overlap": 64,
    "processing/use_gpu": True,
    "processing/max_threads": 8,
    # Paths
    "paths/default_import_dir": "",
    "paths/default_export_dir": "",
    "paths/model_cache_dir": "",
    # AI models
    "ai/auto_download_models": True,
    "ai/model_quality": "balanced",  # fast, balanced, quality
    # User-provided models / external tools (so they aren't re-downloaded)
    "models/starnet_path": "",
    "models/denoise_model": "",
    "models/cosmic_clarity_dir": "",
    # Appearance
    "appearance/split_preview_max": 1024,
    "appearance/histogram_log_scale": True,
    "appearance/pixel_readout_format": "float",  # float, percent
    # Plate solving
    "platesolver/astrometry_net_path": "",
    "platesolver/auto_solve": False,
    "platesolver/astrometry_api_key": "",
    # Auto-update
    "update/check_on_startup": True,
    "update/auto_download": False,
}


class PreferencesDialog(QDialog):
    """Application preferences dialog with tabbed settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumSize(520, 480)
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
        proc_layout.addRow("", self._use_gpu)

        self._tiled = QCheckBox("Tiled processing (for large images)")
        proc_layout.addRow("", self._tiled)

        self._tile_size = QSpinBox()
        self._tile_size.setRange(256, 4096)
        self._tile_size.setSingleStep(256)
        self._tile_size.setSuffix(" px")
        proc_layout.addRow("Tile size:", self._tile_size)

        self._tile_overlap = QSpinBox()
        self._tile_overlap.setRange(8, 256)
        self._tile_overlap.setSingleStep(8)
        self._tile_overlap.setSuffix(" px")
        proc_layout.addRow("Tile overlap:", self._tile_overlap)

        self._max_threads = QSpinBox()
        self._max_threads.setRange(1, 64)
        self._max_threads.setSuffix(" threads")
        proc_layout.addRow("Max CPU threads:", self._max_threads)

        proc_layout.addRow(None, QLabel())  # spacer

        gpu_info = QLabel(
            "<span style='color: #888;'>GPU acceleration requires PyTorch with "
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
        paths_layout.addRow("Default import dir:", import_layout)

        self._export_dir = QLineEdit()
        self._export_dir.setPlaceholderText("Same as source image")
        export_browse = QPushButton("Browse...")
        export_browse.clicked.connect(lambda: self._browse_dir(self._export_dir))
        export_layout = QHBoxLayout()
        export_layout.addWidget(self._export_dir)
        export_layout.addWidget(export_browse)
        paths_layout.addRow("Default export dir:", export_layout)

        self._model_cache = QLineEdit()
        self._model_cache.setPlaceholderText("~/.local/share/Astraios/models")
        model_browse = QPushButton("Browse...")
        model_browse.clicked.connect(lambda: self._browse_dir(self._model_cache))
        model_cache_layout = QHBoxLayout()
        model_cache_layout.addWidget(self._model_cache)
        model_cache_layout.addWidget(model_browse)
        paths_layout.addRow("AI model cache:", model_cache_layout)

        tabs.addTab(paths_tab, "📁 Paths")

        # --- AI Models tab ---
        ai_tab = QWidget()
        ai_layout = QFormLayout(ai_tab)

        self._auto_download = QCheckBox("Auto-download models when needed")
        ai_layout.addRow("", self._auto_download)

        self._model_quality = QComboBox()
        self._model_quality.addItems(["Fast (smaller model)", "Balanced", "Quality (larger model)"])
        ai_layout.addRow("Model quality:", self._model_quality)

        ai_layout.addRow(None, QLabel())

        ai_info = QLabel(
            "<span style='color: #888;'>AI models are downloaded on first use. "
            "Typical size: 50–200 MB per model. Requires internet connection.</span>"
        )
        ai_info.setWordWrap(True)
        ai_layout.addRow("", ai_info)

        ai_layout.addRow(None, QLabel())
        own_models = QLabel(
            "<b>Use models you already have</b><br>"
            "<span style='color: #888;'>Point Astraios at a StarNet binary, a denoise "
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
        ai_layout.addRow("StarNet binary:", sn_layout)

        self._denoise_model = QLineEdit()
        self._denoise_model.setPlaceholderText("a .pt denoise model (optional)")
        dn_browse = QPushButton("Browse...")
        dn_browse.clicked.connect(lambda: self._browse_file(self._denoise_model))
        dn_layout = QHBoxLayout()
        dn_layout.addWidget(self._denoise_model)
        dn_layout.addWidget(dn_browse)
        ai_layout.addRow("AI denoise model:", dn_layout)

        self._cosmic_clarity_dir = QLineEdit()
        self._cosmic_clarity_dir.setPlaceholderText("your Cosmic Clarity model folder")
        cc_browse = QPushButton("Browse...")
        cc_browse.clicked.connect(lambda: self._browse_dir(self._cosmic_clarity_dir))
        cc_layout = QHBoxLayout()
        cc_layout.addWidget(self._cosmic_clarity_dir)
        cc_layout.addWidget(cc_browse)
        ai_layout.addRow("Cosmic Clarity folder:", cc_layout)

        tabs.addTab(ai_tab, "🤖 AI Models")

        # --- Appearance tab ---
        app_tab = QWidget()
        app_layout = QFormLayout(app_tab)

        self._preview_max = QSpinBox()
        self._preview_max.setRange(512, 2048)
        self._preview_max.setSingleStep(256)
        self._preview_max.setSuffix(" px")
        app_layout.addRow("Split preview max size:", self._preview_max)

        self._hist_log = QCheckBox("Use log scale for histogram")
        app_layout.addRow("", self._hist_log)

        self._pixel_format = QComboBox()
        self._pixel_format.addItems(["Float (0.0–1.0)", "Percent (0–100%)", "16-bit (0–65535)"])
        app_layout.addRow("Pixel readout format:", self._pixel_format)

        tabs.addTab(app_tab, "🎨 Appearance")

        # --- Plate Solver tab ---
        ps_tab = QWidget()
        ps_layout = QFormLayout(ps_tab)

        self._auto_solve = QCheckBox("Auto plate solve on image load")
        ps_layout.addRow("", self._auto_solve)

        self._astrometry_path = QLineEdit()
        self._astrometry_path.setPlaceholderText("/usr/bin/solve-field")
        as_browse = QPushButton("Browse...")
        as_browse.clicked.connect(lambda: self._browse_file(self._astrometry_path))
        as_layout = QHBoxLayout()
        as_layout.addWidget(self._astrometry_path)
        as_layout.addWidget(as_browse)
        ps_layout.addRow("Astrometry.net binary:", as_layout)

        self._astrometry_api_key = QLineEdit()
        self._astrometry_api_key.setPlaceholderText("Get free key at nova.astrometry.net")
        self._astrometry_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        ps_layout.addRow("Astrometry.net API key:", self._astrometry_api_key)

        ps_layout.addRow(None, QLabel())

        ps_info = QLabel(
            "<span style='color: #888;'>Plate solving requires either:<br>"
            "• Local: <b>astrometry.net</b> installed on your system, or<br>"
            "• Remote: Free API key from "
            "<a href='https://nova.astrometry.net' style='color: #58a6ff;'>nova.astrometry.net</a>"
            "</span>"
        )
        ps_info.setWordWrap(True)
        ps_info.setOpenExternalLinks(True)
        ps_layout.addRow("", ps_info)

        tabs.addTab(ps_tab, "🔭 Plate Solver")

        # --- Update tab ---
        upd_tab = QWidget()
        upd_layout = QFormLayout(upd_tab)

        self._check_update = QCheckBox("Check for updates on startup")
        upd_layout.addRow("", self._check_update)

        self._auto_download_upd = QCheckBox("Auto-download updates")
        upd_layout.addRow("", self._auto_download_upd)

        upd_layout.addRow(None, QLabel())

        upd_info = QLabel(
            "<span style='color: #888;'>Updates are downloaded in the background. "
            "Installation requires restarting the application.</span>"
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
        self._tile_overlap.setValue(self._get("processing/tile_overlap", 64))
        self._max_threads.setValue(self._get("processing/max_threads", 8))

        self._import_dir.setText(self._get("paths/default_import_dir", ""))
        self._export_dir.setText(self._get("paths/default_export_dir", ""))
        self._model_cache.setText(self._get("paths/model_cache_dir", ""))

        self._auto_download.setChecked(self._get("ai/auto_download_models", True))
        quality = self._get("ai/model_quality", "balanced")
        quality_idx = {"fast": 0, "balanced": 1, "quality": 2}.get(quality, 1)
        self._model_quality.setCurrentIndex(quality_idx)

        self._starnet_path.setText(self._get("models/starnet_path", ""))
        self._denoise_model.setText(self._get("models/denoise_model", ""))
        self._cosmic_clarity_dir.setText(self._get("models/cosmic_clarity_dir", ""))

        self._preview_max.setValue(self._get("appearance/split_preview_max", 1024))
        self._hist_log.setChecked(self._get("appearance/histogram_log_scale", True))
        pixel_fmt = self._get("appearance/pixel_readout_format", "float")
        pixel_idx = {"float": 0, "percent": 1, "16bit": 2}.get(pixel_fmt, 0)
        self._pixel_format.setCurrentIndex(pixel_idx)

        self._auto_solve.setChecked(self._get("platesolver/auto_solve", False))
        self._astrometry_path.setText(self._get("platesolver/astrometry_net_path", ""))
        self._astrometry_api_key.setText(self._get("platesolver/astrometry_api_key", ""))

        self._check_update.setChecked(self._get("update/check_on_startup", True))
        self._auto_download_upd.setChecked(self._get("update/auto_download", False))

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
            "updates/check_on_startup",
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
        self._settings.setValue("processing/tile_overlap", self._tile_overlap.value())
        self._settings.setValue("processing/max_threads", self._max_threads.value())

        self._settings.setValue("paths/default_import_dir", self._import_dir.text())
        self._settings.setValue("paths/default_export_dir", self._export_dir.text())
        self._settings.setValue("paths/model_cache_dir", self._model_cache.text())

        self._settings.setValue("ai/auto_download_models", self._auto_download.isChecked())
        quality_map = {0: "fast", 1: "balanced", 2: "quality"}
        self._settings.setValue(
            "ai/model_quality", quality_map.get(self._model_quality.currentIndex(), "balanced")
        )

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
        self._settings.setValue("platesolver/astrometry_net_path", self._astrometry_path.text())
        self._settings.setValue("platesolver/astrometry_api_key", self._astrometry_api_key.text().strip())

        self._settings.setValue("update/check_on_startup", self._check_update.isChecked())
        self._settings.setValue("update/auto_download", self._auto_download_upd.isChecked())

        self._settings.sync()

    def get_prefs(self) -> dict:
        """Return all preferences as a nested dict."""
        return {
            "processing": {
                "use_gpu": self._use_gpu.isChecked(),
                "tiled_processing": self._tiled.isChecked(),
                "tile_size": self._tile_size.value(),
                "tile_overlap": self._tile_overlap.value(),
                "max_threads": self._max_threads.value(),
            },
            "paths": {
                "default_import_dir": self._import_dir.text(),
                "default_export_dir": self._export_dir.text(),
                "model_cache_dir": self._model_cache.text(),
            },
            "ai": {
                "auto_download_models": self._auto_download.isChecked(),
                "model_quality": {0: "fast", 1: "balanced", 2: "quality"}.get(
                    self._model_quality.currentIndex(), "balanced"
                ),
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
                "astrometry_net_path": self._astrometry_path.text(),
                "astrometry_api_key": self._astrometry_api_key.text(),
            },
            "update": {
                "check_on_startup": self._check_update.isChecked(),
                "auto_download": self._auto_download_upd.isChecked(),
            },
        }
