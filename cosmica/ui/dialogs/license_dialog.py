"""License activation dialog."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from cosmica.licensing.license_manager import LicenseManager


class LicenseDialog(QDialog):
    """Dialog for entering and activating a Cosmica Pro license key."""

    def __init__(self, license_manager: LicenseManager, parent=None):
        super().__init__(parent)
        self._lm = license_manager
        self.setWindowTitle("Cosmica Pro License")
        self.setMinimumWidth(450)
        self.setModal(True)

        layout = QVBoxLayout(self)

        # Status
        self._status_label = QLabel()
        self._update_status()
        layout.addWidget(self._status_label)

        # Key input
        layout.addWidget(QLabel("Enter your license key:"))
        key_row = QHBoxLayout()
        self._key_input = QLineEdit()
        self._key_input.setPlaceholderText("COSMICA-XXXX-XXXX-XXXX-XXXX")
        key_row.addWidget(self._key_input)

        self._activate_btn = QPushButton("Activate")
        self._activate_btn.clicked.connect(self._activate)
        key_row.addWidget(self._activate_btn)
        layout.addLayout(key_row)

        # Result
        self._result_label = QLabel()
        self._result_label.setWordWrap(True)
        layout.addWidget(self._result_label)

        # Buy button
        layout.addSpacing(12)
        info = QLabel(
            "Don't have a key? Get Cosmica Pro for €59 (lifetime) or €7.99/month."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #969696;")
        layout.addWidget(info)

        btn_row = QHBoxLayout()
        buy_btn = QPushButton("Buy Cosmica Pro")
        buy_btn.setObjectName("proButton")
        buy_btn.clicked.connect(self._open_store)
        btn_row.addWidget(buy_btn)

        if self._lm.is_pro:
            deactivate_btn = QPushButton("Deactivate")
            deactivate_btn.setProperty("flat", True)
            deactivate_btn.setObjectName("flatButton")
            deactivate_btn.clicked.connect(self._deactivate)
            btn_row.addWidget(deactivate_btn)

        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setProperty("flat", True)
        close_btn.setObjectName("flatButton")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _update_status(self):
        if self._lm.is_pro:
            self._status_label.setText(
                f"<b style='color:#98c379'>Cosmica Pro — Active</b><br>"
                f"Licensed to: {self._lm.status.email or 'N/A'}"
            )
        else:
            self._status_label.setText(
                "<b style='color:#969696'>Cosmica Free</b><br>"
                "Upgrade to Pro for AI features, batch processing, and more."
            )

    def _activate(self):
        key = self._key_input.text().strip()
        if not key:
            self._result_label.setText("<span style='color:#e5c07b'>Please enter a license key.</span>")
            return

        self._result_label.setText("Validating...")
        self._activate_btn.setEnabled(False)

        # This should ideally be async, but for simplicity:
        status = self._lm.activate(key)

        if status.valid:
            self._result_label.setText(
                "<span style='color:#98c379'>License activated! Restart to enable all Pro features.</span>"
            )
        else:
            self._result_label.setText(
                f"<span style='color:#e06c75'>Activation failed: {status.error}</span>"
            )

        self._activate_btn.setEnabled(True)
        self._update_status()

    def _deactivate(self):
        self._lm.deactivate()
        self._update_status()
        self._result_label.setText("License deactivated.")

    def _open_store(self):
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl("https://cosmica.lemonsqueezy.com/buy"))
