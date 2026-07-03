"""Image Statistics Dialog — display comprehensive per-channel statistics."""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from astraios.core.statistics import ImageStatistics


def _snr_db(snr_linear: float) -> str:
    if snr_linear <= 0:
        return "—"
    return f"{20 * math.log10(snr_linear):.1f} dB"


def _noise_quality_color(snr_linear: float) -> str:
    """Return CSS color based on SNR quality."""
    if snr_linear <= 0:
        return "#888888"
    db = 20 * math.log10(snr_linear)
    if db >= 40:
        return "#44cc44"   # excellent
    elif db >= 25:
        return "#cccc44"   # good
    elif db >= 15:
        return "#cc8844"   # marginal
    return "#cc4444"       # poor


class StatisticsDialog(QDialog):
    """Dialog showing per-channel image statistics and noise estimation."""

    def __init__(self, stats: ImageStatistics, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Statistics")
        self.setMinimumSize(680, 520)

        layout = QVBoxLayout(self)

        # ── Summary row ──────────────────────────────────────────────────────
        linear_str = "Linear (unstretched)" if stats.is_linear else "Non-linear (stretched)"
        summary = QLabel(
            f"{stats.width} × {stats.height} px  |  {stats.n_channels} channel(s)  |  "
            f"{stats.total_pixels:,} pixels  |  {linear_str}"
        )
        summary.setStyleSheet("font-size: 12px; color: #80c0ff; margin-bottom: 8px;")
        layout.addWidget(summary)

        # ── Noise summary cards ───────────────────────────────────────────────
        noise_group = QGroupBox("Noise Estimation")
        noise_grid = QGridLayout(noise_group)
        noise_grid.setSpacing(12)

        headers = ["Channel", "Background", "Noise σ", "SNR (linear)", "SNR (dB)", "Quality"]
        for col, h in enumerate(headers):
            lbl = QLabel(f"<b>{h}</b>")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            noise_grid.addWidget(lbl, 0, col)

        for row, ch in enumerate(stats.channels):
            # Noise σ ≈ MAD × 1.4826  (robust Gaussian estimator)
            noise_sigma = ch.mad * 1.4826
            snr_lin = ch.snr_estimate
            color = _noise_quality_color(snr_lin)

            if snr_lin >= 10 ** (40 / 20):  # > 40 dB
                quality_text = "Excellent"
            elif snr_lin >= 10 ** (25 / 20):  # > 25 dB
                quality_text = "Good"
            elif snr_lin >= 10 ** (15 / 20):  # > 15 dB
                quality_text = "Marginal"
            else:
                quality_text = "Poor"

            cells = [
                ch.name,
                f"{ch.median:.5f}",
                f"{noise_sigma:.5f}",
                f"{snr_lin:.1f}",
                _snr_db(snr_lin),
                quality_text,
            ]
            for col, val in enumerate(cells):
                lbl = QLabel(val)
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == 5:  # quality column
                    lbl.setStyleSheet(f"color: {color}; font-weight: bold;")
                elif col == 4:  # dB column
                    lbl.setStyleSheet(f"color: {color};")
                noise_grid.addWidget(lbl, row + 1, col)

        layout.addWidget(noise_group)

        # ── Full stats table ──────────────────────────────────────────────────
        columns = [
            "Channel", "Mean", "Median", "Std Dev", "MAD", "SNR est.", "Min", "Max",
            "P01", "P99", "Clip Lo%", "Clip Hi%",
        ]
        n_rows = len(stats.channels)
        table = QTableWidget(n_rows, len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        for row, ch in enumerate(stats.channels):
            values = [
                ch.name,
                f"{ch.mean:.5f}",
                f"{ch.median:.5f}",
                f"{ch.std:.5f}",
                f"{ch.mad:.5f}",
                f"{ch.snr_estimate:.1f}",
                f"{ch.min_val:.5f}",
                f"{ch.max_val:.5f}",
                f"{ch.percentile_01:.5f}",
                f"{ch.percentile_99:.5f}",
                f"{ch.clipped_low_pct:.2f}%",
                f"{ch.clipped_high_pct:.2f}%",
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignLeft if col == 0
                    else Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                table.setItem(row, col, item)

        layout.addWidget(table)
