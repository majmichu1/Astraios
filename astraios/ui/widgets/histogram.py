"""Histogram Widget — interactive histogram display for image channels."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget


class HistogramWidget(QWidget):
    """Renders R/G/B/Luminance histograms with log scale and clip indicators."""

    # From the design prototype's HistogramDisplay.
    CHANNEL_COLORS = {
        "red": QColor(255, 100, 100, 180),
        "green": QColor(100, 200, 100, 180),
        "blue": QColor(100, 150, 255, 180),
        "luminance": QColor(230, 237, 243, 110),
        "gray": QColor(230, 237, 243, 180),
    }
    BG = QColor("#0d1117")
    GRID = QColor("#1e2530")
    MARKER = QColor(255, 255, 255, 100)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 40)
        self.setMaximumHeight(160)
        self._data: dict | None = None
        self._log_scale = True
        self._active_channel: str = "RGB"
        # Clip point markers: {"shadow": 0.0, "highlight": 1.0} — normalized [0,1]
        self._clip_shadow: float | None = None
        self._clip_highlight: float | None = None

    def set_histogram_data(self, data: dict):
        """Set histogram data from core.stretch.compute_histogram()."""
        self._data = data
        self.update()

    def clear(self):
        self._data = None
        self.update()

    def set_log_scale(self, enabled: bool):
        self._log_scale = enabled
        self.update()

    def set_active_channel(self, name: str):
        self._active_channel = name
        self.update()

    def _get_stats(self) -> tuple[float, float, float, float] | None:
        """Return (mean, median, sd, clip_pct) derived from the active channel histogram bins."""
        if self._data is None:
            return None
        channel_map = {"RGB": "luminance", "R": "red", "G": "green", "B": "blue", "L": "luminance"}
        key = channel_map.get(self._active_channel, "luminance")
        counts = self._data.get(key)
        if counts is None:
            counts = self._data.get("gray")
        if counts is None:
            return None
        counts = counts.astype(np.float64)
        total = counts.sum()
        if total == 0:
            return None
        n = len(counts)
        centers = (np.arange(n) + 0.5) / n
        mean = float(np.dot(counts, centers) / total)
        cumsum = np.cumsum(counts)
        median_idx = int(np.searchsorted(cumsum, total * 0.5))
        median = float(centers[min(median_idx, n - 1)])
        variance = float(np.dot(counts, (centers - mean) ** 2) / total)
        sd = float(variance ** 0.5)
        clip_pct = float((counts[0] + counts[-1]) / total * 100)
        return mean, median, sd, clip_pct

    def set_clip_points(self, shadow: float | None, highlight: float | None):
        """Set shadow/highlight clip indicator positions in normalized [0, 1] space."""
        self._clip_shadow = shadow
        self._clip_highlight = highlight
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.BG)

        margin = 0
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin

        # Grid: quarters across, thirds down.
        painter.setPen(QPen(self.GRID, 1.0))
        for i in range(1, 4):
            x = margin + w * i / 4
            painter.drawLine(QPointF(x, margin), QPointF(x, margin + h))
        for i in range(1, 3):
            y = margin + h * i / 3
            painter.drawLine(QPointF(margin, y), QPointF(margin + w, y))

        if self._data is None:
            painter.setPen(QPen(QColor("#8b949e")))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                "No histogram data yet",
            )
            painter.end()
            return

        _channel_map = {
            "RGB": ["luminance", "gray", "red", "green", "blue"],
            "R":   ["red"],
            "G":   ["green"],
            "B":   ["blue"],
            "L":   ["luminance", "gray"],
        }
        draw_order = _channel_map.get(self._active_channel, ["luminance", "gray", "red", "green", "blue"])

        for channel_name in draw_order:
            if channel_name not in self._data:
                continue
            counts = self._data[channel_name].astype(np.float64)
            counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
            if counts.max() == 0:
                continue

            if self._log_scale:
                counts = np.log1p(counts)

            max_val = counts.max()
            if max_val > 0:
                counts = counts / max_val

            color = self.CHANNEL_COLORS.get(channel_name, QColor(200, 200, 200, 180))
            n_bins = len(counts)

            path = QPainterPath()
            path.moveTo(margin, margin + h)

            for i in range(n_bins):
                x = margin + (i / n_bins) * w
                y = margin + h - float(counts[i]) * h
                path.lineTo(x, y)

            path.lineTo(margin + w, margin + h)
            path.closeSubpath()

            fill_color = QColor(color)
            fill_color.setAlpha(38)
            painter.fillPath(path, fill_color)

            painter.setPen(QPen(color, 1.0))
            painter.drawPath(path)

        # Black / white point markers: dashed, like the design.
        pen = QPen(self.MARKER, 1.0)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setDashPattern([3, 3])
        painter.setPen(pen)
        if self._clip_shadow is not None and 0.0 < self._clip_shadow < 1.0:
            sx = margin + self._clip_shadow * w
            painter.drawLine(QPointF(sx, margin), QPointF(sx, margin + h))
        if self._clip_highlight is not None and 0.0 < self._clip_highlight < 1.0:
            hx = margin + self._clip_highlight * w
            painter.drawLine(QPointF(hx, margin), QPointF(hx, margin + h))

        painter.end()
