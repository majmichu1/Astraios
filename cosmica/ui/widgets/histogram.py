"""Histogram Widget — interactive histogram display for image channels."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget


class HistogramWidget(QWidget):
    """Renders R/G/B/Luminance histograms with log scale and clip indicators."""

    CHANNEL_COLORS = {
        "red": QColor(220, 50, 50, 180),
        "green": QColor(50, 200, 50, 180),
        "blue": QColor(50, 100, 220, 180),
        "luminance": QColor(200, 200, 200, 120),
        "gray": QColor(200, 200, 200, 180),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 100)
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
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._data is None:
            painter.setPen(QPen(QColor(100, 100, 100)))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No histogram data")
            painter.end()
            return

        margin = 4
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin

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
            fill_color.setAlpha(60)
            painter.fillPath(path, fill_color)

            painter.setPen(QPen(color, 1.0))
            painter.drawPath(path)

        # Draw clip indicator lines
        if self._clip_shadow is not None and 0.0 < self._clip_shadow < 1.0:
            sx = margin + self._clip_shadow * w
            pen = QPen(QColor(80, 160, 255, 220), 1.5)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(int(sx), margin, int(sx), margin + h)
        if self._clip_highlight is not None and 0.0 < self._clip_highlight < 1.0:
            hx = margin + self._clip_highlight * w
            pen = QPen(QColor(255, 200, 80, 220), 1.5)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(int(hx), margin, int(hx), margin + h)

        # Draw border
        painter.setPen(QPen(QColor(60, 60, 60)))
        painter.drawRect(QRectF(margin, margin, w, h))

        painter.end()
