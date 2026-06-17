"""Curves Widget — interactive curve editor with histogram overlay.

Provides a clickable/draggable bezier curve editor for tonal adjustments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from astraios.core.curves import CurvePoints, CurvesParams

if TYPE_CHECKING:
    pass


class CurveEditor(QWidget):
    """Interactive curve editor widget.

    Click to add points, drag to move, right-click to delete.
    Shows a smooth curve through control points with optional histogram overlay.
    """

    curve_changed = pyqtSignal()

    MARGIN = 8
    POINT_RADIUS = 5
    GRID_COLOR = QColor(60, 60, 60)
    CURVE_COLOR = QColor(220, 220, 220)
    POINT_COLOR = QColor(255, 140, 0)
    POINT_HOVER_COLOR = QColor(255, 200, 100)
    HIST_COLOR = QColor(80, 80, 80, 120)

    CHANNEL_NAMES = ["master", "red", "green", "blue"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)

        self._curve = CurvePoints()
        self._histogram: np.ndarray | None = None
        self._dragging_idx: int = -1
        self._hover_idx: int = -1
        # Per-channel points cache: {channel_name: list[tuple[float, float]]}
        self._channel_cache: dict[str, list[tuple[float, float]]] = {}
        self._current_channel: str = "master"

    @property
    def curve(self) -> CurvePoints:
        return self._curve

    @curve.setter
    def curve(self, value: CurvePoints):
        self._curve = value
        self.update()

    def set_histogram(self, counts: np.ndarray | None):
        """Set histogram data to display behind the curve."""
        self._histogram = counts
        self.update()

    def _graph_rect(self) -> QRectF:
        m = self.MARGIN
        return QRectF(m, m, self.width() - 2 * m, self.height() - 2 * m)

    def _value_to_widget(self, vx: float, vy: float) -> QPointF:
        r = self._graph_rect()
        wx = r.left() + vx * r.width()
        wy = r.bottom() - vy * r.height()
        return QPointF(wx, wy)

    def _widget_to_value(self, pos: QPointF) -> tuple[float, float]:
        r = self._graph_rect()
        vx = (pos.x() - r.left()) / r.width()
        vy = (r.bottom() - pos.y()) / r.height()
        return (max(0, min(1, vx)), max(0, min(1, vy)))

    def _point_at(self, pos: QPointF, radius: float = 10) -> int:
        """Find the control point index near the given widget position."""
        for i, (px, py) in enumerate(self._curve.points):
            wp = self._value_to_widget(px, py)
            dx = pos.x() - wp.x()
            dy = pos.y() - wp.y()
            if dx * dx + dy * dy < radius * radius:
                return i
        return -1

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self._graph_rect()

        # Background
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        # Histogram overlay
        if self._histogram is not None and len(self._histogram) > 0:
            hist = self._histogram.astype(np.float64)
            if hist.max() > 0:
                hist = hist / hist.max()
            n_bins = len(hist)
            bin_width = r.width() / n_bins
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.HIST_COLOR)
            for i, h in enumerate(hist):
                bx = r.left() + i * bin_width
                bh = h * r.height()
                painter.drawRect(QRectF(bx, r.bottom() - bh, bin_width + 0.5, bh))

        # Grid lines
        pen = QPen(self.GRID_COLOR, 1)
        painter.setPen(pen)
        for frac in (0.25, 0.5, 0.75):
            x = r.left() + frac * r.width()
            y = r.bottom() - frac * r.height()
            painter.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            painter.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))

        # Diagonal (identity line)
        pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(pen)
        painter.drawLine(self._value_to_widget(0, 0), self._value_to_widget(1, 1))

        # Draw the curve
        lut = self._curve.build_lut()
        # Downsample LUT for drawing
        n_draw = 200
        path = QPainterPath()
        for i in range(n_draw + 1):
            t = i / n_draw
            lut_idx = int(t * (len(lut) - 1))
            val = float(lut[lut_idx])
            pt = self._value_to_widget(t, val)
            if i == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)

        pen = QPen(self.CURVE_COLOR, 2)
        pen.setStyle(Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.drawPath(path)

        # Draw control points
        for i, (px, py) in enumerate(self._curve.points):
            wp = self._value_to_widget(px, py)
            color = self.POINT_HOVER_COLOR if i == self._hover_idx else self.POINT_COLOR
            painter.setPen(QPen(color, 2))
            painter.setBrush(color)
            painter.drawEllipse(wp, self.POINT_RADIUS, self.POINT_RADIUS)

        # Border
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(r)

        painter.end()

    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position()
        idx = self._point_at(pos)

        if event.button() == Qt.MouseButton.LeftButton:
            if idx >= 0:
                self._dragging_idx = idx
            else:
                # Add new point
                vx, vy = self._widget_to_value(pos)
                self._curve.add_point(vx, vy)
                self._dragging_idx = self._point_at(pos)
                self.curve_changed.emit()
                self.update()

        elif event.button() == Qt.MouseButton.RightButton:
            if idx > 0 and idx < len(self._curve.points) - 1:
                self._curve.remove_point(idx)
                self.curve_changed.emit()
                self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()

        if self._dragging_idx >= 0:
            vx, vy = self._widget_to_value(pos)
            self._curve.move_point(self._dragging_idx, vx, vy)
            self.curve_changed.emit()
            self.update()
        else:
            old = self._hover_idx
            self._hover_idx = self._point_at(pos)
            if old != self._hover_idx:
                self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._dragging_idx = -1

    def set_channel(self, channel: str) -> None:
        """Switch to a different channel, caching the current curve points."""
        self._channel_cache[self._current_channel] = list(self._curve.points)
        self._current_channel = channel
        cached = self._channel_cache.get(channel)
        if cached:
            self._curve = CurvePoints()
            self._curve.points = list(cached)
        else:
            self._curve = CurvePoints()
        self.curve_changed.emit()
        self.update()

    def get_points(self) -> list[tuple[float, float]]:
        return list(self._curve.points)

    def set_points(self, points: list[tuple[float, float]]) -> None:
        self._curve = CurvePoints()
        self._curve.points = list(points)
        self.curve_changed.emit()
        self.update()

    def reset(self):
        """Reset the curve to identity."""
        self._curve = CurvePoints()
        self._channel_cache.clear()
        self.curve_changed.emit()
        self.update()

    def get_params(self) -> CurvesParams:
        """Return CurvesParams for the current curve state."""
        return CurvesParams(master=self._curve)
