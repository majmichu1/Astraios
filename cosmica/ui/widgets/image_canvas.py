"""Image Canvas Widget — zoomable, pannable image display with before/after split."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QLabel, QPushButton, QRubberBand, QWidget


class ImageCanvas(QWidget):
    """High-performance image display with zoom, pan, split preview, and overlays."""

    zoom_changed = pyqtSignal(float)
    cursor_position = pyqtSignal(int, int, list)  # x, y, pixel values
    sample_placed = pyqtSignal(float, float)       # image-space x, y
    sample_removed = pyqtSignal(float, float)      # image-space x, y (nearest)
    crop_rect_selected = pyqtSignal(int, int, int, int)  # x, y, w, h in image-space
    crop_mode_changed = pyqtSignal(bool)  # emitted when crop mode enters or exits
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    export_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._pixmap: QPixmap | None = None
        self._pixmap_after: QPixmap | None = None
        self._pixmap_before: QPixmap | None = None  # saved for Before button
        self._view_mode: str = "after"  # "after", "before", "split"
        self._image_data: np.ndarray | None = None
        # Scale from displayed pixmap pixels → full-res image pixels.
        # When _display_image passes a downscaled thumbnail, this is <1.0.
        self._display_scale: float = 1.0

        self._zoom = 1.0
        self._pan_offset = QPointF(0, 0)
        self._dragging = False
        self._drag_start = QPoint()

        self._split_mode = False
        self._split_position = 0.5

        self._fit_to_window = True

        # Sample placement mode (for dynamic background extraction)
        self._sample_mode = False
        self._sample_points: list[tuple[float, float]] = []  # image-space coords

        # WCS/catalog star overlay
        self._overlay_stars: list[tuple[float, float, float]] = []  # (x, y, mag) image-space
        self._show_wcs_overlay = False

        # DSO annotation overlay
        self._dso_annotations: list[tuple[float, float, str, str]] = []  # (x, y, name, type)
        self._show_dso_overlay = False

        # Constellation line overlay
        self._constellation_lines: list[list[tuple[float, float]]] = []  # segments in pixel coords
        self._show_constellation_overlay = False

        # Crop selection mode
        self._crop_mode = False
        self._crop_rubber_band: QRubberBand | None = None
        self._crop_origin: QPoint = QPoint()

        # Grid overlay
        self._show_grid = False

        # Canvas overlay widgets (child widgets paint on top automatically)
        self._info_chip = QLabel(self)
        self._info_chip.setStyleSheet(
            "QLabel { background: rgba(13,17,23,180); color: #8b949e;"
            " border: 1px solid #30363d; border-radius: 4px;"
            " padding: 2px 8px; font-size: 11px; }"
        )
        self._info_chip.hide()

        _ovr_style = (
            "QPushButton { background: rgba(22,27,34,200); color: #e6edf3;"
            " border: 1px solid #30363d; border-radius: 4px;"
            " padding: 3px 10px; font-size: 11px; font-weight: 500; }"
            " QPushButton:hover { background: rgba(48,54,61,220); }"
        )
        self._overlay_undo_btn = QPushButton("← Undo", self)
        self._overlay_undo_btn.setStyleSheet(_ovr_style)
        self._overlay_undo_btn.clicked.connect(self.undo_requested)

        self._overlay_redo_btn = QPushButton("↷ Redo", self)
        self._overlay_redo_btn.setStyleSheet(_ovr_style)
        self._overlay_redo_btn.clicked.connect(self.redo_requested)

        self._overlay_export_btn = QPushButton("↑ Export", self)
        self._overlay_export_btn.setStyleSheet(
            "QPushButton { background: rgba(46,160,67,200); color: #ffffff;"
            " border: none; border-radius: 4px;"
            " padding: 3px 10px; font-size: 11px; font-weight: 600; }"
            " QPushButton:hover { background: rgba(63,185,80,220); }"
        )
        self._overlay_export_btn.clicked.connect(self.export_requested)

    # ── Public API ───────────────────────────────────────────────────────────

    def set_image(self, rgb_array: np.ndarray, image_data: np.ndarray | None = None, display_scale: float = 1.0):
        rgb_array = np.ascontiguousarray(rgb_array)
        h, w, ch = rgb_array.shape
        bytes_per_line = w * ch
        qimg = QImage(rgb_array.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._image_data = image_data
        self._display_scale = display_scale  # pixmap-px → full-res-px multiplier
        if self._fit_to_window:
            self._fit_zoom()
        self.update()

    def set_after_image(self, rgb_array: np.ndarray):
        rgb_array = np.ascontiguousarray(rgb_array)
        h, w, ch = rgb_array.shape
        bytes_per_line = w * ch
        qimg = QImage(rgb_array.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._pixmap_after = QPixmap.fromImage(qimg)
        self.update()

    def clear_after_image(self):
        self._pixmap_after = None
        self._split_mode = False
        self.update()

    def capture_before(self):
        """Save the current displayed pixmap as the 'before' state for Before/After compare."""
        if self._pixmap is not None:
            self._pixmap_before = QPixmap(self._pixmap)

    def set_view_mode(self, mode: str):
        """Set display mode: 'after' | 'before' | 'split'."""
        self._view_mode = mode
        self._split_mode = (mode == "split")
        self.update()

    def set_split_mode(self, enabled: bool):
        self._split_mode = enabled
        if enabled:
            self._view_mode = "split"
        self.update()

    def set_split_position(self, pos: float):
        self._split_position = max(0.0, min(1.0, pos))
        self.update()

    def fit_to_window(self):
        self._fit_to_window = True
        self._fit_zoom()
        self.update()

    def zoom_to(self, level: float):
        self._fit_to_window = False
        self._zoom = max(0.01, min(50.0, level))
        self.zoom_changed.emit(self._zoom)
        self.update()

    def zoom_in(self):
        self.zoom_to(self._zoom * 1.25)

    def zoom_out(self):
        self.zoom_to(self._zoom / 1.25)

    def set_grid_visible(self, visible: bool):
        self._show_grid = visible
        self.update()

    def set_image_info(self, filename: str, dims_str: str):
        self._info_chip.setText(f"{filename}  ·  {dims_str}")
        self._info_chip.adjustSize()
        self._info_chip.show()
        self._reposition_overlays()

    def _reposition_overlays(self):
        m = 8
        self._info_chip.move(m, m)
        bh = self._overlay_undo_btn.sizeHint().height()
        bw_u = self._overlay_undo_btn.sizeHint().width()
        bw_r = self._overlay_redo_btn.sizeHint().width()
        y_bottom = self.height() - bh - m
        self._overlay_undo_btn.move(m, y_bottom)
        self._overlay_redo_btn.move(m + bw_u + 4, y_bottom)
        exp_w = self._overlay_export_btn.sizeHint().width()
        exp_h = self._overlay_export_btn.sizeHint().height()
        self._overlay_export_btn.move(self.width() - exp_w - m, self.height() - exp_h - m)

    # ── Sample mode ──────────────────────────────────────────────────────────

    def set_sample_mode(self, enabled: bool):
        self._sample_mode = enabled
        if enabled:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_sample_points(self, points: list[tuple[float, float]]):
        self._sample_points = list(points)
        self.update()

    def clear_sample_points(self):
        self._sample_points = []
        self.update()

    # ── WCS overlay ──────────────────────────────────────────────────────────

    def set_overlay_stars(self, stars: list[tuple[float, float, float]]):
        """Set catalog stars for WCS overlay. Each entry: (x_img, y_img, magnitude)."""
        self._overlay_stars = stars
        self.update()

    def set_wcs_overlay_visible(self, visible: bool):
        self._show_wcs_overlay = visible
        self.update()

    def set_dso_annotations(self, annotations: list[tuple[float, float, str, str]]):
        """Set DSO annotations. Each entry: (x_img, y_img, name, type_code)."""
        self._dso_annotations = annotations
        self.update()

    def set_dso_overlay_visible(self, visible: bool):
        self._show_dso_overlay = visible
        self.update()

    def set_constellation_lines(self, segments: list[list[tuple[float, float]]]):
        """Set constellation line segments in image-space pixel coords."""
        self._constellation_lines = segments
        self.update()

    def set_constellation_overlay_visible(self, visible: bool):
        self._show_constellation_overlay = visible
        self.update()

    # ── Crop selection mode ───────────────────────────────────────────────────

    def set_crop_mode(self, enabled: bool):
        """Enter/exit interactive crop selection mode.

        While active the user draws a rectangle on the canvas; on mouse-release
        :attr:`crop_rect_selected` is emitted with image-space (x, y, w, h).
        """
        self._crop_mode = enabled
        if not enabled:
            if self._crop_rubber_band:
                self._crop_rubber_band.hide()
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.crop_mode_changed.emit(enabled)

    # ── Paint ────────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        if self._pixmap is None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), Qt.GlobalColor.black)
            painter.setPen(QPen(Qt.GlobalColor.gray))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No image loaded\nDrag & drop FITS/XISF files here")
            painter.end()
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        pw, ph = self._pixmap.width(), self._pixmap.height()
        dst = QRectF(
            self._pan_offset.x(), self._pan_offset.y(),
            pw * self._zoom, ph * self._zoom,
        )
        src = QRectF(0, 0, pw, ph)

        if self._pixmap_after is not None:
            # Live tool preview: current image left, preview result right (highest priority)
            split_x = dst.left() + dst.width() * self._split_position
            painter.setClipRect(QRectF(dst.left(), dst.top(), split_x - dst.left(), dst.height()))
            painter.drawPixmap(dst, self._pixmap, src)
            aw, ah = self._pixmap_after.width(), self._pixmap_after.height()
            painter.setClipRect(QRectF(split_x, dst.top(), dst.right() - split_x, dst.height()))
            painter.drawPixmap(dst, self._pixmap_after, QRectF(0, 0, aw, ah))
            painter.setClipping(False)
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawLine(QPointF(split_x, dst.top()), QPointF(split_x, dst.bottom()))
        elif self._view_mode == "before" and self._pixmap_before is not None:
            bw, bh = self._pixmap_before.width(), self._pixmap_before.height()
            painter.drawPixmap(dst, self._pixmap_before, QRectF(0, 0, bw, bh))
        elif self._view_mode == "split" and self._pixmap_before is not None:
            # Before/After toolbar split: before left, current right
            split_x = dst.left() + dst.width() * self._split_position
            painter.setClipRect(QRectF(dst.left(), dst.top(), split_x - dst.left(), dst.height()))
            bw, bh = self._pixmap_before.width(), self._pixmap_before.height()
            painter.drawPixmap(dst, self._pixmap_before, QRectF(0, 0, bw, bh))
            painter.setClipRect(QRectF(split_x, dst.top(), dst.right() - split_x, dst.height()))
            painter.drawPixmap(dst, self._pixmap, src)
            painter.setClipping(False)
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.drawLine(QPointF(split_x, dst.top()), QPointF(split_x, dst.bottom()))
        else:
            painter.drawPixmap(dst, self._pixmap, src)

        # Overlay positions are stored in full-res coordinates.
        # Divide by full-res dims (not pixmap/thumbnail dims) for correct mapping.
        s = self._display_scale
        full_w = pw / s if s > 0 else pw
        full_h = ph / s if s > 0 else ph

        # Draw grid overlay
        if self._show_grid:
            self._draw_grid(painter, dst)

        # Draw WCS overlay
        if self._show_wcs_overlay and self._overlay_stars:
            self._draw_wcs_overlay(painter, dst, full_w, full_h)

        # Draw DSO annotation overlay
        if self._show_dso_overlay and self._dso_annotations:
            self._draw_dso_annotations(painter, dst, full_w, full_h)

        # Draw constellation line overlay
        if self._show_constellation_overlay and self._constellation_lines:
            self._draw_constellation_overlay(painter, dst, full_w, full_h)

        # Draw background sample points
        if self._sample_points or self._sample_mode:
            self._draw_sample_points(painter, dst, full_w, full_h)

        painter.end()

    def _draw_grid(self, painter: QPainter, dst: QRectF):
        if self._pixmap is None:
            return
        pw = self._pixmap.width()
        ph = self._pixmap.height()

        # ── Grid lines: 4×4 equal cells ──────────────────────────────
        pen = QPen(QColor(255, 255, 255, 40), 1)
        pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(pen)
        for i in range(1, 4):
            x = dst.left() + dst.width() * i / 4
            y = dst.top() + dst.height() * i / 4
            painter.drawLine(QPointF(x, dst.top()), QPointF(x, dst.bottom()))
            painter.drawLine(QPointF(dst.left(), y), QPointF(dst.right(), y))

        # ── Center crosshair ──────────────────────────────────────────
        cx = dst.center().x()
        cy = dst.center().y()
        cl = min(dst.width(), dst.height()) * 0.04
        cross_pen = QPen(QColor(255, 255, 255, 80), 1)
        painter.setPen(cross_pen)
        painter.drawLine(QPointF(cx - cl, cy), QPointF(cx + cl, cy))
        painter.drawLine(QPointF(cx, cy - cl), QPointF(cx, cy + cl))

        # ── Pixel coordinate labels at corners ───────────────────────
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        coord_pen = QPen(QColor(255, 255, 255, 120))
        painter.setPen(coord_pen)
        labels = [
            ("0, 0", 4, 14),
            (f"{pw}, {ph}", dst.width() - 60, dst.height() - 4),
        ]
        for text, lx, ly in labels:
            painter.drawText(QPointF(dst.left() + lx, dst.top() + ly), text)

    def _draw_constellation_overlay(self, painter: QPainter, dst: QRectF, pw: int, ph: int):
        if not self._constellation_lines:
            return
        pen = QPen(QColor(120, 180, 255, 160), 1.2)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for segment in self._constellation_lines:
            if len(segment) < 2:
                continue
            pts = []
            for x_img, y_img in segment:
                wx = dst.left() + (x_img / pw) * dst.width()
                wy = dst.top() + (y_img / ph) * dst.height()
                pts.append(QPointF(wx, wy))
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])

    def _draw_wcs_overlay(self, painter: QPainter, dst: QRectF, pw: int, ph: int):
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)

        for x_img, y_img, mag in self._overlay_stars:
            wx = dst.left() + (x_img / pw) * dst.width()
            wy = dst.top() + (y_img / ph) * dst.height()

            # Circle radius scaled by brightness (brighter = larger)
            r = max(4.0, 12.0 - mag)

            painter.setPen(QPen(QColor(80, 200, 255, 200), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(wx, wy), r, r)

            if mag < 10.0:
                painter.setPen(QPen(QColor(150, 220, 255, 180)))
                painter.drawText(QPointF(wx + r + 2, wy + 4), f"{mag:.1f}")

    def _draw_dso_annotations(self, painter: QPainter, dst: QRectF, pw: int, ph: int):
        # Colour by object type
        _type_colors = {
            "G":   QColor(255, 200,  80, 220),   # galaxies — gold
            "EN":  QColor(255, 100, 100, 220),   # emission nebulae — red
            "PN":  QColor(100, 220, 255, 220),   # planetary nebulae — cyan
            "SNR": QColor(255, 140,  60, 220),   # SNR — orange
            "OC":  QColor(180, 255, 180, 220),   # open clusters — green
            "GC":  QColor(200, 180, 255, 220),   # globular clusters — purple
        }
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)

        for x_img, y_img, name, type_code in self._dso_annotations:
            wx = dst.left() + (x_img / pw) * dst.width()
            wy = dst.top()  + (y_img / ph) * dst.height()
            color = _type_colors.get(type_code, QColor(200, 200, 200, 200))

            # Small circle marker
            painter.setPen(QPen(color, 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(wx, wy), 6.0, 6.0)

            # Name label with subtle drop-shadow
            painter.setPen(QPen(QColor(0, 0, 0, 140), 1))
            painter.drawText(QPointF(wx + 9, wy + 5), name)
            painter.setPen(QPen(color))
            painter.drawText(QPointF(wx + 8, wy + 4), name)

    def _draw_sample_points(self, painter: QPainter, dst: QRectF, pw: int, ph: int):
        for idx, (x_img, y_img) in enumerate(self._sample_points):
            wx = dst.left() + (x_img / pw) * dst.width()
            wy = dst.top() + (y_img / ph) * dst.height()
            r = 8.0
            painter.setPen(QPen(QColor(255, 200, 0, 230), 2))
            painter.setBrush(QColor(255, 200, 0, 40))
            painter.drawEllipse(QPointF(wx, wy), r, r)
            painter.setPen(QPen(QColor(255, 200, 0, 200)))
            font = QFont()
            font.setPointSize(7)
            painter.setFont(font)
            painter.drawText(QPointF(wx + r + 2, wy + 4), str(idx + 1))

        if self._sample_mode:
            # Draw mode indicator
            painter.setPen(QPen(QColor(255, 200, 0)))
            font = QFont()
            font.setPointSize(9)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(QPointF(8, 20), f"Sample mode — {len(self._sample_points)} pts  [RClick=remove]")

    # ── Mouse events ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent):
        img_pos = self._widget_to_image(event.position())

        if self._crop_mode and event.button() == Qt.MouseButton.LeftButton:
            self._crop_origin = event.pos()
            if self._crop_rubber_band is None:
                self._crop_rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
            self._crop_rubber_band.setGeometry(QRect(self._crop_origin, self._crop_origin))
            self._crop_rubber_band.show()
            return

        if self._sample_mode and img_pos is not None:
            s = self._display_scale
            if event.button() == Qt.MouseButton.LeftButton:
                self.sample_placed.emit(img_pos.x() / s, img_pos.y() / s)
                return
            if event.button() == Qt.MouseButton.RightButton:
                self.sample_removed.emit(img_pos.x() / s, img_pos.y() / s)
                return

        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._dragging = True
            self._drag_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._crop_mode and self._crop_rubber_band and self._crop_rubber_band.isVisible():
            self._crop_rubber_band.setGeometry(
                QRect(self._crop_origin, event.pos()).normalized()
            )
            return

        if self._dragging:
            delta = event.pos() - self._drag_start
            self._pan_offset += QPointF(delta.x(), delta.y())
            self._drag_start = event.pos()
            self._fit_to_window = False
            self.update()

        img_pos = self._widget_to_image(event.position())
        if img_pos is not None and self._image_data is not None:
            # Scale from thumbnail-space to full-res-space
            s = self._display_scale
            ix = int(img_pos.x() / s)
            iy = int(img_pos.y() / s)
            data = self._image_data
            if data.ndim == 2:
                if 0 <= iy < data.shape[0] and 0 <= ix < data.shape[1]:
                    self.cursor_position.emit(ix, iy, [float(data[iy, ix])])
            elif data.ndim == 3:
                if 0 <= iy < data.shape[1] and 0 <= ix < data.shape[2]:
                    vals = [float(data[ch, iy, ix]) for ch in range(data.shape[0])]
                    self.cursor_position.emit(ix, iy, vals)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._crop_mode and self._crop_rubber_band and self._crop_rubber_band.isVisible():
            if event.button() == Qt.MouseButton.LeftButton:
                rect = QRect(self._crop_origin, event.pos()).normalized()
                self._crop_rubber_band.hide()
                # Convert widget-space rect corners to image-space
                tl = self._widget_to_image(QPointF(rect.left(), rect.top()))
                br = self._widget_to_image(QPointF(rect.right(), rect.bottom()))
                if tl is not None and br is not None:
                    s = self._display_scale
                    x = max(0, round(tl.x() / s))
                    y = max(0, round(tl.y() / s))
                    w = max(1, round(br.x() / s) - x)
                    h = max(1, round(br.y() / s) - y)
                    self.crop_rect_selected.emit(x, y, w, h)
                # Exit crop mode automatically after selection
                self.set_crop_mode(False)
            return

        if self._dragging:
            self._dragging = False
            cursor = Qt.CursorShape.CrossCursor if self._sample_mode else Qt.CursorShape.ArrowCursor
            self.setCursor(cursor)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        mouse_pos = event.position()
        old_scene = self._widget_to_image(mouse_pos)
        self._fit_to_window = False
        self._zoom = max(0.01, min(50.0, self._zoom * factor))
        new_scene = self._widget_to_image(mouse_pos)
        if old_scene is not None and new_scene is not None:
            dx = (new_scene.x() - old_scene.x()) * self._zoom
            dy = (new_scene.y() - old_scene.y()) * self._zoom
            self._pan_offset += QPointF(dx, dy)
        self.zoom_changed.emit(self._zoom)
        self.update()

    def resizeEvent(self, event):
        if self._fit_to_window:
            self._fit_zoom()
        self._reposition_overlays()
        super().resizeEvent(event)

    def _fit_zoom(self):
        if self._pixmap is None:
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        if pw == 0 or ph == 0:
            return
        self._zoom = min(ww / pw, wh / ph)
        self._pan_offset = QPointF(
            (ww - pw * self._zoom) / 2,
            (wh - ph * self._zoom) / 2,
        )
        self.zoom_changed.emit(self._zoom)

    def _widget_to_image(self, pos: QPointF) -> QPointF | None:
        if self._pixmap is None or self._zoom == 0:
            return None
        x = (pos.x() - self._pan_offset.x()) / self._zoom
        y = (pos.y() - self._pan_offset.y()) / self._zoom
        return QPointF(x, y)
