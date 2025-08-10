from __future__ import annotations

from typing import List, Optional, Callable, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, QRect, QSize, QTimer
from PySide6.QtGui import QImage, QPainter, QPen, QColor
from PySide6.QtWidgets import QWidget

from core.player_state import PlayerState, TrackLite


def _to_qimage(frame: np.ndarray) -> QImage:
    if frame is None:
        return QImage()
    if len(frame.shape) == 3:
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
    else:
        h, w = frame.shape
        return QImage(frame.data, w, h, w, QImage.Format.Format_Grayscale8).copy()


class VideoWidget(QWidget):
    """Widget that renders frames and draws overlays from tracks."""

    def __init__(
        self,
        state: PlayerState,
        parent: Optional[QWidget] = None,
        on_track_selected: Optional[Callable[[int], None]] = None,
    ):
        super().__init__(parent)
        self.state = state
        self.current_frame_image: Optional[QImage] = None
        self.setMinimumSize(640, 360)
        self.on_track_selected = on_track_selected

        # zoom state
        self.zoom: float = 1.0
        self.zoom_center_img: Optional[Tuple[float, float]] = None
        self._hud_text: Optional[str] = None

    def sizeHint(self) -> QSize:
        return QSize(960, 540)

    def set_frame(self, frame: np.ndarray) -> None:
        self.current_frame_image = _to_qimage(frame)
        self.update()

    def paintEvent(self, event):  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if self.current_frame_image is None or self.current_frame_image.isNull():
            return

        # Fit frame into widget while preserving aspect ratio
        target_rect = self._fit_rect(self.current_frame_image.width(), self.current_frame_image.height(), self.rect())

        source_rect = self.current_frame_image.rect()

        # Determine source rect for detailed mode or zoomed overview
        if self.state.current_mode == 'detailed' and self.state.current_track_id is not None:
            det_bbox = self._bbox_for_track_at_frame(self.state.current_track_id, self.state.current_frame)
            if det_bbox:
                x1, y1, x2, y2 = det_bbox
                source_rect = QRect(x1, y1, x2 - x1, y2 - y1)
        elif self.zoom != 1.0:
            img_w = self.current_frame_image.width()
            img_h = self.current_frame_image.height()
            cx = img_w / 2 if self.zoom_center_img is None else self.zoom_center_img[0]
            cy = img_h / 2 if self.zoom_center_img is None else self.zoom_center_img[1]
            half_w = max(1, int((img_w / self.zoom) / 2))
            half_h = max(1, int((img_h / self.zoom) / 2))
            x1 = int(max(0, min(cx - half_w, img_w - 2 * half_w)))
            y1 = int(max(0, min(cy - half_h, img_h - 2 * half_h)))
            source_rect = QRect(x1, y1, min(2 * half_w, img_w), min(2 * half_h, img_h))

        painter.drawImage(target_rect, self.current_frame_image, source_rect)

        # Draw overlays in overview mode (respect source scaling)
        if self.state.current_mode == 'overview' and self.state.video_properties is not None:
            scale_x = target_rect.width() / source_rect.width()
            scale_y = target_rect.height() / source_rect.height()
            offset_x = target_rect.x() - source_rect.x() * scale_x
            offset_y = target_rect.y() - source_rect.y() * scale_y

            pen = QPen(Qt.GlobalColor.green)
            pen.setWidth(2)
            painter.setPen(pen)

            for track in self.state.loaded_tracks:
                if self.state.visible_tracks and track.track_id not in self.state.visible_tracks:
                    continue
                # draw bbox for this frame if available
                for d in track.detections:
                    if d.frame_idx == self.state.current_frame:
                        x1, y1, x2, y2 = d.bbox
                        rx1 = int(offset_x + x1 * scale_x)
                        ry1 = int(offset_y + y1 * scale_y)
                        rx2 = int(offset_x + x2 * scale_x)
                        ry2 = int(offset_y + y2 * scale_y)
                        painter.drawRect(QRect(rx1, ry1, rx2 - rx1, ry2 - ry1))
                        painter.drawText(
                            QRect(rx1, max(0, ry1 - 18), 80, 16), Qt.AlignmentFlag.AlignLeft, f'ID:{track.track_id}'
                        )
                        break

        # HUD text
        if self._hud_text:
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(
                self.rect().adjusted(10, 10, -10, -10),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                self._hud_text,
            )

        painter.end()

    @staticmethod
    def _fit_rect(img_w: int, img_h: int, bounds: QRect) -> QRect:
        if img_w == 0 or img_h == 0:
            return QRect(bounds.x(), bounds.y(), bounds.width(), bounds.height())
        scale = min(bounds.width() / img_w, bounds.height() / img_h)
        w = int(img_w * scale)
        h = int(img_h * scale)
        x = bounds.x() + (bounds.width() - w) // 2
        y = bounds.y() + (bounds.height() - h) // 2
        return QRect(x, y, w, h)

    def _bbox_for_track_at_frame(self, track_id: int, frame_idx: int) -> Optional[Tuple[int, int, int, int]]:
        for t in self.state.loaded_tracks:
            if t.track_id != track_id:
                continue
            for d in t.detections:
                if d.frame_idx == frame_idx:
                    x1, y1, x2, y2 = d.bbox
                    return int(x1), int(y1), int(x2), int(y2)
        return None

    def mousePressEvent(self, event):  # type: ignore[override]
        if self.state.current_mode != 'overview' or not self.on_track_selected:
            return
        # map mouse to image coords using inverse of drawing transform (with zoom)
        target_rect = self._fit_rect(self.current_frame_image.width(), self.current_frame_image.height(), self.rect())
        source_rect = self.current_frame_image.rect()
        if self.zoom != 1.0:
            img_w = self.current_frame_image.width()
            img_h = self.current_frame_image.height()
            cx = img_w / 2 if self.zoom_center_img is None else self.zoom_center_img[0]
            cy = img_h / 2 if self.zoom_center_img is None else self.zoom_center_img[1]
            half_w = max(1, int((img_w / self.zoom) / 2))
            half_h = max(1, int((img_h / self.zoom) / 2))
            x1 = int(max(0, min(cx - half_w, img_w - 2 * half_w)))
            y1 = int(max(0, min(cy - half_h, img_h - 2 * half_h)))
            source_rect = QRect(x1, y1, min(2 * half_w, img_w), min(2 * half_h, img_h))

        if not target_rect.contains(event.position().toPoint()):
            return
        px = (event.position().x() - target_rect.x()) / target_rect.width()
        py = (event.position().y() - target_rect.y()) / target_rect.height()
        img_x = int(source_rect.x() + px * source_rect.width())
        img_y = int(source_rect.y() + py * source_rect.height())

        # find top-most track bbox under cursor
        for t in self.state.loaded_tracks:
            if self.state.visible_tracks and t.track_id not in self.state.visible_tracks:
                continue
            for d in t.detections:
                if d.frame_idx != self.state.current_frame:
                    continue
                x1, y1, x2, y2 = d.bbox
                if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                    self.on_track_selected(t.track_id)
                    return

    def wheelEvent(self, event):  # type: ignore[override]
        # zoom in/out around mouse position
        if self.current_frame_image is None:
            return
        angle = event.angleDelta().y()
        factor = 1.0 + (0.1 if angle > 0 else -0.1)
        new_zoom = float(np.clip(self.zoom * factor, 0.25, 8.0))
        if abs(new_zoom - self.zoom) < 1e-3:
            return
        # update zoom center to mouse position in image coords
        target_rect = self._fit_rect(self.current_frame_image.width(), self.current_frame_image.height(), self.rect())
        if target_rect.contains(event.position().toPoint()):
            px = (event.position().x() - target_rect.x()) / target_rect.width()
            py = (event.position().y() - target_rect.y()) / target_rect.height()
            img_x = int(px * self.current_frame_image.width())
            img_y = int(py * self.current_frame_image.height())
            self.zoom_center_img = (img_x, img_y)
        self.zoom = new_zoom
        self._hud_text = f'Zoom: {self.zoom:.2f}x'
        self.update()
        QTimer.singleShot(1000, self.clear_hud)

    def reset_zoom(self) -> None:
        self.zoom = 1.0
        self.zoom_center_img = None
        self._hud_text = None

    def show_hud(self, text: str) -> None:
        self._hud_text = text
        self.update()

    def clear_hud(self) -> None:
        self._hud_text = None
        self.update()
