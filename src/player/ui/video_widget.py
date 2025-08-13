from __future__ import annotations

from typing import Optional, Callable, Tuple

import numpy as np
import cv2
from PySide6.QtCore import Qt, QRect, QSize, QTimer
from PySide6.QtGui import QImage, QPainter, QPen, QColor
from PySide6.QtWidgets import QWidget

from player.core.player_state import PlayerState
from settings import (
    OUTPUT_WIDTH,
    OUTPUT_HEIGHT,
    TARGET_BBOX_HEIGHT_RATIO,
    SMOOTHING_ALPHA,
    MIN_SCALE,
    MAX_SCALE,
)


def _to_qimage(frame: np.ndarray) -> QImage:
    if frame is None:
        return QImage()
    if len(frame.shape) == 3:
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        # Use BGR888 directly to avoid an extra color conversion
        return QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888).copy()
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
        self.current_frame_np: Optional[np.ndarray] = None
        self.setMinimumSize(640, 360)
        self.on_track_selected = on_track_selected
        # Hint to Qt that we fully paint the widget to avoid unnecessary clears
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        # zoom state (persistent view rectangle in image coords)
        self.zoom: float = 1.0
        self.view_rect_img: Optional[QRect] = None
        self._hud_text: Optional[str] = None
        self._prev_scale_by_track_id: dict[int, float] = {}
        self._color_cache: dict[int, QColor] = {}

    def sizeHint(self) -> QSize:
        return QSize(960, 540)

    def set_frame(self, frame: np.ndarray) -> None:
        self.current_frame_image = _to_qimage(frame)
        self.current_frame_np = frame
        self.update()

    def paintEvent(self, event):  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if self.current_frame_image is None or self.current_frame_image.isNull():
            return

        # Draw based on mode
        if self.state.current_mode == 'detailed' and self.state.current_track_id is not None:
            target_rect = self._fit_aspect_rect(OUTPUT_WIDTH, OUTPUT_HEIGHT, self.rect())
            det_bbox = self._bbox_for_track_at_frame(self.state.current_track_id, self.state.current_frame)
            if det_bbox:
                # Render directly at target_rect size using high-quality OpenCV interpolation
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
                self._draw_detailed_with_padding(painter, det_bbox, target_rect)

            else:
                # Draw full frame into fixed-aspect target
                src = self.current_frame_image.rect()
                is_upscale = target_rect.width() > src.width() or target_rect.height() > src.height()
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, is_upscale)
                painter.drawImage(target_rect, self.current_frame_image, src)
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        else:
            # Fit whole frame and optionally apply interactive zoom view
            target_rect = self._fit_rect(
                self.current_frame_image.width(), self.current_frame_image.height(), self.rect()
            )
            source_rect = self.current_frame_image.rect()
            if self.zoom != 1.0 and self.view_rect_img is not None:
                source_rect = self._clamp_view_rect(
                    self.view_rect_img, self.current_frame_image.width(), self.current_frame_image.height()
                )
            # Toggle high-quality only when upscaling the selected source area
            is_upscale = target_rect.width() > source_rect.width() or target_rect.height() > source_rect.height()
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, is_upscale)
            painter.drawImage(target_rect, self.current_frame_image, source_rect)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

            # Draw overlays in overview mode (respect source scaling)
            if self.state.current_mode == 'overview' and self.state.video_properties is not None:
                scale_x = target_rect.width() / source_rect.width()
                scale_y = target_rect.height() / source_rect.height()
                offset_x = target_rect.x() - source_rect.x() * scale_x
                offset_y = target_rect.y() - source_rect.y() * scale_y

                current_dets = self.state.detections_by_frame.get(self.state.current_frame, [])
                for track_id, d in current_dets:
                    if self.state.visible_tracks and track_id not in self.state.visible_tracks:
                        continue
                    x1, y1, x2, y2 = d.bbox
                    rx1 = int(offset_x + x1 * scale_x)
                    ry1 = int(offset_y + y1 * scale_y)
                    rx2 = int(offset_x + x2 * scale_x)
                    ry2 = int(offset_y + y2 * scale_y)
                    pen = QPen(self._color_for_track(track_id))
                    pen.setWidth(2)
                    painter.setPen(pen)
                    painter.drawRect(QRect(rx1, ry1, rx2 - rx1, ry2 - ry1))
                    painter.drawText(QRect(rx1, max(0, ry1 - 18), 80, 16), Qt.AlignmentFlag.AlignLeft, f'ID:{track_id}')

        # HUD text
        if self._hud_text:
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(
                self.rect().adjusted(10, 10, -10, -10),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                self._hud_text,
            )

        painter.end

    @staticmethod
    def _fit_rect(img_w: int, img_h: int, bounds: QRect) -> QRect:
        if img_w == 0 or img_h == 0:
            return QRect(bounds.x(), bounds.y(), bounds.width(), bounds.height())
        # keep scale reasonable to avoid extreme upscaling
        scale = min(bounds.width() / img_w, bounds.height() / img_h)
        w = int(img_w * scale)
        h = int(img_h * scale)
        x = bounds.x() + (bounds.width() - w) // 2
        y = bounds.y() + (bounds.height() - h) // 2
        return QRect(x, y, w, h)

    @staticmethod
    def _fit_aspect_rect(aspect_w: int, aspect_h: int, bounds: QRect) -> QRect:
        if aspect_w <= 0 or aspect_h <= 0:
            return QRect(bounds.x(), bounds.y(), bounds.width(), bounds.height())
        scale = min(bounds.width() / aspect_w, bounds.height() / aspect_h)
        w = int(aspect_w * scale)
        h = int(aspect_h * scale)
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
        if self.current_frame_image is None or self.current_frame_image.isNull():
            return
        if self.state.current_mode != 'overview' or not self.on_track_selected:
            return
        # map mouse to image coords using inverse of drawing transform (with zoom)
        target_rect = self._fit_rect(self.current_frame_image.width(), self.current_frame_image.height(), self.rect())
        source_rect = (
            self.view_rect_img
            if (self.zoom != 1.0 and self.view_rect_img is not None)
            else self.current_frame_image.rect()
        )

        if not target_rect.contains(event.position().toPoint()):
            return
        px = (event.position().x() - target_rect.x()) / target_rect.width()
        py = (event.position().y() - target_rect.y()) / target_rect.height()
        img_x = int(source_rect.x() + px * source_rect.width())
        img_y = int(source_rect.y() + py * source_rect.height())

        # find top-most track bbox under cursor
        for track_id, d in self.state.detections_by_frame.get(self.state.current_frame, []):
            if self.state.visible_tracks and track_id not in self.state.visible_tracks:
                continue
            x1, y1, x2, y2 = d.bbox
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                self.on_track_selected(track_id)
                return

    def wheelEvent(self, event):  # type: ignore[override]
        # zoom in/out around mouse position; relative to current view only
        if self.current_frame_image is None:
            return
        if self.state.current_mode != 'overview':
            return
        angle = event.angleDelta().y()
        factor = 1.0 + (0.1 if angle > 0 else -0.1)
        new_zoom = float(np.clip(self.zoom * factor, 0.25, 8.0))
        if abs(new_zoom - self.zoom) < 1e-3:
            return

        img_w = self.current_frame_image.width()
        img_h = self.current_frame_image.height()
        target_rect = self._fit_rect(img_w, img_h, self.rect())
        curr_source = (
            self.view_rect_img
            if (self.zoom != 1.0 and self.view_rect_img is not None)
            else self.current_frame_image.rect()
        )
        # Determine zoom center: stationary under mouse for zoom-in; center for zoom-out
        if new_zoom > self.zoom:
            if not target_rect.contains(event.position().toPoint()):
                return
            px = (event.position().x() - target_rect.x()) / max(1, target_rect.width())
            py = (event.position().y() - target_rect.y()) / max(1, target_rect.height())
            img_x = curr_source.x() + px * curr_source.width()
            img_y = curr_source.y() + py * curr_source.height()
        else:
            px = 0.5
            py = 0.5
            img_x = curr_source.center().x()
            img_y = curr_source.center().y()

        # Compute new view relative to current source
        ratio = self.zoom / new_zoom  # <1 zooming in, >1 zooming out
        new_w = max(1, int(curr_source.width() * ratio))
        new_h = max(1, int(curr_source.height() * ratio))
        # Keep the pixel under the mouse stationary by aligning new rect so that
        # img_x maps to the same px position: new_x = img_x - px * new_w
        x1 = int(img_x - px * new_w)
        y1 = int(img_y - py * new_h)
        proposed = QRect(x1, y1, new_w, new_h)

        if new_zoom > self.zoom:
            # Zooming in: clamp to current source to avoid revealing outside content
            bounded = self._clamp_rect_to_bounds(proposed, curr_source)
        else:
            # Zooming out: clamp to full image
            bounded = self._clamp_view_rect(proposed, img_w, img_h)

        self.zoom = new_zoom
        self.view_rect_img = bounded

        self._hud_text = f'Zoom: {self.zoom:.2f}x'
        self.update()
        QTimer.singleShot(1000, self.clear_hud)

    def reset_zoom(self) -> None:
        self.zoom = 1.0
        self.zoom_center_img = None
        self._hud_text = None
        self._prev_scale_by_track_id.clear()

    def show_hud(self, text: str) -> None:
        self._hud_text = text
        self.update()

    def clear_hud(self) -> None:
        self._hud_text = None
        self.update()

    @staticmethod
    def _clamp_view_rect(rect: QRect, img_w: int, img_h: int) -> QRect:
        x = min(max(0, rect.x()), max(0, img_w - rect.width()))
        y = min(max(0, rect.y()), max(0, img_h - rect.height()))
        w = min(rect.width(), img_w)
        h = min(rect.height(), img_h)
        if w <= 0 or h <= 0:
            return QRect(0, 0, img_w, img_h)
        return QRect(x, y, w, h)

    @staticmethod
    def _clamp_rect_to_bounds(rect: QRect, bounds: QRect) -> QRect:
        # Clamp rect entirely within bounds rect
        w = min(rect.width(), bounds.width())
        h = min(rect.height(), bounds.height())
        x = min(max(bounds.x(), rect.x()), bounds.x() + bounds.width() - w)
        y = min(max(bounds.y(), rect.y()), bounds.y() + bounds.height() - h)
        return QRect(x, y, w, h)

    @staticmethod
    def _clamp_rect_to_bounds_preserve_aspect(rect: QRect, bounds: QRect) -> QRect:
        """Clamp rect within bounds while keeping its aspect ratio.

        - Translate inside bounds when possible without resizing.
        - If larger than bounds along any axis, scale down uniformly.
        """
        req_w = max(1, rect.width())
        req_h = max(1, rect.height())
        max_w = max(1, bounds.width())
        max_h = max(1, bounds.height())

        # Uniformly scale down if needed
        if req_w > max_w or req_h > max_h:
            scale = min(max_w / req_w, max_h / req_h)
            new_w = max(1, int(req_w * scale))
            new_h = max(1, int(req_h * scale))
        else:
            new_w = req_w
            new_h = req_h

        # Preserve center, then translate to fit within bounds
        cx = rect.x() + req_w // 2
        cy = rect.y() + req_h // 2
        x = cx - new_w // 2
        y = cy - new_h // 2

        x = max(bounds.x(), min(x, bounds.x() + max_w - new_w))
        y = max(bounds.y(), min(y, bounds.y() + max_h - new_h))

        return QRect(int(x), int(y), int(new_w), int(new_h))

    def _compute_detailed_source_rect(
        self,
        det_bbox: Tuple[int, int, int, int],
        img_w: int,
        img_h: int,
        out_w: int,
        out_h: int,
    ) -> QRect:
        x1, y1, x2, y2 = det_bbox
        bbox_h = max(1, y2 - y1)
        s_inst = (TARGET_BBOX_HEIGHT_RATIO * out_h) / bbox_h
        s_inst = float(np.clip(s_inst, MIN_SCALE, MAX_SCALE))

        track_id = self.state.current_track_id
        if track_id is not None and 0.0 < SMOOTHING_ALPHA < 1.0:
            prev = self._prev_scale_by_track_id.get(track_id)
        else:
            prev = None
        if prev is not None and 0.0 < SMOOTHING_ALPHA < 1.0:
            s = SMOOTHING_ALPHA * prev + (1.0 - SMOOTHING_ALPHA) * s_inst
        else:
            s = s_inst
        if track_id is not None:
            self._prev_scale_by_track_id[track_id] = s

        # bbox centre in original image coords
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        # crop window in scaled-image coordinates
        scx = cx * s
        scy = cy * s
        sx1 = scx - out_w * 0.5
        sy1 = scy - out_h * 0.5
        sx2 = sx1 + out_w
        sy2 = sy1 + out_h
        # map back to original image coords
        ox1 = int(max(0, min(img_w, sx1 / s)))
        oy1 = int(max(0, min(img_h, sy1 / s)))
        ox2 = int(max(0, min(img_w, sx2 / s)))
        oy2 = int(max(0, min(img_h, sy2 / s)))
        rect = QRect(ox1, oy1, max(1, ox2 - ox1), max(1, oy2 - oy1))
        # Ensure fully within image bounds while preserving aspect ratio
        return self._clamp_rect_to_bounds_preserve_aspect(rect, QRect(0, 0, img_w, img_h))

    def _color_for_track(self, track_id: int) -> QColor:
        if track_id in self._color_cache:
            return self._color_cache[track_id]
        # Golden-ratio distributed hue for distinct colors
        golden = 0.618033988749895
        h = (hash(track_id) * golden) % 1.0
        hue = int(h * 359)
        color = QColor.fromHsv(hue, 220, 255)
        self._color_cache[track_id] = color
        return color

    # ------------------------------------------------------------------
    # Detailed-mode renderer with black padding (no stretching)
    # ------------------------------------------------------------------
    def _draw_detailed_with_padding(
        self, painter: QPainter, det_bbox: Tuple[int, int, int, int], target_rect: QRect
    ) -> None:
        if self.current_frame_image is None or self.current_frame_image.isNull():
            return
        if self.current_frame_np is None:
            return
        frame = self.current_frame_np
        img_h, img_w = frame.shape[:2]
        out_w = max(1, int(target_rect.width()))
        out_h = max(1, int(target_rect.height()))

        x1, y1, x2, y2 = det_bbox
        bbox_w = max(1, x2 - x1)
        bbox_h = max(1, y2 - y1)
        # Choose scale to respect height target ratio but also ensure bbox width fits
        s_height = (TARGET_BBOX_HEIGHT_RATIO * out_h) / bbox_h
        s_width_limit = out_w / bbox_w
        s_inst = min(s_height, s_width_limit)
        s_inst = float(np.clip(s_inst, MIN_SCALE, MAX_SCALE))

        track_id = self.state.current_track_id
        prev = self._prev_scale_by_track_id.get(track_id, None) if track_id is not None else None
        if prev is not None and 0.0 < SMOOTHING_ALPHA < 1.0:
            s = SMOOTHING_ALPHA * prev + (1.0 - SMOOTHING_ALPHA) * s_inst
        else:
            s = s_inst
        if track_id is not None:
            self._prev_scale_by_track_id[track_id] = s

        # Compute crop in original image coords directly (avoid scaling full frame)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        crop_w = out_w / s
        crop_h = out_h / s
        win_x1 = cx - crop_w / 2.0
        win_y1 = cy - crop_h / 2.0
        win_x2 = win_x1 + crop_w
        win_y2 = win_y1 + crop_h
        # Intersection with original image
        src_x1 = int(max(0, np.floor(win_x1)))
        src_y1 = int(max(0, np.floor(win_y1)))
        src_x2 = int(min(img_w, np.ceil(win_x2)))
        src_y2 = int(min(img_h, np.ceil(win_y2)))
        # Destination placement within output, preserving scale s
        dst_x1 = int(max(0, np.floor((src_x1 - win_x1) * s)))
        dst_y1 = int(max(0, np.floor((src_y1 - win_y1) * s)))
        dst_x2 = int(min(out_w, np.ceil((src_x2 - win_x1) * s)))
        dst_y2 = int(min(out_h, np.ceil((src_y2 - win_y1) * s)))
        copy_w = max(0, dst_x2 - dst_x1)
        copy_h = max(0, dst_y2 - dst_y1)

        out_np = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        if copy_w > 0 and copy_h > 0:
            src_roi = frame[src_y1:src_y2, src_x1:src_x2]
            # Resize source ROI to destination size (copy_w x copy_h)
            interp = cv2.INTER_LANCZOS4 if s > 1.0 else cv2.INTER_AREA
            resized = cv2.resize(src_roi, (copy_w, copy_h), interpolation=interp)
            out_np[dst_y1:dst_y2, dst_x1:dst_x2] = resized

        # Draw directly with no additional scaling
        composed = _to_qimage(out_np)
        painter.drawImage(target_rect, composed)
