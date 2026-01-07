from __future__ import annotations

import math
from typing import Optional, Callable, Tuple

import numpy as np
import cv2
from PySide6.QtCore import Qt, QRect, QRectF, QSize, QTimer
from PySide6.QtGui import QImage, QPainter, QPen, QColor
from PySide6.QtWidgets import QWidget

from ..core.player_state import PlayerState
from ...settings import (
    OUTPUT_WIDTH,
    OUTPUT_HEIGHT,
    TARGET_BBOX_HEIGHT_RATIO,
    TARGET_BBOX_WIDTH_RATIO,
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
        self.setMouseTracking(True)

        # Overview view params (match frontend/src/ui/player/rendering.ts mental model)
        self.ov_zoom: float = 1.0
        self.ov_offset_x: float = 0.0
        self.ov_offset_y: float = 0.0
        self.hovered_track_id: Optional[int] = None
        self._is_panning: bool = False
        self._pan_last_x: float = 0.0
        self._pan_last_y: float = 0.0

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
            # Match TS frontend: render detailed crop into the full available widget rect (not a fixed-aspect sub-rect).
            target_rect = self.rect()
            det = self._closest_det_for_track_at_frame(self.state.current_track_id, self.state.current_frame)
            if det is not None:
                # Render directly at target_rect size using high-quality OpenCV interpolation
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
                x1, y1, x2, y2 = det.bbox
                self._draw_detailed_with_padding(
                    painter,
                    (int(x1), int(y1), int(x2), int(y2)),
                    bool(getattr(det, 'interpolated', False)),
                    target_rect,
                )

            else:
                # Draw full frame into the widget (best-effort fallback)
                src = self.current_frame_image.rect()
                is_upscale = target_rect.width() > src.width() or target_rect.height() > src.height()
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, is_upscale)
                painter.drawImage(target_rect, self.current_frame_image, src)
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        else:
            # Overview rendering: match frontend/src/ui/player/rendering.ts logic
            vid_w = float(self.current_frame_image.width())
            vid_h = float(self.current_frame_image.height())
            base = self._compute_base_rect(float(self.width()), float(self.height()), vid_w, vid_h)
            s_base = float(base['scale']) * float(self.ov_zoom)
            cx = float(base['x']) + float(base['w']) * 0.5 + float(self.ov_offset_x)
            cy = float(base['y']) + float(base['h']) * 0.5 + float(self.ov_offset_y)

            dx, dy, da = self.state.get_stabilization_at_frame(int(self.state.current_frame))

            painter.save()
            painter.translate(cx, cy)
            painter.scale(s_base, s_base)
            painter.translate(float(dx), float(dy))
            painter.rotate(float(da) * 180.0 / math.pi)

            is_upscale = s_base > 1.0
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, is_upscale)
            painter.drawImage(QRectF(-vid_w * 0.5, -vid_h * 0.5, vid_w, vid_h), self.current_frame_image)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

            if self.state.current_mode == 'overview' and self.hovered_track_id is not None:
                det = self._det_for_track_at_frame(self.hovered_track_id, int(self.state.current_frame))
                if det is not None:
                    x1, y1, x2, y2 = det.bbox
                    rx1 = float(x1) - vid_w * 0.5
                    ry1 = float(y1) - vid_h * 0.5
                    rw = max(1.0, float(x2) - float(x1))
                    rh = max(1.0, float(y2) - float(y1))
                    is_interpolated = bool(getattr(det, 'interpolated', False))
                    pen = QPen(QColor(249, 115, 22) if is_interpolated else QColor(16, 185, 129))
                    pen.setWidthF(2.0 / max(1e-6, s_base))
                    painter.setPen(pen)
                    painter.drawRect(QRectF(rx1, ry1, rw, rh))

            painter.restore()

        # HUD text
        if self._hud_text:
            text_width = painter.fontMetrics().boundingRect(self._hud_text).width()
            text_height = 10
            painter.fillRect(QRect(10, 10, 10 + text_width, 10 + text_height), QColor(255, 255, 255))
            painter.setPen(QPen(QColor(0, 0, 0)))
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

    def _det_for_track_at_frame(self, track_id: int, frame_idx: int):
        for t in self.state.loaded_tracks:
            if t.track_id != track_id:
                continue
            for d in t.detections:
                if d.frame_idx == frame_idx:
                    return d
        return None

    def _closest_det_for_track_at_frame(self, track_id: int, frame_idx: int):
        track = next((t for t in self.state.loaded_tracks if t.track_id == track_id), None)
        if track is None or not track.detections:
            return None
        best = track.detections[0]
        best_dist = abs(int(best.frame_idx) - int(frame_idx))
        for d in track.detections[1:]:
            dist = abs(int(d.frame_idx) - int(frame_idx))
            if dist < best_dist:
                best = d
                best_dist = dist
                if best_dist == 0:
                    break
        return best

    @staticmethod
    def _compute_base_rect(out_w: float, out_h: float, vid_w: float, vid_h: float) -> dict[str, float]:
        if vid_w <= 0 or vid_h <= 0 or out_w <= 0 or out_h <= 0:
            return {'x': 0.0, 'y': 0.0, 'w': 0.0, 'h': 0.0, 'scale': 1.0}
        scale = min(out_w / vid_w, out_h / vid_h)
        disp_w = vid_w * scale
        disp_h = vid_h * scale
        off_x = (out_w - disp_w) / 2.0
        off_y = (out_h - disp_h) / 2.0
        return {'x': off_x, 'y': off_y, 'w': disp_w, 'h': disp_h, 'scale': scale}

    def _pick_track_at_screen_point(self, px: float, py: float) -> Optional[int]:
        if self.current_frame_image is None or self.current_frame_image.isNull():
            return None

        vid_w = float(self.current_frame_image.width())
        vid_h = float(self.current_frame_image.height())
        base = self._compute_base_rect(float(self.width()), float(self.height()), vid_w, vid_h)
        s_base = float(base['scale']) * float(self.ov_zoom)
        if s_base <= 0:
            return None

        dx, dy, da = self.state.get_stabilization_at_frame(int(self.state.current_frame))
        cx = float(base['x']) + float(base['w']) * 0.5 + float(self.ov_offset_x)
        cy = float(base['y']) + float(base['h']) * 0.5 + float(self.ov_offset_y)

        # Invert the overview transform chain:
        # Translate(cx,cy) -> Scale(s) -> Translate(stab) -> Rotate(stab)
        dx0 = px - cx
        dy0 = py - cy
        dx1 = dx0 / s_base
        dy1 = dy0 / s_base
        dx2 = dx1 - float(dx)
        dy2 = dy1 - float(dy)

        cos = math.cos(-float(da))
        sin = math.sin(-float(da))
        dx3 = dx2 * cos - dy2 * sin
        dy3 = dx2 * sin + dy2 * cos

        img_x = dx3 + vid_w * 0.5
        img_y = dy3 + vid_h * 0.5

        for track_id, det in self.state.detections_by_frame.get(int(self.state.current_frame), []):
            if self.state.visible_tracks and track_id not in self.state.visible_tracks:
                continue
            x1, y1, x2, y2 = det.bbox
            if float(x1) <= img_x <= float(x2) and float(y1) <= img_y <= float(y2):
                return int(track_id)

        return None

    def mousePressEvent(self, event):  # type: ignore[override]
        if self.current_frame_image is None or self.current_frame_image.isNull():
            return
        if self.state.current_mode != 'overview' or not self.on_track_selected:
            return
        if event.button() == Qt.MouseButton.RightButton:
            self._is_panning = True
            self._pan_last_x = float(event.position().x())
            self._pan_last_y = float(event.position().y())
            return
        picked = self._pick_track_at_screen_point(float(event.position().x()), float(event.position().y()))
        if picked is not None:
            self.on_track_selected(int(picked))
            return

    def wheelEvent(self, event):  # type: ignore[override]
        # Overview zoom (frontend-like): update zoom and offsets so the point under cursor stays put.
        if self.current_frame_image is None:
            return
        if self.state.current_mode != 'overview':
            return
        angle = event.angleDelta().y()
        factor = 1.0 + (0.1 if angle > 0 else -0.1)
        new_zoom = float(np.clip(self.ov_zoom * factor, 0.25, 8.0))
        if abs(new_zoom - self.ov_zoom) < 1e-6:
            return

        mouse_x = float(event.position().x())
        mouse_y = float(event.position().y())
        base = self._compute_base_rect(
            float(self.width()),
            float(self.height()),
            float(self.current_frame_image.width()),
            float(self.current_frame_image.height()),
        )
        s_old = float(base['scale']) * float(self.ov_zoom)
        s_new = float(base['scale']) * float(new_zoom)
        cx_old = float(base['x']) + float(base['w']) * 0.5 + float(self.ov_offset_x)
        cy_old = float(base['y']) + float(base['h']) * 0.5 + float(self.ov_offset_y)

        if s_old > 0:
            rel_x = mouse_x - cx_old
            rel_y = mouse_y - cy_old
            new_cx = mouse_x - rel_x * (s_new / s_old)
            new_cy = mouse_y - rel_y * (s_new / s_old)
            cx_base = float(base['x']) + float(base['w']) * 0.5
            cy_base = float(base['y']) + float(base['h']) * 0.5
            self.ov_offset_x = new_cx - cx_base
            self.ov_offset_y = new_cy - cy_base

        self.ov_zoom = new_zoom
        self._hud_text = f'Zoom: {self.ov_zoom:.2f}x'
        self.update()
        QTimer.singleShot(1000, self.clear_hud)

    def reset_zoom(self) -> None:
        self.ov_zoom = 1.0
        self.ov_offset_x = 0.0
        self.ov_offset_y = 0.0
        self._hud_text = None
        self._prev_scale_by_track_id.clear()
        self.hovered_track_id = None
        self.update()

    def show_hud(self, text: str) -> None:
        self._hud_text = text
        self.update()

    def clear_hud(self) -> None:
        self._hud_text = None
        self.update()

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if self.state.current_mode != 'overview':
            return
        if self.current_frame_image is None or self.current_frame_image.isNull():
            return
        if self._is_panning and (event.buttons() & Qt.MouseButton.RightButton):
            x = float(event.position().x())
            y = float(event.position().y())
            self.ov_offset_x += x - self._pan_last_x
            self.ov_offset_y += y - self._pan_last_y
            self._pan_last_x = x
            self._pan_last_y = y
            self.update()
            return
        picked = self._pick_track_at_screen_point(float(event.position().x()), float(event.position().y()))
        if picked != self.hovered_track_id:
            self.hovered_track_id = picked
            self.update()

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton:
            self._is_panning = False

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
        bbox_w = max(1, x2 - x1)
        bbox_h = max(1, y2 - y1)
        s_height = (TARGET_BBOX_HEIGHT_RATIO * out_h) / bbox_h
        s_width = (TARGET_BBOX_WIDTH_RATIO * out_w) / bbox_w
        s_inst = min(s_height, s_width)
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
        if track_id < 0:
            return QColor(255, 0, 0)
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
        self,
        painter: QPainter,
        det_bbox: Tuple[int, int, int, int],
        det_interpolated: bool,
        target_rect: QRect,
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
        # Choose scale to respect height and width target ratios (match TS frontend logic)
        s_height = (TARGET_BBOX_HEIGHT_RATIO * out_h) / bbox_h
        s_width = (TARGET_BBOX_WIDTH_RATIO * out_w) / bbox_w
        s_inst = min(s_height, s_width)
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

        # Draw bbox overlay (in the composed output coordinate system)
        rx1 = int((x1 - win_x1) * s)
        ry1 = int((y1 - win_y1) * s)
        rx2 = int((x2 - win_x1) * s)
        ry2 = int((y2 - win_y1) * s)
        w = max(1, rx2 - rx1)
        h = max(1, ry2 - ry1)
        pen = QPen(QColor(249, 115, 22) if det_interpolated else QColor(16, 185, 129))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(QRect(target_rect.x() + rx1, target_rect.y() + ry1, w, h))
