from __future__ import annotations

import os
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

# Make project importable when run as a script
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Ensure ultralytics config dir is the repo one (weights/settings)
server_root = project_root
os.environ.setdefault('YOLO_CONFIG_DIR', str(server_root / 'ultralytics'))

from ultralytics import YOLO  # type: ignore
import torch
from ultralytics.utils.ops import scale_boxes as _ops_scale_boxes

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QToolBar, QMessageBox
from PySide6.QtGui import QAction

from inference.src.player.core.player_state import PlayerState, VideoProperties, TrackLite, DetectionLite
from inference.src.player.core.video_manager import VideoManager
from inference.src.player.ui.video_widget import VideoWidget
from inference.src.settings import YOLO_MODEL_PATH, DETECTOR_IOU_THRESHOLD, DETECTOR_CONFIDENCE_THRESHOLD


@dataclass
class _Params:
    conf: float
    iou: float
    step: float
    agnostic: bool
    classes: list[int] | None  # keep only these classes; None = all
    contain: bool
    contain_thresh: float
    pre_nms: bool


class _SingleFrameDetector:
    """Thin wrapper around YOLO for single-frame detection with adjustable thresholds."""

    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f'YOLO model not found: {model_path}')
        self.model = YOLO(model=str(model_path), verbose=False)

    def detect(
        self,
        frame_bgr: np.ndarray,
        conf: float,
        iou: float,
        *,
        agnostic: bool = False,
        classes: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (boxes_xyxy[N,4], confidences[N], class_ids[N]) for a single frame.

        Args:
                frame_bgr: uint8 BGR frame
                conf: confidence threshold
                iou: IoU threshold
        """
        results = self.model.predict(
            frame_bgr,
            conf=float(conf),
            iou=float(iou),
            agnostic_nms=bool(agnostic),
            classes=classes if classes and len(classes) > 0 else None,
            verbose=False,
            save=False,
            stream=False,
        )
        if not results:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )
        boxes = _to_numpy(r.boxes.xyxy)
        confs = _to_numpy(r.boxes.conf).reshape(-1)
        cls = (
            _to_numpy(r.boxes.cls).reshape(-1).astype(np.int32)
            if getattr(r.boxes, 'cls', None) is not None
            else np.zeros_like(confs, dtype=np.int32)
        )
        return boxes, confs, cls

    def detect_raw_no_nms(
        self,
        frame_bgr: np.ndarray,
        conf: float,
        classes: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run model forward and extract boxes/conf without any NMS.

        Returns boxes scaled to original frame size (xyxy), confidences, class_ids.
        """
        # Ensure predictor initialized with proper shapes
        predictor = getattr(self.model, 'predictor', None)
        if predictor is None or not hasattr(predictor, 'preprocess') or not hasattr(predictor, 'inference'):
            _ = self.model.predict(frame_bgr[..., ::-1], conf=0.001, iou=0.7, verbose=False, save=False)
            predictor = getattr(self.model, 'predictor', None)
        if predictor is None:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )
        # Preprocess expects RGB uint8 array batch
        frame_rgb = frame_bgr[..., ::-1]
        batch = np.expand_dims(frame_rgb, axis=0)
        pred_any = predictor
        func_pre = getattr(pred_any, 'preprocess', None)
        func_inf = getattr(pred_any, 'inference', None)
        if not callable(func_pre) or not callable(func_inf):
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )
        im = func_pre(batch)
        results = func_inf(im)
        # results may be (preds, feats) or just (preds)
        preds_any = results[0] if isinstance(results, (list, tuple)) else results
        if isinstance(preds_any, (list, tuple)):
            preds_any = preds_any[0]
        preds_tensor = preds_any.detach().cpu() if isinstance(preds_any, torch.Tensor) else None
        if preds_tensor is None:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )
        # Ensure shape is (B, no, N)
        p = preds_tensor
        if p.ndim == 3 and p.shape[1] > p.shape[2]:
            # likely (B, N, no)
            p = p.permute(0, 2, 1)
        p = p[0] if p.ndim == 3 else p
        if p.shape[0] == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )
        # Now p is (no, N)
        boxes_xywh = p[:4, :].T  # (N,4)
        cls_logits = p[4:, :].T  # (N,nc) or (N,1)
        if cls_logits.numel() == 0:
            confs = torch.ones((boxes_xywh.shape[0],), dtype=torch.float32)
            cls_ids = torch.zeros_like(confs, dtype=torch.int64)
        else:
            confs, cls_ids = torch.max(torch.sigmoid(cls_logits), dim=1)
        # Apply only confidence filtering in raw pre-NMS mode (floor at 0.1)
        thresh = max(0.1, float(conf))
        mask = confs >= thresh
        boxes_xywh = boxes_xywh[mask]
        confs = confs[mask]
        cls_ids = cls_ids[mask]
        if boxes_xywh.numel() == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )
        # Convert to xyxy in preprocessed image space
        x, y, w, h = boxes_xywh.unbind(1)
        boxes_xyxy = torch.stack([x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0], dim=1)
        # im is a tensor returned by Ultralytics (batch, ch, h, w)
        if isinstance(im, torch.Tensor):
            in_h, in_w = int(im.shape[-2]), int(im.shape[-1])
        else:
            arr_shape = getattr(im, 'shape', None)
            in_h = int(arr_shape[-2]) if arr_shape is not None else frame_bgr.shape[0]
            in_w = int(arr_shape[-1]) if arr_shape is not None else frame_bgr.shape[1]
        orig_h, orig_w = frame_bgr.shape[:2]
        scaled = _ops_scale_boxes((in_h, in_w), boxes_xyxy, (orig_h, orig_w)).round()
        return (
            scaled.numpy().astype(np.float32),
            confs.numpy().astype(np.float32),
            cls_ids.numpy().astype(np.int32),
        )


def _to_numpy(tensor_or_array):
    try:
        return tensor_or_array.cpu().numpy()
    except AttributeError:
        return np.array(tensor_or_array)


def _ioa(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-min-area (containment score) for boxes a and b.

    Boxes are [x1, y1, x2, y2]. Returns 0..1.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = max(1e-9, min(area_a, area_b))
    return float(inter / denom)


def _suppress_by_containment(boxes: np.ndarray, confs: np.ndarray, thresh: float) -> np.ndarray:
    """Return boolean mask of boxes to keep by suppressing nested boxes.

    A box is removed if it is highly contained in another box with higher or equal
    confidence according to IoA >= thresh.
    """
    n = len(boxes)
    if n == 0:
        return np.zeros((0,), dtype=bool)
    order = np.argsort(-confs)  # high to low
    keep = np.ones(n, dtype=bool)
    for i_idx in range(n):
        i = order[i_idx]
        if not keep[i]:
            continue
        for j_idx in range(i_idx + 1, n):
            j = order[j_idx]
            if not keep[j]:
                continue
            score = _ioa(boxes[j], boxes[i])  # is j inside i?
            if score >= thresh:
                keep[j] = False
    return keep


class DetectionTunerWindow(QMainWindow):
    def __init__(self, video_path: Path, params: _Params, model: Path, models: list[Path] | None = None):
        super().__init__()
        self.setWindowTitle('Windsurf Detection Tuner')

        self.params = params
        self.video = VideoManager(video_path)
        # Model management
        self.models: list[Path] = []
        if models is not None:
            self.models = [p for p in models if p.exists() and p.suffix.lower() == '.pt']
        start_model = model if model.exists() and model.suffix.lower() == '.pt' else YOLO_MODEL_PATH
        if start_model.exists() and start_model.suffix.lower() == '.pt' and start_model not in self.models:
            self.models.insert(0, start_model)
        if not self.models:
            raise RuntimeError('No YOLO .pt models found to load')
        self.model_idx: int = 0
        for i, p in enumerate(self.models):
            if p.resolve() == start_model.resolve():
                self.model_idx = i
                break
        self.detector = _SingleFrameDetector(self.models[self.model_idx])
        # Class names from model, if available
        try:
            self.class_names = self.detector.model.names  # type: ignore[attr-defined]
        except Exception:
            self.class_names = {}

        # Player state + widget
        self.state = PlayerState()
        self.state.reset(
            input_video_path=video_path.as_posix(),
            video_properties=VideoProperties(
                fps=self.video.fps,
                width=self.video.width,
                height=self.video.height,
                total_frames=self.video.total_frames,  # type: ignore[arg-type]
            ),
            loaded_tracks=[],
        )

        self.video_widget = VideoWidget(self.state)
        self.setCentralWidget(self.video_widget)

        # Toolbar with keybinds button
        tb = QToolBar('Help', self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        action_help = QAction('Keybinds', self)
        action_help.setStatusTip('Show all keyboard shortcuts')
        action_help.triggered.connect(self._show_keybinds)
        tb.addAction(action_help)

        # Seek to first frame and run initial detection
        self.state.current_frame = 0
        self.video.seek_frame(0)
        _, frame = self.video.read_frame()
        if frame is not None:
            self.video_widget.set_frame(frame)
        self._run_and_update()

        # Keep a small timer to refresh HUD fading
        self._timer = QTimer(self)
        self._timer.timeout.connect(lambda: None)
        self._timer.start(100)

    # --------------- Core update logic --------------- #
    def _set_frame_index(self, idx: int) -> None:
        idx = max(0, min(idx, self.video.total_frames - 1))
        self.state.current_frame = idx
        self.video.seek_frame(idx)
        _, frame = self.video.read_frame()
        if frame is not None:
            self.video_widget.set_frame(frame)
        self._run_and_update()

    def _run_and_update(self) -> None:
        frame_idx = int(self.state.current_frame)
        # Use current frame image from widget (already BGR)
        frame = self.video_widget.current_frame_np
        if frame is None:
            return

        if self.params.pre_nms:
            boxes, confs, cls_ids = self.detector.detect_raw_no_nms(
                frame,
                self.params.conf,
                classes=self.params.classes,
            )
        else:
            boxes, confs, cls_ids = self.detector.detect(
                frame,
                self.params.conf,
                self.params.iou,
                agnostic=self.params.agnostic,
                classes=self.params.classes,
            )

        # Optional containment-based suppression for single-class nested boxes
        if (not self.params.pre_nms) and self.params.contain and len(boxes) > 1:
            keep = _suppress_by_containment(boxes, confs, self.params.contain_thresh)
            boxes = boxes[keep]
            confs = confs[keep]
            cls_ids = cls_ids[keep] if cls_ids is not None and len(cls_ids) == len(keep) else cls_ids

        # Build transient TrackLites: one per detection at the current frame
        tracks: List[TrackLite] = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            tl = TrackLite(
                track_id=i + 1,
                start_frame=frame_idx,
                end_frame=frame_idx,
                start_time=frame_idx / max(1.0, self.video.fps),
                duration=0.0,
                detection_count=1,
                detections=[
                    DetectionLite(
                        frame_idx=frame_idx,
                        bbox=[x1, y1, x2, y2],
                        confidence=float(confs[i]),
                        interpolated=False,
                    )
                ],
            )
            tracks.append(tl)

        self.state.loaded_tracks = tracks
        # Rebuild fast lookups for overlays
        self.state.visible_tracks = self.state._extract_visible_tracks()
        self.state.detections_by_frame = self.state._rebuild_detection_index()

        self._update_hud(num=len(tracks), cls_ids=cls_ids)
        self.video_widget.update()

    def _update_hud(self, num: int, cls_ids: np.ndarray | None = None) -> None:
        cls_part = ''
        if self.params.classes is not None and len(self.params.classes) > 0:
            names = (
                [self.class_names.get(c, str(c)) for c in self.params.classes]
                if isinstance(self.class_names, dict)
                else [str(c) for c in self.params.classes]
            )
            cls_part = f' | classes {",".join(map(str, names))}'
        if cls_ids is not None and len(cls_ids) > 0 and isinstance(self.class_names, dict):
            unique = {int(c) for c in cls_ids.tolist()}
            names = [self.class_names.get(c, str(c)) for c in sorted(unique)]
            cls_part += f' | seen {",".join(map(str, names))}'
        model_name = self.models[self.model_idx].name if 0 <= self.model_idx < len(self.models) else 'unknown'
        mode_tag = 'pre-NMS' if self.params.pre_nms else 'post-NMS'
        msg = (
            f'frame {self.state.current_frame + 1}/{self.video.total_frames} | '
            f'conf {self.params.conf:.2f} | iou {self.params.iou:.2f} | '
            f'model {model_name} | agnostic {"on" if self.params.agnostic else "off"} | '
            f'contain {self.params.contain_thresh:.2f} {"on" if self.params.contain else "off"} | '
            f'{mode_tag} | dets {num}'
            f'{cls_part}'
        )
        self.video_widget.show_hud(msg)

    def _show_keybinds(self) -> None:
        text = (
            'Navigation:\n'
            '  Left/Right: previous/next frame\n'
            '  Ctrl+Left/Right: -/+ 30 frames\n'
            '  Shift+Left/Right: -/+ 5 seconds\n'
            '\n'
            'Detection thresholds:\n'
            '  C / V: decrease / increase confidence\n'
            '  I / O: decrease / increase IoU\n'
            '\n'
            'NMS & filtering:\n'
            '  A: toggle class-agnostic NMS\n'
            '  [ / ]: cycle single kept class (if model has names)\n'
            '  \\: clear class filter\n'
            '\n'
            'Containment suppression (single-class dup reduction):\n'
            '  K: toggle containment suppression\n'
            '  N / M: decrease / increase containment threshold\n'
            '\n'
            'Model switching:\n'
            '  , / . : previous / next model in list\n'
            '\n'
            'Pre-NMS toggle:\n'
            '  P: toggle pre-NMS\n'
        )
        QMessageBox.information(self, 'Keybinds', text)

    # --------------- Key handling --------------- #
    def keyPressEvent(self, event):  # type: ignore[override]
        key = event.key()
        mods = event.modifiers()

        # Frame navigation
        if key == Qt.Key.Key_Left and not (
            mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
        ):
            self._set_frame_index(self.state.current_frame - 1)
            return
        if key == Qt.Key.Key_Right and not (
            mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
        ):
            self._set_frame_index(self.state.current_frame + 1)
            return
        if (mods & Qt.KeyboardModifier.ShiftModifier) and key == Qt.Key.Key_Left:
            step = int(max(1, round(self.video.fps * 5)))
            self._set_frame_index(self.state.current_frame - step)
            return
        if (mods & Qt.KeyboardModifier.ShiftModifier) and key == Qt.Key.Key_Right:
            step = int(max(1, round(self.video.fps * 5)))
            self._set_frame_index(self.state.current_frame + step)
            return
        if (mods & Qt.KeyboardModifier.ControlModifier) and key == Qt.Key.Key_Left:
            self._set_frame_index(self.state.current_frame - 30)
            return
        if (mods & Qt.KeyboardModifier.ControlModifier) and key == Qt.Key.Key_Right:
            self._set_frame_index(self.state.current_frame + 30)
            return

        # Threshold tuning
        if key == Qt.Key.Key_C:  # decrease conf
            self.params.conf = float(np.clip(self.params.conf - self.params.step, 0.0, 1.0))
            self._run_and_update()
            return
        if key == Qt.Key.Key_V:  # increase conf
            self.params.conf = float(np.clip(self.params.conf + self.params.step, 0.0, 1.0))
            self._run_and_update()
            return
        if key == Qt.Key.Key_I:  # decrease iou
            self.params.iou = float(np.clip(self.params.iou - self.params.step, 0.0, 1.0))
            self._run_and_update()
            return
        if key == Qt.Key.Key_O:  # increase iou
            self.params.iou = float(np.clip(self.params.iou + self.params.step, 0.0, 1.0))
            self._run_and_update()
            return

        # Toggle class-agnostic NMS
        if key == Qt.Key.Key_A:
            self.params.agnostic = not self.params.agnostic
            self._run_and_update()
            return

        # Class filtering: [ / ] to cycle a single kept class, \\ to clear
        if key == Qt.Key.Key_BracketLeft:
            class_count = len(self.class_names) if isinstance(self.class_names, dict) else 0
            if class_count > 0:
                cur = (self.params.classes[0] if self.params.classes else 0) - 1
                if cur < 0:
                    cur = class_count - 1
                self.params.classes = [int(cur)]
                self._run_and_update()
                return
        if key == Qt.Key.Key_BracketRight:
            class_count = len(self.class_names) if isinstance(self.class_names, dict) else 0
            if class_count > 0:
                cur = (self.params.classes[0] if self.params.classes else -1) + 1
                if cur >= class_count:
                    cur = 0
                self.params.classes = [int(cur)]
                self._run_and_update()
                return
        if key == Qt.Key.Key_Backslash:
            self.params.classes = None
            self._run_and_update()
            return

        # Containment suppression toggle and threshold (K/N/M)
        if key == Qt.Key.Key_K:
            self.params.contain = not self.params.contain
            self._run_and_update()
            return
        if key == Qt.Key.Key_N:
            self.params.contain_thresh = float(np.clip(self.params.contain_thresh - 0.05, 0.0, 1.0))
            if self.params.contain:
                self._run_and_update()
            else:
                self._update_hud(num=len(self.state.loaded_tracks))
            return
        if key == Qt.Key.Key_M:
            self.params.contain_thresh = float(np.clip(self.params.contain_thresh + 0.05, 0.0, 1.0))
            if self.params.contain:
                self._run_and_update()
            else:
                self._update_hud(num=len(self.state.loaded_tracks))
            return

        # Toggle pre-NMS/raw rendering (P)
        if key == Qt.Key.Key_P:
            self.params.pre_nms = not self.params.pre_nms
            self._run_and_update()
            return

        # Model switching: , previous | . next
        if key == Qt.Key.Key_Comma and self.models:
            self.model_idx = (self.model_idx - 1) % len(self.models)
            self.detector = _SingleFrameDetector(self.models[self.model_idx])
            try:
                self.class_names = self.detector.model.names  # type: ignore[attr-defined]
            except Exception:
                self.class_names = {}
            self._run_and_update()
            return
        if key == Qt.Key.Key_Period and self.models:
            self.model_idx = (self.model_idx + 1) % len(self.models)
            self.detector = _SingleFrameDetector(self.models[self.model_idx])
            try:
                self.class_names = self.detector.model.names  # type: ignore[attr-defined]
            except Exception:
                self.class_names = {}
            self._run_and_update()
            return

        super().keyPressEvent(event)


def _discover_models(models_dir: Path) -> list[Path]:
    if not models_dir.exists() or not models_dir.is_dir():
        return []
    return sorted([p for p in models_dir.glob('*.pt') if p.is_file()])


def main() -> None:
    parser = argparse.ArgumentParser(description='Tune detection thresholds on a chosen frame.')
    parser.add_argument('video', type=str, help='Path to input video')
    parser.add_argument('--conf', type=float, default=float(DETECTOR_CONFIDENCE_THRESHOLD), help='Initial confidence')
    parser.add_argument('--iou', type=float, default=float(DETECTOR_IOU_THRESHOLD), help='Initial IoU')
    parser.add_argument('--step', type=float, default=0.02, help='Adjustment step for conf/iou')
    parser.add_argument('--agnostic', action='store_true', help='Use class-agnostic NMS')
    parser.add_argument('--classes', type=str, default='', help='Optional comma-separated class ids to keep')
    parser.add_argument('--contain', action='store_true', help='Enable containment-based suppression')
    parser.add_argument(
        '--contain-thresh', type=float, default=0.80, help='Containment IoA threshold (intersection/min-area)'
    )
    parser.add_argument('--model', type=str, default='', help='Path to a YOLO .pt model to start with')
    parser.add_argument('--models-dir', type=str, default='', help='Directory with YOLO .pt models to cycle through')
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f'Video not found: {video_path}')
        return

    classes = [int(x) for x in args.classes.split(',') if x.strip().isdigit()] if args.classes else None
    params = _Params(
        conf=float(args.conf),
        iou=float(args.iou),
        step=float(args.step),
        agnostic=bool(args.agnostic),
        classes=classes,
        contain=bool(args.contain),
        contain_thresh=float(args.contain_thresh),
        pre_nms=False,
    )

    app = QApplication(sys.argv)
    start_model = Path(args.model).resolve() if args.model else YOLO_MODEL_PATH.resolve()
    models_root = Path(args.models_dir).resolve() if args.models_dir else YOLO_MODEL_PATH.parent.resolve()
    model_list = _discover_models(models_root)
    if start_model.exists() and start_model.suffix.lower() == '.pt' and start_model not in model_list:
        model_list = [start_model] + model_list
    win = DetectionTunerWindow(video_path, params, start_model, model_list)
    win.show()
    app.exec()


if __name__ == '__main__':
    main()
