from __future__ import annotations

import sys
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Make project importable when run as a script
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QToolBar, QMessageBox
from PySide6.QtGui import QAction

from server.inference.src.player.core.player_state import PlayerState, VideoProperties, TrackLite, DetectionLite
from server.inference.src.player.core.video_manager import VideoManager
from server.inference.src.player.ui.video_widget import VideoWidget
from server.inference.src.util.video_io import get_video_properties, VideoInfo
from server.inference.src.tracking.detector import SurferDetector
from server.inference.src.tracking.preprocessing.preprocessor import Preprocessor
from server.inference.src.common_types import Detection, Track
from server.inference.src.settings import (
    YOLO_MODEL_PATH,
    REID_MODEL_PATH,
    GREEDY_PREPROCESSOR_MIN_IOU,
    GREEDY_PREPROCESSOR_MIN_COSINE_SIMILARITY,
    GREEDY_PREPROCESSOR_MAX_FRAME_DISTANCE,
    GREEDY_PREPROCESSOR_MIN_IOU_MATCHES_SINGLE_TRACK,
)


def _detections_to_initial_tracks(dets: list[Detection]) -> list[Track]:
    return [Track(track_id=i + 1, sorted_detections=[d]) for i, d in enumerate(dets)]


def _to_tracklites(tracks: list[Track], video_props: VideoInfo) -> list[TrackLite]:
    out: list[TrackLite] = []
    for t in tracks:
        out.append(
            TrackLite(
                track_id=int(t.track_id),
                start_frame=int(t.start_frame),
                end_frame=int(t.end_frame),
                start_time=float(t.start_frame / max(1.0, video_props.fps)),
                duration=float(t.duration_frames / max(1.0, video_props.fps)),
                detection_count=len(t.sorted_detections),
                detections=[
                    DetectionLite(
                        frame_idx=int(d.frame_idx),
                        bbox=[int(d.bbox.x1), int(d.bbox.y1), int(d.bbox.x2), int(d.bbox.y2)],
                        confidence=float(d.confidence),
                    )
                    for d in t.sorted_detections
                ],
            )
        )
    return out


@dataclass
class GreedyParams:
    min_iou: float
    min_cos: float
    max_gap: int
    min_iou_match: float
    step_f: float = 0.02


class GreedyTunerWindow(QMainWindow):
    def __init__(self, video_path: Path, params: GreedyParams):
        super().__init__()
        self.setWindowTitle('Greedy Preprocessor Tuner')

        self.params = params
        self.video = VideoManager(video_path)
        self.video_props = get_video_properties(video_path)

        # Load/cached detections once
        self.detector = SurferDetector(yolo_model_path=YOLO_MODEL_PATH, reid_model_path=REID_MODEL_PATH)
        self.detections: list[Detection] = self.detector.run_object_detection_on_video(video_path)
        self.initial_tracks: list[Track] = _detections_to_initial_tracks(self.detections)

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

        # Toolbar with keybinds
        tb = QToolBar('Help', self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        action_help = QAction('Keybinds', self)
        action_help.setStatusTip('Show all keyboard shortcuts')
        action_help.triggered.connect(self._show_keybinds)
        tb.addAction(action_help)

        # Show first frame
        self.state.current_frame = 0
        self.video.seek_frame(0)
        _, frame = self.video.read_frame()
        if frame is not None:
            self.video_widget.set_frame(frame)

        # First compute and render
        self._recompute_and_update()

        # Small timer for HUD fading
        self._timer = QTimer(self)
        self._timer.timeout.connect(lambda: None)
        self._timer.start(100)

    # ---------------------- Core logic ---------------------- #
    def _recompute_and_update(self) -> None:
        pre = Preprocessor(
            greedy_min_iou=self.params.min_iou,
            greedy_min_cosine_similarity=self.params.min_cos,
            greedy_max_frame_distance=self.params.max_gap,
            greedy_min_iou_matches_single_track=self.params.min_iou_match,
        )
        tracks = pre.track(list(self.initial_tracks), self.video_props)
        self.state.loaded_tracks = _to_tracklites(tracks, self.video_props)
        self.state.visible_tracks = self.state._extract_visible_tracks()
        self.state.detections_by_frame = self.state._rebuild_detection_index()
        self._update_hud(num_tracks=len(tracks), num_dets=len(self.detections))
        self.video_widget.update()

    def _update_hud(self, num_tracks: int, num_dets: int) -> None:
        msg = (
            f'Greedy: IOU {self.params.min_iou:.2f} | COS {self.params.min_cos:.2f} | '
            f'maxGap {self.params.max_gap} | IOU_match {self.params.min_iou_match:.2f} | '
            f'tracks {num_tracks} | dets {num_dets} | frame {self.state.current_frame + 1}/{self.video.total_frames}'
        )
        self.video_widget.show_hud(msg)
        QTimer.singleShot(1000, self.video_widget.clear_hud)

    def _show_keybinds(self) -> None:
        text = (
            'Navigation:\n'
            '  Left/Right: previous/next frame\n'
            '  Ctrl+Left/Right: -/+ 30 frames\n'
            '  Shift+Left/Right: -/+ 5 seconds\n'
            '\n'
            'Greedy params:\n'
            '  1 / 2: decrease / increase min IOU\n'
            '  3 / 4: decrease / increase min cosine similarity\n'
            '  5 / 6: decrease / increase max frame distance\n'
            '  7 / 8: decrease / increase min IOU match per det\n'
        )
        QMessageBox.information(self, 'Keybinds', text)

    # ---------------------- Controls ---------------------- #
    def keyPressEvent(self, event):  # type: ignore[override]
        key = event.key()
        mods = event.modifiers()

        # Frame navigation
        if key == Qt.Key.Key_Left and not (
            mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
        ):
            self._seek_frame(self.state.current_frame - 1)
            return
        if key == Qt.Key.Key_Right and not (
            mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
        ):
            self._seek_frame(self.state.current_frame + 1)
            return
        if (mods & Qt.KeyboardModifier.ShiftModifier) and key == Qt.Key.Key_Left:
            step = int(max(1, round(self.video.fps * 5)))
            self._seek_frame(self.state.current_frame - step)
            return
        if (mods & Qt.KeyboardModifier.ShiftModifier) and key == Qt.Key.Key_Right:
            step = int(max(1, round(self.video.fps * 5)))
            self._seek_frame(self.state.current_frame + step)
            return
        if (mods & Qt.KeyboardModifier.ControlModifier) and key == Qt.Key.Key_Left:
            self._seek_frame(self.state.current_frame - 30)
            return
        if (mods & Qt.KeyboardModifier.ControlModifier) and key == Qt.Key.Key_Right:
            self._seek_frame(self.state.current_frame + 30)
            return

        # Params
        if key == Qt.Key.Key_1:
            self.params.min_iou = float(np.clip(self.params.min_iou - self.params.step_f, 0.0, 1.0))
            self._recompute_and_update()
            return
        if key == Qt.Key.Key_2:
            self.params.min_iou = float(np.clip(self.params.min_iou + self.params.step_f, 0.0, 1.0))
            self._recompute_and_update()
            return
        if key == Qt.Key.Key_3:
            self.params.min_cos = float(np.clip(self.params.min_cos - self.params.step_f, 0.0, 1.0))
            self._recompute_and_update()
            return
        if key == Qt.Key.Key_4:
            self.params.min_cos = float(np.clip(self.params.min_cos + self.params.step_f, 0.0, 1.0))
            self._recompute_and_update()
            return
        if key == Qt.Key.Key_5:
            self.params.max_gap = int(max(0, self.params.max_gap - 1))
            self._recompute_and_update()
            return
        if key == Qt.Key.Key_6:
            self.params.max_gap = int(self.params.max_gap + 1)
            self._recompute_and_update()
            return
        if key == Qt.Key.Key_7:
            self.params.min_iou_match = float(np.clip(self.params.min_iou_match - self.params.step_f, 0.0, 1.0))
            self._recompute_and_update()
            return
        if key == Qt.Key.Key_8:
            self.params.min_iou_match = float(np.clip(self.params.min_iou_match + self.params.step_f, 0.0, 1.0))
            self._recompute_and_update()
            return

        super().keyPressEvent(event)

    def _seek_frame(self, frame: int) -> None:
        frame = max(0, min(frame, self.video.total_frames - 1))
        self.state.current_frame = frame
        self.video.seek_frame(frame)
        _, frame_img = self.video.read_frame()
        if frame_img is not None:
            self.video_widget.set_frame(frame_img)
        self.video_widget.update()


def main() -> None:
    parser = argparse.ArgumentParser(description='Tune Greedy preprocessor hyperparameters with live track overlay.')
    parser.add_argument('video', type=str, help='Path to input video')
    parser.add_argument('--iou', type=float, default=float(GREEDY_PREPROCESSOR_MIN_IOU), help='min IOU threshold')
    parser.add_argument(
        '--cos', type=float, default=float(GREEDY_PREPROCESSOR_MIN_COSINE_SIMILARITY), help='min cosine similarity'
    )
    parser.add_argument(
        '--gap', type=int, default=int(GREEDY_PREPROCESSOR_MAX_FRAME_DISTANCE), help='max frame distance'
    )
    parser.add_argument(
        '--iou-match',
        type=float,
        default=float(GREEDY_PREPROCESSOR_MIN_IOU_MATCHES_SINGLE_TRACK),
        help='min IOU to consider same track',
    )
    parser.add_argument('--step', type=float, default=0.02, help='float step for thresholds')
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f'Video not found: {video_path}')
        return

    params = GreedyParams(
        min_iou=float(args.iou),
        min_cos=float(args.cos),
        max_gap=int(args.gap),
        min_iou_match=float(args.iou_match),
        step_f=float(args.step),
    )

    app = QApplication(sys.argv)
    win = GreedyTunerWindow(video_path, params)
    win.show()
    app.exec()


if __name__ == '__main__':
    main()
