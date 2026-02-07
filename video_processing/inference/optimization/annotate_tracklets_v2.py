from __future__ import annotations

import sys
import glob
import pickle
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Make project importable when run as a script
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from video_processing.inference.src.util.video_io import get_video_properties, VideoInfo
from video_processing.inference.src.tracking.detector import SurferDetector
from video_processing.inference.src.tracking.preprocessing.preprocessor import TrackPreProcessor
from video_processing.inference.src.tracking.track_processing import TrackRTSSmoothing
from video_processing.inference.src.common_types import Detection, Track
from video_processing.inference.src.player.core.player_state import Metadata, VideoProperties, TrackLite, DetectionLite
from video_processing.inference.src.settings import YOLO_MODEL_PATH
from video_processing.inference.src.player.core.video_manager import VideoManager
from video_processing.inference.src.tracking.renderable_tracks import prepare_renderable_tracks
from video_processing.inference.src.visualization.stabilize import Transform, compute_stabilization_transforms_gmc
from video_processing.inference.src.tracking.ilp_tracker import ILPTracker
from video_processing.inference.optimization.optimization_util import AssignmentKey, build_assignment_from_tracks

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QWidget,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
)


def _to_qimage(frame) -> QImage:
    if frame is None:
        return QImage()
    # QImage(memoryview) requires C-contiguous buffers.
    if isinstance(frame, np.ndarray) and not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    h, w = frame.shape[:2]
    if len(frame.shape) == 3:
        ch = frame.shape[2]
        bytes_per_line = ch * w
        return QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888).copy()
    return QImage(frame.data, w, h, w, QImage.Format.Format_Grayscale8).copy()


def _detections_to_initial_tracks(detections: list[Detection]) -> list[Track]:
    return [Track(track_id=i + 1, sorted_detections=[det]) for i, det in enumerate(detections)]


def _build_metadata(
    tracks: list[Track],
    input_path: Path,
    video_props: VideoInfo,
    *,
    raw_motion_transforms: list[Transform],
) -> Metadata:
    render_tracks = prepare_renderable_tracks(
        tracks,
        video_width=video_props.width,
        video_height=video_props.height,
        raw_motion_transforms=raw_motion_transforms,
    )
    return Metadata(
        input_video_path=input_path.absolute().as_posix(),
        video_properties=VideoProperties(
            fps=video_props.fps,
            width=video_props.width,
            height=video_props.height,
            total_frames=video_props.approximate_total_frames,
        ),
        tracks=[
            TrackLite(
                track_id=t.track_id,
                start_frame=t.start_frame,
                end_frame=t.end_frame,
                start_time=t.start_frame / max(1, video_props.fps),
                duration=t.duration_frames / max(1, video_props.fps),
                detection_count=len(t.sorted_detections),
                detections=[
                    DetectionLite(
                        frame_idx=det.frame_idx,
                        bbox=[int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2)],
                        confidence=float(det.confidence),
                        interpolated=det.interpolated,
                        boom=[float(det.boom.point.x), float(det.boom.point.y), float(det.boom.conf)],
                        mast_tip=[float(det.mast_tip.point.x), float(det.mast_tip.point.y), float(det.mast_tip.conf)],
                        anchor=[float(det.anchor.x), float(det.anchor.y)],
                        scale=float(det.scale),
                    )
                    for det in rt.sorted_detections
                ],
            )
            for t, rt in zip(tracks, render_tracks)
        ],
    )


def _save_golden_metadata(
    tracks: list[Track],
    input_path: Path,
    output_dir: Path,
    video_props: VideoInfo,
    *,
    raw_motion_transforms: list[Transform],
    filename: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f'{input_path.stem}.golden.tracks.pkl'
    out_path = output_dir / filename
    metadata = _build_metadata(tracks, input_path, video_props, raw_motion_transforms=raw_motion_transforms)
    with open(out_path, 'wb') as f:
        pickle.dump(metadata, f)
    return out_path


class ClickableCrop(QLabel):
    def __init__(self, pre_id: int, on_left: Callable[[int], None], on_right: Callable[[int], None]):
        super().__init__()
        self.pre_id = int(pre_id)
        self._on_left = on_left
        self._on_right = on_right
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_left(int(self.pre_id))
            return
        if event.button() == Qt.MouseButton.RightButton:
            self._on_right(int(self.pre_id))
            return
        super().mousePressEvent(event)


class CropGridWindow(QMainWindow):
    def __init__(
        self,
        *,
        title: str,
        on_crop_left: Callable[[int], None],
        on_crop_right: Optional[Callable[[int], None]] = None,
        on_key: Callable[[Qt.Key], None],
    ) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self._on_crop_left = on_crop_left
        self._on_crop_right = on_crop_right
        self._on_key = on_key

        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.status = QLabel('', root)
        self.status.setWordWrap(True)
        layout.addWidget(self.status, stretch=0)

        self.scroll = QScrollArea(root)
        self.scroll.setWidgetResizable(True)
        self.grid_container = QWidget(self.scroll)
        self.grid = QGridLayout(self.grid_container)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(8)
        self.scroll.setWidget(self.grid_container)
        layout.addWidget(self.scroll, stretch=1)

        self.setCentralWidget(root)

        # Shortcuts (these should work regardless of which child widget has focus).
        for key, seq in (
            (Qt.Key.Key_Up, QKeySequence('Up')),
            (Qt.Key.Key_Down, QKeySequence('Down')),
            (Qt.Key.Key_Escape, QKeySequence('Esc')),
        ):
            sc = QShortcut(seq, self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda k=key: self._on_key(k))  # type: ignore[arg-type]

    def set_status(self, text: str) -> None:
        self.status.setText(str(text))

    def set_items(self, items: list[tuple[int, QImage, str]]) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        cols = 4
        for idx, (pre_id, img, caption) in enumerate(items):
            cell = QWidget(self.grid_container)
            v = QVBoxLayout(cell)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(2)

            def _noop(_pid: int) -> None:
                return

            crop = ClickableCrop(
                int(pre_id),
                self._on_crop_left,
                self._on_crop_right if self._on_crop_right is not None else _noop,
            )
            crop.setPixmap(QPixmap.fromImage(img))
            crop.setToolTip(f'pre {int(pre_id)}')
            v.addWidget(crop)

            lbl = QLabel(str(caption), cell)
            lbl.setWordWrap(True)
            v.addWidget(lbl)

            r = idx // cols
            c = idx % cols
            self.grid.addWidget(cell, r, c)

    def keyPressEvent(self, event):  # type: ignore[override]
        # Keep default key handling for typing/navigation within the window.
        super().keyPressEvent(event)


class TrackletAnnotatorV2Controller:
    """
    v2 UX (two always-visible crop windows):
      - Unassigned window: click a crop to assign it to the current id (id==0 => discard)
      - Current-id window: click a crop to unassign it (moves back to Unassigned)
      - Up/Down changes current id (including 0)
      - Esc saves (requires no unassigned)
    """

    def __init__(
        self,
        *,
        input_video: Path,
        video_props: VideoInfo,
        pre_tracks: list[Track],
        raw_motion_transforms: list[Transform],
        initial_assignments: dict[int, int] | None = None,
        output_dir: Path,
        on_finished: Optional[Callable[[], None]] = None,
    ):
        self.input_video = input_video
        self.video_props = video_props
        self.pre_tracks = pre_tracks
        self.raw_motion_transforms = raw_motion_transforms
        self.output_dir = output_dir
        self.on_finished = on_finished

        self._pre_by_id: dict[int, Track] = {int(t.track_id): t for t in pre_tracks}
        self._frames_by_pre: dict[int, set[int]] = {
            int(t.track_id): {int(d.frame_idx) for d in t.sorted_detections} for t in pre_tracks
        }

        self.assignments: dict[int, int] = dict(initial_assignments or {})  # pre_id -> golden_id (0 discard, >0 track)

        # Crop cache: pre_id -> QImage
        self._start_crop_cache: dict[int, QImage] = {}
        self._crop_video = VideoManager(input_video)
        self._detail_video = VideoManager(input_video)
        self._tracklet_frames_window: CropGridWindow | None = None

        max_existing = max((int(v) for v in self.assignments.values() if int(v) > 0), default=1)
        self.current_id: int = int(max_existing)
        self._last_message: str | None = None

        self.current_window = CropGridWindow(
            title='Current Track',
            on_crop_left=self._unassign_pre_id,
            on_crop_right=self._show_tracklet_frames,
            on_key=self._on_key,
        )
        self.unassigned_window = CropGridWindow(
            title='Unassigned',
            on_crop_left=self._assign_pre_id_to_current,
            on_crop_right=self._show_tracklet_frames,
            on_key=self._on_key,
        )

        def _controls_bar(parent: QWidget, *, include_save: bool) -> QWidget:
            bar = QWidget(parent)
            lay = QHBoxLayout(bar)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(8)
            btn_down = QPushButton('ID - (Down)', bar)
            btn_down.clicked.connect(lambda: self._on_key(Qt.Key.Key_Down))
            btn_up = QPushButton('ID + (Up)', bar)
            btn_up.clicked.connect(lambda: self._on_key(Qt.Key.Key_Up))
            lay.addWidget(btn_down)
            lay.addWidget(btn_up)
            if include_save:
                btn_save = QPushButton('Save (Esc)', bar)
                btn_save.clicked.connect(self._finalize_and_save)
                lay.addWidget(btn_save)
            lay.addStretch(1)
            return bar

        current_bar = _controls_bar(self.current_window, include_save=True)
        unassigned_bar = _controls_bar(self.unassigned_window, include_save=False)

        # Insert right below the status label.
        self.current_window.centralWidget().layout().insertWidget(1, current_bar)  # type: ignore[union-attr]
        self.unassigned_window.centralWidget().layout().insertWidget(1, unassigned_bar)  # type: ignore[union-attr]

        self.current_window.show()
        self.unassigned_window.show()

        self._refresh_windows()

    def _unassigned_pre_ids(self) -> list[int]:
        return sorted([int(t.track_id) for t in self.pre_tracks if int(t.track_id) not in self.assignments])

    # --------------------------- crops/grid --------------------------- #
    def _crop_for_pre_id(self, pre_id: int) -> Optional[QImage]:
        pre_id = int(pre_id)
        cached = self._start_crop_cache.get(pre_id)
        if cached is not None and not cached.isNull():
            return cached

        t = self._pre_by_id.get(pre_id)
        if t is None or not t.sorted_detections:
            return None
        det = t.sorted_detections[0]
        frame_idx = int(det.frame_idx)
        bbox = det.bbox

        self._crop_video.seek_frame(frame_idx)
        _, frame = self._crop_video.read_frame()
        if frame is None:
            return None

        x1 = max(0, min(int(bbox.x1), int(self.video_props.width)))
        y1 = max(0, min(int(bbox.y1), int(self.video_props.height)))
        x2 = max(0, min(int(bbox.x2), int(self.video_props.width)))
        y2 = max(0, min(int(bbox.y2), int(self.video_props.height)))
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None

        qimg = _to_qimage(crop)
        if qimg.isNull():
            return None
        qimg = self._scale_to_min_side(qimg, target_px=100)
        self._start_crop_cache[pre_id] = qimg
        return qimg

    def _scale_to_min_side(self, img: QImage, *, target_px: int) -> QImage:
        w = int(img.width())
        h = int(img.height())
        if w <= 0 or h <= 0:
            return img
        target_px = max(1, int(target_px))
        if min(w, h) == target_px:
            return img
        if w <= h:
            return img.scaledToWidth(target_px, Qt.TransformationMode.SmoothTransformation)
        return img.scaledToHeight(target_px, Qt.TransformationMode.SmoothTransformation)

    # --------------------------- interactions --------------------------- #
    def _assign_pre_id_to_current(self, pre_id: int) -> None:
        pre_id = int(pre_id)
        gid = int(self.current_id)
        if gid < 0:
            return
        if gid > 0:
            ok, reason = self._can_assign_without_overlap(pre_id, gid)
            if not ok:
                self._set_message(f'Cannot assign pre {pre_id} -> ID {gid}: {reason}')
                self._refresh_windows()
                return
        self.assignments[int(pre_id)] = int(gid)
        self._set_message(f'Assigned pre {pre_id} -> ID {gid}')
        self._refresh_windows()

    def _unassign_pre_id(self, pre_id: int) -> None:
        pre_id = int(pre_id)
        prev = self.assignments.pop(int(pre_id), None)
        if prev is None:
            self._set_message(f'pre {pre_id} is already unassigned')
        else:
            self._set_message(f'Unassigned pre {pre_id} from ID {int(prev)}')
        self._refresh_windows()

    def _can_assign_without_overlap(self, pre_id: int, golden_id: int) -> tuple[bool, str]:
        if int(golden_id) == 0:
            return True, ''
        candidate = self._frames_by_pre.get(int(pre_id))
        if not candidate:
            return False, 'tracklet has no frames'
        for assigned_pre_id, gid in self.assignments.items():
            if int(gid) != int(golden_id):
                continue
            used = self._frames_by_pre.get(int(assigned_pre_id), set())
            if candidate.isdisjoint(used):
                continue
            overlap = candidate.intersection(used)
            first = min(overlap) if overlap else None
            if first is None:
                return False, f'overlap with pre {int(assigned_pre_id)}'
            return (
                False,
                f'overlap with pre {int(assigned_pre_id)} at frame {int(first)} (overlap_frames={len(overlap)})',
            )
        return True, ''

    # --------------------------- saving --------------------------- #
    def _finalize_and_save(self) -> None:
        unassigned = self._unassigned_pre_ids()
        if unassigned:
            self._set_message(f'Cannot save: {len(unassigned)} unassigned remain')
            self._refresh_windows()
            return

        group_to_dets: dict[int, list[Detection]] = {}
        for pre_id, gid in self.assignments.items():
            t = self._pre_by_id.get(int(pre_id))
            if t is None:
                continue
            group_to_dets.setdefault(int(gid), []).extend(t.sorted_detections)

        merged: list[Track] = []
        for gid, dets in group_to_dets.items():
            dets.sort(key=lambda d: d.frame_idx)
            merged.append(Track(track_id=int(gid), sorted_detections=dets))

        # Rendering metadata expects dense per-frame tracks; densify before saving.
        merged = TrackRTSSmoothing().track(merged, self.video_props, self.raw_motion_transforms)

        golden_path = _save_golden_metadata(
            merged,
            self.input_video,
            self.output_dir,
            self.video_props,
            raw_motion_transforms=self.raw_motion_transforms,
        )
        self._set_message(f'Saved: {golden_path}')
        self._refresh_windows()
        try:
            if self.on_finished is not None:
                self.on_finished()
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._crop_video.release()
        except Exception:
            pass
        try:
            self._detail_video.release()
        except Exception:
            pass
        try:
            self.current_window.close()
        except Exception:
            pass
        try:
            self.unassigned_window.close()
        except Exception:
            pass
        try:
            if self._tracklet_frames_window is not None:
                self._tracklet_frames_window.close()
        except Exception:
            pass

    def _on_key(self, key: Qt.Key) -> None:
        if key == Qt.Key.Key_Escape:
            self._finalize_and_save()
            return
        if key == Qt.Key.Key_Up:
            self.current_id = int(self.current_id) + 1
            self._set_message(f'Current ID: {int(self.current_id)}')
            self._refresh_windows()
            return
        if key == Qt.Key.Key_Down:
            self.current_id = max(0, int(self.current_id) - 1)
            self._set_message(f'Current ID: {int(self.current_id)}')
            self._refresh_windows()
            return

    def _set_message(self, msg: str | None) -> None:
        self._last_message = str(msg) if msg else None

    def _refresh_windows(self) -> None:
        gid = int(self.current_id)

        # current window items: pre_ids with assignment == gid
        current_pre_ids = sorted([int(pid) for pid, g in self.assignments.items() if int(g) == gid])
        current_items: list[tuple[int, QImage, str]] = []
        for pre_id in current_pre_ids:
            img = self._crop_for_pre_id(pre_id)
            if img is None or img.isNull():
                continue
            t = self._pre_by_id.get(int(pre_id))
            if t and t.sorted_detections:
                f0 = int(t.sorted_detections[0].frame_idx)
                span = int(t.sorted_detections[-1].frame_idx) - int(t.sorted_detections[0].frame_idx) + 1
                c = t.sorted_detections[0].bbox.center
                cx, cy = int(round(float(c.x))), int(round(float(c.y)))
            else:
                f0 = -1
                span = 0
                cx = cy = -1
            current_items.append((pre_id, img, f'f0={f0} | span={span} | c=({cx},{cy})'))

        unassigned_pre_ids = self._unassigned_pre_ids()
        unassigned_items: list[tuple[int, QImage, str]] = []
        for pre_id in unassigned_pre_ids:
            img = self._crop_for_pre_id(pre_id)
            if img is None or img.isNull():
                continue
            t = self._pre_by_id.get(int(pre_id))
            if t and t.sorted_detections:
                f0 = int(t.sorted_detections[0].frame_idx)
                span = int(t.sorted_detections[-1].frame_idx) - int(t.sorted_detections[0].frame_idx) + 1
                c = t.sorted_detections[0].bbox.center
                cx, cy = int(round(float(c.x))), int(round(float(c.y)))
            else:
                f0 = -1
                span = 0
                cx = cy = -1
            unassigned_items.append((pre_id, img, f'f0={f0} | span={span} | c=({cx},{cy})'))

        self.current_window.setWindowTitle(f'Current ID = {gid} (click to unassign)')
        self.current_window.set_items(current_items)
        self.unassigned_window.setWindowTitle('Unassigned (click to assign to current id)')
        self.unassigned_window.set_items(unassigned_items)

        unassigned_count = len(unassigned_pre_ids)
        base_current = (
            f'Current ID: {gid} | assigned: {len(current_pre_ids)} | unassigned: {unassigned_count} | Up/Down changes ID | Esc saves'
        )
        if self._last_message:
            base_current = f'{base_current}\n{self._last_message}'
        self.current_window.set_status(base_current)
        self.unassigned_window.set_status(
            f'Unassigned: {unassigned_count} | click assigns to ID={gid} (0=discard)'
        )

    def _show_tracklet_frames(self, pre_id: int) -> None:
        pre_id = int(pre_id)
        t = self._pre_by_id.get(pre_id)
        if t is None or not t.sorted_detections:
            self._set_message(f'Cannot open tracklet pre {pre_id}: no detections')
            self._refresh_windows()
            return

        items: list[tuple[int, QImage, str]] = []
        for det in t.sorted_detections:
            f = int(det.frame_idx)
            bbox = det.bbox
            self._detail_video.seek_frame(f)
            _, frame = self._detail_video.read_frame()
            if frame is None:
                continue
            x1 = max(0, min(int(bbox.x1), int(self.video_props.width)))
            y1 = max(0, min(int(bbox.y1), int(self.video_props.height)))
            x2 = max(0, min(int(bbox.x2), int(self.video_props.width)))
            y2 = max(0, min(int(bbox.y2), int(self.video_props.height)))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue
            qimg = _to_qimage(crop)
            if qimg.isNull():
                continue
            qimg = self._scale_to_min_side(qimg, target_px=100)
            c = bbox.center
            cx, cy = int(round(float(c.x))), int(round(float(c.y)))
            items.append((pre_id, qimg, f'f={f} | c=({cx},{cy})'))

        if self._tracklet_frames_window is None:
            self._tracklet_frames_window = CropGridWindow(
                title='Tracklet frames',
                on_crop_left=lambda _pid: None,
                on_key=self._on_key,
            )
        self._tracklet_frames_window.setWindowTitle(f'Tracklet pre {pre_id} (frames={len(items)})')
        self._tracklet_frames_window.set_status('Right-click any crop to open this view.')
        self._tracklet_frames_window.set_items(items)
        self._tracklet_frames_window.show()


class _BatchController:
    def __init__(self, video_paths: list[Path], output_dir: Path, *, no_ilp: bool):
        self.video_paths = video_paths
        self.output_dir = output_dir
        self.no_ilp = bool(no_ilp)
        self._app: QApplication | None = None
        self._idx: int = -1
        self._win: TrackletAnnotatorV2 | None = None

    def _next(self) -> None:
        self._idx += 1
        if self._idx >= len(self.video_paths):
            if self._app is not None:
                self._app.quit()
            return
        video = self.video_paths[self._idx]

        if (self.output_dir / f'{video.stem}.golden.tracks.pkl').exists():
            self._next()
            return

        video_props = get_video_properties(video)
        detector = SurferDetector(yolo_model_path=YOLO_MODEL_PATH)
        detections = detector.run_object_detection_on_video(video.as_posix())
        tracks: list[Track] = _detections_to_initial_tracks(detections)
        transforms = compute_stabilization_transforms_gmc(video.as_posix())
        tracks = TrackPreProcessor().track(tracks, video_props, transforms)

        initial_assignments: dict[int, int] = {}
        if not self.no_ilp:
            initial_assignments = _initial_assignments_from_ilp(tracks, video_props, transforms)

        self._win = TrackletAnnotatorV2Controller(
            input_video=video,
            video_props=video_props,
            pre_tracks=tracks,
            raw_motion_transforms=transforms,
            initial_assignments=initial_assignments,
            output_dir=self.output_dir,
            on_finished=self._next,
        )
        # windows are shown by the controller

    def run(self) -> None:
        self._app = QApplication(sys.argv)
        self._next()
        self._app.exec()


def main() -> None:
    parser = argparse.ArgumentParser(description='Crop/grid-only tracklet annotator (v2).')
    parser.add_argument('videos', type=str, nargs='+', help='Path(s) or glob pattern(s) to input video(s)')
    parser.add_argument('--output-dir', type=str, default='tmp', help='Directory to write golden metadata')
    parser.add_argument('--no-ilp', action='store_true', help='Do not initialize groups with ILPTracker')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    video_paths: list[Path] = []
    for pat in args.videos:
        expanded = [Path(p) for p in glob.glob(pat)]
        if not expanded:
            p = Path(pat)
            if p.exists():
                expanded = [p]
        video_paths.extend(expanded)
    video_paths = sorted({p.resolve() for p in video_paths if p.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv'}})
    if not video_paths:
        print('No input videos found for given patterns.')
        return

    controller = _BatchController(video_paths, output_dir, no_ilp=bool(args.no_ilp))
    controller.run()


def _initial_assignments_from_ilp(
    pre_tracks: list[Track],
    video_props: VideoInfo,
    transforms: list[Transform],
) -> dict[int, int]:
    """
    Run ILPTracker to create *combined tracks*, then map each pre-tracklet to an ILP track_id.

    Mapping is done via detection identity keys (frame + bbox ints). If a pre-tracklet maps cleanly to one ILP id,
    we initialize it; otherwise we leave it unassigned for manual resolution.
    """
    if not pre_tracks:
        return {}

    pred_tracks = ILPTracker(video_path=None).track(pre_tracks, video_props, transforms)
    pred_assign = build_assignment_from_tracks(pred_tracks)

    out: dict[int, int] = {}
    for t in pre_tracks:
        pre_id = int(t.track_id)
        ids: set[int] = set()
        for det in t.sorted_detections:
            k = AssignmentKey(
                frame_idx=int(det.frame_idx),
                x1=int(det.bbox.x1),
                y1=int(det.bbox.y1),
                x2=int(det.bbox.x2),
                y2=int(det.bbox.y2),
            )
            pred_id = pred_assign.get(k)
            if pred_id is not None:
                ids.add(int(pred_id))
        if len(ids) == 1:
            out[pre_id] = int(next(iter(ids)))

    return out


if __name__ == '__main__':
    main()
