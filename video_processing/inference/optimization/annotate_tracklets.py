from __future__ import annotations

import sys
import glob
import pickle
import argparse
from pathlib import Path
from typing import Callable, Optional


# Make project importable when run as a script
this_file = Path(__file__).resolve()
project_root = this_file.parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from video_processing.inference.src.util.video_io import get_video_properties, VideoInfo
from video_processing.inference.src.tracking.detector import SurferDetector
from video_processing.inference.src.tracking.preprocessing.preprocessor import TrackPreProcessor
from video_processing.inference.src.common_types import Detection, Track
from video_processing.inference.src.player.core.player_state import Metadata, VideoProperties, TrackLite, DetectionLite
from video_processing.inference.src.settings import YOLO_MODEL_PATH
from video_processing.inference.src.player.core.video_manager import VideoManager
from video_processing.inference.src.player.ui.video_widget import VideoWidget
from video_processing.inference.src.tracking.track_processing import prepare_renderable_tracks

from video_processing.inference.src.visualization.stabilize import compute_stabilization_transforms_gmc

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QMainWindow


"""USAGE:

"spacebar" : next golden id
"backspace" : undo last assignment
"escape" : finalize and save
"d" : mark selected tracklet as discard (golden id 0)
"left arrow" : previous unassigned tracklet start
"right arrow" : next unassigned tracklet start
"control + left arrow" : previous 30 frames
"control + right arrow" : next 30 frames
"shift + left arrow" : previous 5 frames
"shift + right arrow" : next 5 frames
"""

DISCARD_GOLDEN_ID = 0


def _detections_to_initial_tracks(detections: list[Detection]) -> list[Track]:
    return [Track(track_id=i + 1, sorted_detections=[det]) for i, det in enumerate(detections)]


def _build_metadata(tracks: list[Track], input_path: Path, video_props: VideoInfo) -> Metadata:
    render_tracks = prepare_renderable_tracks(tracks, video_width=video_props.width, video_height=video_props.height)
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
                        anchor=[int(det.anchor.x), int(det.anchor.y)],
                        scale=float(det.scale),
                    )
                    for det in rt.sorted_detections
                ],
            )
            for t, rt in zip(tracks, render_tracks)
        ],
    )


def _save_golden_metadata(
    tracks: list[Track], input_path: Path, output_dir: Path, video_props: VideoInfo, *, filename: str | None = None
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f'{input_path.stem}.golden.tracks.pkl'
    out_path = output_dir / filename
    metadata = _build_metadata(tracks, input_path, video_props)
    with open(out_path, 'wb') as f:
        pickle.dump(metadata, f)
    return out_path


def _to_tracklites(tracks: list[Track], video_props: VideoInfo) -> list[TrackLite]:
    render_tracks = prepare_renderable_tracks(tracks, video_width=video_props.width, video_height=video_props.height)
    out: list[TrackLite] = []
    for t, rt in zip(tracks, render_tracks):
        out.append(
            TrackLite(
                # Initialize with negative IDs so labels are hidden until assigned
                track_id=-int(t.track_id),
                start_frame=int(t.start_frame),
                end_frame=int(t.end_frame),
                start_time=float(t.start_frame / max(1, video_props.fps)),
                duration=float(t.duration_frames / max(1, video_props.fps)),
                detection_count=len(t.sorted_detections),
                detections=[
                    DetectionLite(
                        frame_idx=int(d.frame_idx),
                        bbox=[int(d.bbox.x1), int(d.bbox.y1), int(d.bbox.x2), int(d.bbox.y2)],
                        confidence=float(d.confidence),
                        interpolated=d.interpolated,
                        boom=[float(d.boom.point.x), float(d.boom.point.y), float(d.boom.conf)],
                        mast_tip=[float(d.mast_tip.point.x), float(d.mast_tip.point.y), float(d.mast_tip.conf)],
                        anchor=[int(d.anchor.x), int(d.anchor.y)],
                        scale=float(d.scale),
                    )
                    for d in rt.sorted_detections
                ],
            )
        )
    return out


class TrackAnnotatorWindow(QMainWindow):
    def __init__(
        self,
        input_video: Path,
        video_props: VideoInfo,
        preprocessed_tracks: list[Track],
        output_dir: Path,
        on_finished: Optional[Callable[[], None]] = None,
    ):
        super().__init__()
        self.setWindowTitle('Windsurf Tracklet Annotator')

        from video_processing.inference.src.player.core.player_state import PlayerState

        self.state = PlayerState()
        self.state.reset(
            input_video_path=input_video.as_posix(),
            video_properties=VideoProperties(
                fps=video_props.fps,
                width=video_props.width,
                height=video_props.height,
                total_frames=video_props.approximate_total_frames,
            ),
            loaded_tracks=_to_tracklites(preprocessed_tracks, video_props),
        )

        self.video = VideoManager(input_video)
        self.output_dir = output_dir
        self.input_video = input_video
        self.video_props = video_props
        self.pre_tracks = preprocessed_tracks
        self.on_finished = on_finished

        # Map from original preprocessed track id -> TrackLite object (stable reference)
        self.pre_id_to_tracklite: dict[int, TrackLite] = {}
        for tl in self.state.loaded_tracks:
            if int(tl.track_id) < 0:
                self.pre_id_to_tracklite[-int(tl.track_id)] = tl

        # golden assignments: preprocessed track id -> golden id
        self.assignments: dict[int, int] = {}
        self.history: list[tuple[int, int | None]] = []  # (pre_id, prev_golden_or_None)
        self.current_golden_id: int = 1
        self.selected_pre_id: int | None = None

        self.video_widget = VideoWidget(self.state, on_track_selected=self._on_track_clicked)
        self.setCentralWidget(self.video_widget)

        # Show first frame
        self.state.current_frame = 0
        self._on_seek(0)

        # Small timer for HUD fading consistency
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: None)
        self.timer.start(100)

        self._update_hud()

    # -------------------------- Interaction logic -------------------------- #
    def _on_seek(self, frame: int) -> None:
        self.state.current_frame = max(0, min(frame, self.video.total_frames - 1))
        self.video.seek_frame(self.state.current_frame)
        _, frame_img = self.video.read_frame()
        if frame_img is not None:
            self.video_widget.set_frame(frame_img)
        # keep overlays consistent
        self.video_widget.update()

    def _step(self, delta: int) -> None:
        idx, frame_img = self.video.advance_by(max(1, abs(delta)))
        if idx >= 0 and frame_img is not None:
            self.state.current_frame = idx
            self.video_widget.set_frame(frame_img)
            self.video_widget.update()

    def _on_track_clicked(self, track_id: int) -> None:
        # Only accept clicks on unassigned (negative-labeled) tracklets
        if int(track_id) > 0:
            self._update_hud(brief=f'Already assigned (golden {track_id})')
            return
        pre_id = int(-track_id)
        self.selected_pre_id = int(pre_id)

        # Check temporal overlap with existing assignments for this golden id
        if not self._can_assign_without_overlap(pre_id, self.current_golden_id):
            self._update_hud(brief='Overlap conflict: cannot assign to this golden id')
            return
        self._set_assignment(
            pre_id, self.current_golden_id, brief=f'Assigned pre {pre_id} -> golden {self.current_golden_id}'
        )

    def _set_assignment(self, pre_id: int, golden_id: int, *, brief: str | None = None) -> None:
        prev = self.assignments.get(int(pre_id))
        self.history.append((int(pre_id), prev))
        self.assignments[int(pre_id)] = int(golden_id)
        self._apply_display_id(int(pre_id), int(golden_id))
        self._update_hud(brief=brief)

    def _discard_selected(self) -> None:
        # Prefer hovered tracklet in overview (works even before clicking).
        hovered = self.video_widget.hovered_track_id
        if hovered is not None and int(hovered) < 0:
            pre_id = int(-hovered)
        else:
            pre_id = self.selected_pre_id if self.selected_pre_id is not None else None

        if pre_id is None:
            self._update_hud(brief='No tracklet selected/hovered')
            return

        self.selected_pre_id = int(pre_id)
        self._set_assignment(int(pre_id), int(DISCARD_GOLDEN_ID), brief=f'Discard pre {int(pre_id)} (golden 0)')

    def _undo(self) -> None:
        if not self.history:
            self._update_hud(brief='Nothing to undo')
            return
        pre_id, prev_golden = self.history.pop()
        if prev_golden is None:
            self.assignments.pop(pre_id, None)
            # revert display to negative id (hidden)
            self._apply_display_id(pre_id, -pre_id)
        else:
            self.assignments[pre_id] = prev_golden
            self._apply_display_id(pre_id, prev_golden)
        self._update_hud(brief='Undone last assignment')

    def _update_hud(self, brief: str | None = None) -> None:
        total = len(self.pre_tracks)
        assigned = len(self.assignments)
        base = f'Golden ID: {self.current_golden_id} | assigned {assigned}/{total}'
        text = f'{brief} | {base}' if brief else base
        self.video_widget.show_hud(text)

    def keyPressEvent(self, event):  # type: ignore[override]
        key = event.key()
        mods = event.modifiers()
        if key == Qt.Key.Key_Space:
            self.current_golden_id += 1
            self._update_hud(brief='Next golden id')
            total = len(self.pre_tracks)
            assigned = len(self.assignments)
            if assigned == total:
                self._finalize_and_save()
            else:
                # Seek to beginning for next assignment round
                unassigned = self._unassigned_tracks()
                earliest_unassigned_frame = min(unassigned, key=lambda x: x[0])[0]
                self._on_seek(earliest_unassigned_frame)
            return
        if key == Qt.Key.Key_Backspace:
            self._undo()
            return
        if key == Qt.Key.Key_D:
            self._discard_selected()
            return
        if key == Qt.Key.Key_Escape:
            self._finalize_and_save()
            return
        if key == Qt.Key.Key_Left and not (
            mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
        ):
            self._seek_to_unassigned(-1)
            return
        if key == Qt.Key.Key_Right and not (
            mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
        ):
            self._seek_to_unassigned(+1)
            return
        if (mods & Qt.KeyboardModifier.ControlModifier) and key == Qt.Key.Key_Left:
            self._on_seek(self.state.current_frame - 1)
            return
        if (mods & Qt.KeyboardModifier.ControlModifier) and key == Qt.Key.Key_Right:
            self._on_seek(self.state.current_frame + 1)
            return
        if (mods & Qt.KeyboardModifier.ShiftModifier) and key == Qt.Key.Key_Left:
            self._on_seek(
                self.state.current_frame - int(self.state.video_properties.fps * 5)
                if self.state.video_properties
                else 0
            )
            return
        if (mods & Qt.KeyboardModifier.ShiftModifier) and key == Qt.Key.Key_Right:
            self._on_seek(
                self.state.current_frame + int(self.state.video_properties.fps * 5)
                if self.state.video_properties
                else 0
            )
            return
        super().keyPressEvent(event)

    # --------------------------- Display management ------------------------- #
    def _apply_display_id(self, pre_id: int, new_id: int) -> None:
        # Update track id for this specific preprocessed track's TrackLite
        tl = self.pre_id_to_tracklite.get(int(pre_id))
        if tl is not None:
            tl.track_id = int(new_id)
        # Rebuild fast lookups
        self.state.visible_tracks = self.state._extract_visible_tracks()
        self.state.detections_by_frame = self.state._rebuild_detection_index()
        # Refresh overlay
        self.video_widget.update()

    def _can_assign_without_overlap(self, pre_id: int, golden_id: int) -> bool:
        if int(golden_id) == int(DISCARD_GOLDEN_ID):
            return True
        # Frames of the candidate preprocessed track
        frames_candidate: set[int] = set()
        for t in self.pre_tracks:
            if int(t.track_id) == int(pre_id):
                frames_candidate = {int(d.frame_idx) for d in t.sorted_detections}
                break
        if not frames_candidate:
            return False
        # Frames already used by tracks assigned to this golden id
        frames_used: set[int] = set()
        for assigned_pre_id, gid in self.assignments.items():
            if int(gid) != int(golden_id):
                continue
            for t in self.pre_tracks:
                if int(t.track_id) == int(assigned_pre_id):
                    frames_used.update(int(d.frame_idx) for d in t.sorted_detections)
                    break
        # No overlap allowed
        return frames_candidate.isdisjoint(frames_used)

    # ------------------------- Unassigned navigation ------------------------ #
    def _unassigned_tracks(self) -> list[tuple[int, int]]:
        return [
            (int(t.start_frame), int(t.track_id)) for t in self.pre_tracks if int(t.track_id) not in self.assignments
        ]

    def _seek_to_unassigned(self, direction: int) -> None:
        unassigned = self._unassigned_tracks()

        if not unassigned:
            self._update_hud(brief='All tracklets assigned')
            return

        curr = int(self.state.current_frame)
        if direction > 0:
            # Next with start > current
            candidates = [u for u in unassigned if u[0] > curr]
            if not candidates:
                self._update_hud(brief='No next unassigned')
                return
            target_frame, target_pre = min(candidates, key=lambda x: x[0])
        else:
            candidates = [u for u in unassigned if u[0] < curr]
            if not candidates:
                self._update_hud(brief='No previous unassigned')
                return
            target_frame, target_pre = max(candidates, key=lambda x: x[0])

        self._on_seek(int(target_frame))
        self._update_hud(brief=f'Go to unassigned pre {int(target_pre)} @ {int(target_frame)}')

    # ----------------------------- Finalization ----------------------------- #
    def _finalize_and_save(self) -> None:
        # Ensure all preprocessed tracks have an assignment
        unassigned = [int(t.track_id) for t in self.pre_tracks if int(t.track_id) not in self.assignments]
        if unassigned:
            msg = f'Cannot save: {len(unassigned)} unassigned tracklets remain. Assign all before saving.'
            print(msg)
            self._update_hud(brief='Unassigned tracklets remain. Assign all before saving.')
            return

        # Group detections by golden id
        group_to_dets: dict[int, list] = {}
        for t in self.pre_tracks:
            gid = self.assignments[int(t.track_id)]
            group_to_dets.setdefault(gid, []).extend(t.sorted_detections)

        merged: list[Track] = []
        for gid, dets in group_to_dets.items():
            dets.sort(key=lambda d: d.frame_idx)
            merged.append(Track(track_id=int(gid), sorted_detections=dets))

        golden_path = _save_golden_metadata(merged, self.input_video, self.output_dir, self.video_props)
        print(f'Saved golden metadata: {golden_path}')
        print(f'Saved player metadata: {self.output_dir / f"{self.input_video.stem}.tracks.pkl"}')
        try:
            if self.on_finished is not None:
                self.on_finished()
        finally:
            self.close()


class _BatchAnnotatorController:
    def __init__(self, video_paths: list[Path], output_dir: Path):
        self.video_paths = video_paths
        self.output_dir = output_dir
        self._app: QApplication | None = None
        self._idx: int = -1
        self._win: TrackAnnotatorWindow | None = None

    def _next(self) -> None:
        self._idx += 1
        if self._idx >= len(self.video_paths):
            if self._app is not None:
                self._app.quit()
            return
        video = self.video_paths[self._idx]

        # if already processed, skip
        if (self.output_dir / f'{video.stem}.golden.tracks.pkl').exists():
            self._next()
            return

        video_props = get_video_properties(video)
        detector = SurferDetector(yolo_model_path=YOLO_MODEL_PATH)
        detections = detector.run_object_detection_on_video(video.as_posix())
        tracks: list[Track] = _detections_to_initial_tracks(detections)
        transforms = compute_stabilization_transforms_gmc(video.as_posix())
        tracks = TrackPreProcessor().track(tracks, video_props, transforms)

        self._win = TrackAnnotatorWindow(video, video_props, tracks, self.output_dir, on_finished=self._next)
        self._win.show()

    def run(self) -> None:
        self._app = QApplication(sys.argv)
        self._next()
        self._app.exec()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run detection + greedy preprocessor, then interactively merge tracklets to build a golden standard.'
    )
    parser.add_argument('videos', type=str, nargs='+', help='Path(s) or glob pattern(s) to input video(s)')
    parser.add_argument('--output-dir', type=str, default='tmp', help='Directory to write golden metadata')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Expand patterns
    video_paths: list[Path] = []
    for pat in args.videos:
        expanded = [Path(p) for p in glob.glob(pat)]
        if not expanded:
            # allow literal path that may not glob-expand on Windows
            p = Path(pat)
            if p.exists():
                expanded = [p]
        video_paths.extend(expanded)
    video_paths = sorted({p.resolve() for p in video_paths if p.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv'}})
    if not video_paths:
        print('No input videos found for given patterns.')
        return

    # Batch GUI annotator over multiple videos
    controller = _BatchAnnotatorController(video_paths, output_dir)
    controller.run()


if __name__ == '__main__':
    main()
