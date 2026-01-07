from __future__ import annotations
from pathlib import Path
import json
import pickle
from typing import Optional, List

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QMainWindow, QWidget, QFileDialog, QVBoxLayout, QMessageBox

from ..core.player_state import Metadata, PlayerState, VideoProperties
from ..core.video_manager import VideoManager
from ..ui.video_widget import VideoWidget
from ..ui.timeline_widget import TimelineWidget
from ..ui.controls_widget import ControlsWidget


class MainWindow(QMainWindow):
    def __init__(self, start_directory: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle('Windsurf Player')

        self.state = PlayerState()
        self.state.reset(
            input_video_path='',
            video_properties=VideoProperties(fps=0, width=0, height=0, total_frames=0),
            loaded_tracks=[],
        )

        self.video: Optional[VideoManager] = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)  # adjusted after video load

        # UI layout
        central = QWidget()
        root_layout = QVBoxLayout(central)

        self.video_widget = VideoWidget(self.state, on_track_selected=self._enter_detailed)
        self.timeline = TimelineWidget(self.state)
        self.timeline.frameChanged.connect(self._on_seek)

        self.controls = ControlsWidget(
            on_play_pause=self._toggle_play,
            on_speed_down=lambda: self._bump_speed(down=True),
            on_speed_up=lambda: self._bump_speed(down=False),
            on_prev_video=lambda: self._load_adjacent(-1),
            on_next_video=lambda: self._load_adjacent(1),
        )

        # Single-pane layout (removed track sidebar)
        root_layout.addWidget(self.video_widget, 1)
        root_layout.addWidget(self.timeline, 0)
        root_layout.addWidget(self.controls, 0)
        # Keep UI from ballooning
        self.controls.setMaximumHeight(48)
        self.timeline.setMaximumHeight(56)
        self.setCentralWidget(central)

        # Startup flow: pick a directory, auto-load first video, then stop auto-advance
        self.output_dir = start_directory or self._ask_output_dir()
        self.metadata_files: List[Path] = []
        self.current_metadata_index: int = -1
        if self.output_dir:
            self._load_first_tracks_file(self.output_dir)

    # ------------------------------ UI actions ------------------------------ #
    def _toggle_play(self) -> None:
        self.state.is_playing = not self.state.is_playing
        self._update_timer_interval()

    def _bump_speed(self, down: bool) -> None:
        rates = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        try:
            idx = rates.index(self.state.playback_speed)
        except ValueError:
            idx = 2
        idx = max(0, min(len(rates) - 1, idx - 1 if down else idx + 1))
        self.state.playback_speed = rates[idx]
        self.video_widget.show_hud(f'Speed: {self.state.playback_speed}x')
        QTimer.singleShot(1000, self.video_widget.clear_hud)
        self._update_timer_interval()

    def _on_seek(self, frame: int) -> None:
        if not self.video:
            return
        self.state.current_frame = max(0, min(frame, self.video.total_frames - 1))
        self.video.seek_frame(self.state.current_frame)
        _, frame_img = self.video.read_frame()
        if frame_img is not None:
            self.video_widget.set_frame(frame_img)
        self.timeline.update()
        # no time accumulation; frame-locked playback

    def _step_next(self) -> None:
        """Advance by exactly one frame using grab/read path for speed."""
        if not self.video:
            return
        self.state.is_playing = False
        idx, frame_img = self.video.advance_by(1)
        if idx < 0 or frame_img is None:
            return
        self.state.current_frame = idx
        self.video_widget.set_frame(frame_img)
        self.timeline.update()

    def _on_visibility_changed(self, visible: set[int]) -> None:
        self.video_widget.update()

    # ----------------------------- Playback loop ---------------------------- #
    def _tick(self) -> None:
        if not (self.state.is_playing and self.video):
            return
        # Frame-locked playback: advance exactly one frame per tick
        idx, frame_img = self.video.advance_by(1)
        if idx < 0 or frame_img is None:
            self.state.is_playing = False
            return
        self.state.current_frame = idx
        # if we are in detailed mode but the track has no further detections, switch to overview mode
        if self.state.current_mode == 'detailed':
            detections = self.state.detections_by_frame.get(idx, [])
            has_detections = any(track_id == self.state.current_track_id for track_id, det in detections)
            if not has_detections:
                self.state.current_mode = 'overview'
                self.state.current_track_id = None
                self.video_widget.update()
        self.video_widget.set_frame(frame_img)
        self.timeline.update()

    # --------------------------- Loading and setup -------------------------- #
    def _ask_output_dir(self) -> Optional[Path]:
        dlg = QFileDialog(self, 'Select output directory with .tracks.pkl files')
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        if dlg.exec():
            sel = dlg.selectedFiles()
            if sel:
                return Path(sel[0])
        return None

    def _load_first_tracks_file(self, directory: Path) -> None:
        self.metadata_files = sorted(directory.glob('*.tracks.pkl'))
        if not self.metadata_files:
            QMessageBox.warning(self, 'No metadata', 'No .tracks.pkl files found in the selected directory.')
            return
        self._load_metadata_by_index(0)

    def _load_metadata_by_index(self, index: int) -> None:
        if index < 0 or index >= len(self.metadata_files):
            return
        self.current_metadata_index = index
        self._load_metadata(self.metadata_files[index])

    def _load_metadata(self, metadata_path: Path) -> None:
        with open(metadata_path, 'rb') as f:
            metadata: Metadata = pickle.load(f)

        stabilization_by_frame = self._load_stabilization_by_frame(metadata_path, metadata.video_properties.total_frames)
        self.state.reset(
            input_video_path=metadata.input_video_path,
            video_properties=metadata.video_properties,
            loaded_tracks=metadata.tracks,
            stabilization_by_frame=stabilization_by_frame,
        )

        self.setWindowTitle(f'Windsurf Player - {Path(metadata.input_video_path).name}')

        self._open_video(Path(metadata.input_video_path))

    @staticmethod
    def _load_stabilization_by_frame(metadata_path: Path, total_frames: int) -> list[tuple[float, float, float]]:
        """
        Load per-frame stabilization deltas from the local pipeline output.

        Expected sibling file: `<stem>.stabilization_transforms.json` next to `<stem>.tracks.pkl`.
        Returns a dense list of (dx, dy, da) (da in radians), length `total_frames` when possible.
        """
        if total_frames <= 0:
            return []

        name = metadata_path.name
        if name.endswith('.tracks.pkl'):
            stem = name[: -len('.tracks.pkl')]
        else:
            stem = metadata_path.stem

        path = metadata_path.with_name(f'{stem}.stabilization_transforms.json')
        if not path.exists():
            return [(0.0, 0.0, 0.0) for _ in range(int(total_frames))]

        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            transforms = payload.get('transforms', [])
        except Exception:
            return [(0.0, 0.0, 0.0) for _ in range(int(total_frames))]

        out: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0) for _ in range(int(total_frames))]

        if isinstance(transforms, list) and len(transforms) == int(total_frames) and all(
            isinstance(t, dict) for t in transforms
        ):
            # Dense list; allow either implicit index or explicit frame_idx.
            for i, t in enumerate(transforms):
                frame_idx = int(t.get('frame_idx', i))
                if 0 <= frame_idx < len(out):
                    out[frame_idx] = (float(t.get('dx', 0.0)), float(t.get('dy', 0.0)), float(t.get('da', 0.0)))
            return out

        if isinstance(transforms, list):
            for t in transforms:
                if not isinstance(t, dict):
                    continue
                frame_idx = int(t.get('frame_idx', -1))
                if 0 <= frame_idx < len(out):
                    out[frame_idx] = (float(t.get('dx', 0.0)), float(t.get('dy', 0.0)), float(t.get('da', 0.0)))

        return out

    def _open_video(self, video_path: Path) -> None:
        if self.video:
            self.video.release()
        self.video = VideoManager(video_path)
        self.state.current_frame = 0
        self._on_seek(0)
        self._update_timer_interval()

    def _update_timer_interval(self) -> None:
        if not self.video or self.video.fps <= 0:
            return
        # one frame per tick, factoring speed
        effective_fps = max(0.1, self.video.fps * float(self.state.playback_speed))
        self.timer.setInterval(int(max(5, min(1000.0, 1000.0 / effective_fps))))

    # ----------------------------- Key bindings ----------------------------- #
    def keyPressEvent(self, event):  # type: ignore[override]
        # Ensure main window consumes Spacebar instead of focused buttons
        if event.key() == Qt.Key.Key_Space:
            self._toggle_play()
            return
        if event.key() == Qt.Key.Key_Left and not (
            event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
        ):
            # Frame-precise back step that ignores timer accumulation
            self.state.is_playing = False
            self._accumulated_frames = 0.0
            self._on_seek(self.state.current_frame - 1)
            return
        if event.key() == Qt.Key.Key_Right and not (
            event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
        ):
            # Fast path: avoid seek for +1 frame
            self._step_next()
            return
        # modifiers first
        if (event.modifiers() & Qt.KeyboardModifier.ControlModifier) and event.key() == Qt.Key.Key_Left:
            self._on_seek(
                self.state.current_frame - int(self.state.video_properties.fps * 30)
                if self.state.video_properties
                else 0
            )
            return
        if (event.modifiers() & Qt.KeyboardModifier.ControlModifier) and event.key() == Qt.Key.Key_Right:
            self._on_seek(
                self.state.current_frame + int(self.state.video_properties.fps * 30)
                if self.state.video_properties
                else 0
            )
            return
        if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) and event.key() == Qt.Key.Key_Left:
            self._on_seek(
                self.state.current_frame - int(self.state.video_properties.fps * 5)
                if self.state.video_properties
                else 0
            )
            return
        if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) and event.key() == Qt.Key.Key_Right:
            self._on_seek(
                self.state.current_frame + int(self.state.video_properties.fps * 5)
                if self.state.video_properties
                else 0
            )
            return

        if event.key() == Qt.Key.Key_Minus:
            self._bump_speed(down=True)
        elif event.key() == Qt.Key.Key_Plus or event.key() == Qt.Key.Key_Equal:
            self._bump_speed(down=False)
        # plain left/right handled above for snappy stepping
        elif event.key() == Qt.Key.Key_Q:
            self.close()
        elif event.key() == Qt.Key.Key_Escape:
            self.state.current_mode = 'overview'
            self.video_widget.update()
        elif event.key() == Qt.Key.Key_N:
            self._load_adjacent(1)
        elif event.key() == Qt.Key.Key_P:
            self._load_adjacent(-1)

    def _load_adjacent(self, delta: int) -> None:
        if not self.metadata_files:
            return
        nxt = self.current_metadata_index + delta
        self._load_metadata_by_index(nxt)

    def _enter_detailed(self, track_id: int) -> None:
        self.state.current_track_id = track_id
        self.state.current_mode = 'detailed'
        self.video_widget.reset_zoom()
        self.video_widget.update()
