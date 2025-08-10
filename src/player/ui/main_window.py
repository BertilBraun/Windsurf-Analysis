from __future__ import annotations
from pathlib import Path
from typing import Optional, List

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QMainWindow, QWidget, QFileDialog, QVBoxLayout, QMessageBox

from core.player_state import PlayerState
from core.video_manager import VideoManager
from core.metadata_loader import load_tracks_metadata
from ui.video_widget import VideoWidget
from ui.timeline_widget import TimelineWidget
from ui.controls_widget import ControlsWidget


class MainWindow(QMainWindow):
    def __init__(self, start_directory: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle('Windsurf Player')

        self.state = PlayerState()
        self.video: Optional[VideoManager] = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)  # ~60 FPS UI timer

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

    def _on_seek(self, frame: int) -> None:
        if not self.video:
            return
        self.state.current_frame = max(0, min(frame, self.video.total_frames - 1))
        self.video.seek_frame(self.state.current_frame)
        _, frame_img = self.video.read_frame()
        if frame_img is not None:
            self.video_widget.set_frame(frame_img)
        self.timeline.update()

    def _on_visibility_changed(self, visible: set[int]) -> None:
        self.video_widget.update()

    # ----------------------------- Playback loop ---------------------------- #
    def _tick(self) -> None:
        if not (self.state.is_playing and self.video):
            return
        step = max(1, int(self.state.playback_speed))
        next_frame = self.state.current_frame + step
        if next_frame >= self.video.total_frames:
            # stop at end; do not auto-advance to next video
            self.state.is_playing = False
            return
        self._on_seek(next_frame)

    # --------------------------- Loading and setup -------------------------- #
    def _ask_output_dir(self) -> Optional[Path]:
        dlg = QFileDialog(self, 'Select output directory with .tracks.json files')
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        if dlg.exec():
            sel = dlg.selectedFiles()
            if sel:
                return Path(sel[0])
        return None

    def _load_first_tracks_file(self, directory: Path) -> None:
        self.metadata_files = sorted(directory.glob('*.tracks.json'))
        if not self.metadata_files:
            QMessageBox.warning(self, 'No metadata', 'No .tracks.json files found in the selected directory.')
            return
        self._load_metadata_by_index(0)

    def _load_metadata_by_index(self, index: int) -> None:
        if not (0 <= index < len(self.metadata_files)):
            return
        self.current_metadata_index = index
        self._load_metadata(self.metadata_files[index])

    def _load_metadata(self, json_path: Path) -> None:
        input_video_path, video_props, tracks = load_tracks_metadata(json_path)
        self.state.reset_for_new_video()
        self.state.video_properties = video_props
        self.state.loaded_tracks = tracks
        self.state.visible_tracks = {t.track_id for t in tracks}
        self.state.input_video_path = input_video_path
        self.setWindowTitle(f'Windsurf Player - {Path(input_video_path).name}')

        self._open_video(Path(input_video_path))

    def _open_video(self, video_path: Path) -> None:
        if self.video:
            self.video.release()
        self.video = VideoManager(video_path)
        self.state.current_frame = 0
        self._on_seek(0)

    # ----------------------------- Key bindings ----------------------------- #
    def keyPressEvent(self, event):  # type: ignore[override]
        # Ensure main window consumes Spacebar instead of focused buttons
        if event.key() == Qt.Key.Key_Space:
            self._toggle_play()
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
        elif event.key() == Qt.Key.Key_Left:
            self._on_seek(self.state.current_frame - 1)
        elif event.key() == Qt.Key.Key_Right:
            self._on_seek(self.state.current_frame + 1)
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
        if 0 <= nxt < len(self.metadata_files):
            self._load_metadata_by_index(nxt)

    def _enter_detailed(self, track_id: int) -> None:
        self.state.current_track_id = track_id
        self.state.current_mode = 'detailed'
        self.video_widget.reset_zoom()
        self.video_widget.update()
