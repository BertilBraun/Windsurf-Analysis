from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton


class ControlsWidget(QWidget):
    def __init__(
        self,
        on_play_pause: Callable[[], None],
        on_speed_down: Callable[[], None],
        on_speed_up: Callable[[], None],
        on_prev_video: Callable[[], None],
        on_next_video: Callable[[], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.on_play_pause = on_play_pause
        self.on_speed_down = on_speed_down
        self.on_speed_up = on_speed_up
        self.on_prev_video = on_prev_video
        self.on_next_video = on_next_video

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        self.play_btn = QPushButton('Play/Pause')
        self.play_btn.clicked.connect(self.on_play_pause)  # type: ignore[arg-type]
        self.slow_btn = QPushButton('Speed -')
        self.slow_btn.clicked.connect(self.on_speed_down)  # type: ignore[arg-type]
        self.fast_btn = QPushButton('Speed +')
        self.fast_btn.clicked.connect(self.on_speed_up)  # type: ignore[arg-type]
        self.prev_btn = QPushButton('Prev Video')
        self.prev_btn.clicked.connect(self.on_prev_video)  # type: ignore[arg-type]
        self.next_btn = QPushButton('Next Video')
        self.next_btn.clicked.connect(self.on_next_video)  # type: ignore[arg-type]

        layout.addWidget(self.play_btn, 0)
        layout.addWidget(self.slow_btn, 0)
        layout.addWidget(self.fast_btn, 0)
        layout.addWidget(self.prev_btn, 0)
        layout.addWidget(self.next_btn, 0)

        # Prevent buttons from stealing Spacebar focus
        self.play_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slow_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.fast_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.prev_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.next_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
