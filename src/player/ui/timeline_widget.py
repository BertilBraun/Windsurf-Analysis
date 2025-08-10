from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRect, Signal
from PySide6.QtGui import QPainter, QColor, QMouseEvent
from PySide6.QtWidgets import QWidget

from core.player_state import PlayerState


class TimelineWidget(QWidget):
    frameChanged = Signal(int)

    def __init__(self, state: PlayerState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.state = state
        self.setMinimumHeight(40)
        self.setSizePolicy(self.sizePolicy().horizontalPolicy(), self.sizePolicy().verticalPolicy())

    def paintEvent(self, event):  # type: ignore[override]
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        if not self.state.video_properties:
            return

        total = max(1, self.state.video_properties.total_frames)
        # Draw current position
        x = int(self.width() * (self.state.current_frame / total))
        painter.fillRect(QRect(x - 1, 0, 2, self.height()), QColor(200, 200, 200))

        # Draw simple track spans
        for t in self.state.loaded_tracks:
            lx = int(self.width() * (t.start_frame / total))
            rx = int(self.width() * (t.end_frame / total))
            painter.fillRect(
                QRect(lx, self.height() // 2, max(1, rx - lx), self.height() // 2), QColor(80, 160, 240, 140)
            )

        painter.end()

    def mousePressEvent(self, event: QMouseEvent):  # type: ignore[override]
        if not self.state.video_properties:
            return
        total = max(1, self.state.video_properties.total_frames)
        ratio = min(1.0, max(0.0, event.position().x() / max(1, self.width())))
        frame = int(ratio * total)
        self.frameChanged.emit(frame)
