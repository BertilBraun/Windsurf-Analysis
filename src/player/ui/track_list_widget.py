from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QListWidget, QListWidgetItem

from core.player_state import PlayerState


class TrackListWidget(QWidget):
    def __init__(
        self, state: PlayerState, on_visibility_changed: Callable[[set[int]], None], parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.state = state
        self.on_visibility_changed = on_visibility_changed

        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)
        self.list_widget.itemChanged.connect(self._on_item_changed)  # type: ignore[arg-type]

    def refresh(self) -> None:
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for t in self.state.loaded_tracks:
            item = QListWidgetItem(f'Track {t.track_id}')
            item.setCheckState(
                Qt.CheckState.Checked
                if (not self.state.visible_tracks or t.track_id in self.state.visible_tracks)
                else Qt.CheckState.Unchecked
            )
            item.setData(256, t.track_id)
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

    def _on_item_changed(self, item: QListWidgetItem) -> None:  # type: ignore[override]
        track_id = int(item.data(256))
        if item.checkState():
            self.state.visible_tracks.add(track_id)
        else:
            self.state.visible_tracks.discard(track_id)
        self.on_visibility_changed(set(self.state.visible_tracks))
