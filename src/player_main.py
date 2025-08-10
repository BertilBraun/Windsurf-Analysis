from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QApplication

from player.ui.main_window import MainWindow


def run_player(start_dir: Optional[str] = None) -> None:
    app = QApplication(sys.argv)
    window = MainWindow(start_directory=Path(start_dir) if start_dir else None)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    run_player()
