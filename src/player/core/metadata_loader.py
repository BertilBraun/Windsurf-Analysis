from __future__ import annotations

import pickle
from pathlib import Path

from player.core.player_state import Metadata


def load_tracks_metadata(path: Path) -> Metadata:
    with open(path, 'rb') as f:
        return pickle.load(f)
