from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

from core.player_state import PlayerState, TrackLite, DetectionLite, VideoProperties


def load_tracks_metadata(path: Path) -> Tuple[str, VideoProperties, List[TrackLite]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    input_video_path: str = data['input_video_path']
    vp_raw = data['video_properties']
    video_props = VideoProperties(
        fps=float(vp_raw['fps']),
        width=int(vp_raw['width']),
        height=int(vp_raw['height']),
        total_frames=int(vp_raw['total_frames']),
    )

    tracks: List[TrackLite] = []
    for t in data['tracks']:
        dets = [
            DetectionLite(
                frame_idx=int(d['frame_idx']),
                bbox=[int(b) for b in d['bbox']],
                confidence=float(d['confidence']),
            )
            for d in t['detections']
        ]
        tracks.append(
            TrackLite(
                track_id=int(t['track_id']),
                start_frame=int(t['start_frame']),
                end_frame=int(t['end_frame']),
                start_time=float(t['start_time']),
                duration=float(t['duration']),
                detection_count=int(t['detection_count']),
                detections=dets,
            )
        )

    return input_video_path, video_props, tracks
