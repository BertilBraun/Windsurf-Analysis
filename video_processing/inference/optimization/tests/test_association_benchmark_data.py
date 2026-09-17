from __future__ import annotations

import pickle
from pathlib import Path

from video_processing.inference.optimization.association_benchmark_data import (
    load_golden_sequence,
    resolve_video_path,
)
from video_processing.inference.src.player.core.player_state import (
    DetectionLite,
    Metadata,
    TrackLite,
    VideoProperties,
)


def _detection(frame_index: int, *, interpolated: bool) -> DetectionLite:
    return DetectionLite(
        frame_idx=frame_index,
        bbox=[frame_index, 0, frame_index + 10, 10],
        confidence=0.9,
        interpolated=interpolated,
        boom=[0.0, 0.0, 0.0],
        mast_tip=[0.0, 0.0, 0.0],
        anchor=[0, 0],
        scale=1.0,
    )


def test_resolve_video_path_prefers_case_insensitive_sibling(tmp_path: Path) -> None:
    golden_path = tmp_path / 'MVI_5348.golden.tracks.pkl'
    video_path = tmp_path / 'MVI_5348.MP4'
    video_path.touch()

    resolved = resolve_video_path(golden_path, 'C:/stale/MVI_5348.MP4')

    assert resolved == video_path.resolve()


def test_load_filters_interpolated_observations(tmp_path: Path) -> None:
    golden_path = tmp_path / 'sample.golden.tracks.pkl'
    (tmp_path / 'sample.mp4').touch()
    metadata = Metadata(
        input_video_path='C:/stale/sample.mp4',
        video_properties=VideoProperties(fps=30.0, width=100, height=100, total_frames=3),
        tracks=[
            TrackLite(
                track_id=1,
                start_frame=0,
                end_frame=2,
                start_time=0.0,
                duration=0.1,
                detection_count=3,
                detections=[
                    _detection(0, interpolated=False),
                    _detection(1, interpolated=True),
                    _detection(2, interpolated=False),
                ],
            )
        ],
    )
    with golden_path.open('wb') as file:
        pickle.dump(metadata, file)

    sequence = load_golden_sequence(golden_path)

    assert len(sequence.gold_assignment) == 2
    assert [detection.frame_idx for detection in sequence.original_tracks[0].detections] == [0, 2]
    assert sequence.original_tracks[0].detection_count == 2
