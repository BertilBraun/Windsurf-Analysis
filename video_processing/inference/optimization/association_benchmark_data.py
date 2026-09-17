from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from video_processing.inference.optimization.optimization_util import (
    AssignmentKey,
    _extract_embeddings_for_tracklets,
    _load_golden,
    _to_track_with_embeddings,
)
from video_processing.inference.src.common_types import Detection, Track
from video_processing.inference.src.player.core.player_state import DetectionLite, Metadata, TrackLite


VIDEO_SUFFIXES = frozenset({'.mp4', '.mov', '.m4v'})


@dataclass(frozen=True)
class GoldenSequence:
    golden_path: Path
    video_path: Path
    metadata: Metadata
    original_tracks: tuple[TrackLite, ...]
    gold_assignment: dict[AssignmentKey, int]

    @property
    def name(self) -> str:
        return self.golden_path.name.removesuffix('.golden.tracks.pkl')


def observation_key(detection: DetectionLite | Detection) -> AssignmentKey:
    if isinstance(detection, DetectionLite):
        x1, y1, x2, y2 = detection.bbox
        return AssignmentKey(int(detection.frame_idx), int(x1), int(y1), int(x2), int(y2))
    return AssignmentKey(
        int(detection.frame_idx),
        int(detection.bbox.x1),
        int(detection.bbox.y1),
        int(detection.bbox.x2),
        int(detection.bbox.y2),
    )


def resolve_video_path(golden_path: Path, metadata_path: str) -> Path:
    sequence_stem = golden_path.name.removesuffix('.golden.tracks.pkl').casefold()
    sibling_candidates = sorted(
        path.resolve()
        for path in golden_path.parent.iterdir()
        if path.is_file() and path.suffix.casefold() in VIDEO_SUFFIXES and path.stem.casefold() == sequence_stem
    )
    if len(sibling_candidates) == 1:
        return sibling_candidates[0]
    if len(sibling_candidates) > 1:
        raise ValueError(f'Ambiguous sibling videos for {golden_path.name}: {sibling_candidates}')

    stored_path = Path(metadata_path)
    if stored_path.is_file():
        return stored_path.resolve()
    raise FileNotFoundError(f'No video found for {golden_path.name}; stored path is {metadata_path!r}.')


def load_golden_sequence(golden_path: Path) -> GoldenSequence:
    metadata = _load_golden(golden_path)
    video_path = resolve_video_path(golden_path, metadata.input_video_path)
    original_tracks: list[TrackLite] = []
    assignment: dict[AssignmentKey, int] = {}

    seen_track_ids: set[int] = set()
    for track in metadata.tracks:
        if track.track_id in seen_track_ids:
            raise ValueError(f'Duplicate gold track id {track.track_id} in {golden_path.name}.')
        seen_track_ids.add(track.track_id)
        original_detections = [detection for detection in track.detections if not detection.interpolated]
        if not original_detections:
            continue
        for detection in original_detections:
            key = observation_key(detection)
            if key in assignment:
                raise ValueError(f'Duplicate observation key {key} in {golden_path.name}.')
            assignment[key] = int(track.track_id)
        original_tracks.append(
            replace(
                track,
                start_frame=min(detection.frame_idx for detection in original_detections),
                end_frame=max(detection.frame_idx for detection in original_detections),
                detection_count=len(original_detections),
                detections=original_detections,
            )
        )

    return GoldenSequence(
        golden_path=golden_path.resolve(),
        video_path=video_path,
        metadata=metadata,
        original_tracks=tuple(original_tracks),
        gold_assignment=assignment,
    )


def load_golden_sequences(golden_directory: Path) -> list[GoldenSequence]:
    paths = sorted(golden_directory.glob('*.golden.tracks.pkl'))
    if not paths:
        raise ValueError(f'No golden files found in {golden_directory}.')
    return [load_golden_sequence(path) for path in paths]


def materialize_original_detections(sequence: GoldenSequence) -> list[Detection]:
    tracklets = list(sequence.original_tracks)
    embeddings = _extract_embeddings_for_tracklets(str(sequence.video_path), tracklets)
    detections: list[Detection] = []
    for tracklet, tracklet_embeddings in zip(tracklets, embeddings):
        reconstructed_track = _to_track_with_embeddings(tracklet, tracklet_embeddings)
        detections.extend(reconstructed_track.sorted_detections)
    detections.sort(key=lambda detection: observation_key(detection))
    if {observation_key(detection) for detection in detections} != set(sequence.gold_assignment):
        raise ValueError(f'Reconstructed observations differ from golden keys for {sequence.name}.')
    return detections


def singleton_tracks(detections: list[Detection]) -> list[Track]:
    return [
        Track(track_id=index + 1, sorted_detections=[detection.copy()]) for index, detection in enumerate(detections)
    ]
