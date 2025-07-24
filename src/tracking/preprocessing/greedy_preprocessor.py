import logging
from enum import Enum

from collections import defaultdict


from tracking.preprocessing.filter_non_surfers import filter_non_surfers_from_tracks
from video_io import VideoInfo
from common_types import Detection, FrameIndex, Track, TrackId, cosine_similarity, histogram_similarity

from typing import Callable
import numpy as np


class _ComparisonResult(Enum):
    MATCH = 'match'
    MAY_MATCH = 'may_match'
    NO_MATCH = 'no_match'


def _calc_pairwise(a: Track, b: Track, metric: Callable[[Detection, Detection], float]) -> float:
    """Calculate pairwise metric between all detections in two tracks."""
    total = 0
    count = 0
    for da in a.sorted_detections:
        for db in b.sorted_detections:
            total += metric(da, db)
            count += 1
    return total / count if count > 0 else 0.0


def _mean_embedding(t: Track) -> np.ndarray:
    """Calculate the mean embedding of a track."""
    return np.mean([d.embedding for d in t.sorted_detections], axis=0)


def pairwise_cosine_similarity(a: Track, b: Track) -> float:
    """Calculate pairwise cosine similarity between all detections in two tracks."""
    return _calc_pairwise(a, b, lambda a, b: cosine_similarity(a.embedding, b.embedding))


def mean_embedding_cosine_similarity(a: Track, b: Track) -> float:
    """Calculate cosine similarity between mean embeddings of two tracks."""
    return cosine_similarity(
        _mean_embedding(a),
        _mean_embedding(b),
    )


def pairwise_histogram_similarity(a: Track, b: Track) -> float:
    """Calculate pairwise histogram similarity between all detections in two tracks."""
    return _calc_pairwise(a, b, lambda a, b: histogram_similarity(a, b))


def _mean_embedding_histogram(t: Track) -> np.ndarray:
    """Calculate the mean embedding of a track."""
    return np.mean([d.color_histogram for d in t.sorted_detections], axis=0)


def mean_embedding_histogram_similarity(a: Track, b: Track) -> float:
    """Calculate histogram similarity between mean embeddings of two tracks."""
    return cosine_similarity(
        _mean_embedding_histogram(a),
        _mean_embedding_histogram(b),
    )


def pairwise_squared_cosine_similarity(a: Track, b: Track) -> float:
    """Calculate pairwise squared cosine similarity between all detections in two tracks."""
    return _calc_pairwise(a, b, lambda a, b: cosine_similarity(a.embedding, b.embedding) ** 2)


def prop_embeddings_sim(a: Track, b: Track, min_sim: float = 0.5) -> float:
    """Count how many embeddings in two tracks have cosine similarity above a threshold."""
    count = 0
    cnt = 0
    for da in a.sorted_detections:
        for db in b.sorted_detections:
            cnt += 1
            if cosine_similarity(da.embedding, db.embedding) >= min_sim:
                count += 1
    return count / cnt if cnt > 0 else 0.0


class GreedyPreprocessor:
    def __init__(
        self,
        greedy_min_iou: float = 0.5,
        greedy_min_cosine_similarity: float = 0.7,
        greedy_max_frame_distance: int = 5,
        greedy_min_iou_matches_single_track: float = 0.1,
    ):
        self.greedy_min_iou = greedy_min_iou
        self.greedy_min_cosine_similarity = greedy_min_cosine_similarity
        self.greedy_max_frame_distance = greedy_max_frame_distance
        self.greedy_min_iou_matches_single_track = greedy_min_iou_matches_single_track
        self.min_iou_matches_single_track = greedy_min_iou_matches_single_track

    def track_detections(self, detections: list[Detection], video_properties: VideoInfo) -> list[Track]:
        logging.info(f'{"=" * 80} Running greedy preprocessor {len(detections)} detections {"=" * 80}')

        tracks = self._preprocess_detections(detections)

        if False:  # TODO: reenable? seems to work well without it
            kept, removed = filter_non_surfers_from_tracks(tracks)
            logging.info(f'Greedy preprocessor kept {len(kept)} tracks and removed {len(removed)} tracks')
        else:
            kept = tracks

        # for the kept track, I want to experiment:
        # I want to compute the average embedding vector of each track
        # I want to compute the average cosine similarity between the embedding vectors of each detection of a track with the average embedding vector of the track
        # I want to compute the average cosine similarity between the embedding vectors of each detection of other tracks with the average embedding vector of the track
        # I want to compute the pairwise cosine similarity between the average embedding vectors of each of the tracks

        import numpy as np

        average_embeddings = {}
        for track in kept:
            average_embedding = np.mean([d.embedding for d in track.sorted_detections], axis=0)
            average_embeddings[track.track_id] = average_embedding

            average_cosine_similarity = 0
            for detection in track.sorted_detections:
                average_cosine_similarity += cosine_similarity(detection.embedding, average_embedding)
            average_cosine_similarity /= len(track.sorted_detections)
            print(
                f'Track {track.track_id} average cosine similarity with its own detections: {average_cosine_similarity} ({len(track.sorted_detections)} detections)'
            )

            for other_track in kept:
                if other_track.track_id == track.track_id:
                    continue
                if (
                    other_track.start_frame() < track.end_frame()  # It does not start after the track
                    or other_track.start_frame() > track.end_frame() + 30  # It starts too far in the future
                ):
                    continue
                average_cosine_similarity = 0
                for detection in other_track.sorted_detections:
                    average_cosine_similarity += cosine_similarity(detection.embedding, average_embedding)
                average_cosine_similarity /= len(other_track.sorted_detections)
                print(
                    f'Track {track.track_id} average cosine similarity with track {other_track.track_id}: {average_cosine_similarity} ({len(other_track.sorted_detections)} detections)'
                )

        for i in range(len(kept)):
            for j in range(len(kept)):
                track_i = kept[i]
                track_j = kept[j]
                if (
                    track_i.start_frame() < track_j.end_frame()  # It does not start after the track
                    or track_i.start_frame() > track_j.end_frame() + 30  # It starts too far in the future
                    or i == j
                ):
                    continue
                average_cosine_similarity = cosine_similarity(
                    average_embeddings[track_i.track_id], average_embeddings[track_j.track_id]
                )
                print(
                    f'Pairwise cosine similarity between track {track_j.track_id} and track {track_i.track_id}: {average_cosine_similarity} ({len(track_i.sorted_detections)} vs {len(track_j.sorted_detections)} detections)'
                )

                average_pairwise_cosine_similarity = 0
                number_of_detections_with_good_similarity = 0
                for detection_i in track_i.sorted_detections:
                    for detection_j in track_j.sorted_detections:
                        sim = cosine_similarity(detection_i.embedding, detection_j.embedding)
                        if sim >= self.greedy_min_cosine_similarity:
                            number_of_detections_with_good_similarity += 1
                        average_pairwise_cosine_similarity += sim
                average_pairwise_cosine_similarity /= len(track_i.sorted_detections) * len(track_j.sorted_detections)
                print(
                    f'Average pairwise cosine similarity between track {track_j.track_id} and track {track_i.track_id}: {average_pairwise_cosine_similarity}'
                )
                print(
                    f'Number of detections with good similarity: {number_of_detections_with_good_similarity}/{len(track_i.sorted_detections) * len(track_j.sorted_detections)} ({number_of_detections_with_good_similarity / (len(track_i.sorted_detections) * len(track_j.sorted_detections)) * 100:.2f}%)'
                )
        return kept

    def _preprocess_detections(self, detections: list[Detection]) -> list[Track]:
        """Greedily stiches detections onto tracks as long as both IOU and cosine similarity are high."""

        # We match greedily if:
        # the bounding box of a detection overlaps only with a single active track
        # or both iou and cosine similarity are high enough to continue the track.
        #
        # We only match against active tracks.
        # Tracks become stale if they:
        # - are too old
        # - have been considered for a match but ot chosen
        # - have been matched by multiple detections in the same frame

        detections_by_frame: dict[FrameIndex, list[Detection]] = defaultdict(list)
        for det in detections:
            detections_by_frame[det.frame_idx].append(det)

        # Sort detections by frame index to process them in order.
        sorted_frame_indices = sorted(detections_by_frame.keys())

        next_track_id = 1

        # these tracks have been detected and can be matched to further detections
        active_tracks: list[Track] = []
        # these tracks have been detected but can't match further detections
        stale_tracks: list[Track] = []
        stale_track_ids: set[TrackId] = set()

        for frame_idx in sorted_frame_indices:
            matches_this_frame = []
            for detection in detections_by_frame[frame_idx]:
                clean_matches: list[Track] = []
                mby_matches: list[Track] = []

                for track in active_tracks:
                    comparison_result = self._compare_detection_to_track(track, detection)
                    if comparison_result == _ComparisonResult.MATCH:
                        # Track matches the detection, continue it
                        clean_matches.append(track)
                    elif comparison_result == _ComparisonResult.MAY_MATCH:
                        mby_matches.append(track)

                if len(clean_matches) == 1:
                    matches_this_frame.append((clean_matches[0], detection))
                elif len(clean_matches) == 0 and len(mby_matches) == 1:
                    matches_this_frame.append((mby_matches[0], detection))
                else:
                    # no clear match found, create a new track for this detection
                    new_track = Track(track_id=next_track_id, sorted_detections=[])
                    matches_this_frame.append((new_track, detection))
                    next_track_id += 1

                    # all clean and mby matches are stale because we couldn't cleanly match them
                    for track in clean_matches + mby_matches:
                        if track.track_id not in stale_track_ids:
                            stale_track_ids.add(track.track_id)
                            stale_tracks.append(track)

            # now we construct tracks for any match that is not a duplicate.
            # Tracks that are matched by multiple detections will become stale (the detections will be assigned to new tracks).
            detections_per_track = defaultdict(list)
            tracks_per_track_id = {}
            for track, detection in matches_this_frame:
                detections_per_track[track.track_id].append(detection)
                tracks_per_track_id[track.track_id] = track

            for track_id, detections in detections_per_track.items():
                track = tracks_per_track_id[track_id]
                if len(detections) > 1:
                    stale_track_ids.add(track_id)
                    stale_tracks.append(track)
                    for detection in detections:
                        active_tracks.append(Track(track_id=next_track_id, sorted_detections=[detection]))
                        next_track_id += 1
                else:
                    assert len(detections) == 1
                    track.sorted_detections.append(detections[0])
                    # JANK: for new tracks which haven't been in active_tracks yet, we need to add them
                    if track not in active_tracks:
                        active_tracks.append(track)

            # Too old tracks are stale
            for track in active_tracks:
                if track.end().frame_idx + self.greedy_max_frame_distance < frame_idx:
                    if track.track_id not in stale_track_ids:
                        stale_track_ids.add(track.track_id)
                        stale_tracks.append(track)
            # update stale / active tracks once per frame
            active_tracks = [track for track in active_tracks if track not in stale_tracks]

        return stale_tracks + active_tracks

    def _compare_detection_to_track(self, track: Track, detection: Detection) -> _ComparisonResult:
        iou = track.end().bbox.iou(detection.bbox)

        if iou < self.min_iou_matches_single_track:
            return _ComparisonResult.NO_MATCH

        n = len(track.sorted_detections)
        average_sim = sum(cosine_similarity(d.embedding, detection.embedding) for d in track.sorted_detections) / n

        if iou >= self.greedy_min_iou and average_sim >= self.greedy_min_cosine_similarity:
            return _ComparisonResult.MATCH

        return _ComparisonResult.MAY_MATCH
