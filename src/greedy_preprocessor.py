#!/usr/bin/env python3

import z3
import logging
from enum import Enum

from collections import defaultdict


from video_io import VideoInfo
from common_types import Detection, FrameIndex, Track, TrackId, cosine_similarity

class _ComparisonResult(Enum):
    MATCH = 'match'
    MAY_MATCH = 'may_match'
    NO_MATCH = 'no_match'


class GreedyPreprocessor:
    def __init__(
        self,
        greedy_min_iou: float = 0.8,
        greedy_min_cosine_similarity: float = 0.5,
        greedy_max_frame_distance: int = 5,
        greedy_min_iou_matches_single_track: float = 0.2,
    ):
        self.greedy_min_iou = greedy_min_iou
        self.greedy_min_cosine_similarity = greedy_min_cosine_similarity
        self.greedy_max_frame_distance = greedy_max_frame_distance
        self.greedy_min_iou_matches_single_track = greedy_min_iou_matches_single_track
        self.min_iou_matches_single_track = greedy_min_iou_matches_single_track

    def _compare_detection_to_track(self, track: Track, detection: Detection) -> _ComparisonResult:
        iou = track.end().bbox.iou(detection.bbox)

        if iou < self.min_iou_matches_single_track:
            return _ComparisonResult.NO_MATCH

        # TODO cosine similarity to more than just one frame?
        n = len(track.sorted_detections)
        average_sim = sum(cosine_similarity(d.feat, detection.feat) for d in track.sorted_detections) / n

        if iou >= self.greedy_min_iou and average_sim >= self.greedy_min_cosine_similarity:
            return _ComparisonResult.MATCH

        return _ComparisonResult.MAY_MATCH

    def _preprocess_detections(self, detections_by_frame: dict[int, list[Detection]]) -> list[Track]:
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
                    matches_this_frame.append(
                        (clean_matches[0], detection)
                    )
                    # track = clean_matches[0]
                    # track.sorted_detections.append(detection)
                elif len(clean_matches) == 0 and len(mby_matches) == 1:
                    # track = mby_matches[0]
                    # track.sorted_detections.append(detection)
                    matches_this_frame.append(
                        (mby_matches[0], detection)
                    )
                else:
                    # no clear match found, create a new track for this detection
                    new_track = Track(track_id=next_track_id, sorted_detections=[])
                    matches_this_frame.append(
                        (new_track, detection)
                    )
                    # active_tracks.append(new_track)
                    next_track_id += 1

                    # all clean and mby matches are stale because we couldn't cleanly match them
                    for track in clean_matches + mby_matches:
                        if track.track_id not in stale_track_ids:
                            stale_track_ids.add(track.track_id)
                            stale_tracks.append(track)
            # now we construct tracks for any match that is not a duplicate. Tracks that are matched by multiple detections will become stale (the detections will be assigned to the new track).
            detections_per_track = defaultdict(list)
            tracks_per_track_id = {}
            for track, detection in matches_this_frame:
                detections_per_track[track.track_id].append(detection)
                tracks_per_track_id[track.track_id] = track
            for track_id, dets in detections_per_track.items():
                track = tracks_per_track_id[track_id]
                if len(dets) > 1:
                    stale_track_ids.add(track_id)
                    stale_tracks.append(track)
                    for detection in dets:
                        active_tracks.append(Track(track_id=next_track_id, sorted_detections=[detection]))
                        next_track_id += 1
                else:
                    assert len(dets) == 1
                    track.sorted_detections.append(dets[0])
                    # JANK
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

    def track_detections(
        self,
        detections: list[Detection],
        video_properties: VideoInfo | None = None,
    ) -> list[Track]:
        logging.info(
            f"{'=' * 80} Running greedy preprocessor {len(detections)} detections {'=' * 80}")
        detections_by_frame: dict[FrameIndex, list[Detection]] = defaultdict(list)
        for det in detections:
            detections_by_frame[det.frame_idx].append(det)
        return self._preprocess_detections(detections_by_frame)
