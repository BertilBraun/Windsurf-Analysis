import logging
from enum import Enum
from collections import defaultdict


from ...util.video_io import VideoInfo
from ...common_types import Detection, FrameIndex, Track, TrackId
from ...util.similarity_helpers import cosine_similarity


class _ComparisonResult(Enum):
    MATCH = 'match'
    MAY_MATCH = 'may_match'
    NO_MATCH = 'no_match'


class GreedyTrackStitcher:
    def __init__(
        self,
        greedy_min_iou: float,
        greedy_min_cosine_similarity: float,
        greedy_max_frame_distance: int,
        greedy_ema_alpha: float,
    ):
        self.min_iou = greedy_min_iou
        self.min_cosine_similarity = greedy_min_cosine_similarity
        self.max_frame_distance = greedy_max_frame_distance
        self.ema_alpha = greedy_ema_alpha

    def track(self, tracks: list[Track], video_properties: VideoInfo) -> list[Track]:
        """Greedily stiches detections onto tracks as long as both IOU and cosine similarity are high."""
        logging.info(f'{"=" * 30} Running greedy preprocessor with {len(tracks)} tracks {"=" * 30}')

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
        for track in tracks:
            assert len(track.sorted_detections) == 1, 'Greedy preprocessor only supports single-detection tracks'
            for det in track.sorted_detections:
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
                    comparison_result = self._compare_detection_to_track(
                        track, detection, [tr for tr in active_tracks if tr != track]
                    )
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
                if track.end.frame_idx + self.max_frame_distance < frame_idx:
                    if track.track_id not in stale_track_ids:
                        stale_track_ids.add(track.track_id)
                        stale_tracks.append(track)
            # update stale / active tracks once per frame
            active_tracks = [track for track in active_tracks if track not in stale_tracks]

        return stale_tracks + active_tracks

    def _compare_detection_to_track(
        self, track: Track, detection: Detection, all_other_tracks: list[Track]
    ) -> _ComparisonResult:
        iou = track.end.bbox.iou(detection.bbox)

        # if iou < self.min_iou_matches_single_track:
        #     return _ComparisonResult.NO_MATCH

        ema_embedding = track.start.embedding
        for d in track.sorted_detections:
            ema_embedding = self.ema_alpha * ema_embedding + (1 - self.ema_alpha) * d.embedding
        # average_sim = sum(cosine_similarity(d.embedding, detection.embedding) for d in track.sorted_detections) / len(track.sorted_detections)
        average_sim = cosine_similarity(ema_embedding, detection.embedding)

        if average_sim < self.min_cosine_similarity:
            return _ComparisonResult.NO_MATCH

        if iou >= self.min_iou:
            return _ComparisonResult.MATCH

        # if the bbox is so far away from all other tracks, we can match it
        if all(
            detection.bbox.iou(other_track.end.bbox) <= 1e-6
            and detection.bbox.center.distance_to(other_track.end.bbox.center) > detection.bbox.width * 2.0
            for other_track in all_other_tracks
        ):
            return _ComparisonResult.MAY_MATCH

        return _ComparisonResult.NO_MATCH  # TODO change back to MAY_MATCH?
