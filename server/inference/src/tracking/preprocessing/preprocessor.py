from server.inference.src.visualization.stabilize import Transform
from ...util.video_io import VideoInfo
from ...common_types import Track
from ...settings import (
    GREEDY_PREPROCESSOR_APPEARANCE_PROBABILITY_STRICT,
    GREEDY_PREPROCESSOR_APPEARANCE_PROBABILITY_LOOSE,
    GREEDY_PREPROCESSOR_MOTION_PROBABILITY_STRICT,
    GREEDY_PREPROCESSOR_MOTION_PROBABILITY_LOOSE,
    GREEDY_PREPROCESSOR_MAX_FRAME_DISTANCE,
    GREEDY_PREPROCESSOR_EMA_ALPHA,
)
from .greedy_track_stitcher import GreedyTrackStitcher
# from .filter_non_surfers import FilterNonSurfers


class TrackPreProcessor:
    def __init__(
        self,
        appearance_probability_strict: float = GREEDY_PREPROCESSOR_APPEARANCE_PROBABILITY_STRICT,
        appearance_probability_loose: float = GREEDY_PREPROCESSOR_APPEARANCE_PROBABILITY_LOOSE,
        motion_probability_strict: float = GREEDY_PREPROCESSOR_MOTION_PROBABILITY_STRICT,
        motion_probability_loose: float = GREEDY_PREPROCESSOR_MOTION_PROBABILITY_LOOSE,
        max_frame_distance: int = GREEDY_PREPROCESSOR_MAX_FRAME_DISTANCE,
        ema_alpha: float = GREEDY_PREPROCESSOR_EMA_ALPHA,
        # non_surfer_min_frames: int = 5,
        # non_surfer_similarity_thresh: float = 0.8,
        debug_video_path: str | None = None,
    ):
        self.greedy_track_stitcher = GreedyTrackStitcher(
            appearance_probability_strict=appearance_probability_strict,
            appearance_probability_loose=appearance_probability_loose,
            motion_probability_strict=motion_probability_strict,
            motion_probability_loose=motion_probability_loose,
            max_frame_distance=max_frame_distance,
            ema_alpha=ema_alpha,
            # debug_video_path=debug_video_path,
        )
        # self.filter_non_surfers = FilterNonSurfers(
        #     min_frames=non_surfer_min_frames,
        #     similarity_thresh=non_surfer_similarity_thresh,
        # )

    def track(self, tracks: list[Track], video_properties: VideoInfo, transforms: list[Transform]) -> list[Track]:
        tracks = self.greedy_track_stitcher.track(tracks, video_properties, transforms)
        # tracks = self.filter_non_surfers.track(tracks, video_properties)
        return tracks
