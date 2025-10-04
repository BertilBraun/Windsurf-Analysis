from ...util.video_io import VideoInfo
from ...common_types import Track
from ...settings import (
    GREEDY_PREPROCESSOR_MIN_IOU,
    GREEDY_PREPROCESSOR_MIN_COSINE_SIMILARITY,
    GREEDY_PREPROCESSOR_MAX_FRAME_DISTANCE,
    GREEDY_PREPROCESSOR_EMA_ALPHA,
)
from .greedy_track_stitcher import GreedyTrackStitcher
# from .filter_non_surfers import FilterNonSurfers


class Preprocessor:
    def __init__(
        self,
        appearance_strict: float = 0.05,
        appearance_loose: float = 0.15,
        motion_strict: float = 0.2,
        motion_loose: float = 5.0,
        max_frame_distance: int = 6,  # stale cutoff (frames)
        ema_alpha: float = 0.6,  # appearance EMA smoothing
        # non_surfer_min_frames: int = 5,
        # non_surfer_similarity_thresh: float = 0.8,
    ):
        self.greedy_track_stitcher = GreedyTrackStitcher(
            greedy_min_iou=greedy_min_iou,
            greedy_min_cosine_similarity=greedy_min_cosine_similarity,
            greedy_max_frame_distance=greedy_max_frame_distance,
            greedy_ema_alpha=greedy_ema_alpha,
        )
        # self.filter_non_surfers = FilterNonSurfers(
        #     min_frames=non_surfer_min_frames,
        #     similarity_thresh=non_surfer_similarity_thresh,
        # )

    def track(self, tracks: list[Track], video_properties: VideoInfo, transforms: list[Transform]) -> list[Track]:
        tracks = self.greedy_track_stitcher.track(tracks, video_properties, transforms)
        # TODO tracks = OldGreedyTrackStitcher().track(tracks, video_properties)
        # tracks = self.filter_non_surfers.track(tracks, video_properties)
        return tracks
