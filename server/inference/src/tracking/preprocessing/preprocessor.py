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
        greedy_min_iou: float = GREEDY_PREPROCESSOR_MIN_IOU,
        greedy_min_cosine_similarity: float = GREEDY_PREPROCESSOR_MIN_COSINE_SIMILARITY,
        greedy_max_frame_distance: int = GREEDY_PREPROCESSOR_MAX_FRAME_DISTANCE,
        greedy_ema_alpha: float = GREEDY_PREPROCESSOR_EMA_ALPHA,
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

    def track(self, tracks: list[Track], video_properties: VideoInfo) -> list[Track]:
        tracks = self.greedy_track_stitcher.track(tracks, video_properties)
        # tracks = self.filter_non_surfers.track(tracks, video_properties)
        return tracks
