from __future__ import annotations

import numpy as np

from video_processing.inference.src.common_types import BoundingBox, Detection, Keypoint, Point, Track
from video_processing.inference.src.tracking.ilp_tracker import (
    ILPTracker,
    _bbox_shape_ratios,
    _robust_fragment_embedding,
)
from video_processing.inference.src.tracking.reid.ReIDColorABStripeHistogram import ReIDColorABStripeHistogram
from video_processing.inference.src.util.similarity_helpers import HellingerEmbedding


def _detection(frame_index: int, embedding: HellingerEmbedding) -> Detection:
    hidden_keypoint = Keypoint(Point(0, 0), 0.0)
    return Detection(
        bbox=BoundingBox(10, 10, 30, 50),
        embedding=embedding,
        confidence=1.0,
        frame_idx=frame_index,
        boom=hidden_keypoint,
        mast_tip=hidden_keypoint,
    )


def test_foreground_mask_excludes_border_background() -> None:
    crop = np.full((100, 100, 3), (180, 120, 60), dtype=np.uint8)
    crop[20:80, 35:65] = (0, 0, 255)

    mask = ReIDColorABStripeHistogram()._compute_mask(crop)

    assert mask.dtype == np.bool_
    assert mask[50, 50]
    assert not mask[5, 5]


def test_foreground_mask_can_be_disabled() -> None:
    crop = np.zeros((20, 30, 3), dtype=np.uint8)

    mask = ReIDColorABStripeHistogram(use_mask=False)._compute_mask(crop)

    assert mask.all()


def test_robust_fragment_embedding_trims_appearance_outlier() -> None:
    primary = HellingerEmbedding(np.array([1.0, 0.0], dtype=np.float32))
    nearby = HellingerEmbedding(np.array([0.99, 0.01], dtype=np.float32))
    outlier = HellingerEmbedding(np.array([0.0, 1.0], dtype=np.float32))
    track = Track(
        track_id=1,
        sorted_detections=[
            _detection(0, primary),
            _detection(1, primary),
            _detection(2, nearby),
            _detection(3, outlier),
        ],
    )

    prototype = _robust_fragment_embedding(track, keep_fraction=0.75)

    assert prototype.distance(primary) < prototype.distance(outlier)
    assert prototype.distance(primary) < 0.001


def test_ilp_motion_uses_position_and_size_by_default() -> None:
    assert not ILPTracker().use_position_only


def test_bbox_shape_ratios_are_symmetric() -> None:
    small = BoundingBox(0, 0, 10, 20)
    large = BoundingBox(0, 0, 30, 20)

    assert _bbox_shape_ratios(small, large) == _bbox_shape_ratios(large, small)
    assert _bbox_shape_ratios(small, large) == (3.0, 3.0)
