from __future__ import annotations

import math
import numpy as np

from dataclasses import dataclass
from typing import Iterator


@dataclass
class Point:
    x: int
    y: int

    def __iter__(self) -> Iterator[int]:
        return iter((self.x, self.y))

    def distance_to(self, other: Point) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        assert int(x1) <= int(x2) and int(y1) <= int(y2), f'Bounding boxes must be valid ({x1}<={x2}, {y1}<={y2})'
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Point:
        return Point(int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2))

    def __iter__(self) -> Iterator[int]:
        return iter((self.x1, self.y1, self.x2, self.y2))

    def copy(self) -> BoundingBox:
        return BoundingBox(self.x1, self.y1, self.x2, self.y2)

    def interpolate(self, other: BoundingBox, alpha: float) -> BoundingBox:
        return BoundingBox(
            int((1 - alpha) * self.x1 + alpha * other.x1),
            int((1 - alpha) * self.y1 + alpha * other.y1),
            int((1 - alpha) * self.x2 + alpha * other.x2),
            int((1 - alpha) * self.y2 + alpha * other.y2),
        )

    def iou(self, other: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) with another bounding box."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection_area == 0:
            return 0.0

        self_area = self.width * self.height
        other_area = other.width * other.height
        union_area = self_area + other_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0


@dataclass
class Detection:
    bbox: BoundingBox
    feat: np.ndarray
    confidence: float
    frame_idx: FrameIndex


FrameIndex = int


@dataclass
class Track(Detection):
    track_id: int | None

    def copy(self) -> Track:
        return Track(
            bbox=self.bbox.copy(),
            confidence=self.confidence,
            track_id=self.track_id,
            feat=self.feat,
            frame_idx=self.frame_idx,
        )

    def interpolate(self, other: Track, alpha: float) -> Track:
        """Interpolate between this track and another track.

        Args:
            other: Target track to interpolate towards
            alpha: Interpolation factor (0.0 = this track, 1.0 = other track)

        Returns:
            New Track with interpolated values
        """
        # Interpolate frame index
        interpolated_frame_idx = int((1 - alpha) * self.frame_idx + alpha * other.frame_idx)

        # Interpolate bounding box
        interpolated_bbox = self.bbox.interpolate(other.bbox, alpha)

        # Interpolate confidence
        interpolated_confidence = (1 - alpha) * self.confidence + alpha * other.confidence
        interpolated_confidence *= 0.7  # Reduce confidence for interpolated detections

        # Interpolate feature
        interpolated_feat = (1 - alpha) * self.feat + alpha * other.feat

        return Track(
            frame_idx=interpolated_frame_idx,
            bbox=interpolated_bbox,
            confidence=interpolated_confidence,
            track_id=self.track_id,
            feat=interpolated_feat,
        )


TrackId = int | None
TrackerInput = dict[TrackId, list[Track]]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
