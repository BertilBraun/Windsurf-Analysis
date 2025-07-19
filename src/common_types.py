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

    def overlaps(self, other):
        """Check if this bounding box overlaps with another."""
        return not (self.x2 < other.x1 or self.x1 > other.x2 or self.y2 < other.y1 or self.y1 > other.y2)


@dataclass
class Detection:
    bbox: BoundingBox
    feat: np.ndarray
    confidence: float
    frame_idx: FrameIndex

    def copy(self) -> Detection:
        return Detection(
            bbox=self.bbox.copy(),
            feat=self.feat.copy(),
            confidence=self.confidence,
            frame_idx=self.frame_idx,
        )

    def interpolate(self, other: Detection, alpha: float) -> Detection:
        new_bbox = self.bbox.interpolate(other.bbox, alpha)
        new_feat = (1 - alpha) * self.feat + alpha * other.feat
        new_confidence = (1 - alpha) * self.confidence + alpha * other.confidence
        new_frame_idx = int((1 - alpha) * self.frame_idx + alpha * other.frame_idx)

        return Detection(bbox=new_bbox, feat=new_feat, confidence=new_confidence, frame_idx=new_frame_idx)


FrameIndex = int
TrackId = int | None


@dataclass
class Track:
    track_id: TrackId
    sorted_detections: list[Detection]

    def copy(self) -> Track:
        new_sorted_detections = [d.copy() for d in self.sorted_detections]
        return Track(
            track_id=self.track_id,
            sorted_detections=new_sorted_detections,
        )

    def end(self) -> Detection:
        """Return the last detection in the track."""
        if not self.sorted_detections:
            raise ValueError("Track has no detections.")
        return self.sorted_detections[-1]

    def start(self) -> Detection:
        """Return the first detection in the track."""
        if not self.sorted_detections:
            raise ValueError("Track has no detections.")
        return self.sorted_detections[0]

    def start_frame(self) -> int:
        """Return the frame index of the first detection in the track."""
        return self.start().frame_idx

    def end_frame(self) -> int:
        """Return the frame index of the last detection in the track."""
        return self.end().frame_idx


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
