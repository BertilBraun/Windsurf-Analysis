from __future__ import annotations

import math

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


@dataclass
class Detection:
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None


@dataclass
class Track:
    frame_idx: int
    bbox: BoundingBox
    confidence: float
    embedding: list[float]

    def copy(self) -> Track:
        return Track(
            self.frame_idx,
            self.bbox.copy(),
            self.confidence,
            self.embedding.copy(),
        )

    def histogram_similarity(self, other: Track) -> float:
        """Calculate similarity between two tracks based on their hue histograms.

        Uses Bhattacharyya coefficient for histogram comparison.

        Args:
            other: Another Track to compare against

        Returns:
            Similarity score between 0.0 (no similarity) and 1.0 (identical)
        """
        if len(self.embedding) != len(other.embedding):
            raise ValueError('Histograms must have the same number of bins')

        # Bhattacharyya coefficient: sum of sqrt(p_i * q_i) for all bins
        similarity = sum(math.sqrt(self.embedding[i] * other.embedding[i]) for i in range(len(self.embedding)))

        return similarity

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

        # Interpolate hue histogram (bin by bin)
        interpolated_histogram = [
            (1 - alpha) * self.embedding[i] + alpha * other.embedding[i] for i in range(len(self.embedding))
        ]

        return Track(interpolated_frame_idx, interpolated_bbox, interpolated_confidence, interpolated_histogram)


TrackId = int | None
TrackerInput = dict[TrackId, list[Track]]
