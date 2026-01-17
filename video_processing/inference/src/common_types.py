from __future__ import annotations

from functools import cache
import math
import numpy as np

from dataclasses import dataclass
from typing import Iterator

from .util.similarity_helpers import Embedding


def interpolate(a: float, b: float, alpha: float) -> float:
    return (1 - alpha) * a + alpha * b


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def __iter__(self) -> Iterator[int]:
        return iter((self.x, self.y))

    def distance_to(self, other: Point) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def interpolate(self, other: Point, alpha: float) -> Point:
        return Point(int(interpolate(self.x, other.x, alpha)), int(interpolate(self.y, other.y, alpha)))

    def __mul__(self, other: float) -> Point:
        return Point(int(self.x * other), int(self.y * other))

    def __truediv__(self, other: float) -> Point:
        return Point(int(self.x / other), int(self.y / other))

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)

    def __neg__(self) -> Point:
        return Point(-self.x, -self.y)

    def __abs__(self) -> Point:
        return Point(abs(self.x), abs(self.y))

    def __bool__(self) -> bool:
        return self.x != 0 or self.y != 0


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self):
        assert int(self.x1) <= int(self.x2) and int(self.y1) <= int(self.y2), (
            f'Bounding boxes must be valid ({self.x1}<={self.x2}, {self.y1}<={self.y2})'
        )
        assert (
            isinstance(self.x1, int)
            and isinstance(self.y1, int)
            and isinstance(self.x2, int)
            and isinstance(self.y2, int)
        ), 'Bounding boxes must be integers'

    @staticmethod
    def from_center_wh(cx: float, cy: float, w: float, h: float) -> BoundingBox:
        return BoundingBox(int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Point:
        return Point(int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2))

    @property
    def diagonal_length(self) -> float:
        return math.sqrt(self.width**2 + self.height**2)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center_wh(self) -> np.ndarray:
        return np.array([self.center.x, self.center.y, self.width, self.height], dtype=np.float64)

    def __iter__(self) -> Iterator[int]:
        return iter((self.x1, self.y1, self.x2, self.y2))

    def copy(self) -> BoundingBox:
        return BoundingBox(self.x1, self.y1, self.x2, self.y2)

    def interpolate(self, other: BoundingBox, alpha: float) -> BoundingBox:
        center = self.center.interpolate(other.center, alpha)
        width = int(interpolate(self.width, other.width, alpha))
        height = int(interpolate(self.height, other.height, alpha))
        return BoundingBox(
            center.x - width // 2,
            center.y - height // 2,
            center.x + width // 2,
            center.y + height // 2,
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

    def scale(self, scale_factor: float) -> BoundingBox:
        cx, cy = self.center
        return BoundingBox(
            int(cx - self.width * scale_factor / 2),
            int(cy - self.height * scale_factor / 2),
            int(cx + self.width * scale_factor / 2),
            int(cy + self.height * scale_factor / 2),
        )

    def clamp(self, min_x: int, min_y: int, max_x: int, max_y: int) -> BoundingBox:
        return BoundingBox(
            max(min_x, self.x1),
            max(min_y, self.y1),
            min(max_x, self.x2),
            min(max_y, self.y2),
        )

    def __hash__(self) -> int:
        return hash((self.x1, self.y1, self.x2, self.y2))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoundingBox):
            return False
        return self.x1 == other.x1 and self.y1 == other.y1 and self.x2 == other.x2 and self.y2 == other.y2

    def __add__(self, other: BoundingBox) -> BoundingBox:
        return BoundingBox(
            self.x1 + other.x1,
            self.y1 + other.y1,
            self.x2 + other.x2,
            self.y2 + other.y2,
        )

    def __mul__(self, other: float) -> BoundingBox:
        return BoundingBox(
            int(self.x1 * other),
            int(self.y1 * other),
            int(self.x2 * other),
            int(self.y2 * other),
        )

    def __truediv__(self, other: float) -> BoundingBox:
        return BoundingBox(
            int(self.x1 / other),
            int(self.y1 / other),
            int(self.x2 / other),
            int(self.y2 / other),
        )


@dataclass
class Detection:
    bbox: BoundingBox
    embedding: Embedding
    confidence: float
    frame_idx: FrameIndex
    interpolated: bool = False

    def copy(self) -> Detection:
        return Detection(
            bbox=self.bbox.copy(),
            embedding=self.embedding,
            confidence=self.confidence,
            frame_idx=self.frame_idx,
            interpolated=self.interpolated,
        )

    def interpolate(self, other: Detection, alpha: float, frame_idx: FrameIndex) -> Detection:
        new_bbox = self.bbox.interpolate(other.bbox, alpha)
        new_embedding = self.embedding.interpolate(other.embedding, alpha)
        new_confidence = interpolate(self.confidence, other.confidence, alpha)

        return Detection(
            bbox=new_bbox,
            embedding=new_embedding,
            confidence=new_confidence,
            frame_idx=frame_idx,
            interpolated=True,
        )

    def __hash__(self) -> int:
        return hash((self.bbox, self.confidence, self.frame_idx))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Detection):
            return False
        return self.bbox == other.bbox and self.confidence == other.confidence and self.frame_idx == other.frame_idx


FrameIndex = int
TrackId = int


@dataclass
class Track:
    track_id: TrackId
    sorted_detections: list[Detection]

    def __init__(self, track_id: TrackId, sorted_detections: list[Detection]):
        self.track_id = track_id
        self.sorted_detections = sorted(sorted_detections, key=lambda d: d.frame_idx)

    # @Perf Because sorted_detections is not static. Leave this until it becomes an issue (simple > perf)
    @property
    def detections_by_frame(self) -> dict[FrameIndex, Detection]:
        return {d.frame_idx: d for d in self.sorted_detections}

    def copy(self) -> Track:
        new_sorted_detections = [d.copy() for d in self.sorted_detections]
        return Track(
            track_id=self.track_id,
            sorted_detections=new_sorted_detections,
        )

    @property
    def start(self) -> Detection:
        """Return the first detection in the track."""
        if not self.sorted_detections:
            raise ValueError('Track has no detections.')
        return self.sorted_detections[0]

    @property
    def end(self) -> Detection:
        """Return the last detection in the track."""
        if not self.sorted_detections:
            raise ValueError('Track has no detections.')
        return self.sorted_detections[-1]

    @property
    def start_frame(self) -> FrameIndex:
        """Return the frame index of the first detection in the track."""
        return self.start.frame_idx

    @property
    def end_frame(self) -> FrameIndex:
        """Return the frame index of the last detection in the track."""
        return self.end.frame_idx

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    def alive_at_frame(self, frame_idx: FrameIndex) -> bool:
        """Check if the track has a detection at the given frame index."""
        return frame_idx >= self.start_frame and frame_idx <= self.end_frame

    def get_most_recent_detection_at_frame(self, frame_idx: FrameIndex) -> Detection:
        """Get the most recent detection at the given frame index."""
        if not self.alive_at_frame(frame_idx):
            raise ValueError(f'Track {self.track_id} is not alive at frame {frame_idx}.')
        for i in range(frame_idx, self.start_frame - 1, -1):
            if i in self.detections_by_frame:
                return self.detections_by_frame[i]
        assert False

    @cache
    def mean_embedding(self, ema: float = 0.9) -> Embedding:
        assert self.sorted_detections, 'Track has no detections.'
        embedding: Embedding = self.sorted_detections[0].embedding
        for detection in self.sorted_detections[1:]:
            embedding = embedding.interpolate(detection.embedding, ema)  # type: ignore
        return embedding

        return Embedding.mean([d.embedding for d in self.sorted_detections])

    def mean_embedding_reverse(self, ema: float = 0.9) -> Embedding:
        assert self.sorted_detections, 'Track has no detections.'
        embedding: Embedding = self.sorted_detections[-1].embedding
        for detection in self.sorted_detections[-2::-1]:
            embedding = embedding.interpolate(detection.embedding, ema)  # type: ignore
        return embedding

    def frame_gap(self, other: Track) -> int:
        return other.start_frame - self.end_frame - 1

    def __hash__(self) -> int:
        return hash((self.track_id, tuple(self.sorted_detections)))
