from __future__ import annotations

import math
import numpy as np
import cv2

from dataclasses import dataclass
from typing import Iterator


def compute_color_histogram(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    """
    Compute HSV color histogram for a bounding box region, relative to the entire image.

    Args:
        image: BGR image array (H, W, 3)
        bbox: BoundingBox object defining the region

    Returns:
        Difference histogram (bbox_hist - image_hist) with 256+16+8 = 280 total values
    """
    # Extract the region of interest
    roi = image[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]

    # Handle empty or invalid ROI
    if roi.size == 0:
        return np.zeros(256 + 16 + 8)

    # Convert both ROI and full image to HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_full = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Compute histograms for ROI
    hist_h_roi = cv2.calcHist([hsv_roi], [0], None, [256], [0, 256])
    hist_s_roi = cv2.calcHist([hsv_roi], [1], None, [16], [0, 256])
    hist_v_roi = cv2.calcHist([hsv_roi], [2], None, [8], [0, 256])

    # Compute histograms for full image
    hist_h_full = cv2.calcHist([hsv_full], [0], None, [256], [0, 256])
    hist_s_full = cv2.calcHist([hsv_full], [1], None, [16], [0, 256])
    hist_v_full = cv2.calcHist([hsv_full], [2], None, [8], [0, 256])

    # Normalize histograms to [0, 1] range
    hist_h_roi = hist_h_roi.flatten()
    hist_s_roi = hist_s_roi.flatten()
    hist_v_roi = hist_v_roi.flatten()

    hist_h_full = hist_h_full.flatten()
    hist_s_full = hist_s_full.flatten()
    hist_v_full = hist_v_full.flatten()

    # Normalize ROI histograms
    if hist_h_roi.sum() > 0:
        hist_h_roi = hist_h_roi / hist_h_roi.sum()
    if hist_s_roi.sum() > 0:
        hist_s_roi = hist_s_roi / hist_s_roi.sum()
    if hist_v_roi.sum() > 0:
        hist_v_roi = hist_v_roi / hist_v_roi.sum()

    # Normalize full image histograms
    if hist_h_full.sum() > 0:
        hist_h_full = hist_h_full / hist_h_full.sum()
    if hist_s_full.sum() > 0:
        hist_s_full = hist_s_full / hist_s_full.sum()
    if hist_v_full.sum() > 0:
        hist_v_full = hist_v_full / hist_v_full.sum()

    # Compute difference histograms (ROI - full image)
    hist_h_diff = hist_h_roi - hist_h_full
    hist_s_diff = hist_s_roi - hist_s_full
    hist_v_diff = hist_v_roi - hist_v_full

    # Normalize difference histograms
    if hist_h_diff.sum() > 0:
        hist_h_diff = hist_h_diff / hist_h_diff.sum()
    if hist_s_diff.sum() > 0:
        hist_s_diff = hist_s_diff / hist_s_diff.sum()
    if hist_v_diff.sum() > 0:
        hist_v_diff = hist_v_diff / hist_v_diff.sum()

    # Concatenate all difference histograms into a single feature vector
    return np.concatenate([hist_h_diff])  # TODO , hist_s_diff, hist_v_diff])
    return np.concatenate([hist_h_diff, hist_s_diff, hist_v_diff])


@dataclass
class Point:
    x: int
    y: int

    def __iter__(self) -> Iterator[int]:
        return iter((self.x, self.y))

    def distance_to(self, other: Point) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def interpolate(self, other: Point, alpha: float) -> Point:
        return Point(int((1 - alpha) * self.x + alpha * other.x), int((1 - alpha) * self.y + alpha * other.y))


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
        center = self.center.interpolate(other.center, alpha)
        width = int((1 - alpha) * self.width + alpha * other.width)
        height = int((1 - alpha) * self.height + alpha * other.height)
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


@dataclass
class Detection:
    bbox: BoundingBox
    feat: np.ndarray
    confidence: float
    frame_idx: FrameIndex
    color_histogram: np.ndarray  # HSV difference histogram: 256(H) + 16(S) + 8(V) = 280 total values

    def copy(self) -> Detection:
        return Detection(
            bbox=self.bbox.copy(),
            feat=self.feat.copy(),
            confidence=self.confidence,
            frame_idx=self.frame_idx,
            color_histogram=self.color_histogram.copy(),
        )

    def interpolate(self, other: Detection, alpha: float) -> Detection:
        new_bbox = self.bbox.interpolate(other.bbox, alpha)
        new_feat = (1 - alpha) * self.feat + alpha * other.feat
        new_confidence = (1 - alpha) * self.confidence + alpha * other.confidence
        new_frame_idx = int((1 - alpha) * self.frame_idx + alpha * other.frame_idx)
        # Interpolate color histograms
        new_color_histogram = (1 - alpha) * self.color_histogram + alpha * other.color_histogram

        return Detection(
            bbox=new_bbox,
            feat=new_feat,
            confidence=new_confidence,
            frame_idx=new_frame_idx,
            color_histogram=new_color_histogram,
        )


FrameIndex = int
TrackId = int | None


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

    def start(self) -> Detection:
        """Return the first detection in the track."""
        if not self.sorted_detections:
            raise ValueError('Track has no detections.')
        return self.sorted_detections[0]

    def end(self) -> Detection:
        """Return the last detection in the track."""
        if not self.sorted_detections:
            raise ValueError('Track has no detections.')
        return self.sorted_detections[-1]

    def start_frame(self) -> int:
        """Return the frame index of the first detection in the track."""
        return self.start().frame_idx

    def end_frame(self) -> int:
        """Return the frame index of the last detection in the track."""
        return self.end().frame_idx

    def alive_at_frame(self, frame_idx: FrameIndex) -> bool:
        """Check if the track has a detection at the given frame index."""
        return frame_idx >= self.start_frame() and frame_idx <= self.end_frame()

    def get_most_recent_detection_at_frame(self, frame_idx: FrameIndex) -> Detection:
        """Get the most recent detection at the given frame index."""
        if not self.alive_at_frame(frame_idx):
            raise ValueError(f'Track {self.track_id} is not alive at frame {frame_idx}.')
        for i in range(frame_idx, self.start_frame() - 1, -1):
            if i in self.detections_by_frame:
                return self.detections_by_frame[i]
        assert False


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def histogram_similarity(det1: Detection, det2: Detection) -> float:
    """
    Compute similarity between two detections based on their color histograms.

    Args:
        det1: First detection
        det2: Second detection

    Returns:
        Cosine similarity between the color histograms (0-1 range)
    """
    return cosine_similarity(det1.color_histogram, det2.color_histogram)
