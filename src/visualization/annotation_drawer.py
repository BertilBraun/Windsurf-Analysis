import cv2
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

from common_types import BoundingBox, TrackId


@dataclass
class Annotation:
    track_id: TrackId
    bbox: BoundingBox
    confidence: float


class AnnotationDrawer:
    """Handles drawing annotations and tracking trails on video frames"""

    # Define a palette of easily distinguishable colors (BGR format for OpenCV)
    COLOR_PALETTE = [
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (0, 0, 255),  # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 255),  # Orange-red
        (128, 255, 0),  # Lime
        (255, 0, 128),  # Pink
        (0, 255, 128),  # Spring green
        (128, 255, 255),  # Light yellow
        (255, 128, 255),  # Light magenta
        (255, 255, 128),  # Light cyan
        (64, 64, 255),  # Light red
        (255, 64, 64),  # Light blue
        (64, 255, 64),  # Light green
        (192, 192, 192),  # Light gray
        (128, 128, 0),  # Olive
    ]

    def __init__(self, max_track_length: int = 30):
        self.track_history: dict[TrackId, list[tuple[float, float]]] = defaultdict(list)
        self.max_track_length = max_track_length
        self.track_colors: dict[TrackId, tuple[int, int, int]] = {}

    def _get_track_color(self, track_id: TrackId) -> tuple[int, int, int]:
        """Get or assign a unique color for a track ID"""
        if track_id not in self.track_colors:
            color_index = len(self.track_colors) % len(self.COLOR_PALETTE)
            self.track_colors[track_id] = self.COLOR_PALETTE[color_index]
        return self.track_colors[track_id]

    def draw_detections_with_trails(self, frame: np.ndarray, annotations: list[Annotation]) -> np.ndarray:
        """Draw detection bounding boxes, labels, and tracking trails on a frame"""
        annotated_frame = frame.copy()

        # Draw tracking trails first (so they appear behind boxes)
        annotated_frame = self._draw_tracking_trails(annotated_frame, annotations)

        annotated_frame = self._draw_detections_only(annotated_frame, annotations)

        return annotated_frame

    def _draw_tracking_trails(self, frame: np.ndarray, annotations: list[Annotation]) -> np.ndarray:
        """Draw tracking trails for detected objects"""
        annotated_frame = frame.copy()

        for annotation in annotations:
            track = self.track_history[annotation.track_id]
            color = self._get_track_color(annotation.track_id)

            # Add current position to track
            track.append((float(annotation.bbox.center.x), float(annotation.bbox.center.y)))

            # Limit track history length
            if len(track) > self.max_track_length:
                track.pop(0)

            # Draw tracking trail with track-specific color (lighter version)
            if len(track) > 1:
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                # Make trail color lighter by adding white
                trail_color = tuple(min(255, int(c * 0.7 + 255 * 0.3)) for c in color)
                cv2.polylines(annotated_frame, [points], isClosed=False, color=trail_color, thickness=3)

        return annotated_frame

    def _draw_detections_only(self, frame: np.ndarray, annotations: list[Annotation]) -> np.ndarray:
        """Draw only detection bounding boxes and labels (no trails)"""
        annotated_frame = frame.copy()

        for annotation in annotations:
            x1, y1, x2, y2 = annotation.bbox
            color = self._get_track_color(annotation.track_id)

            # Draw bounding box with track-specific color
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Prepare label text
            label = f'{annotation.confidence:.2f} ID:{annotation.track_id}'

            # Draw label background with track-specific color
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)

            # Draw label text (use white or black text based on color brightness for readability)
            brightness = sum(color) / 3
            text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        return annotated_frame
