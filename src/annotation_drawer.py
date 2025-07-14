import cv2
import numpy as np
from collections import defaultdict
from detector import Detection


class AnnotationDrawer:
    """Handles drawing annotations and tracking trails on video frames"""

    def __init__(self, max_track_length: int = 30):
        self.track_history = defaultdict(lambda: [])
        self.max_track_length = max_track_length

    def draw_detections_only(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw only detection bounding boxes and labels (no trails)"""
        annotated_frame = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Prepare label text
            label = f'{detection.class_name}: {detection.confidence:.2f}'
            if detection.track_id is not None:
                label += f' ID:{detection.track_id}'

            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)

            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return annotated_frame

    def draw_detections_with_trails(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw detection bounding boxes, labels, and tracking trails on a frame"""
        annotated_frame = frame.copy()

        # Draw tracking trails first (so they appear behind boxes)
        self._draw_tracking_trails(annotated_frame, detections)

        return self.draw_detections_only(annotated_frame, detections)

    def _draw_tracking_trails(self, frame: np.ndarray, detections: list[Detection]) -> None:
        """Draw tracking trails for detected objects"""
        for detection in detections:
            if detection.track_id is not None:
                # Convert bbox center for trail
                x1, y1, x2, y2 = detection.bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                track = self.track_history[detection.track_id]

                # Add current position to track
                track.append((float(center_x), float(center_y)))

                # Limit track history length
                if len(track) > self.max_track_length:
                    track.pop(0)

                # Draw tracking trail
                if len(track) > 1:
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)
