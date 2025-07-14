import os
import numpy as np
from typing import Generator, Iterator
from dataclasses import dataclass

from ultralytics import YOLO
from ultralytics.engine.results import Results

DEFAULT_MODEL_NAME = 'yolo11n.pt'

IOU_THRESHOLD = 0.2
CONFIDENCE_THRESHOLD = 0.1
BATCH_SIZE = 32


def _to_numpy(tensor_or_array):
    """Convert PyTorch tensor or array-like object to numpy array"""
    try:
        # Try PyTorch tensor conversion first
        return tensor_or_array.cpu().numpy()
    except AttributeError:
        # Fall back to numpy array conversion
        return np.array(tensor_or_array)


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2)

    def __iter__(self) -> Iterator[int]:
        return iter((self.x1, self.y1, self.x2, self.y2))


@dataclass
class Detection:
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None


class SurferDetector:
    """Pure detection and tracking class for surfers in video"""

    def __init__(self):
        self.model = YOLO(DEFAULT_MODEL_NAME)

    def detect_and_track_video(self, video_path: os.PathLike) -> Generator[list[Detection], None, None]:
        """Run batched inference on entire video, return generator of (frame, detections)"""
        results = self.model.track(
            str(video_path),
            iou=IOU_THRESHOLD,
            conf=CONFIDENCE_THRESHOLD,
            batch=BATCH_SIZE,
            persist=True,
            stream=True,
            verbose=False,
        )

        for result in results:
            yield self._extract_detections(result)

    def _extract_detections(self, result: Results) -> list[Detection]:
        """Extract detection information for further processing"""
        detections: list[Detection] = []

        if result.boxes is not None:
            # Convert tensors to numpy arrays using utility function
            boxes = _to_numpy(result.boxes.xyxy)
            confidences = _to_numpy(result.boxes.conf)
            class_ids = _to_numpy(result.boxes.cls)

            track_ids = None
            if result.boxes.id is not None:
                track_ids = _to_numpy(result.boxes.id)

            for i in range(len(boxes)):
                detection = Detection(
                    bbox=BoundingBox(
                        x1=boxes[i][0],
                        y1=boxes[i][1],
                        x2=boxes[i][2],
                        y2=boxes[i][3],
                    ),
                    confidence=float(confidences[i]),
                    class_id=int(class_ids[i]),
                    class_name=result.names[int(class_ids[i])],
                    track_id=int(track_ids[i]) if track_ids is not None else None,
                )
                detections.append(detection)

        return detections
