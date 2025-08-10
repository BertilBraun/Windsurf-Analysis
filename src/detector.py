import os
import logging
import numpy as np
from typing import Generator

from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.engine.results import Results


from reid import ReID
from settings import YOLO_MODEL_PATH, REID_MODEL_PATH, MIN_TRACKING_FPS, IOU_THRESHOLD, CONFIDENCE_THRESHOLD, BATCH_SIZE
from video_io import get_video_properties
from common_types import BoundingBox, Detection, compute_color_histogram


class SurferDetector:
    """Pure detection and tracking class for surfers in video"""

    def __init__(self):
        logging.info(f'Using model: {YOLO_MODEL_PATH}')
        if not YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(f'Model {YOLO_MODEL_PATH} not found')

        self.model = YOLO(YOLO_MODEL_PATH, verbose=False)
        self.reid_model = ReID(model_path=REID_MODEL_PATH)

    def run_object_detection_on_video(self, video_path: os.PathLike | str) -> Generator[Detection, None, None]:
        """Run batched inference on entire video, return generator of (frame, detections)"""

        video_props = get_video_properties(video_path)
        skip_frames = video_props.fps // MIN_TRACKING_FPS

        results = self.model.predict(
            str(video_path),
            iou=IOU_THRESHOLD,
            conf=CONFIDENCE_THRESHOLD,
            batch=BATCH_SIZE,
            vid_stride=skip_frames,
            stream=True,
            verbose=False,
        )

        for frame_index, result in tqdm(
            enumerate(results), total=video_props.total_frames // skip_frames, desc='Processing video'
        ):
            yield from self._extract_detections(result, frame_index * skip_frames)

    def _extract_detections(self, result: Results, frame_idx: int) -> list[Detection]:
        """Extract detection information for further processing"""

        if result.boxes is None or len(result.boxes) == 0:
            return []

        detections: list[Detection] = []

        # Convert tensors to numpy arrays using utility function
        boxes = _to_numpy(result.boxes.xyxy)
        confidences = _to_numpy(result.boxes.conf)

        # Get the original frame data for histogram computation
        orig_img = result.orig_img

        reid_feats = self.reid_model.get_features(
            boxes, orig_img
        )  # TODO run extractor batched over all frames in the results

        for i in range(len(boxes)):
            bbox = BoundingBox(
                x1=boxes[i][0],
                y1=boxes[i][1],
                x2=boxes[i][2],
                y2=boxes[i][3],
            )

            # Normalize embedding
            embedding = reid_feats[i] / np.linalg.norm(reid_feats[i])

            # Compute color histogram for this detection
            color_histogram = compute_color_histogram(orig_img, bbox)

            detection = Detection(
                bbox=bbox,
                embedding=embedding,
                confidence=confidences[i],
                frame_idx=frame_idx,
                color_histogram=color_histogram,
            )
            detections.append(detection)

        return detections


def _to_numpy(tensor_or_array):
    """Convert PyTorch tensor or array-like object to numpy array"""
    try:
        # Try PyTorch tensor conversion first
        return tensor_or_array.cpu().numpy()
    except AttributeError:
        # Fall back to numpy array conversion
        return np.array(tensor_or_array)
