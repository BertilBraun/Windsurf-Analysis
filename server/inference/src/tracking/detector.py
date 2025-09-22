import os
import logging
from pathlib import Path
import numpy as np

current_dir = Path(__file__).parent
server_root_dir = current_dir.parent.parent.parent

os.environ['YOLO_CONFIG_DIR'] = str(server_root_dir / 'ultralytics')

from ultralytics import YOLO


from .reid import ReID
from ..settings import (
    USE_GPU,
    YOLO_MODEL_PATH,
    MIN_TRACKING_FPS,
    IOU_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    BATCH_SIZE,
)
from ..util.cache import cache_to_file
from ..util.video_io import get_video_properties
from ..common_types import BoundingBox, Detection


class SurferDetector:
    """Pure detection and tracking class for surfers in video"""

    def __init__(self, yolo_model_path: os.PathLike | str):
        logging.info(f'Using model: {yolo_model_path}')
        yolo_model_path = Path(yolo_model_path)

        if not yolo_model_path.exists():
            raise FileNotFoundError(f'YOLO model {yolo_model_path} not found')

        self.yolo_model = YOLO(model=yolo_model_path, verbose=False)
        self.reid_model = ReID()

    @cache_to_file(
        'yolo_detections',
        ignore_args=[0],
        additional_args=[YOLO_MODEL_PATH, IOU_THRESHOLD, CONFIDENCE_THRESHOLD, BATCH_SIZE],
    )
    def run_object_detection_on_video(self, video_path: str) -> list[Detection]:
        """Run batched inference on entire video and return all detections as a list.

        This performs YOLO inference in a streamed fashion but accumulates lightweight
        metadata (frame index, boxes and confidences) and then
        runs ReID in larger batches across frames to improve GPU utilization.
        """

        video_props = get_video_properties(video_path)
        skip_frames = max(1, video_props.fps // MIN_TRACKING_FPS)

        results = self.yolo_model.predict(
            str(video_path),
            iou=IOU_THRESHOLD,
            conf=CONFIDENCE_THRESHOLD,
            batch=BATCH_SIZE,
            vid_stride=skip_frames,
            stream=True,
            save=False,
            half=USE_GPU,
            verbose=False,
        )

        # Accumulate per-detection info first (without ReID), then do ReID in batches
        pending_crops: list[np.ndarray] = []
        pending_meta: list[tuple[BoundingBox, float, int]] = []
        all_detections: list[Detection] = []

        for frame_index, result in enumerate(results):
            frame_idx = frame_index * skip_frames
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = _to_numpy(result.boxes.xyxy)
            confidences = _to_numpy(result.boxes.conf)
            orig_img = result.orig_img

            # Prepare crops and metadata
            for i in range(len(boxes)):
                bbox = BoundingBox(
                    x1=boxes[i][0],
                    y1=boxes[i][1],
                    x2=boxes[i][2],
                    y2=boxes[i][3],
                )

                # Crop for ReID (BGR)
                x1, y1, x2, y2 = map(int, (bbox.x1, bbox.y1, bbox.x2, bbox.y2))
                h, w = orig_img.shape[:2]
                x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                crop = orig_img[y1c:y2c, x1c:x2c]

                # Skip invalid crops
                if crop.size == 0:
                    continue

                bbox_clipped = BoundingBox(x1c, y1c, x2c, y2c)
                pending_crops.append(crop)
                pending_meta.append((bbox_clipped, float(confidences[i]), frame_idx))

                # If we have enough crops, flush a ReID batch
                if len(pending_crops) >= BATCH_SIZE:
                    _flush_reid_batch(self.reid_model, pending_crops, pending_meta, all_detections)

        # Flush remaining crops
        if pending_crops:
            _flush_reid_batch(self.reid_model, pending_crops, pending_meta, all_detections)

        return all_detections


def _flush_reid_batch(
    reid_model: ReID,
    pending_crops: list[np.ndarray],
    pending_meta: list[tuple[BoundingBox, float, int]],
    output_detections: list[Detection],
) -> None:
    """Encode a batch of crops with ReID and append as Detection objects.

    pending_meta must align 1:1 with pending_crops and contain:
      (bbox, confidence, frame_idx)
    """
    if not pending_crops:
        return

    features = reid_model.get_features_for_crops(pending_crops)
    assert features.shape[0] == len(pending_meta)

    for i, (bbox, confidence, frame_idx) in enumerate(pending_meta):
        embedding = features[i]
        # Ensure numerical safety: normalize if slightly off
        norm = np.linalg.norm(embedding)
        if norm > 0 and abs(norm - 1.0) > 1e-3:
            embedding = embedding / norm

        output_detections.append(
            Detection(
                bbox=bbox,
                embedding=embedding,
                confidence=confidence,
                frame_idx=frame_idx,
            )
        )

    pending_crops.clear()
    pending_meta.clear()


def _to_numpy(tensor_or_array):
    """Convert PyTorch tensor or array-like object to numpy array"""
    try:
        # Try PyTorch tensor conversion first
        return tensor_or_array.cpu().numpy()
    except AttributeError:
        # Fall back to numpy array conversion
        return np.array(tensor_or_array)
