import os
import logging
from pathlib import Path
from typing import Literal
import numpy as np
from dataclasses import dataclass

current_dir = Path(__file__).parent
server_root_dir = current_dir.parent.parent.parent

os.environ['YOLO_CONFIG_DIR'] = str(server_root_dir / 'ultralytics')

from ultralytics import YOLO


from .reid import ReID, ReIDColorHistogram, ReIDViT, ReIDOSNet
from ..settings import (
    USE_GPU,
    YOLO_MODEL_PATH,
    MIN_TRACKING_FPS,
    DETECTOR_IOU_THRESHOLD,
    DETECTOR_CONFIDENCE_THRESHOLD,
    DETECTOR_BATCH_SIZE,
    OSNET_REID_MODEL_PATH,
    REID_MODEL_TYPE,
)
from ..util.cache import cache_to_file
from ..util.video_io import get_video_properties
from ..common_types import BoundingBox, Detection, FrameIndex


@dataclass
class _RawDetection:
    bbox: BoundingBox
    confidence: float
    frame_idx: FrameIndex
    crop: np.ndarray


class SurferDetector:
    """Pure detection and tracking class for surfers in video"""

    def __init__(self, yolo_model_path: os.PathLike | str):
        logging.info(f'Using model: {yolo_model_path}')
        yolo_model_path = Path(yolo_model_path)

        if not yolo_model_path.exists():
            raise FileNotFoundError(f'YOLO model {yolo_model_path} not found')

        self.yolo_model = YOLO(model=yolo_model_path, verbose=False)
        self.reid_model = _init_reid_model(REID_MODEL_TYPE)

    def run_object_detection_on_video(self, video_path: str) -> list[Detection]:
        """Two-pass pipeline: cached YOLO detection+crops, then cached ReID features."""
        raw_detections = self.run_detection_pass(video_path)
        return self.run_reid_pass(raw_detections)

    @cache_to_file(
        'yolo_detections_raw',
        ignore_args=[0],
        additional_args=[
            YOLO_MODEL_PATH,
            DETECTOR_IOU_THRESHOLD,
            DETECTOR_CONFIDENCE_THRESHOLD,
            DETECTOR_BATCH_SIZE,
            MIN_TRACKING_FPS,
        ],
    )
    def run_detection_pass(self, video_path: str) -> list[_RawDetection]:
        """Run YOLO once and persist crops+metadata. Returns raw detections."""

        video_props = get_video_properties(video_path)
        skip_frames = max(1, video_props.fps // MIN_TRACKING_FPS)

        results = self.yolo_model.predict(
            str(video_path),
            iou=DETECTOR_IOU_THRESHOLD,
            conf=DETECTOR_CONFIDENCE_THRESHOLD,
            batch=DETECTOR_BATCH_SIZE,
            vid_stride=skip_frames,
            stream=True,
            save=False,
            half=USE_GPU,
            verbose=False,
        )

        raw_detections: list[_RawDetection] = []

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

                h, w = orig_img.shape[:2]
                bbox = bbox.clamp(0, 0, w, h)

                if bbox.area <= 0:  # Skip invalid crops
                    continue

                raw_detections.append(
                    _RawDetection(
                        bbox=bbox,
                        confidence=float(confidences[i]),
                        crop=orig_img[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2],
                        frame_idx=frame_idx,
                    )
                )

        return raw_detections

    @cache_to_file('reid_features', ignore_args=[0], additional_args=[REID_MODEL_TYPE])
    def run_reid_pass(self, raw_detections: list[_RawDetection]) -> list[Detection]:
        """Compute embeddings for saved crops based on current ReID model.

        Cached by (REID_MODEL_TYPE, det_key) so changing ReID invalidates only this pass.
        """

        # Batch crops for efficiency
        all_detections: list[Detection] = []
        pending_crops: list[np.ndarray] = []
        pending_meta: list[tuple[BoundingBox, float, int]] = []

        for rd in raw_detections:
            crop = rd.crop
            if crop is None or crop.size == 0:
                continue
            pending_crops.append(crop)
            pending_meta.append((rd.bbox, rd.confidence, rd.frame_idx))
            if len(pending_crops) >= DETECTOR_BATCH_SIZE:
                _flush_reid_batch(self.reid_model, pending_crops, pending_meta, all_detections)

        # Flush remaining crops
        if pending_crops:
            _flush_reid_batch(self.reid_model, pending_crops, pending_meta, all_detections)

        return all_detections


def _init_reid_model(model_type: Literal['color_hist', 'osnet', 'vit']) -> ReID:
    if model_type == 'color_hist':
        return ReIDColorHistogram()
    if model_type == 'osnet':
        return ReIDOSNet(model_path=OSNET_REID_MODEL_PATH)
    if model_type == 'vit':
        return ReIDViT()
    raise ValueError(f'Unknown REID_MODEL_TYPE: {model_type}')


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
    assert len(features) == len(pending_meta)

    for i, (bbox, confidence, frame_idx) in enumerate(pending_meta):
        output_detections.append(
            Detection(
                bbox=bbox,
                embedding=features[i],
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
