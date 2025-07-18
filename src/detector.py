import os
import logging
import numpy as np
from typing import Generator

from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

from video_io import get_video_properties
import settings
from common_types import BoundingBox, Detection


def log_detection_settings():
    settings_str = '\n'.join(
        f'{k}: {v}' for k, v in settings.__dict__.items() if not k.startswith('__') and not callable(v) and k.isupper()
    )
    logging.info(f'Detection settings: \n{settings_str}')


class SurferDetector:
    """Pure detection and tracking class for surfers in video"""

    def __init__(self):
        logging.info(f'Using model: {settings.YOLO_MODEL_PATH}')
        if not settings.YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(f'Model {settings.YOLO_MODEL_PATH} not found')

        self.model = YOLO(settings.YOLO_MODEL_PATH, verbose=False)

        self.model.add_callback('on_predict_start', self._on_predict_start)

        log_detection_settings()

    def detect_and_track_video(
        self, video_path: os.PathLike | str
    ) -> Generator[tuple[int, np.ndarray, list[Detection]], None, None]:
        """Run batched inference on entire video, return generator of (frame, detections)"""

        video_props = get_video_properties(video_path)
        skip_frames = video_props.fps // settings.MIN_TRACKING_FPS

        results = self.model.predict(
            str(video_path),
            iou=settings.IOU_THRESHOLD,
            conf=settings.CONFIDENCE_THRESHOLD,
            batch=settings.BATCH_SIZE,
            vid_stride=skip_frames,
            stream=True,
            verbose=False,
        )

        for frame_index, result in tqdm(
            enumerate(results), total=video_props.total_frames // skip_frames, desc='Processing video'
        ):
            yield frame_index * skip_frames, result.orig_img, self._extract_detections(result)

    def _extract_detections(self, result: Results) -> list[Detection]:
        """Extract detection information for further processing"""
        detections: list[Detection] = []

        if result.boxes is not None:
            # Convert tensors to numpy arrays using utility function
            boxes = _to_numpy(result.boxes.xyxy)
            confidences = _to_numpy(result.boxes.conf)
            feats = _to_numpy(result.feats)

            for i in range(len(boxes)):
                detection = Detection(
                    bbox=BoundingBox(
                        x1=boxes[i][0],
                        y1=boxes[i][1],
                        x2=boxes[i][2],
                        y2=boxes[i][3],
                    ),
                    feat=feats[i],
                    confidence=confidences[i],
                )
                detections.append(detection)

        return detections

    def _on_predict_start(self, predictor: object) -> None:
        """Initialize trackers for object tracking during prediction.

        Args:
            predictor (ultralytics.engine.predictor.BasePredictor): The predictor object to initialize trackers for.
        """
        predictor._feats = None  # type: ignore  # reset in case used earlier

        # Register hook to extract input of Detect layer
        def pre_hook(module, input):
            predictor._feats = list(input[0])  # type: ignore  # unroll to new list to avoid mutation in forward

        predictor._hook = predictor.model.model.model[-1].register_forward_pre_hook(pre_hook)  # type: ignore


def _to_numpy(tensor_or_array):
    """Convert PyTorch tensor or array-like object to numpy array"""
    try:
        # Try PyTorch tensor conversion first
        return tensor_or_array.cpu().numpy()
    except AttributeError:
        # Fall back to numpy array conversion
        return np.array(tensor_or_array)
