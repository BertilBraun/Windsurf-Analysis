import os
import logging
import yaml
import numpy as np
from pathlib import Path
from typing import Generator

from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

from video_io import get_video_properties
import settings
from common_types import Detection, BoundingBox


def log_detection_settings():
    settings_str = "\n".join(
        f"{k}: {v}" for k, v in settings.__dict__.items() if not k.startswith('__') and not callable(v) and k.isupper()
    )
    logging.info(f'Detection settings: \n{settings_str}')



class SurferDetector:
    """Pure detection and tracking class for surfers in video"""

    def __init__(self):
        logger = logging.getLogger(__name__)
        logger.info(f'Using model: {settings.YOLO_MODEL_PATH}')
        model_path = Path(settings.YOLO_MODEL_PATH)
        if not model_path.exists():
            logging.warning(f'Model {model_path} not found, using default model')
        self.model = YOLO(settings.YOLO_MODEL_PATH, verbose=False)
        log_detection_settings()
        self.tracker_config_file = self._write_tracker_config()

    def detect_and_track_video(
        self, video_path: os.PathLike | str
    ) -> Generator[tuple[int, np.ndarray, list[Detection]], None, None]:
        """Run batched inference on entire video, return generator of (frame, detections)"""

        video_props = get_video_properties(video_path)
        skip_frames = video_props.fps // settings.MIN_TRACKING_FPS

        results = self.model.track(
            str(video_path),
            iou=settings.IOU_THRESHOLD,
            conf=settings.CONFIDENCE_THRESHOLD,
            batch=settings.BATCH_SIZE,
            vid_stride=skip_frames,
            tracker=str(self.tracker_config_file),
            stream=True,
            verbose=False,
        )

        for frame_index, result in tqdm(
            enumerate(results), total=video_props.total_frames // skip_frames, desc='Processing video'
        ):
            yield frame_index * skip_frames, result.orig_img, self._extract_detections(result)

    def _write_tracker_config(self) -> Path:
        file_name = Path(__file__).parent / 'botsort.yaml'
        yaml_content = {
            'tracker_type': 'botsort',
            'track_high_thresh': settings.BOTS_TRACK_HIGH_THRESH,  # threshold for the first association
            'track_low_thresh': settings.BOTS_TRACK_LOW_THRESH,  # threshold for the second association
            'new_track_thresh': settings.BOTS_NEW_TRACK_THRESH,  # threshold for init new track if the detection does not match any tracks
            'track_buffer': settings.BOTS_TRACK_BUFFER,  # buffer to calculate the time when to remove tracks
            'match_thresh': settings.BOTS_MATCH_THRESH,  # threshold for matching tracks
            'fuse_score': settings.BOTS_FUSE_SCORE,  # Whether to fuse confidence scores with the iou distances before matching
            # min_box_area: 10  # threshold for min box areas(for tracker evaluation, not used for now)
            #
            # BoT-SORT settings
            'gmc_method': settings.BOTS_GMC_METHOD,  # method of global motion compensation
            # ReID model related thresh
            'proximity_thresh': settings.BOTS_PROXIMITY_THRESH,  # minimum IoU for valid match with ReID
            'appearance_thresh': settings.BOTS_APPEARANCE_THRESH,  # minimum appearance similarity for ReID
            'with_reid': settings.BOTS_WITH_REID,
            'model': settings.BOTS_MODEL,  # uses native features if detector is YOLO else yolo11n-cls.pt
        }

        with open(file_name, 'w') as f:
            yaml.dump(yaml_content, f)
        return file_name

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


def _to_numpy(tensor_or_array):
    """Convert PyTorch tensor or array-like object to numpy array"""
    try:
        # Try PyTorch tensor conversion first
        return tensor_or_array.cpu().numpy()
    except AttributeError:
        # Fall back to numpy array conversion
        return np.array(tensor_or_array)
