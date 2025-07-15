import os
import numpy as np
from typing import Generator

from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

from video_io import get_video_properties
from settings import DEFAULT_MODEL_NAME, IOU_THRESHOLD, CONFIDENCE_THRESHOLD, BATCH_SIZE, MIN_TRACKING_FPS
from common_types import Detection, BoundingBox


class SurferDetector:
    """Pure detection and tracking class for surfers in video"""

    def __init__(self):
        self.model = YOLO(DEFAULT_MODEL_NAME, verbose=True)
        # TODO does the model have to be reset?
        # TODO does the model have to be moved to the GPU?

    def detect_and_track_video(
        self, video_path: os.PathLike | str
    ) -> Generator[tuple[int, np.ndarray, list[Detection]], None, None]:
        """Run batched inference on entire video, return generator of (frame, detections)"""

        video_props = get_video_properties(video_path)
        skip_frames = video_props.fps // MIN_TRACKING_FPS

        # TODO write the yaml here live? Be able to change the parameters :)

        results = self.model.track(
            str(video_path),
            iou=IOU_THRESHOLD,
            conf=CONFIDENCE_THRESHOLD,
            batch=BATCH_SIZE,
            vid_stride=skip_frames,
            tracker='botsort.yaml',
            # track_buffer=MIN_TRACKING_FPS * 10,  # 10sec sensible?
            # with_reid=True,  # TODO try
            persist=True,  # TODO True?
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
