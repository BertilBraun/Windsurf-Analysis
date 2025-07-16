import os
from pathlib import Path


DEFAULT_MODEL_NAME = Path(__file__).parent / '../train/models/100epochs.pt'

if not os.path.exists(DEFAULT_MODEL_NAME):
    print(f'WARNING: Model {DEFAULT_MODEL_NAME} not found, using default model')
    DEFAULT_MODEL_NAME = 'yolo11n.pt'

# YOLO settings

IOU_THRESHOLD = 0.2
CONFIDENCE_THRESHOLD = 0.1
BATCH_SIZE = 32

MIN_TRACKING_FPS = 25

# Tracking settings

# Minimum percentage of total frames a track must appear in (default 20%)
MIN_FRAME_PERCENTAGE = 20

MAX_TEMPORAL_DISTANCE_SECONDS = 10.0
MAX_SPATIAL_DISTANCE_BB = 2.5  # One Bounding Box Width
HISTOGRAM_SIMILARITY_THRESHOLD = 0.9


SMOOTHING_WINDOW_SIZE = 2  # TODO more or less?


STANDARD_OUTPUT_DIR = 'individual_surfers'
