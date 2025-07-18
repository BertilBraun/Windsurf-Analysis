from pathlib import Path


STANDARD_OUTPUT_DIR = 'individual_surfers'

# YOLO settings
YOLO_MODEL_PATH = Path(__file__).parent / '../train/models/100epochs.pt'
IOU_THRESHOLD = 0.2
CONFIDENCE_THRESHOLD = 0.1
BATCH_SIZE = 32

# Tracking preprocessing settings
MIN_TRACKING_FPS = 25


# Track postprocessing settings
# Minimum percentage of total frames a track must appear in (default 20%)
MIN_FRAME_PERCENTAGE = 20

MAX_TEMPORAL_DISTANCE_SECONDS = 10.0
MAX_SPATIAL_DISTANCE_BB = 2.5  # One Bounding Box Width
HISTOGRAM_SIMILARITY_THRESHOLD = 0.9

SMOOTHING_WINDOW_SIZE = 2  # TODO more or less?
