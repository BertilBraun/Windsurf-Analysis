DEFAULT_MODEL_NAME = 'yolo11n.pt'

# YOLO settings

IOU_THRESHOLD = 0.2
CONFIDENCE_THRESHOLD = 0.1
BATCH_SIZE = 32

MIN_TRACKING_FPS = 15

# Tracking settings

# Minimum percentage of total frames a track must appear in (default 30%)
MIN_FRAME_PERCENTAGE = 30
# Maximum frame gap to allow merging of non-overlapping tracks (default 150 frames = 5 seconds at 30fps)
MAX_MERGE_FRAME_GAP = 150


STANDARD_OUTPUT_DIR = 'individual_surfers'
