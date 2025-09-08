import torch
from pathlib import Path

USE_GPU = torch.cuda.is_available()

STANDARD_OUTPUT_DIR = 'individual_surfers'
NUM_PARALLEL_VIDEO_WORKERS = 4

# YOLO settings
inference_root_folder = Path(__file__).parent.parent
YOLO_MODEL_PATH = inference_root_folder / 'weights/yolo_models/windsurfing/best.pt'
REID_MODEL_PATH = inference_root_folder / 'weights/reid_models/common/osnet_ain_x1_0_msmt17.pth'

IOU_THRESHOLD = 0.2
CONFIDENCE_THRESHOLD = 0.75
BATCH_SIZE = 32

# Tracking preprocessing settings
MIN_TRACKING_FPS = 25
MAX_OVERLAP_LENGTH_SECONDS = 10

# Track postprocessing settings
# Minimum percentage of total frames a track must appear in (default 5%)
MIN_FRAME_PERCENTAGE = 5

SMOOTHING_WINDOW_SIZE = 1  # TODO more or less?

VIDEO_SUFFIX_SECONDS = 1.0

# Greedy preprocessor settings
GREEDY_PREPROCESSOR_MIN_IOU = 0.2
GREEDY_PREPROCESSOR_MIN_COSINE_SIMILARITY = 0.85
GREEDY_PREPROCESSOR_MAX_FRAME_DISTANCE = 5
GREEDY_PREPROCESSOR_MIN_IOU_MATCHES_SINGLE_TRACK = 0.02

# Greedy tracker settings
# Greedy tracker merges tracks by average embedding cosine similarity of the tracks until no more possible merges exist, or the max average embedding cosine similarity is below the threshold
GREEDY_MIN_IOU_MATCHES_SINGLE_TRACK = 0.5
GREEDY_SHORT_TRACK_MIN_FRAMES = 10
GREEDY_MIN_COSINE_SIMILARITY = 0.8


# Discrete optimization tracker settings
# Start cost should discourage frivolous starts but not overpower strong local matches
OPTIMIZER_W_START = 4.0


OPTIMIZER_SHORT_MIN_LINK_IOU = 0.1
OPTIMIZER_SHORT_MIN_LINK_COS = 0.2
OPTIMIZER_SHORT_W_LINK_IOU = 0.3
OPTIMIZER_SHORT_W_LINK_APP = 1.2
OPTIMIZER_SHORT_W_LINK_GAP = 0.02
# the amount of frames to look forward and backwards for appearance.
# For now these are not weighted by distance so keep small
OPTIMIZER_SHORT_LINK_COST_APPEARANCE_WINDOW_RADIUS = 10


OPTIMIZER_LONG_MIN_LINK_IOU = 0.0
OPTIMIZER_LONG_MIN_LINK_COS = 0.6
OPTIMIZER_LONG_W_LINK_IOU = 0.1
OPTIMIZER_LONG_W_LINK_APP = 2.0
OPTIMIZER_LONG_W_LINK_GAP = 2.0  # multiplied by the percentage of max track gap


OPTIMIZER_TIMEOUT_SECONDS = 60

# Video splicing settings
OUTPUT_WIDTH = 1000  # width of the written video
OUTPUT_HEIGHT = 1000  # height of the written video
TARGET_BBOX_HEIGHT_RATIO = 0.70  # bbox should fill ~60 % of output height
SMOOTHING_ALPHA = 0.0  # 0 = no smoothing, 0.8 ≈ keep 80 % of the previous scale
MIN_SCALE = 0.2  # avoid over/under-zoom
MAX_SCALE = 10.0  # avoid over/under-zoom
