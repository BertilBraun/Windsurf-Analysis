import torch
from pathlib import Path
from typing import Literal

USE_GPU = torch.cuda.is_available()

EPS = 1e-9

STANDARD_OUTPUT_DIR = 'tmp/standard'
NUM_PARALLEL_VIDEO_WORKERS = 4

# YOLO settings
inference_root_folder = Path(__file__).parent.parent
YOLO_MODEL_PATH = inference_root_folder / 'weights/yolo_models/windsurfing/best.pt'
OSNET_REID_MODEL_PATH = inference_root_folder / 'weights/reid_models/common/osnet_ain_x1_0_msmt17.pth'
OSNET_REID_MODEL_PATH = inference_root_folder / 'weights/reid_models/common/osnet_ain_x1_0_imagenet.pth'

# ReID model selection
REID_MODEL_TYPE: Literal['color_hist', 'osnet', 'vit', 'color_ab_stripe_hist'] = 'color_ab_stripe_hist'  # 'color_hist'

DETECTOR_IOU_THRESHOLD = 0.2
DETECTOR_CONFIDENCE_THRESHOLD = 0.6
DETECTOR_BATCH_SIZE = 32

# Tracking preprocessing settings
MIN_TRACKING_FPS = 300  # Disabled for now
MAX_OVERLAP_LENGTH_SECONDS = 5

# Track postprocessing settings
# Minimum percentage of total frames a track must appear in (default 5%)
MIN_FRAME_PERCENTAGE = 5

SMOOTHING_WINDOW_SIZE = 1  # TODO more or less?

VIDEO_SUFFIX_SECONDS = 1.0

# Greedy preprocessor settings
GREEDY_PREPROCESSOR_APPEARANCE_PROBABILITY_STRICT = 0.95
GREEDY_PREPROCESSOR_APPEARANCE_PROBABILITY_LOOSE = 0.20
GREEDY_PREPROCESSOR_MOTION_PROBABILITY_STRICT = 0.95
GREEDY_PREPROCESSOR_MOTION_PROBABILITY_LOOSE = 0.5
GREEDY_PREPROCESSOR_MAX_FRAME_DISTANCE = 6  # stale cutoff (frames)
GREEDY_PREPROCESSOR_EMA_ALPHA = 0.6  # appearance EMA smoothing

# Greedy tracker settings
# Greedy tracker merges tracks by average embedding cosine similarity of the tracks until no more possible merges exist, or the max average embedding cosine similarity is below the threshold
GREEDY_MIN_IOU_MATCHES_SINGLE_TRACK = 0.5
GREEDY_SHORT_TRACK_MIN_FRAMES = 10
GREEDY_MIN_COSINE_SIMILARITY = 0.8


# Discrete optimization tracker settings
# Start cost should discourage frivolous starts but not overpower strong local matches
OPTIMIZER_W_START = 9.75


OPTIMIZER_SHORT_MIN_LINK_IOU = 0.055
OPTIMIZER_SHORT_MIN_LINK_COS = 0.403
OPTIMIZER_SHORT_W_LINK_IOU = 0.880
OPTIMIZER_SHORT_W_LINK_APP = 1.144
OPTIMIZER_SHORT_W_LINK_GAP = 0.049
# the amount of frames to look forward and backwards for appearance.
# For now these are not weighted by distance so keep small
OPTIMIZER_SHORT_LINK_COST_APPEARANCE_WINDOW_RADIUS = 1


OPTIMIZER_LONG_MIN_LINK_IOU = 0.060
OPTIMIZER_LONG_MIN_LINK_COS = 0.721
OPTIMIZER_LONG_W_LINK_IOU = 0.222
OPTIMIZER_LONG_W_LINK_APP = 2.397
OPTIMIZER_LONG_W_LINK_GAP = 0.006  # multiplied by the percentage of max track gap


OPTIMIZER_TIMEOUT_SECONDS = 60

# Video splicing settings
OUTPUT_WIDTH = 1000  # width of the written video
OUTPUT_HEIGHT = 1000  # height of the written video
TARGET_BBOX_HEIGHT_RATIO = 0.70  # bbox should fill ~60 % of output height
SMOOTHING_ALPHA = 0.0  # 0 = no smoothing, 0.8 ≈ keep 80 % of the previous scale
MIN_SCALE = 0.2  # avoid over/under-zoom
MAX_SCALE = 10.0  # avoid over/under-zoom
