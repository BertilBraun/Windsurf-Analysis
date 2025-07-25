from pathlib import Path


STANDARD_OUTPUT_DIR = 'individual_surfers'
NUM_PARALLEL_VIDEO_WORKERS = 4

# YOLO settings
YOLO_MODEL_PATH = Path(__file__).parent / '../train/models/100epochs.pt'
REID_MODEL_PATH = Path(__file__).parent / 'weights/osnet_ain_x1_0_msmt17.pth'

IOU_THRESHOLD = 0.2
CONFIDENCE_THRESHOLD = 0.25
BATCH_SIZE = 32

# Tracking preprocessing settings
MIN_TRACKING_FPS = 25
MAX_OVERLAP_LENGTH_SECONDS = 10

# Track postprocessing settings
# Minimum percentage of total frames a track must appear in (default 20%)
MIN_FRAME_PERCENTAGE = 20

SMOOTHING_WINDOW_SIZE = 2  # TODO more or less?

VIDEO_SUFFIX_SECONDS = 1.0

# Greedy preprocessor settings
GREEDY_PREPROCESSOR_MIN_IOU = 0.5
GREEDY_PREPROCESSOR_MIN_COSINE_SIMILARITY = 0.7
GREEDY_PREPROCESSOR_MAX_FRAME_DISTANCE = 5
GREEDY_PREPROCESSOR_MIN_IOU_MATCHES_SINGLE_TRACK = 0.1

# Greedy tracker settings
# Greedy tracker merges tracks by average embedding cosine similarity of the tracks until no more possible merges exist, or the max average embedding cosine similarity is below the threshold
GREEDY_MIN_IOU_MATCHES_SINGLE_TRACK = 0.5
GREEDY_SHORT_TRACK_MIN_FRAMES = 10
GREEDY_MIN_COSINE_SIMILARITY = 0.8


# Discrete optimization tracker settings
OPTIMIZER_MIN_LINK_IOU = 0.0
OPTIMIZER_MIN_LINK_COS = -1.0
OPTIMIZER_W_LINK_IOU = 0.2
OPTIMIZER_W_LINK_APP = 1.0
OPTIMIZER_W_LINK_GAP = 0.001
# should be scaled according to number of estimated starts / tracks and number links required
OPTIMIZER_W_START = 10.0
# the amount of frames to look forward and backwards for appearance.
# For now these are not weighted by distance so keep small
OPTIMIZER_LINK_COST_APPEARANCE_WINDOW_RADIUS = 10


OPTIMIZER_TIMEOUT_SECONDS = 60
