import os
import logging
from pathlib import Path


STANDARD_OUTPUT_DIR = 'individual_surfers'

# YOLO settings
YOLO_MODEL_PATH = Path(__file__).parent / '../train/models/100epochs.pt'
IOU_THRESHOLD = 0.2
CONFIDENCE_THRESHOLD = 0.1
BATCH_SIZE = 32

# Tracking preprocessing settings
MIN_TRACKING_FPS = 25

# Botsort settings
BOTS_TRACK_HIGH_THRESH = 0.2  # threshold for the first association
BOTS_TRACK_LOW_THRESH = 0.1  # threshold for the second association
BOTS_NEW_TRACK_THRESH = 0.4  # threshold for init new track if the detection does not match any tracks
BOTS_TRACK_BUFFER = MIN_TRACKING_FPS * 10  # buffer to calculate the time when to remove tracks
BOTS_MATCH_THRESH = 0.8  # threshold for matching tracks
BOTS_FUSE_SCORE = False  # Whether to fuse confidence scores with the iou distances before matching
# min_box_area: 10  # threshold for min box areas(for tracker evaluation not used for now)
#
# BoT-SORT settings
BOTS_GMC_METHOD = 'sparseOptFlow'  # method of global motion compensation
# ReID model related thresh
BOTS_PROXIMITY_THRESH = 0.05  # minimum IoU for valid match with ReID
BOTS_APPEARANCE_THRESH = 0.8  # minimum appearance similarity for ReID
BOTS_WITH_REID = True
BOTS_MODEL = 'auto'  # uses native features if detector is YOLO else yolo11n-cls.pt

# Track postprocessing settings
# Minimum percentage of total frames a track must appear in (default 20%)
MIN_FRAME_PERCENTAGE = 20

MAX_TEMPORAL_DISTANCE_SECONDS = 10.0
MAX_SPATIAL_DISTANCE_BB = 2.5  # One Bounding Box Width
HISTOGRAM_SIMILARITY_THRESHOLD = 0.9

SMOOTHING_WINDOW_SIZE = 2  # TODO more or less?
