from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from server.inference.bot_sort.bot_sort import BoTSORT

from ..common_types import BoundingBox, Detection, Track
from ..tracking.tracking import Tracker
from ..util.video_io import VideoInfo, VideoReader
from ..visualization.stabilize import Transform  # dx, dy, da (radians), frame_idx


@dataclass
class _BotSortArgs:
    # Detection score thresholds
    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.05
    new_track_thresh: float = 0.7

    # Buffer/aging
    track_buffer: int = 30

    # Association thresholds
    proximity_thresh: float = 0.5  # IoU distance mask (1 - IoU)
    appearance_thresh: float = 0.5  # embedding distance cap (after internal scaling)
    match_thresh: float = 0.8  # final assignment cost limit

    # Camera motion compensation
    cmc_method: str = 'sparseOptFlow'  # {'sparseOptFlow','orb','sift','ecc','none'}

    # Verbosity hooks used by GMC when reading from file; safe defaults here
    name: str = ''
    ablation: int = 0


class BotSortTracker(Tracker):
    def __init__(
        self,
        vid_file_path: str,
        track_high_thresh: float = 0.40409851756977044,
        track_low_thresh: float = 0.0382033230726261,
        new_track_thresh: float = 0.6486597332163248,
        track_buffer: int = 300,  # 10sec * FPS
        proximity_thresh: float = 0.8974397185401413,
        appearance_thresh: float = 0.4422175665323802,
        match_thresh: float = 0.6275465323377439,
        cmc_method: str = 'sparseOptFlow',
    ) -> None:
        self.vid_file_path = vid_file_path
        self.args = _BotSortArgs(
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            proximity_thresh=proximity_thresh,
            appearance_thresh=appearance_thresh,
            match_thresh=match_thresh,
            cmc_method=cmc_method,
        )

    def track(self, tracks: List[Track], video_properties: VideoInfo, transforms: List[Transform]) -> List[Track]:
        # Group single-detection inputs into per-frame lists
        by_frame: Dict[int, List[Detection]] = {}
        for t in tracks:
            assert len(t.sorted_detections) == 1, 'Input must be one detection per Track.'
            d = t.sorted_detections[0]
            by_frame.setdefault(d.frame_idx, []).append(d)

        bot_sort = BoTSORT(self.args, video_properties.fps)

        # Collect per-track detections over time
        tid2dets: Dict[int, List[Detection]] = {}

        # Index transforms by frame for O(1) access
        frame_to_transform: Dict[int, Transform] = {int(t.frame_idx): t for t in transforms}
        # We use per-frame delta camera motion (prev -> curr)

        if 1:
            # Process with video frames for debug overlays
            with VideoReader(self.vid_file_path) as reader:
                for f, frame in reader.read_frames():
                    dets = by_frame.get(f, [])
                    last_detections = by_frame.get(f - 1, [])
                    # Build per-frame delta warp from cumulative transforms
                    t = frame_to_transform.get(int(f))
                    if t is not None:
                        c, s = float(np.cos(t.da)), float(np.sin(t.da))
                        dx, dy = float(t.dx), float(t.dy)
                        H3 = np.array([[c, -s, dx], [s, c, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
                    else:
                        H3 = np.eye(3, dtype=np.float64)
                    ext_warp = H3[:2, :3]

                    active_tracks = bot_sort.update(dets, last_detections, ext_warp, frame)

                    for st in active_tracks:
                        if st.frame_id == bot_sort.frame_id and st.is_activated:
                            tlbr = st.tlbr
                            bbox = BoundingBox(int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3]))
                            emb = st.curr_feat if st.curr_feat is not None else st.smooth_feat
                            if emb is None:
                                emb = np.zeros(128, dtype=np.float32)
                            det = Detection(bbox=bbox, embedding=emb, confidence=float(st.score), frame_idx=f)
                            tid2dets.setdefault(int(st.track_id), []).append(det)
        else:
            # Process per frame without re-reading the video
            active_tracks = []
            for f in range(video_properties.total_frames):
                dets = by_frame.get(f, [])
                last_detections = by_frame.get(f - 1, [])
                # Build per-frame delta warp from cumulative transforms
                t = frame_to_transform.get(int(f))
                if t is not None:
                    c, s = float(np.cos(t.da)), float(np.sin(t.da))
                    dx, dy = float(t.dx), float(t.dy)
                    H3 = np.array([[c, -s, dx], [s, c, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
                else:
                    H3 = np.eye(3, dtype=np.float64)
                ext_warp = H3[:2, :3]

                active_tracks = bot_sort.update(dets, last_detections, ext_warp, None)

                for st in active_tracks:
                    if st.frame_id == bot_sort.frame_id and st.is_activated:
                        tlbr = st.tlbr
                        bbox = BoundingBox(int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3]))
                        emb = st.curr_feat if st.curr_feat is not None else st.smooth_feat
                        if emb is None:
                            emb = np.zeros(128, dtype=np.float32)
                        det = Detection(bbox=bbox, embedding=emb, confidence=float(st.score), frame_idx=f)
                        tid2dets.setdefault(int(st.track_id), []).append(det)

            # Record matched updates (those updated at current internal frame)
            # Use the last processed frame index, if any
            last_frame_idx = max(by_frame.keys()) if by_frame else None
            if last_frame_idx is not None:
                for st in active_tracks:
                    if st.frame_id == bot_sort.frame_id and st.is_activated:
                        tlbr = st.tlbr  # (x1,y1,x2,y2)
                        bbox = BoundingBox(int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3]))
                        emb = st.curr_feat if st.curr_feat is not None else st.smooth_feat
                        if emb is None:
                            # Fallback to zeros if no feature is available (should be rare)
                            emb = np.zeros(128, dtype=np.float32)
                        det = Detection(bbox=bbox, embedding=emb, confidence=float(st.score), frame_idx=last_frame_idx)
                        tid2dets.setdefault(int(st.track_id), []).append(det)

        # Build final Track objects
        out_tracks: List[Track] = []
        for tid, dets in tid2dets.items():
            dets.sort(key=lambda d: d.frame_idx)
            out_tracks.append(Track(track_id=int(tid), sorted_detections=dets))
        return out_tracks
