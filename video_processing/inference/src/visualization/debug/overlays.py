from __future__ import annotations

import cv2
import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Tuple


from ...motion.kalman_filter import KFState
from ...motion.cmc import CMC
from ...common_types import Detection, Track
from .draw import draw_bounding_box, draw_text, draw_arrow, compose_side_by_side


class Overlay:
    def apply(self, image: np.ndarray) -> None:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class DetectionsOverlay(Overlay):
    detections: List[Detection]

    def apply(self, image: np.ndarray) -> None:
        for i, d in enumerate(self.detections):
            draw_bounding_box(image, d.bbox, (255, 255, 255))
            # draw label at bottom right of bbox
            draw_text(image, f'Det {i}', (d.bbox.x2, d.bbox.y2), (255, 255, 255))


@dataclass
class KalmanOverlay(Overlay):
    kalman_states_by_track_id: Dict[int, KFState]
    camera_motion_compensator: CMC
    target_frame_index: int | None = None

    def apply(self, image: np.ndarray) -> None:
        # draw predicted bbox at target frame index (or current image index unknown; draw at alpha=0.0 via display_bbox)
        for track_id, state in self.kalman_states_by_track_id.items():
            if self.target_frame_index is not None:
                state = state.predict_to(int(self.target_frame_index), self.camera_motion_compensator)
            bbox = state.display_bbox(alpha=0.0)
            draw_bounding_box(image, bbox, (0, 255, 0), label=f'KF id={track_id}')
            # draw velocity vector
            vx, vy = state.mean[4], state.mean[5]
            draw_arrow(
                image,
                (bbox.center.x, bbox.center.y),
                (round(bbox.center.x + vx), round(bbox.center.y + vy)),
                (0, 255, 0),
            )


@dataclass
class CameraMotionTrailOverlay(Overlay):
    camera_motion_compensator: CMC
    history_points: List[Tuple[float, float]]

    def __init__(self, camera_motion_compensator: CMC) -> None:
        self.camera_motion_compensator = camera_motion_compensator
        self.history_points = []

    def apply(self, image: np.ndarray) -> None:
        # Attempt to fetch current frame transform by size, fall back to last
        try:
            # TODO: implement
            # The CMC stores transforms in a dict keyed by frame index
            # We cannot know the current frame index here; rely on caller to update history externally if needed.
            pass
        except Exception:
            pass
        img_h, img_w = image.shape[:2]
        center = (img_w // 2, img_h // 2)
        pts = [center]
        for dt in self.history_points:
            last_pt = pts[-1]
            nxt = (int(round(last_pt[0] + dt[0])), int(round(last_pt[1] + dt[1])))
            pts.append(nxt)
        n = len(pts)
        colors_list: List[Tuple[int, int, int]] = []
        if n > 1:
            idx = (np.linspace(0, 255, n - 1)).astype(np.uint8)
            cmap = cv2.applyColorMap(idx, cv2.COLORMAP_AUTUMN)
            for i in range(n - 1):
                r = int(cmap[i, 0, 2])
                g = int(cmap[i, 0, 1])
                b = int(cmap[i, 0, 0])
                colors_list.append((b, g, r))
        else:
            colors_list = [(0, 255, 255)] * 1
        for i in range(1, n):
            draw_arrow(image, pts[i - 1], pts[i], colors_list[i - 1], 2)


@dataclass
class EdgeMetrics:
    motion_nll: float
    appearance_nll: float
    gap_nll: float
    total_cost: float
    average_mahalanobis_squared: float
    average_log_determinant: float
    gap_frames: int


def compose_fragment_pair_view(
    track_a: Track,
    track_b: Track,
    metrics: EdgeMetrics,
    kalman_end_state: KFState,
    camera_motion_compensator: CMC,
    frame_a: np.ndarray,
    frame_b: np.ndarray,
) -> np.ndarray:
    vis_a = frame_a.copy()
    vis_b = frame_b.copy()

    bb_a = track_a.end.bbox
    bb_b = track_b.start.bbox
    draw_bounding_box(vis_a, bb_a, (0, 255, 0), label=f'A id={track_a.track_id} end f={track_a.end_frame}')
    draw_bounding_box(vis_b, bb_b, (0, 255, 255), label=f'B id={track_b.track_id} start f={track_b.start_frame}')

    bbox = kalman_end_state.display_bbox(alpha=0.0)
    draw_bounding_box(vis_a, bbox, (255, 0, 0), label='KF A end')
    if kalman_end_state.mean.shape[0] >= 6:
        vx = float(kalman_end_state.mean[4])
        vy = float(kalman_end_state.mean[5])
        cx, cy = bbox.center
        draw_arrow(vis_a, (cx, cy), (round(cx + vx), round(cy + vy)), (255, 0, 0), 2)

    pred_b = kalman_end_state.predict_to(track_b.start_frame, camera_motion_compensator)
    g2 = float(pred_b.gating_distance(track_b.start.bbox.center_wh))
    bbox = pred_b.display_bbox(alpha=0.0)
    draw_bounding_box(vis_b, bbox, (255, 0, 255), label=f'KF→B pred g^2={g2:.2f}')

    banner_h = 36
    banner = np.full((banner_h, max(vis_a.shape[1], vis_b.shape[1]) * 2, 3), 15, dtype=np.uint8)
    text = (
        f'Δf={track_a.frame_gap(track_b)} '
        f'avg_d2={metrics.average_mahalanobis_squared:.3f} avg_logdet={metrics.average_log_determinant:.3f} '
        f'C_mot={metrics.motion_nll:.4f}  C_app={metrics.appearance_nll:.4f}  C_gap={metrics.gap_nll:.4f}  C_tot={metrics.total_cost:.4f}'
    )
    draw_text(banner, text, (8, 24), (230, 230, 230), 0.6, 1)
    return compose_side_by_side(banner, vis_a, vis_b, target_height=max(vis_a.shape[0], vis_b.shape[0]))
