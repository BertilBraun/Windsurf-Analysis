"""
Renderable track preparation.

This module converts dense per-frame `Track` objects into `RenderableTrack`s by computing:
  - an anchor point per frame (for stable camera anchoring),
  - a per-frame scale factor (to normalize mast length within a track).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from ..common_types import Detection, Point, RenderableDetection, RenderableTrack, Track, interpolate
from ..visualization.stabilize import Transform


def _assert_dense_per_frame(detections: list[Detection], track_id: int) -> None:
    if not detections:
        return
    start_frame = int(detections[0].frame_idx)
    for i, det in enumerate(detections):
        expected = start_frame + i
        assert int(det.frame_idx) == expected, (
            f'Track {track_id} must be dense per frame in prepare_renderable_tracks '
            f'(expected frame_idx={expected}, got frame_idx={det.frame_idx}).'
        )


# --------------------------------------------------------------------------------------
# Anchor smoothing knobs (single implementation)
# --------------------------------------------------------------------------------------

# Neighborhood weights for robust mean smoothing. Keys are frame offsets relative to the current frame.
ANCHOR_SMOOTH_NEIGHBOR_WEIGHTS: dict[int, float] = {
    -2: 0.25,
    -1: 1.00,
    0: 4.00,
    1: 1.00,
    2: 0.25,
}

# Huber threshold (pixels) for robust reweighting.
ANCHOR_SMOOTH_ROBUST_MEAN_C_PX: float = 50.0

# Around this bbox size, keypoint localization becomes unreliable enough that anchoring should
# move back towards the bbox center.
ANCHOR_KEYPOINT_SIZE_BLEND_CENTER_PX: float = 50.0
ANCHOR_KEYPOINT_SIZE_BLEND_RANGE_PX: float = 10.0

# Bbox fallback anchor uses the horizontal center and a point lower in the box than the true center.
ANCHOR_BBOX_FALLBACK_Y_FRACTION: float = 0.6


@dataclass(frozen=True)
class FloatPoint:
    x: float
    y: float

    def distance_to(self, other: FloatPoint) -> float:
        dx = float(self.x) - float(other.x)
        dy = float(self.y) - float(other.y)
        return math.hypot(dx, dy)

    def interpolate(self, other: FloatPoint, alpha: float) -> FloatPoint:
        a = float(alpha)
        return FloatPoint(
            x=(1.0 - a) * float(self.x) + a * float(other.x),
            y=(1.0 - a) * float(self.y) + a * float(other.y),
        )

    def __add__(self, other: FloatPoint) -> FloatPoint:
        return FloatPoint(x=float(self.x) + float(other.x), y=float(self.y) + float(other.y))

    def __mul__(self, scalar: float) -> FloatPoint:
        s = float(scalar)
        return FloatPoint(x=float(self.x) * s, y=float(self.y) * s)

    def __truediv__(self, scalar: float) -> FloatPoint:
        s = float(scalar)
        if s == 0.0:
            return self
        return FloatPoint(x=float(self.x) / s, y=float(self.y) / s)


def _to_float_point(p: Point) -> FloatPoint:
    return FloatPoint(float(p.x), float(p.y))


def _rotation_cos_sin(angle_rad: float) -> tuple[float, float]:
    return (math.cos(float(angle_rad)), math.sin(float(angle_rad)))


class MotionModel:
    """
    Rigid 2D motion model for mapping points between frames.

    Uses a per-frame raw motion delta convention:
      Transform(dx,dy,da,frame_idx=k) maps points from frame (k-1) -> k via:
        p_k = R(da) * p_{k-1} + [dx, dy]
    """

    def __init__(
        self,
        *,
        raw_motion_transforms: list[Transform],
        start_frame: int,
        frame_count: int,
    ) -> None:
        self.start_frame = int(start_frame)
        self.frame_count = int(frame_count)

        raw_by_frame: dict[int, Transform] = {int(t.frame_idx): t for t in raw_motion_transforms}

        # Cumulative transform from `start_frame` to frame (start_frame + i):
        #   p_i = R[i] * p_0 + T[i]
        self._c: list[float] = [1.0] * self.frame_count  # cos(theta)
        self._s: list[float] = [0.0] * self.frame_count  # sin(theta)
        self._tx: list[float] = [0.0] * self.frame_count
        self._ty: list[float] = [0.0] * self.frame_count

        for rel_idx in range(1, self.frame_count):
            abs_frame = self.start_frame + rel_idx
            # Delta (abs_frame-1 -> abs_frame) is stored at frame_idx=abs_frame.
            t = raw_by_frame.get(int(abs_frame))
            if t is None:
                dx = dy = da = 0.0
            else:
                dx, dy, da = float(t.dx), float(t.dy), float(t.da)

            c_d, s_d = _rotation_cos_sin(da)
            c_prev, s_prev = self._c[rel_idx - 1], self._s[rel_idx - 1]

            # R_new = R_delta * R_prev  (angle addition)
            c_new = c_d * c_prev - s_d * s_prev
            s_new = s_d * c_prev + c_d * s_prev

            # T_new = R_delta * T_prev + dT
            tx_prev, ty_prev = self._tx[rel_idx - 1], self._ty[rel_idx - 1]
            tx_new = c_d * tx_prev - s_d * ty_prev + dx
            ty_new = s_d * tx_prev + c_d * ty_prev + dy

            self._c[rel_idx] = float(c_new)
            self._s[rel_idx] = float(s_new)
            self._tx[rel_idx] = float(tx_new)
            self._ty[rel_idx] = float(ty_new)

    def _relative_params(self, from_rel: int, to_rel: int) -> tuple[float, float, float, float]:
        c_to, s_to = self._c[to_rel], self._s[to_rel]
        c_from, s_from = self._c[from_rel], self._s[from_rel]

        # R_rel = R_to * R_from^{-1} = R_to * R_from^T (rotation matrices)
        c_rel = c_to * c_from + s_to * s_from
        s_rel = s_to * c_from - c_to * s_from

        # t_rel = t_to - R_rel * t_from
        tx_from, ty_from = self._tx[from_rel], self._ty[from_rel]
        tx_rf = c_rel * tx_from - s_rel * ty_from
        ty_rf = s_rel * tx_from + c_rel * ty_from
        tx_rel = self._tx[to_rel] - tx_rf
        ty_rel = self._ty[to_rel] - ty_rf
        return float(c_rel), float(s_rel), float(tx_rel), float(ty_rel)

    def map_point_abs(self, from_frame: int, to_frame: int, p: FloatPoint) -> FloatPoint:
        from_rel = int(from_frame) - self.start_frame
        to_rel = int(to_frame) - self.start_frame
        if from_rel == to_rel:
            return p
        if not (0 <= from_rel < self.frame_count and 0 <= to_rel < self.frame_count):
            return p
        c_rel, s_rel, tx_rel, ty_rel = self._relative_params(from_rel, to_rel)
        x = c_rel * float(p.x) - s_rel * float(p.y) + tx_rel
        y = s_rel * float(p.x) + c_rel * float(p.y) + ty_rel
        return FloatPoint(x=float(x), y=float(y))


def _anchor_robust_mean(
    points: list[FloatPoint],
    *,
    frame_start: int,
    raw_motion_transforms: list[Transform],
) -> list[FloatPoint]:
    if not points:
        return []

    weights = dict(ANCHOR_SMOOTH_NEIGHBOR_WEIGHTS)
    weights.setdefault(0, 1.0)
    robust_c_px = float(ANCHOR_SMOOTH_ROBUST_MEAN_C_PX)

    motion: Optional[MotionModel] = None
    if len(points) >= 2 and False:  # TODO reenable? - currently way worse when enabled
        motion = MotionModel(
            raw_motion_transforms=raw_motion_transforms,
            start_frame=int(frame_start),
            frame_count=int(len(points)),
        )

    offsets = sorted(weights.keys(), key=lambda k: abs(int(k)))
    out: list[FloatPoint] = []

    for i in range(len(points)):
        current_abs = int(frame_start + i)
        current_pt = points[i]

        samples: list[tuple[FloatPoint, float]] = []
        for rel_off in offsets:
            j = int(i + int(rel_off))
            if j < 0 or j >= len(points):
                continue
            w = float(weights.get(int(rel_off), 0.0))
            if w <= 0.0:
                continue

            neighbor = points[j]
            if motion is not None:
                neighbor_abs = int(frame_start + j)
                neighbor = motion.map_point_abs(neighbor_abs, current_abs, neighbor)
            samples.append((neighbor, w))

        if not samples:
            out.append(current_pt)
            continue

        def mean_with(scale: list[float]) -> FloatPoint:
            acc = FloatPoint(0.0, 0.0)
            acc_w = 0.0
            for (p, w0), s in zip(samples, scale):
                w = float(w0) * float(s)
                if w <= 0.0:
                    continue
                acc = acc + p * w
                acc_w += w
            return current_pt if acc_w <= 0.0 else (acc / acc_w)

        m0 = mean_with([1.0 for _ in samples])
        if robust_c_px > 0.0:
            scale2: list[float] = []
            for p, _w in samples:
                r = p.distance_to(m0)
                scale2.append(1.0 if r <= robust_c_px else (robust_c_px / max(1e-9, float(r))))
            out.append(mean_with(scale2))
        else:
            out.append(m0)

    return out


def _smooth_mean(values: list[float], radius: int) -> list[float]:
    assert 0 <= radius, f'Radius {radius} must be positive'
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values) - 1, i + radius)
        window = values[lo : hi + 1]
        out.append(float(sum(window) / max(1, len(window))))
    return out


def _median(values: list[float]) -> float:
    if not values:
        return 1.0
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def calculate_anchors(
    detections: list[Detection],
    *,
    missing_grace_frames: int,
    video_width: int,
    video_height: int,
    raw_motion_transforms: list[Transform],
) -> list[Point]:
    """
    Compute per-frame anchors for a dense, per-frame list of detections.

    Behavior:
      - Prefer a point above the boom when boom is visible.
      - Blend back towards a lower bbox fallback point when detections are too small for reliable keypoints.
      - Short gaps (<= missing_grace_frames) between two visible boom keypoints are interpolated (anchor->anchor).
      - Longer missing spans blend to/from the bbox fallback point over missing_grace_frames to avoid snapping.
      - Final anchors are smoothed with a robust (Huber) neighborhood mean; when raw camera-motion transforms are
        provided, neighbor samples are mapped into the current frame before smoothing.
    """
    assert missing_grace_frames > 0
    assert video_width > 0
    assert video_height > 0

    frame_start = int(detections[0].frame_idx) if detections else 0

    bbox_centers = [_to_float_point(d.bbox.center) for d in detections]
    bbox_fallback_anchors = [
        FloatPoint(
            float(d.bbox.center.x),
            float(d.bbox.y1) + float(d.bbox.height) * float(ANCHOR_BBOX_FALLBACK_Y_FRACTION),
        )
        for d in detections
    ]
    boom_vis = [d.boom.is_visible for d in detections]
    visible_indices = [i for i, v in enumerate(boom_vis) if v]

    bbox_top_y = [float(d.bbox.y1) for d in detections]
    boom_or_center_x = [
        float(d.boom.point.x) if boom_vis[i] else float(bbox_centers[i].x) for i, d in enumerate(detections)
    ]

    def proxy_mast_tip(i: int) -> FloatPoint:
        return FloatPoint(float(boom_or_center_x[i]), float(bbox_top_y[i]))

    def preferred_anchor(i: int) -> FloatPoint:
        # Keep the anchor slightly above the boom/mast intersection, but do not let mast-tip noise move the crop.
        # The final neighborhood smoother absorbs frame-to-frame jitter.
        if boom_vis[i]:
            bbox_size = min(float(detections[i].bbox.width), float(detections[i].bbox.height))
            boom = _to_float_point(detections[i].boom.point)
            keypoint_anchor = proxy_mast_tip(i).interpolate(boom, 0.85)
            blend_range = max(1e-6, float(ANCHOR_KEYPOINT_SIZE_BLEND_RANGE_PX))
            keypoint_weight = (bbox_size - float(ANCHOR_KEYPOINT_SIZE_BLEND_CENTER_PX)) / blend_range + 0.5
            keypoint_weight = max(0.0, min(1.0, keypoint_weight))
            return bbox_fallback_anchors[i].interpolate(keypoint_anchor, keypoint_weight)
        return bbox_fallback_anchors[i]

    if not visible_indices:
        smoothed = _anchor_robust_mean(
            bbox_fallback_anchors,
            frame_start=frame_start,
            raw_motion_transforms=raw_motion_transforms,
        )
        return [Point(int(p.x), int(p.y)) for p in smoothed]

    anchors = list(bbox_fallback_anchors)
    visible_set = set(visible_indices)
    for i in visible_indices:
        anchors[i] = preferred_anchor(i)

    short_gap_indices: set[int] = set()
    for a_i, b_i in zip(visible_indices[:-1], visible_indices[1:]):
        gap = b_i - a_i - 1
        if 0 < gap <= missing_grace_frames:
            p0 = preferred_anchor(a_i)
            p1 = preferred_anchor(b_i)
            for k in range(1, gap + 1):
                t = float(k) / float(gap + 1)
                idx = a_i + k
                anchors[idx] = p0.interpolate(p1, t)
                short_gap_indices.add(idx)

    out_point: list[Optional[FloatPoint]] = [None for _ in detections]
    out_weight: list[float] = [0.0 for _ in detections]
    in_point: list[Optional[FloatPoint]] = [None for _ in detections]
    in_weight: list[float] = [0.0 for _ in detections]

    def set_out(idx: int, pt: FloatPoint, w: float) -> None:
        if out_point[idx] is None or w > out_weight[idx]:
            out_point[idx] = pt
            out_weight[idx] = float(w)

    def set_in(idx: int, pt: FloatPoint, w: float) -> None:
        if in_point[idx] is None or w > in_weight[idx]:
            in_point[idx] = pt
            in_weight[idx] = float(w)

    # Leading missing: if > grace, interpolate bbox fallback point at first-grace -> anchor (at first)
    first = visible_indices[0]
    if first > missing_grace_frames:
        start_idx = first - missing_grace_frames
        start_pt = bbox_fallback_anchors[start_idx]
        end_pt = preferred_anchor(first)
        for idx in range(start_idx, first):
            if idx in visible_set or idx in short_gap_indices:
                continue
            t = float(idx - start_idx) / float(missing_grace_frames)
            set_in(idx, start_pt.interpolate(end_pt, t), w=float(missing_grace_frames - (first - idx) + 1))

    # Internal long gaps: anchor->bbox fallback then bbox fallback->anchor (with overlap blending if needed)
    for a_i, b_i in zip(visible_indices[:-1], visible_indices[1:]):
        gap = b_i - a_i - 1
        if gap <= missing_grace_frames:
            continue

        # Out: anchor at a_i -> bbox fallback point at a_i+grace
        out_target_idx = a_i + missing_grace_frames
        out_target = bbox_fallback_anchors[out_target_idx]
        for idx in range(a_i + 1, min(out_target_idx, b_i - 1) + 1):
            if idx in visible_set or idx in short_gap_indices:
                continue
            t = float(idx - a_i) / float(missing_grace_frames)
            set_out(
                idx,
                preferred_anchor(a_i).interpolate(out_target, t),
                w=float(missing_grace_frames - (idx - a_i) + 1),
            )

        # In: bbox fallback point at b_i-grace -> anchor at b_i
        in_start_idx = b_i - missing_grace_frames
        in_start = bbox_fallback_anchors[in_start_idx]
        for idx in range(max(in_start_idx, a_i + 1), b_i):
            if idx in visible_set or idx in short_gap_indices:
                continue
            t = float(idx - in_start_idx) / float(missing_grace_frames)
            set_in(
                idx,
                in_start.interpolate(preferred_anchor(b_i), t),
                w=float(missing_grace_frames - (b_i - idx) + 1),
            )

    # Trailing missing:
    # - <= grace: hold last anchor
    # - > grace: interpolate anchor -> bbox fallback point at last+grace
    last = visible_indices[-1]
    trailing = (len(detections) - 1) - last
    if trailing <= missing_grace_frames:
        for idx in range(last + 1, len(detections)):
            anchors[idx] = anchors[last]
    else:
        out_target_idx = last + missing_grace_frames
        out_target = bbox_fallback_anchors[out_target_idx]
        for idx in range(last + 1, out_target_idx + 1):
            if idx in visible_set or idx in short_gap_indices:
                continue
            t = float(idx - last) / float(missing_grace_frames)
            set_out(
                idx,
                preferred_anchor(last).interpolate(out_target, t),
                w=float(missing_grace_frames - (idx - last) + 1),
            )
        # Beyond out_target_idx stays at the bbox fallback point (default).

    # Apply (possibly blended) transitions.
    for idx in range(len(detections)):
        if idx in visible_set or idx in short_gap_indices:
            continue
        p_out = out_point[idx]
        p_in = in_point[idx]
        if p_out is not None and p_in is not None:
            w_out = max(0.0, float(out_weight[idx]))
            w_in = max(0.0, float(in_weight[idx]))
            w_sum = w_out + w_in
            anchors[idx] = p_out if w_sum <= 0.0 else (p_out * w_out + p_in * w_in) / w_sum
        elif p_out is not None:
            anchors[idx] = p_out
        elif p_in is not None:
            anchors[idx] = p_in

    smoothed = _anchor_robust_mean(
        anchors,
        frame_start=frame_start,
        raw_motion_transforms=raw_motion_transforms,
    )
    return [Point(int(p.x), int(p.y)) for p in smoothed]


def calculate_scales(
    detections: list[Detection],
    anchors: list[Point],
    *,
    missing_grace_frames: int,
    mast_smooth_radius: int,
    video_height: int,
    target_mast_fill: float = 0.75,
    target_bbox_fill: float = 0.9,
) -> list[float]:
    """
    Compute per-frame normalized crop height fractions (0..1) for a dense, per-frame list of detections.

    The "length" signal is preferred as dist(boom, mast_tip) when both keypoints are visible.
    Missing spans are filled with a fallback using bbox top edge and anchors; then smoothed.
    The resulting crop height is chosen so the mast occupies ~target_mast_fill of the crop height.
    """
    assert missing_grace_frames > 0
    assert mast_smooth_radius > 0
    assert video_height > 0
    assert 0.0 < float(target_mast_fill) < 1.0
    assert 0.0 < float(target_bbox_fill) <= 1.0
    grace = int(missing_grace_frames)

    boom_vis = [d.boom.is_visible for d in detections]
    mast_vis = [d.mast_tip.is_visible for d in detections]

    len_vis = [boom_vis[i] and mast_vis[i] for i in range(len(detections))]
    raw_len: list[Optional[float]] = [
        detections[i].boom.point.distance_to(detections[i].mast_tip.point) if len_vis[i] else None
        for i in range(len(detections))
    ]

    def bbox_fallback_len(i: int) -> float:
        return float(abs(anchors[i].y - detections[i].bbox.y1))

    visible_len_indices = [i for i, v in enumerate(len_vis) if v]
    filled_len = [bbox_fallback_len(i) for i in range(len(detections))]
    if visible_len_indices:
        for i in visible_len_indices:
            length = raw_len[i]
            assert length is not None
            filled_len[i] = length

        # interpolate short gaps between visible lengths
        for a_i, b_i in zip(visible_len_indices[:-1], visible_len_indices[1:]):
            gap = b_i - a_i
            if 0 < gap <= grace:
                v0 = filled_len[a_i]
                v1 = filled_len[b_i]
                for k in range(1, gap):
                    t = float(k) / float(gap)
                    filled_len[a_i + k] = interpolate(v0, v1, t)

        # trailing missing: hold last visible up to grace, then bbox fallback
        last = visible_len_indices[-1]
        for i in range(last + 1, len(detections)):
            if (i - last) <= grace:
                filled_len[i] = filled_len[last]
            else:
                filled_len[i] = bbox_fallback_len(i)

    smoothed_len = _smooth_mean(filled_len, int(mast_smooth_radius))
    crop_h_px_mast = [float(length) / target_mast_fill for length in smoothed_len]
    crop_h_px_bbox = [float(d.bbox.height) / target_bbox_fill for d in detections]
    crop_h_px = [max(hm, hb, 1.0) for hm, hb in zip(crop_h_px_mast, crop_h_px_bbox)]
    crop_h_norm = [float(h) / float(video_height) for h in crop_h_px]
    return [max(1e-6, min(1.0, v)) for v in crop_h_norm]


def prepare_renderable_tracks(
    tracks: list[Track],
    *,
    video_width: int,
    video_height: int,
    raw_motion_transforms: list[Transform],
    missing_grace_frames: int = 5,
    mast_smooth_radius: int = 15,
    target_mast_fill: float = 0.75,
) -> list[RenderableTrack]:
    """
    Post-processing step for the rendering pipeline.

    Produces RenderableTrack/RenderableDetection objects with stable per-frame:
      - anchor (Point)
      - scale (float)  (normalized crop height fraction, 0..1)
    """
    out_tracks: list[RenderableTrack] = []
    for track in tracks:
        detections = track.sorted_detections
        if not detections:
            out_tracks.append(RenderableTrack(track_id=track.track_id, sorted_detections=[]))
            continue

        _assert_dense_per_frame(detections, track.track_id)
        anchors = calculate_anchors(
            detections,
            missing_grace_frames=missing_grace_frames,
            video_width=video_width,
            video_height=video_height,
            raw_motion_transforms=raw_motion_transforms,
        )
        scales = calculate_scales(
            detections,
            anchors,
            missing_grace_frames=missing_grace_frames,
            mast_smooth_radius=mast_smooth_radius,
            video_height=video_height,
            target_mast_fill=target_mast_fill,
        )

        out_tracks.append(
            RenderableTrack(
                track_id=track.track_id,
                sorted_detections=[
                    RenderableDetection(
                        bbox=detection.bbox,
                        confidence=float(detection.confidence),
                        frame_idx=int(detection.frame_idx),
                        interpolated=bool(detection.interpolated),
                        boom=detection.boom,
                        mast_tip=detection.mast_tip,
                        anchor=anchor,
                        scale=scale,
                    )
                    for detection, anchor, scale in zip(detections, anchors, scales)
                ],
            )
        )

    return out_tracks
