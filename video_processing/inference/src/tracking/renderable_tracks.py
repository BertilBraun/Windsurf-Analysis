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


# --------------------------------------------------------------------------------------
# Tuning knobs (edit for quick iteration)
# --------------------------------------------------------------------------------------

# Stabilization transform index offset used when mapping neighbor points into the current frame.
# This is intentionally a simple integer shift so you can quickly test whether your per-frame deltas
# are off-by-one (common when mixing "prev->curr" vs "anchored-at-frame" transform conventions).
ANCHOR_SMOOTH_TRANSFORM_INDEX_OFFSET = 0

# Neighborhood weights for anchor smoothing. Keys are frame offsets relative to the current frame.
# Example: {-2:0.25, -1:1.0, 0:4.0, 1:0.5, 2:0.25}
ANCHOR_SMOOTH_NEIGHBOR_WEIGHTS: dict[int, float] = {
    -2: 0.25,
    -1: 1.00,
    0: 4.00,
    1: 1.00,
    2: 0.25,
}

# Anchor smoothing method:
# - "weighted_mean": simple weighted average (default; tends to preserve motion better)
# - "local_fit": weighted local regression, evaluated at k=0 (can look "floaty" if the window is too wide)
# - "robust_mean": weighted mean with one robust reweighting step (Huber) against outliers
ANCHOR_SMOOTH_METHOD: str = 'robust_mean'

# Whether to use background-derived camera transforms to map neighbor samples into the current frame before smoothing.
# In practice, this often hurts because anchors are on a moving subject (not background). Keep it off by default.
ANCHOR_SMOOTH_USE_CAMERA_TRANSFORMS: bool = False

# Motion-aware smoothing model for anchors after mapping neighbor samples into the current frame.
# - "linear": fit position + velocity over the local window, then evaluate at k=0 (reduces jitter with less "motion averaging")
# - "quadratic": also fits acceleration (can help on turns, but is easier to overfit when samples are sparse/noisy)
ANCHOR_SMOOTH_MODEL: str = 'linear'

# Optional robust fitting (Huber) threshold in pixels. Set 0 to disable.
ANCHOR_SMOOTH_ROBUST_C_PX: float = 0.0

# Optional max-distance gate for including a neighbor sample (in pixels). This was useful for guarding against
# single-frame pose outliers, but it can also suppress smoothing during legitimate fast subject motion.
# Set to `None` to disable and rely on robust fitting instead.
ANCHOR_SMOOTH_MAX_NEIGHBOR_DISTANCE_FRAME_PERCENT: Optional[float] = None

# Optional *motion-compensated residual* downweighting for the weighted-mean smoother.
# Idea: neighbors far from a simple local constant-velocity prediction are more likely to be detection jitter/outliers.
#
# This helps keep smoothing active during legitimate fast motion (where absolute distance can be large) while still
# rejecting "spiky" measurements.
#
# Only applied when `ANCHOR_SMOOTH_USE_CAMERA_TRANSFORMS` is False (i.e. we are smoothing in raw image space).
ANCHOR_SMOOTH_RESIDUAL_HUBER_C_PX: float = 0.0

# Robust mean (Huber) threshold in pixels. Applied only when `ANCHOR_SMOOTH_METHOD="robust_mean"`.
ANCHOR_SMOOTH_ROBUST_MEAN_C_PX: float = 50.0


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
            x=(1.0 - a) * float(self.x) + a * float(other.x), y=(1.0 - a) * float(self.y) + a * float(other.y)
        )

    def __add__(self, other: FloatPoint) -> FloatPoint:
        return FloatPoint(x=float(self.x) + float(other.x), y=float(self.y) + float(other.y))

    def __sub__(self, other: FloatPoint) -> FloatPoint:
        return FloatPoint(x=float(self.x) - float(other.x), y=float(self.y) - float(other.y))

    def __mul__(self, scalar: float) -> FloatPoint:
        s = float(scalar)
        return FloatPoint(x=float(self.x) * s, y=float(self.y) * s)

    def __truediv__(self, scalar: float) -> FloatPoint:
        s = float(scalar)
        if s == 0:
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
        transform_index_offset: int,
    ) -> None:
        self.start_frame = int(start_frame)
        self.frame_count = int(frame_count)
        self.transform_index_offset = int(transform_index_offset)

        raw_by_frame: dict[int, Transform] = {int(t.frame_idx): t for t in raw_motion_transforms}

        # Cumulative transform from `start_frame` to frame (start_frame + i):
        #   p_i = R[i] * p_0 + T[i]
        self._c: list[float] = [1.0] * self.frame_count  # cos(theta)
        self._s: list[float] = [0.0] * self.frame_count  # sin(theta)
        self._tx: list[float] = [0.0] * self.frame_count
        self._ty: list[float] = [0.0] * self.frame_count

        for rel_idx in range(1, self.frame_count):
            prev_abs = self.start_frame + rel_idx - 1
            # Delta (prev_abs -> prev_abs+1) is stored at frame_idx=(prev_abs+1).
            # Apply the optional index offset by shifting that lookup key.
            key = int(prev_abs + 1 + self.transform_index_offset)
            t = raw_by_frame.get(key)
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

    def _map_rel(self, from_rel: int, to_rel: int, p: FloatPoint) -> FloatPoint:
        if from_rel == to_rel:
            return p
        if not (0 <= from_rel < self.frame_count and 0 <= to_rel < self.frame_count):
            return p

        c_rel, s_rel, tx_rel, ty_rel = self._relative_params(from_rel, to_rel)

        x = c_rel * float(p.x) - s_rel * float(p.y) + tx_rel
        y = s_rel * float(p.x) + c_rel * float(p.y) + ty_rel
        return FloatPoint(x=float(x), y=float(y))

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
        return self._map_rel(from_rel, to_rel, p)

    def relative_rigid_params_abs(self, from_frame: int, to_frame: int) -> tuple[float, float, float, float]:
        """
        Return rigid mapping parameters (c, s, tx, ty) such that:
          p_to = R(c,s) * p_from + [tx, ty]
        in absolute (pixel) coordinates.
        """
        from_rel = int(from_frame) - self.start_frame
        to_rel = int(to_frame) - self.start_frame
        if from_rel == to_rel:
            return (1.0, 0.0, 0.0, 0.0)
        if not (0 <= from_rel < self.frame_count and 0 <= to_rel < self.frame_count):
            return (1.0, 0.0, 0.0, 0.0)
        return self._relative_params(from_rel, to_rel)


def _anchor_weighted(
    points: list[FloatPoint],
    *,
    frame_start: int,
    raw_motion_transforms: Optional[list[Transform]],
    transform_index_offset: int,
    neighbor_weights: dict[int, float],
    max_neighbor_distance: Optional[float] = None,
) -> list[FloatPoint]:
    if not points:
        return []

    weights = dict(neighbor_weights)
    weights.setdefault(0, 1.0)

    motion: Optional[MotionModel] = None
    if raw_motion_transforms is not None and len(points) >= 2:
        motion = MotionModel(
            raw_motion_transforms=raw_motion_transforms,
            start_frame=int(frame_start),
            frame_count=int(len(points)),
            transform_index_offset=int(transform_index_offset),
        )

    out: list[FloatPoint] = []
    for i in range(len(points)):
        current_abs = int(frame_start + i)
        current_pt = points[i]
        residual_huber_c_px = float(ANCHOR_SMOOTH_RESIDUAL_HUBER_C_PX)
        use_residual_gating = motion is None and residual_huber_c_px > 0.0

        # Simple local velocity estimate (central difference) for residual gating in raw image space.
        vx = vy = 0.0
        if use_residual_gating:
            prev_pt = points[i - 1] if i - 1 >= 0 else None
            next_pt = points[i + 1] if i + 1 < len(points) else None
            if prev_pt is not None and next_pt is not None:
                vx = 0.5 * (float(next_pt.x) - float(prev_pt.x))
                vy = 0.5 * (float(next_pt.y) - float(prev_pt.y))
            elif prev_pt is not None:
                vx = float(current_pt.x) - float(prev_pt.x)
                vy = float(current_pt.y) - float(prev_pt.y)
            elif next_pt is not None:
                vx = float(next_pt.x) - float(current_pt.x)
                vy = float(next_pt.y) - float(current_pt.y)

        acc = FloatPoint(0.0, 0.0)
        acc_w = 0.0

        for rel_off, w in sorted(weights.items(), key=lambda kv: abs(int(kv[0]))):
            j = int(i + int(rel_off))
            if j < 0 or j >= len(points):
                continue
            w = float(w)
            if w <= 0:
                continue

            neighbor = points[j]
            if motion is not None:
                neighbor_abs = int(frame_start + j)
                neighbor = motion.map_point_abs(neighbor_abs, current_abs, neighbor)

            if max_neighbor_distance is not None and neighbor.distance_to(current_pt) > float(max_neighbor_distance):
                continue

            # Downweight samples that deviate strongly from a local constant-velocity model.
            if use_residual_gating:
                k = float(rel_off)
                pred = FloatPoint(x=float(current_pt.x) + vx * k, y=float(current_pt.y) + vy * k)
                residual = neighbor.distance_to(pred)
                if residual > residual_huber_c_px:
                    w *= residual_huber_c_px / max(1e-9, float(residual))

            acc = acc + neighbor * w
            acc_w += w

        out.append(current_pt if acc_w <= 0 else (acc / acc_w))

    return out


def _anchor_robust_mean(
    points: list[FloatPoint],
    *,
    frame_start: int,
    raw_motion_transforms: Optional[list[Transform]],
    transform_index_offset: int,
    neighbor_weights: dict[int, float],
    robust_c_px: float,
    max_neighbor_distance: Optional[float] = None,
) -> list[FloatPoint]:
    """
    Weighted mean with one robust reweighting step.

    Steps per frame:
      1) Compute weighted mean from neighbor samples (optionally mapped into current frame).
      2) Reweight each sample using a Huber factor based on its distance to the mean.
      3) Compute the final weighted mean.
    """
    if not points:
        return []

    weights = dict(neighbor_weights)
    weights.setdefault(0, 1.0)
    robust_c_px = float(robust_c_px)
    if robust_c_px <= 0.0:
        return _anchor_weighted(
            points,
            frame_start=int(frame_start),
            raw_motion_transforms=raw_motion_transforms,
            transform_index_offset=int(transform_index_offset),
            neighbor_weights=neighbor_weights,
            max_neighbor_distance=max_neighbor_distance,
        )

    motion: Optional[MotionModel] = None
    if raw_motion_transforms is not None and len(points) >= 2:
        motion = MotionModel(
            raw_motion_transforms=raw_motion_transforms,
            start_frame=int(frame_start),
            frame_count=int(len(points)),
            transform_index_offset=int(transform_index_offset),
        )

    out: list[FloatPoint] = []
    for i in range(len(points)):
        current_abs = int(frame_start + i)
        current_pt = points[i]

        samples: list[tuple[FloatPoint, float]] = []
        for rel_off, w in sorted(weights.items(), key=lambda kv: abs(int(kv[0]))):
            j = int(i + int(rel_off))
            if j < 0 or j >= len(points):
                continue
            w = float(w)
            if w <= 0.0:
                continue

            neighbor = points[j]
            if motion is not None:
                neighbor_abs = int(frame_start + j)
                neighbor = motion.map_point_abs(neighbor_abs, current_abs, neighbor)

            if max_neighbor_distance is not None and neighbor.distance_to(current_pt) > float(max_neighbor_distance):
                continue

            samples.append((neighbor, w))

        if not samples:
            out.append(current_pt)
            continue

        def mean_with(ws: list[float]) -> FloatPoint:
            acc = FloatPoint(0.0, 0.0)
            acc_w = 0.0
            for (p, w0), w1 in zip(samples, ws):
                w = float(w0) * float(w1)
                if w <= 0.0:
                    continue
                acc = acc + p * w
                acc_w += w
            return current_pt if acc_w <= 0.0 else (acc / acc_w)

        m0 = mean_with([1.0 for _ in samples])
        ws2: list[float] = []
        for p, _w in samples:
            r = p.distance_to(m0)
            if r <= robust_c_px:
                g = 1.0
            else:
                g = robust_c_px / max(1e-9, float(r))
            ws2.append(float(g))
        m1 = mean_with(ws2)
        out.append(m1)

    return out


def _solve_linear_wls(samples: list[tuple[int, float]], weights: list[float]) -> tuple[float, float]:
    # Fit y = b0 + b1*k
    s0 = s1 = s2 = 0.0
    t0 = t1 = 0.0
    for (k, y), w in zip(samples, weights):
        kk = float(k)
        w = float(w)
        s0 += w
        s1 += w * kk
        s2 += w * kk * kk
        t0 += w * float(y)
        t1 += w * kk * float(y)
    det = s0 * s2 - s1 * s1
    if abs(det) < 1e-9 or s0 <= 0.0:
        mu = t0 / max(1e-9, s0)
        return (float(mu), 0.0)
    b0 = (t0 * s2 - t1 * s1) / det
    b1 = (t1 * s0 - t0 * s1) / det
    return (float(b0), float(b1))


def _solve_quadratic_wls(samples: list[tuple[int, float]], weights: list[float]) -> tuple[float, float, float]:
    # Fit y = b0 + b1*k + b2*k^2 using normal equations and Gaussian elimination.
    a00 = a01 = a02 = 0.0
    a11 = a12 = 0.0
    a22 = 0.0
    b0 = b1 = b2 = 0.0
    for (k, y), w in zip(samples, weights):
        kk = float(k)
        k2 = kk * kk
        w = float(w)
        a00 += w
        a01 += w * kk
        a02 += w * k2
        a11 += w * kk * kk
        a12 += w * kk * k2
        a22 += w * k2 * k2
        yy = float(y)
        b0 += w * yy
        b1 += w * kk * yy
        b2 += w * k2 * yy

    M = [
        [a00, a01, a02, b0],
        [a01, a11, a12, b1],
        [a02, a12, a22, b2],
    ]

    for col in range(3):
        pivot = col
        for r in range(col + 1, 3):
            if abs(M[r][col]) > abs(M[pivot][col]):
                pivot = r
        if abs(M[pivot][col]) < 1e-9:
            b0_lin, b1_lin = _solve_linear_wls(samples, weights)
            return (float(b0_lin), float(b1_lin), 0.0)
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]

        inv = 1.0 / M[col][col]
        for c in range(col, 4):
            M[col][c] *= inv
        for r in range(3):
            if r == col:
                continue
            factor = M[r][col]
            if factor == 0.0:
                continue
            for c in range(col, 4):
                M[r][c] -= factor * M[col][c]

    return (float(M[0][3]), float(M[1][3]), float(M[2][3]))


def _smooth_anchors_local_fit(
    points: list[FloatPoint],
    *,
    frame_start: int,
    raw_motion_transforms: Optional[list[Transform]],
    transform_index_offset: int,
    neighbor_weights: dict[int, float],
    model: str,
    robust_c_px: float,
    max_neighbor_distance: Optional[float] = None,
) -> list[FloatPoint]:
    if not points:
        return []

    weights = dict(neighbor_weights)
    weights.setdefault(0, 1.0)

    model = str(model).lower().strip()
    if model not in {'linear', 'quadratic'}:
        model = 'linear'
    robust_c_px = float(robust_c_px)

    motion: Optional[MotionModel] = None
    if raw_motion_transforms is not None and len(points) >= 2:
        motion = MotionModel(
            raw_motion_transforms=raw_motion_transforms,
            start_frame=int(frame_start),
            frame_count=int(len(points)),
            transform_index_offset=int(transform_index_offset),
        )

    out: list[FloatPoint] = []
    for i in range(len(points)):
        current_abs = int(frame_start + i)
        current_pt = points[i]

        ks: list[int] = []
        xs: list[float] = []
        ys: list[float] = []
        base_w: list[float] = []

        for rel_off, w in weights.items():
            j = int(i + int(rel_off))
            if j < 0 or j >= len(points):
                continue
            w = float(w)
            if w <= 0.0:
                continue

            neighbor = points[j]
            if motion is not None:
                neighbor_abs = int(frame_start + j)
                neighbor = motion.map_point_abs(neighbor_abs, current_abs, neighbor)

            if max_neighbor_distance is not None and neighbor.distance_to(current_pt) > float(max_neighbor_distance):
                continue

            ks.append(int(rel_off))
            xs.append(float(neighbor.x))
            ys.append(float(neighbor.y))
            base_w.append(float(w))

        if not ks:
            out.append(current_pt)
            continue

        def fit_for(values: list[float], w_in: list[float]) -> float:
            paired = list(zip(ks, values))
            if model == 'quadratic' and len(paired) >= 3:
                b0, _b1, _b2 = _solve_quadratic_wls(paired, w_in)
                return float(b0)
            b0, _b1 = _solve_linear_wls(paired, w_in)
            return float(b0)

        x0 = fit_for(xs, base_w)
        y0 = fit_for(ys, base_w)

        if robust_c_px > 0.0 and len(ks) >= 3:
            # One IRLS step with Huber weights on 2D residuals.
            w2: list[float] = []
            for x, y, w in zip(xs, ys, base_w):
                r = math.hypot(float(x) - float(x0), float(y) - float(y0))
                if r <= robust_c_px:
                    g = 1.0
                else:
                    g = robust_c_px / max(1e-9, r)
                w2.append(float(w) * float(g))
            x0 = fit_for(xs, w2)
            y0 = fit_for(ys, w2)

        out.append(FloatPoint(x=float(x0), y=float(y0)))

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
    raw_motion_transforms: Optional[list[Transform]] = None,
    anchor_smooth_transform_index_offset: int = ANCHOR_SMOOTH_TRANSFORM_INDEX_OFFSET,
    anchor_smooth_neighbor_weights: dict[int, float] = ANCHOR_SMOOTH_NEIGHBOR_WEIGHTS,
) -> list[Point]:
    """
    Compute per-frame anchors for a dense, per-frame list of detections.

    Behavior:
      - Prefer a point along the mast segment when boom is visible.
      - Short gaps (<= missing_grace_frames) between two visible boom keypoints are interpolated (anchor->anchor).
        This interpolation is suppressed if it would move the anchor more than 5% of the frame diagonal in one step
        (to avoid large jumps from outlier detections).
      - Longer missing spans blend to/from bbox centers over missing_grace_frames to avoid snapping.
      - Final anchors are smoothed with a fixed weighted window [prev:1, curr:4, next:0.5].
    """
    assert missing_grace_frames > 0
    assert video_width > 0
    assert video_height > 0

    max_anchor_step_px: Optional[float]
    if ANCHOR_SMOOTH_MAX_NEIGHBOR_DISTANCE_FRAME_PERCENT is None:
        max_anchor_step_px = None
    else:
        max_anchor_step_px = float(ANCHOR_SMOOTH_MAX_NEIGHBOR_DISTANCE_FRAME_PERCENT) * math.hypot(
            float(video_width), float(video_height)
        )

    bbox_centers = [_to_float_point(d.bbox.center) for d in detections]
    boom_vis = [d.boom.is_visible for d in detections]
    mast_vis = [d.mast_tip.is_visible for d in detections]
    visible_indices = [i for i, v in enumerate(boom_vis) if v]

    bbox_top_y = [int(d.bbox.y1) for d in detections]
    boom_or_center_x = [
        int(d.boom.point.x) if boom_vis[i] else int(bbox_centers[i].x) for i, d in enumerate(detections)
    ]

    def proxy_mast_tip(i: int) -> FloatPoint:
        # Proxy when the mast tip keypoint isn't visible: use bbox top edge with a reasonable X.
        return FloatPoint(float(boom_or_center_x[i]), float(bbox_top_y[i]))

    def fill_mast_tip_points() -> list[FloatPoint]:
        """
        Build a dense per-frame mast-tip point series, using:
          - visible mast tips when available,
          - interpolation for short gaps,
          - bbox-top proxy for long gaps, with interpolation into/out of proxy around re-appearance.
        """
        grace = int(missing_grace_frames)
        tip_vis = [bool(v) for v in mast_vis]
        visible_tip_indices = [i for i, v in enumerate(tip_vis) if v]

        # Default: proxy everywhere.
        tip: list[FloatPoint] = [proxy_mast_tip(i) for i in range(len(detections))]
        if not visible_tip_indices:
            return tip

        # Place visible tips.
        for i in visible_tip_indices:
            tip[i] = _to_float_point(detections[i].mast_tip.point)

        # Interpolate short gaps between visible tips.
        for a_i, b_i in zip(visible_tip_indices[:-1], visible_tip_indices[1:]):
            gap = b_i - a_i - 1
            if 0 < gap <= grace:
                p0 = _to_float_point(detections[a_i].mast_tip.point)
                p1 = _to_float_point(detections[b_i].mast_tip.point)
                for k in range(1, gap + 1):
                    t = float(k) / float(gap + 1)
                    tip[a_i + k] = p0.interpolate(p1, t)

        # Leading: if first visible is far away, interpolate proxy->tip over the last `grace` frames before it.
        first = visible_tip_indices[0]
        if first > grace:
            start_idx = first - grace
            p0 = proxy_mast_tip(start_idx)
            p1 = _to_float_point(detections[first].mast_tip.point)
            for idx in range(start_idx, first):
                t = float(idx - start_idx) / float(grace)
                tip[idx] = p0.interpolate(p1, t)

        # Trailing: if last visible is far away from the end, interpolate tip->proxy over the next `grace` frames.
        last = visible_tip_indices[-1]
        trailing = (len(detections) - 1) - last
        if trailing > grace:
            end_idx = last + grace
            p0 = _to_float_point(detections[last].mast_tip.point)
            p1 = proxy_mast_tip(end_idx)
            for idx in range(last + 1, end_idx + 1):
                t = float(idx - last) / float(grace)
                tip[idx] = p0.interpolate(p1, t)

        # Internal long gaps: ease out to proxy and back in from proxy.
        for a_i, b_i in zip(visible_tip_indices[:-1], visible_tip_indices[1:]):
            gap = b_i - a_i - 1
            if gap <= grace:
                continue

            out_end = min(len(detections) - 1, a_i + grace)
            in_start = max(0, b_i - grace)

            # Ease tip[a_i] -> proxy(out_end)
            p0 = _to_float_point(detections[a_i].mast_tip.point)
            p1 = proxy_mast_tip(out_end)
            for idx in range(a_i + 1, out_end + 1):
                t = float(idx - a_i) / float(grace)
                tip[idx] = p0.interpolate(p1, t)

            # Middle: proxy (already default).

            # Ease proxy(in_start) -> tip[b_i]
            p0 = proxy_mast_tip(in_start)
            p1 = _to_float_point(detections[b_i].mast_tip.point)
            for idx in range(in_start, b_i):
                t = float(idx - in_start) / float(grace)
                tip[idx] = p0.interpolate(p1, t)

        return tip

    filled_mast_tip = fill_mast_tip_points()

    def preferred_anchor(i: int) -> FloatPoint:
        # With a centered crop, anchoring on the boom can cut off the mast tip (boom is close to one end of the mast).
        # Use a point along the mast segment; when the true mast tip isn't visible, `filled_mast_tip` eases into a
        # bbox-top proxy and back out, avoiding large jumps when the mast reappears.
        if boom_vis[i]:
            tip = filled_mast_tip[i]
            boom = _to_float_point(detections[i].boom.point)
            # Bias strongly towards the boom so the anchor stays close to the rider/boom while
            # still leaving headroom for the mast tip in the crop.
            return tip.interpolate(boom, 0.85)
        return bbox_centers[i]

    method = str(ANCHOR_SMOOTH_METHOD).lower().strip()
    use_camera = bool(ANCHOR_SMOOTH_USE_CAMERA_TRANSFORMS) and raw_motion_transforms is not None
    transforms_for_smoothing = raw_motion_transforms if use_camera else None

    if not visible_indices:
        if method == 'robust_mean':
            smoothed = _anchor_robust_mean(
                bbox_centers,
                frame_start=int(detections[0].frame_idx) if detections else 0,
                raw_motion_transforms=transforms_for_smoothing,
                transform_index_offset=int(anchor_smooth_transform_index_offset),
                neighbor_weights=anchor_smooth_neighbor_weights,
                robust_c_px=float(ANCHOR_SMOOTH_ROBUST_MEAN_C_PX),
                max_neighbor_distance=max_anchor_step_px,
            )
        elif method == 'local_fit':
            smoothed = _smooth_anchors_local_fit(
                bbox_centers,
                frame_start=int(detections[0].frame_idx) if detections else 0,
                raw_motion_transforms=transforms_for_smoothing,
                transform_index_offset=int(anchor_smooth_transform_index_offset),
                neighbor_weights=anchor_smooth_neighbor_weights,
                model=ANCHOR_SMOOTH_MODEL,
                robust_c_px=ANCHOR_SMOOTH_ROBUST_C_PX,
                max_neighbor_distance=max_anchor_step_px,
            )
        else:
            smoothed = _anchor_weighted(
                bbox_centers,
                frame_start=int(detections[0].frame_idx) if detections else 0,
                raw_motion_transforms=transforms_for_smoothing,
                transform_index_offset=int(anchor_smooth_transform_index_offset),
                neighbor_weights=anchor_smooth_neighbor_weights,
                max_neighbor_distance=max_anchor_step_px,
            )
        return [Point(p.x, p.y) for p in smoothed]

    anchors = list(bbox_centers)
    visible_set = set(visible_indices)
    for i in visible_indices:
        anchors[i] = preferred_anchor(i)

    short_gap_indices: set[int] = set()
    for a_i, b_i in zip(visible_indices[:-1], visible_indices[1:]):
        gap = b_i - a_i - 1
        if 0 < gap <= missing_grace_frames:
            p0 = preferred_anchor(a_i)
            p1 = preferred_anchor(b_i)
            current_anchor = anchors[a_i]
            for k in range(1, gap + 1):
                t = float(k) / float(gap + 1)
                idx = a_i + k
                candidate = p0.interpolate(p1, t)
                if candidate.distance_to(current_anchor) <= max_anchor_step_px:
                    anchors[idx] = candidate
                    short_gap_indices.add(idx)
                current_anchor = anchors[idx]

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

    # Leading missing: if > grace, interpolate bbox(center at first-grace) -> anchor (at first)
    first = visible_indices[0]
    if first > missing_grace_frames:
        start_idx = first - missing_grace_frames
        start_pt = bbox_centers[start_idx]
        end_pt = preferred_anchor(first)
        for idx in range(start_idx, first):
            if idx in visible_set or idx in short_gap_indices:
                continue
            t = float(idx - start_idx) / float(missing_grace_frames)
            set_in(idx, start_pt.interpolate(end_pt, t), w=float(missing_grace_frames - (first - idx) + 1))

    # Internal long gaps: anchor->bbox then bbox->anchor (with overlap blending if needed)
    for a_i, b_i in zip(visible_indices[:-1], visible_indices[1:]):
        gap = b_i - a_i - 1
        if gap <= missing_grace_frames:
            continue

        # Out: anchor at a_i -> bbox center at a_i+grace
        out_target_idx = a_i + missing_grace_frames
        out_target = bbox_centers[out_target_idx]
        for idx in range(a_i + 1, min(out_target_idx, b_i - 1) + 1):
            if idx in visible_set or idx in short_gap_indices:
                continue
            t = float(idx - a_i) / float(missing_grace_frames)
            set_out(
                idx,
                preferred_anchor(a_i).interpolate(out_target, t),
                w=float(missing_grace_frames - (idx - a_i) + 1),
            )

        # In: bbox center at b_i-grace -> anchor at b_i
        in_start_idx = b_i - missing_grace_frames
        in_start = bbox_centers[in_start_idx]
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
    # - > grace: interpolate anchor -> bbox center at last+grace
    last = visible_indices[-1]
    trailing = (len(detections) - 1) - last
    if trailing <= missing_grace_frames:
        for idx in range(last + 1, len(detections)):
            anchors[idx] = anchors[last]
    else:
        out_target_idx = last + missing_grace_frames
        out_target = bbox_centers[out_target_idx]
        for idx in range(last + 1, out_target_idx + 1):
            if idx in visible_set or idx in short_gap_indices:
                continue
            t = float(idx - last) / float(missing_grace_frames)
            set_out(
                idx,
                preferred_anchor(last).interpolate(out_target, t),
                w=float(missing_grace_frames - (idx - last) + 1),
            )
        # Beyond out_target_idx stays at bbox centers (default).

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

    if method == 'robust_mean':
        smoothed = _anchor_robust_mean(
            anchors,
            frame_start=int(detections[0].frame_idx),
            raw_motion_transforms=transforms_for_smoothing,
            transform_index_offset=int(anchor_smooth_transform_index_offset),
            neighbor_weights=anchor_smooth_neighbor_weights,
            robust_c_px=float(ANCHOR_SMOOTH_ROBUST_MEAN_C_PX),
            max_neighbor_distance=max_anchor_step_px,
        )
    elif method == 'local_fit':
        smoothed = _smooth_anchors_local_fit(
            anchors,
            frame_start=int(detections[0].frame_idx),
            raw_motion_transforms=transforms_for_smoothing,
            transform_index_offset=int(anchor_smooth_transform_index_offset),
            neighbor_weights=anchor_smooth_neighbor_weights,
            model=ANCHOR_SMOOTH_MODEL,
            robust_c_px=ANCHOR_SMOOTH_ROBUST_C_PX,
            max_neighbor_distance=max_anchor_step_px,
        )
    else:
        smoothed = _anchor_weighted(
            anchors,
            frame_start=int(detections[0].frame_idx),
            raw_motion_transforms=transforms_for_smoothing,
            transform_index_offset=int(anchor_smooth_transform_index_offset),
            neighbor_weights=anchor_smooth_neighbor_weights,
            max_neighbor_distance=max_anchor_step_px,
        )
    # Return `Point` for backward-compat with downstream code, but keep float precision in x/y.
    return [Point(p.x, p.y) for p in smoothed]


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
        # Keep in float space; anchors may be subpixel floats (stored in Point for compat).
        return float(abs(float(getattr(anchors[i], 'y')) - float(detections[i].bbox.y1)))

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
    missing_grace_frames: int = 5,
    mast_smooth_radius: int = 15,
    target_mast_fill: float = 0.75,
    raw_motion_transforms: Optional[list[Transform]] = None,
    anchor_smooth_transform_index_offset: int = ANCHOR_SMOOTH_TRANSFORM_INDEX_OFFSET,
    anchor_smooth_neighbor_weights: dict[int, float] = ANCHOR_SMOOTH_NEIGHBOR_WEIGHTS,
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
            anchor_smooth_transform_index_offset=int(anchor_smooth_transform_index_offset),
            anchor_smooth_neighbor_weights=anchor_smooth_neighbor_weights,
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
