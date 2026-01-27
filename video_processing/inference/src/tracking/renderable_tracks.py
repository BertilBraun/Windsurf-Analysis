"""
Renderable track preparation.

This module converts dense per-frame `Track` objects into `RenderableTrack`s by computing:
  - an anchor point per frame (for stable camera anchoring),
  - a per-frame scale factor (to normalize mast length within a track).
"""

from __future__ import annotations

import math
from typing import Optional

from ..common_types import Detection, Point, RenderableDetection, RenderableTrack, Track, interpolate


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


def _anchor_weighted(points: list[Point], *, max_neighbor_distance: Optional[float] = None) -> list[Point]:
    out: list[Point] = []
    for i in range(len(points)):
        acc = Point(0, 0)
        acc_w = 0.0
        if i - 1 >= 0:
            if max_neighbor_distance is None or points[i - 1].distance_to(points[i]) <= max_neighbor_distance:
                acc += points[i - 1] * 1.0
                acc_w += 1.0
        acc += points[i] * 4.0
        acc_w += 4.0
        if i + 1 < len(points):
            if max_neighbor_distance is None or points[i + 1].distance_to(points[i]) <= max_neighbor_distance:
                acc += points[i + 1] * 0.5
                acc_w += 0.5
        out.append(acc / acc_w)
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

    MAX_DISTANCE_FRAME_PERCENT = 0.05
    max_anchor_step_px = MAX_DISTANCE_FRAME_PERCENT * math.hypot(float(video_width), float(video_height))

    bbox_centers = [d.bbox.center for d in detections]
    boom_vis = [d.boom.is_visible for d in detections]
    mast_vis = [d.mast_tip.is_visible for d in detections]
    visible_indices = [i for i, v in enumerate(boom_vis) if v]

    bbox_top_y = [int(d.bbox.y1) for d in detections]
    boom_or_center_x = [
        int(d.boom.point.x) if boom_vis[i] else int(bbox_centers[i].x) for i, d in enumerate(detections)
    ]

    def proxy_mast_tip(i: int) -> Point:
        # Proxy when the mast tip keypoint isn't visible: use bbox top edge with a reasonable X.
        return Point(int(boom_or_center_x[i]), int(bbox_top_y[i]))

    def fill_mast_tip_points() -> list[Point]:
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
        tip: list[Point] = [proxy_mast_tip(i) for i in range(len(detections))]
        if not visible_tip_indices:
            return tip

        # Place visible tips.
        for i in visible_tip_indices:
            tip[i] = detections[i].mast_tip.point

        # Interpolate short gaps between visible tips.
        for a_i, b_i in zip(visible_tip_indices[:-1], visible_tip_indices[1:]):
            gap = b_i - a_i - 1
            if 0 < gap <= grace:
                p0 = detections[a_i].mast_tip.point
                p1 = detections[b_i].mast_tip.point
                for k in range(1, gap + 1):
                    t = float(k) / float(gap + 1)
                    tip[a_i + k] = p0.interpolate(p1, t)

        # Leading: if first visible is far away, interpolate proxy->tip over the last `grace` frames before it.
        first = visible_tip_indices[0]
        if first > grace:
            start_idx = first - grace
            p0 = proxy_mast_tip(start_idx)
            p1 = detections[first].mast_tip.point
            for idx in range(start_idx, first):
                t = float(idx - start_idx) / float(grace)
                tip[idx] = p0.interpolate(p1, t)

        # Trailing: if last visible is far away from the end, interpolate tip->proxy over the next `grace` frames.
        last = visible_tip_indices[-1]
        trailing = (len(detections) - 1) - last
        if trailing > grace:
            end_idx = last + grace
            p0 = detections[last].mast_tip.point
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
            p0 = detections[a_i].mast_tip.point
            p1 = proxy_mast_tip(out_end)
            for idx in range(a_i + 1, out_end + 1):
                t = float(idx - a_i) / float(grace)
                tip[idx] = p0.interpolate(p1, t)

            # Middle: proxy (already default).

            # Ease proxy(in_start) -> tip[b_i]
            p0 = proxy_mast_tip(in_start)
            p1 = detections[b_i].mast_tip.point
            for idx in range(in_start, b_i):
                t = float(idx - in_start) / float(grace)
                tip[idx] = p0.interpolate(p1, t)

        return tip

    filled_mast_tip = fill_mast_tip_points()

    def preferred_anchor(i: int) -> Point:
        # With a centered crop, anchoring on the boom can cut off the mast tip (boom is close to one end of the mast).
        # Use a point along the mast segment; when the true mast tip isn't visible, `filled_mast_tip` eases into a
        # bbox-top proxy and back out, avoiding large jumps when the mast reappears.
        if boom_vis[i]:
            tip = filled_mast_tip[i]
            boom = detections[i].boom.point
            # Bias strongly towards the boom so the anchor stays close to the rider/boom while
            # still leaving headroom for the mast tip in the crop.
            return tip.interpolate(boom, 0.85)
        return bbox_centers[i]

    if not visible_indices:
        return _anchor_weighted(bbox_centers, max_neighbor_distance=max_anchor_step_px)

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

    out_point: list[Optional[Point]] = [None for _ in detections]
    out_weight: list[float] = [0.0 for _ in detections]
    in_point: list[Optional[Point]] = [None for _ in detections]
    in_weight: list[float] = [0.0 for _ in detections]

    def set_out(idx: int, pt: Point, w: float) -> None:
        if out_point[idx] is None or w > out_weight[idx]:
            out_point[idx] = pt
            out_weight[idx] = float(w)

    def set_in(idx: int, pt: Point, w: float) -> None:
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

    return _anchor_weighted(anchors, max_neighbor_distance=max_anchor_step_px)


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
        return float(abs(int(anchors[i].y) - int(detections[i].bbox.y1)))

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
