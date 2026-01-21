"""
Renderable track preparation.

This module converts dense per-frame `Track` objects into `RenderableTrack`s by computing:
  - an anchor point per frame (for stable camera anchoring),
  - a per-frame scale factor (to normalize mast length within a track).
"""

from __future__ import annotations

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


def _anchor_weighted(points: list[Point]) -> list[Point]:
    out: list[Point] = []
    for i in range(len(points)):
        acc = Point(0, 0)
        acc_w = 0.0
        if i - 1 >= 0:
            acc += points[i - 1] * 1.0
            acc_w += 1.0
        acc += points[i] * 4.0
        acc_w += 4.0
        if i + 1 < len(points):
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


def calculate_anchors(detections: list[Detection], *, missing_grace_frames: int) -> list[Point]:
    """
    Compute per-frame anchors for a dense, per-frame list of detections.

    Behavior:
      - Prefer boom keypoint when visible.
      - Short gaps (<= missing_grace_frames) between two visible boom keypoints are interpolated (anchor->anchor).
      - Longer missing spans blend to/from bbox centers over missing_grace_frames to avoid snapping.
      - Final anchors are smoothed with a fixed weighted window [prev:1, curr:4, next:0.5].
    """
    assert missing_grace_frames > 0

    bbox_centers = [d.bbox.center for d in detections]
    boom_vis = [d.boom.is_visible for d in detections]
    visible_indices = [i for i, v in enumerate(boom_vis) if v]

    if not visible_indices:
        return _anchor_weighted(bbox_centers)

    anchors = list(bbox_centers)
    visible_set = set(visible_indices)
    for i in visible_indices:
        anchors[i] = detections[i].boom.point

    short_gap_indices: set[int] = set()
    for a_i, b_i in zip(visible_indices[:-1], visible_indices[1:]):
        gap = b_i - a_i - 1
        if 0 < gap <= missing_grace_frames:
            p0 = detections[a_i].boom.point
            p1 = detections[b_i].boom.point
            for k in range(1, gap + 1):
                t = float(k) / float(gap + 1)
                idx = a_i + k
                anchors[idx] = p0.interpolate(p1, t)
                short_gap_indices.add(idx)

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
        end_pt = detections[first].boom.point
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
                detections[a_i].boom.point.interpolate(out_target, t),
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
                in_start.interpolate(detections[b_i].boom.point, t),
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
                detections[last].boom.point.interpolate(out_target, t),
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

    return _anchor_weighted(anchors)


def calculate_scales(
    detections: list[Detection],
    anchors: list[Point],
    *,
    missing_grace_frames: int,
    mast_smooth_radius: int,
) -> list[float]:
    """
    Compute per-frame scaling factors for a dense, per-frame list of detections.

    The "length" signal is preferred as dist(boom, mast_tip) when both keypoints are visible.
    Missing spans are filled with a fallback using bbox top edge and anchors; then smoothed.
    Scales are normalized per-track so the reference mast length is constant across frames.
    """
    assert missing_grace_frames > 0
    assert mast_smooth_radius > 0
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
    if not visible_len_indices:
        filled_len = [bbox_fallback_len(i) for i in range(len(detections))]
    else:
        filled_len: list[float] = [bbox_fallback_len(i) for i in range(len(detections))]
        for i in visible_len_indices:
            assert raw_len[i] is not None
            filled_len[i] = float(raw_len[i])

        # interpolate short gaps between visible lengths
        for a_i, b_i in zip(visible_len_indices[:-1], visible_len_indices[1:]):
            gap = b_i - a_i - 1
            if 0 < gap <= grace:
                v0 = float(filled_len[a_i])
                v1 = float(filled_len[b_i])
                for k in range(1, gap + 1):
                    t = float(k) / float(gap + 1)
                    filled_len[a_i + k] = float(interpolate(v0, v1, t))

        # trailing missing: hold last visible up to grace, then bbox fallback
        last = visible_len_indices[-1]
        for i in range(last + 1, len(detections)):
            if (i - last) <= grace:
                filled_len[i] = float(filled_len[last])
            else:
                filled_len[i] = bbox_fallback_len(i)

    smoothed_len = _smooth_mean(filled_len, int(mast_smooth_radius))
    ref_candidates = [smoothed_len[i] for i in range(len(detections)) if len_vis[i]]
    ref_len = _median(ref_candidates if ref_candidates else smoothed_len)
    ref_len = max(1.0, float(ref_len))

    return [float(ref_len / max(1.0, float(length))) for length in smoothed_len]


def prepare_renderable_tracks(
    tracks: list[Track],
    *,
    missing_grace_frames: int = 5,
    mast_smooth_radius: int = 15,
) -> list[RenderableTrack]:
    """
    Post-processing step for the rendering pipeline.

    Produces RenderableTrack/RenderableDetection objects with stable per-frame:
      - anchor (Point)
      - scale (float)  (normalized so mast length is constant per track)
    """
    out_tracks: list[RenderableTrack] = []
    for track in tracks:
        detections = track.sorted_detections
        if not detections:
            out_tracks.append(RenderableTrack(track_id=track.track_id, sorted_detections=[]))
            continue

        _assert_dense_per_frame(detections, track.track_id)
        anchors = calculate_anchors(detections, missing_grace_frames=missing_grace_frames)
        scales = calculate_scales(
            detections,
            anchors,
            missing_grace_frames=missing_grace_frames,
            mast_smooth_radius=mast_smooth_radius,
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
