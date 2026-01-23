from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import cv2
import numpy as np
from tqdm import tqdm


def _add_video_processing_to_path() -> None:
    import sys

    scripts_dir = Path(__file__).resolve().parent
    video_processing_dir = scripts_dir.parents[0]
    sys.path.insert(0, str(scripts_dir))
    sys.path.insert(0, str(video_processing_dir))


_add_video_processing_to_path()

from inference.src.player.core.player_state import Metadata  # noqa: E402
from inference.src.util.video_io import VideoReader, VideoWriter, get_video_properties  # noqa: E402
from inference.src.visualization.stabilize import (  # noqa: E402
    STABLE_GFTT_BLOCK_SIZE,
    STABLE_GFTT_MAX_CORNERS,
    STABLE_GFTT_MIN_DISTANCE,
    STABLE_GFTT_QUALITY_LEVEL,
    STABLE_SMOOTHING_WINDOW,
)

StabilizerName = Literal['masked_vidstab', 'gmc', 'vidstab', 'none']


@dataclass(frozen=True)
class Detection:
    frame_idx: int
    bbox: tuple[int, int, int, int]  # x1,y1,x2,y2 in pixels
    interpolated: bool
    anchor: tuple[int, int]  # pixel coords
    scale: float  # normalized crop height (0..1) relative to source height


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            'Generate a 3-panel homepage demo video:\n'
            '  left: (top) raw, (bottom) tracking overlays\n'
            '  right: detailed centered rider view (render full-width, then crop center 50% into right half)\n'
            '\n'
            'Use --with-stabilization to also warp frames (experimental).'
        )
    )
    p.add_argument('--input-video', type=str, required=True)
    p.add_argument('--output-video', type=str, default=None)
    p.add_argument('--track-id', type=int, required=True)

    p.add_argument('--pipeline-output-dir', type=str, default='tmp/homepage_demo_run')
    p.add_argument('--reuse-pipeline-output', action='store_true')
    p.add_argument(
        '--stabilization-transforms',
        type=str,
        default=None,
        help='Optional path to `*.stabilization_transforms.json` (required for tracking trail).',
    )

    p.add_argument('--skip-orientation', action='store_true')
    p.add_argument(
        '--orientation-model-path',
        type=str,
        default=None,
        help='Defaults to `video_processing/inference/weights/orientation_fixer/best.pt`.',
    )
    p.add_argument('--yolo-model-path', type=str, default=None)

    p.add_argument(
        '--stabilizer',
        type=str,
        default='masked_vidstab',
        choices=['masked_vidstab', 'gmc', 'vidstab', 'none'],
        help='Only used when generating pipeline outputs (ignored with --reuse-pipeline-output).',
    )
    p.add_argument('--limit-frames', type=int, default=None)
    p.add_argument('--smoothing-window', type=int, default=int(STABLE_SMOOTHING_WINDOW))
    p.add_argument('--mask-margin-px', type=int, default=20)
    p.add_argument('--processing-max-dim', type=float, default=None)
    p.add_argument('--gmc-downscale', type=int, default=2)
    p.add_argument('--use-detector-crops', action='store_true')
    p.add_argument('--masked-max-corners', type=int, default=int(STABLE_GFTT_MAX_CORNERS))
    p.add_argument('--masked-quality-level', type=float, default=float(STABLE_GFTT_QUALITY_LEVEL))
    p.add_argument('--masked-min-distance', type=float, default=float(STABLE_GFTT_MIN_DISTANCE))
    p.add_argument('--masked-block-size', type=int, default=int(STABLE_GFTT_BLOCK_SIZE))

    p.add_argument('--left-zoom', type=float, default=1.0, help='Zoom-in for shaky warp to avoid borders.')
    p.add_argument('--trail-length', type=int, default=30, help='Number of bbox-center points in the trail.')
    p.add_argument('--output-width', type=int, default=None)
    p.add_argument('--output-height', type=int, default=None)
    p.add_argument('--right-zoom-mul', type=float, default=0.85, help='Extra zoom multiplier for detailed crop.')
    p.add_argument(
        '--with-stabilization',
        action='store_true',
        help='Enable stabilization warps (experimental); omit to keep frames unwarped.',
    )
    return p.parse_args()


def _load_metadata(path: Path) -> Metadata:
    with open(path, 'rb') as f:
        return pickle.load(f)


def _load_corrections(path: Path, *, frame_count: int) -> list[tuple[float, float, float]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    transforms = payload.get('transforms', [])
    out: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0) for _ in range(int(frame_count))]
    if isinstance(transforms, list):
        for i, t in enumerate(transforms):
            if not isinstance(t, dict):
                continue
            frame_idx = int(t.get('frame_idx', i))
            if 0 <= frame_idx < len(out):
                out[frame_idx] = (float(t.get('dx', 0.0)), float(t.get('dy', 0.0)), float(t.get('da', 0.0)))
    return out


def _find_first(dir_path: Path, pattern: str) -> Path | None:
    matches = sorted(dir_path.glob(pattern))
    return matches[0] if matches else None


def _extract_track_detections(metadata: Metadata, track_id: int) -> list[Detection]:
    track = next((t for t in metadata.tracks if int(t.track_id) == int(track_id)), None)
    if track is None:
        raise ValueError(f'Track id {track_id} not found in metadata')
    dets: list[Detection] = []
    for d in track.detections:
        x1, y1, x2, y2 = (int(d.bbox[0]), int(d.bbox[1]), int(d.bbox[2]), int(d.bbox[3]))
        anchor = getattr(d, 'anchor', None)
        if isinstance(anchor, list) and len(anchor) >= 2:
            ax, ay = int(anchor[0]), int(anchor[1])
        else:
            ax = int((x1 + x2) * 0.5)
            ay = int((y1 + y2) * 0.5)
        scale = float(getattr(d, 'scale', 1.0))
        dets.append(
            Detection(
                frame_idx=int(d.frame_idx),
                bbox=(x1, y1, x2, y2),
                interpolated=bool(d.interpolated),
                anchor=(ax, ay),
                scale=scale,
            )
        )
    dets.sort(key=lambda x: x.frame_idx)
    return dets


def _closest_detection(dets: list[Detection], frame_idx: int) -> Detection:
    if not dets:
        raise ValueError('Track has no detections')
    lo, hi = 0, len(dets) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if dets[mid].frame_idx < frame_idx:
            lo = mid + 1
        else:
            hi = mid
    cand = dets[lo]
    if lo > 0:
        prev = dets[lo - 1]
        if abs(prev.frame_idx - frame_idx) <= abs(cand.frame_idx - frame_idx):
            return prev
    return cand


def _mat3_translate(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, float(tx)], [0.0, 1.0, float(ty)], [0.0, 0.0, 1.0]], dtype=np.float64)


def _mat3_rotate(da: float) -> np.ndarray:
    c = float(np.cos(float(da)))
    s = float(np.sin(float(da)))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _centered_transform(dx: float, dy: float, da: float) -> np.ndarray:
    # Match the player chain (Canvas/QPainter):
    #   translate(dx,dy) then rotate(da)
    # Canvas composes as CTM = CTM * T; CTM = CTM * R, so the matrix is T @ R.
    return _mat3_translate(dx, dy) @ _mat3_rotate(da)


def _pixel_space_transform(centered: np.ndarray, w: int, h: int) -> np.ndarray:
    cx = (float(w) - 1.0) * 0.5
    cy = (float(h) - 1.0) * 0.5
    return _mat3_translate(cx, cy) @ centered @ _mat3_translate(-cx, -cy)


def _warp_affine(frame_bgr: np.ndarray, src_to_dst: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    inv = np.linalg.inv(src_to_dst)
    M = inv[:2, :].astype(np.float32)
    return cv2.warpAffine(
        frame_bgr,
        M,
        dsize=(int(w), int(h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _zoom_crop(frame_bgr: np.ndarray, zoom: float) -> tuple[np.ndarray, np.ndarray]:
    zoom = float(max(1.0, zoom))
    if zoom <= 1.0 + 1e-9:
        return frame_bgr, np.eye(3, dtype=np.float64)

    h, w = frame_bgr.shape[:2]
    crop_w = max(1, int(round(float(w) / zoom)))
    crop_h = max(1, int(round(float(h) / zoom)))
    x0 = max(0, (w - crop_w) // 2)
    y0 = max(0, (h - crop_h) // 2)
    crop = frame_bgr[y0 : y0 + crop_h, x0 : x0 + crop_w]
    resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LANCZOS4 if zoom > 1.0 else cv2.INTER_AREA)

    sx = float(w) / float(crop_w)
    sy = float(h) / float(crop_h)
    crop_to_full = np.array([[sx, 0.0, -float(x0) * sx], [0.0, sy, -float(y0) * sy], [0.0, 0.0, 1.0]])
    return resized, crop_to_full


def _mat3_scale(sx: float, sy: float) -> np.ndarray:
    return np.array([[float(sx), 0.0, 0.0], [0.0, float(sy), 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _transform_points(M: np.ndarray, pts: Iterable[tuple[float, float]]) -> np.ndarray:
    arr = np.array([[float(x), float(y), 1.0] for x, y in pts], dtype=np.float64).T  # (3,N)
    out = (M @ arr).T  # (N,3)
    return out[:, :2]


def _draw_polyline(img: np.ndarray, pts_xy: np.ndarray, color_bgr: tuple[int, int, int], thickness: int) -> None:
    if pts_xy.shape[0] < 2:
        return
    pts = pts_xy.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color_bgr, thickness=int(thickness), lineType=cv2.LINE_AA)


def _overlay_line(
    img: np.ndarray, p1: tuple[float, float], p2: tuple[float, float], color_bgr: tuple[int, int, int], a: float
) -> None:
    a = float(np.clip(a, 0.0, 1.0))
    if a <= 0:
        return
    overlay = img.copy()
    cv2.line(
        overlay,
        (int(round(p1[0])), int(round(p1[1]))),
        (int(round(p2[0])), int(round(p2[1]))),
        color=color_bgr,
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, a, img, 1.0 - a, 0.0, dst=img)


def _detailed_crop(
    frame_bgr: np.ndarray,
    *,
    out_w: int,
    out_h: int,
    det: Detection,
    zoom_mul: float,
) -> np.ndarray:
    # Match frontend/src/ui/player/rendering.ts + inference/src/player/ui/video_widget.py detailed mode:
    # - crop centered on the pose-derived anchor
    # - det.scale is the normalized crop height (0..1) relative to the source height
    MIN_CROP_NORM = 0.05
    MAX_CROP_NORM = 1.0

    src_h, src_w = frame_bgr.shape[:2]
    ax, ay = det.anchor

    crop_h_norm = float(np.clip(float(det.scale), MIN_CROP_NORM, MAX_CROP_NORM))
    z = float(max(1e-6, float(zoom_mul)))

    crop_h = float(crop_h_norm * float(src_h)) / z
    max_crop_h_from_width = float(float(src_w) * float(out_h) / max(1.0, float(out_w)))
    crop_h = float(np.clip(crop_h, 1.0, min(float(src_h), max_crop_h_from_width)))
    s = float(float(out_h) / max(1e-6, crop_h))

    cx = float(ax)
    cy = float(ay)
    crop_w = float(out_w) / s
    win_x1 = cx - crop_w / 2.0
    win_y1 = cy - crop_h / 2.0
    win_x2 = win_x1 + crop_w
    win_y2 = win_y1 + crop_h

    src_x1 = int(max(0, np.floor(win_x1)))
    src_y1 = int(max(0, np.floor(win_y1)))
    src_x2 = int(min(src_w, np.ceil(win_x2)))
    src_y2 = int(min(src_h, np.ceil(win_y2)))

    dst_x1 = int(max(0, np.floor((float(src_x1) - win_x1) * s)))
    dst_y1 = int(max(0, np.floor((float(src_y1) - win_y1) * s)))
    dst_x2 = int(min(out_w, np.ceil((float(src_x2) - win_x1) * s)))
    dst_y2 = int(min(out_h, np.ceil((float(src_y2) - win_y1) * s)))

    copy_w = max(0, dst_x2 - dst_x1)
    copy_h = max(0, dst_y2 - dst_y1)

    out = np.zeros((int(out_h), int(out_w), 3), dtype=np.uint8)
    if copy_w > 0 and copy_h > 0 and src_x2 > src_x1 and src_y2 > src_y1:
        roi = frame_bgr[src_y1:src_y2, src_x1:src_x2]
        interp = cv2.INTER_LANCZOS4 if s > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(roi, (copy_w, copy_h), interpolation=interp)
        out[dst_y1:dst_y2, dst_x1:dst_x2] = resized
    return out


def _main() -> int:
    args = _parse_args()
    input_video = Path(args.input_video)
    if not input_video.exists():
        raise SystemExit(f'Input video not found: {input_video}')

    apply_stabilization_warp = bool(getattr(args, 'with_stabilization', False))

    pipeline_root = Path(args.pipeline_output_dir)
    run_dir = pipeline_root / input_video.stem

    if args.reuse_pipeline_output:
        upright_video = _find_first(run_dir, '*.upright.mp4')
        metadata_path = _find_first(run_dir, '*.tracks.pkl')
        transforms_path = (
            Path(args.stabilization_transforms)
            if args.stabilization_transforms
            else _find_first(run_dir, '*.stabilization_transforms.json')
        )
        if upright_video is None or metadata_path is None or transforms_path is None:
            raise SystemExit(
                f'Missing pipeline outputs in {run_dir}. Run once without --reuse-pipeline-output to generate them.'
            )
    else:
        import local_modal_pipeline_player  # noqa: E402

        res = local_modal_pipeline_player.run_local_pipeline(
            input_video,
            output_dir=pipeline_root,
            skip_orientation=bool(args.skip_orientation),
            orientation_model_path=args.orientation_model_path,
            yolo_model_path=args.yolo_model_path,
            stabilizer=args.stabilizer,
            processing_max_dim=args.processing_max_dim,
            mask_margin_px=int(args.mask_margin_px),
            gmc_downscale=int(args.gmc_downscale),
            limit_frames=args.limit_frames,
            use_detector_crops=bool(args.use_detector_crops),
            masked_vidstab_debug_dir=None,
            masked_vidstab_debug_every_n=1,
            smoothing_window=int(args.smoothing_window),
            masked_max_corners=int(args.masked_max_corners),
            masked_quality_level=float(args.masked_quality_level),
            masked_min_distance=float(args.masked_min_distance),
            masked_block_size=int(args.masked_block_size),
        )
        upright_video = res.upright_video_path
        metadata_path = res.metadata_path
        transforms_path = (
            Path(args.stabilization_transforms) if args.stabilization_transforms else res.stabilization_transforms_path
        )
        if transforms_path is None:
            raise SystemExit('No stabilization transforms written by pipeline')

    props = get_video_properties(upright_video)
    frame_count = int(props.total_frames)
    if args.limit_frames is not None:
        frame_count = min(frame_count, int(args.limit_frames))

    metadata = _load_metadata(metadata_path)
    dets = _extract_track_detections(metadata, int(args.track_id))
    corrections = _load_corrections(Path(transforms_path), frame_count=frame_count)

    out_w = int(args.output_width) if args.output_width is not None else int(props.width)
    out_h = int(args.output_height) if args.output_height is not None else int(props.height)

    left_w = out_w // 2
    right_w = out_w - left_w
    left_h = out_h // 2
    right_h = out_h

    left_zoom = float(max(1.0, args.left_zoom))
    trail_len = max(2, int(args.trail_length))

    out_path = Path(args.output_video) if args.output_video else run_dir / f'{input_video.stem}.mp4'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with VideoReader(upright_video) as reader, VideoWriter(out_path, out_w, out_h, int(props.fps)) as writer:
        for frame_idx, frame_bgr in tqdm(reader.read_frames(), desc='Processing frames', total=frame_count):
            frame_idx = int(frame_idx)
            if frame_idx >= frame_count:
                break

            frame_bgr = frame_bgr.astype(np.uint8, copy=False)
            if frame_bgr.shape[1] != int(props.width) or frame_bgr.shape[0] != int(props.height):
                frame_bgr = cv2.resize(frame_bgr, (int(props.width), int(props.height)), interpolation=cv2.INTER_AREA)

            dx, dy, da = corrections[frame_idx]

            if apply_stabilization_warp:
                M_draw = _pixel_space_transform(_centered_transform(dx, dy, da), int(props.width), int(props.height))
                M_stable = np.linalg.inv(M_draw)
                M_shaky = M_draw

                stabilized = _warp_affine(frame_bgr, M_stable)
                shaky = _warp_affine(frame_bgr, M_shaky)
                shaky, crop_to_full = _zoom_crop(shaky, left_zoom)
                P_shaky = crop_to_full @ M_shaky
            else:
                stabilized = frame_bgr
                shaky = frame_bgr
                P_shaky = np.eye(3, dtype=np.float64)

            # Left panels
            top_left = cv2.resize(shaky, (left_w, left_h), interpolation=cv2.INTER_AREA)
            bottom_left = top_left.copy()

            det = _closest_detection(dets, frame_idx)
            x1, y1, x2, y2 = det.bbox
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            P_left_panel = (
                _mat3_scale(float(left_w) / float(props.width), float(left_h) / float(props.height)) @ P_shaky
            )
            poly = _transform_points(P_left_panel, corners)
            _draw_polyline(bottom_left, poly, (0, 255, 0), thickness=2)

            # BBox center trail (white, fading)
            P_panel = _mat3_scale(float(left_w) / float(props.width), float(left_h) / float(props.height)) @ P_shaky
            pts: list[tuple[float, float]] = []
            for i in range(trail_len):
                j = max(0, frame_idx - i)
                det_j = _closest_detection(dets, j)
                x1j, y1j, x2j, y2j = det_j.bbox
                cxj = (float(x1j) + float(x2j)) * 0.5
                cyj = (float(y1j) + float(y2j)) * 0.5
                pxy = _transform_points(P_panel, [(cxj, cyj)])[0]
                pts.append((float(pxy[0]), float(pxy[1])))

            for i in range(len(pts) - 1):
                a = max(0.15, 1.0 - float(i) / float(len(pts)))
                _overlay_line(bottom_left, pts[i], pts[i + 1], (255, 255, 255), a)
            cv2.circle(
                bottom_left,
                (int(round(pts[0][0])), int(round(pts[0][1]))),
                3,
                (255, 255, 255),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

            # Right: match player detailed mode (no stabilization): render at full size, then take center 50% width.
            right_full = _detailed_crop(
                frame_bgr,
                out_w=out_w,
                out_h=out_h,
                det=det,
                zoom_mul=float(args.right_zoom_mul),
            )
            x0 = max(0, (out_w - right_w) // 2)
            x1c = min(out_w, x0 + right_w)
            right = right_full[:, x0:x1c]

            out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            out[0:left_h, 0:left_w] = top_left
            out[left_h : left_h + left_h, 0:left_w] = bottom_left
            out[0:right_h, left_w : left_w + right_w] = right

            writer.write_frame(out)

    print('Wrote video to:', out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
