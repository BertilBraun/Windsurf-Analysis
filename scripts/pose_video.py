#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm


def _read_video_fps(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'Failed to open video: {video_path}')
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    return float(fps)


def _read_video_total_frames(video_path: Path) -> int | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    return total if total > 0 else None


def _build_video_writer(output_path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    fourcc_fn = getattr(cv2, 'VideoWriter_fourcc')
    fourcc = fourcc_fn(*('mp4v' if output_path.suffix.lower() in {'.mp4', '.mov', '.m4v'} else 'XVID'))
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'Failed to open VideoWriter for: {output_path}')
    return writer


def _to_numpy(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    obj = data
    if hasattr(obj, 'detach'):
        try:
            obj = obj.detach()
        except Exception:
            pass
    if hasattr(obj, 'cpu'):
        try:
            obj = obj.cpu()
        except Exception:
            pass
    if hasattr(obj, 'numpy'):
        try:
            return obj.numpy()
        except Exception:
            pass
    return np.asarray(obj)


def _serialize_keypoints(result: Any) -> list[dict[str, Any]]:
    if result.keypoints is None or result.keypoints.xy is None:
        return []

    keypoints_xy_np: np.ndarray = _to_numpy(result.keypoints.xy)

    scores = result.keypoints.conf
    scores_np: np.ndarray | None = None
    if scores is not None:
        scores_np = _to_numpy(scores)

    frame_people: list[dict[str, Any]] = []
    num_people = keypoints_xy_np.shape[0]
    for person_idx in range(num_people):
        person_dict: dict[str, Any] = {
            'id': int(person_idx),
            'keypoints': keypoints_xy_np[person_idx].tolist(),  # shape (K, 2)
        }
        if scores_np is not None:
            person_dict['scores'] = scores_np[person_idx].tolist()  # shape (K,)
        frame_people.append(person_dict)

    return frame_people


def run_pose_detection(
    input_video: Path,
    output_video: Path | None,
    output_json: Path | None,
    model_path: str,
    conf: float,
    iou: float,
    imgsz: int,
    stride: int,
    device: str | int,
    augment: bool,
    kp_conf: float,
    min_kpts: int,
    min_box_frac: float,
    require_torso: bool,
    ignore_face: bool,
    ignore_wrists: bool,
    min_persist: int,
) -> None:
    model = YOLO(model_path)

    writer: cv2.VideoWriter | None = None
    serialized_frames: list[dict[str, Any]] = []

    input_fps = _read_video_fps(input_video)
    effective_fps = max(input_fps / max(stride, 1), 1.0)

    # Inference context handled by Ultralytics; no manual eval() to keep types simple
    results = model.predict(
        source=str(input_video),
        stream=True,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        vid_stride=stride,
        augment=augment,
        verbose=False,
    )

    # === Gating, progress, and temporal persistence helpers ===
    total_frames = _read_video_total_frames(input_video)
    total_iters = int(total_frames / max(stride, 1)) if total_frames is not None else None
    COCO_SKELETON_EDGES: list[tuple[int, int]] = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),  # face
        (5, 6),  # shoulders
        (5, 7),
        (7, 9),  # left arm
        (6, 8),
        (8, 10),  # right arm
        (5, 11),
        (6, 12),  # torso
        (11, 12),  # hips
        (11, 13),
        (13, 15),  # left leg
        (12, 14),
        (14, 16),  # right leg
    ]

    FACE_KPT_IDXS = {0, 1, 2, 3, 4}
    WRIST_KPT_IDXS = {9, 10}

    def draw_keypoints_and_skeleton_gated(
        frame: np.ndarray, kpts_xy: np.ndarray, kpts_scores: np.ndarray | None
    ) -> None:
        K = kpts_xy.shape[0]
        valid = np.ones((K,), dtype=bool)
        if kpts_scores is not None:
            valid &= kpts_scores >= kp_conf
        if ignore_face:
            for i in FACE_KPT_IDXS:
                if i < K:
                    valid[i] = False
        if ignore_wrists:
            for i in WRIST_KPT_IDXS:
                if i < K:
                    valid[i] = False

        for i in range(K):
            if not valid[i]:
                continue
            x, y = kpts_xy[i]
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1, lineType=cv2.LINE_AA)

        for i, j in COCO_SKELETON_EDGES:
            if i < K and j < K and valid[i] and valid[j]:
                x1, y1 = kpts_xy[i]
                x2, y2 = kpts_xy[j]
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1, lineType=cv2.LINE_AA)

    def skeleton_is_valid(kpts_xy: np.ndarray, kpts_scores: np.ndarray | None, frame_w: int, frame_h: int) -> bool:
        K = kpts_xy.shape[0]
        valid = np.ones((K,), dtype=bool)
        if kpts_scores is not None:
            valid &= kpts_scores >= kp_conf
        if ignore_face:
            for i in FACE_KPT_IDXS:
                if i < K:
                    valid[i] = False
        if ignore_wrists:
            for i in WRIST_KPT_IDXS:
                if i < K:
                    valid[i] = False

        if int(np.sum(valid)) < min_kpts:
            return False

        if require_torso:
            must_have = [5, 6, 11, 12]  # shoulders and hips
            for idx in must_have:
                if idx >= K:
                    return False
                if kpts_scores is not None and kpts_scores[idx] < kp_conf:
                    return False

        x1, y1, x2, y2 = compute_bbox_from_keypoints(kpts_xy, valid)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        min_area = min_box_frac * (frame_w * frame_h)
        if area < min_area:
            return False
        return True

    prev_boxes_counts: list[tuple[tuple[int, int, int, int], int]] = []
    iterator = tqdm(results, total=total_iters, desc='Pose', unit='f')
    for frame_index, result in enumerate(iterator):
        frame_bgr: np.ndarray = result.orig_img.copy()
        h, w = frame_bgr.shape[:2]

        if output_video and writer is None:
            writer = _build_video_writer(output_video, w, h, effective_fps)

        # Extract predictions
        frame_people: list[dict[str, Any]] = []
        curr_boxes: list[tuple[int, int, int, int]] = []
        kept: list[tuple[np.ndarray, np.ndarray | None, tuple[int, int, int, int]]] = []
        if result.keypoints is not None and result.keypoints.xy is not None:
            kxy = _to_numpy(result.keypoints.xy)  # (N, K, 2)
            ksc = _to_numpy(result.keypoints.conf) if result.keypoints.conf is not None else None  # (N, K)
            num = kxy.shape[0]
            for i in range(num):
                if not skeleton_is_valid(kxy[i], None if ksc is None else ksc[i], w, h):
                    continue
                # Build mask for bbox
                K = kxy[i].shape[0]
                valid_mask = np.ones((K,), dtype=bool)
                if ksc is not None:
                    valid_mask &= ksc[i] >= kp_conf
                if ignore_face:
                    for idx in FACE_KPT_IDXS:
                        if idx < K:
                            valid_mask[idx] = False
                if ignore_wrists:
                    for idx in WRIST_KPT_IDXS:
                        if idx < K:
                            valid_mask[idx] = False
                bbox = compute_bbox_from_keypoints(kxy[i], valid_mask)
                kept.append((kxy[i], None if ksc is None else ksc[i], bbox))
                curr_boxes.append(bbox)

        # Update persistence
        prev_boxes_counts = update_persistence(prev_boxes_counts, curr_boxes, iou_thresh=0.3)

        # Draw only persisted
        for kpts, scores, bbox in kept:
            count = 1
            for pb, c in prev_boxes_counts:
                if bbox_iou(pb, bbox) >= 0.3:
                    count = c
                    break
            if count < max(1, int(min_persist)):
                continue
            draw_keypoints_and_skeleton_gated(frame_bgr, kpts, scores)
            person_entry: dict[str, Any] = {'id': int(len(frame_people)), 'keypoints': kpts.tolist()}
            if scores is not None:
                person_entry['scores'] = scores.tolist()
            frame_people.append(person_entry)

        serialized_frames.append({'frame_index': int(frame_index * max(stride, 1)), 'people': frame_people})

        if writer is not None:
            writer.write(frame_bgr)

    if writer is not None:
        writer.release()

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open('w', encoding='utf-8') as f:
            json.dump(
                {
                    'source': str(input_video),
                    'model': str(model_path),
                    'imgsz': imgsz,
                    'conf': conf,
                    'iou': iou,
                    'stride': stride,
                    'augment': augment,
                    'kp_conf': kp_conf,
                    'min_kpts': min_kpts,
                    'min_box_frac': min_box_frac,
                    'require_torso': require_torso,
                    'ignore_face': ignore_face,
                    'ignore_wrists': ignore_wrists,
                    'min_persist': min_persist,
                    'frames': serialized_frames,
                },
                f,
                indent=2,
            )


# COCO-17 keypoint skeleton edges for visualization
COCO_SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # face
    (5, 6),  # shoulders
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (5, 11),
    (6, 12),  # torso
    (11, 12),  # hips
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
]


def _expand_and_clip_box(
    x1: float, y1: float, x2: float, y2: float, expand: float, w: int, h: int
) -> tuple[int, int, int, int]:
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = (x2 - x1) * expand
    bh = (y2 - y1) * expand
    nx1 = max(int(cx - 0.5 * bw), 0)
    ny1 = max(int(cy - 0.5 * bh), 0)
    nx2 = min(int(cx + 0.5 * bw), w - 1)
    ny2 = min(int(cy + 0.5 * bh), h - 1)
    return nx1, ny1, nx2, ny2


def _draw_keypoints_and_skeleton(frame: np.ndarray, kpts_xy: np.ndarray, color=(0, 255, 0)) -> None:
    for x, y in kpts_xy:
        cv2.circle(frame, (int(x), int(y)), 2, color, -1, lineType=cv2.LINE_AA)
    for i, j in COCO_SKELETON_EDGES:
        if i < len(kpts_xy) and j < len(kpts_xy):
            x1, y1 = kpts_xy[i]
            x2, y2 = kpts_xy[j]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1, lineType=cv2.LINE_AA)


# Keypoint index groups for COCO-17
FACE_KPT_IDXS = {0, 1, 2, 3, 4}
WRIST_KPT_IDXS = {9, 10}


def compute_bbox_from_keypoints(keypoints_xy: np.ndarray, valid_mask: np.ndarray) -> tuple[int, int, int, int]:
    if not np.any(valid_mask):
        return 0, 0, 0, 0
    pts = keypoints_xy[valid_mask]
    x1 = int(np.min(pts[:, 0]))
    y1 = int(np.min(pts[:, 1]))
    x2 = int(np.max(pts[:, 0]))
    y2 = int(np.max(pts[:, 1]))
    return x1, y1, x2, y2


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def update_persistence(
    prev_boxes_counts: list[tuple[tuple[int, int, int, int], int]],
    curr_boxes: list[tuple[int, int, int, int]],
    iou_thresh: float = 0.3,
) -> list[tuple[tuple[int, int, int, int], int]]:
    updated: list[tuple[tuple[int, int, int, int], int]] = []
    for curr in curr_boxes:
        best = 0.0
        best_count = 0
        for prev_box, prev_count in prev_boxes_counts:
            iou = bbox_iou(prev_box, curr)
            if iou > best:
                best = iou
                best_count = prev_count
        new_count = best_count + 1 if best >= iou_thresh else 1
        updated.append((curr, new_count))
    return updated


def draw_keypoints_and_skeleton_gated(
    frame: np.ndarray,
    kpts_xy: np.ndarray,
    kpts_scores: np.ndarray | None,
    kp_conf: float,
    ignore_face: bool,
    ignore_wrists: bool,
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    K = kpts_xy.shape[0]
    valid = np.ones((K,), dtype=bool)
    if kpts_scores is not None:
        valid &= kpts_scores >= kp_conf
    if ignore_face:
        for i in FACE_KPT_IDXS:
            if i < K:
                valid[i] = False
    if ignore_wrists:
        for i in WRIST_KPT_IDXS:
            if i < K:
                valid[i] = False

    for i in range(K):
        if not valid[i]:
            continue
        x, y = kpts_xy[i]
        cv2.circle(frame, (int(x), int(y)), 2, color, -1, lineType=cv2.LINE_AA)

    for i, j in COCO_SKELETON_EDGES:
        if i < K and j < K and valid[i] and valid[j]:
            x1, y1 = kpts_xy[i]
            x2, y2 = kpts_xy[j]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1, lineType=cv2.LINE_AA)


def skeleton_is_valid(
    kpts_xy: np.ndarray,
    kpts_scores: np.ndarray | None,
    kp_conf: float,
    min_kpts: int,
    frame_w: int,
    frame_h: int,
    min_box_frac: float,
    require_torso: bool,
    ignore_face: bool,
    ignore_wrists: bool,
) -> bool:
    K = kpts_xy.shape[0]
    valid = np.ones((K,), dtype=bool)
    if kpts_scores is not None:
        valid &= kpts_scores >= kp_conf
    if ignore_face:
        for i in FACE_KPT_IDXS:
            if i < K:
                valid[i] = False
    if ignore_wrists:
        for i in WRIST_KPT_IDXS:
            if i < K:
                valid[i] = False

    if int(np.sum(valid)) < min_kpts:
        return False

    if require_torso:
        must_have = [5, 6, 11, 12]  # shoulders and hips
        for idx in must_have:
            if idx >= K:
                return False
            if kpts_scores is not None and kpts_scores[idx] < kp_conf:
                return False

    x1, y1, x2, y2 = compute_bbox_from_keypoints(kpts_xy, valid)
    area = max(0, x2 - x1) * max(0, y2 - y1)
    min_area = min_box_frac * (frame_w * frame_h)
    if area < min_area:
        return False
    return True


def run_pose_with_detector_crops(
    input_video: Path,
    output_video: Path | None,
    output_json: Path | None,
    pose_model_path: str,
    det_model_path: str,
    conf: float,
    det_conf: float,
    iou: float,
    imgsz: int,
    stride: int,
    device: str | int,
    expand: float,
    # performance knobs
    det_imgsz: int,
    det_every: int,
    pose_batch: int,
    max_det: int,
    max_crops: int,
    det_min_box_frac: float,
    kp_conf: float,
    min_kpts: int,
    min_box_frac: float,
    require_torso: bool,
    ignore_face: bool,
    ignore_wrists: bool,
    min_persist: int,
) -> None:
    pose_model = YOLO(pose_model_path)
    det_model = YOLO(det_model_path)
    # Inference context handled by Ultralytics; no manual eval()

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video: {input_video}')

    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    effective_fps = max(input_fps / max(stride, 1), 1.0)

    writer: cv2.VideoWriter | None = None
    if output_video:
        writer = _build_video_writer(output_video, width, height, effective_fps)

    serialized_frames: list[dict[str, Any]] = []

    frame_index = 0
    prev_boxes_counts: list[tuple[tuple[int, int, int, int], int]] = []
    last_bboxes: np.ndarray | None = None
    since_det = 10**9
    # Progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_iters = int(total_frames / max(stride, 1)) if total_frames > 0 else None
    with tqdm(total=total_iters, desc='Det+Pose', unit='f') as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % max(stride, 1) != 0:
                frame_index += 1
                continue

        run_detection = (since_det >= max(1, int(det_every))) or (last_bboxes is None)
        if run_detection:
            det_results = det_model.predict(
                source=[frame],
                conf=det_conf,
                iou=iou,
                imgsz=det_imgsz,
                device=device,
                verbose=False,
                classes=[0],
                max_det=max_det,
            )[0]
            bboxes = _to_numpy(det_results.boxes.xyxy) if det_results.boxes is not None else np.zeros((0, 4))
            # Filter tiny boxes and cap count
            if bboxes.size > 0:
                areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                min_area = det_min_box_frac * (width * height)
                keep = areas >= min_area
                bboxes = bboxes[keep]
                if bboxes.shape[0] > 0:
                    order = np.argsort(-areas[keep])
                    bboxes = bboxes[order[: max(1, int(max_crops))]]
            last_bboxes = bboxes
            since_det = 0
        else:
            bboxes = last_bboxes if last_bboxes is not None else np.zeros((0, 4))
            since_det += 1

        crops: list[np.ndarray] = []
        crop_metas: list[tuple[int, int, int, int]] = []
        for box in bboxes:
            x1, y1, x2, y2 = box.tolist()
            ex1, ey1, ex2, ey2 = _expand_and_clip_box(x1, y1, x2, y2, expand=expand, w=width, h=height)
            crop = frame[ey1:ey2, ex1:ex2]
            if crop.size == 0:
                continue
            crops.append(crop)
            crop_metas.append((ex1, ey1, ex2, ey2))

        frame_people: list[dict[str, Any]] = []
        if crops:
            pose_results = pose_model.predict(
                source=crops, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False, batch=pose_batch
            )
            # Map keypoints back to full frame and draw
            kept: list[tuple[np.ndarray, np.ndarray | None, tuple[int, int, int, int]]] = []
            curr_boxes: list[tuple[int, int, int, int]] = []
            for res, meta in zip(pose_results, crop_metas):
                if res.keypoints is None or res.keypoints.xy is None:
                    continue
                kpts_xy_np: np.ndarray = _to_numpy(res.keypoints.xy)
                kpts_sc_np: np.ndarray | None = (
                    _to_numpy(res.keypoints.conf) if res.keypoints.conf is not None else None
                )
                ex1, ey1, _, _ = meta
                for k in range(kpts_xy_np.shape[0]):
                    k_local = kpts_xy_np[k]
                    k_global = k_local.copy()
                    k_global[:, 0] += ex1
                    k_global[:, 1] += ey1
                    s_global = kpts_sc_np[k] if kpts_sc_np is not None else None
                    # Validate skeleton
                    K = k_global.shape[0]
                    valid_mask = np.ones((K,), dtype=bool)
                    if s_global is not None:
                        valid_mask &= s_global >= kp_conf
                    if ignore_face:
                        for idx in (0, 1, 2, 3, 4):
                            if idx < K:
                                valid_mask[idx] = False
                    if ignore_wrists:
                        for idx in (9, 10):
                            if idx < K:
                                valid_mask[idx] = False
                    # Simple validity checks
                    num_valid = int(np.sum(valid_mask))
                    if num_valid < min_kpts:
                        continue
                    # Torso requirement
                    if require_torso:
                        for idx in (5, 6, 11, 12):
                            if idx >= K:
                                valid_mask = None
                                break
                            if s_global is not None and s_global[idx] < kp_conf:
                                valid_mask = None
                                break
                        if valid_mask is None:
                            continue
                    # Area threshold
                    x1 = int(np.min(k_global[valid_mask][:, 0]))
                    y1 = int(np.min(k_global[valid_mask][:, 1]))
                    x2 = int(np.max(k_global[valid_mask][:, 0]))
                    y2 = int(np.max(k_global[valid_mask][:, 1]))
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    if area < (min_box_frac * width * height):
                        continue
                    bbox = (x1, y1, x2, y2)
                    kept.append((k_global, s_global, bbox))
                    curr_boxes.append(bbox)

            # Persistence across frames
            prev_boxes_counts = update_persistence(prev_boxes_counts, curr_boxes, iou_thresh=0.3)
            for k_global, s_global, bbox in kept:
                count = 1
                for pb, c in prev_boxes_counts:
                    if bbox_iou(pb, bbox) >= 0.3:
                        count = c
                        break
                if count < max(1, int(min_persist)):
                    continue
                draw_keypoints_and_skeleton_gated(frame, k_global, s_global, kp_conf, ignore_face, ignore_wrists)
                person_entry: dict[str, Any] = {'id': int(len(frame_people)), 'keypoints': k_global.tolist()}
                if s_global is not None:
                    person_entry['scores'] = s_global.tolist()
                frame_people.append(person_entry)

        serialized_frames.append(
            {
                'frame_index': int(frame_index),
                'people': frame_people,
            }
        )

        if writer is not None:
            writer.write(frame)

        frame_index += 1
        pbar.update(1)

    cap.release()
    if writer is not None:
        writer.release()

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open('w', encoding='utf-8') as f:
            json.dump(
                {
                    'source': str(input_video),
                    'pose_model': str(pose_model_path),
                    'det_model': str(det_model_path),
                    'imgsz': imgsz,
                    'conf': conf,
                    'det_conf': det_conf,
                    'iou': iou,
                    'stride': stride,
                    'expand': expand,
                    'kp_conf': kp_conf,
                    'min_kpts': min_kpts,
                    'min_box_frac': min_box_frac,
                    'require_torso': require_torso,
                    'ignore_face': ignore_face,
                    'ignore_wrists': ignore_wrists,
                    'min_persist': min_persist,
                    'frames': serialized_frames,
                },
                f,
                indent=2,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run human pose detection on a video using YOLOv8-pose.')
    parser.add_argument('input', type=Path, help='Path to input video file')
    parser.add_argument(
        '--output-video', type=Path, default=None, help='Path to save annotated video (e.g., output.mp4)'
    )
    parser.add_argument('--output-json', type=Path, default=None, help='Path to save keypoints JSON')
    parser.add_argument(
        '--model', default='yolov8x-pose-p6.pt', help='Ultralytics pose model (e.g., yolov8x-pose-p6.pt)'
    )
    parser.add_argument('--conf', type=float, default=0.15, help='Pose confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--imgsz', type=int, default=1280, help='Inference image size')
    parser.add_argument('--stride', type=int, default=1, help='Process every Nth frame (vid_stride)')
    parser.add_argument('--augment', action='store_true', help='Enable test-time augmentation for pose inference')
    parser.add_argument('--use-detector', action='store_true', help='Use a detector to crop around people before pose')
    parser.add_argument('--det-model', default='yolov8x.pt', help='Detector model for cropping (COCO)')
    parser.add_argument('--det-conf', type=float, default=0.25, help='Detector confidence threshold')
    parser.add_argument('--expand', type=float, default=1.6, help='Box expansion factor for crops')
    # Performance knobs
    parser.add_argument('--det-imgsz', type=int, default=640, help='Detector image size')
    parser.add_argument('--det-every', type=int, default=1, help='Run detector every N processed frames')
    parser.add_argument('--pose-batch', type=int, default=16, help='Pose batch size for crop inference')
    parser.add_argument('--max-det', type=int, default=30, help='Max detections per frame for detector')
    parser.add_argument('--max-crops', type=int, default=12, help='Limit number of person crops per frame')
    parser.add_argument(
        '--det-min-box-frac', type=float, default=8e-4, help='Filter tiny detector boxes by area fraction'
    )
    # Gating and temporal smoothing
    parser.add_argument('--kp-conf', type=float, default=0.3, help='Min keypoint confidence to draw/use')
    parser.add_argument('--min-kpts', type=int, default=6, help='Min number of valid keypoints per person')
    parser.add_argument('--min-box-frac', type=float, default=5e-4, help='Min bbox area as fraction of frame')
    parser.add_argument('--require-torso', action='store_true', default=True, help='Require shoulders and hips')
    parser.add_argument('--no-require-torso', dest='require_torso', action='store_false')
    parser.add_argument('--ignore-face', action='store_true', default=True, help='Ignore face keypoints/edges')
    parser.add_argument('--no-ignore-face', dest='ignore_face', action='store_false')
    parser.add_argument('--ignore-wrists', action='store_true', default=True, help='Ignore wrist keypoints/edges')
    parser.add_argument('--no-ignore-wrists', dest='ignore_wrists', action='store_false')
    parser.add_argument('--min-persist', type=int, default=2, help='Require detection to persist N frames')
    parser.add_argument('--cpu', action='store_true', help='Force CPU even if CUDA is available')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device: str | int = 'cpu' if args.cpu or not torch.cuda.is_available() else 0

    if not args.output_video and not args.output_json:
        # Default to writing an annotated video next to the input
        inferred_out = args.input.with_name(args.input.stem + '_pose.mp4')
        args.output_video = inferred_out

    if args.use_detector:
        run_pose_with_detector_crops(
            input_video=args.input,
            output_video=args.output_video,
            output_json=args.output_json,
            pose_model_path=args.model,
            det_model_path=args.det_model,
            conf=args.conf,
            det_conf=args.det_conf,
            iou=args.iou,
            imgsz=args.imgsz,
            stride=max(1, int(args.stride)),
            device=device,
            expand=args.expand,
            det_imgsz=args.det_imgsz,
            det_every=max(1, int(args.det_every)),
            pose_batch=max(1, int(args.pose_batch)),
            max_det=max(1, int(args.max_det)),
            max_crops=max(1, int(args.max_crops)),
            det_min_box_frac=float(args.det_min_box_frac),
            kp_conf=args.kp_conf,
            min_kpts=args.min_kpts,
            min_box_frac=args.min_box_frac,
            require_torso=args.require_torso,
            ignore_face=args.ignore_face,
            ignore_wrists=args.ignore_wrists,
            min_persist=args.min_persist,
        )
    else:
        run_pose_detection(
            input_video=args.input,
            output_video=args.output_video,
            output_json=args.output_json,
            model_path=args.model,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            stride=max(1, int(args.stride)),
            device=device,
            augment=bool(args.augment),
            kp_conf=args.kp_conf,
            min_kpts=args.min_kpts,
            min_box_frac=args.min_box_frac,
            require_torso=args.require_torso,
            ignore_face=args.ignore_face,
            ignore_wrists=args.ignore_wrists,
            min_persist=args.min_persist,
        )


if __name__ == '__main__':
    main()
