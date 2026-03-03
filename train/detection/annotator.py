#!/usr/bin/env python3
"""
annotate.py – minimal multi-box annotator with grow/shrink toggle

Mouse:
    LMB hold + move : "paint" to grow bbox (bbox expands to include your path)

Keys:
    r         : undo last box
    Space     : accept / save frame
    w/a/s/d  : move/resize selected box edge (mode: grow)
    W/A/S/D  : move/resize selected box edge (mode: shrink)
    q / e     : rotate 90° CCW / CW (also clickable buttons)
    Esc       : quit
    ','       : previous frame
    '.'       : next frame
    'x'       : skip frame
    'f'       : empty frame
    backspace : undo last save
"""

import os
import re
import cv2
import random
import argparse
import numpy as np
from typing import Optional, Tuple
from screen_utils import get_screen_size, get_window_monitor_size, overlay_screen_warning

try:
    from hard_windows import HardWindow, load_hard_windows
except ImportError:
    from train.detection.hard_windows import HardWindow, load_hard_windows

from pathlib import Path

# ---------- CLI -------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('video_dir', type=Path)
ap.add_argument('output_dir', type=Path)
ap.add_argument('--samples', type=int, default=2000)
ap.add_argument(
    '--windows-file',
    type=Path,
    default=None,
    help='Optional hard-window file. If set, new samples use listed peak frames instead of random frame draws.',
)
args = ap.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

VID_EXT = {'.mp4', '.mov', '.avi', '.mkv'}
ADJUST_BB_SIZE = 0.02  # 1% of current bounding box
videos = [p for p in args.video_dir.rglob('*') if p.suffix.lower() in VID_EXT]
if not videos and args.windows_file is None:
    raise SystemExit('No videos found in', args.video_dir)

fcounts = []
for v in videos:
    c = cv2.VideoCapture(str(v))
    fcounts.append(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) or 1)
    c.release()
weights = [c / sum(fcounts) for c in fcounts] if fcounts else []
frame_count_cache = {str(v.resolve()): c for v, c in zip(videos, fcounts)}
hard_windows: list[HardWindow] = load_hard_windows(args.windows_file) if args.windows_file is not None else []
hard_window_idx = 0
active_hard_window: Optional[HardWindow] = None
if args.windows_file is not None and not hard_windows:
    raise SystemExit('No valid hard windows found in', args.windows_file)


def resize_to_max(img, max_side=2048) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img, scale


def resize_to_fit_screen(
    img: np.ndarray,
    screen_size: Optional[Tuple[int, int]],
    margins: Tuple[int, int] = (50, 50),
    max_side: int = 8096,
) -> Tuple[np.ndarray, float]:
    """Resize up/down so the image fits within the screen (with margins)."""
    if img is None:
        return img, 1.0

    h, w = img.shape[:2]
    max_w = None
    max_h = None
    if screen_size is not None:
        screen_w, screen_h = screen_size
        margin_w, margin_h = margins
        max_w = max(200, int(screen_w - margin_w))
        max_h = max(200, int(screen_h - margin_h))

    if max_w is None or max_h is None:
        # Screen size unknown: only downscale to max_side (avoid surprise upscaling).
        scale = min(1.0, max_side / max(h, w))
    else:
        # Fit within screen (no crop), but allow upscaling to fill the available space.
        scale_screen = min(max_w / max(1, w), max_h / max(1, h))
        scale_cap = max_side / max(h, w)
        scale = min(scale_screen, scale_cap)

    scale = max(0.01, float(scale))
    if abs(scale - 1.0) > 1e-6:
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    return img, scale


# ---------- globals ---------------------------------------------------------
drawing = False
mx, my = 0, 0
boxes = []  # list of [x1,y1,x2,y2] floats in ORIGINAL image pixel coords
img: Optional[np.ndarray] = None
disp: Optional[np.ndarray] = None
orig_img: Optional[np.ndarray] = None
scale_factor: float = 1.0
last_saved: Optional[Tuple[Path, int, int]] = None  # (vpath, frame_no, sid)
fallback_screen_size: Optional[Tuple[int, int]] = get_screen_size()
rotation_k: int = 0  # number of 90° clockwise rotations applied (0..3)
paint_box: Optional[Tuple[float, float, float, float]] = None  # in ORIGINAL coords while drawing

UI_PAD = 10
UI_BTN_W = 150
UI_BTN_H = 34
UI_GAP = 10


def _ui_button_rects():
    ccw = (UI_PAD, UI_PAD, UI_PAD + UI_BTN_W, UI_PAD + UI_BTN_H)
    cw = (UI_PAD + UI_BTN_W + UI_GAP, UI_PAD, UI_PAD + 2 * UI_BTN_W + UI_GAP, UI_PAD + UI_BTN_H)
    return ccw, cw


def _pt_in_rect(x: int, y: int, rect) -> bool:
    x1, y1, x2, y2 = rect
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def _draw_ui(canvas: np.ndarray) -> np.ndarray:
    """Overlay rotate buttons + current rotation on the canvas (in place)."""
    global rotation_k
    if canvas is None:
        return canvas
    H, W = canvas.shape[:2]
    ccw, cw = _ui_button_rects()

    # Only draw if it fits reasonably
    if ccw[2] > W - 2 or cw[2] > W - 2 or ccw[3] > H - 2:
        return canvas

    def draw_btn(rect, label):
        x1, y1, x2, y2 = rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 30, 30), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (220, 220, 220), 1)
        cv2.putText(
            canvas,
            label,
            (x1 + 10, y2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

    draw_btn(ccw, 'Rotate CCW (Q)')
    draw_btn(cw, 'Rotate CW (E)')

    rot = (rotation_k % 4) * 90
    cv2.putText(
        canvas,
        f'Rotation: {rot} deg',
        (UI_PAD, UI_PAD + UI_BTN_H + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f'Rotation: {rot} deg',
        (UI_PAD, UI_PAD + UI_BTN_H + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _rotate_boxes_ccw(bx_list, w: int, h: int):
    # CCW: (x, y) -> (h - 1 - y, x), new size (w' = h, h' = w)
    out = []
    for x1, y1, x2, y2 in bx_list:
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        pts = [(h - 1 - y, x) for (x, y) in corners]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        out.append([float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))])
    return out


def _rotate_boxes_cw(bx_list, w: int, h: int):
    # CW: (x, y) -> (y, w - 1 - x), new size (w' = h, h' = w)
    out = []
    for x1, y1, x2, y2 in bx_list:
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        pts = [(y, w - 1 - x) for (x, y) in corners]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        out.append([float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))])
    return out


def rotate_current(delta_quarters: int):
    """Rotate current frame + boxes by N*90 degrees (positive=CW, negative=CCW)."""
    global img, orig_img, boxes, drawing, rotation_k, scale_factor, paint_box
    if orig_img is None:
        return

    steps = int(delta_quarters)
    if steps == 0:
        return

    drawing = False
    paint_box = None
    if steps > 0:
        for _ in range(steps % 4):
            h, w = orig_img.shape[:2]
            orig_img = cv2.rotate(orig_img, cv2.ROTATE_90_CLOCKWISE)
            boxes = _rotate_boxes_cw(boxes, w=w, h=h)
            rotation_k = (rotation_k + 1) % 4
    else:
        for _ in range((-steps) % 4):
            h, w = orig_img.shape[:2]
            orig_img = cv2.rotate(orig_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            boxes = _rotate_boxes_ccw(boxes, w=w, h=h)
            rotation_k = (rotation_k - 1) % 4

    img, scale_factor = resize_to_fit_screen(orig_img, get_window_monitor_size('annotate') or fallback_screen_size)
    redraw(img, boxes)


def mouse_cb(event, x, y, flags, param):
    global drawing, boxes, disp, mx, my, paint_box
    mx, my = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        ccw, cw = _ui_button_rects()
        if _pt_in_rect(x, y, ccw):
            rotate_current(-1)
            return
        if _pt_in_rect(x, y, cw):
            rotate_current(1)
            return
        if orig_img is None:
            return
        h0, w0 = orig_img.shape[:2]
        ox = float(max(0.0, min(w0 - 1.0, x / max(1e-8, scale_factor))))
        oy = float(max(0.0, min(h0 - 1.0, y / max(1e-8, scale_factor))))
        drawing = True
        paint_box = (ox, oy, ox, oy)
        redraw(img, boxes)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if paint_box is None or orig_img is None:
            return
        h0, w0 = orig_img.shape[:2]
        ox = float(max(0.0, min(w0 - 1.0, x / max(1e-8, scale_factor))))
        oy = float(max(0.0, min(h0 - 1.0, y / max(1e-8, scale_factor))))
        x1, y1, x2, y2 = paint_box
        paint_box = (min(x1, ox), min(y1, oy), max(x2, ox), max(y2, oy))
        redraw(img, boxes)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if paint_box is None:
            return
        x1, y1, x2, y2 = paint_box
        paint_box = None
        # Prevent zero or negative area
        if x2 > x1 + 1 and y2 > y1 + 1:
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
        redraw(img, boxes)


def redraw(base, bx_list, highlight_last=True):
    """draw all boxes on a copy of base, highlight last box if any"""
    global disp, paint_box
    if base is None:
        disp = None
        return
    canvas: np.ndarray = base.copy()
    n = len(bx_list)
    for i, bx in enumerate(bx_list):
        color = (0, 255, 0)  # green for normal
        thickness = 2
        if highlight_last and i == n - 1 and not drawing:
            color = (0, 0, 255)  # red
            thickness = 3
        x1 = int(bx[0] * scale_factor)
        y1 = int(bx[1] * scale_factor)
        x2 = int(bx[2] * scale_factor)
        y2 = int(bx[3] * scale_factor)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

    if drawing and paint_box is not None:
        x1, y1, x2, y2 = paint_box
        cv2.rectangle(
            canvas,
            (int(x1 * scale_factor), int(y1 * scale_factor)),
            (int(x2 * scale_factor), int(y2 * scale_factor)),
            (255, 255, 0),
            2,
        )
    disp = canvas


def adjust_last(dx1=0, dy1=0, dx2=0, dy2=0):
    """Fine-tune the last box, clamped to image, by ADJUST_BB_SIZE% of bbox size."""
    if not boxes:
        return
    if orig_img is None:
        return
    h, w = orig_img.shape[:2]
    x1, y1, x2, y2 = boxes[-1]

    box_w = max(1, abs(x2 - x1))
    box_h = max(1, abs(y2 - y1))
    step_x = max(1, int(ADJUST_BB_SIZE * box_w))
    step_y = max(1, int(ADJUST_BB_SIZE * box_h))

    nx1 = max(0, min(x1 + dx1 * step_x, x2 - 1))
    ny1 = max(0, min(y1 + dy1 * step_y, y2 - 1))
    nx2 = min(w - 1, max(x2 + dx2 * step_x, nx1 + 1))
    ny2 = min(h - 1, max(y2 + dy2 * step_y, ny1 + 1))
    boxes[-1] = [nx1, ny1, nx2, ny2]
    redraw(img, boxes)


def save_sample(vpath: Path):
    global last_saved, last_sid, sample_count
    last_sid += 1
    sample_count += 1

    last_saved = (vpath, frame_no, last_sid)
    if not boxes:
        print('⚠ No boxes')
    stem = f'{vpath.stem}_sample_{last_sid:04d}'
    jpg = args.output_dir / f'{stem}.jpg'
    txt = args.output_dir / f'{stem}.txt'

    # Ensure we have an original image to save
    if orig_img is None:
        print('⚠ Original image not available; skipping save')
        return False

    cv2.imwrite(str(jpg), orig_img)
    h0, w0 = orig_img.shape[:2]
    with open(txt, 'w') as f:
        for x1, y1, x2, y2 in boxes:
            ox1 = max(0.0, min(w0 - 1.0, float(x1)))
            oy1 = max(0.0, min(h0 - 1.0, float(y1)))
            ox2 = max(0.0, min(w0 - 1.0, float(x2)))
            oy2 = max(0.0, min(h0 - 1.0, float(y2)))

            cx = (ox1 + ox2) / 2.0 / w0
            cy = (oy1 + oy2) / 2.0 / h0
            bw = (ox2 - ox1) / w0
            bh = (oy2 - oy1) / h0
            f.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
    print(f'✔ {jpg.name} ({len(boxes)} boxes)')
    os.system(f"notify-send 'Saved' '{jpg.name}'")
    return True


# ---------- main loop -------------------------------------------------------
cv2.namedWindow('annotate')
cv2.setMouseCallback('annotate', mouse_cb)

# Start sid with number of already existing .jpg files


def get_max_sid(output_dir):
    # Looks for patterns like *_sample_XXXX.jpg
    sid_pat = re.compile(r'_sample_(\d{4})\.jpg$')
    max_sid = -1
    for f in output_dir.glob('*.jpg'):
        m = sid_pat.search(f.name)
        if m:
            s = int(m.group(1))
            max_sid = max(max_sid, s)
    return max_sid


last_sid = get_max_sid(args.output_dir)
sample_count = len(list(args.output_dir.glob('*.jpg')))
print(f'Starting at sample id {last_sid} (existing: {sample_count})')

frame_state = {}  # To remember which frame you are on for each video, for advanced uses if needed.


def _frame_count_for(vpath: Path) -> int:
    key = str(vpath.resolve())
    if key in frame_count_cache:
        return frame_count_cache[key]
    cap = cv2.VideoCapture(str(vpath))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.release()
    frame_count_cache[key] = count
    return count


def _pick_random_frame():
    idx = random.choices(range(len(videos)), weights)[0]
    vpath = videos[idx]
    fcnt = fcounts[idx]
    return vpath, fcnt, random.randint(0, fcnt - 1), None


def _pick_next_source():
    global hard_window_idx, active_hard_window
    if hard_windows:
        if hard_window_idx >= len(hard_windows):
            return None
        active_hard_window = hard_windows[hard_window_idx]
        hard_window_idx += 1
        vpath = active_hard_window.video_path
        fcnt = _frame_count_for(vpath)
        frame_no = min(max(int(active_hard_window.peak_frame), 0), max(0, fcnt - 1))
        return vpath, fcnt, frame_no, active_hard_window
    active_hard_window = None
    return _pick_random_frame()


initial_source = _pick_next_source()
if initial_source is None:
    raise SystemExit('No frames available to annotate.')
vpath, fcnt, frame_no, active_hard_window = initial_source

while sample_count < args.samples:
    cap = cv2.VideoCapture(str(vpath))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        # If frame not ok, pick another source frame
        next_source = _pick_next_source()
        if next_source is None:
            break
        vpath, fcnt, frame_no, active_hard_window = next_source
        continue

    orig_img = frame
    img, scale_factor = resize_to_fit_screen(orig_img, get_window_monitor_size('annotate') or fallback_screen_size)
    rotation_k = 0
    paint_box = None
    boxes.clear()
    redraw(img, boxes)
    h, w = img.shape[:2]
    last_eff_screen: Optional[Tuple[int, int]] = None

    while True:
        eff_screen = get_window_monitor_size('annotate') or fallback_screen_size
        if orig_img is not None and eff_screen is not None and eff_screen != last_eff_screen:
            img, scale_factor = resize_to_fit_screen(orig_img, eff_screen)
            redraw(img, boxes)
            last_eff_screen = eff_screen

        window_suffix = ''
        if active_hard_window is not None:
            window_suffix = (
                f' [window {hard_window_idx}/{len(hard_windows)} '
                f'{active_hard_window.start_frame + 1}-{active_hard_window.end_frame + 1}]'
            )
        cv2.setWindowTitle(
            'annotate',
            f'annotate [{sample_count}/{args.samples}] {vpath.name} (frame {frame_no + 1}/{fcnt}) rot={rotation_k * 90}°{window_suffix}',
        )
        to_show = disp if disp is not None else (img if img is not None else np.zeros((10, 10, 3), dtype=np.uint8))
        to_show = to_show.copy()
        to_show = _draw_ui(to_show)
        to_show = overlay_screen_warning(to_show, eff_screen)
        cv2.imshow('annotate', to_show)
        key = cv2.waitKey(20)

        if key == 27:  # Esc
            cv2.destroyAllWindows()
            raise SystemExit('Aborted')

        elif key == ord('r'):
            if boxes:
                boxes.pop()
                redraw(img, boxes)

        elif key in (ord('q'), ord('Q')):
            rotate_current(-1)
        elif key in (ord('e'), ord('E')):
            rotate_current(1)

        elif key == 32:  # Space -> accept/save
            if save_sample(vpath):
                next_source = _pick_next_source()
                if next_source is None:
                    sample_count = args.samples
                else:
                    vpath, fcnt, frame_no, active_hard_window = next_source
            break

        # Fine-tune last box
        elif key == ord('w'):
            adjust_last(dy1=-1)
        elif key == ord('W'):
            adjust_last(dy1=1)
        elif key == ord('a'):
            adjust_last(dx1=-1)
        elif key == ord('A'):
            adjust_last(dx1=1)
        elif key == ord('s'):
            adjust_last(dy2=1)
        elif key == ord('S'):
            adjust_last(dy2=-1)
        elif key == ord('d'):
            adjust_last(dx2=1)
        elif key == ord('D'):
            adjust_last(dx2=-1)

        # Previous frame
        elif key == ord(','):
            if boxes:
                save_sample(vpath)
            frame_no = max(0, frame_no - 1)
            cap = cv2.VideoCapture(str(vpath))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            cap.release()
            if ok:
                orig_img = frame
                img, scale_factor = resize_to_fit_screen(
                    orig_img, get_window_monitor_size('annotate') or fallback_screen_size
                )
                rotation_k = 0
                paint_box = None
                boxes.clear()
                redraw(img, boxes)
            # Stay in this frame for annotation

        # Next frame
        elif key == ord('.'):
            if boxes:
                save_sample(vpath)
            frame_no = min(fcnt - 1, frame_no + 1)
            cap = cv2.VideoCapture(str(vpath))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            cap.release()
            if ok:
                orig_img = frame
                img, scale_factor = resize_to_fit_screen(
                    orig_img, get_window_monitor_size('annotate') or fallback_screen_size
                )
                rotation_k = 0
                paint_box = None
                boxes.clear()
                redraw(img, boxes)
            # Stay in this frame for annotation

        elif key in (ord('f'), ord('F')):
            # store the empty frame and move to a new source frame
            boxes.clear()
            save_sample(vpath)
            next_source = _pick_next_source()
            if next_source is None:
                sample_count = args.samples
            else:
                vpath, fcnt, frame_no, active_hard_window = next_source
            break

        # Backspace: Undo last save, re-display that frame for re-annotation
        elif key == 8:
            if last_saved is not None:
                vpath_del, frame_no_del, sid_del = last_saved
                stem = f'{vpath_del.stem}_sample_{sid_del:04d}'
                jpg = args.output_dir / f'{stem}.jpg'
                txt = args.output_dir / f'{stem}.txt'
                # Delete files if they exist
                if jpg.exists():
                    jpg.unlink()
                    print(f'Deleted: {jpg.name}')
                    os.system(f"notify-send 'Deleted' '{jpg.name}'")
                else:
                    print(f'File not found: {jpg.name}')
                    os.system(f"notify-send 'File not found' '{jpg.name}'")
                if txt.exists():
                    txt.unlink()
                    print(f'Deleted: {txt.name}')
                else:
                    print(f'File not found: {txt.name}')
                    os.system(f"notify-send 'File not found' '{txt.name}'")
                # Set up to re-display that frame
                img = None
                boxes.clear()
                cap = cv2.VideoCapture(str(vpath_del))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no_del)
                ok, frame = cap.read()
                cap.release()
                if ok:
                    orig_img = frame
                    img, scale_factor = resize_to_fit_screen(
                        orig_img, get_window_monitor_size('annotate') or fallback_screen_size
                    )
                    rotation_k = 0
                    paint_box = None
                    redraw(img, boxes)
                    vpath = vpath_del
                    fcnt = _frame_count_for(vpath)
                    frame_no = frame_no_del
                    last_sid = max(0, sid_del)
                    sample_count -= 1
                last_saved = None  # only allow undoing the most recent
            else:
                os.system("notify-send 'Nothing to undo' 'No saved frame to delete.'")
        # skipping
        elif key == ord('x'):
            # Skip this frame, do not save
            next_source = _pick_next_source()
            if next_source is None:
                sample_count = args.samples
            else:
                vpath, fcnt, frame_no, active_hard_window = next_source
            break

cv2.destroyAllWindows()
print('Done!')

cv2.destroyAllWindows()
print('Done!')
