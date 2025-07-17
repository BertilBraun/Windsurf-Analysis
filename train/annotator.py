#!/usr/bin/env python3
"""
annotate.py – minimal multi-box annotator with grow/shrink toggle

Mouse:
    LMB drag : draw box

Keys:
    r        : undo last box
    Space    : accept / save frame
    w/a/s/d  : move/resize last box (mode: grow or shrink)
    q        : toggle shrink/grow mode
    Esc      : quit
"""

import os
import re
import cv2
import random
import argparse

from pathlib import Path

# ---------- CLI -------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('video_dir', type=Path)
ap.add_argument('output_dir', type=Path)
ap.add_argument('--samples', type=int, default=1000)
args = ap.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

VID_EXT = {'.mp4', '.mov', '.avi', '.mkv'}
ADJUST_BB_SIZE = 0.02  # 1% of current bounding box
videos = [p for p in args.video_dir.rglob('*') if p.suffix.lower() in VID_EXT]
if not videos:
    raise SystemExit('No videos found in', args.video_dir)

fcounts = []
for v in videos:
    c = cv2.VideoCapture(str(v))
    fcounts.append(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) or 1)
    c.release()
weights = [c / sum(fcounts) for c in fcounts]


def resize_to_max(img, max_side=2048):
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


# ---------- globals ---------------------------------------------------------
drawing = False
ix = iy = 0
mx, my = 0, 0
boxes = []  # list of [x1,y1,x2,y2] floats
img = disp = None
grow_mode = False  # True: grow (green), False: shrink (red)
last_saved = None  # (vpath, frame_no, sid)


def mouse_cb(event, x, y, flags, param):
    global drawing, ix, iy, boxes, disp, grow_mode, mx, my
    mx, my = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        grow_mode = False
        drawing, ix, iy = True, x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        redraw(img, boxes + [[ix, iy, x, y]])
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        h, w = img.shape[:2]
        x1 = float(max(0, min(ix, x)))
        y1 = float(max(0, min(iy, y)))
        x2 = float(min(w - 1, max(ix, x)))
        y2 = float(min(h - 1, max(iy, y)))
        # Prevent zero or negative area
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
        redraw(img, boxes)


def redraw(base, bx_list, highlight_last=True):
    """draw all boxes on a copy of base, highlight last box if any"""
    global disp
    disp = base.copy()
    n = len(bx_list)
    for i, bx in enumerate(bx_list):
        color = (0, 255, 0)  # green for normal
        thickness = 2
        if highlight_last and i == n - 1 and not drawing:
            color = (0, 255, 0) if grow_mode else (0, 0, 255)  # green or red
            thickness = 3
        cv2.rectangle(disp, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), color, thickness)


def adjust_last(dx1=0, dy1=0, dx2=0, dy2=0):
    """Fine-tune the last box, clamped to image, by ADJUST_BB_SIZE% of bbox size."""
    if not boxes:
        return
    h, w = img.shape[:2]
    x1, y1, x2, y2 = boxes[-1]

    box_w = max(1, abs(x2 - x1))
    box_h = max(1, abs(y2 - y1))
    step_x = max(1, int(ADJUST_BB_SIZE * box_w))
    step_y = max(1, int(ADJUST_BB_SIZE * box_h))

    # In grow mode, move border outward; in shrink, move inward.
    factor = 1 if grow_mode else -1

    nx1 = max(0, min(x1 + factor * dx1 * step_x, x2 - 1))
    ny1 = max(0, min(y1 + factor * dy1 * step_y, y2 - 1))
    nx2 = min(w - 1, max(x2 + factor * dx2 * step_x, nx1 + 1))
    ny2 = min(h - 1, max(y2 + factor * dy2 * step_y, ny1 + 1))
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
    cv2.imwrite(str(jpg), img)
    h, w = img.shape[:2]
    with open(txt, 'w') as f:
        for x1, y1, x2, y2 in boxes:
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
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

idx = random.choices(range(len(videos)), weights)[0]
vpath, fcnt = videos[idx], fcounts[idx]
frame_no = random.randint(0, fcnt - 1)

while sample_count < args.samples:
    cap = cv2.VideoCapture(str(vpath))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        # If frame not ok, pick another random video/frame
        idx = random.choices(range(len(videos)), weights)[0]
        vpath, fcnt = videos[idx], fcounts[idx]
        frame_no = random.randint(0, fcnt - 1)
        continue

    grow_mode = False
    img = resize_to_max(frame)
    boxes.clear()
    redraw(img, boxes)
    h, w = img.shape[:2]

    while True:
        cv2.setWindowTitle(
            'annotate', f'annotate [{sample_count}/{args.samples}] {vpath.name} (frame {frame_no + 1}/{fcnt})'
        )
        cv2.imshow('annotate', disp)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # Esc
            cv2.destroyAllWindows()
            raise SystemExit('Aborted')

        elif key == ord('r'):
            grow_mode = False
            if boxes:
                boxes.pop()
                redraw(img, boxes)

        elif key == 32:  # Space -> accept/save
            grow_mode = False
            if save_sample(vpath):
                # Next: Pick random frame/video
                idx = random.choices(range(len(videos)), weights)[0]
                vpath, fcnt = videos[idx], fcounts[idx]
                frame_no = random.randint(0, fcnt - 1)
            break

        elif key == ord('q'):  # toggle grow/shrink
            grow_mode = not grow_mode
            mode = 'Grow' if grow_mode else 'Shrink'
            print(f'Mode: {mode}')
            redraw(img, boxes)

        # Fine-tune last box
        elif key == ord('w'):
            adjust_last(dy1=-1)
        elif key == ord('a'):
            adjust_last(dx1=-1)
        elif key == ord('s'):
            adjust_last(dy2=1)
        elif key == ord('d'):
            adjust_last(dx2=1)

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
                img = resize_to_max(frame)
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
                img = resize_to_max(frame)
                boxes.clear()
                redraw(img, boxes)
            # Stay in this frame for annotation

        elif key == ord('e'):
            # store the empty frame with no boxes and advance to the next frame
            save_sample(vpath)
            frame_no = min(fcnt - 1, frame_no + 1)
            cap = cv2.VideoCapture(str(vpath))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            cap.release()
            if ok:
                img = resize_to_max(frame)
                boxes.clear()
                redraw(img, boxes)

        # Tab: Reset box start during drawing
        elif key == 9 and drawing:
            ix = mx
            iy = my
            redraw(img, boxes + [[ix, iy, mx, my]])

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
                    img = resize_to_max(frame)
                    redraw(img, boxes)
                    vpath = vpath_del
                    fcnt = fcounts[videos.index(vpath)]
                    frame_no = frame_no_del
                    last_sid = max(0, sid_del)
                    sample_count -= 1
                last_saved = None  # only allow undoing the most recent
            else:
                os.system("notify-send 'Nothing to undo' 'No saved frame to delete.'")
        # skipping
        elif key == ord('x'):
            # Skip this frame, do not save
            idx = random.choices(range(len(videos)), weights)[0]
            vpath, fcnt = videos[idx], fcounts[idx]
            frame_no = random.randint(0, fcnt - 1)
            break

cv2.destroyAllWindows()
print('Done!')

cv2.destroyAllWindows()
print('Done!')
