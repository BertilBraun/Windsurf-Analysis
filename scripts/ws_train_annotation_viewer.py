#!/usr/bin/env python3

import argparse, cv2, subprocess
from pathlib import Path
from util import mby_notify, setup_logging
import logging

def draw_boxes(img_path):
    img = cv2.imread(str(img_path))
    txt_path = img_path.with_suffix(".txt")
    if img is None:
        return None
    if txt_path.exists():
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, cx, cy, w, h = map(float, parts)
                ih, iw = img.shape[:2]
                x1 = int((cx - w / 2) * iw)
                y1 = int((cy - h / 2) * ih)
                x2 = int((cx + w / 2) * iw)
                y2 = int((cy + h / 2) * ih)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img

def main():
    setup_logging(notify_on_error=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir", type=Path)
    ap.add_argument("--fps", type=float, default=5, help="Frames per second")
    args = ap.parse_args()

    images = sorted(args.sample_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No jpg images found in {args.sample_dir}")

    win = "sample_viewer"
    delay = max(1, int(1000 // args.fps))
    idx = 0
    playing = True

    cv2.namedWindow(win)
    while True:
        if not images:
            logging.info("No more images left. Exiting.")
            break

        img = draw_boxes(images[idx])
        if img is not None:
            cv2.imshow(win, img)
            cv2.setWindowTitle(win, f"[{idx+1}/{len(images)}] {images[idx].name}")
        else:
            logging.error(f"Could not load {images[idx].name}")

        key = cv2.waitKey(delay if playing else 0) & 0xFF

        if key in (27, ord('q')):
            break
        elif key == ord(' '):  # Space: toggle play/pause
            playing = not playing
        elif key == 81:  # Left arrow
            if idx > 0:
                idx -= 1
        elif key == 83:  # Right arrow
            if idx < len(images) - 1:
                idx += 1
        elif key == 8 and not playing:  # Backspace and paused
            img_path = images[idx]
            txt_path = img_path.with_suffix(".txt")
            try:
                img_path.unlink()
            except FileNotFoundError:
                logging.error(f"Could not delete {img_path.name}, file not found.")
                continue
            try:
                txt_path.unlink()
            except FileNotFoundError:
                logging.error(f"Could not delete {txt_path.name}, file not found.")
            msg = f"Deleted {img_path.name} and {txt_path.name if txt_path.exists() else '[no txt]'}"
            logging.info(msg)
            mby_notify("Image Deleted", msg)
            # Remove from list and update idx
            images.pop(idx)
            if idx >= len(images):
                idx = len(images) - 1
            if not images:
                logging.info("No more images left. Exiting.")
                mby_notify("No Images Left", "All images have been deleted.")
                break
            continue
        elif playing:
            if idx < len(images) - 1:
                idx += 1
            else:
                playing = False  # stop at last image

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
