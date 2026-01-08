from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _add_video_processing_to_path() -> None:
    import sys

    video_processing_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(video_processing_dir))


_add_video_processing_to_path()

from inference.src.util.video_io import VideoReader, VideoWriter  # noqa: E402


@dataclass(frozen=True)
class FrameTransform:
    frame_idx: int
    dx: float
    dy: float
    da: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create a synthetic 'shaky' video by applying per-frame random translations, "
            "then cropping a fixed margin so no black borders appear."
        )
    )
    p.add_argument("--input-video", type=str, required=True)
    p.add_argument("--output-video", type=str, required=True)
    p.add_argument("--transforms-out", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--max-dx", type=float, default=20.0, help="Max translation in +/− pixels (x).")
    p.add_argument("--max-dy", type=float, default=20.0, help="Max translation in +/− pixels (y).")
    p.add_argument(
        "--max-da-deg",
        type=float,
        default=0.0,
        help="Optional max rotation in +/− degrees. If non-zero, you must set crop margins explicitly.",
    )

    p.add_argument(
        "--crop-margin-x",
        type=int,
        default=None,
        help="Crop margin in pixels (x). Defaults to ceil(max-dx) if max-da-deg==0.",
    )
    p.add_argument(
        "--crop-margin-y",
        type=int,
        default=None,
        help="Crop margin in pixels (y). Defaults to ceil(max-dy) if max-da-deg==0.",
    )
    p.add_argument(
        "--limit-frames",
        type=int,
        default=None,
        help="Optionally limit number of processed frames (useful for quick tests).",
    )
    p.add_argument("--fourcc", type=str, default="mp4v")
    return p.parse_args()


def _warp_and_crop(
    frame_bgr: np.ndarray,
    *,
    dx: float,
    dy: float,
    da_rad: float,
    crop_margin_x: int,
    crop_margin_y: int,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]

    if abs(da_rad) > 0:
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, da_rad * 180.0 / np.pi, 1.0)
        M[0, 2] += float(dx)
        M[1, 2] += float(dy)
    else:
        M = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)

    warped = cv2.warpAffine(
        frame_bgr,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    x1, x2 = crop_margin_x, w - crop_margin_x
    y1, y2 = crop_margin_y, h - crop_margin_y
    return warped[y1:y2, x1:x2]


def main() -> int:
    args = _parse_args()

    input_video = Path(args.input_video)
    output_video = Path(args.output_video)
    transforms_out = Path(args.transforms_out)

    rng = np.random.default_rng(int(args.seed))

    max_da_deg = float(args.max_da_deg)
    if max_da_deg != 0.0 and (args.crop_margin_x is None or args.crop_margin_y is None):
        raise SystemExit("--max-da-deg requires explicit --crop-margin-x/--crop-margin-y to avoid black borders.")

    crop_margin_x = int(args.crop_margin_x) if args.crop_margin_x is not None else int(np.ceil(float(args.max_dx)))
    crop_margin_y = int(args.crop_margin_y) if args.crop_margin_y is not None else int(np.ceil(float(args.max_dy)))
    crop_margin_x = max(0, crop_margin_x)
    crop_margin_y = max(0, crop_margin_y)

    transforms: list[FrameTransform] = []

    with VideoReader(input_video) as reader:
        props = reader.get_properties()
        out_w = int(props.width - 2 * crop_margin_x)
        out_h = int(props.height - 2 * crop_margin_y)
        if out_w <= 0 or out_h <= 0:
            raise SystemExit(
                f"Crop margins too large for input size {props.width}x{props.height}: "
                f"({crop_margin_x}, {crop_margin_y})."
            )

        limit_frames = int(args.limit_frames) if args.limit_frames is not None else None

        with VideoWriter(
            output_video,
            width=out_w,
            height=out_h,
            fps=int(props.fps),
            fourcc=str(args.fourcc),
        ) as writer:
            for frame_idx, frame in reader.read_frames():
                frame_idx = int(frame_idx)
                if limit_frames is not None and frame_idx >= limit_frames:
                    break

                dx = float(rng.uniform(-float(args.max_dx), float(args.max_dx)))
                dy = float(rng.uniform(-float(args.max_dy), float(args.max_dy)))
                da_rad = float(rng.uniform(-max_da_deg, max_da_deg) * np.pi / 180.0)

                out = _warp_and_crop(
                    frame,
                    dx=dx,
                    dy=dy,
                    da_rad=da_rad,
                    crop_margin_x=crop_margin_x,
                    crop_margin_y=crop_margin_y,
                )
                if out.shape[1] != out_w or out.shape[0] != out_h:
                    raise RuntimeError(
                        f"Unexpected output frame size {out.shape[1]}x{out.shape[0]} (expected {out_w}x{out_h})"
                    )
                writer.write_frame(out)

                transforms.append(FrameTransform(frame_idx=frame_idx, dx=dx, dy=dy, da=da_rad))

    payload = {
        "input_video": str(input_video),
        "output_video": str(output_video),
        "seed": int(args.seed),
        "max_dx": float(args.max_dx),
        "max_dy": float(args.max_dy),
        "max_da_deg": float(max_da_deg),
        "crop_margin_x": int(crop_margin_x),
        "crop_margin_y": int(crop_margin_y),
        "frames": [t.__dict__ for t in transforms],
    }
    transforms_out.parent.mkdir(parents=True, exist_ok=True)
    transforms_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote video: {output_video}")
    print(f"Wrote transforms: {transforms_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

