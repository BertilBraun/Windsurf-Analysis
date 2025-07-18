from vidstab import VidStab
from pathlib import Path
import os
import logging


def compute_vidstab_transforms(input_video: str | os.PathLike) -> VidStab | None:
    """
    Compute stabilization transforms for a video (in memory).
    Returns:
        VidStab object with computed transforms.
    """
    # TODO: Make this work for short videos
    try:
        stabilizer = VidStab()
        stabilizer.gen_transforms(input_path=input_video)
    except ValueError as e:
        logging.error(f"Error computing transforms for {input_video}: {e}")
    return None


def stabilize_with_transforms(input_video: str | os.PathLike, output_video: str | os.PathLike, stabilizer: VidStab):
    """
    Apply given transforms to input video, write stabilized video to output path.
    """
    stabilizer.stabilize(
        input_path=input_video,
        output_path=output_video,
    )


def stabilize(input_video: str | os.PathLike, output_video: str | os.PathLike | None = None):
    """
    Stabilize a video using VidStab.
    Args:
        input_video (str | os.PathLike): Path to the input video file.
        output_video (str | os.PathLike): Path to save the stabilized video file.
    """
    input_video = Path(input_video)
    if output_video is None:
        output_video = input_video.with_name(f"{input_video.stem}_stabilized{input_video.suffix}")

    stabilizer = compute_vidstab_transforms(input_video)
    if stabilizer is None:
        return
    stabilize_with_transforms(input_video, output_video, stabilizer)
