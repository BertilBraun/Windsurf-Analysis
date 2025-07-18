import os

from pathlib import Path
from vidstab import VidStab

from video_io import get_video_properties


def compute_vidstab_transforms(input_video: str | os.PathLike) -> VidStab:
    """
    Compute stabilization transforms for a video (in memory).
    Returns:
        VidStab object with computed transforms.
    """
    video_properties = get_video_properties(input_video)

    stabilizer = VidStab()
    stabilizer.gen_transforms(input_path=input_video, smoothing_window=min(30, video_properties.total_frames - 1))

    return stabilizer


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
        output_video = input_video.with_name(f'{input_video.stem}_stabilized{input_video.suffix}')

    stabilizer = compute_vidstab_transforms(input_video)
    stabilize_with_transforms(input_video, output_video, stabilizer)
