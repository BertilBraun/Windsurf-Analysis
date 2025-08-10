import os

from vidstab import VidStab

from video_io import get_video_properties


def compute_vidstab_transforms(input_video: str | os.PathLike) -> VidStab:
    """
    Compute stabilization transforms for a video (in memory).
    Returns:
        VidStab object with computed transforms.
    """
    video_properties = get_video_properties(input_video)

    stabilizer = VidStab('DENSE')
    stabilizer.gen_transforms(input_path=input_video, smoothing_window=min(20, video_properties.total_frames - 1))

    return stabilizer
