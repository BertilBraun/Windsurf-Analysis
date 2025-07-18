import os

from pathlib import Path
from vidstab import VidStab
from vidstab.frame_queue import FrameQueue
import cv2

from video_io import get_video_properties
from copy import deepcopy


class VidStabWithoutVideoCapture:
    def __init__(self, vid_stab: VidStab):
        """A VidStab without the video capture can be send to other processes.

        Args:
            vid_stab (VidStab): The VidStab object to be used for stabilization will be modified.
        """
        fq = vid_stab.frame_queue
        vid_stab.frame_queue = FrameQueue()
        self.vid_stab = deepcopy(vid_stab)
        vid_stab.frame_queue = fq

    def get_vid_stab(self, video_path: os.PathLike | str) -> VidStab:
        """Get the VidStab object."""
        vid_stab = deepcopy(self.vid_stab)
        vid_stab.frame_queue = FrameQueue()
        vid_stab.frame_queue.set_frame_source(cv2.VideoCapture(str(Path(video_path).resolve())))
        return vid_stab


def compute_vidstab_transforms(input_video: str | os.PathLike) -> VidStabWithoutVideoCapture:
    """
    Compute stabilization transforms for a video (in memory).
    Returns:
        VidStab object with computed transforms.
    """
    video_properties = get_video_properties(input_video)

    stabilizer = VidStab()
    stabilizer.gen_transforms(input_path=input_video, smoothing_window=min(30, video_properties.total_frames - 1))

    return VidStabWithoutVideoCapture(stabilizer)


def stabilize_with_transforms(input_video: str | os.PathLike, output_video: str | os.PathLike, stabilizer: VidStabWithoutVideoCapture):
    """
    Apply given transforms to input video, write stabilized video to output path.
    """
    stabilizer.get_vid_stab(Path(input_video)).stabilize(
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
