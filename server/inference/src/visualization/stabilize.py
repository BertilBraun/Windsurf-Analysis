import os
import numpy as np
from typing import NamedTuple

from vidstab import VidStab

from ..util.video_io import get_video_properties

Transform = NamedTuple('Transform', [('dx', float), ('dy', float), ('da', float)])


def compute_stabilization_transforms(input_video: str | os.PathLike) -> list[Transform]:
    """
    Compute stabilization transforms for a video (in memory).
    Returns:
        List of transforms.
    """
    video_properties = get_video_properties(input_video)

    stabilizer = VidStab('DENSE')
    stabilizer.gen_transforms(input_path=input_video, smoothing_window=min(20, video_properties.total_frames - 1))

    assert stabilizer.transforms is not None
    return [Transform(dx, dy, da) for dx, dy, da in stabilizer.transforms]


def stabilize_video(input_video: str | os.PathLike, output_video: str | os.PathLike, transforms: list[Transform]):
    stabilizer = VidStab()
    stabilizer.transforms = np.array([[transform.dx, transform.dy, transform.da] for transform in transforms])
    stabilizer.apply_transforms(input_video, output_video, border_type='reflect', show_progress=False)
