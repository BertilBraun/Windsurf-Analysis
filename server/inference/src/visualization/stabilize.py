import os
import numpy as np
from typing import Dict, NamedTuple, Optional

from vidstab import VidStab

from server.inference.src.util.cache import cache_to_file
from ..util.video_io import VideoReader
from server.inference.bot_sort.gmc import GMC

from ..util.video_io import get_video_properties

Transform = NamedTuple('Transform', [('dx', float), ('dy', float), ('da', float), ('frame_idx', int)])


@cache_to_file('vidstab_transforms')
def compute_stabilization_transforms(input_video: str | os.PathLike) -> list[Transform]:
    """
    Compute stabilization transforms for a video (in memory).
    Returns:
        List of transforms.
    """
    video_properties = get_video_properties(input_video)

    stabilizer = VidStab()  # TODO: 'DENSE')
    stabilizer.gen_transforms(input_path=input_video, smoothing_window=min(20, video_properties.total_frames - 1))

    assert stabilizer.transforms is not None
    transforms = [
        Transform(float(dx), float(dy), float(da), frame_idx + 1)
        for frame_idx, (dx, dy, da) in enumerate(stabilizer.transforms)
    ]
    # Prepend a true zero transform at frame 0 to keep indices aligned and avoid a first-frame spike
    return [Transform(0.0, 0.0, 0.0, 0)] + transforms


def stabilize_video(input_video: str | os.PathLike, output_video: str | os.PathLike, transforms: list[Transform]):
    stabilizer = VidStab()
    stabilizer.transforms = np.array([[transform.dx, transform.dy, transform.da] for transform in transforms])
    stabilizer.apply_transforms(input_video, output_video, show_progress=False)


@cache_to_file('gmc_transforms')
def compute_stabilization_transforms_gmc(
    input_video: str | os.PathLike,
    method: str = 'sparseOptFlow',
    downscale: int = 2,
    detections_by_frame: Optional[Dict[int, np.ndarray]] = None,
) -> list[Transform]:
    """
    Compute stabilization transforms using the same GMC method used by BoTSORT (e.g., sparseOptFlow),
    and return cumulative absolute offsets for dx, dy, da per frame.
    """
    gmc = GMC(method=method, downscale=downscale)
    cumulative = np.eye(3, dtype=np.float64)
    transforms: list[Transform] = []

    with VideoReader(input_video) as reader:
        for f, frame in reader.read_frames():
            dets = None
            if detections_by_frame is not None:
                dets = detections_by_frame.get(int(f))
            H_delta = gmc.apply(frame, dets)
            if not isinstance(H_delta, np.ndarray) or H_delta.shape != (2, 3):
                H_delta = np.eye(2, 3, dtype=np.float64)
            H_delta3 = np.eye(3, dtype=np.float64)
            H_delta3[:2, :3] = H_delta.astype(np.float64, copy=False)
            # Compose to get cumulative transform from frame 0 to frame f
            cumulative = H_delta3 @ cumulative

            # Convert to stabilization-style offsets (invert camera motion)
            dx = float(-cumulative[0, 2])
            dy = float(-cumulative[1, 2])
            da = float(np.arctan2(cumulative[1, 0], cumulative[0, 0]))
            transforms.append(Transform(dx, dy, da, int(f)))

    # Ensure we have at least a zero transform for frame 0
    if not transforms or transforms[0].frame_idx != 0:
        transforms = [Transform(0.0, 0.0, 0.0, 0)] + transforms
    return transforms
