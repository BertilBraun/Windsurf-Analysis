import os
import numpy as np
from typing import NamedTuple

from vidstab import VidStab

from server.inference.src.util.cache import cache_to_file
from ..util.video_io import VideoReader
from server.inference.bot_sort.gmc import GMC

from ..util.video_io import get_video_properties

Transform = NamedTuple('Transform', [('dx', float), ('dy', float), ('da', float), ('frame_idx', int)])


@cache_to_file('vidstab_transforms')
def compute_stabilization_transforms(input_video: str | os.PathLike) -> list[Transform]:
    """
    Compute per-frame stabilization deltas using VidStab's generated transforms.
    Output is dx, dy, da for each frame relative to the previous frame (no cumulative, no prepend).
    """
    video_properties = get_video_properties(input_video)

    stabilizer = VidStab()  # TODO: 'DENSE')
    stabilizer.gen_transforms(input_path=input_video, smoothing_window=min(20, video_properties.total_frames - 1))

    assert stabilizer.transforms is not None
    # VidStab stores cumulative-like smoothing outputs per frame index starting from 1 typically.
    # Convert to per-frame deltas: delta[i] = params[i] - params[i-1]
    raw = [(float(dx), float(dy), float(da)) for (dx, dy, da) in stabilizer.transforms]
    deltas: list[Transform] = []
    prev_dx, prev_dy, prev_da = 0.0, 0.0, 0.0
    for i, (dx, dy, da) in enumerate(raw, start=1):
        ddx = dx - prev_dx
        ddy = dy - prev_dy
        dda = da - prev_da
        deltas.append(Transform(ddx, ddy, dda, i))
        prev_dx, prev_dy, prev_da = dx, dy, da
    return deltas


def stabilize_video(input_video: str | os.PathLike, output_video: str | os.PathLike, transforms: list[Transform]):
    stabilizer = VidStab()
    stabilizer.transforms = np.array([[transform.dx, transform.dy, transform.da] for transform in transforms])
    stabilizer.apply_transforms(input_video, output_video, show_progress=False)


@cache_to_file('gmc_transforms')
def compute_stabilization_transforms_gmc(input_video: str | os.PathLike, downscale: int = 2) -> list[Transform]:
    """
    Compute per-frame camera motion deltas using the GMC method,
    and return dx, dy, da for each frame relative to the previous frame.
    """
    gmc = GMC(downscale=downscale)
    transforms: list[Transform] = []

    with VideoReader(input_video) as reader:
        for f, frame in reader.read_frames():
            H_delta = gmc.apply(frame)
            if not isinstance(H_delta, np.ndarray) or H_delta.shape != (2, 3):
                H_delta = np.eye(2, 3, dtype=np.float64)
            # Extract rigid delta parameters from H (prev -> curr)
            R00, tx = float(H_delta[0, 0]), float(H_delta[0, 2])
            R10, ty = float(H_delta[1, 0]), float(H_delta[1, 2])
            da = float(np.arctan2(R10, R00))
            dx = float(tx)
            dy = float(ty)
            transforms.append(Transform(dx, dy, da, int(f)))

    # Ensure we have at least a zero transform for frame 0
    if not transforms or transforms[0].frame_idx != 0:
        transforms = [Transform(0.0, 0.0, 0.0, 0)] + transforms
    return transforms
