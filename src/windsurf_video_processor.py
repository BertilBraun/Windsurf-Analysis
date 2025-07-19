import os
import logging
from typing import Callable, TypeVar

from tqdm import tqdm
from pathlib import Path
from debug_drawer import generate_debug_video_worker_function
from helpers import log_and_reraise
from stabilize import compute_vidstab_transforms
import logging


from video_io import VideoReader, VideoWriter, get_video_properties
from detector import SurferDetector
from surfer_tracker import process_detections_into_tracks
from annotation_drawer import Annotation, AnnotationDrawer
from stabilize import stabilize
from concurrent.futures import ProcessPoolExecutor

from discrete_opt_tracker import DiscreteOptimizationTracker

from common_types import Track

import video_splicing


P = TypeVar('P')
R = TypeVar('R')


class WindsurfingVideoProcessor:
    """Main video processing orchestrator"""

    def __init__(self, draw_annotations: bool = False, output_dir: str = 'individual_surfers', dry_run: bool = False):
        self.surf_detector = SurferDetector()
        # TODO: parameterize
        self.priority_mp_executor = ProcessPoolExecutor(max_workers=1)
        self.high_mp_executor = ProcessPoolExecutor(max_workers=4)
        self.draw_annotations = draw_annotations
        self.output_dir = output_dir
        self.dry_run = dry_run

    def process_video(self, input_path: os.PathLike):
        """Main video processing pipeline with batched YOLO inference"""
        logger = logging.getLogger(__name__)

        props = get_video_properties(input_path)
        logger.info(f'Processing video: {props.width}x{props.height}, {props.fps} FPS, {props.total_frames} frames')

        # start stabilizer computation in background
        # stabilizer_future = self.submit_high_priority_task(compute_vidstab_transforms, input_path)

        # run detection and tracking
        detections = list(self.surf_detector.run_object_detection_on_video(input_path))

        # wait for stabilizer computation to finish
        # stabilizer = stabilizer_future.result()

        processed_tracks = process_detections_into_tracks(input_path, detections, DiscreteOptimizationTracker())

        if not self.dry_run:
            self.submit_low_priority_task(
                _generate_individual_videos_worker_function, (processed_tracks, input_path, self.output_dir)
            )

        if self.draw_annotations:
            self.submit_low_priority_task(
                _generate_annotated_video_worker_function, (processed_tracks, input_path, self.output_dir)
            )

            self.submit_low_priority_task(
                generate_debug_video_worker_function, (detections, processed_tracks, None, input_path, self.output_dir)
            )

    def finalize(self):
        self.high_mp_executor.shutdown(wait=True)
        self.priority_mp_executor.shutdown(wait=True)

    def submit_low_priority_task(self, func: Callable[[P], R], args: P, **kwargs):
        return self.high_mp_executor.submit(
            log_and_reraise, func, args, helpers_log_and_reraise_output_dir=self.output_dir, **kwargs
        )

    def submit_high_priority_task(self, func: Callable[[P], R], args: P, **kwargs):
        return self.priority_mp_executor.submit(
            log_and_reraise, func, args, helpers_log_and_reraise_output_dir=self.output_dir, **kwargs
        )


def _stabilize_individual_video_worker_function(args: tuple[os.PathLike, os.PathLike]) -> None:
    logger = logging.getLogger(__name__)
    input_file, output_file = args
    logger.info(f'Stabilizing {input_file} -> {output_file}')
    if stabilize(input_file, output_file):
        logger.info(f'Stabilized {input_file} -> {output_file}')
        os.unlink(input_file)


def _generate_individual_videos_worker_function(args: tuple[list[Track], os.PathLike, os.PathLike | str]) -> None:
    tracks, input_path, output_dir = args
    individual_videos = video_splicing.generate_individual_videos(tracks, input_path, output_dir)

    with ProcessPoolExecutor(max_workers=4) as executor:
        for individual_video in individual_videos:
            output_file = Path(individual_video).with_suffix('.stabilized.mp4')
            executor.submit(_stabilize_individual_video_worker_function, (individual_video, output_file))


def _generate_annotated_video_worker_function(args: tuple[list[Track], os.PathLike, os.PathLike | str]) -> None:
    tracks, input_path, output_dir = args
    annotation_drawer = AnnotationDrawer()

    annotated_video_path = Path(output_dir) / f'{Path(input_path).stem}+00_annotated.mp4'
    logging.info(f'Writing annotated video to {annotated_video_path}')

    with VideoReader(input_path) as reader:
        video_props = reader.get_properties()
        with VideoWriter(annotated_video_path, video_props.width, video_props.height, video_props.fps) as writer:
            for frame_index, frame in tqdm(
                reader.read_frames(), total=video_props.total_frames, desc='Drawing annotations'
            ):
                annotations = [
                    Annotation(track.track_id, detection.bbox, detection.confidence)
                    for track in tracks
                    for detection in track.sorted_detections
                    if detection.frame_idx == frame_index
                ]

                writer.write_frame(annotation_drawer.draw_detections_with_trails(frame, annotations))
