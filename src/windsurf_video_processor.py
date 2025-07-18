import os
import logging

from tqdm import tqdm
from pathlib import Path
import numpy as np
from vidstab import VidStab
from debug_drawer import generate_debug_video_worker_function
from stabilize import compute_vidstab_transforms


from video_io import VideoReader, VideoWriter, get_video_properties
from detector import SurferDetector
from surfer_tracker import SurferTracker
from annotation_drawer import Annotation, AnnotationDrawer
from stabilize import stabilize
from concurrent.futures import ProcessPoolExecutor

from common_types import Track, TrackerInput, Detection

import video_splicing


def _detector_mp_init_surf_detector():
    global surf_detector
    surf_detector = SurferDetector()


def _run_detect(video_path: os.PathLike) -> list[Detection]:
    """Run detection and tracking on a video file"""
    global surf_detector
    logger = logging.getLogger(__name__)
    logger.info(f'Starting detection on {video_path}')
    return list(surf_detector.detect_and_track_video(video_path))


def _run_stabilize(video_path: os.PathLike) -> VidStab | None:
    return compute_vidstab_transforms(video_path)


class WindsurfingVideoProcessor:
    """Main video processing orchestrator"""

    def __init__(self, draw_annotations: bool = False, output_dir: str = 'individual_surfers', dry_run: bool = False):
        # TODO: parameterize
        self.detector_mp = ProcessPoolExecutor(max_workers=2, initializer=_detector_mp_init_surf_detector)
        self.high_mp_executor = ProcessPoolExecutor(max_workers=4)
        self.draw_annotations = draw_annotations
        self.output_dir = output_dir
        self.dry_run = dry_run

    def process_video(self, input_path: os.PathLike):
        """Main video processing pipeline with batched YOLO inference"""
        logger = logging.getLogger(__name__)

        surfer_tracker = SurferTracker()

        props = get_video_properties(input_path)
        logger.info(f'Processing video: {props.width}x{props.height}, {props.fps} FPS, {props.total_frames} frames')

        detector_future = self.detector_mp.submit(_run_detect, input_path)
        stabilizer_future = self.high_mp_executor.submit(_run_stabilize, input_path)

        detections = detector_future.result()
        stabilizer = stabilizer_future.result()

        for detection in detections:
            surfer_tracker.add_detection(detection.frame_idx, detection)

        processed_tracks = surfer_tracker.process_tracks(input_path)


        if not self.dry_run:
            self.high_mp_executor.submit(
                _generate_individual_videos_worker_function, (processed_tracks, input_path, self.output_dir)
            )

        if self.draw_annotations:
            all_tracks = {track_id: tracks for track_id, tracks in processed_tracks.items()}
            # YOLO tracks only:
            tracks = (
                Track(detection.bbox, detection.feat, detection.confidence, frame_idx, None)
                for (frame_idx, detections) in surfer_tracker.detections.items()
                for detection in detections
            )
            all_tracks = {i: [t] for i, t in enumerate(tracks)}

            self.high_mp_executor.submit(
                _generate_annotated_video_worker_function,
                (all_tracks, input_path, self.output_dir)
            )

            self.high_mp_executor.submit(
                generate_debug_video_worker_function,
                (detections, input_path, self.output_dir)
            )


            # self.annotated_video_generator.submit((all_tracks, input_path, self.output_dir))
            # self.debug_video_generator.submit((surfer_tracker.detections, input_path, self.output_dir))

    def finalize(self):
        self.high_mp_executor.shutdown(wait=True)
        self.detector_mp.shutdown(wait=True)


def _stabilize_individual_video_worker_function(args: tuple[os.PathLike, os.PathLike]) -> None:
    logger = logging.getLogger(__name__)
    input_file, output_file = args
    logger.info(f'Stabilizing {input_file} -> {output_file}')
    if stabilize(input_file, output_file):
        logger.info(f'Stabilized {input_file} -> {output_file}')
        os.unlink(input_file)


def _generate_individual_videos_worker_function(args: tuple[TrackerInput, os.PathLike, os.PathLike | str]) -> None:
    tracks, input_path, output_dir = args
    individual_videos = video_splicing.generate_individual_videos(tracks, input_path, output_dir)

    with ProcessPoolExecutor(max_workers=4) as executor:
        for individual_video in individual_videos:
            output_file = Path(individual_video).with_suffix('.stabilized.mp4')
            executor.submit(
                _stabilize_individual_video_worker_function, (individual_video, output_file)
            )


def _generate_annotated_video_worker_function(args: tuple[TrackerInput, os.PathLike, os.PathLike | str]) -> None:
    tracks, input_path, output_dir = args
    annotation_drawer = AnnotationDrawer()

    annotated_video_path = Path(output_dir) / f'{Path(input_path).stem}+00_annotated.mp4'

    with VideoReader(input_path) as reader:
        video_props = reader.get_properties()
        with VideoWriter(annotated_video_path, video_props.width, video_props.height, video_props.fps) as writer:
            for frame_index, frame in tqdm(
                reader.read_frames(), total=video_props.total_frames, desc='Drawing annotations'
            ):
                annotations = [
                    Annotation(track_id, track.bbox, track.confidence)
                    for track_id, tracks in tracks.items()
                    for track in tracks
                    if track.frame_idx == frame_index
                ]

                writer.write_frame(annotation_drawer.draw_detections_with_trails(frame, annotations))
