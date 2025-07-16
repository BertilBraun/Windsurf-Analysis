import os

from tqdm import tqdm
from pathlib import Path


from video_io import VideoReader, VideoWriter, get_video_properties
from detector import SurferDetector
from surfer_tracker import SurferTracker
from annotation_drawer import Annotation, AnnotationDrawer
from stabilize import stabilize_ffmpeg

from common_types import TrackerInput
from worker_pool import WorkerPool

import video_splicing


class WindsurfingVideoProcessor:
    def __init__(self, draw_annotations: bool, output_dir: os.PathLike | str):
        self.detector = SurferDetector()
        self.individual_video_generator = WorkerPool(_generate_individual_videos_worker_function, num_workers=1)
        self.annotated_video_generator = WorkerPool(_write_annotated_video_worker_function, num_workers=1)
        self.draw_annotations = draw_annotations
        self.output_dir = output_dir

    def process_video(self, input_path: os.PathLike):
        """Main video processing pipeline with batched YOLO inference"""

        surfer_tracker = SurferTracker()

        props = get_video_properties(input_path)
        print(f'Processing video: {props.width}x{props.height}, {props.fps} FPS, {props.total_frames} frames')

        for frame_index, frame, detections in self.detector.detect_and_track_video(input_path):
            for detection in detections:
                if detection.track_id is not None:
                    surfer_tracker.add_detection(frame_index, detection, frame)

        processed_tracks = surfer_tracker.process_tracks(input_path)
        self.individual_video_generator.submit((processed_tracks, input_path, self.output_dir))

        if self.draw_annotations:
            all_tracks = {track_id: tracks for track_id, tracks in processed_tracks.items()}
            # YOLO tracks only:
            all_tracks = {track_id: tracks for track_id, tracks in surfer_tracker.track_inputs.items()}

            self.annotated_video_generator.submit((all_tracks, input_path, self.output_dir))

    def finalize(self):
        self.individual_video_generator.stop()
        self.annotated_video_generator.stop()


def _stabilize_individual_video_worker_function(args: tuple[os.PathLike, os.PathLike]) -> None:
    input_file, output_file = args
    print(f'Stabilizing {input_file} -> {output_file}')
    if stabilize_ffmpeg(input_file, output_file):
        print(f'Stabilized {input_file} -> {output_file}')
        os.unlink(input_file)


def _generate_individual_videos_worker_function(args: tuple[TrackerInput, os.PathLike, os.PathLike | str]) -> None:
    tracks, input_path, output_dir = args
    individual_videos = video_splicing.generate_individual_videos(tracks, input_path, output_dir)

    with WorkerPool(_stabilize_individual_video_worker_function, num_workers=4) as stabilizer:
        for individual_video in individual_videos:
            output_file = Path(individual_video).with_suffix('.stabilized.mp4')
            stabilizer.submit((individual_video, output_file))


def _write_annotated_video_worker_function(args: tuple[TrackerInput, os.PathLike, os.PathLike | str]) -> None:
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
