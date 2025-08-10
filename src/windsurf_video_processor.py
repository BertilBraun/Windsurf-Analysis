import os
import logging
import pickle
from typing import Callable, Sequence, TypeVar

from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from helpers import log_and_reraise

from player.core.player_state import DetectionLite, Metadata, TrackLite, VideoProperties
from video_io import VideoInfo, VideoReader, VideoWriter, get_video_properties
from detector import SurferDetector
from visualization.annotation_drawer import Annotation, AnnotationDrawer

from tracking.tracking import Tracker
from tracking.track_processing import TrackFilteringSmoothingRelabeling
from tracking.discrete_opt_tracker import DiscreteILPTracker
from tracking.preprocessing.greedy_preprocessor import GreedyPreprocessor
from tracking.greedy_tracker import GreedyTracker  # noqa: F401 (imported for optional use)


from visualization.debug_drawer import generate_debug_video_worker_function, debug_track_similarities
from visualization.video_splicing import generate_individual_videos
from visualization.stabilize import compute_vidstab_transforms
from visualization.track_graph_viz import visualize_tracks  # noqa: F401 (optional debugging import)

from common_types import Detection, Track


P = TypeVar('P')
R = TypeVar('R')


class WindsurfingVideoProcessor:
    """Main video processing orchestrator"""

    def __init__(
        self,
        draw_annotations: bool,
        output_dir: str,
        generate_videos: bool,
        debug_views: bool,
        parallel_workers: int,
        stabilize: bool,
    ):
        self.surf_detector = SurferDetector()
        self.executor = ProcessPoolExecutor(max_workers=parallel_workers)
        self.draw_annotations = draw_annotations
        self.output_dir = Path(output_dir)
        self.generate_videos = generate_videos
        self.debug_views = debug_views
        self.stabilize = stabilize

    def process_video(self, input_path: os.PathLike):
        """Main video processing pipeline with batched YOLO inference"""
        logger = logging.getLogger(__name__)

        input_path = Path(input_path)

        props = get_video_properties(input_path)
        logger.info(f'Processing video: {props.width}x{props.height}, {props.fps} FPS, {props.total_frames} frames')

        # run detection and tracking
        detections = self.surf_detector.run_object_detection_on_video(input_path)

        processed_tracks = _process_detections_into_tracks(
            detections,
            props,
            trackers=[
                GreedyPreprocessor(),
                # GreedyTracker(),
                DiscreteILPTracker(),
                TrackFilteringSmoothingRelabeling(),
            ],
        )

        # Always save compact track metadata for the interactive player
        _save_tracks_metadata(processed_tracks, input_path, self.output_dir, props)

        if self.generate_videos:
            self.submit_task(
                _generate_individual_videos_worker_function,
                (processed_tracks, input_path, self.output_dir, self.stabilize),
            )

        if self.draw_annotations:
            self.submit_task(
                _generate_annotated_video_worker_function,
                (processed_tracks, input_path, self.output_dir),
            )

        if self.debug_views:
            self.submit_task(
                generate_debug_video_worker_function,
                (detections, processed_tracks, input_path, self.output_dir),
            )
            self.submit_task(
                debug_track_similarities,
                (processed_tracks, input_path, self.output_dir, props),
            )

    def finalize(self):
        self.executor.shutdown(wait=True)

    def submit_task(self, func: Callable[[P], R], args: P, **kwargs):
        return self.executor.submit(
            log_and_reraise, func, args, helpers_log_and_reraise_output_dir=self.output_dir, **kwargs
        )


def _process_detections_into_tracks(
    detections: list[Detection], video_properties: VideoInfo, trackers: Sequence[Tracker]
) -> list[Track]:
    """Process collected tracks and return processed track data for video generation"""
    logger = logging.getLogger(__name__)

    if not detections:
        logger.warning('No tracks available for processing')
        return []

    processed_tracks = [Track(track_id=i, sorted_detections=[detection]) for i, detection in enumerate(detections)]
    for tracker in trackers:
        processed_tracks = tracker.track(processed_tracks, video_properties)

        # Show a timeline of the tracks with all possible merge options
        # visualize_tracks(processed_tracks, str(original_video_path))

    if not processed_tracks:
        logger.warning('No valid tracks found for video generation')
        return []

    # Log track statistics
    logger.info(f'After processing: {len(processed_tracks)} tracks remaining')
    for track in processed_tracks:
        duration_frames = track.sorted_detections[-1].frame_idx - track.sorted_detections[0].frame_idx
        duration_seconds = duration_frames / video_properties.fps
        frame_percentage = duration_frames / video_properties.total_frames
        logger.info(
            f'  Track {track.track_id}: {len(track.sorted_detections)} detections, {duration_seconds:.1f}s ({frame_percentage * 100:.1f}%)'
        )

    return processed_tracks


def _generate_individual_videos_worker_function(args: tuple[list[Track], os.PathLike, Path, bool]) -> None:
    tracks, input_path, output_dir, stabilize = args
    individual_videos = generate_individual_videos(tracks, input_path, output_dir)

    if stabilize:
        video_stabilizer = compute_vidstab_transforms(input_path)
        for individual_video in individual_videos:
            output_file = Path(individual_video).with_suffix('.stabilized.mp4')
            video_stabilizer.stabilize(input_path=individual_video, output_path=output_file, use_stored_transforms=True)


def _generate_annotated_video_worker_function(args: tuple[list[Track], os.PathLike, Path]) -> None:
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


def _save_tracks_metadata(tracks: list[Track], input_path: Path, output_dir: Path, video_props: VideoInfo) -> None:
    """Save compact track metadata for later loading by the player interface.

    The metadata excludes embeddings to keep file sizes small.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = Metadata(
        input_video_path=input_path.absolute().as_posix(),
        video_properties=VideoProperties(
            fps=video_props.fps,
            width=video_props.width,
            height=video_props.height,
            total_frames=video_props.total_frames,
        ),
        tracks=[
            TrackLite(
                track_id=track.track_id,
                start_frame=track.start_frame(),
                end_frame=track.end_frame(),
                start_time=track.start_frame() / video_props.fps,
                duration=(track.end_frame() - track.start_frame()) / video_props.fps,
                detection_count=len(track.sorted_detections),
                detections=[
                    DetectionLite(
                        frame_idx=det.frame_idx,
                        bbox=[int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2)],
                        confidence=float(det.confidence),
                    )
                    for det in track.sorted_detections
                ],
            )
            for track in tracks
        ],
    )

    with open(output_dir / f'{input_path.stem}.tracks.pkl', 'wb') as f:
        pickle.dump(metadata, f)
