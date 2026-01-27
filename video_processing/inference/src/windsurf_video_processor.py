import os
import logging
import pickle
from typing import Callable, Sequence, TypeVar

from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


from .settings import YOLO_MODEL_PATH
from .util.helpers import log_and_reraise

from .player.core.player_state import DetectionLite, Metadata, TrackLite, VideoProperties
from .util.video_io import VideoInfo, VideoReader, VideoWriter, get_video_properties
from .visualization.annotation_drawer import Annotation, AnnotationDrawer

from .tracking.detector import SurferDetector
from .tracking.tracking import Tracker
from .tracking.track_processing import TrackPostProcessing, prepare_renderable_tracks
from .tracking.preprocessing.preprocessor import TrackPreProcessor
from .tracking.iterative_ilp_tracker import IterativeILPTracker
from .tracking.ilp_tracker import ILPTracker
# from .tracking.discrete_opt_tracker import DiscreteOptTracker
# from .tracking.greedy_tracker import GreedyTracker
# from .tracking.oc_sort import OCSortEmbedTracker


from .visualization.debug_drawer import generate_debug_video_worker_function, debug_track_similarities
from .visualization.video_splicing import generate_individual_videos
from .visualization.stabilize import (
    Transform,
    compute_stabilization_transforms,
    stabilize_video,
    compute_stabilization_transforms_gmc,
)
from .visualization.track_graph_viz import visualize_tracks  # noqa: F401 (optional debugging import)

from .common_types import Detection, Track


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
        yolo_model_path: os.PathLike | str = YOLO_MODEL_PATH,
    ):
        self.surf_detector = SurferDetector(yolo_model_path=yolo_model_path)
        self.executor = ProcessPoolExecutor(max_workers=parallel_workers)
        self.draw_annotations = draw_annotations
        self.output_dir = Path(output_dir)
        self.generate_videos = generate_videos
        self.debug_views = debug_views
        self.stabilize = stabilize

    def process_video(self, input_path: os.PathLike) -> Metadata:
        """Main video processing pipeline with batched YOLO inference"""
        logger = logging.getLogger(__name__)

        input_path = Path(input_path)

        props = get_video_properties(input_path)
        logger.info(f'Processing video: {props.width}x{props.height}, {props.fps} FPS, {props.total_frames} frames')

        # run detection and tracking
        detections = self.surf_detector.run_object_detection_on_video(input_path.as_posix())

        transforms = compute_stabilization_transforms_gmc(input_path.as_posix(), downscale=2)
        # transforms = compute_stabilization_transforms(input_path.as_posix())

        processed_tracks = _process_detections_into_tracks(
            detections,
            props,
            transforms,
            trackers=[
                TrackPreProcessor(debug_video_path=input_path.as_posix()),
                # # GreedyTracker(),
                # DiscreteILPTracker(),
                #  OCSortEmbedTracker(
                #      det_hi=0.60,
                #      det_lo=0.05,
                #      iou_thr=0.40,
                #      inertia=0.12,
                #      delta_t=3,
                #      min_hits=2,
                #      reid_sim_thr=0.82,
                #      ambig_iou_margin=0.25,
                #      sim_margin=0.12,
                #      w_mot=0.10,
                #      spawn_suppress_iou=0.55,
                #      spawn_suppress_sim=0.85,
                #      maha_gate=20.0,
                #      output_tentative=True,
                #      output_tentative_max=3,
                #      dedup_enable=False,
                #  ),
                # IterativeILPTracker(video_path=input_path.as_posix()),
                ILPTracker(video_path=input_path.as_posix()),
                TrackPostProcessing(),
            ],
        )

        # Always save compact track metadata for the interactive player
        metadata = _save_tracks_metadata(processed_tracks, input_path, self.output_dir, props)

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

        return metadata

    def finalize(self):
        self.executor.shutdown(wait=True)

    def submit_task(self, func: Callable[[P], R], args: P, **kwargs):
        return self.executor.submit(
            log_and_reraise, func, args, helpers_log_and_reraise_output_dir=self.output_dir, **kwargs
        )


def _process_detections_into_tracks(
    detections: list[Detection], video_properties: VideoInfo, transforms: list[Transform], trackers: Sequence[Tracker]
) -> list[Track]:
    """Process collected tracks and return processed track data for video generation"""
    logger = logging.getLogger(__name__)

    if not detections:
        logger.warning('No tracks available for processing')
        return []

    processed_tracks = [Track(track_id=i, sorted_detections=[detection]) for i, detection in enumerate(detections)]
    for tracker in trackers:
        processed_tracks = tracker.track(processed_tracks, video_properties, transforms)

        # Show a timeline of the tracks with all possible merge options
        # visualize_tracks(processed_tracks, str(original_video_path))

    if not processed_tracks:
        logger.warning('No valid tracks found for video generation')
        return []

    # Log track statistics
    logger.info(f'After processing: {len(processed_tracks)} tracks remaining')
    for track in processed_tracks:
        duration_seconds = track.duration_frames / video_properties.fps
        frame_percentage = track.duration_frames / video_properties.total_frames
        logger.info(
            f'  Track {track.track_id}: {len(track.sorted_detections)} detections, {duration_seconds:.1f}s ({frame_percentage * 100:.1f}%)'
        )

    return processed_tracks


def _generate_individual_videos_worker_function(args: tuple[list[Track], os.PathLike, Path, bool]) -> None:
    tracks, input_path, output_dir, stabilize = args
    individual_videos = generate_individual_videos(tracks, input_path, output_dir)

    if stabilize:
        video_stabilizer = compute_stabilization_transforms(input_path)
        for individual_video in individual_videos:
            output_file = Path(individual_video).with_suffix('.stabilized.mp4')
            stabilize_video(input_video=individual_video, output_video=output_file, transforms=video_stabilizer)


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


def _save_tracks_metadata(tracks: list[Track], input_path: Path, output_dir: Path, video_props: VideoInfo) -> Metadata:
    """Save compact track metadata for later loading by the player interface.

    The metadata excludes embeddings to keep file sizes small.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    render_tracks = prepare_renderable_tracks(tracks, video_width=video_props.width, video_height=video_props.height)

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
                start_frame=track.start_frame,
                end_frame=track.end_frame,
                start_time=track.start_frame / video_props.fps,
                duration=track.duration_frames / video_props.fps,
                detection_count=len(track.sorted_detections),
                detections=[
                    DetectionLite(
                        frame_idx=det.frame_idx,
                        bbox=[int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2)],
                        confidence=float(det.confidence),
                        interpolated=bool(det.interpolated),
                        boom=[float(det.boom.point.x), float(det.boom.point.y), float(det.boom.conf)],
                        mast_tip=[float(det.mast_tip.point.x), float(det.mast_tip.point.y), float(det.mast_tip.conf)],
                        anchor=[int(det.anchor.x), int(det.anchor.y)],
                        scale=float(det.scale),
                    )
                    for det in rtrack.sorted_detections
                ],
            )
            for track, rtrack in zip(tracks, render_tracks)
        ],
    )

    with open(output_dir / f'{input_path.stem}.tracks.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    return metadata
