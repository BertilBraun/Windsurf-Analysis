import os

from video_io import get_video_properties
from track_processor import TrackProcessor, TrackerInput
from video_splicer import VideoSplicer


class SurferTracker:
    """Main orchestrator for track processing and video generation"""

    def __init__(self):
        self.track_processor = TrackProcessor()
        self.video_splicer = VideoSplicer()

    def process_tracks(
        self,
        original_video_path: os.PathLike,
        track_inputs: TrackerInput,
        output_dir: os.PathLike | str,
    ):
        """Complete pipeline: process tracks and generate individual videos.

        Args:
            original_video_path: Path to original high-resolution video
            track_inputs: TrackerInput object containing the tracks to process
            output_dir: Directory to save individual videos
        """
        # Get video properties for track processing
        video_properties = get_video_properties(original_video_path)

        # Process tracks
        processed_tracks = self.track_processor.process_tracks(track_inputs, video_properties.total_frames)

        if not processed_tracks:
            print('No valid tracks found for video generation')
            return

        print(f'After processing: {len(processed_tracks)} tracks remaining')
        for track_id, track_data in processed_tracks.items():
            duration_frames = track_data[-1].frame_idx - track_data[0].frame_idx
            duration_seconds = duration_frames / video_properties.fps
            frame_percentage = duration_frames / video_properties.total_frames
            print(
                f'  Track {track_id}: {len(track_data)} detections, {duration_seconds:.1f}s ({frame_percentage * 100:.1f}%)'
            )

        # Generate individual videos
        self.video_splicer.generate_individual_videos(processed_tracks, original_video_path, output_dir)
