import os
from typing import Iterator
import modal

# download all the videos '/{id}_upright.mp4' from the volume
volume = modal.Volume.from_name('windsurf-analysis-volume', create_if_missing=True)


# download the video to the local directory
def download_video(video_path: str, local_path: str):
    video: Iterator[bytes] = volume.read_file(video_path)
    with open(local_path, 'wb') as f:
        for chunk in video:
            f.write(chunk)
    print(f'Downloaded video: {video_path}')


OUTPUT_DIR = 'videos'
os.makedirs(OUTPUT_DIR, exist_ok=True)

for video in volume.listdir('/'):
    if video.path.endswith('_upright.mp4'):
        download_video(video.path, OUTPUT_DIR + '/' + video.path)
        volume.remove_file(video.path)
