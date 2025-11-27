import modal

# delete all files in the volume
volume = modal.Volume.from_name('windsurf-analysis-volume', create_if_missing=True)
for file in volume.listdir('/'):
    extension = file.path.split('.')[-1]
    if extension not in ['mp4', 'json', 'bin']:
        continue
    volume.remove_file(file.path)
