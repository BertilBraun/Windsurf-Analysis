import modal

# delete all files in the volume
volume = modal.Volume.from_name('windsurf-analysis-volume', create_if_missing=True)


def cleanup_folder(folder: str):
    print(f'Cleaning up folder: {folder}')
    for file in volume.listdir(folder):
        extension = file.path.split('.')[-1]
        if '.' not in file.path:
            cleanup_folder(file.path)
            continue
        if extension not in ['mp4', 'json', 'bin']:
            continue
        print(f'Deleting file: {file.path}')
        volume.remove_file(file.path)
    # TODO somehow delete the folder once it's empty


cleanup_folder('/')
