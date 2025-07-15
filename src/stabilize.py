import os
import tempfile
import subprocess
import logging


def stabilize_ffmpeg(input_file: os.PathLike | str, output_file: os.PathLike | str) -> bool:
    with tempfile.NamedTemporaryFile(suffix='.trf', delete=False, dir=None) as trf_tmp:
        trf_file = trf_tmp.name

    cmd_detect = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel',
        'error',
        '-i',
        input_file,
        '-vf',
        f'vidstabdetect=shakiness=5:accuracy=15:result={trf_file}',
        '-f',
        'null',
        '-',
        '-y',
    ]
    cmd_stab = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel',
        'error',
        '-i',
        input_file,
        '-vf',
        f'vidstabtransform=smoothing=30:input={trf_file},unsharp=5:5:0.8:3:3:0.4',
        '-c:v',
        'libx264',
        '-c:a',
        'copy',
        output_file,
        '-y',
    ]
    try:
        subprocess.run(cmd_detect, check=True)
        subprocess.run(cmd_stab, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f'  !! ffmpeg stabilization failed for {input_file}: {e}')
        return False
    finally:
        try:
            os.remove(trf_file)
        except Exception:
            pass
