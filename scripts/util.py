#!/usr/bin/env python3

import tempfile
import subprocess
import logging
import os
from datetime import datetime
import socket
import json
import select
from pathlib import Path
import time


class NotifySendHandler(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            try:
                msg = self.format(record)
                subprocess.run(['notify-send', '--urgency=critical', 'ERROR', msg])
            except Exception:
                self.handleError(record)


def mby_notify(message, title='Notification', urgency='normal'):
    """
    Send a desktop notification using notify-send. Silently fails if notify-send is not available.
    """
    try:
        subprocess.run(['notify-send', '--urgency', urgency, title, message])
    except Exception:
        pass


def setup_logging(log_file=None, notify_on_error=False):
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file = f'{log_file}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    if notify_on_error:
        handlers.append(NotifySendHandler())
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)


def get_tmp_file(suffix):
    return tempfile.mkstemp(suffix=suffix, dir=None)[1]


def stabilize_ffmpeg(input_file: os.PathLike, output_file: os.PathLike) -> bool:
    trf_file = get_tmp_file('.trf')

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


class MpvIPCTask:
    """A context manager for running mpv with IPC support."""

    def __init__(self, video_path: Path | str, lua_script_path: Path | str, socket_dir=None):
        self.video_path = Path(video_path)
        self.lua_script_path = Path(lua_script_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f'Video file {self.video_path} does not exist')
        if not self.lua_script_path.exists():
            raise FileNotFoundError(f'Lua script {self.lua_script_path} does not exist')
        self.socket_path = tempfile.mktemp(prefix='mpv-sock-', dir=socket_dir)
        self.proc = None
        self.sock = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def start(self):
        mpv_cmd = [
            'mpv',
            '--no-terminal',
            '--keep-open=yes',
            f'--input-ipc-server={self.socket_path}',
            f'--script={str(self.lua_script_path.resolve())}',
            str(self.video_path.resolve()),
        ]
        self.proc = subprocess.Popen(mpv_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Wait for socket
        start = time.time()
        while not Path(self.socket_path).exists():
            if self.proc.poll() is not None:
                raise RuntimeError('mpv exited before creating IPC socket')
            if time.time() - start > 5.0:
                raise RuntimeError(f'Timeout waiting for mpv socket {self.socket_path}')
            time.sleep(0.05)
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.setblocking(False)
        try:
            self.sock.connect(self.socket_path)
        except Exception as e:
            self.close()
            raise RuntimeError(f'Cannot connect to mpv IPC: {e}')

    def close(self):
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()
        try:
            Path(self.socket_path).unlink(missing_ok=True)
        except Exception:
            pass

    def poll_events(self, timeout=0.1):
        events = []
        ready, _, _ = select.select([self.sock], [], [], timeout)
        if ready:
            try:
                data = self.sock.recv(4096)
                if not data:
                    return []
                for line in data.splitlines():
                    try:
                        ev = json.loads(line)
                        events.append(ev)
                    except Exception:
                        continue
            except BlockingIOError:
                pass
        return events

    def send_command(self, cmd: list, req_id=0):
        payload = json.dumps({'command': cmd, 'request_id': req_id}) + '\n'
        self.sock.sendall(payload.encode())

    def get_property(self, prop: str, timeout=2.0):
        req_id = int(time.time() * 1e3) & 0xFFFF
        cmd = {'command': ['get_property', prop], 'request_id': req_id}
        self.sock.sendall((json.dumps(cmd) + '\n').encode())
        start = time.time()
        buffer = b''
        while True:
            ready, _, _ = select.select([self.sock], [], [], 0.2)
            if ready:
                try:
                    data = self.sock.recv(4096)
                    if not data:
                        return None
                    buffer += data
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        try:
                            resp = json.loads(line)
                            if resp.get('request_id') == req_id:
                                return resp.get('data')
                        except Exception:
                            continue
                except BlockingIOError:
                    pass
            if time.time() - start > timeout:
                logging.info('MpvViewer.get_property: timeout waiting for property')
                return None

    def show_text(self, text, duration_s=3):
        self.send_command(['show-text', text, duration_s * 1000])

    def is_running(self):
        return self.proc.poll() is None if self.proc else False
