from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


WINDOWS_FILE_HEADER = '# hard_windows_v1'


@dataclass(frozen=True)
class HardWindow:
    video_path: Path
    start_frame: int
    end_frame: int
    peak_frame: int
    score: float
    notes: str = ''


def _clamp_window(window: HardWindow) -> HardWindow:
    start_frame = max(0, int(window.start_frame))
    end_frame = max(start_frame, int(window.end_frame))
    peak_frame = min(max(int(window.peak_frame), start_frame), end_frame)
    return HardWindow(
        video_path=Path(window.video_path),
        start_frame=start_frame,
        end_frame=end_frame,
        peak_frame=peak_frame,
        score=float(window.score),
        notes=str(window.notes),
    )


def load_hard_windows(path: Path) -> list[HardWindow]:
    src_path = Path(path)
    if not src_path.exists():
        raise FileNotFoundError(str(src_path))

    windows: list[HardWindow] = []
    for raw_line in src_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith('#'):
            continue

        parts = raw_line.split('\t')
        if len(parts) < 5:
            continue

        video_raw = parts[0].strip()
        try:
            start_frame = int(parts[1])
            end_frame = int(parts[2])
            peak_frame = int(parts[3])
            score = float(parts[4])
        except ValueError:
            continue
        notes = parts[5].strip() if len(parts) > 5 else ''

        video_path = Path(video_raw)
        if not video_path.is_absolute():
            video_path = (src_path.parent / video_path).resolve()

        windows.append(
            _clamp_window(
                HardWindow(
                    video_path=video_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    peak_frame=peak_frame,
                    score=score,
                    notes=notes,
                )
            )
        )
    return windows


def save_hard_windows(path: Path, windows: list[HardWindow], *, relative_to_file: bool = False) -> None:
    dst_path = Path(path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [WINDOWS_FILE_HEADER, '# video_path\tstart_frame\tend_frame\tpeak_frame\tscore\tnotes']
    base_dir = dst_path.parent.resolve()
    for window in windows:
        clean = _clamp_window(window)
        video_path = Path(clean.video_path)
        if relative_to_file:
            try:
                video_path = video_path.resolve().relative_to(base_dir)
            except ValueError:
                video_path = video_path.resolve()
        else:
            video_path = video_path.resolve()

        notes = str(clean.notes).replace('\t', ' ').replace('\n', ' ').strip()
        lines.append(
            f'{video_path.as_posix()}\t{clean.start_frame}\t{clean.end_frame}\t{clean.peak_frame}\t{clean.score:.6f}\t{notes}'
        )

    dst_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
